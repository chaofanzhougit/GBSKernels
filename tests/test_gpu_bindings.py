"""GPU backend (nanobind extension) -- validated on CPU via the host-shim build.

Builds ``bindings/gbskernels_ext`` in host-shim mode (the CUDA kernels compiled as
plain C++ against ``core/preflight/cuda``, grid-emulated on CPU) and checks that
the ``backend="gpu"`` path of the public API returns the same results as the CPU
reference. So the entire Python -> kernel marshalling is validated without a GPU;
the real GPU build (nvcc) of the same sources runs on the device.

Skips cleanly if ``cmake`` or ``nanobind`` is unavailable (e.g. a minimal runner).
The build is cached in ``bindings/build_host`` and reused across runs.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.layer5

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "bindings" / "build_host"


def _toolchain_ok() -> bool:
    if shutil.which("cmake") is None:
        return False
    try:
        import nanobind  # noqa: F401
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def gpu_ext():
    if not _toolchain_ok():
        pytest.skip("cmake / nanobind not available")
    so = list(BUILD.glob("gbskernels_ext*.so"))
    if not so:
        subprocess.run(
            ["cmake", "-S", str(REPO / "bindings"), "-B", str(BUILD),
             "-DGBS_HOST_SHIM=ON", f"-DPython_EXECUTABLE={sys.executable}"],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(["cmake", "--build", str(BUILD), "-j"],
                       check=True, capture_output=True, text=True)
    # fresh import of the package so it picks up the freshly built extension
    sys.path.insert(0, str(BUILD))
    import gbskernels
    importlib.reload(gbskernels)
    assert gbskernels.gpu_available(), "extension built but not importable"
    return gbskernels


def _rng_stack(n, seed, sym=False, scale=1.0):
    g = np.random.default_rng(seed)
    a = scale * (g.standard_normal((16, n, n)) + 1j * g.standard_normal((16, n, n)))
    if sym:
        a = a + a.transpose(0, 2, 1)
    return np.ascontiguousarray(a)


@pytest.mark.parametrize("func,sym", [("perm", False), ("haf", True), ("lhaf", True)])
def test_gpu_backend_matches_cpu(gpu_ext, func, sym):
    fn = getattr(gpu_ext, f"{func}_batched")
    mats = _rng_stack(6, seed=hash(func) % 1000, sym=sym)
    cpu = fn(mats, backend="cpu")
    gpu = fn(mats, backend="gpu")
    assert gpu.shape == cpu.shape and gpu.dtype == np.complex128
    assert np.max(np.abs(cpu - gpu)) < 1e-9


def test_gpu_permanent_coop_dispatch_matches_cpu(gpu_ext):
    # n >= 12 permanents auto-dispatch to the cooperative kernel inside host_api
    # (perf; measured ~5x on a 4090). The public GPU path must still match the CPU
    # reference -- through both the one-shot binding and the device-resident
    # Workspace/Session (which has its own coop branch).
    mats = _rng_stack(14, seed=11)  # n=14 >= crossover (12)
    cpu = gpu_ext.perm_batched(mats, backend="cpu")
    # n=14 permanents are large (~1e7), so compare RELATIVE error (the coop kernel
    # regroups the Glynn sum, so agreement is FP64-level, not bit-exact).
    rel = lambda a: float(np.max(np.abs(a - cpu) / np.maximum(np.abs(cpu), 1e-300)))
    gpu = gpu_ext.perm_batched(mats, backend="gpu")          # -> gbs_perm_host -> coop
    assert rel(gpu) < 1e-9, "one-shot coop dispatch != CPU"
    with gpu_ext.Workspace(backend="gpu") as ws:             # -> Session -> coop
        wsres = ws.perm_batched(mats)
    assert rel(wsres) < 1e-9, "session coop dispatch != CPU"


@pytest.mark.parametrize("func,batched,single,mk,mpname", [
    ("perm", "perm_batched", "perm",
     lambda d: __import__("bench._inputs", fromlist=["x"]).make_cancellation_matrix(6, d, 1), "permanent_mp"),
    ("haf", "haf_batched", "haf",
     lambda d: __import__("bench._inputs", fromlist=["x"]).cancellation_hafnian(d, 1), "hafnian_mp"),
    ("lhaf", "lhaf_batched", "lhaf",
     lambda d: __import__("bench._inputs", fromlist=["x"]).cancellation_loop_hafnian(d, 1), "loop_hafnian_mp"),
    ("tor", "tor_batched", "tor",
     lambda d: __import__("bench._inputs", fromlist=["x"]).cancellation_torontonian(d), "torontonian_mp"),
])
def test_gpu_auto_indicator_and_dd_rerun(gpu_ext, func, batched, single, mk, mpname):
    # GPU precision="auto" for all four: the *_kappa kernel emits the cancellation
    # indicator (sum|term|) alongside the FP64 result in one device pass; the host
    # reruns only the risky elements in DD. A uniform batch mixing well-conditioned
    # and heavy-cancellation inputs must select FP64 vs DD per element and recover
    # accuracy on the risky one (where plain FP64 has lost digits).
    import mpmath
    import highprec_ref

    mpref = getattr(highprec_ref, mpname)
    mats = [mk(1e-1), mk(1e-10), mk(1e-1)]  # idx1 cancels heavily
    out, diags = getattr(gpu_ext, batched)(mats, precision="auto", backend="gpu",
                                           return_diagnostics=True)
    assert [d["tier"] for d in diags] == ["fp64", "dd", "fp64"], func
    assert diags[1]["cancellation"] > gpu_ext._AUTO_KAPPA_MAX  # on-device kappa flagged it
    with mpmath.workdps(60):
        exact = mpref(mats[1], dps=60)
        rel_auto = float(abs(mpmath.mpc(complex(out[1])) - exact) / abs(exact))
        fp64 = getattr(gpu_ext, single)(mats[1], backend="gpu")
        rel_fp64 = float(abs(mpmath.mpc(complex(fp64)) - exact) / abs(exact))
    assert rel_auto < 1e-12, f"{func}: DD rerun should recover accuracy"
    assert rel_fp64 > 1e4 * rel_auto, f"{func}: FP64 should have degraded vs the DD rerun"
    # the well-conditioned elements are the plain FP64 result (no rerun)
    assert out[0] == complex(getattr(gpu_ext, single)(mats[0], backend="gpu"))


def test_gpu_workspace_auto_bucketing(gpu_ext):
    # P0.1: precision="auto" through the Workspace is bucket-wise per-element auto -- the
    # ragged result is identical to evaluating each matrix's auto alone (the central
    # bucketing invariant, now honored for 'auto'). Mix sizes AND conditioning so both the
    # buckets and the per-element FP64/DD tier selection vary.
    import bench._inputs as I
    mats = [I.physical_hafnian(4, 1), I.cancellation_hafnian(1e-10, 1),     # 4x4 fp64; 8x8 dd
            I.physical_hafnian(8, 2), I.cancellation_hafnian(1e-10, 3)]     # 8x8 fp64; 8x8 dd
    with gpu_ext.Workspace(backend="gpu") as ws:
        got = ws.haf_batched(mats, precision="auto")
    ref = np.array([complex(gpu_ext.haf(m, precision="auto", backend="gpu")) for m in mats])
    assert got.shape == (4,) and got.dtype == np.complex128
    assert np.array_equal(got, ref), "Workspace auto != per-matrix auto (bucketing changed values)"


def test_gpu_auto_over_dd_cap_falls_back_to_mpmath(gpu_ext):
    # P0.3: GPU auto's rerun tier is DD, whose cap (haf_dd=16) is below the FP64 cap
    # (haf=20). For a size BETWEEN them a risky element cannot rerun in DD, so it reruns on
    # the CPU in mpmath ('ref') -- a DEFINED, precision-preserving fallback, not the old
    # data-dependent crash. A well-conditioned same-size element stays FP64.
    import mpmath
    import highprec_ref
    import bench._inputs as I
    d, delta = 18, 1e-11                                   # 16 < 18 <= 20
    a = f = b = e = c = 1.0
    blk = np.array([[0, a, b, c], [a, 0, delta - 2.0, e], [b, delta - 2.0, 0, f], [c, e, f, 0]], complex)
    g = np.random.default_rng(5); R = g.standard_normal((d - 4, d - 4)); R = R + R.T; np.fill_diagonal(R, 0.0)
    risky = np.zeros((d, d), complex); risky[:4, :4] = blk; risky[4:, 4:] = R
    risky = np.ascontiguousarray(risky)
    well = I.physical_hafnian(d, 6)
    out, diags = gpu_ext.haf_batched([well, risky], precision="auto", backend="gpu",
                                     return_diagnostics=True)
    assert diags[0]["tier"] == "fp64"
    assert diags[1]["tier"] == "ref", "an over-DD-cap risky element must rerun on CPU mpmath"
    with mpmath.workdps(40):
        exact = highprec_ref.hafnian_mp(risky, dps=40)
        rel = float(abs(mpmath.mpc(complex(out[1])) - exact) / abs(exact))
    assert rel < 1e-12, "the mpmath fallback recovered accuracy where FP64 had cancelled"


def test_gpu_sampler_over_cap_routes_to_precision_preserving_cpu(gpu_ext, monkeypatch):
    # P0.2: in the GPU sampler, a submatrix beyond the GPU haf cap falls back to the CPU at a
    # precision AT LEAST as accurate as requested -- a requested 'dd' is served as mpmath
    # 'ref' (CPU has no DD), never silently as CPU fp64. Spy on the routing (fast: no mpmath).
    import bench._inputs as I
    from sampling import sampler
    small = I.physical_hafnian(6, 1)
    big = I.physical_hafnian(24, 2)                        # 24 > GPU haf cap -> CPU fallback
    seen: dict = {}
    real_batched = sampler.gbskernels.haf_batched

    def spy(mats, precision="fp64", backend="cpu"):
        seen["precision"], seen["backend"] = precision, backend
        return real_batched(mats, precision="fp64", backend="cpu")  # keep it fast; we check routing

    monkeypatch.setattr(sampler.gbskernels, "haf_batched", spy)
    with gpu_ext.Workspace(backend="gpu") as ws:
        out = sampler._eval_haf_batch([small, big], ws, "dd")
    assert seen == {"precision": "ref", "backend": "cpu"}, "over-cap 'dd' must route to CPU 'ref'"
    assert np.isfinite(out[1])


def test_gpu_torontonian_real_domain(gpu_ext):
    O = _rng_stack(6, seed=7, sym=True, scale=0.1)
    O = np.ascontiguousarray(np.real(O).astype(np.complex128))  # physical real O
    cpu = gpu_ext.tor_batched(O, backend="cpu")
    gpu = gpu_ext.tor_batched(O, backend="gpu")
    assert np.max(np.abs(cpu - gpu)) < 1e-9


def test_gpu_perm_dd_matches_reference(gpu_ext):
    from cpu_ref.permanent import permanent_glynn_dd

    A = _rng_stack(6, seed=3)
    gpu_dd = gpu_ext.perm_batched(A, precision="dd", backend="gpu")
    ref = np.array([permanent_glynn_dd(m) for m in A])
    assert np.max(np.abs(gpu_dd - ref)) < 1e-12


def test_gpu_haf_dd_holds_where_fp64_cancels(gpu_ext):
    # The DD hafnian (hardest kernel) must track the mpmath reference at machine
    # precision on a cancellation-heavy input where FP64 has lost many digits.
    import mpmath
    from highprec_ref import hafnian_mp

    delta = 1e-9
    a = f = b = e = c = 1.0
    d = delta - 2.0
    B = np.array([[0, a, b, c], [a, 0, d, e], [b, d, 0, f], [c, e, f, 0]], float)
    g = np.random.default_rng(2)
    R = g.standard_normal((4, 4)); R = R + R.T; np.fill_diagonal(R, 0.0)
    A = np.zeros((8, 8)); A[:4, :4] = B; A[4:, 4:] = R
    A = np.ascontiguousarray(A.astype(complex))

    dd = gpu_ext.haf_batched(A[None], precision="dd", backend="gpu")[0]
    fp = gpu_ext.haf_batched(A[None], precision="fp64", backend="gpu")[0]
    with mpmath.workdps(60):
        exact = hafnian_mp(A, dps=60)
        err_dd = float(abs(mpmath.mpc(dd) - exact) / abs(exact))
        err_fp = float(abs(mpmath.mpc(fp) - exact) / abs(exact))
    assert err_dd < 1e-13          # DD holds at machine precision
    assert err_fp > 1e-6           # FP64 has genuinely degraded here
    assert err_dd < err_fp / 1e4   # DD is dramatically better


def test_gpu_lhaf_dd_holds_where_fp64_cancels(gpu_ext):
    import mpmath
    from highprec_ref import loop_hafnian_mp

    delta = 1e-9
    B = np.array([[1.0, 2.0], [2.0, delta - 2.0]])  # lhaf(B) = delta
    g = np.random.default_rng(1)
    R = g.standard_normal((4, 4)); R = R + R.T
    A = np.zeros((6, 6)); A[:2, :2] = B; A[2:, 2:] = R
    A = np.ascontiguousarray(A.astype(complex))
    dd = gpu_ext.lhaf_batched(A[None], precision="dd", backend="gpu")[0]
    fp = gpu_ext.lhaf_batched(A[None], precision="fp64", backend="gpu")[0]
    with mpmath.workdps(60):
        ex = loop_hafnian_mp(A, dps=60)
        err_dd = float(abs(mpmath.mpc(dd) - ex) / abs(ex))
        err_fp = float(abs(mpmath.mpc(fp) - ex) / abs(ex))
    assert err_dd < 1e-13 and err_fp > 1e-6 and err_dd < err_fp / 1e4


def test_gpu_tor_dd_holds_where_fp64_cancels(gpu_ext):
    import mpmath
    from highprec_ref import torontonian_mp

    # single-mode O = diag(a,a): tor = a/(1-a), the kernel's 1/sqrt(det)-1 cancels
    O = np.ascontiguousarray(np.array([[1e-9, 0.0], [0.0, 1e-9]], dtype=complex))
    dd = gpu_ext.tor_batched(O[None], precision="dd", backend="gpu")[0]
    fp = gpu_ext.tor_batched(O[None], precision="fp64", backend="gpu")[0]
    with mpmath.workdps(60):
        ex = torontonian_mp(O, dps=60)
        err_dd = float(abs(mpmath.mpc(dd) - ex) / abs(ex))
        err_fp = float(abs(mpmath.mpc(fp) - ex) / abs(ex))
    # single-mode tor is a 2-term cancellation -> FP64 degrades to ~1e-7 (less than
    # the power-trace cases), but DD is still machine-precision and far better.
    assert err_dd < 1e-13 and err_fp > 1e-8 and err_dd < err_fp / 1e4


@pytest.mark.parametrize("func,sym,real,d", [("perm", False, False, 6), ("haf", True, False, 6),
                                             ("lhaf", True, False, 5), ("tor", True, True, 6)])
def test_gpu_resident_output_dlpack(gpu_ext, func, sym, real, d):
    # Workspace v2: the result is left ON THE DEVICE and returned as a zero-copy,
    # DLPack-exportable handle (on the host-shim CPU "device" a numpy array; on a real
    # GPU build a CuPy array). It must (a) hold the same values as the host-result
    # path, (b) be consumable zero-copy via the DLPack protocol, (c) report the right
    # DLPack device, and (d) be an independent buffer (not the reused session output).
    # lhaf d=5 exercises odd-N (augmented) on the resident path too.
    A = _rng_stack(d, seed=hash(func) % 100, sym=sym)
    if real:
        A = np.ascontiguousarray(np.real(A).astype(np.complex128) * 0.1)
    with gpu_ext.Workspace(backend="gpu") as ws:
        host = getattr(ws, f"{func}_batched")(A)
        dev = getattr(ws, f"{func}_resident")(A)
        rt = np.from_dlpack(dev)                       # DLPack zero-copy consumption
        assert np.allclose(rt, host), f"{func}: resident != host-result path"
        assert np.shares_memory(rt, np.asarray(dev)), "from_dlpack must be zero-copy"
        assert dev.__dlpack_device__()[0] == 1, "host-shim build -> kDLCPU device"
        # a second held result is a distinct buffer (resident output is not the
        # session's reused d_out)
        dev2 = getattr(ws, f"{func}_resident")(A * (2.0 if not real else 1.0))
        assert np.from_dlpack(dev) is not np.from_dlpack(dev2)


def test_gpu_resident_output_errors(gpu_ext):
    # device-resident output is GPU + fp64 + uniform-batch only; clear errors else.
    A = _rng_stack(6, seed=1)
    with gpu_ext.Workspace(backend="cpu") as cpu_ws:
        with pytest.raises(NotImplementedError, match="backend='gpu'"):
            cpu_ws.perm_resident(A)
    with gpu_ext.Workspace(backend="gpu") as ws:
        with pytest.raises(NotImplementedError, match="fp64"):
            ws.perm_resident(A, precision="dd")
        with pytest.raises(ValueError, match="uniform"):  # ragged -> use *_batched
            ws.perm_resident([np.eye(2), np.eye(3)])


def test_gpu_resident_torch_interop_if_available(gpu_ext):
    # If PyTorch is present it consumes the same DLPack handle zero-copy (the protocol
    # is framework-agnostic; on a real GPU build this is the CuPy/PyTorch-GPU path).
    torch = pytest.importorskip("torch")
    A = _rng_stack(6, seed=4)
    with gpu_ext.Workspace(backend="gpu") as ws:
        dev = ws.perm_resident(A)
        t = torch.from_dlpack(dev)
        assert np.allclose(t.numpy(), ws.perm_batched(A))


def test_gpu_single_eval_and_errors(gpu_ext):
    assert gpu_ext.perm(np.array([[1.0, 2.0], [3.0, 4.0]]), backend="gpu") == pytest.approx(10.0)
    with pytest.raises(NotImplementedError):  # 'ref' is mpmath (CPU), not a GPU kernel
        gpu_ext.perm_batched(_rng_stack(4, 1), precision="ref", backend="gpu")
    with pytest.raises(ValueError):  # ragged batch not supported on GPU
        gpu_ext.perm_batched([np.eye(2), np.eye(3)], backend="gpu")


def test_gpu_rejects_oversize_input(gpu_ext):
    # Beyond the per-kernel local-buffer limit must raise, not overflow.
    big_haf = np.ascontiguousarray(np.zeros((1, 22, 22), dtype=complex))  # haf cap is 20
    with pytest.raises(ValueError, match="exceeds the kernel"):
        gpu_ext.haf_batched(big_haf, backend="gpu")
    big_haf_dd = np.ascontiguousarray(np.zeros((1, 18, 18), dtype=complex))  # haf_dd cap 16
    with pytest.raises(ValueError, match="exceeds the kernel"):
        gpu_ext.haf_batched(big_haf_dd, precision="dd", backend="gpu")


@pytest.mark.parametrize("n,prec", [(1, "fp64"), (3, "fp64"), (5, "fp64"),
                                    (7, "fp64"), (3, "dd"), (5, "dd")])
def test_gpu_loop_hafnian_odd_dimension(gpu_ext, n, prec):
    # The GPU loop hafnian now accepts ODD N: it augments to (N+1) with a self-loop-1
    # vertex (zero off-diagonals) -- which can only loop, leaving lhaf invariant -- and
    # runs the validated even-N kernel. Must match the CPU reference (which handles
    # any N) for both fp64 and DD. This is what puts displaced GBS on the GPU.
    A = _rng_stack(n, seed=10 + n, sym=True)[:4]   # symmetric, nonzero diagonal (loops)
    cpu = gpu_ext.lhaf_batched(A, backend="cpu")
    gpu = gpu_ext.lhaf_batched(A, precision=prec, backend="gpu")
    assert np.max(np.abs(gpu - cpu) / np.maximum(np.abs(cpu), 1e-300)) < 1e-9
    # a single odd-N call works too, and through the device-resident Workspace
    assert abs(complex(gpu_ext.lhaf(A[0], backend="gpu")) - complex(cpu[0])) < 1e-9 * (abs(cpu[0]) + 1)
    with gpu_ext.Workspace(backend="gpu") as ws:
        assert np.max(np.abs(ws.lhaf_batched(A) - cpu)) < 1e-9


def test_end_to_end_throughput_harness(gpu_ext, tmp_path):
    # The public-path throughput harness runs, records median+IQR + commit, and
    # its built-in GPU-vs-CPU checksum agreement holds (correctness inside the
    # benchmark). Timing is host-emulated here; the harness/structure is the point.
    from bench.throughput_end_to_end import run as run_e2e

    artifact, path = run_e2e(batch=16, repeats=3, out_dir=tmp_path)
    assert path.exists()
    assert artifact["kind"] == "throughput_end_to_end"
    assert artifact["gpu_backend"] in {"gpu", "host-shim"}
    assert artifact["params"]["regime"] == "physical"            # shared-generator regime recorded
    assert "winner" not in artifact and "score" not in artifact  # no composite
    assert artifact["all_backends_agree"] is True                # P0.8 aggregate gate (host-shim: GPU==CPU)
    for r in artifact["summary"]:
        assert r["backends_agree"], f"GPU vs CPU mismatch at {r['func']} dim {r['matrix_dim']}"
        assert "evals_per_sec_median" in r["gpu"] and "evals_per_sec_iqr" in r["gpu"]


def test_e2e_checksum_disagreement_exits_nonzero(monkeypatch, tmp_path):
    # P0.8: a public-path checksum disagreement must FAIL the run (exit 2) so the GPU
    # session aborts instead of publishing a number from a backend mismatch. Drive main()
    # with a stubbed run() that reports a disagreeing cell (no GPU needed).
    import bench.throughput_end_to_end as e2e

    fake = {"gpu_backend": "gpu", "commit": "x", "container_digest": "y",
            "all_backends_agree": False,
            "summary": [{"func": "haf", "matrix_dim": 8, "gpu": {"evals_per_sec_median": 1.0},
                         "cpu": {"evals_per_sec_median": 1.0}, "backends_agree": False}]}
    monkeypatch.setattr(e2e, "run", lambda *a, **k: (fake, tmp_path / "x.json"))
    monkeypatch.setattr(sys, "argv", ["e2e", "--batch", "8", "--repeats", "2"])
    with pytest.raises(SystemExit) as ei:
        e2e.main()
    assert ei.value.code == 2


def test_gpu_dd_torontonian_rejects_complex_input(gpu_ext):
    # Real double-double torontonian must refuse complex O, not silently drop imag.
    O = np.ascontiguousarray((np.eye(4) * 0.1 + 1j * np.full((4, 4), 0.01)))
    with pytest.raises(ValueError, match="real-domain only"):
        gpu_ext.tor_batched(O[None], precision="dd", backend="gpu")
    # a real O is accepted
    Oreal = np.ascontiguousarray((np.eye(4) * 0.1).astype(complex))
    assert np.isfinite(abs(gpu_ext.tor_batched(Oreal[None], precision="dd", backend="gpu")[0]))
