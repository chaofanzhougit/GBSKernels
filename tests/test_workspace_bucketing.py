"""Workspace bucketing -- the device-resident + bucketing v1 contract.

Proves the central invariant of ``docs/device_resident_contract.md``: feeding a
*ragged* batch (matrices of mixed sizes -- the shape a GBS sampler's growing-
submatrix chain produces) through ``gbskernels.Workspace`` groups it into per-size
launches and scatters the results back, so the result for input ``i`` is identical
to evaluating its matrix alone. Bucketing changes grouping, never values.

The CPU-backend tests need no toolchain. The GPU-backend tests build the nanobind
extension in host-shim mode (CUDA kernels as plain C++, grid-emulated on CPU) and
validate the same invariant through the real binding path -- CPU-first, before any
GPU session. They skip cleanly where cmake/nanobind are unavailable.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import gbskernels

pytestmark = pytest.mark.layer5

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "bindings" / "build_host"


# --- input builders (ragged, sampler-shaped) --------------------------------

def _mats(func: str, dims, seed: int = 0):
    """A ragged list of `dims`-sized matrices appropriate to `func`."""
    g = np.random.default_rng(seed)
    out = []
    for k, d in enumerate(dims):
        a = g.standard_normal((d, d)) + 1j * g.standard_normal((d, d))
        if func == "tor":  # physical real O, small norm (I - O_S well-conditioned)
            a = np.real(a) * 0.1
            a = a + a.T
            a = a.astype(np.complex128)
        elif func in ("haf", "lhaf"):  # symmetric
            a = a + a.T
        out.append(np.ascontiguousarray(a.astype(np.complex128)))
    return out


# functions x the dims a sampler mixes (even for haf/lhaf/tor; within GPU caps)
_RAGGED = {
    "perm": [3, 5, 3, 4, 5, 2],
    "haf": [4, 6, 4, 8, 6, 4],
    "lhaf": [4, 6, 4, 6],
    "tor": [4, 6, 4, 8],  # matrix dim = 2 * modes
}


# --- CPU backend: the invariant needs no GPU --------------------------------

@pytest.mark.parametrize("func", ["perm", "haf", "lhaf", "tor"])
def test_cpu_bucketing_equals_per_matrix_evaluation(func):
    dims = _RAGGED[func]
    mats = _mats(func, dims, seed=hash(func) % 100)
    standalone = np.array(
        [getattr(gbskernels, func)(m, backend="cpu") for m in mats]
    )
    with gbskernels.Workspace(backend="cpu") as ws:
        got = getattr(ws, f"{func}_batched")(mats)
    # Same CPU reference either way -> bit-for-bit identical, not just close.
    assert got.shape == (len(mats),) and got.dtype == np.complex128
    assert np.array_equal(got, standalone), f"{func}: bucketing changed values"


def test_bucketing_preserves_input_order_under_interleaved_sizes():
    # Interleave sizes so buckets are non-contiguous; every output slot must map
    # back to its own input regardless of how matrices were grouped.
    dims = [4, 6, 4, 8, 6, 4, 8]
    mats = _mats("haf", dims, seed=7)
    with gbskernels.Workspace(backend="cpu") as ws:
        got = ws.haf_batched(mats)
    for i, m in enumerate(mats):
        assert got[i] == gbskernels.haf(m, backend="cpu"), f"slot {i} misplaced"


def test_empty_and_single_size_degenerate_cases():
    with gbskernels.Workspace(backend="cpu") as ws:
        empty = ws.haf_batched([])
        assert empty.shape == (0,) and empty.dtype == np.complex128
        # a single-size ragged list is just one bucket == the uniform path
        uniform = _mats("haf", [6, 6, 6], seed=1)
        got = ws.haf_batched(uniform)
        ref = gbskernels.haf_batched(uniform, backend="cpu")
        assert np.array_equal(got, ref)


def test_non_square_input_is_rejected_with_index():
    with gbskernels.Workspace(backend="cpu") as ws:
        bad = [np.eye(4), np.zeros((4, 5)), np.eye(4)]
        with pytest.raises(ValueError, match=r"#1"):
            ws.haf_batched(bad)


def test_closed_workspace_rejects_and_double_close_is_noop():
    ws = gbskernels.Workspace(backend="cpu")
    ws.__enter__()
    ws.close()
    with pytest.raises(RuntimeError, match="closed"):
        ws.haf_batched([np.eye(2)])
    ws.close()  # idempotent -- must not raise


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        gbskernels.Workspace(backend="tpu")


# --- GPU backend (host-shim): same invariant through the binding path -------

def _toolchain_ok() -> bool:
    if shutil.which("cmake") is None:
        return False
    try:
        import nanobind  # noqa: F401
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def gpu_pkg():
    if not _toolchain_ok():
        pytest.skip("cmake / nanobind not available")
    if not list(BUILD.glob("gbskernels_ext*.so")):
        subprocess.run(
            ["cmake", "-S", str(REPO / "bindings"), "-B", str(BUILD),
             "-DGBS_HOST_SHIM=ON", f"-DPython_EXECUTABLE={sys.executable}"],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(["cmake", "--build", str(BUILD), "-j"],
                       check=True, capture_output=True, text=True)
    sys.path.insert(0, str(BUILD))
    import gbskernels as pkg
    importlib.reload(pkg)
    assert pkg.gpu_available()
    return pkg


@pytest.mark.parametrize("func", ["perm", "haf", "lhaf", "tor"])
def test_gpu_bucketing_matches_cpu_reference(gpu_pkg, func):
    dims = _RAGGED[func]
    mats = _mats(func, dims, seed=hash(func) % 100)
    ref = np.array([getattr(gpu_pkg, func)(m, backend="cpu") for m in mats])
    with gpu_pkg.Workspace(backend="gpu") as ws:
        got = getattr(ws, f"{func}_batched")(mats)
    assert got.shape == ref.shape
    assert np.max(np.abs(got - ref)) < 1e-9, f"{func}: GPU bucketing != CPU ref"


def test_gpu_workspace_uses_resident_session_and_reuses_buffers(gpu_pkg):
    # Residency half of the contract: the Workspace holds one device session whose
    # buffers grow monotonically and are REUSED by smaller/equal batches -- a
    # smaller batch after a bigger one must not reallocate. ``reallocs`` is the
    # binding's diagnostic witness for exactly this (also gated in C++ by
    # core/check_session.cu). Driven here through the public Workspace API.
    with gpu_pkg.Workspace(backend="gpu") as ws:
        assert ws._session is not None, "Workspace should hold a resident session"
        ws.haf_batched(_mats("haf", [4, 4, 4], seed=10))   # first alloc
        r1 = ws._session.reallocs()
        ws.haf_batched(_mats("haf", [8, 8, 8], seed=11))    # bigger -> grow
        r2 = ws._session.reallocs()
        ws.haf_batched(_mats("haf", [4, 4, 4], seed=12))    # smaller -> reuse
        r3 = ws._session.reallocs()
    assert (r1, r2, r3) == (1, 2, 2), f"expected grow,grow,reuse; got {(r1, r2, r3)}"


def test_gpu_bucketing_rejects_oversize_member_naming_its_index(gpu_pkg):
    # A single oversize matrix in an otherwise-valid ragged batch must raise and
    # name *which* input it was (the hafnian GPU cap is 20).
    mats = _mats("haf", [6, 6], seed=2)
    oversize = np.ascontiguousarray(np.zeros((22, 22), dtype=complex))
    mats = [mats[0], oversize, mats[1]]
    with gpu_pkg.Workspace(backend="gpu") as ws:
        with pytest.raises(ValueError, match=r"#1.*exceeds the kernel"):
            ws.haf_batched(mats)


def test_gpu_workspace_runs_the_real_sampler_ragged_workload(gpu_pkg):
    # L5 capstone (contract sec.3): the actual GBS sampler's pattern-hafnian list
    # -- genuinely ragged (the vacuum pattern is 0x0, others grow with photon
    # number) -- runs end-to-end on the GPU binding path via Workspace bucketing
    # and reproduces the CPU sampler's hafnians. This is the workload the whole
    # contract exists for (docs/DESIGN.md §2.3).
    from sampling.gbs import _submatrix, fock_patterns, random_gbs_kernel

    B, _r = random_gbs_kernel(m=3, seed=4)
    even = [p for p in fock_patterns(3, cutoff=3) if sum(p) % 2 == 0]
    mats = [_submatrix(B, p) for p in even]
    assert len({m.shape[0] for m in mats}) >= 3, "workload must be heterogeneous"
    cpu = gpu_pkg.haf_batched(mats, backend="cpu")
    with gpu_pkg.Workspace(backend="gpu") as ws:
        gpu = ws.haf_batched(mats)
    assert np.max(np.abs(gpu - cpu)) < 1e-9, "GPU sampler workload != CPU reference"
