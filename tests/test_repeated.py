"""Repeated-row loop hafnian: the finite-difference sieve.

Ground truth is the *expanded* matrix evaluated by the already-validated
references (``cpu_ref.lhaf`` / ``cpu_ref.haf``); Layer 2 adds The Walrus's own
repeated-row implementation on the case its API expresses (loop weights =
diagonal). The public API and the (host-shim) GPU kernel are pinned against the
CPU sieve, and the sampler's opt-in sieve path must reproduce the standard
chain's conditional weights.
"""

from __future__ import annotations

import numpy as np
import pytest

import cpu_ref
import gbskernels
from cpu_ref.repeated import lhaf_repeated, sieve_term_count

pytestmark = pytest.mark.layer5


def _base(M: int, seed: int, cplx: bool = True):
    g = np.random.default_rng(seed)
    A = g.standard_normal((M, M))
    if cplx:
        A = A + 1j * g.standard_normal((M, M))
    A = A + A.T
    gam = g.standard_normal(M) + (1j * g.standard_normal(M) if cplx else 0.0)
    return A, gam


def _expand(A, gamma, reps):
    idx = [i for i, ni in enumerate(reps) for _ in range(int(ni))]
    if not idx:
        return np.empty((0, 0), dtype=np.complex128)
    E = np.array(np.asarray(A, dtype=np.complex128)[np.ix_(idx, idx)])
    np.fill_diagonal(E, np.asarray(gamma, dtype=np.complex128)[idx])
    return E


PATTERNS = [(0, 0, 0), (1, 1, 0), (2, 1, 1), (2, 0, 2), (3, 2, 1), (1, 1, 1),
            (4, 3, 2), (5, 0, 1), (2, 2, 2)]


@pytest.mark.parametrize("cplx", [False, True])
def test_sieve_equals_expanded_loop_hafnian(cplx):
    A, gam = _base(3, 42, cplx)
    for reps in PATTERNS:
        truth = cpu_ref.lhaf(_expand(A, gam, reps))
        got = lhaf_repeated(A, gam, reps)
        assert abs(got - truth) <= 1e-11 * max(1.0, abs(truth)), reps


def test_sieve_gamma_zero_is_plain_hafnian_of_expansion():
    A, _ = _base(3, 7)
    for reps in [(2, 2, 0), (2, 1, 1), (4, 2, 2), (3, 1, 0)]:
        truth = cpu_ref.haf(_expand(A, np.zeros(3), reps))
        got = lhaf_repeated(A, None, reps)
        assert abs(got - truth) <= 1e-11 * max(1.0, abs(truth)), reps
    # odd total photons with zero loops -> exactly zero
    assert lhaf_repeated(A, None, (2, 1, 0)) == 0.0


def test_sieve_term_count_and_validation():
    assert sieve_term_count((3, 2, 1)) == 4 * 3 * 2
    assert sieve_term_count(()) == 1
    A, gam = _base(3, 1)
    with pytest.raises(ValueError, match="reps"):
        lhaf_repeated(A, gam, (1, 2)) # wrong length
    with pytest.raises(ValueError, match="non-negative"):
        lhaf_repeated(A, gam, (1, -1, 0))


@pytest.mark.layer2
def test_sieve_vs_the_walrus_repeated():
    thewalrus = pytest.importorskip("thewalrus")
    A, _ = _base(4, 11)
    gam = np.diag(A).copy() # hafnian_repeated: loops = the (repeated) diagonal
    for reps in [(1, 1, 1, 1), (2, 1, 0, 1), (3, 2, 1, 0), (2, 2, 2, 2)]:
        w = complex(thewalrus.hafnian_repeated(np.asarray(A), rpt=list(reps), loop=True))
        got = lhaf_repeated(A, gam, reps)
        assert abs(got - w) <= 1e-9 * max(1.0, abs(w)), reps


def test_public_api_single_batched_and_gpu():
    A, gam = _base(3, 5)
    reps = np.array([[2, 1, 1], [1, 1, 0], [3, 0, 2]], dtype=np.int32)
    vec = gbskernels.lhaf_repeated(A, gam, reps)
    assert vec.shape == (3,)
    for i, r in enumerate(reps):
        single = gbskernels.lhaf_repeated(A, gam, r)
        assert single == vec[i] # batched == looped
        truth = cpu_ref.lhaf(_expand(A, gam, r))
        assert abs(vec[i] - truth) <= 1e-11 * max(1.0, abs(truth))

    ext = gbskernels._load_gpu_ext()
    if ext is not None and hasattr(ext, "lhaf_repeated"):
        gpu = gbskernels.lhaf_repeated(A, gam, reps, backend="gpu")
        assert np.max(np.abs(gpu - vec)) <= 1e-11 * max(1.0, float(np.max(np.abs(vec))))


def test_sampler_sieve_weights_match_standard_chain():
    from sampling import sampler as smod

    g = np.random.default_rng(3)
    k, cutoff = 3, 4
    # a valid reduced A-matrix from a pure state (reuse the sampler's own builder)
    m = 4
    r = g.uniform(0.15, 0.35, m)
    z = (g.standard_normal((m, m)) + 1j * g.standard_normal((m, m))) / np.sqrt(2.0)
    U, rr = np.linalg.qr(z)
    ph = np.diagonal(rr).copy()
    ph /= np.abs(ph)
    U = U * ph
    X, Y = U.real, U.imag
    S = np.block([[X, -Y], [Y, X]]) @ np.diag(np.concatenate([np.exp(-r), np.exp(r)]))
    cov = S @ S.T # hbar=2: cov = (hbar/2) S S^T
    A = smod._reduced_A(cov, k, hbar=2.0)

    from math import factorial

    uniq = np.array([[0, 0], [1, 0], [2, 1], [1, 3]], dtype=np.int64)
    inv_fac = np.array([1.0 / factorial(j) for j in range(cutoff + 1)])
    std = smod._conditional_weights(A, uniq, k, cutoff + 1, inv_fac, None, "fp64")
    sieve = smod._conditional_weights(A, uniq, k, cutoff + 1, inv_fac, None, "fp64",
                                      repeated_sieve=True)
    assert np.allclose(std, sieve, rtol=1e-9, atol=1e-13)

    ext = gbskernels._load_gpu_ext()
    if ext is not None and hasattr(ext, "lhaf_repeated"):
        ws = gbskernels.Workspace() # GPU route: one lhaf_repeated launch/step
        gpu = smod._conditional_weights(A, uniq, k, cutoff + 1, inv_fac, ws, "fp64",
                                        repeated_sieve=True)
        assert np.allclose(gpu, sieve, rtol=1e-11, atol=1e-15)


@pytest.mark.slow
def test_sampler_sieve_end_to_end_distribution():
    """sample(repeated_sieve=True) draws from the same distribution as the
    standard chain (weights agree to fp64 rounding -> TV within sampling noise)."""
    from sampling import sampler as smod

    g = np.random.default_rng(9)
    m = 3
    r = g.uniform(0.15, 0.35, m)
    z = (g.standard_normal((m, m)) + 1j * g.standard_normal((m, m))) / np.sqrt(2.0)
    U, rr = np.linalg.qr(z)
    ph = np.diagonal(rr).copy()
    ph /= np.abs(ph)
    U = U * ph
    X, Y = U.real, U.imag
    S = np.block([[X, -Y], [Y, X]]) @ np.diag(np.concatenate([np.exp(-r), np.exp(r)]))
    cov = S @ S.T

    N = 1500
    a = smod.sample(cov, N, cutoff=4, seed=5)
    b = smod.sample(cov, N, cutoff=4, seed=6, repeated_sieve=True)

    def emp(s):
        d = {}
        for x in s:
            t = tuple(int(v) for v in x)
            d[t] = d.get(t, 0) + 1.0 / len(s)
        return d

    pa, pb = emp(a), emp(b)
    tv = 0.5 * sum(abs(pa.get(t, 0.0) - pb.get(t, 0.0)) for t in set(pa) | set(pb))
    assert tv < 4.0 / np.sqrt(N)

    # the GPU route is now real: it must draw a valid sample set
    import gbskernels
    if gbskernels._load_gpu_ext() is not None:
        s_gpu = smod.sample(cov, 50, cutoff=3, backend="gpu", seed=4,
                            repeated_sieve=True)
        assert s_gpu.shape == (50, m) and (s_gpu >= 0).all()
    with pytest.raises(NotImplementedError, match="fp64"):
        smod.sample(cov, 4, cutoff=3, precision="auto", repeated_sieve=True)

def test_repeated_ab_harness_smoke():
    """The R4 device-A/B harness runs end-to-end (host-shim in CI) and its
    honesty gate (sieve == expanded checksum) holds on a tiny workload."""
    ext = gbskernels._load_gpu_ext()
    if ext is None or not hasattr(ext, "lhaf_repeated"):
        pytest.skip("gbskernels_ext with lhaf_repeated not built")
    from bench.repeated_ab import run

    art = run(modes=2, qs=[2, 3], batch=8, repeats=2, warmup=1, seed=0)
    assert art["rows"][0]["values_agree_rel"] <= 1e-8
    assert art["rows"][0]["speedup_sieve_over_expanded"] > 0
    assert "provenance" in art and art["gpu_backend_kind"] in ("gpu", "host-shim")


def test_default_sample_and_benchmark_measure_the_same_path(monkeypatch):
    """Regression for the benchmark-semantics bug: sample()'s auto-sieve default
    and samples_per_second()'s default must resolve identically, so the reported
    default samples/sec IS the default path (the bench's plain rows previously
    pinned repeated_sieve=False while sample() defaulted to the sieve)."""
    import gbskernels
    from sampling import sampler as smod

    ext = gbskernels._load_gpu_ext()
    if ext is None or not hasattr(ext, "lhaf_repeated"):
        pytest.skip("extension without lhaf_repeated")

    g = np.random.default_rng(11)
    m = 4
    r = g.uniform(0.15, 0.35, m)
    z = (g.standard_normal((m, m)) + 1j * g.standard_normal((m, m))) / np.sqrt(2.0)
    U, rr = np.linalg.qr(z)
    ph = np.diagonal(rr).copy()
    ph /= np.abs(ph)
    U = U * ph
    X, Y = U.real, U.imag
    S = np.block([[X, -Y], [Y, X]]) @ np.diag(np.concatenate([np.exp(-r), np.exp(r)]))
    cov = S @ S.T
    calls = {"n": 0}
    real = gbskernels.lhaf_repeated

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(gbskernels, "lhaf_repeated", spy)

    # default sample() at gpu/fp64/cutoff>=4/modes<=8 -> the sieve path
    smod.sample(cov, 8, cutoff=4, backend="gpu", seed=0)
    assert calls["n"] > 0, "default sample() should auto-route to the sieve here"

    # default samples_per_second() must take the SAME path and say so
    calls["n"] = 0
    row = smod.samples_per_second(cov, 8, cutoff=4, backend="gpu", seed=0)
    assert calls["n"] > 0, "default samples_per_second() must measure the default path"
    assert row["repeated_sieve_effective"] is True
    assert row["backend"] == "gpu" # default row: unsuffixed label

    # explicit pins are labeled and honored
    calls["n"] = 0
    row = smod.samples_per_second(cov, 8, cutoff=4, backend="gpu", seed=0,
                                  repeated_sieve=False)
    assert calls["n"] == 0 and row["backend"] == "gpu+nosieve"
    assert row["repeated_sieve_effective"] is False
    row = smod.samples_per_second(cov, 8, cutoff=4, backend="gpu", seed=0,
                                  repeated_sieve=True)
    assert calls["n"] > 0 and row["backend"] == "gpu+sieve"


def test_certified_sieve_enclosure_and_bit_identity():
    """Certified sieve values are bit-identical to the plain sieve on
    both backends, and the bound encloses the mpmath expanded reference --
    including a near-cancelling gamma (adversarial-ish) pattern."""
    import gbskernels
    from highprec_ref import loop_hafnian_mp

    g = np.random.default_rng(21)
    M = 4
    z = (g.standard_normal((M, M)) + 1j * g.standard_normal((M, M))) / np.sqrt(2)
    A = (z + z.T) / 2
    gam = (g.standard_normal(M) + 1j * g.standard_normal(M)) * 0.3
    reps = np.array([[1, 2, 0, 3], [2, 2, 2, 2], [0, 0, 1, 1], [4, 0, 2, 2]],
                    dtype=np.int32)

    vc, dc = gbskernels.lhaf_repeated(A, gam, reps, backend="cpu", certified=True)
    vp_c = gbskernels.lhaf_repeated(A, gam, reps, backend="cpu")
    assert np.array_equal(vc, vp_c)

    ext = gbskernels._load_gpu_ext()
    pairs = [(vc, dc)]
    if ext is not None and hasattr(ext, "lhaf_repeated_certified"):
        vg, dg = gbskernels.lhaf_repeated(A, gam, reps, backend="gpu", certified=True)
        vp = gbskernels.lhaf_repeated(A, gam, reps, backend="gpu")
        assert np.array_equal(vg, vp)
        pairs.append((vg, dg))

    for i, row in enumerate(reps):
        idx = np.repeat(np.arange(M), row)
        E = A[np.ix_(idx, idx)].copy()
        np.fill_diagonal(E, gam[idx])
        ex = complex(loop_hafnian_mp(E, dps=50))
        for v, d in pairs:
            assert abs(v[i] - ex) <= d["abs_error_bound"][i]
            assert d["tier"] == "certified-fp64"

    # scalar form + over-cap refusal (bounds must be inf, never finite lies)
    v1, d1 = gbskernels.lhaf_repeated(A, gam, np.array([1, 1, 0, 0]), certified=True)
    assert isinstance(v1, complex) and np.isfinite(d1["abs_error_bound"])


def test_sampler_certified_weights_kept_mass():
    """certified_weights=True: draws bit-identical to the plain sieve chain
    (the certificate rides along, never changes sampling); diagnostics carry
    finite per-mode kept-TV bounds; guards refuse resident/non-fp64."""
    import gbskernels
    from sampling import sampler as smod

    ext = gbskernels._load_gpu_ext()
    if ext is None or not hasattr(ext, "lhaf_repeated_certified"):
        pytest.skip("extension without certified sieve")
    g = np.random.default_rng(31)
    m = 4
    r = g.uniform(0.15, 0.3, m)
    z = (g.standard_normal((m, m)) + 1j * g.standard_normal((m, m))) / np.sqrt(2)
    U, rr = np.linalg.qr(z)
    ph = np.diagonal(rr).copy()
    ph /= np.abs(ph)
    U = U * ph
    X, Y = U.real, U.imag
    S = np.block([[X, -Y], [Y, X]]) @ np.diag(np.concatenate([np.exp(-r), np.exp(r)]))
    cov = S @ S.T

    s1, d = smod.sample(cov, 40, cutoff=4, backend="gpu", seed=5,
                        certified_weights=True, return_diagnostics=True)
    s0 = smod.sample(cov, 40, cutoff=4, backend="gpu", seed=5)
    assert np.array_equal(s0, s1)
    kc = d["kept_mass_certified"]
    assert np.isfinite(kc["total_tv_kept_bound"])
    assert kc["total_tv_kept_bound"] < 1e-8
    assert len(kc["per_mode_tv_bound_max"]) == m

    with pytest.raises(NotImplementedError, match="resident"):
        smod.sample(cov, 4, cutoff=4, backend="gpu", resident=True,
                    certified_weights=True)
    with pytest.raises(NotImplementedError, match="fp64"):
        smod.sample(cov, 4, cutoff=4, precision="auto", certified_weights=True)
    with pytest.raises(ValueError, match="repeated_sieve"):
        smod.sample(cov, 4, cutoff=4, backend="gpu", repeated_sieve=False,
                    certified_weights=True)


def test_certified_sieve_bit_identity_at_large_N():
    """High-2 regression: the certified sieve value must be bit-identical to the
    plain sieve at N past ~16 (where a re-associated coeff recurrence diverges).
    Exercises both backends over reps summing to 16/20/28."""
    import gbskernels

    g = np.random.default_rng(44)
    M = 4
    z = (g.standard_normal((M, M)) + 1j * g.standard_normal((M, M))) / np.sqrt(2)
    A = (z + z.T) / 2
    gam = (g.standard_normal(M) + 1j * g.standard_normal(M)) * 0.3
    reps = np.array([[8, 8, 0, 0], [10, 10, 0, 0], [7, 7, 7, 7]], dtype=np.int32) # N=16,20,28

    for backend in ("cpu", "gpu"):
        ext = gbskernels._load_gpu_ext()
        if backend == "gpu" and (ext is None or not hasattr(ext, "lhaf_repeated_certified")):
            continue
        plain = gbskernels.lhaf_repeated(A, gam, reps, backend=backend)
        cert, _ = gbskernels.lhaf_repeated(A, gam, reps, backend=backend, certified=True)
        assert np.array_equal(plain, cert), f"{backend}: certified != plain at large N"


def test_certified_weights_cpu_and_default_paths():
    """High-1 regression: certified_weights=True must be honored on the CPU
    backend and with repeated_sieve left at its None default (the auto-resolution
    used to run first and reject it); explicit False still conflicts."""
    from sampling import sampler as smod

    g = np.random.default_rng(7)
    m = 4
    r = g.uniform(0.15, 0.3, m)
    z = (g.standard_normal((m, m)) + 1j * g.standard_normal((m, m))) / np.sqrt(2)
    U, rr = np.linalg.qr(z)
    ph = np.diagonal(rr).copy()
    ph /= np.abs(ph)
    U = U * ph
    X, Y = U.real, U.imag
    S = np.block([[X, -Y], [Y, X]]) @ np.diag(np.concatenate([np.exp(-r), np.exp(r)]))
    cov = S @ S.T

    # CPU backend + default None repeated_sieve: works, draws unchanged, bound finite
    s1, d = smod.sample(cov, 30, cutoff=4, backend="cpu", seed=9,
                        certified_weights=True, return_diagnostics=True)
    s0 = smod.sample(cov, 30, cutoff=4, backend="cpu", seed=9, repeated_sieve=True)
    assert np.array_equal(s0, s1)
    assert np.isfinite(d["kept_mass_certified"]["total_tv_kept_bound"])

    with pytest.raises(ValueError, match="not False"):
        smod.sample(cov, 4, cutoff=4, backend="cpu", repeated_sieve=False,
                    certified_weights=True)


def test_lhaf_repeated_rejects_negative_reps():
    """Medium-1 regression: negative repetition counts are rejected before any
    backend dispatch (the GPU odometer assumes non-negative)."""
    import gbskernels

    A = np.eye(3, dtype=np.complex128)
    gam = np.zeros(3, dtype=np.complex128)
    for backend in ("cpu", "gpu"):
        if backend == "gpu" and gbskernels._load_gpu_ext() is None:
            continue
        with pytest.raises(ValueError, match="non-negative"):
            gbskernels.lhaf_repeated(A, gam, np.array([2, -1, 0], dtype=np.int32),
                                     backend=backend)
