"""v3 fully on-device conditional sampler -- distributional validation.

``sampler.sample(backend="gpu", resident=True)`` runs the WHOLE chain on the device (prefix
state, submatrix gather, conditional hafnians, inverse-CDF + cuRAND draw) -- no per-mode host
round trip. cuRAND != numpy RNG, so it is NOT seed-identical to the hybrid/CPU sampler; its
empirical distribution is validated against the exact GBS distribution (TV distance) and a
chi-square test, the same way the hybrid sampler is.

Builds the nanobind host-shim extension (CUDA kernels as host C++); skips cleanly where
cmake/nanobind are unavailable or the extension predates ``sample_resident``.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.layer5, pytest.mark.slow]

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


def _state(m: int, seed: int):
    """A zero-displacement pure GBS state -> (cov, r, B = U tanh(r) U^T). Modest squeezing so the
    photon distribution fits a small cutoff (the resident sampler is cap-limited to small modes/
    cutoff: 2*m*cutoff <= the hafnian cap)."""
    g = np.random.default_rng(seed)
    r = g.uniform(0.2, 0.4, m)
    z = (g.standard_normal((m, m)) + 1j * g.standard_normal((m, m))) / np.sqrt(2.0)
    U, rr = np.linalg.qr(z); ph = np.diagonal(rr).copy(); ph /= np.abs(ph); U = U * ph
    S = (np.block([[U.real, -U.imag], [U.imag, U.real]])
         @ np.block([[np.diag(np.exp(-r)), np.zeros((m, m))],
                     [np.zeros((m, m)), np.diag(np.exp(r))]]))
    return S @ S.T, r, U @ np.diag(np.tanh(r)) @ U.T


@pytest.fixture(scope="module")
def resident_sampler():
    if not _toolchain_ok():
        pytest.skip("cmake / nanobind not available")
    if not list(BUILD.glob("gbskernels_ext*.so")):
        subprocess.run(["cmake", "-S", str(REPO / "bindings"), "-B", str(BUILD),
                        "-DGBS_HOST_SHIM=ON", f"-DPython_EXECUTABLE={sys.executable}"],
                       check=True, capture_output=True, text=True)
        subprocess.run(["cmake", "--build", str(BUILD), "-j"], check=True, capture_output=True, text=True)
    sys.path.insert(0, str(BUILD))
    import gbskernels
    importlib.reload(gbskernels)
    if not gbskernels.gpu_available() or not hasattr(gbskernels._load_gpu_ext(), "sample_resident"):
        pytest.skip("extension predates sample_resident (rebuild bindings/)")
    from sampling import sampler
    importlib.reload(sampler)
    return sampler


def _tv(p: dict, q: dict) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def _emp(samps) -> dict:
    d: dict = {}
    for x in samps:
        t = tuple(int(v) for v in x); d[t] = d.get(t, 0) + 1
    return {k: v / len(samps) for k, v in d.items()}


def test_resident_sampler_matches_exact_distribution(resident_sampler):
    from sampling import gbs
    cov, r, B = _state(2, seed=5)             # 2 modes; 2*M*cutoff = 16 <= cap 20
    cutoff, N = 4, 12000
    samp = resident_sampler.sample(cov, N, cutoff=cutoff, backend="gpu", precision="fp64",
                                   resident=True, seed=7)
    assert samp.shape == (N, 2) and samp.min() >= 0 and samp.max() <= cutoff
    pats, probs = gbs.probabilities(B, r, cutoff + 1)
    exact = {k: p for k, p in zip(pats, probs) if max(k) <= cutoff}
    Z = sum(exact.values()); exact = {k: v / Z for k, v in exact.items()}
    tv = _tv(_emp(samp), exact)
    assert tv < 0.04, f"TV(resident sampler, exact) = {tv:.4f} too large (noise ~ {1/np.sqrt(N):.4f})"


def test_resident_guards(resident_sampler):
    cov, _, _ = _state(2, seed=1)
    with pytest.raises(ValueError, match="backend='gpu'"):
        resident_sampler.sample(cov, 10, cutoff=4, backend="cpu", resident=True)
    with pytest.raises(ValueError, match="fp64-only"):
        resident_sampler.sample(cov, 10, cutoff=4, backend="gpu", precision="dd", resident=True)
    with pytest.raises(ValueError, match="hafnian cap"):  # 2*M*cutoff = 2*2*8 = 32 > 20
        resident_sampler.sample(cov, 10, cutoff=8, backend="gpu", resident=True)


def test_resident_guards_are_explicit():
    """The early-status limits must be explicit errors, never silent bypasses:
    sieve, diagnostics, non-fp64, and the cap all refuse with clear messages."""
    import numpy as np
    import pytest

    from sampling import sampler as smod

    g = np.random.default_rng(0)
    m = 3
    r = g.uniform(0.15, 0.3, m)
    z = (g.standard_normal((m, m)) + 1j * g.standard_normal((m, m))) / np.sqrt(2)
    U, rr = np.linalg.qr(z)
    ph = np.diagonal(rr).copy()
    ph /= np.abs(ph)
    U = U * ph
    X, Y = U.real, U.imag
    S = np.block([[X, -Y], [Y, X]]) @ np.diag(np.concatenate([np.exp(-r), np.exp(r)]))
    cov = S @ S.T

    with pytest.raises(NotImplementedError, match="sieve"):
        smod.sample(cov, 4, cutoff=3, backend="gpu", resident=True,
                    repeated_sieve=True)
    with pytest.raises(ValueError, match="fp64-only"):
        smod.sample(cov, 4, cutoff=3, backend="gpu", resident=True, precision="auto")
    with pytest.raises(ValueError, match="diagnostics"):
        smod.sample(cov, 4, cutoff=3, backend="gpu", resident=True,
                    return_diagnostics=True)
    with pytest.raises(ValueError, match="backend='gpu'"):
        smod.sample(cov, 4, cutoff=3, backend="cpu", resident=True)
