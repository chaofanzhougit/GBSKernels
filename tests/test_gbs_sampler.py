"""Conditional GBS sampler -- distributional validation (the real-workload capstone).

``sampling.sampler.sample`` draws photon-number samples by the reduced-covariance
chain rule, routing each mode's conditional-probability hafnian batch through a
``gbskernels.Workspace`` on the GPU backend. Validated against the exact distribution
and The Walrus's independent sampler with **TV distance** and a **chi-square** test,
and the GPU (host-shim) path is checked to reproduce the CPU samples bit-for-bit.

The Walrus high-level path needs a NumPy-2.0 compatibility shim, applied as a
shim so the exact-distribution + oracle-sampler comparisons run here.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.layer5, pytest.mark.slow]  # distributional, Walrus sampler oracle

if not hasattr(np, "find_common_type"):
    np.find_common_type = lambda a, s: np.result_type(*a, *s)

pytest.importorskip("thewalrus")
import thewalrus.symplectic as sym  # noqa: E402
from thewalrus.samples import hafnian_sample_state  # noqa: E402

from sampling import gbs, sampler  # noqa: E402

HBAR = 2.0
REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "bindings" / "build_host"


def _state(m: int, seed: int):
    """A zero-displacement pure GBS state -> (B = U tanh(r) U^T, r, cov).

    Modest squeezing keeps the chain's reduced submatrices small so the pure-Python
    cpu_ref hafnian (the test oracle) stays fast -- the GPU kernel is what makes the
    deep-photon regime cheap, measured in bench.sampler_throughput on a device."""
    g = np.random.default_rng(seed)
    r = g.uniform(0.15, 0.35, m)
    z = (g.standard_normal((m, m)) + 1j * g.standard_normal((m, m))) / np.sqrt(2.0)
    U, rr = np.linalg.qr(z); ph = np.diagonal(rr).copy(); ph /= np.abs(ph); U = U * ph
    B = U @ np.diag(np.tanh(r)) @ U.T
    S = sym.interferometer(U) @ sym.squeezing(r, np.zeros(m))
    return B, r, (HBAR / 2.0) * S @ S.T


def _emp(samps) -> dict:
    d: dict = {}
    for x in samps:
        t = tuple(int(v) for v in x)
        d[t] = d.get(t, 0) + 1
    n = len(samps)
    return {k: v / n for k, v in d.items()}


def _tv(p: dict, q: dict) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def test_sampler_tv_vs_exact_distribution():
    B, r, cov = _state(3, seed=11)
    cutoff, N = 4, 9000
    samp = sampler.sample(cov, N, cutoff=cutoff, seed=1)
    pats, probs = gbs.probabilities(B, r, cutoff + 1)  # exclusive cutoff -> max photons = cutoff
    exact = {k: p for k, p in zip(pats, probs) if max(k) <= cutoff}
    Z = sum(exact.values())
    exact = {k: v / Z for k, v in exact.items()}
    tv = _tv(_emp(samp), exact)
    assert tv < 0.03, f"TV(sampler, exact) = {tv:.4f} too large (noise ~ {1/np.sqrt(N):.4f})"


def test_sampler_matches_thewalrus_tv_and_chisquare():
    _, _, cov = _state(3, seed=22)
    cutoff, N = 4, 9000
    mine = sampler.sample(cov, N, cutoff=cutoff, seed=2)
    wal = np.asarray(hafnian_sample_state(cov, N, cutoff=cutoff + 1))
    assert _tv(_emp(mine), _emp(wal)) < 0.03

    # chi-square: are the two samplers' total-photon distributions consistent?
    from scipy import stats
    tmax = int(max(mine.sum(1).max(), wal.sum(1).max()))
    om = np.array([np.sum(mine.sum(1) == t) for t in range(tmax + 1)], float)
    ow = np.array([np.sum(wal.sum(1) == t) for t in range(tmax + 1)], float)
    keep = (om + ow) >= 10                       # merge sparse tail bins
    table = np.vstack([om[keep], ow[keep]])
    _, p, _, _ = stats.chi2_contingency(table)
    assert p > 1e-3, f"chi-square rejects sampler==Walrus (p={p:.2e})"


def test_sampler_normalization_and_shape():
    _, _, cov = _state(3, seed=33)
    samp = sampler.sample(cov, 2000, cutoff=4, seed=7)
    assert samp.shape == (2000, 3)
    assert samp.min() >= 0 and samp.max() <= 4      # within the cutoff
    rate = sampler.samples_per_second(cov, 1000, cutoff=4, backend="cpu")
    assert rate["samples_per_sec"] > 0 and rate["modes"] == 3


# --- GPU backend (host-shim): the chain runs through the Workspace, matches CPU ---

def _toolchain_ok() -> bool:
    if shutil.which("cmake") is None:
        return False
    try:
        import nanobind  # noqa: F401
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def gpu_sampler():
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
    assert gbskernels.gpu_available()
    importlib.reload(sampler)
    return sampler


def test_sampler_gpu_matches_cpu(gpu_sampler):
    _, _, cov = _state(3, seed=44)
    N = 4000
    cpu = gpu_sampler.sample(cov, N, cutoff=4, backend="cpu", seed=3)
    # repeated_sieve=False pins the Workspace/powertrace path: this test asserts
    # ROUTING equality (GPU Workspace == CPU, bit-for-bit); the auto-sieve default
    # is a different evaluator with its own distributional tests (test_repeated).
    gpu = gpu_sampler.sample(cov, N, cutoff=4, backend="gpu", seed=3,
                             repeated_sieve=False)  # via Workspace
    assert np.array_equal(cpu, gpu), "GPU-routed sampler must reproduce the CPU samples"
