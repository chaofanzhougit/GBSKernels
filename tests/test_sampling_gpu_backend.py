"""The sampling / physics layer runs on the GPU backend -- closing the integration
gap that it previously only used CPU defaults.

Builds the nanobind extension in host-shim mode and drives the GBS permanent,
hafnian, and torontonian sampling paths through ``backend="gpu"``, which routes the
ragged pattern matrices through a :class:`gbskernels.Workspace` (real bucketing +
device-buffer residency). Each must match the CPU backend. Skips if the toolchain
is absent. (The displaced GBS path stays CPU: its odd-photon submatrices hit the
GPU loop hafnian's even-N-only limit.)
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
def gpu_sampling():
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
    import gbskernels
    importlib.reload(gbskernels)
    assert gbskernels.gpu_available()
    from sampling import boson_sampling, gbs
    return gbs, boson_sampling


def test_gbs_pnr_probabilities_gpu_matches_cpu(gpu_sampling):
    gbs, _ = gpu_sampling
    B, r = gbs.random_gbs_kernel(m=3, seed=1)
    _, p_cpu = gbs.probabilities(B, r, cutoff=4, backend="cpu")
    _, p_gpu = gbs.probabilities(B, r, cutoff=4, backend="gpu")  # haf via Workspace
    assert np.max(np.abs(p_cpu - p_gpu)) < 1e-9


def test_boson_sampling_gpu_matches_cpu(gpu_sampling):
    _, bs = gpu_sampling
    g = np.random.default_rng(0)
    z = (g.standard_normal((4, 4)) + 1j * g.standard_normal((4, 4))) / np.sqrt(2.0)
    U, _ = np.linalg.qr(z)
    _, p_cpu = bs.probabilities(U, backend="cpu")
    _, p_gpu = bs.probabilities(U, backend="gpu")  # perm via Workspace
    assert np.max(np.abs(p_cpu - p_gpu)) < 1e-9


def test_displaced_gbs_gpu_matches_cpu(gpu_sampling):
    # Displaced GBS (loop hafnian) on the GPU -- the last GPU physics corner. Its
    # submatrices include ODD sizes (displacement gives odd photon patterns weight),
    # which the GPU loop hafnian now handles via the augmentation identity. Routed
    # through a Workspace; must match the CPU backend, and odd patterns must carry
    # real mass (so the odd-size loop hafnians are actually exercised on the GPU).
    gbs, _ = gpu_sampling
    m, cutoff = 3, 4
    B, r = gbs.random_gbs_kernel(m=m, seed=2)
    g = np.random.default_rng(9)
    alpha = 0.3 * (g.standard_normal(m) + 1j * g.standard_normal(m))
    pats, p_cpu = gbs.displaced_probabilities(B, r, alpha, cutoff, backend="cpu")
    _, p_gpu = gbs.displaced_probabilities(B, r, alpha, cutoff, backend="gpu")
    assert np.max(np.abs(p_cpu - p_gpu)) < 1e-9, "displaced GBS GPU != CPU"
    odd = np.array([sum(p) % 2 == 1 for p in pats])
    assert p_gpu[odd].sum() > 1e-2, "odd photon patterns (odd-size lhaf) must carry mass"


def test_torontonian_threshold_gpu_matches_cpu(gpu_sampling):
    gbs, _ = gpu_sampling
    pytest.importorskip("thewalrus")
    import thewalrus.symplectic as sym

    g = np.random.default_rng(3)
    m = 3
    r = g.uniform(0.2, 0.5, m)
    z = (g.standard_normal((m, m)) + 1j * g.standard_normal((m, m))) / np.sqrt(2.0)
    U, rr = np.linalg.qr(z); ph = np.diagonal(rr).copy(); ph /= np.abs(ph); U = U * ph
    cov = (2.0 / 2.0) * (sym.interferometer(U) @ sym.squeezing(r, np.zeros(m))) @ \
          (sym.interferometer(U) @ sym.squeezing(r, np.zeros(m))).T  # hbar=2
    cpu = gbs.torontonian_threshold_probabilities(cov, backend="cpu")
    gpu = gbs.torontonian_threshold_probabilities(cov, backend="gpu")  # tor via Workspace
    assert max(abs(cpu[c] - gpu[c]) for c in cpu) < 1e-9
    assert sum(gpu.values()) == pytest.approx(1.0, abs=1e-9)
