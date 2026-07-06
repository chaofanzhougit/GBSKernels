"""Torontonian -- five-layer verification (on its physical, real-O domain).

The torontonian arises for threshold (click/no-click) detectors. Its physical
inputs are real ``O`` matrices from real Gaussian covariances; the tests use that
domain, where the value is unambiguous and all references agree.
"""

from __future__ import annotations

import itertools

import mpmath
import numpy as np
import pytest

from cpu_ref import torontonian
from highprec_ref import torontonian_mp


def real_O(n: int, seed: int, scale: float = 0.12) -> np.ndarray:
    """Random real symmetric 2n x 2n matrix of small norm (keeps I - O_S regular)."""
    g = np.random.default_rng(seed)
    M = g.standard_normal((2 * n, 2 * n)) * scale
    return (M + M.T) / 2


# --- Layer 1: independent closed forms & multiplicativity -------------------


@pytest.mark.layer1
@pytest.mark.parametrize("seed", range(6))
def test_n1_closed_form(seed):
    # tor(O) = 1/sqrt(det(I - O)) - 1 for a single mode (2x2 O).
    O = real_O(1, seed)
    expected = 1.0 / np.sqrt(np.linalg.det(np.eye(2) - O)) - 1.0
    assert torontonian(O) == pytest.approx(expected, rel=1e-10)


@pytest.mark.layer1
def test_zero_matrix_is_zero():
    # All det(I - O_S) = 1, so tor = sum_S (-1)^(n-|S|) = (1-1)^n = 0 for n>=1.
    assert torontonian(np.zeros((4, 4))) == pytest.approx(0.0, abs=1e-12)
    assert torontonian(np.zeros((6, 6))) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.layer1
@pytest.mark.parametrize("seed", range(5))
def test_direct_sum_multiplicativity(seed):
    # Decoupled modes (block structure in xxpp) -> torontonian factorizes.
    g = np.random.default_rng(seed)
    # mode A: 2x2 block at indices (0, nA+...) ; build two independent groups
    nA, nB = 1, 2
    OA = real_O(nA, seed)
    OB = real_O(nB, seed + 100)
    n = nA + nB
    O = np.zeros((2 * n, 2 * n))
    # place OA on modes [0..nA), OB on modes [nA..n) in xxpp ordering
    a_modes = list(range(nA))
    b_modes = list(range(nA, n))
    a_idx = a_modes + [m + n for m in a_modes]
    b_idx = b_modes + [m + n for m in b_modes]
    O[np.ix_(a_idx, a_idx)] = OA
    O[np.ix_(b_idx, b_idx)] = OB
    assert torontonian(O) == pytest.approx(torontonian(OA) * torontonian(OB), rel=1e-9)


# --- Layer 2: The Walrus differential oracle (real O) -----------------------

thewalrus = pytest.importorskip("thewalrus", reason="Layer 2 needs The Walrus")


@pytest.mark.layer2
@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", range(4))
def test_matches_thewalrus_real(n, seed):
    O = real_O(n, seed=50 * n + seed)
    expected = complex(thewalrus.tor(O, recursive=False))
    assert torontonian(O) == pytest.approx(expected, rel=1e-8, abs=1e-11)


@pytest.mark.layer2
@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("seed", range(3))
def test_matches_thewalrus_physical_pure_state(n, seed):
    # The real use case: O from an actual pure Gaussian state covariance.
    from thewalrus.quantum import Qmat, Xmat
    from thewalrus.random import random_covariance

    V = random_covariance(n, hbar=2, pure=True)
    Q = Qmat(V, hbar=2)
    O = (Xmat(n) @ (np.eye(2 * n) - np.linalg.inv(Q))).real
    expected = complex(thewalrus.tor(O, recursive=False))
    assert torontonian(O) == pytest.approx(expected, rel=1e-7, abs=1e-10)


# --- Layer 3: invariants ----------------------------------------------------


@pytest.mark.layer3
@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("seed", range(5))
def test_mode_permutation_invariance(n, seed):
    O = real_O(n, seed=seed)
    g = np.random.default_rng(seed ^ 0x77)
    p = g.permutation(n)
    full = np.concatenate([p, p + n])  # permute x and p indices together
    assert torontonian(O[np.ix_(full, full)]) == pytest.approx(torontonian(O), rel=1e-9)


# --- reference validation ---------------------------------------------------


@pytest.mark.layer1
@pytest.mark.parametrize("n", [1, 2, 3])
def test_mp_matches_fp64_on_real_O(n):
    O = real_O(n, seed=7)
    assert complex(torontonian_mp(O, dps=50)) == pytest.approx(torontonian(O), rel=1e-12)
