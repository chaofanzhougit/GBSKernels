"""The exact real torontonian matrix vs the legacy entrywise real cast.

``sampling.gbs.threshold_O_xxpp`` builds the torontonian input in the
quadrature (xxpp) basis, where it is real BY CONSTRUCTION and provably
torontonian-equivalent to the complex-basis ``O = I - Q^{-1}`` (restriction to
a click pattern keeps mode pairs in both blocks, which commutes with the fixed
block unitary relating the bases, so every determinant in the
inclusion-exclusion is identical).

The legacy shortcut ``Re(I - Q^{-1})`` is a DIFFERENT matrix whenever the state
has complex pair correlations. These tests keep both facts true at the library
level, on synthetic states small enough for mpmath:

1. exactness: tor(Ox_S) equals the complex-basis torontonian and reproduces
   the validated ``torontonian_threshold_probabilities`` distribution;
2. the legacy cast stays measurably wrong on a complex-correlation state
   (so the bug cannot silently return);
3. for real-structured states the two coincide -- why the sampler-level
   differential tests against The Walrus never caught the cast.
"""

from __future__ import annotations

import numpy as np
import pytest

from highprec_ref import torontonian_mp
from sampling import gbs

HBAR = 2.0


def test_threshold_helper_is_part_of_the_public_module_surface():
    assert "threshold_O_xxpp" in gbs.__all__


def _lossy_complex_state(m: int, seed: int, eta: float = 0.6) -> np.ndarray:
    """xxpp covariance of a lossy GBS state with a COMPLEX interferometer
    (nonzero xp correlations -> complex <aa> -> max|imag(I - Q^{-1})| = O(0.1),
    the structure that exposes the legacy cast)."""
    g = np.random.default_rng(seed)
    r = g.uniform(0.5, 0.9, m)
    z = (g.standard_normal((m, m)) + 1j * g.standard_normal((m, m))) / np.sqrt(2.0)
    U, rr = np.linalg.qr(z)
    ph = np.diagonal(rr).copy()
    ph /= np.abs(ph)
    U = U * ph
    nb = np.sinh(r) ** 2
    mm = np.sinh(r) * np.cosh(r)
    aiaj = U.T @ np.diag(mm) @ U
    aidaj = U.conj().T @ np.diag(nb) @ U
    x = np.eye(m) + np.real(aidaj + aidaj.conj().T) + 2 * np.real(aiaj)
    p = np.eye(m) + np.real(aidaj + aidaj.conj().T) - 2 * np.real(aiaj)
    xp = 2 * np.imag(aiaj) + 2 * np.imag(aidaj)
    cov = np.block([[x, xp], [xp.T, p]])
    return eta * cov + (1 - eta) * np.eye(2 * m)  # uniform loss keeps it Gaussian


def _pattern_prob_via_Ox(cov: np.ndarray, c: tuple[int, ...]) -> float:
    m = len(c)
    Ox, lsd = gbs.threshold_O_xxpp(cov, hbar=HBAR)
    S = [i for i in range(m) if c[i]]
    idx = S + [i + m for i in S]
    t = torontonian_mp(np.ascontiguousarray(Ox[np.ix_(idx, idx)]), dps=40)
    return float(t) * float(np.exp(-lsd))


@pytest.mark.parametrize("m", [3, 4])
def test_Ox_equals_complex_basis_torontonian(m):
    """tor(Ox_S) == tor(O_S) for the complex-basis O, every pattern."""
    cov = _lossy_complex_state(m, seed=11 + m)
    Ox, _ = gbs.threshold_O_xxpp(cov, hbar=HBAR)
    Q = gbs._qmat(cov, HBAR)
    Oc = np.eye(2 * m) - np.linalg.inv(Q)
    assert np.abs(Oc.imag).max() > 0.02  # the state genuinely exercises the cast
    # Ox and Oc carry independent fp64 inversion noise (~1e-15 at the input),
    # so their EXACT torontonians agree to ~1e-8 relative, not to mpmath
    # precision -- still six orders sharper than the legacy-cast error.
    for bits in range(1, 2 ** m):
        S = [i for i in range(m) if (bits >> i) & 1]
        idx = S + [i + m for i in S]
        tx = torontonian_mp(np.ascontiguousarray(Ox[np.ix_(idx, idx)]), dps=40)
        tc = complex(torontonian_mp(np.ascontiguousarray(Oc[np.ix_(idx, idx)]), dps=40))
        assert abs(tc.imag) <= 1e-12 * max(1.0, abs(tc.real))
        assert float(tx) == pytest.approx(tc.real, rel=1e-8)


def test_Ox_is_exactly_symmetric_binary64():
    """Inversion roundoff must not leak an ambiguous triangle to tor_single."""
    cov = _lossy_complex_state(8, seed=19)
    Ox, _ = gbs.threshold_O_xxpp(cov, hbar=HBAR)

    assert Ox.dtype == np.float64
    assert Ox.flags.c_contiguous
    assert np.all(np.isfinite(Ox))
    assert np.array_equal(Ox, Ox.T)


@pytest.mark.parametrize("m", [3])
def test_Ox_reproduces_validated_threshold_distribution(m):
    """P(S) = tor(Ox_S) e^{-log sqrt det Q} matches the Walrus-validated
    complex batched path for every pattern (and hence sums to 1)."""
    cov = _lossy_complex_state(m, seed=29)
    probs = gbs.torontonian_threshold_probabilities(cov, hbar=HBAR)
    for c, p_ref in probs.items():
        if sum(c) == 0:
            continue
        assert _pattern_prob_via_Ox(cov, c) == pytest.approx(p_ref, rel=1e-10)


def test_legacy_real_cast_is_a_different_matrix():
    """Re(I - Q^{-1}) must keep FAILING on complex-correlation states; if this
    test ever breaks, someone reintroduced the pre-2026-07-10 cast bug."""
    m = 4
    cov = _lossy_complex_state(m, seed=15)
    Q = gbs._qmat(cov, HBAR)
    O_legacy = np.real(np.eye(2 * m) - np.linalg.inv(Q))
    c = tuple([1] * m)
    S = list(range(m))
    idx = S + [i + m for i in S]
    t_legacy = float(torontonian_mp(np.ascontiguousarray(O_legacy[np.ix_(idx, idx)]), dps=40))
    p_exact = _pattern_prob_via_Ox(cov, c)
    _, lsd = gbs.threshold_O_xxpp(cov, hbar=HBAR)
    p_legacy = t_legacy * float(np.exp(-lsd))
    assert abs(p_legacy - p_exact) / p_exact > 0.05


def test_real_structured_state_makes_the_cast_harmless():
    """With a REAL interferometer the complex-basis O is already real, the two
    forms coincide, and the legacy cast changed nothing -- which is why the
    sampler-level differential tests never caught it."""
    m = 3
    g = np.random.default_rng(7)
    r = g.uniform(0.4, 0.8, m)
    U, _ = np.linalg.qr(g.standard_normal((m, m)))  # real orthogonal
    nb, mm = np.sinh(r) ** 2, np.sinh(r) * np.cosh(r)
    aiaj = U.T @ np.diag(mm) @ U
    aidaj = U.T @ np.diag(nb) @ U
    x = np.eye(m) + 2 * aidaj + 2 * aiaj
    p = np.eye(m) + 2 * aidaj - 2 * aiaj
    cov = 0.7 * np.block([[x, np.zeros((m, m))], [np.zeros((m, m)), p]]) \
        + 0.3 * np.eye(2 * m)
    Q = gbs._qmat(cov, HBAR)
    Oc = np.eye(2 * m) - np.linalg.inv(Q)
    assert np.abs(Oc.imag).max() < 1e-12
    # Ox and Re(Oc) live in different bases (their ENTRIES differ), but for a
    # real-structured state Re(Oc) == Oc, so the legacy cast changed no VALUE:
    Ox, _ = gbs.threshold_O_xxpp(cov, hbar=HBAR)
    idx = list(range(m)) + [i + m for i in range(m)]
    t_cast = complex(torontonian_mp(np.ascontiguousarray(Oc.real[np.ix_(idx, idx)]), dps=40)).real
    t_exact = float(torontonian_mp(np.ascontiguousarray(Ox[np.ix_(idx, idx)]), dps=40))
    assert t_exact == pytest.approx(t_cast, rel=1e-8)
