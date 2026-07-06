"""Displaced GBS (loop hafnian) end-to-end -- the displaced corner of Layer 4.

Closes the documented gap that the sampler only validated zero-displacement PNR GBS.
Validates ``sampling.gbs.displaced_probabilities`` (the loop-hafnian path) three ways:

1. **alpha = 0 reduces exactly** to the hafnian ``probabilities`` (zero loop weights
   -> the loop hafnian is the hafnian) -- ties it to the already-validated path.
2. **sums to ~1** over a photon cutoff (truncation error only).
3. **per-pattern agreement with The Walrus**, whose ``probabilities`` uses an
   independent (multidimensional-Hermite) method -- not the loop hafnian -- so this
   is a genuine cross-method check (docs/DESIGN.md §1.3 #3, sec.8 Layer 4).

(3) needs The Walrus's high-level path, which calls the NumPy-2.0-removed
``np.find_common_type``; we apply our own verified fix as a shim
(``np.result_type``) so the oracle runs in this environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from sampling import gbs

# The Walrus 0.21 high-level path uses the removed np.find_common_type under NumPy
# 2.x; apply the compat fix as a shim so the oracle runs (same semantics).
if not hasattr(np, "find_common_type"):
    np.find_common_type = lambda array_types, scalar_types: np.result_type(
        *array_types, *scalar_types
    )

pytest.importorskip("thewalrus")
import thewalrus.symplectic as sym  # noqa: E402
from thewalrus import quantum as twq  # noqa: E402

pytestmark = pytest.mark.slow  # distributional validation vs The Walrus

HBAR = 2.0


def _displaced_state(m: int, seed: int):
    """A displaced pure GBS state. Returns (B = U tanh(r) U^T, r, alpha, mu, cov),
    with alpha the Husimi displacement in The Walrus convention."""
    g = np.random.default_rng(seed)
    r = g.uniform(0.1, 0.45, m)
    z = (g.standard_normal((m, m)) + 1j * g.standard_normal((m, m))) / np.sqrt(2.0)
    U, rr = np.linalg.qr(z)
    ph = np.diagonal(rr).copy(); ph /= np.abs(ph); U = U * ph
    B = U @ np.diag(np.tanh(r)) @ U.T
    a_in = 0.3 * (g.standard_normal(m) + 1j * g.standard_normal(m))
    mu = np.sqrt(2.0 * HBAR) * np.concatenate([a_in.real, a_in.imag])
    S = sym.interferometer(U) @ sym.squeezing(r, np.zeros(m))
    cov = (HBAR / 2.0) * S @ S.T
    alpha = twq.complex_to_real_displacements(mu, hbar=HBAR)[:m]
    return B, r, alpha, mu, cov


def test_displaced_reduces_to_hafnian_at_zero_displacement():
    B, r = gbs.random_gbs_kernel(m=3, seed=2)
    _, p_haf = gbs.probabilities(B, r, cutoff=4)
    _, p_dis = gbs.displaced_probabilities(B, r, np.zeros(3), cutoff=4)
    assert np.allclose(p_dis, p_haf, atol=1e-12), "alpha=0 must reduce to the hafnian path"


# Cutoffs chosen just above the truncation tolerance: this displaced state's mean
# photon number is < ~1.4/mode, so cutoff 5 captures >99% of the mass. Deeper cutoffs
# only added (7,7,7)-style patterns that need a 42x42 loop hafnian in pure Python yet
# contribute ~nothing -- 390s for no extra validation (see git history).
@pytest.mark.parametrize("m,cutoff", [(2, 5), (3, 5)])
def test_displaced_sums_to_one(m, cutoff):
    B, r, alpha, mu, cov = _displaced_state(m, seed=10 + m)
    total = gbs.displaced_total(B, r, alpha, cutoff)
    assert total == pytest.approx(1.0, abs=2e-2)  # finite-cutoff truncation


# Per-pattern agreement with The Walrus is exact at any cutoff (independent of
# truncation), so a modest cutoff validates it just as well and far faster.
@pytest.mark.parametrize("m,cutoff", [(1, 8), (2, 6), (3, 5)])
def test_displaced_matches_thewalrus_per_pattern(m, cutoff):
    B, r, alpha, mu, cov = _displaced_state(m, seed=100 + m)
    # the self-built kernel is the negative of the convention's A-matrix block (the
    # squeezing-quadrature sign displaced_probabilities accounts for internally)
    assert np.allclose(-B, twq.Amat(cov, hbar=HBAR).conj()[:m, :m], atol=1e-10)

    patterns, p_mine = gbs.displaced_probabilities(B, r, alpha, cutoff)
    p_ref = np.real(twq.probabilities(mu, cov, cutoff, hbar=HBAR)).reshape(-1)
    assert len(patterns) == p_ref.size
    assert np.max(np.abs(p_mine - p_ref)) < 1e-9, "displaced GBS != The Walrus per pattern"
    # displacement gives odd-photon patterns nonzero weight (unlike zero-displacement)
    odd = np.array([sum(p) % 2 == 1 for p in patterns])
    assert p_mine[odd].sum() > 1e-3
