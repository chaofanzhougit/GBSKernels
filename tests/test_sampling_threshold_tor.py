"""Torontonian threshold distribution end-to-end -- the threshold corner of Layer 4.

Closes the gap that the sampler produced threshold statistics only by marginalizing
the hafnian PNR distribution. ``sampling.gbs.torontonian_threshold_probabilities``
computes the click-pattern distribution DIRECTLY from the torontonian. Validated:

1. **sums to 1 exactly** -- a genuine distribution over click patterns (no cutoff);
2. **per-pattern vs The Walrus** ``threshold_detection_prob`` (independent);
3. **equals the marginalized hafnian PNR distribution** -- a torontonian-vs-hafnian
   consistency check *through the physics*, oracle-independent (the two routes to the
   same threshold distribution must agree, up to the PNR cutoff's truncation).
"""

from __future__ import annotations

import numpy as np
import pytest

from sampling import gbs

if not hasattr(np, "find_common_type"):  # NumPy-2.0 compat shim
    np.find_common_type = lambda a, s: np.result_type(*a, *s)

pytest.importorskip("thewalrus")
import thewalrus as tw  # noqa: E402
import thewalrus.symplectic as sym  # noqa: E402

pytestmark = pytest.mark.slow  # distributional validation vs The Walrus

HBAR = 2.0


def _zero_disp_state(m: int, seed: int):
    """A zero-displacement pure GBS state: returns (B = U tanh(r) U^T, r, cov)."""
    g = np.random.default_rng(seed)
    r = g.uniform(0.2, 0.5, m)
    z = (g.standard_normal((m, m)) + 1j * g.standard_normal((m, m))) / np.sqrt(2.0)
    U, rr = np.linalg.qr(z)
    ph = np.diagonal(rr).copy(); ph /= np.abs(ph); U = U * ph
    B = U @ np.diag(np.tanh(r)) @ U.T
    S = sym.interferometer(U) @ sym.squeezing(r, np.zeros(m))
    cov = (HBAR / 2.0) * S @ S.T
    return B, r, cov


@pytest.mark.parametrize("m", [2, 3])
def test_tor_threshold_sums_to_one(m):
    _, _, cov = _zero_disp_state(m, seed=5 + m)
    probs = gbs.torontonian_threshold_probabilities(cov)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)  # exact, no truncation
    assert all(p > -1e-12 for p in probs.values())


@pytest.mark.parametrize("m", [2, 3])
def test_tor_threshold_matches_thewalrus_per_pattern(m):
    _, _, cov = _zero_disp_state(m, seed=20 + m)
    mu = np.zeros(2 * m)
    probs = gbs.torontonian_threshold_probabilities(cov)
    for c, p in probs.items():
        ref = float(np.real(tw.threshold_detection_prob(mu, cov, np.array(c), hbar=HBAR)))
        assert abs(p - ref) < 1e-9, f"click {c}: {p} vs Walrus {ref}"


@pytest.mark.parametrize("m,cutoff", [(2, 11), (3, 8)])
def test_tor_threshold_matches_marginalized_pnr(m, cutoff):
    B, r, cov = _zero_disp_state(m, seed=30 + m)
    tor_probs = gbs.torontonian_threshold_probabilities(cov)          # exact (torontonian)
    marg = gbs.threshold_probabilities(B, r, cutoff)                  # marginalized hafnian PNR
    for c, p in tor_probs.items():
        assert abs(p - marg.get(c, 0.0)) < 5e-3, f"click {c}: tor {p} vs marg {marg.get(c)}"
