"""Layer 4 -- end-to-end Gaussian boson sampling via the hafnian kernels.

Tests the hafnian *in its actual scientific use*: computing GBS detection
probabilities. The headline check is the physical sum-to-one law over a
truncated Fock space, which is independent of any external library and fails
loudly if the hafnians are even slightly wrong. Per-element agreement with The
Walrus's own hafnian (Layer 2) and marginal consistency round it out.

(The Walrus 0.21's high-level ``probabilities``/``state_vector`` path is
unavailable under NumPy 2.0 -- it calls the removed ``np.find_common_type`` in
its Hermite-multidimensional code -- so the end-to-end distribution is validated
via sum-to-one + the hafnian oracle rather than against that broken path.)
"""

from __future__ import annotations

import numpy as np
import pytest

from sampling.gbs import (
    fock_patterns,
    probabilities,
    random_gbs_kernel,
    threshold_probabilities,
    threshold_total,
    total_probability,
)

pytestmark = [pytest.mark.layer4, pytest.mark.slow]  # statistical, mpmath/Walrus-heavy


@pytest.mark.parametrize("m", [1, 2, 3])
@pytest.mark.parametrize("seed", range(4))
def test_probabilities_sum_to_one(m, seed):
    B, r = random_gbs_kernel(m, seed=seed)
    cutoff = 14 if m <= 2 else 9
    assert total_probability(B, r, cutoff) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("m", [1, 2, 3])
@pytest.mark.parametrize("seed", range(3))
def test_probabilities_nonnegative_and_real(m, seed):
    B, r = random_gbs_kernel(m, seed=seed)
    _, probs = probabilities(B, r, cutoff=8)
    assert probs.dtype == np.float64
    assert (probs >= 0.0).all()


@pytest.mark.parametrize("seed", range(3))
def test_odd_total_photon_number_has_zero_probability(seed):
    # Zero-displacement Gaussian states emit photons in pairs: P = 0 for odd sum.
    B, r = random_gbs_kernel(3, seed=seed)
    patterns, probs = probabilities(B, r, cutoff=5)
    for nbar, p in zip(patterns, probs):
        if sum(nbar) % 2 == 1:
            assert p == 0.0


def test_vacuum_probability_is_product_of_sech():
    # P(0,...,0) = 1 / prod cosh r_i  (haf of empty matrix = 1).
    B, r = random_gbs_kernel(3, seed=1)
    patterns, probs = probabilities(B, r, cutoff=6)
    p0 = probs[patterns.index((0, 0, 0))]
    assert p0 == pytest.approx(1.0 / np.prod(np.cosh(r)), rel=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_batched_equals_looped(seed):
    # The batched hafnian path must equal singly-evaluated hafnians (the GBS
    # workload is exactly haf_batched; docs/DESIGN.md §7/sec.8).
    import gbskernels
    from sampling.gbs import _submatrix

    B, r = random_gbs_kernel(2, seed=seed)
    patterns = [p for p in fock_patterns(2, 8) if sum(p) % 2 == 0]
    mats = [_submatrix(B, p) for p in patterns]
    batched = gbskernels.haf_batched(mats, precision="fp64")
    looped = np.array([gbskernels.haf(M, precision="fp64") for M in mats])
    assert np.array_equal(batched, looped)


@pytest.mark.parametrize("m,seed", [(2, 0), (2, 1), (3, 0)])
def test_per_element_hafnian_matches_thewalrus(m, seed):
    # Each GBS amplitude's hafnian must match The Walrus's hafnian (Layer-2
    # oracle applied inside the Layer-4 pipeline).
    thewalrus = pytest.importorskip("thewalrus")
    from sampling.gbs import _submatrix
    import gbskernels

    B, r = random_gbs_kernel(m, seed=seed)
    for nbar in fock_patterns(m, 4):
        if sum(nbar) % 2 or sum(nbar) == 0:
            continue
        M = _submatrix(B, nbar)
        assert gbskernels.haf(M) == pytest.approx(complex(thewalrus.hafnian(M)), rel=1e-7, abs=1e-12)


@pytest.mark.parametrize("m", [1, 2, 3])
@pytest.mark.parametrize("seed", range(3))
def test_threshold_detection_distribution_valid(m, seed):
    # Threshold (click/no-click) statistics, marginalized from the hafnian PNR
    # distribution, form a valid distribution over the 2^m click patterns.
    B, r = random_gbs_kernel(m, seed=seed)
    cutoff = 14 if m <= 2 else 9
    thr = threshold_probabilities(B, r, cutoff)
    assert len(thr) <= 2 ** m
    assert all(v >= 0.0 for v in thr.values())
    assert threshold_total(B, r, cutoff) == pytest.approx(1.0, abs=1e-3)


def test_threshold_no_click_equals_vacuum_probability():
    # The all-no-click outcome is exactly the vacuum Fock probability 1/prod cosh r.
    B, r = random_gbs_kernel(3, seed=2)
    thr = threshold_probabilities(B, r, cutoff=8)
    assert thr[(0, 0, 0)] == pytest.approx(1.0 / np.prod(np.cosh(r)), rel=1e-9)


@pytest.mark.parametrize("seed", range(3))
def test_marginal_consistency(seed):
    # Summing the joint distribution over mode 2 gives mode 1's marginal, which
    # must itself be a valid (sums-to-~1) single-mode distribution.
    B, r = random_gbs_kernel(2, seed=seed)
    patterns, probs = probabilities(B, r, cutoff=14)
    marg0 = {}
    for (n0, n1), p in zip(patterns, probs):
        marg0[n0] = marg0.get(n0, 0.0) + p
    assert sum(marg0.values()) == pytest.approx(1.0, abs=1e-3)
    assert all(v >= 0.0 for v in marg0.values())
