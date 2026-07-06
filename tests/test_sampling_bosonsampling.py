"""Layer 4 -- statistical / end-to-end via standard boson sampling.

Tests the permanent *in its actual scientific use*: as the amplitude of a boson
sampling event. By unitarity the output probabilities form a valid distribution
(sum to 1) -- an end-to-end correctness check that depends on no external
library, and that drives the batched API on a sampling-shaped workload.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench._inputs import haar_unitary
from sampling import output_configurations, probabilities, total_probability

import gbskernels

pytestmark = pytest.mark.layer4

CASES = [(3, 2), (4, 2), (4, 3), (5, 3), (3, 3), (6, 3)]


@pytest.mark.parametrize("m,n", CASES)
@pytest.mark.parametrize("seed", range(4))
def test_probabilities_sum_to_one(m, n, seed):
    U = haar_unitary(m, seed=seed)
    assert total_probability(U, in_modes=tuple(range(n))) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("m,n", CASES)
@pytest.mark.parametrize("seed", range(4))
def test_probabilities_are_nonnegative_and_real(m, n, seed):
    U = haar_unitary(m, seed=seed)
    _, probs = probabilities(U, in_modes=tuple(range(n)))
    assert probs.dtype == np.float64
    assert (probs >= -1e-15).all()


def test_uniform_input_on_identity_is_deterministic_pattern():
    # On the identity interferometer, photons stay put: P(input pattern) = 1.
    U = np.eye(4, dtype=np.complex128)
    configs, probs = probabilities(U, in_modes=(0, 1))
    p = dict(zip(configs, probs))
    assert p[(0, 1)] == pytest.approx(1.0, abs=1e-12)
    assert sum(probs) == pytest.approx(1.0, abs=1e-12)


def test_batched_matches_looped_on_sampling_submatrices():
    # The sampling layer's batched permanents must equal singly-evaluated ones.
    U = haar_unitary(5, seed=2)
    in_modes = (0, 1, 2)
    configs = output_configurations(5, 3)
    mats = [U[np.ix_(np.array(t), np.array(in_modes))] for t in configs]
    batched = gbskernels.perm_batched(mats, precision="fp64")
    looped = np.array([gbskernels.perm(M, precision="fp64") for M in mats])
    assert np.array_equal(batched, looped)
