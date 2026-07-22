from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "examples" / "jiuzhang"
sys.path.insert(0, str(HERE))

import coherence_family as cf  # noqa: E402


def test_classical_and_physical_boundaries():
    n = np.array([0.1, 0.7])
    classical, physical = cf.coherence_bounds(n)
    assert np.all(classical == n)
    assert np.all(physical > classical)
    assert cf.classify_moment(n, classical)["classical_positive_p"] is True
    assert cf.classify_moment(n, physical)["classical_positive_p"] is False
    with pytest.raises(ValueError, match="outside"):
        cf.classify_moment(n, physical * 1.01)


def test_pair_covariance_has_expected_moments():
    n = np.array([0.25])
    m = np.array([0.2])
    cov = cf.input_cov_xxpp_from_pair_moments(n, m)
    expected = np.array([
        [1.5, 0.4, 0.0, 0.0],
        [0.4, 1.5, 0.0, 0.0],
        [0.0, 0.0, 1.5, -0.4],
        [0.0, 0.0, -0.4, 1.5],
    ])
    assert np.allclose(cov, expected)


def test_coherence_coordinate_is_bounded():
    n = np.array([0.2, 1.0])
    assert np.allclose(cf.anomalous_moment(n, 0), 0)
    assert np.allclose(cf.anomalous_moment(n, 1), np.sqrt(n * (n + 1)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        cf.anomalous_moment(n, 1.1)


def test_classical_excess_parameterization_crosses_boundary_at_zero():
    n = np.array([0.2, 1.0])
    assert np.allclose(cf.excess_moment(n, 0), n)
    assert cf.classify_moment(n, cf.excess_moment(n, -0.5))["classical_positive_p"]
    assert not cf.classify_moment(n, cf.excess_moment(n, 0.5))["classical_positive_p"]


def test_excess_endpoints_equal_published_pair_construction():
    import q7_construction as q7

    r = np.array([0.2, 0.5])
    n = np.sinh(r) ** 2
    pairing = q7.pair_beamsplitter(2 * len(r))
    for kind, eta in (("squashed", 0.0), ("squeezed", 1.0)):
        expected = pairing @ q7.input_cov_xxpp(r, kind) @ pairing.T
        actual = cf.input_cov_xxpp_from_pair_moments(n, cf.excess_moment(n, eta))
        assert np.allclose(actual, expected, atol=2e-15)
