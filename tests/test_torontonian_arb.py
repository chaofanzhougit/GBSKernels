"""Focused tests for the independent Arb torontonian oracle."""

from __future__ import annotations

import json
from fractions import Fraction

import numpy as np
import pytest
from flint import ctx

from highprec_ref.torontonian_arb import (
    ArbDomainError,
    ArbPrecisionError,
    DyadicEndpoint,
    torontonian_arb,
    torontonian_block_diagonal_arb,
)

pytestmark = pytest.mark.layer1


def _xxpp_from_mode_blocks(blocks: np.ndarray) -> np.ndarray:
    n_modes = len(blocks)
    matrix = np.zeros((2 * n_modes, 2 * n_modes), dtype=np.float64)
    for mode, block in enumerate(blocks):
        indices = (mode, mode + n_modes)
        matrix[np.ix_(indices, indices)] = block
    return matrix


def test_dense_diagonal_interval_contains_exact_binary64_closed_form():
    value = 0.2
    n_modes = 3
    matrix = value * np.eye(2 * n_modes)

    result = torontonian_arb(matrix, target_bits=90)
    exact_value = Fraction.from_float(value)
    closed_form = (exact_value / (1 - exact_value)) ** n_modes

    assert result.contains(closed_form)
    assert result.subset_count == 1 << n_modes
    assert result.method == "dense-subset-determinants"
    assert result.minimum_determinant_lower.to_fraction() > 0


def test_factorized_blocks_match_dense_xxpp_definition():
    # Both determinants are rational squares: the exact factors are 1 and 1/3.
    blocks = np.array(
        [
            [[3 / 8, 3 / 8], [3 / 8, 3 / 8]],
            [[3 / 16, 5 / 16], [5 / 16, 3 / 16]],
        ],
        dtype=np.float64,
    )
    dense = torontonian_arb(_xxpp_from_mode_blocks(blocks), target_bits=100)
    factorized = torontonian_block_diagonal_arb(blocks, target_bits=100)

    assert dense.contains(Fraction(1, 3))
    assert factorized.contains(Fraction(1, 3))
    assert dense.lower.to_fraction() <= factorized.upper.to_fraction()
    assert factorized.lower.to_fraction() <= dense.upper.to_fraction()


def test_adaptive_precision_and_context_restoration():
    old_precision = ctx.prec
    result = torontonian_arb(
        0.2 * np.eye(4),
        initial_precision_bits=64,
        max_precision_bits=256,
        target_bits=100,
    )

    assert result.precision_bits > 64
    assert ctx.prec == old_precision


def test_precision_cap_fails_closed_without_an_enclosure():
    with pytest.raises(ArbPrecisionError, match="64 bits"):
        torontonian_arb(
            0.2 * np.eye(4),
            initial_precision_bits=64,
            max_precision_bits=64,
            target_bits=100,
        )


def test_endpoints_roundtrip_through_strict_json_without_rounding():
    result = torontonian_arb(0.25 * np.eye(2), target_bits=80)
    payload = result.to_dict()
    encoded = json.dumps(payload, allow_nan=False)
    decoded = json.loads(encoded)

    lower = DyadicEndpoint.from_dict(decoded["lower"])
    upper = DyadicEndpoint.from_dict(decoded["upper"])
    assert lower == result.lower
    assert upper == result.upper
    assert isinstance(decoded["lower"]["mantissa"], str)


def test_factorized_reference_scales_to_32_modes():
    blocks = np.repeat((0.25 * np.eye(2))[None, :, :], 32, axis=0)
    result = torontonian_block_diagonal_arb(blocks, target_bits=90)

    assert result.contains(Fraction(1, 3) ** 32)
    assert result.n_modes == 32
    assert result.subset_count == 1 << 32
    assert result.method == "factorized-mode-blocks"
    assert result.minimum_determinant_lower.to_fraction() <= Fraction(9, 16) ** 32


def test_nonpositive_determinant_is_a_domain_error():
    with pytest.raises(ArbDomainError, match="nonpositive"):
        torontonian_arb(np.eye(2))


def test_minimum_determinant_bound_includes_the_empty_subset():
    dense = torontonian_arb(-0.25 * np.eye(2))
    blocks = torontonian_block_diagonal_arb(
        np.array([-0.25 * np.eye(2, dtype=np.float64)])
    )

    assert dense.minimum_determinant_lower.to_fraction() == 1
    assert blocks.minimum_determinant_lower.to_fraction() == 1


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 3)),
        np.zeros((3, 3)),
        np.array([[0.0, 1.0j], [1.0j, 0.0]]),
        np.array([[np.nan, 0.0], [0.0, 0.0]]),
    ],
)
def test_dense_reference_rejects_invalid_inputs(bad):
    with pytest.raises(ValueError):
        torontonian_arb(bad)
