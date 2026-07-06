"""Validate the hafnian arbitrary-precision reference itself (Layer 1)."""

from __future__ import annotations

from fractions import Fraction
from math import prod

import mpmath
import numpy as np
import pytest

from conftest import symmetric_matrix
from cpu_ref import hafnian_naive
from highprec_ref import hafnian_mp

pytestmark = pytest.mark.layer1


def _exact_haf_via_rationals(A: np.ndarray, dps: int):
    """Hafnian from EXACT rational inputs -- independent of mpmath's float path,
    pinning that hafnian_mp retains full FP64 inputs (the workprec guard)."""
    N = A.shape[0]
    with mpmath.workdps(dps):
        def to_mp(x):
            f = Fraction(float(x))
            return mpmath.mpf(f.numerator) / mpmath.mpf(f.denominator)

        E = [[mpmath.mpc(to_mp(A[i, j].real), to_mp(A[i, j].imag)) for j in range(N)]
             for i in range(N)]

        def rec(rem):
            if not rem:
                return mpmath.mpc(1)
            i, rest = rem[0], rem[1:]
            tot = mpmath.mpc(0)
            for k in range(len(rest)):
                tot += E[i][rest[k]] * rec(rest[:k] + rest[k + 1 :])
            return tot

        return rec(tuple(range(N)))


@pytest.mark.parametrize("n_pairs", range(1, 5))
def test_mp_all_ones_is_exact_double_factorial(n_pairs):
    N = 2 * n_pairs
    assert hafnian_mp(np.ones((N, N)), dps=50) == mpmath.mpf(prod(range(1, N, 2)))


@pytest.mark.parametrize("N", [2, 4, 6])
@pytest.mark.parametrize("seed", range(4))
def test_mp_agrees_with_naive(N, seed):
    A = symmetric_matrix(N, seed=43 * N + seed)
    assert abs(complex(hafnian_mp(A, dps=60)) - hafnian_naive(A)) < 1e-12 * (
        abs(hafnian_naive(A)) + 1.0
    )


@pytest.mark.parametrize("N", [4, 6])
def test_mp_matches_exact_rational_ground_truth(N):
    A = symmetric_matrix(N, seed=7)
    with mpmath.workdps(70):
        ref = hafnian_mp(A, dps=70)
        truth = _exact_haf_via_rationals(A, 70)
        assert abs(ref - truth) <= mpmath.mpf(10) ** -55 * abs(truth)
