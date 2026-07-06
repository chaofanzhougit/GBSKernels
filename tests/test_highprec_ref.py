"""Validate the arbitrary-precision ground truth itself.

The mpmath reference is trusted by Layer 5, so it must earn that trust against
the same independent footing as everything else: the definition (naive),
closed forms, and exactness on integer inputs.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import mpmath
import numpy as np
import pytest

from bench._inputs import make_cancellation_matrix
from conftest import complex_matrix
from cpu_ref import permanent_naive
from highprec_ref import permanent_mp

pytestmark = pytest.mark.layer1


def _exact_perm_via_rationals(A: np.ndarray, dps: int):
    """Permanent from EXACT rational inputs (Fraction is lossless on binary64).

    Independent of mpmath's float conversion path, so it pins whether
    ``permanent_mp`` actually retains the full FP64 inputs or silently truncates
    them to the global working precision -- the exact bug the workprec guard in
    ``_to_mp_matrix`` prevents.
    """
    n = A.shape[0]
    with mpmath.workdps(dps):
        def to_mp(x: float):
            f = Fraction(float(x))
            return mpmath.mpf(f.numerator) / mpmath.mpf(f.denominator)

        E = [[mpmath.mpc(to_mp(A[i, j].real), to_mp(A[i, j].imag)) for j in range(n)]
             for i in range(n)]
        tot = mpmath.mpc(0)
        for sg in itertools.permutations(range(n)):
            p = mpmath.mpc(1)
            for i in range(n):
                p *= E[i][sg[i]]
            tot += p
        return tot


@pytest.mark.parametrize("delta", [1e-2, 1e-9])
def test_mp_matches_exact_rational_ground_truth_to_full_precision(delta):
    # The reference must be high-precision even when the caller's global mpmath
    # precision is the default (~15 digits): inputs carry sub-1e-12 structure
    # (0.5 + delta) that a truncating converter would destroy.
    A = make_cancellation_matrix(6, delta=delta, seed=3)
    with mpmath.workdps(70):
        ref = permanent_mp(A, dps=70)
        truth = _exact_perm_via_rationals(A, 70)
        assert abs(ref - truth) <= mpmath.mpf(10) ** -55 * abs(truth)


@pytest.mark.parametrize("n", range(1, 7))
def test_mp_all_ones_is_exactly_n_factorial(n):
    val = permanent_mp(np.ones((n, n)), dps=50)
    assert val == mpmath.mpf(math.factorial(n))  # exact, integer-valued


@pytest.mark.parametrize("n", range(2, 7))
@pytest.mark.parametrize("seed", range(5))
def test_mp_agrees_with_naive_definition(n, seed):
    A = complex_matrix(n, seed=271 * n + seed)
    mp_val = complex(permanent_mp(A, dps=60))
    naive = permanent_naive(A)
    assert abs(mp_val - naive) < 1e-12 * (abs(naive) + 1.0)


def test_mp_integer_matrix_is_exact():
    # Integer inputs -> integer permanent, recovered exactly at high precision.
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert permanent_mp(A, dps=50) == mpmath.mpf(10)
