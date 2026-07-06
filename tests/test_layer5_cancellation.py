"""Layer 5 -- cancellation & conditioning: the measured FP64 boundary (docs/DESIGN.md §6).

These tests encode the project's central numerical claim as executable checks:

1. the arbitrary-precision reference stays *exact* under cancellation;
2. FP64 relative error is governed by the Glynn condition number ``kappa``
   (``rel_err ~ kappa * eps``) -- the boundary is *measured*, not assumed;
3. on a deliberately cancellation-heavy matrix FP64 visibly breaks while the
   reference survives -- the boundary is *real*.

The double-double tier that is meant to survive this on the GPU is validated in
a rented-GPU session against this same reference (docs/DESIGN.md §8, Layer 5).
"""

from __future__ import annotations

import math

import mpmath
import numpy as np
import pytest

from bench._inputs import (
    make_cancellation_matrix,
    pm1_matrix,
    random_complex,
    unit_modulus_complex,
)
from cpu_ref import cancellation_ratio, permanent_glynn, permanent_naive
from highprec_ref import permanent_mp

pytestmark = pytest.mark.layer5

EPS = float(np.finfo(np.float64).eps)


# --- (1) the reference is exact under cancellation --------------------------


@pytest.mark.parametrize("n", range(2, 9))
@pytest.mark.parametrize("seed", range(4))
def test_reference_exact_on_pm1_integer_permanent(n, seed):
    # +/-1 matrices have integer permanents; naive is exact in float for n! < 2^53
    # (n <= 9), and the mpmath reference must reproduce that integer exactly.
    A = pm1_matrix(n, seed=seed)
    exact_int = round(permanent_naive(A).real)
    assert permanent_mp(A, dps=50) == mpmath.mpf(exact_int)


@pytest.mark.parametrize("delta", [1e-2, 1e-6, 1e-10])
def test_reference_matches_factorized_value_under_cancellation(delta):
    # The block-diagonal permanent factorizes exactly (direct-sum
    # multiplicativity); the reference must satisfy perm(A) = perm(B)*perm(R) to
    # full precision even as FP64 loses most of its digits. (Note: the block's
    # own permanent is 2*delta' for the *stored* delta', not the nominal delta --
    # the increment 1e-12 rounds against 0.5 -- so we factorize, not hard-code it.)
    n = 6
    A = make_cancellation_matrix(n, delta=delta, seed=3)
    # Do the multiply/subtract IN mpmath (workdps), else they round at the
    # default global precision and the comparison collapses to FP64.
    with mpmath.workdps(70):
        expected = permanent_mp(A[:2, :2], dps=70) * permanent_mp(A[2:, 2:], dps=70)
        got = permanent_mp(A, dps=70)
        assert abs(got - expected) <= mpmath.mpf(10) ** -50 * abs(expected)


# --- (2) FP64 error is governed by the condition number kappa ---------------


def _inputs_for_kappa_characterization():
    cases = []
    for seed in range(6):
        cases.append(random_complex(6, seed=seed))
        cases.append(unit_modulus_complex(7, seed=seed))
    for delta in [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11]:
        cases.append(make_cancellation_matrix(6, delta=delta, seed=seed))
    return cases


@pytest.mark.parametrize("A", _inputs_for_kappa_characterization())
def test_fp64_error_bounded_by_condition_number(A):
    exact = complex(permanent_mp(A, dps=60))
    approx = permanent_glynn(A)
    kappa = cancellation_ratio(A, perm_value=exact)
    rel_err = abs(approx - exact) / max(abs(exact), 1e-300)
    # Backward-stable summation: rel_err <= C * kappa * eps. Generous C.
    assert rel_err <= 1e3 * max(kappa, 1.0) * EPS


# --- (3) the boundary is real: FP64 breaks where the reference holds ---------


def test_fp64_visibly_breaks_under_heavy_cancellation():
    A = make_cancellation_matrix(6, delta=1e-12, seed=1)
    exact = complex(permanent_mp(A, dps=80))
    approx = permanent_glynn(A)
    rel_err = abs(approx - exact) / abs(exact)
    kappa = cancellation_ratio(A, perm_value=exact)
    assert kappa > 1e10  # genuinely ill-conditioned for Glynn
    assert rel_err > 1e-6  # FP64 has lost many digits here...
    # ...while the reference is converged: it agrees with itself across working
    # precisions far beyond FP64 (compared in mpmath), so the boundary is a
    # property of FP64, not of the problem being unsolvable.
    # (kappa ~ 1e12 costs even the reference ~12 of its digits, so it converges
    # to ~50 digits, not infinitely -- still ~45 orders beyond FP64's failure.)
    with mpmath.workdps(90):
        assert abs(permanent_mp(A, dps=90) - permanent_mp(A, dps=55)) <= (
            mpmath.mpf(10) ** -40 * abs(permanent_mp(A, dps=90))
        )


def test_fp64_error_grows_monotonically_as_cancellation_increases():
    errs = []
    for delta in [1e-3, 1e-6, 1e-9, 1e-12]:
        A = make_cancellation_matrix(6, delta=delta, seed=2)
        exact = complex(permanent_mp(A, dps=80))
        errs.append(abs(permanent_glynn(A) - exact) / abs(exact))
    # each tenfold-smaller delta (tenfold-larger kappa) costs ~1000x accuracy
    assert errs[0] < errs[1] < errs[2] < errs[3]
    assert errs[-1] > 1e3 * errs[0]
