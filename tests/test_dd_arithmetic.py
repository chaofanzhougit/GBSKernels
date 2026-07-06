"""Validate the double-double arithmetic and the DD permanent (docs/DESIGN.md §6).

Two things must hold for the DD tier to be trustworthy:

1. the error-free transforms are *exact* (TwoSum/TwoProd recover the rounding
   error to the bit), and DD ops agree with mpmath at ~31 digits;
2. the DD permanent restores accuracy where the FP64 permanent cancels --
   tracking the mpmath reference to machine precision across the whole
   conditioning range that breaks FP64.

This is the CPU validation the GPU DD kernel (core/dd.cuh, core/*.cu) must later
reproduce in a rented-GPU session.
"""

from __future__ import annotations

import mpmath
import numpy as np
import pytest

from bench._inputs import make_cancellation_matrix, random_complex
from cpu_ref import cancellation_ratio, permanent_glynn
from cpu_ref import dd
from cpu_ref.permanent import permanent_glynn_dd
from highprec_ref import permanent_mp

pytestmark = pytest.mark.layer5


# --- (1) the EFTs are exact -------------------------------------------------

@pytest.mark.parametrize("a,b", [(1.0, 2.0**-60), (0.1, 0.2), (1e8, 1e-8), (3.0, 7.0)])
def test_two_sum_is_exact(a, b):
    s, e = dd.two_sum(a, b)
    # s + e reproduces a + b with no rounding (checked in mpmath)
    with mpmath.workdps(60):
        assert mpmath.mpf(s) + mpmath.mpf(e) == mpmath.mpf(a) + mpmath.mpf(b)


@pytest.mark.parametrize("a,b", [(0.1, 0.3), (1.0 + 2**-53, 1.0 - 2**-53), (7.0, 11.0), (1e-9, 3e8)])
def test_two_prod_is_exact(a, b):
    p, e = dd.two_prod(a, b)
    with mpmath.workdps(60):
        assert mpmath.mpf(p) + mpmath.mpf(e) == mpmath.mpf(a) * mpmath.mpf(b)


def test_dd_mul_matches_mpmath_to_dd_precision():
    a, b = 1.0 + 1e-20, 1.0 - 3e-18  # structure below FP64 resolution
    ah = dd.r_add((1.0, 0.0), (1e-20, 0.0))
    bh = dd.r_add((1.0, 0.0), (-3e-18, 0.0))
    prod = dd.r_mul(ah, bh)
    with mpmath.workdps(50):
        exact = (mpmath.mpf(1) + mpmath.mpf("1e-20")) * (mpmath.mpf(1) - mpmath.mpf("3e-18"))
        got = mpmath.mpf(prod[0]) + mpmath.mpf(prod[1])
        assert abs(got - exact) <= mpmath.mpf(10) ** -29 * abs(exact)


# --- (2) DD permanent restores accuracy under cancellation ------------------

@pytest.mark.parametrize("seed", range(4))
def test_dd_agrees_with_fp64_and_reference_when_well_conditioned(seed):
    A = random_complex(6, seed=seed)
    with mpmath.workdps(60):
        exact = permanent_mp(A, dps=60)
        rel = abs(mpmath.mpc(permanent_glynn_dd(A)) - exact) / abs(exact)
        assert rel < 1e-14  # at least as good as FP64


@pytest.mark.parametrize("delta", [1e-2, 1e-6, 1e-10, 1e-12, 1e-14])
def test_dd_survives_cancellation_that_breaks_fp64(delta):
    A = make_cancellation_matrix(6, delta=delta, seed=1)
    with mpmath.workdps(80):
        exact = permanent_mp(A, dps=80)
        kappa = cancellation_ratio(A, perm_value=complex(exact))
        err_fp64 = float(abs(mpmath.mpc(permanent_glynn(A)) - exact) / abs(exact))
        err_dd = float(abs(mpmath.mpc(permanent_glynn_dd(A)) - exact) / abs(exact))
    # DD stays at ~machine precision regardless of conditioning...
    assert err_dd < 1e-14
    # ...and once FP64 has genuinely degraded, DD is dramatically better.
    if kappa > 1e6:
        assert err_dd < err_fp64 / 1e3


def test_dd_holds_where_fp64_has_no_correct_digits():
    A = make_cancellation_matrix(6, delta=1e-14, seed=1)
    with mpmath.workdps(80):
        exact = permanent_mp(A, dps=80)
        err_fp64 = float(abs(mpmath.mpc(permanent_glynn(A)) - exact) / abs(exact))
        err_dd = float(abs(mpmath.mpc(permanent_glynn_dd(A)) - exact) / abs(exact))
    assert err_fp64 > 1e-4    # FP64 has lost ~all precision
    assert err_dd < 1e-13     # DD is still essentially exact
