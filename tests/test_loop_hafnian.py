"""Loop hafnian -- five-layer verification.

The loop hafnian allows points matched to themselves (loops, weighted by the
diagonal); it is what Gaussian states with nonzero displacement need (anchor
sec.2.1) and reduces to the ordinary hafnian when the diagonal is zero.
"""

from __future__ import annotations

from fractions import Fraction

import mpmath
import numpy as np
import pytest

from conftest import symmetric_matrix
from cpu_ref import (
    hafnian_powertrace,
    loop_hafnian_naive,
    loop_hafnian_powertrace,
)
from highprec_ref import loop_hafnian_mp

# --- Layer 1: independent combinatorial ground truth ------------------------


@pytest.mark.layer1
def test_2x2_closed_form():
    # lhaf([[a,b],[b,c]]) = b + a*c (pair, or two loops)
    a, b, c = 2.0, 3.0, 5.0
    A = np.array([[a, b], [b, c]])
    assert loop_hafnian_naive(A) == pytest.approx(b + a * c)
    assert loop_hafnian_powertrace(A) == pytest.approx(b + a * c)


@pytest.mark.layer1
def test_1x1_is_the_loop():
    assert loop_hafnian_naive(np.array([[7.0]])) == pytest.approx(7.0)


@pytest.mark.layer1
def test_diagonal_matrix_is_product_of_diagonal():
    # No off-diagonal edges -> every point must loop -> product of the diagonal.
    d = np.array([2.0, 3.0, 5.0, 7.0])
    A = np.diag(d)
    assert loop_hafnian_naive(A) == pytest.approx(float(np.prod(d)))
    assert loop_hafnian_powertrace(A) == pytest.approx(float(np.prod(d)))


@pytest.mark.layer1
def test_all_ones_is_telephone_number():
    # lhaf(J_N) counts involutions of N elements (pairs + fixed points) =
    # the telephone numbers T(N): 1, 1, 2, 4, 10, 26, 76, 232, ...
    telephone = [1, 1, 2, 4, 10, 26, 76, 232]
    for N in range(1, 8):
        val = loop_hafnian_naive(np.ones((N, N)))
        assert val == pytest.approx(float(telephone[N]))


@pytest.mark.layer1
@pytest.mark.parametrize("N", [2, 4, 6])
@pytest.mark.parametrize("seed", range(5))
def test_naive_and_powertrace_agree(N, seed):
    A = symmetric_matrix(N, seed=200 * N + seed)
    assert loop_hafnian_powertrace(A) == pytest.approx(loop_hafnian_naive(A), rel=1e-9)


# --- Layer 2: The Walrus differential oracle (loop=True) --------------------

thewalrus = pytest.importorskip("thewalrus", reason="Layer 2 needs The Walrus")


@pytest.mark.layer2
@pytest.mark.parametrize("N", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("seed", range(4))
def test_matches_thewalrus_loop(N, seed):
    A = symmetric_matrix(N, seed=29 * N + seed)
    expected = complex(thewalrus.hafnian(A, loop=True))
    assert loop_hafnian_naive(A) == pytest.approx(expected, rel=1e-7, abs=1e-10)
    if N % 2 == 0:
        assert loop_hafnian_powertrace(A) == pytest.approx(expected, rel=1e-7, abs=1e-10)


# --- Layer 3: invariants ----------------------------------------------------


@pytest.mark.layer3
@pytest.mark.parametrize("N", [2, 4, 6])
@pytest.mark.parametrize("seed", range(5))
def test_reduces_to_ordinary_hafnian_when_diagonal_zero(N, seed):
    A = symmetric_matrix(N, seed=seed, zero_diag=True)
    assert loop_hafnian_powertrace(A) == pytest.approx(hafnian_powertrace(A), rel=1e-9)


@pytest.mark.layer3
@pytest.mark.parametrize("N", [2, 3, 4, 5])
@pytest.mark.parametrize("seed", range(5))
def test_permutation_invariance(N, seed):
    A = symmetric_matrix(N, seed=seed)
    g = np.random.default_rng(seed ^ 0x1234)
    p = g.permutation(N)
    assert loop_hafnian_naive(A[np.ix_(p, p)]) == pytest.approx(
        loop_hafnian_naive(A), rel=1e-8
    )


# --- reference validation ---------------------------------------------------


@pytest.mark.layer1
@pytest.mark.parametrize("N", [3, 4])
def test_mp_matches_exact_rational_ground_truth(N):
    A = symmetric_matrix(N, seed=11)
    with mpmath.workdps(70):
        ref = loop_hafnian_mp(A, dps=70)

        def to_mp(x):
            f = Fraction(float(x))
            return mpmath.mpf(f.numerator) / mpmath.mpf(f.denominator)

        E = [[mpmath.mpc(to_mp(A[i, j].real), to_mp(A[i, j].imag)) for j in range(N)]
             for i in range(N)]

        def rec(rem):
            if not rem:
                return mpmath.mpc(1)
            i, rest = rem[0], rem[1:]
            tot = E[i][i] * rec(rest)
            for k in range(len(rest)):
                tot += E[i][rest[k]] * rec(rest[:k] + rest[k + 1 :])
            return tot

        truth = rec(tuple(range(N)))
        assert abs(ref - truth) <= mpmath.mpf(10) ** -55 * abs(truth)
