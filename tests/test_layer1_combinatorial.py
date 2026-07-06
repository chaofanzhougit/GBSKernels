"""Layer 1 -- independent combinatorial ground truth (shares no code with The Walrus).

The whole point of the permanent is counting, so we count by brute force at small
sizes and check exact closed forms. These are bedrock: they depend on no existing
library, only on the definition.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import binary_matrix, complex_matrix, count_perfect_matchings_bipartite
from cpu_ref import permanent_glynn, permanent_naive, permanent_ryser

pytestmark = pytest.mark.layer1

ALGOS = (permanent_naive, permanent_ryser, permanent_glynn)


@pytest.mark.parametrize("algo", ALGOS)
@pytest.mark.parametrize("n", range(1, 8))
def test_identity_permanent_is_one(algo, n):
    assert algo(np.eye(n)) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("algo", ALGOS)
@pytest.mark.parametrize("n", range(1, 8))
def test_all_ones_permanent_is_n_factorial(algo, n):
    # perm(J_n) = n!  (every permutation contributes a product of ones)
    assert algo(np.ones((n, n))) == pytest.approx(float(math.factorial(n)), rel=1e-9)


@pytest.mark.parametrize("algo", ALGOS)
@pytest.mark.parametrize("n", range(1, 7))
def test_triangular_permanent_is_diagonal_product(algo, n):
    # An upper-triangular matrix forces sigma = identity, so perm = prod of diag.
    g = np.random.default_rng(100 + n)
    A = np.triu(g.uniform(-1, 1, size=(n, n)))
    assert algo(A) == pytest.approx(float(np.prod(np.diag(A))), rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("n", range(1, 8))
@pytest.mark.parametrize("seed", range(6))
def test_binary_permanent_counts_perfect_matchings(n, seed):
    A = binary_matrix(n, seed=seed)
    expected = count_perfect_matchings_bipartite(A)  # independent backtracking
    for algo in ALGOS:
        assert algo(A) == pytest.approx(float(expected), abs=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_direct_sum_multiplicativity(seed):
    # perm(A (+) B) = perm(A) * perm(B) for the block-diagonal direct sum.
    A = complex_matrix(3, seed=seed)
    B = complex_matrix(4, seed=seed + 50)
    blk = np.block([[A, np.zeros((3, 4))], [np.zeros((4, 3)), B]])
    lhs = permanent_glynn(blk)
    rhs = permanent_glynn(A) * permanent_glynn(B)
    assert lhs == pytest.approx(rhs, rel=1e-9)


@pytest.mark.parametrize("n", range(2, 8))
@pytest.mark.parametrize("seed", range(8))
def test_three_algorithms_agree_complex(n, seed):
    # Three different algorithms (definition, Ryser, Glynn) must agree.
    A = complex_matrix(n, seed=1000 * n + seed)
    p_naive = permanent_naive(A)
    p_ryser = permanent_ryser(A)
    p_glynn = permanent_glynn(A)
    assert p_ryser == pytest.approx(p_naive, rel=1e-9, abs=1e-12)
    assert p_glynn == pytest.approx(p_naive, rel=1e-9, abs=1e-12)


def test_known_small_value():
    # perm([[1,2],[3,4]]) = 1*4 + 2*3 = 10 ; perm([[a,b],[c,d]]) = ad + bc.
    assert permanent_glynn(np.array([[1.0, 2.0], [3.0, 4.0]])) == pytest.approx(10.0)
    # 3x3 hand value: perm of all-distinct via cofactor-style expansion.
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
    # perm = 1*(5*10+6*8) + 2*(4*10+6*7) + 3*(4*8+5*7) = 98 + 164 + 201 = 463
    assert permanent_glynn(A) == pytest.approx(463.0)
