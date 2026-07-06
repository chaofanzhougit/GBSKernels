"""Layer 1 (hafnian) -- independent combinatorial ground truth (no Walrus).

Count perfect matchings by brute force and check exact closed forms. Bedrock:
depends on no existing library, only on the definition of the hafnian.
"""

from __future__ import annotations

from math import prod

import numpy as np
import pytest

from conftest import count_perfect_matchings_graph, symmetric_matrix
from cpu_ref import hafnian_naive, hafnian_powertrace

pytestmark = pytest.mark.layer1

ALGOS = (hafnian_naive, hafnian_powertrace)


def _double_factorial_odd(N: int) -> int:
    # (N-1)!! for even N = number of perfect matchings of the complete graph K_N
    return prod(range(1, N, 2))


@pytest.mark.parametrize("algo", ALGOS)
@pytest.mark.parametrize("n_pairs", range(1, 5))
def test_all_ones_hafnian_is_double_factorial(algo, n_pairs):
    N = 2 * n_pairs
    assert algo(np.ones((N, N))) == pytest.approx(float(_double_factorial_odd(N)))


@pytest.mark.parametrize("algo", ALGOS)
def test_empty_and_odd_sizes(algo):
    assert algo(np.zeros((0, 0))) == pytest.approx(1.0)  # empty product
    assert algo(np.ones((3, 3))) == pytest.approx(0.0)   # odd size -> 0
    assert algo(np.ones((5, 5))) == pytest.approx(0.0)


@pytest.mark.parametrize("algo", ALGOS)
@pytest.mark.parametrize("n_pairs", range(1, 5))
def test_offdiagonal_2x2_is_a01(algo, n_pairs):
    # haf of a single 2x2 block is just its off-diagonal entry; direct-sum of
    # blocks multiplies, so a block-diagonal of 2x2s gives the product.
    blocks = [np.array([[0.0, float(k + 1)], [float(k + 1), 0.0]]) for k in range(n_pairs)]
    A = np.zeros((2 * n_pairs, 2 * n_pairs))
    for k, B in enumerate(blocks):
        A[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] = B
    assert algo(A) == pytest.approx(float(prod(range(1, n_pairs + 1))))


@pytest.mark.parametrize("N", [2, 4, 6, 8])
@pytest.mark.parametrize("seed", range(6))
def test_binary_hafnian_counts_graph_perfect_matchings(N, seed):
    # Symmetric 0/1 adjacency (zero diagonal): hafnian == #perfect matchings.
    g = np.random.default_rng(seed)
    U = (g.random((N, N)) < 0.5).astype(float)
    A = np.triu(U, 1)
    A = A + A.T  # symmetric, zero diagonal
    expected = count_perfect_matchings_graph(A)  # independent backtracking
    for algo in ALGOS:
        assert algo(A) == pytest.approx(float(expected), abs=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_direct_sum_multiplicativity(seed):
    # haf(A (+) B) = haf(A) haf(B)
    A = symmetric_matrix(4, seed=seed)
    B = symmetric_matrix(6, seed=seed + 31)
    blk = np.zeros((10, 10), dtype=complex)
    blk[:4, :4] = A
    blk[4:, 4:] = B
    assert hafnian_powertrace(blk) == pytest.approx(
        hafnian_powertrace(A) * hafnian_powertrace(B), rel=1e-9
    )


@pytest.mark.parametrize("N", [2, 4, 6, 8])
@pytest.mark.parametrize("seed", range(6))
def test_naive_and_powertrace_agree(N, seed):
    A = symmetric_matrix(N, seed=100 * N + seed)
    assert hafnian_powertrace(A) == pytest.approx(hafnian_naive(A), rel=1e-9, abs=1e-12)
