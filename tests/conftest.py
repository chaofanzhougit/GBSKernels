"""Shared fixtures and independent helpers for the verification suite."""

from __future__ import annotations

import numpy as np
import pytest


def real_matrix(n: int, seed: int, scale: float = 1.0) -> np.ndarray:
    """Random real matrix, entries ~ U(-scale, scale)."""
    g = np.random.default_rng(seed)
    return g.uniform(-scale, scale, size=(n, n))


def complex_matrix(n: int, seed: int, scale: float = 1.0) -> np.ndarray:
    """Random complex matrix, real/imag parts ~ U(-scale, scale)."""
    g = np.random.default_rng(seed)
    re = g.uniform(-scale, scale, size=(n, n))
    im = g.uniform(-scale, scale, size=(n, n))
    return re + 1j * im


def binary_matrix(n: int, seed: int, p: float = 0.5) -> np.ndarray:
    """Random 0/1 matrix with edge probability ``p`` (a bipartite adjacency)."""
    g = np.random.default_rng(seed)
    return (g.random((n, n)) < p).astype(np.float64)


def count_perfect_matchings_bipartite(adj: np.ndarray) -> int:
    """Count perfect matchings of a bipartite graph by backtracking.

    Independent of every permanent *formula*: it assigns each left vertex (row)
    a distinct unused right vertex (column) with an edge, via recursive
    backtracking. For a 0/1 matrix this equals the permanent, which is exactly
    the Layer-1 ground-truth identity we want to check -- established here by a
    structurally different routine than the sum-over-permutations definition.
    """
    n = adj.shape[0]
    used = [False] * n

    def rec(row: int) -> int:
        if row == n:
            return 1
        total = 0
        for col in range(n):
            if adj[row, col] != 0 and not used[col]:
                used[col] = True
                total += rec(row + 1)
                used[col] = False
        return total

    return rec(0)


def symmetric_matrix(n: int, seed: int, scale: float = 1.0, zero_diag: bool = False) -> np.ndarray:
    """Random complex symmetric n x n matrix (A == A.T, not Hermitian)."""
    g = np.random.default_rng(seed)
    G = scale * (g.uniform(-1, 1, (n, n)) + 1j * g.uniform(-1, 1, (n, n)))
    A = G + G.T
    if zero_diag:
        np.fill_diagonal(A, 0.0)
    return A


def count_perfect_matchings_graph(adj: np.ndarray) -> int:
    """Count perfect matchings of an undirected graph by backtracking.

    Independent of every hafnian *formula*: it pairs the lowest-indexed unmatched
    vertex with each available neighbour and recurses. For a 0/1 symmetric
    adjacency (zero diagonal) this equals the hafnian -- the Layer-1 ground-truth
    identity for the hafnian, via a structurally different routine than the
    sum-over-matchings definition in cpu_ref.
    """
    n = adj.shape[0]
    if n % 2 == 1:
        return 0
    matched = [False] * n

    def rec() -> int:
        i = 0
        while i < n and matched[i]:
            i += 1
        if i == n:
            return 1
        matched[i] = True
        total = 0
        for j in range(i + 1, n):
            if not matched[j] and adj[i, j] != 0:
                matched[j] = True
                total += rec()
                matched[j] = False
        matched[i] = False
        return total

    return rec()


def rel_error(approx: complex, exact: complex) -> float:
    """Relative error |approx - exact| / max(|exact|, tiny)."""
    denom = max(abs(complex(exact)), 1e-300)
    return abs(complex(approx) - complex(exact)) / denom


@pytest.fixture(scope="session")
def has_thewalrus() -> bool:
    try:
        import thewalrus  # noqa: F401

        return True
    except Exception:
        return False
