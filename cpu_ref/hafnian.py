"""Hafnian of a symmetric matrix — CPU reference implementations.

For a symmetric ``2n x 2n`` matrix ``A``, ``haf(A) = sum_{M in PMP(2n)}
prod_{(i,j) in M} A[i,j]`` summed over perfect matchings of ``2n`` points. It
counts (weighted) perfect matchings of a general graph and is #P-hard. The
ordinary hafnian ignores the diagonal (no vertex matches itself); the loop
hafnian (Phase 3) restores it.

Two algorithms:

* :func:`hafnian_naive` -- the literal sum over perfect matchings, by recursion
  (match the first free vertex to each other, recurse). O((2n-1)!!). Ground
  truth for small sizes.
* :func:`hafnian_powertrace` -- the Bjorklund-Cygan-Pilipczuk power-trace
  formula used by The Walrus, O(n^3 2^n): a signed sum over the ``2^n`` subsets
  of the ``n`` index-pairs, each contributing the ``lambda^n`` coefficient of
  ``exp(sum_k tr((B_S X)^k)/(2k) lambda^k)`` recovered from power traces via the
  exp/Newton recurrence. This is the GPU-target algorithm and shares the
  signed-subset-sum skeleton with the permanent (docs/DESIGN.md §5).
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from .permanent import _as_square_complex

__all__ = [
    "haf",
    "hafnian_naive",
    "hafnian_powertrace",
    "exp_coeff_from_kg",
    "_power_traces",
]


def hafnian_naive(A: Any) -> complex:
    """O((2n-1)!!) hafnian: literal sum over perfect matchings. Small sizes only."""
    M = _as_square_complex(A)
    N = M.shape[0]
    if N == 0:
        return complex(1.0)
    if N % 2 == 1:
        return complex(0.0)

    def rec(rem: tuple[int, ...]) -> complex:
        if not rem:
            return 1.0 + 0j
        i, rest = rem[0], rem[1:]
        total = 0j
        for k in range(len(rest)):
            j = rest[k]
            total += M[i, j] * rec(rest[:k] + rest[k + 1 :])
        return total

    return complex(rec(tuple(range(N))))


def exp_coeff_from_kg(kg: np.ndarray, n: int) -> complex:
    """``[lambda^n] exp(sum_{k>=1} g_k lambda^k)`` given ``kg[k] = k * g_k``.

    Via the standard exp-of-generating-function (Newton) recurrence: with
    ``E(lambda) = exp(sum_k g_k lambda^k)``, ``E' = (sum_k k g_k lambda^{k-1}) E``
    gives ``e_j = (1/j) sum_{k=1}^j (k g_k) e_{j-k}``, ``e_0 = 1``. Shared by the
    ordinary and loop hafnians; they differ only in how ``kg`` is built.
    """
    e = np.zeros(n + 1, dtype=np.complex128)
    e[0] = 1.0
    for j in range(1, n + 1):
        acc = 0j
        for k in range(1, j + 1):
            acc += kg[k] * e[j - k]
        e[j] = acc / j
    return complex(e[n])


def _power_traces(C: np.ndarray, n: int) -> np.ndarray:
    """``p[k] = tr(C^k)`` for ``k = 1..n`` (``p[0]`` unused)."""
    p = np.zeros(n + 1, dtype=np.complex128)
    if C.shape[0] == 0:
        return p
    P = C.copy()
    for k in range(1, n + 1):
        p[k] = np.trace(P)
        if k < n:
            P = P @ C
    return p


def _exp_newton_coeff(BX: np.ndarray, n: int) -> complex:
    """``[lambda^n] exp(sum_k tr(BX^k)/(2k) lambda^k)`` (ordinary-hafnian kernel)."""
    if BX.shape[0] == 0:
        return 1.0 + 0j if n == 0 else 0j
    p = _power_traces(BX, n)
    return exp_coeff_from_kg(p / 2.0, n)  # kg[k] = k * (p_k/(2k)) = p_k/2


def hafnian_powertrace(A: Any) -> complex:
    """Hafnian via the Bjorklund-Cygan-Pilipczuk power-trace formula (ordinary).

    ``haf(A) = sum_{S subset [n]} (-1)^(n-|S|) [lambda^n] exp(sum_k tr((B_S X)^k)/(2k) lambda^k)``
    where ``B_S`` is the principal submatrix on the index-pairs in ``S`` and ``X``
    swaps the two indices within each pair. The diagonal is zeroed (ordinary
    hafnian).
    """
    M = _as_square_complex(A).copy()
    N = M.shape[0]
    if N == 0:
        return complex(1.0)
    if N % 2 == 1:
        return complex(0.0)
    np.fill_diagonal(M, 0.0)  # ordinary hafnian ignores the diagonal
    n = N // 2

    total = 0j
    for m in range(n + 1):
        for S in itertools.combinations(range(n), m):
            if m == 0:
                BX = np.empty((0, 0), dtype=np.complex128)
            else:
                idx = [p for i in S for p in (2 * i, 2 * i + 1)]
                B = M[np.ix_(idx, idx)]
                swap = [q for t in range(m) for q in (2 * t + 1, 2 * t)]
                BX = B[:, swap]  # B @ X : swap columns within each pair
            total += ((-1) ** (n - m)) * _exp_newton_coeff(BX, n)
    return complex(total)


def haf(A: Any) -> complex:
    """Hafnian of symmetric ``A`` (FP64 tier). Default CPU reference = power-trace."""
    return hafnian_powertrace(A)
