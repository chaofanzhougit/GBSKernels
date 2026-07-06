"""Permanent of a square matrix — CPU reference implementations.

The permanent is ``perm(A) = sum_{sigma in S_n} prod_i A[i, sigma(i)]`` — the
determinant with every sign set to ``+1``, which is exactly what removes the
cancellation that makes the determinant easy and makes the permanent #P-hard
(Valiant 1979). For a 0/1 matrix it counts perfect matchings of the associated
bipartite graph (see Layer-1 tests).

Three independent algorithms, in increasing sophistication, so they can
cross-validate one another (a bug shared by all three is far less likely than a
bug in one):

* :func:`permanent_naive`  -- O(n!) sum over permutations. The literal
  definition; ground truth for tiny ``n``.
* :func:`permanent_ryser`  -- Ryser's inclusion-exclusion, O(n * 2^n), with a
  Gray-code walk so each subset is an O(n) rank-1 update of the previous one.
* :func:`permanent_glynn`  -- the Glynn / Balasubramanian-Bax-Franklin-Glynn
  (BB-FG) form, O(n * 2^(n-1)): half the subsets of Ryser via the sign symmetry.
  This is The Walrus's default ``perm`` algorithm and the one the GPU kernel
  mirrors (docs/DESIGN.md §5: "implement first").

All three accept any real/complex array-like and compute in ``complex128``
(the FP64 tier). The Gray-code walks here are the CPU twin of the GPU
``subset_engine`` (docs/DESIGN.md §5): one shared skeleton, enumerate 2^k subsets,
each an O(1)/O(n) delta-update of the last.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from . import dd

__all__ = [
    "perm",
    "permanent_naive",
    "permanent_ryser",
    "permanent_glynn",
    "permanent_glynn_dd",
]


def _as_square_complex(A: Any) -> np.ndarray:
    """Validate and coerce input to a square ``complex128`` matrix."""
    M = np.ascontiguousarray(A, dtype=np.complex128)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"permanent requires a square 2-D matrix, got shape {M.shape}")
    return M


def permanent_naive(A: Any) -> complex:
    """O(n!) permanent: literal sum over all permutations. Tiny ``n`` only.

    This is the most obviously-correct implementation and serves as independent
    ground truth in the Layer-1 suite.
    """
    M = _as_square_complex(A)
    n = M.shape[0]
    if n == 0:
        return complex(1.0)  # empty product
    rows = range(n)
    total = 0j
    for sigma in itertools.permutations(range(n)):
        prod = 1.0 + 0j
        for i in rows:
            prod *= M[i, sigma[i]]
        total += prod
    return complex(total)


def permanent_ryser(A: Any) -> complex:
    """Ryser's formula via a Gray-code subset walk, O(n * 2^n).

    ``perm(A) = (-1)^n * sum_{S subset [n]} (-1)^|S| * prod_i (sum_{j in S} A[i,j])``.

    The Gray code visits the 2^n column subsets so that consecutive subsets
    differ by exactly one column, making the per-step update an O(n) add/subtract
    of one column to the running row-sum vector.
    """
    M = _as_square_complex(A)
    n = M.shape[0]
    if n == 0:
        return complex(1.0)

    rowsum = np.zeros(n, dtype=np.complex128)  # rowsum[i] = sum_{j in S} A[i,j]
    total = 0j
    sign = 1  # (-1)^|S|; S starts empty -> +1, but its product term is 0 for n>=1
    prev_gray = 0
    for i in range(1, 1 << n):
        gray = i ^ (i >> 1)
        j = (gray ^ prev_gray).bit_length() - 1  # the single column that flipped
        if (gray >> j) & 1:  # column j entered S
            rowsum += M[:, j]
        else:  # column j left S
            rowsum -= M[:, j]
        sign = -sign
        total += sign * np.prod(rowsum)
        prev_gray = gray

    if n & 1:  # multiply by (-1)^n
        total = -total
    return complex(total)


def permanent_glynn(A: Any) -> complex:
    """Glynn / BB-FG formula via a Gray-code sign walk, O(n * 2^(n-1)).

    ``perm(A) = 2^-(n-1) * sum_{delta} (prod_k delta_k) * prod_i (sum_j delta_j A[i,j])``
    over ``delta in {+1,-1}^n`` with ``delta_0`` fixed to ``+1`` (the symmetry
    that halves Ryser). The Gray code flips one ``delta_k`` per step, a
    ``+-2 * A[:,k]`` rank-1 update of the running row-sum vector; flipping any
    sign negates the running product-sign.
    """
    M = _as_square_complex(A)
    n = M.shape[0]
    if n == 0:
        return complex(1.0)

    rowsum = M.sum(axis=1).astype(np.complex128)  # delta = all +1
    total = np.prod(rowsum)
    sign = 1
    prev_gray = 0
    # vary delta_1..delta_{n-1} (n-1 bits); bit set <-> delta = -1
    for i in range(1, 1 << (n - 1)):
        gray = i ^ (i >> 1)
        k = (gray ^ prev_gray).bit_length() - 1
        col = k + 1  # column 0 is the fixed +1
        if (gray >> k) & 1:  # delta_col: +1 -> -1
            rowsum -= 2.0 * M[:, col]
        else:  # delta_col: -1 -> +1
            rowsum += 2.0 * M[:, col]
        sign = -sign
        total += sign * np.prod(rowsum)
        prev_gray = gray

    return complex(total / (1 << (n - 1)))


def permanent_glynn_dd(A: Any) -> complex:
    """Glynn/BB-FG permanent accumulated in double-double precision.

    Bit-for-bit the same Gray-code walk as :func:`permanent_glynn`, but every
    add/sub/mul runs in DD (``cpu_ref.dd``, the mirror of ``core/dd.cuh``). This
    is the CPU stand-in for the GPU DD tier: it restores accuracy where the FP64
    alternating sum cancels (docs/DESIGN.md §6), and validates the DD algorithm
    against the mpmath reference before any GPU port. ~1-2 orders slower than
    FP64; small ``n``.
    """
    M = _as_square_complex(A)
    n = M.shape[0]
    if n == 0:
        return complex(1.0)

    # entries as complex DD
    E = [[dd.c_from(complex(M[r, c])) for c in range(n)] for r in range(n)]

    rowsum = []  # delta = all +1
    for r in range(n):
        acc = dd.c_from(0j)
        for c in range(n):
            acc = dd.c_add(acc, E[r][c])
        rowsum.append(acc)

    def product() -> dd.Cplx:
        p = rowsum[0]
        for r in range(1, n):
            p = dd.c_mul(p, rowsum[r])
        return p

    total = product()
    sign = 1
    prev_gray = 0
    for i in range(1, 1 << (n - 1)):
        gray = i ^ (i >> 1)
        k = (gray ^ prev_gray).bit_length() - 1
        col = k + 1
        step = -2.0 if ((gray >> k) & 1) else 2.0
        for r in range(n):
            rowsum[r] = dd.c_add(rowsum[r], dd.c_mul_real(E[r][col], step))
        sign = -sign
        p = product()
        total = dd.c_add(total, p) if sign > 0 else dd.c_sub(total, p)
        prev_gray = gray

    total = dd.c_mul_real(total, 1.0 / (1 << (n - 1)))
    return dd.c_to_complex(total)


def perm(A: Any) -> complex:
    """Permanent of ``A`` (FP64 tier). Default CPU reference = Glynn/BB-FG."""
    return permanent_glynn(A)
