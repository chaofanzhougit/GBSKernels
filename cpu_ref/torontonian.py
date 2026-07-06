"""Torontonian of an even-dimensional matrix — CPU reference implementation.

The torontonian arises for **threshold** (click / no-click) detectors, where one
sums hafnian-like contributions over all photon numbers; it collapses that
infinite sum into a single function (docs/DESIGN.md §2.1):

    tor(O) = sum_{S subset [n]} (-1)^(n-|S|) / sqrt(det(I - O_S))

for a ``2n x 2n`` matrix ``O`` (``n`` modes). ``O_S`` is the principal submatrix
on the indices of the modes in ``S``, in **xxpp** ordering: mode ``i`` owns
indices ``i`` and ``i + n`` (this ordering and the ``(-1)^(n-|S|)`` sign were
pinned numerically against The Walrus -- see tests). Per-subset cost is a
determinant of a principal submatrix.

This is the same signed-subset-sum skeleton as the other three functions (anchor
sec.5); only the per-subset kernel differs (a determinant instead of a
power-trace / row-sum product). The CPU reference enumerates subsets directly for
clarity; the GPU kernel can walk the Gray code with incremental rank-1 Cholesky
updates of ``I - O_S`` (docs/DESIGN.md §5).

**Domain note.** The torontonian's physical inputs are **real** ``O`` matrices
built from real Gaussian covariance matrices, and on real ``O`` this matches both
of The Walrus's implementations and the mpmath reference exactly. For a generic
*complex* ``O`` the ``sqrt`` of a complex determinant is branch-dependent and
even The Walrus's own recursive and non-recursive routines disagree, so the
value is convention-dependent there; we validate (and intend) the real domain.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

__all__ = ["tor", "torontonian"]


def _as_even_complex(O: Any) -> tuple[np.ndarray, int]:
    M = np.ascontiguousarray(O, dtype=np.complex128)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"torontonian requires a square matrix, got shape {M.shape}")
    if M.shape[0] % 2 != 0:
        raise ValueError(f"torontonian requires even dimension, got {M.shape[0]}")
    return M, M.shape[0] // 2


def torontonian(O: Any) -> complex:
    """Torontonian of the ``2n x 2n`` matrix ``O`` (xxpp ordering)."""
    M, n = _as_even_complex(O)
    if n == 0:
        return complex(1.0)

    total = 0j
    for m in range(n + 1):
        for S in itertools.combinations(range(n), m):
            if m == 0:
                det = 1.0 + 0j # empty submatrix: det(I) = 1
            else:
                idx = [i for i in S] + [i + n for i in S] # xxpp: x-indices then p-indices
                sub = M[np.ix_(idx, idx)]
                det = np.linalg.det(np.eye(2 * m) - sub)
            total += ((-1) ** (n - m)) / np.sqrt(complex(det))
    return complex(total)


def tor(O: Any) -> complex:
    """Torontonian of ``O`` (FP64 tier)."""
    return torontonian(O)
