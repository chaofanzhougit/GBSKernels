"""Torontonian at arbitrary precision via mpmath (independent ground truth).

The subset-sum-over-determinants formula evaluated with mpmath determinants and
square roots at a configurable working precision. Reuses the precision-guarded
converter so binary64 inputs are captured exactly
(see [[mpmath-input-conversion-precision]]).
"""

from __future__ import annotations

import itertools
from typing import Any

import mpmath

from .permanent import _to_mp_matrix

__all__ = ["torontonian_mp"]


def torontonian_mp(O: Any, dps: int = 60) -> Any:
    """High-precision torontonian of the ``2n x 2n`` matrix ``O`` at ``dps`` digits."""
    rows, N, is_complex = _to_mp_matrix(O)
    if N % 2 != 0:
        raise ValueError(f"torontonian requires even dimension, got {N}")
    n = N // 2

    with mpmath.workdps(dps):
        if n == 0:
            return mpmath.mpf(1)
        total = mpmath.mpc(0) if is_complex else mpmath.mpf(0)
        for m in range(n + 1):
            for S in itertools.combinations(range(n), m):
                if m == 0:
                    det = mpmath.mpf(1)
                else:
                    idx = [i for i in S] + [i + n for i in S]
                    size = 2 * m
                    # I - O_S as an mpmath matrix
                    sub = mpmath.matrix(size, size)
                    for a in range(size):
                        for b in range(size):
                            sub[a, b] = (1 if a == b else 0) - rows[idx[a]][idx[b]]
                    det = mpmath.det(sub)
                total += ((-1) ** (n - m)) / mpmath.sqrt(det)
        return total
