"""Hafnian at arbitrary precision via mpmath (independent ground truth).

The literal sum over perfect matchings, evaluated in mpmath at a configurable
working precision. Independent of the FP64 power-trace path (different algorithm
entirely), so agreement is real evidence. Reuses the precision-guarded converter
from :mod:`highprec_ref.permanent` so binary64 inputs are captured exactly
regardless of the caller's global mpmath precision.
"""

from __future__ import annotations

from typing import Any

import mpmath

from .permanent import _to_mp_matrix

__all__ = ["hafnian_mp"]


def hafnian_mp(A: Any, dps: int = 60) -> Any:
    """High-precision ordinary hafnian of symmetric ``A`` at ``dps`` digits.

    Recursive perfect-matching sum (ignores the diagonal). O((2n-1)!!); keep
    ``2n`` small (<= ~12).
    """
    rows, N, is_complex = _to_mp_matrix(A)
    with mpmath.workdps(dps):
        one = mpmath.mpc(1) if is_complex else mpmath.mpf(1)
        zero = mpmath.mpc(0) if is_complex else mpmath.mpf(0)
        if N == 0:
            return one
        if N % 2 == 1:
            return zero

        def rec(rem: tuple[int, ...]):
            if not rem:
                return one
            i, rest = rem[0], rem[1:]
            total = zero
            for k in range(len(rest)):
                j = rest[k]
                total = total + rows[i][j] * rec(rest[:k] + rest[k + 1 :])
            return total

        return rec(tuple(range(N)))
