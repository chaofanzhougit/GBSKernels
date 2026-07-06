"""Loop hafnian at arbitrary precision via mpmath (independent ground truth).

The literal recursive sum over loop matchings (pairs + singleton loops),
evaluated in mpmath. Works for any ``N``. Reuses the precision-guarded converter
so binary64 inputs are captured exactly (see [[mpmath-input-conversion-precision]]).
"""

from __future__ import annotations

from typing import Any

import mpmath

from .permanent import _to_mp_matrix

__all__ = ["loop_hafnian_mp"]


def loop_hafnian_mp(A: Any, dps: int = 60) -> Any:
    """High-precision loop hafnian of symmetric ``A`` at ``dps`` digits."""
    rows, N, is_complex = _to_mp_matrix(A)
    with mpmath.workdps(dps):
        one = mpmath.mpc(1) if is_complex else mpmath.mpf(1)
        zero = mpmath.mpc(0) if is_complex else mpmath.mpf(0)
        if N == 0:
            return one

        def rec(rem: tuple[int, ...]):
            if not rem:
                return one
            i, rest = rem[0], rem[1:]
            total = rows[i][i] * rec(rest)  # i as a loop
            for k in range(len(rest)):
                j = rest[k]
                total = total + rows[i][j] * rec(rest[:k] + rest[k + 1 :])
            return total

        return rec(tuple(range(N)))
