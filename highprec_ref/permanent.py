"""Permanent at arbitrary precision via mpmath (independent ground truth).

Ryser's inclusion-exclusion evaluated in ``mpmath`` at a configurable working
precision. Kept structurally simple and independent of the FP64 ``cpu_ref``
implementation (plain Python lists + mpmath scalars, no numpy arithmetic) so
that agreement between the two is genuine evidence, not a shared-bug artifact.

The input matrix is converted to mpmath *exactly* (each ``float`` maps to the
binary64 value it already holds), so the high-precision result is the permanent
of the very same matrix the FP64 path saw — which is what makes a relative-error
comparison against this reference meaningful (docs/DESIGN.md §6).
"""

from __future__ import annotations

from typing import Any

import mpmath
import numpy as np

__all__ = ["permanent_mp"]


def _to_mp_matrix(A: Any) -> tuple[list[list[Any]], int, bool]:
    """Return (rows of mpmath scalars, n, is_complex), exact from binary64."""
    M = np.ascontiguousarray(A)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"permanent requires a square 2-D matrix, got shape {M.shape}")
    n = M.shape[0]
    is_complex = np.iscomplexobj(M)
    Mc = M.astype(np.complex128)
    rows: list[list[Any]] = []
    # Capture each binary64 entry EXACTLY. mpmath.mpf(float) rounds to the
    # *current* working precision, so this must run at >= 53 bits regardless of
    # the caller's global dps -- otherwise the "high-precision" reference is
    # silently truncated to ~15 digits on its own inputs (a real bug this guards).
    with mpmath.workprec(64):
        for i in range(n):
            row = []
            for j in range(n):
                z = Mc[i, j]
                if is_complex:
                    row.append(mpmath.mpc(float(z.real), float(z.imag)))
                else:
                    row.append(mpmath.mpf(float(z.real)))
            rows.append(row)
    return rows, n, is_complex


def permanent_mp(A: Any, dps: int = 60) -> Any:
    """High-precision permanent of ``A`` at ``dps`` decimal digits (Ryser).

    Returns an ``mpmath.mpf`` for real input or ``mpmath.mpc`` for complex input.
    Use ``dps=50..80`` for ground truth at small ``n``; cost is O(n * 2^n) mpmath
    operations, so keep ``n`` small (<= ~14).
    """
    rows, n, is_complex = _to_mp_matrix(A)
    with mpmath.workdps(dps):
        zero = mpmath.mpc(0) if is_complex else mpmath.mpf(0)
        if n == 0:
            return mpmath.mpf(1)

        rowsum = [zero for _ in range(n)]  # rowsum[i] = sum_{j in S} A[i,j]
        total = zero
        sign = 1
        prev_gray = 0
        for i in range(1, 1 << n):
            gray = i ^ (i >> 1)
            j = (gray ^ prev_gray).bit_length() - 1
            if (gray >> j) & 1:
                for r in range(n):
                    rowsum[r] = rowsum[r] + rows[r][j]
            else:
                for r in range(n):
                    rowsum[r] = rowsum[r] - rows[r][j]
            sign = -sign
            prod = rowsum[0]
            for r in range(1, n):
                prod = prod * rowsum[r]
            total = total + (prod if sign > 0 else -prod)
            prev_gray = gray

        if n & 1:
            total = -total
        return total
