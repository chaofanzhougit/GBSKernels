"""Conditioning diagnostics for the permanent algorithms.

The permanent's *definition* is an all-plus sum, but the fast *algorithms*
(Ryser, Glynn) are alternating signed sums, so they suffer catastrophic
cancellation when the result is small relative to the magnitude of the terms
(docs/DESIGN.md §6). The relevant condition number for our Glynn-based FP64 path is

    kappa(A) = (sum over Glynn terms of |term|) / |perm(A)|

A standard backward-stable summation loses about ``log10(kappa)`` decimal digits,
so ``rel_err_fp64 ~ kappa * eps``. This module makes ``kappa`` measurable, which
is what turns "FP64 is sometimes wrong" into a *characterized boundary*.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from .hafnian import _exp_newton_coeff, _power_traces, exp_coeff_from_kg
from .permanent import _as_square_complex, permanent_glynn
from .torontonian import _as_even_complex

__all__ = ["glynn_abs_term_sum", "hafnian_abs_term_sum", "loop_hafnian_abs_term_sum",
           "torontonian_abs_term_sum", "cancellation_ratio", "summation_condition_number"]


def glynn_abs_term_sum(A: Any) -> float:
    """Sum of the absolute values of the Glynn/BB-FG terms, ``sum |term_i|``.

    Same Gray-code walk as :func:`cpu_ref.permanent.permanent_glynn`, but it
    accumulates ``prod_r |rowsum_r|`` instead of the signed product. This is the
    numerator of the summation condition number for the Glynn FP64 path.
    """
    M = _as_square_complex(A)
    n = M.shape[0]
    if n == 0:
        return 1.0

    rowsum = M.sum(axis=1).astype(np.complex128)
    acc = float(np.prod(np.abs(rowsum)))
    prev_gray = 0
    for i in range(1, 1 << (n - 1)):
        gray = i ^ (i >> 1)
        k = (gray ^ prev_gray).bit_length() - 1
        col = k + 1
        if (gray >> k) & 1:
            rowsum -= 2.0 * M[:, col]
        else:
            rowsum += 2.0 * M[:, col]
        acc += float(np.prod(np.abs(rowsum)))
        prev_gray = gray
    return acc / (1 << (n - 1))


def cancellation_ratio(A: Any, perm_value: complex | None = None) -> float:
    """``kappa(A)`` -- Glynn summation condition number (see module docstring).

    ``perm_value`` may be supplied (e.g. a high-precision permanent) to avoid
    recomputing it and to make ``kappa`` independent of FP64 error in the
    denominator; otherwise the FP64 Glynn permanent is used.
    """
    p = permanent_glynn(A) if perm_value is None else perm_value
    denom = abs(complex(p))
    if denom == 0.0:
        return float("inf")
    return glynn_abs_term_sum(A) / denom


def hafnian_abs_term_sum(A: Any) -> float:
    """``sum_S |[lambda^n] exp(...)|`` -- the power-trace hafnian's term magnitudes
    (mirrors :func:`cpu_ref.hafnian.hafnian_powertrace`, accumulating |coeff|)."""
    M = _as_square_complex(A).copy()
    N = M.shape[0]
    if N == 0:
        return 1.0
    if N % 2 == 1:
        return 0.0
    np.fill_diagonal(M, 0.0)
    n = N // 2
    acc = 0.0
    for m in range(n + 1):
        for S in itertools.combinations(range(n), m):
            if m == 0:
                BX = np.empty((0, 0), dtype=np.complex128)
            else:
                idx = [p for i in S for p in (2 * i, 2 * i + 1)]
                B = M[np.ix_(idx, idx)]
                swap = [q for t in range(m) for q in (2 * t + 1, 2 * t)]
                BX = B[:, swap]
            acc += abs(_exp_newton_coeff(BX, n))
    return acc


def loop_hafnian_abs_term_sum(A: Any) -> float:
    """``sum_S |coeff_S|`` for the power-trace loop hafnian (even N; mirrors
    :func:`cpu_ref.loop_hafnian.loop_hafnian_powertrace`)."""
    M = _as_square_complex(A)
    N = M.shape[0]
    if N == 0:
        return 1.0
    if N % 2 == 1:
        raise ValueError("loop_hafnian_abs_term_sum requires even N")
    n = N // 2
    acc = 0.0
    for m in range(1, n + 1):
        for S in itertools.combinations(range(n), m):
            idx = [p for i in S for p in (2 * i, 2 * i + 1)]
            B = M[np.ix_(idx, idx)]
            size = 2 * m
            Xs = np.zeros((size, size), dtype=np.complex128)
            for t in range(m):
                Xs[2 * t, 2 * t + 1] = 1.0
                Xs[2 * t + 1, 2 * t] = 1.0
            C = B @ Xs
            d = np.diag(B).copy()
            p = _power_traces(C, n)
            kg = np.zeros(n + 1, dtype=np.complex128)
            Ckm1 = np.eye(size, dtype=np.complex128)
            for k in range(1, n + 1):
                v_k = 0.5 * (d @ Xs @ Ckm1 @ d)
                kg[k] = p[k] / 2.0 + k * v_k
                if k < n:
                    Ckm1 = Ckm1 @ C
            acc += abs(exp_coeff_from_kg(kg, n))
    return acc


def torontonian_abs_term_sum(O: Any) -> float:
    """``sum_S |1/sqrt(det(I - O_S))|`` (mirrors :func:`cpu_ref.torontonian`)."""
    M, n = _as_even_complex(O)
    if n == 0:
        return 1.0
    acc = 0.0
    for m in range(n + 1):
        for S in itertools.combinations(range(n), m):
            if m == 0:
                det = 1.0 + 0j
            else:
                idx = [i for i in S] + [i + n for i in S]
                sub = M[np.ix_(idx, idx)]
                det = np.linalg.det(np.eye(2 * m) - sub)
            acc += abs(1.0 / np.sqrt(complex(det)))
    return acc


_ABS_TERM_SUM = {
    "perm": glynn_abs_term_sum,
    "haf": hafnian_abs_term_sum,
    "lhaf": loop_hafnian_abs_term_sum,
    "tor": torontonian_abs_term_sum,
}


def summation_condition_number(func: str, A: Any, value: complex | None = None) -> float:
    """``kappa = sum|terms| / |f(A)|`` for ``func in {perm, haf, lhaf, tor}`` -- the
    cancellation indicator the ``precision="auto"`` tier selector uses
    (``rel_err_fp64 ~ kappa * eps``). ``value`` (the FP64 result) is reused as the
    denominator if given, else recomputed."""
    if func not in _ABS_TERM_SUM:
        raise ValueError(f"unknown function {func!r}; expected one of {sorted(_ABS_TERM_SUM)}")
    if value is None:
        import cpu_ref
        value = getattr(cpu_ref, func)(A)
    denom = abs(complex(value))
    if denom == 0.0:
        return float("inf")
    return _ABS_TERM_SUM[func](A) / denom
