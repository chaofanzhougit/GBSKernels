"""Double-double (DD) arithmetic on the CPU — mirror of ``core/dd.cuh``.

A DD value is an unevaluated sum of two float64 (``hi + lo``, ~31 decimal
digits) built from error-free transformations (Dekker/Knuth). This module is the
**CPU twin** of the GPU ``core/dd.cuh``: it implements the same EFTs so the DD
*algorithms* (and the FP64-vs-DD accuracy boundary, docs/DESIGN.md §6) can be
validated against the mpmath reference now, without a GPU. The CUDA kernels then
only need to reproduce already-verified numerics.

One deliberate difference, numerically equivalent: ``core/dd.cuh`` forms the
product error with a hardware FMA (``two_prod``); Python 3.12 has no
``math.fma``, so here ``two_prod`` uses Dekker's 27-bit splitting. Both yield the
*exact* rounding error of ``a*b``, so the DD results are identical.

Scalars are plain Python floats (IEEE binary64); a real DD is a ``(hi, lo)``
tuple, a complex DD a ``(re_hi, re_lo, im_hi, im_lo)`` tuple. Functions over
classes for speed in the inner permanent loop.
"""

from __future__ import annotations

from typing import Tuple

Real = Tuple[float, float]      # hi, lo
Cplx = Tuple[float, float, float, float]  # re_hi, re_lo, im_hi, im_lo

__all__ = ["two_sum", "two_prod", "r_add", "r_sub", "r_mul", "c_from", "c_add",
           "c_sub", "c_mul", "c_mul_real", "c_to_complex"]

_SPLITTER = 134217729.0  # 2^27 + 1, for Dekker splitting


# --- error-free transforms --------------------------------------------------

def two_sum(a: float, b: float) -> Real:
    """fl(a+b) and its exact error (Knuth, no magnitude assumption)."""
    s = a + b
    bb = s - a
    err = (a - (s - bb)) + (b - bb)
    return s, err


def _split(a: float) -> Real:
    c = _SPLITTER * a
    hi = c - (c - a)
    return hi, a - hi


def two_prod(a: float, b: float) -> Real:
    """fl(a*b) and its exact error via Dekker splitting (FMA-free)."""
    p = a * b
    ahi, alo = _split(a)
    bhi, blo = _split(b)
    err = ((ahi * bhi - p) + ahi * blo + alo * bhi) + alo * blo
    return p, err


# --- real DD ----------------------------------------------------------------

def _quick_two_sum(a: float, b: float) -> Real:
    s = a + b
    return s, b - (s - a)


def r_add(a: Real, b: Real) -> Real:
    s, e = two_sum(a[0], b[0])
    e += a[1] + b[1]
    return _quick_two_sum(s, e)


def r_sub(a: Real, b: Real) -> Real:
    return r_add(a, (-b[0], -b[1]))


def r_mul(a: Real, b: Real) -> Real:
    p, e = two_prod(a[0], b[0])
    e += a[0] * b[1] + a[1] * b[0]
    return _quick_two_sum(p, e)


# --- complex DD -------------------------------------------------------------

def c_from(z: complex) -> Cplx:
    return (z.real, 0.0, z.imag, 0.0)


def c_add(a: Cplx, b: Cplx) -> Cplx:
    rh, rl = r_add((a[0], a[1]), (b[0], b[1]))
    ih, il = r_add((a[2], a[3]), (b[2], b[3]))
    return (rh, rl, ih, il)


def c_sub(a: Cplx, b: Cplx) -> Cplx:
    rh, rl = r_sub((a[0], a[1]), (b[0], b[1]))
    ih, il = r_sub((a[2], a[3]), (b[2], b[3]))
    return (rh, rl, ih, il)


def c_mul(a: Cplx, b: Cplx) -> Cplx:
    ar, ai = (a[0], a[1]), (a[2], a[3])
    br, bi = (b[0], b[1]), (b[2], b[3])
    rr = r_sub(r_mul(ar, br), r_mul(ai, bi))  # re = ar*br - ai*bi
    ii = r_add(r_mul(ar, bi), r_mul(ai, br))  # im = ar*bi + ai*br
    return (rr[0], rr[1], ii[0], ii[1])


def c_mul_real(a: Cplx, s: float) -> Cplx:
    rh, rl = r_mul((a[0], a[1]), (s, 0.0))
    ih, il = r_mul((a[2], a[3]), (s, 0.0))
    return (rh, rl, ih, il)


def c_to_complex(a: Cplx) -> complex:
    return complex(a[0] + a[1], a[2] + a[3])
