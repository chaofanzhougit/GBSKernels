"""Loop hafnian with repeated rows/columns — the finite-difference sieve (R4).

The GBS sampling workload evaluates (loop) hafnians of matrices built by
*repeating* rows/columns of a small base matrix: a photon pattern
``n = (n_1..n_M)`` selects mode ``i``'s row/column ``n_i`` times, so the
expanded matrix has dimension ``N = sum(n_i)`` and the direct power-trace cost
``O(N^3 2^(N/2))``. When the pattern has multiplicities (collisions — exactly
the deep-photon regime), the sieve computes the same value in
``O(prod(n_i + 1) * (M^2 + N))``: orders of magnitude cheaper. This is the
repeated-row / finite-difference-sieve method of Kan (2008) as used for GBS by
Bulmer et al. (arXiv:2108.01622, The Walrus's ``hafnian(A, rpt, loop=True)``)
— reimplemented here from the identities below, sharing no code with either.

Parametrization (matches ``sampling/gbs._loop_submatrix`` exactly): the
expanded matrix is ``A`` with row/col ``i`` repeated ``n_i`` times — so copies
of ``i`` pair with copies of ``j != i`` at weight ``A_ij`` and with each other
at weight ``A_ii`` — and its *diagonal* (the loop weights) is ``gamma_i`` on
every copy of ``i``. ``gamma = 0`` gives the ordinary hafnian of the expansion.

Derivation (two algebraic identities; both verified numerically against the
expanded reference in the tests):

1. *Moment form.* The loop hafnian is the Isserlis/Wick matching sum, i.e. the
   formal Gaussian moment ``lhaf = E[prod_i y_i^{n_i}]`` under the pairing
   rules ``E[y_i y_j] = A_ij``, singleton ``-> gamma_i`` (a polynomial identity
   in the entries, so complex ``A``/``gamma`` are fine).
2. *Polarization (the sieve).* ``prod_i y_i^{n_i} = (1/(N! 2^N)) *
   sum_{v <= n} (-1)^{|v|} prod_i C(n_i, v_i) * (x_v . y)^N`` with
   ``x_v = n - 2v``; and the one-dimensional moment is closed-form:
   ``E[(x.y)^N] = sum_k C(N,2k) (2k-1)!! sigma^{2k} mu^{N-2k}`` with
   ``sigma^2 = x^T A x``, ``mu = x . gamma``.

Accuracy note (honest): the sieve is itself an alternating sum — better
conditioned than inclusion–exclusion (the reason arXiv:2108.01622 adopted it)
but not cancellation-free; the FP64 accuracy boundary applies as everywhere
else in this library. :func:`lhaf_repeated_certified` is the R1-machinery
variant: the same walk carrying a running error bound (standard model,
u = 2^-53; complex multiply <= 3u; any-order dots via gamma-constants;
integer data -- x, the binomial prefactors -- is exact, the E[s^N]
coefficients and N! are charged u-relative where they exceed 2^53), returning
``(value, bound)`` with ``|value - lhaf| <= bound`` (enclosure gated vs mpmath
in the tests, adversarial families included).
"""

from __future__ import annotations

from math import comb, factorial
from typing import Any

import numpy as np

from .permanent import _as_square_complex

__all__ = ["lhaf_repeated", "lhaf_repeated_certified", "sieve_term_count"]


def sieve_term_count(reps: Any) -> int:
    """``prod(n_i + 1)`` — the sieve's term count (its cost driver), vs the
    power-trace's ``2^(N/2)`` on the expansion. The dispatch decision quantity."""
    n = np.asarray(reps, dtype=np.int64)
    return int(np.prod(n + 1)) if n.size else 1


def lhaf_repeated(A: Any, gamma: Any, reps: Any) -> complex:
    """Loop hafnian of ``A`` expanded by ``reps`` with loop weights ``gamma``.

    Equals ``cpu_ref.lhaf(expanded)`` (and, for ``gamma = 0``,
    ``cpu_ref.haf(expanded)``) exactly in exact arithmetic; the tests pin both.
    """
    M = _as_square_complex(A)
    n = np.asarray(reps, dtype=np.int64)
    if n.ndim != 1 or n.shape[0] != M.shape[0]:
        raise ValueError(f"reps must be a length-{M.shape[0]} vector of counts, got {n.shape}")
    if (n < 0).any():
        raise ValueError("reps must be non-negative")
    g = (np.zeros(M.shape[0], dtype=np.complex128) if gamma is None
         else np.ascontiguousarray(gamma, dtype=np.complex128))
    if g.shape != (M.shape[0],):
        raise ValueError(f"gamma must be a length-{M.shape[0]} vector, got {g.shape}")

    N = int(n.sum())
    if N == 0:
        return complex(1.0)

    # E[s^N] coefficients C(N, 2k)*(2k-1)!!, exact integers.
    kmax = N // 2
    coeff = np.empty(kmax + 1, dtype=np.float64)
    for k in range(kmax + 1):
        df = 1
        for t in range(2 * k - 1, 0, -2):
            df *= t
        coeff[k] = float(comb(N, 2 * k) * df)

    total = 0j
    for v in np.ndindex(*(n + 1)):
        # v <-> n-v symmetry: x flips sign, S_N picks (-1)^N, the sign picks
        # (-1)^N too -- equal contributions. Keep v lexicographically <= n-v,
        # double the strict half (the self-paired v = n/2 term counts once).
        mirror = tuple(int(ni - vi) for ni, vi in zip(n, v))
        if v > mirror:
            continue
        va = np.asarray(v)
        x = (n - 2 * va).astype(np.float64)
        c = 1.0 if (int(va.sum()) % 2 == 0) else -1.0
        if v < mirror:
            c *= 2.0
        for ni, vi in zip(n, va):
            c *= comb(int(ni), int(vi))
        sig2 = complex(x @ M @ x)
        mu = complex(x @ g)
        # sum_k coeff[k] * sig2^k * mu^(N-2k), powers by running products
        mupow = np.empty(kmax + 1, dtype=np.complex128)
        mupow[kmax] = mu if N % 2 else 1.0           # mu^(N-2*kmax)
        for k in range(kmax - 1, -1, -1):
            mupow[k] = mupow[k + 1] * (mu * mu)
        SN = 0j
        sk = 1.0 + 0j
        for k in range(kmax + 1):
            SN += coeff[k] * sk * mupow[k]
            sk *= sig2
        total += c * SN
    return complex(total / (factorial(N) * 2.0 ** N))


_U = 2.0 ** -53
_C_MUL = 3.0 * _U


def _gam(k: float) -> float:
    ku = k * _U
    return ku / (1.0 - ku) * (1.0 + 4.0 * _U)


def _pmul(a, ea, b, eb):
    """(a±ea)(b±eb) pair product: value fl(a*b), bound per Lemma B1."""
    v = a * b
    aa, ab = abs(a), abs(b)
    return v, aa * eb + ab * ea + ea * eb + _C_MUL * aa * ab + 8e-323


def lhaf_repeated_certified(A: Any, gamma: Any, reps: Any) -> tuple[complex, float]:
    """Certified sieve: ``(value, bound)`` with ``|value - lhaf_exact| <= bound``.

    The value path is arithmetic-identical to :func:`lhaf_repeated` (same
    operation order), so the certificate covers the number the plain sieve
    returns; the bound follows every step of the running-error-bound
    machinery. CPU reference; the GPU kernel mirrors it.
    """
    M = _as_square_complex(A)
    n = np.asarray(reps, dtype=np.int64)
    if n.ndim != 1 or n.shape[0] != M.shape[0]:
        raise ValueError(f"reps must be a length-{M.shape[0]} vector of counts, got {n.shape}")
    if (n < 0).any():
        raise ValueError("reps must be non-negative")
    g = (np.zeros(M.shape[0], dtype=np.complex128) if gamma is None
         else np.ascontiguousarray(gamma, dtype=np.complex128))
    if g.shape != (M.shape[0],):
        raise ValueError(f"gamma must be a length-{M.shape[0]} vector, got {g.shape}")

    N = int(n.sum())
    if N == 0:
        return complex(1.0), 0.0
    Mdim = M.shape[0]
    Mabs = np.abs(M)                     # hoisted magnitudes (the CSE lesson)
    gabs = np.abs(g)
    c_dot = _gam(3.0 * (Mdim * Mdim + Mdim) + 6.0)   # generous any-order dot
    c_dotv = _gam(3.0 * Mdim + 6.0)

    kmax = N // 2
    coeff = np.empty(kmax + 1, dtype=np.float64)
    ecoeff = np.empty(kmax + 1, dtype=np.float64)
    for k in range(kmax + 1):
        df = 1
        for t in range(2 * k - 1, 0, -2):
            df *= t
        ci = comb(N, 2 * k) * df
        coeff[k] = float(ci)
        # exact if the integer fits 2^53; else the double conversion rounds
        ecoeff[k] = 0.0 if ci < 2 ** 53 else _U * float(ci)

    total = 0j
    e_tot = 0.0
    for v in np.ndindex(*(n + 1)):
        mirror = tuple(int(ni - vi) for ni, vi in zip(n, v))
        if v > mirror:
            continue
        va = np.asarray(v)
        x = (n - 2 * va).astype(np.float64)          # exact small integers
        c = 1.0 if (int(va.sum()) % 2 == 0) else -1.0
        if v < mirror:
            c *= 2.0
        for ni, vi in zip(n, va):
            c *= comb(int(ni), int(vi))              # exact (< 2^53 asserted below)
        xa = np.abs(x)
        sig2 = complex(x @ M @ x)
        e_sig = c_dot * float(xa @ Mabs @ xa) + Mdim * Mdim * 8e-323
        mu = complex(x @ g)
        e_mu = c_dotv * float(xa @ gabs) + Mdim * 8e-323
        # mu-power ladder (value order == plain sieve exactly)
        mupow = np.empty(kmax + 1, dtype=np.complex128)
        e_mupow = np.empty(kmax + 1, dtype=np.float64)
        if N % 2:
            mupow[kmax], e_mupow[kmax] = mu, e_mu
        else:
            mupow[kmax], e_mupow[kmax] = 1.0, 0.0
        mu2, e_mu2 = _pmul(mu, e_mu, mu, e_mu)
        for k in range(kmax - 1, -1, -1):
            mupow[k], e_mupow[k] = _pmul(mupow[k + 1], e_mupow[k + 1], mu2, e_mu2)
        SN = 0j
        e_SN = 0.0
        sk, e_sk = 1.0 + 0j, 0.0
        for k in range(kmax + 1):
            t1, e1 = _pmul(coeff[k], ecoeff[k], sk, e_sk)
            t2, e2 = _pmul(t1, e1, mupow[k], e_mupow[k])
            SN = SN + t2
            e_SN = e_SN + e2 + _U * abs(SN)
            sk, e_sk = _pmul(sk, e_sk, sig2, e_sig)
        total = total + c * SN
        e_tot = e_tot + abs(c) * e_SN + _C_MUL * abs(c) * abs(SN) + _U * abs(total)
    D = factorial(N) * 2.0 ** N                      # N! exact < 2^53 iff N <= 18
    e_D_rel = 0.0 if factorial(N) < 2 ** 53 else _gam(float(N))
    out = total / D
    bound = e_tot / D + abs(out) * (e_D_rel + _U)
    # terminal inflation for the RN bound arithmetic itself (cpu_ref regime;
    # generous op count, matching cpu_ref/certified.py's convention)
    ops = 64.0 * (kmax + 2.0) * float(np.prod(n + 1, dtype=np.float64)) + 64.0
    bound *= 1.0 / (1.0 - min(16.0 * _U * ops, 0.5))
    return complex(out), float(bound)
