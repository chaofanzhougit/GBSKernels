"""Certified evaluation — rigorous a-posteriori error bounds.

For the permanent and the hafnian, compute the **same FP64 value as the shipped
reference algorithm, bit for bit** (:func:`cpu_ref.permanent.permanent_glynn`,
:func:`cpu_ref.hafnian.hafnian_powertrace` — identical operations in identical
order), and alongside it a bound ``E`` with

    |computed - exact| <= E (exact = the true real/complex-arithmetic value)

so the certificate applies to the very value the fast path returns, not to some
alternative computation. This is the rigorous counterpart of the ``kappa``
cancellation indicator (``cpu_ref.diagnostics``): the dominant term of ``E`` is
``~ u * sum_S |term_S|`` — the same quantity ``kappa`` estimates, but carried as
a proven running bound (Higham, *Accuracy and Stability of Numerical
Algorithms*, ch. 3) instead of a calibrated heuristic. No existing
implementation of these functions publishes such a bound (see
the GPU kernels mirror this in CUDA:
per-instruction directed rounding makes the device tier *stronger* than this
reference).

Validity model (stated precisely, so the claim is checkable):

* IEEE-754 binary64, round-to-nearest: every scalar operation satisfies
  ``fl(a op b) = (a op b)(1 + delta)``, ``|delta| <= u = 2**-53``; complex
  multiplication satisfies ``|fl(xy) - xy| <= C_MUL * |x||y|`` with
  ``C_MUL = 3u > sqrt(2)*gamma_2`` (Higham sec. 3.6).
* Sums/dot products may be evaluated in **any** order (sequential, pairwise,
  blocked BLAS, with or without FMA): length-``d`` accumulations are bounded
  with a generous ``gamma``-constant covering all conventional (non-Strassen)
  orders. numpy's complex ``@`` is conventional GEMM.
* ``|.|`` upper bounds inflate ``abs()`` by ``1 + 2**-50`` (covers any faithful
  libm ``hypot``); each multiply carries an absolute underflow slop ``UFL``.
* The bound accumulators are themselves FP64; their own rounding is absorbed by
  a final multiplicative inflation ``1/(1 - 16u*OPS)`` from a generous
  operation-count over-estimate (guarded: raises if sizes ever make it
  non-negligible).

Scope: all four functions (``perm``, ``haf``, ``lhaf``, ``tor``).
Tightness is *measured*, not assumed: the test contract asserts the enclosure
contains the mpmath value on every input, including the adversarial
cancellation families, and tracks bound/true-error ratios.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from .hafnian import exp_coeff_from_kg
from .permanent import _as_square_complex

__all__ = ["certified_permanent", "certified_hafnian", "certified_loop_hafnian",
           "certified_torontonian", "certified"]

U = 2.0**-53 # unit roundoff, binary64 round-to-nearest
C_MUL = 3.0 * U # complex multiply relative bound (> sqrt(2)*gamma_2)
ABS_SAFE = 1.0 + 2.0**-50 # |.| upper-bound safety (faithful hypot)
UFL = 8e-323 # absolute underflow slop per multiply (16*eta_min)
MIN_NORMAL = np.finfo(np.float64).tiny


def _ru_add(a: float, b: float) -> float:
    """Binary64 addition rounded outward toward +inf for nonnegative inputs."""
    if a == 0.0:
        return float(b)
    if b == 0.0:
        return float(a)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value = np.float64(a) + np.float64(b)
    return float(np.nextafter(value, np.inf))


def _ru_mul(a: float, b: float) -> float:
    """Binary64 multiplication rounded outward toward +inf for nonnegative inputs."""
    if a == 0.0 or b == 0.0:
        return 0.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value = np.float64(a) * np.float64(b)
    return float(np.nextafter(value, np.inf))


def _ru_div(a: float, b: float) -> float:
    """Binary64 division rounded outward toward +inf for a >= 0 and b > 0."""
    if a == 0.0:
        return 0.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        value = np.float64(a) / np.float64(b)
    return float(np.nextafter(value, np.inf))


def _rd_mul(a: float, b: float) -> float:
    """Nonnegative lower bound for the product of nonnegative binary64 inputs."""
    if a == 0.0 or b == 0.0:
        return 0.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value = np.float64(a) * np.float64(b)
    return max(0.0, float(np.nextafter(value, -np.inf)))


def _rd_sub(a: float, b: float) -> float:
    """Nonnegative lower bound for a-b; callers establish a > b first."""
    with np.errstate(under="ignore", invalid="ignore"):
        value = np.float64(a) - np.float64(b)
    return max(0.0, float(np.nextafter(value, -np.inf)))


def _gamma(k: float) -> float:
    """gamma_k = k*u / (1 - k*u), the standard-model accumulation constant."""
    ku = k * U
    if ku >= 0.5:
        raise ValueError("problem size outside the certified model's range")
    return ku / (1.0 - ku)


def _inflate(bound: float, ops: float) -> float:
    """Absorb the bound accumulators' own FP64 rounding: multiply by
    1/(1 - 16u*OPS) for a generous over-count OPS of bound-arithmetic ops."""
    slack = 16.0 * U * ops
    if slack >= 2.0**-16:
        raise ValueError("operation count too large for the certified model")
    return bound / (1.0 - slack)


def _absu(z) -> Any:
    """Upper bound on |z| (scalar or array)."""
    return np.nextafter(np.abs(z) * ABS_SAFE + UFL, np.inf)


def _absl(z) -> Any:
    """Lower bound on |z| (scalar or array)."""
    return np.maximum(0.0, np.nextafter(
        np.abs(z) * (2.0 - ABS_SAFE) - UFL, -np.inf))


# --------------------------------------------------------------------------
# permanent — certified Glynn/BB-FG (value path identical to permanent_glynn)
# --------------------------------------------------------------------------

def certified_permanent(A: Any) -> tuple[complex, float]:
    """``(value, bound)``: ``value`` is bit-for-bit ``permanent_glynn(A)``;
    ``|value - perm(A)| <= bound`` rigorously (module validity model).

    The bound tracks (a) the initial row sums (any-order ``gamma_n``), (b) the
    Gray-code delta walk — each of the ``2^(n-1)`` steps adds one rounding to
    every row-sum element, the price of the delta reuse the GPU kernel shares —
    (c) each signed product term (polydisc propagation of the row-sum bounds
    plus the product's own relative error), and (d) the alternating
    accumulation, whose ``u*sum|term|`` character is exactly the rigorous
    ``kappa``.
    """
    M = _as_square_complex(A)
    n = M.shape[0]
    if n == 0:
        return complex(1.0), 0.0

    # --- value path: identical operations, identical order, to permanent_glynn
    rowsum = M.sum(axis=1).astype(np.complex128)

    # --- bound state
    g_sum = _gamma(n) # any-order length-n sum
    e_r = g_sum * np.sum(_absu(M), axis=1) + n * UFL

    def product_with_bound() -> tuple[np.complex128, float]:
        """The actual NumPy product plus an op-by-op pair enclosure.

        The shadow product propagates absolute multiply error through every
        later factor. If NumPy ever chooses a different reduction result, fail
        closed rather than attach this order-specific bound to another value.
        """
        p = np.complex128(rowsum[0])
        ep = float(e_r[0])
        for r in range(1, n):
            A_, B_ = float(_absu(p)), float(_absu(rowsum[r]))
            er = float(e_r[r])
            p = np.complex128(p * rowsum[r])
            ep = A_ * er + B_ * ep + ep * er + C_MUL * A_ * B_ + UFL
        value = np.prod(rowsum)
        if complex(value) != complex(p):
            return value, float("inf")
        return value, ep

    total, e_tot = product_with_bound()

    sign = 1
    prev_gray = 0
    for i in range(1, 1 << (n - 1)):
        gray = i ^ (i >> 1)
        k = (gray ^ prev_gray).bit_length() - 1
        col = k + 1
        if (gray >> k) & 1: # 2.0*M[:,col] is exact (power of 2)
            rowsum -= 2.0 * M[:, col]
        else:
            rowsum += 2.0 * M[:, col]
        e_r += U * _absu(rowsum) + UFL
        sign = -sign
        term, term_bound = product_with_bound()
        total += sign * term # sign flip exact
        e_tot += term_bound + U * float(_absu(total)) + UFL
        prev_gray = gray

    total = total / (1 << (n - 1)) # power-of-two scale: exact
    e_tot = e_tot / float(1 << (n - 1)) + UFL

    ops = 8.0 * (n + 4) * float(1 << max(n - 1, 0))
    return complex(total), _inflate(e_tot, ops)


# --------------------------------------------------------------------------
# hafnian — certified power-trace (value path identical to hafnian_powertrace)
# --------------------------------------------------------------------------

def _certified_power_traces(C: np.ndarray, n: int):
    """``(p, e_tr)`` with ``p`` bit-for-bit ``_power_traces(C, n)`` (C exact) and
    ``|p[k] - tr(C_true^k)| <= e_tr[k]``: entrywise matmul bounds
    ``E_k = E_{k-1}|C| + c_dot*|P_{k-1}||C|`` plus the trace-sum rounding."""
    d = C.shape[0]
    c_dot = _gamma(3.0 * d + 6.0) # any-order complex dot, len d (generous)
    Cabs = _absu(C)
    p = np.zeros(n + 1, dtype=np.complex128)
    e_tr = np.zeros(n + 1)
    P = C.copy()
    E = np.zeros((d, d)) # entrywise |P_hat - P_true| bound
    g_trace = _gamma(d)
    for k in range(1, n + 1):
        p[k] = np.trace(P)
        e_tr[k] = float(np.sum(np.diag(E))) + g_trace * float(np.sum(_absu(np.diag(P))))
        if k < n:
            Pabs = _absu(P)
            E = E @ Cabs + c_dot * (Pabs @ Cabs) + d * UFL
            P = P @ C
    return p, e_tr


def _newton_from_kg(kg: np.ndarray, e_kg: np.ndarray, n: int) -> tuple[complex, float]:
    """``([lambda^n] exp(sum_k g_k lambda^k), bound)`` from ``kg[k] = k*g_k`` with
    per-coefficient bounds ``e_kg``; values mirror ``exp_coeff_from_kg`` exactly."""
    e = np.zeros(n + 1, dtype=np.complex128)
    e[0] = 1.0
    eb = np.zeros(n + 1) # bounds for e[j]
    for j in range(1, n + 1):
        acc = 0j
        acc_b = 0.0
        for k in range(1, j + 1):
            prod = kg[k] * e[j - k]
            A_, B_ = float(_absu(kg[k])), float(_absu(e[j - k]))
            prod_b = A_ * eb[j - k] + B_ * e_kg[k] + e_kg[k] * eb[j - k] \
                + C_MUL * A_ * B_ + UFL
            acc = acc + prod
            acc_b = acc_b + prod_b + U * float(_absu(acc))
        e[j] = acc / j
        eb[j] = acc_b / j + U * float(_absu(e[j]))
    return complex(e[n]), float(eb[n])


def _certified_exp_newton(C: np.ndarray, n: int) -> tuple[complex, float]:
    """``([lambda^n] exp(sum_k tr(C^k)/(2k) lambda^k), bound)`` for one subset.

    Mirrors ``_power_traces`` + ``exp_coeff_from_kg`` operation-for-operation
    (see the split helpers); ``kg = p/2`` is a power-of-two scale, exact."""
    if C.shape[0] == 0:
        return (1.0 + 0j if n == 0 else 0j), 0.0
    p, e_tr = _certified_power_traces(C, n)
    return _newton_from_kg(p / 2.0, e_tr / 2.0, n)


def certified_hafnian(A: Any) -> tuple[complex, float]:
    """``(value, bound)``: ``value`` is bit-for-bit ``hafnian_powertrace(A)``;
    ``|value - haf(A)| <= bound`` rigorously (module validity model).

    The subset loop mirrors :func:`cpu_ref.hafnian.hafnian_powertrace`
    (same ``itertools.combinations`` order, same submatrix construction — both
    exact — same accumulation); each subset contributes its per-subset bound
    plus one rounding of the running sum, so the final bound's dominant term is
    the rigorous form of the ``kappa`` cancellation indicator.
    """
    M = _as_square_complex(A).copy()
    N = M.shape[0]
    if N == 0:
        return complex(1.0), 0.0
    if N % 2 == 1:
        return complex(0.0), 0.0
    np.fill_diagonal(M, 0.0)
    n = N // 2

    total = 0j
    e_tot = 0.0
    ops = 0.0
    for m in range(n + 1):
        for S in itertools.combinations(range(n), m):
            if m == 0:
                coeff, cb = _certified_exp_newton(
                    np.empty((0, 0), dtype=np.complex128), n)
            else:
                idx = [q for i in S for q in (2 * i, 2 * i + 1)]
                B = M[np.ix_(idx, idx)]
                swap = [q for t in range(m) for q in (2 * t + 1, 2 * t)]
                coeff, cb = _certified_exp_newton(B[:, swap], n)
            total += ((-1) ** (n - m)) * coeff # sign exact
            e_tot += cb + U * float(_absu(total))
            ops += 24.0 * n * (2 * m) ** 3 + 40.0 * n * n + 64.0
    return complex(total), _inflate(e_tot, ops)


# --------------------------------------------------------------------------
# loop hafnian — certified (even N mirrors loop_hafnian_powertrace; odd N
# mirrors loop_hafnian_naive), value paths bit-for-bit
# --------------------------------------------------------------------------

def certified_loop_hafnian(A: Any) -> tuple[complex, float]:
    """``(value, bound)``: ``value`` is bit-for-bit ``cpu_ref.lhaf(A)``
    (power-trace mirror for even N, naive-recursion mirror for odd N);
    ``|value - lhaf(A)| <= bound`` rigorously (module validity model).

    Even N adds the loop term ``v_k = (1/2) d^T X C^{k-1} d`` to the hafnian
    machinery: ``d`` and ``d@Xs`` are exact (diagonal copy / permutation), the
    ``C^{k-1}`` chain carries the same entrywise matmul bounds as the power
    traces, and the two dot products carry input-error + any-order-dot terms.
    """
    M = _as_square_complex(A)
    N = M.shape[0]
    if N == 0:
        return complex(1.0), 0.0
    if N % 2 == 1:
        return _lhaf_naive_pair(M)
    n = N // 2

    total = 0j
    e_tot = 0.0
    ops = 0.0
    for m in range(n + 1):
        for S in itertools.combinations(range(n), m):
            if m == 0:
                continue # mirrors loop_hafnian_powertrace
            idx = [q for i in S for q in (2 * i, 2 * i + 1)]
            B = M[np.ix_(idx, idx)] # exact copy
            size = 2 * m
            Xs = np.zeros((size, size), dtype=np.complex128)
            for t in range(m):
                Xs[2 * t, 2 * t + 1] = 1.0
                Xs[2 * t + 1, 2 * t] = 1.0
            C = B @ Xs # permutation matmul: exact
            d = np.diag(B).copy() # exact

            p, e_p = _certified_power_traces(C, n)

            c_dot = _gamma(3.0 * size + 6.0)
            Cabs = _absu(C)
            dabs = _absu(d)
            kg = np.zeros(n + 1, dtype=np.complex128)
            e_kg = np.zeros(n + 1)
            Ckm1 = np.eye(size, dtype=np.complex128) # C^0, exact
            E_km1 = np.zeros((size, size))
            dXs = d @ Xs # vector permutation: exact
            dXs_abs = _absu(dXs)
            for k in range(1, n + 1):
                # value path mirrors 0.5 * (d @ Xs @ Ckm1 @ d), left-to-right
                w = dXs @ Ckm1
                e_w = dXs_abs @ E_km1 + c_dot * (dXs_abs @ _absu(Ckm1)) + size * UFL
                v_k = 0.5 * (w @ d) # trailing dot, then exact halving
                e_dot = float(e_w @ dabs) + c_dot * float(_absu(w) @ dabs) + size * UFL
                e_vk = 0.5 * e_dot
                tmp = k * v_k # fl integer multiply
                e_tmp = k * e_vk + U * float(_absu(tmp))
                kg[k] = p[k] / 2.0 + tmp # p/2 exact; one fl add
                e_kg[k] = e_p[k] / 2.0 + e_tmp + U * float(_absu(kg[k]))
                if k < n:
                    E_km1 = E_km1 @ Cabs + c_dot * (_absu(Ckm1) @ Cabs) + size * UFL
                    Ckm1 = Ckm1 @ C
            coeff, cb = _newton_from_kg(kg, e_kg, n)
            total += ((-1) ** (n - m)) * coeff
            e_tot += cb + U * float(_absu(total))
            ops += 48.0 * n * size**3 + 60.0 * n * n + 64.0
    return complex(total), _inflate(e_tot, ops)


def _lhaf_naive_pair(M: np.ndarray) -> tuple[complex, float]:
    """Certified mirror of ``loop_hafnian_naive`` (any N; used for odd N)."""
    def rec(rem: tuple[int, ...]):
        if not rem:
            return 1.0 + 0j, 0.0
        i, rest = rem[0], rem[1:]
        v0, e0 = rec(rest)
        total = M[i, i] * v0 # i as a loop (M entry exact)
        a0 = float(_absu(M[i, i]))
        e_tot = a0 * e0 + C_MUL * a0 * float(_absu(v0)) + UFL
        for k in range(len(rest)): # i paired with rest[k]
            j = rest[k]
            v1, e1 = rec(rest[:k] + rest[k + 1:])
            term = M[i, j] * v1
            a1 = float(_absu(M[i, j]))
            e_term = a1 * e1 + C_MUL * a1 * float(_absu(v1)) + UFL
            total = total + term # mirrors +=
            e_tot = e_tot + e_term + U * float(_absu(total))
        return total, e_tot

    v, e = rec(tuple(range(M.shape[0])))
    n = M.shape[0]
    fact = 1.0
    for x in range(2, n + 1):
        fact *= x
    return complex(v), _inflate(float(e), 8.0 * n * fact)


# --------------------------------------------------------------------------
# torontonian — certified (own LU in pair arithmetic; value within tier
# tolerance of the reference, the bound covers THIS value)
# --------------------------------------------------------------------------
#
# Pair-arithmetic toolkit for the scalar complex ops the LU needs. Each op's
# bound is derived in its docstring under the module validity model; anything
# non-finite or too-uncertain degrades to bound = inf ("cannot certify"),
# never to a silent claim.

def _p_mul(a, ea: float, b, eb: float):
    v = a * b
    A_, B_ = float(_absu(a)), float(_absu(b))
    return v, A_ * eb + B_ * ea + ea * eb + C_MUL * A_ * B_ + UFL


def _p_div(a, ea: float, b, eb: float):
    """(a +- ea) / (b +- eb) with the textbook component division.

    Algorithmic error (den = c^2+d^2 has theta_3; each numerator dot has
    gamma_3 <= 3.1u absolute over |a||b| by Cauchy-Schwarz; final real divides
    u each): <= 8u*A_hi*B_hi/den_lo + 8u*|v|. Perturbation:
    (ea*B_hi + eb*A_hi) / (B_lo*(B_lo-eb)). Upper magnitudes appear only in
    numerators and lower magnitudes only in guards/denominators. The unscaled
    component formula is evaluated only when both lower denominator products,
    every nonzero numerator component product, and every nontrivial numerator
    dot result are guaranteed normal. The perturbation bound is evaluated in a
    factorized form with outward-rounded positive arithmetic, so a tiny ``ea``
    cannot disappear before division by a tiny ``|b|``. Guarded: invalid
    bounds, eb >= B_lo/2, any unsupported subnormal value operation, or
    non-finite intermediates -> NaN value and bound inf.
    """
    B_hi = float(_absu(b))
    B_lo = float(_absl(b))
    A_hi = float(_absu(a))
    refused = (complex(np.nan, np.nan), float("inf"))
    if (not np.isfinite(A_hi) or not np.isfinite(B_hi)
            or not np.isfinite(ea) or not np.isfinite(eb)
            or ea < 0.0 or eb < 0.0 or B_lo <= 0.0 or eb >= 0.5 * B_lo):
        return refused
    b_pert_lo = _rd_sub(B_lo, eb)
    normal_factor = _ru_div(MIN_NORMAL, B_lo)
    if B_lo < normal_factor or b_pert_lo < normal_factor:
        return refused
    den_lo = _rd_mul(_rd_mul(B_lo, B_lo), 1.0 - 8.0 * U)
    pert_den_lo = _rd_mul(B_lo, b_pert_lo)
    if (not np.isfinite(den_lo) or not np.isfinite(pert_den_lo)
            or den_lo <= 0.0 or pert_den_lo <= 0.0):
        return refused

    ar, ai = float(a.real), float(a.imag)
    c, dd_ = float(b.real), float(b.imag)

    def normal_product_or_zero(x: float, y: float) -> bool:
        if x == 0.0 or y == 0.0:
            return True
        return abs(x) >= _ru_div(MIN_NORMAL, abs(y))

    component_pairs = ((ar, c), (ai, dd_), (ai, c), (ar, dd_))
    if not all(normal_product_or_zero(x, y) for x, y in component_pairs):
        return refused

    den = c * c + dd_ * dd_
    if not np.isfinite(den) or den <= 0.0:
        return refused
    re0, re1 = ar * c, ai * dd_
    im0, im1 = ai * c, ar * dd_
    num_re = re0 + re1
    num_im = im0 - im1

    def normal_dot_or_exact_zero(value: float, pair0, pair1) -> bool:
        terms_exactly_zero = ((pair0[0] == 0.0 or pair0[1] == 0.0)
                              and (pair1[0] == 0.0 or pair1[1] == 0.0))
        return terms_exactly_zero or (np.isfinite(value) and abs(value) >= MIN_NORMAL)

    if (not normal_dot_or_exact_zero(num_re, (ar, c), (ai, dd_))
            or not normal_dot_or_exact_zero(num_im, (ai, c), (ar, dd_))):
        return refused

    vre = num_re / den
    vim = num_im / den
    v = complex(vre, vim)
    if not (np.isfinite(vre) and np.isfinite(vim)):
        return refused

    if ar == 0.0 and ai == 0.0:
        e_alg = 16.0 * UFL
    else:
        ab_hi = _ru_mul(A_hi, B_hi)
        e_alg = _ru_add(
            _ru_div(_ru_mul(8.0 * U, ab_hi), den_lo),
            _ru_add(_ru_mul(8.0 * U, float(_absu(v))), 16.0 * UFL),
        )

    # Algebraically identical to
    #   (ea*B_hi + eb*A_hi) / (B_lo*(B_lo-eb)),
    # but this ordering keeps tiny perturbations from underflowing before the
    # small denominator amplifies them.
    e_pert = _ru_add(
        _ru_mul(_ru_div(ea, b_pert_lo), _ru_div(B_hi, B_lo)),
        _ru_mul(_ru_div(eb, B_lo), _ru_div(A_hi, b_pert_lo)),
    )
    return v, _ru_add(e_alg, e_pert)


def _p_inv_sqrt(z, ez: float):
    """(1/sqrt(z +- ez), bound), principal branch.

    Branch guard: if the disc (z, ez) can touch the negative real axis (or 0),
    the principal square root is not certifiable -> bound inf. A-posteriori
    sqrt bound: with s = fl(sqrt(z)) and delta >= |s^2 - z| + ez, if
    delta <= |z|/2 then |s - sqrt(z_true)| <= delta / sqrt(|z|) (monotonicity
    of t*(2*sqrt|z| - t)). The reciprocal reuses _p_div on (1, 0)/(s, t).
    """
    Zm = float(np.abs(z))  # value path
    Z_hi = float(_absu(z))
    Z_lo = float(_absl(z))
    if not np.isfinite(Z_hi) or Z_lo <= 0.0 or ez >= 0.5 * Z_lo:
        return complex(np.nan), float("inf")
    if z.real < 0.0 and abs(z.imag) <= ez: # disc may cross the branch cut
        return complex(np.sqrt(z)), float("inf")
    s = complex(np.sqrt(z))
    sq, e_sq = _p_mul(s, 0.0, s, 0.0) # fl(s*s) with its own bound
    delta = (float(_absu(sq - z)) + e_sq
             + U * (float(_absu(sq)) + Z_hi) + ez)
    if delta > 0.5 * Z_lo:
        return s, float("inf")
    t = delta / np.sqrt(Z_lo * (1.0 - 4.0 * U))
    return _p_div(1.0 + 0j, 0.0, s, t)


def _certified_det_lu(Ap: list[list[tuple[Any, float]]]) -> tuple[complex, float]:
    """Determinant of a small complex matrix held as (value, bound) pairs, by
    LU with partial pivoting carried in pair arithmetic. The pivot CHOICE needs
    no certification (any pivot order yields det up to the tracked sign); the
    arithmetic of the chosen sequence is what the bounds follow."""
    k = len(Ap)
    if k == 0:
        return 1.0 + 0j, 0.0
    A = [row[:] for row in Ap]
    det, e_det = 1.0 + 0j, 0.0
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(A[r][col][0]))
        if piv != col:
            A[col], A[piv] = A[piv], A[col]
            det, e_det = -det, e_det # sign flip exact
        pv, pe = A[col][col]
        det, e_det = _p_mul(det, e_det, pv, pe)
        if not np.isfinite(e_det):
            return complex(det), float("inf")
        if abs(pv) == 0.0:
            return complex(det * 0.0), e_det # exactly singular pivot: det=0 path
        for r in range(col + 1, k):
            f, fe = _p_div(*A[r][col], pv, pe)
            for c in range(col + 1, k):
                m_, me_ = _p_mul(f, fe, *A[col][c])
                v0, e0 = A[r][c]
                v1 = v0 - m_
                A[r][c] = (v1, e0 + me_ + U * float(_absu(v1)) + UFL)
    return complex(det), float(e_det)


def certified_torontonian(O: Any) -> tuple[complex, float]:
    """``(value, bound)`` for the torontonian, xxpp ordering.

    Unlike perm/haf/lhaf, the value path is the certified module's own LU (the
    numpy reference uses LAPACK internals we cannot bound), so ``value`` agrees
    with ``cpu_ref.tor`` to tier tolerance rather than bit-for-bit — the bound
    covers *this returned value*, which is what the API hands back. Physical
    (real ``O``) inputs are the validated domain; a subset whose determinant
    disc could touch the branch cut degrades that evaluation's bound to inf
    (never a silent claim), as does a non-finite intermediate.
    """
    from .torontonian import _as_even_complex

    M, n = _as_even_complex(O)
    if n == 0:
        return complex(1.0), 0.0

    total = 0j
    e_tot = 0.0
    ops = 0.0
    for m in range(n + 1):
        for S in itertools.combinations(range(n), m):
            if m == 0:
                term, tb = 1.0 + 0j, 0.0 # det(I) = 1, 1/sqrt(1) = 1
            else:
                idx = [i for i in S] + [i + n for i in S]
                sub = M[np.ix_(idx, idx)] # exact copy
                k = 2 * m
                # I - sub: off-diagonal negation exact; diagonal one fl subtract
                Ap = []
                for r in range(k):
                    row = []
                    for c in range(k):
                        if r == c:
                            v = 1.0 - sub[r, c]
                            row.append((v, U * float(_absu(v)) + UFL))
                        else:
                            row.append((-sub[r, c], 0.0))
                    Ap.append(row)
                det, e_det = _certified_det_lu(Ap)
                term, tb = _p_inv_sqrt(det, e_det)
                ops += 10.0 * k**3 + 40.0 * k + 64.0
            if (m - n) % 2 == 0: # (-1)^(n-m) exact
                total = total + term
            else:
                total = total - term
            e_tot += tb + U * float(_absu(total)) + UFL
    return complex(total), _inflate(e_tot, ops)


_CERTIFIED = {"perm": certified_permanent, "haf": certified_hafnian,
              "lhaf": certified_loop_hafnian, "tor": certified_torontonian}


def certified(func: str, A: Any) -> tuple[complex, float]:
    """Dispatch: ``(value, abs_error_bound)`` for all four functions."""
    if func not in _CERTIFIED:
        raise NotImplementedError(
            f"precision='certified' has no evaluator for {func!r} "
            f"(implemented: {sorted(_CERTIFIED)})."
        )
    return _CERTIFIED[func](A)
