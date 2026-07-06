"""Loop hafnian of a symmetric matrix — CPU reference implementations.

The loop hafnian generalizes the hafnian to allow "loops" — a point matched to
itself, weighted by the diagonal entry. It is what Gaussian states with nonzero
displacement (finite mean field) require (docs/DESIGN.md §2.1); it reduces to the
ordinary hafnian when the diagonal is zero.

    lhaf(A) = sum_{M in LPM(N)} prod_{(i,j) in M, i!=j} A[i,j] * prod_{(i) in M} A[i,i]

over "loop perfect matchings" LPM(N): partitions of the N points into pairs and
singletons (loops). Unlike the ordinary hafnian, N may be odd (leftover points
become loops).

Two algorithms:

* :func:`loop_hafnian_naive` -- literal recursive sum over loop matchings; works
  for any ``N``. Ground truth.
* :func:`loop_hafnian_powertrace` -- the power-trace formula for even ``N``,
  reusing the ordinary-hafnian subset engine and adding the diagonal (loop) term

      v_k = (1/2) d^T X (B_S X)^{k-1} d

  to the generating-function exponent, where ``d`` is the diagonal of the
  selected submatrix. This term was derived and then pinned numerically against
  the naive enumeration AND The Walrus ``loop=True`` (see tests). The GPU-target
  algorithm; shares the subset skeleton with the hafnian (docs/DESIGN.md §5).
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from .hafnian import _power_traces, exp_coeff_from_kg
from .permanent import _as_square_complex

__all__ = ["lhaf", "loop_hafnian_naive", "loop_hafnian_powertrace"]


def loop_hafnian_naive(A: Any) -> complex:
    """O((N-1)!!-ish) loop hafnian: literal recursive sum over loop matchings.

    Works for any ``N`` (even or odd). Small sizes only.
    """
    M = _as_square_complex(A)
    N = M.shape[0]
    if N == 0:
        return complex(1.0)

    def rec(rem: tuple[int, ...]) -> complex:
        if not rem:
            return 1.0 + 0j
        i, rest = rem[0], rem[1:]
        total = M[i, i] * rec(rest)  # i as a loop
        for k in range(len(rest)):   # i paired with rest[k]
            j = rest[k]
            total += M[i, j] * rec(rest[:k] + rest[k + 1 :])
        return total

    return complex(rec(tuple(range(N))))


def loop_hafnian_powertrace(A: Any) -> complex:
    """Loop hafnian via the power-trace formula with the diagonal term (even N).

    ``lhaf(A) = sum_{S subset [n]} (-1)^(n-|S|) [lambda^n]
        exp(sum_k [ tr(C^k)/(2k) + (1/2) d^T X C^{k-1} d ] lambda^k)``
    with ``C = B_S X``, ``d = diag(B_S)``, ``n = N/2``. Reduces to the ordinary
    hafnian when the diagonal is zero (``v_k = 0``).
    """
    M = _as_square_complex(A)
    N = M.shape[0]
    if N == 0:
        return complex(1.0)
    if N % 2 == 1:
        raise ValueError(
            "loop_hafnian_powertrace requires even N; use loop_hafnian_naive for odd N"
        )
    n = N // 2

    total = 0j
    for m in range(n + 1):
        for S in itertools.combinations(range(n), m):
            if m == 0:
                continue  # empty subset contributes [lambda^n] exp(0) = 0 for n>=1
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
            # loop term v_k = (1/2) d^T X C^{k-1} d ; kg[k] = k*(p_k/(2k) + v_k)
            kg = np.zeros(n + 1, dtype=np.complex128)
            Ckm1 = np.eye(size, dtype=np.complex128)  # C^0
            for k in range(1, n + 1):
                v_k = 0.5 * (d @ Xs @ Ckm1 @ d)
                kg[k] = p[k] / 2.0 + k * v_k
                if k < n:
                    Ckm1 = Ckm1 @ C
            total += ((-1) ** (n - m)) * exp_coeff_from_kg(kg, n)
    return complex(total)


def lhaf(A: Any) -> complex:
    """Loop hafnian of symmetric ``A`` (FP64 tier).

    Uses the power-trace algorithm for even ``N`` and the naive enumeration for
    odd ``N`` (the power-trace form here is derived for even ``N``).
    """
    M = _as_square_complex(A)
    if M.shape[0] % 2 == 1:
        return loop_hafnian_naive(M)
    return loop_hafnian_powertrace(M)
