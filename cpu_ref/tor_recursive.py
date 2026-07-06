"""Recursive prefix-Cholesky torontonian.

Same value as :func:`cpu_ref.torontonian` — ``tor(O) = sum_S (-1)^(n-|S|) /
sqrt(det(I - O_S))`` over mode subsets, xxpp ordering — computed by walking the
subset lattice as a **tree** and *reusing the parent's Cholesky factor* instead
of recomputing each subset determinant from scratch. This is the recursive
algorithm of Kaposi, Kolarovszki et al. (arXiv:2109.04528, shipped CPU-side in
Piquasso); reimplemented here from the idea, sharing no code, as the reference
the GPU kernel (``core/tor_recursive.cu``) mirrors.

The two structural facts that make it work:

* **Mode-major ordering.** Build ``I - O_S`` with each included mode's (x, p)
  rows adjacent, appended in mode order. Simultaneous row/col permutations
  leave the determinant unchanged, so this is the same determinant as the
  xxpp-blocked reference — but now *including one more mode appends two
  trailing rows/columns*.
* **Prefix property.** The Cholesky factor of a leading principal submatrix is
  the leading block of the factor. So a DFS over "exclude/include mode j"
  needs ONE factor buffer: including appends two rows (O(size^2) work),
  backtracking just forgets them (no downdating, nothing recomputed), and
  ``det = prod diag(L)^2`` is maintained as a per-level running product.

Per-subset amortized cost drops from O(|S|^3) (fresh determinant) to O(|S|^2)
(the appended rows), the polynomial-factor speedup the paper measures. The
factorization is the complex-symmetric ``L L^T`` (no conjugation; principal
branch on the diagonal square roots). On the physical domain (real ``O``,
``I - O_S`` SPD) this is the ordinary real Cholesky and unconditionally sound;
off it there is no pivoting freedom inside the tree walk (pivoting would break
the prefix reuse), so a breakdown surfaces as NaN/Inf rather than silently —
callers fall back to the direct reference, exactly like the real-Cholesky
kernel (the real-Cholesky physical-domain path).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .torontonian import _as_even_complex

__all__ = ["torontonian_recursive"]


def torontonian_recursive(O: Any) -> complex:
    """Torontonian of the ``2n x 2n`` matrix ``O`` via prefix-Cholesky DFS."""
    M, n = _as_even_complex(O)
    if n == 0:
        return complex(1.0)

    # I - O in xxpp; index of mode i's components: x -> i, p -> i + n.
    IO = np.eye(2 * n, dtype=np.complex128) - M

    dim = 2 * n
    L = np.zeros((dim, dim), dtype=np.complex128) # the ONE factor buffer
    modes = np.empty(n, dtype=np.int64) # included modes, in order
    total = 0.0 + 0.0j

    def append_mode(j: int, size: int, count: int) -> complex | None:
        """Append mode j's two rows to L (rows size, size+1) given `count`
        already-included modes; return the det contribution (l_xx * l_pp)^2,
        or None on breakdown (zero/invalid pivot)."""
        for t in range(2): # t=0: x-row, t=1: p-row
            r = size + t
            gj = j if t == 0 else j + n # global index of the new row
            for c in range(r + 1):
                # global index of column c: mode modes[c // 2], comp c % 2
                mc = j if (c // 2 == count) else modes[c // 2]
                gc = (mc if (c % 2 == 0) else mc + n)
                s = IO[gj, gc]
                acc = s - np.dot(L[r, :c], L[c, :c]) # plain L L^T: no conjugation
                if c < r:
                    piv = L[c, c]
                    if piv == 0:
                        return None
                    L[r, c] = acc / piv
                else:
                    val = np.sqrt(acc)
                    if not np.isfinite(val.real) or not np.isfinite(val.imag) or val == 0:
                        return None
                    L[r, r] = val
        return (L[size, size] * L[size + 1, size + 1]) ** 2

    def rec(mode: int, count: int, detprod: complex) -> None:
        nonlocal total
        if mode == n:
            total += ((-1) ** (n - count)) / np.sqrt(detprod)
            return
        rec(mode + 1, count, detprod) # exclude `mode`
        modes[count] = mode # include `mode`
        d = append_mode(mode, 2 * count, count)
        if d is None:
            raise FloatingPointError(
                "prefix-Cholesky breakdown (no pivoting inside the tree walk); "
                "use cpu_ref.torontonian for this input"
            )
        rec(mode + 1, count + 1, detprod * d)

    rec(0, 0, 1.0 + 0.0j)
    return complex(total)
