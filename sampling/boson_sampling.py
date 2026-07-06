"""Standard boson sampling probabilities via the permanent (docs/DESIGN.md §2.2).

For an ``m``-mode interferometer ``U`` fed ``n`` single photons in distinct input
modes ``S``, the probability of detecting output configuration ``T`` (a multiset
of ``n`` modes) is

    P(T) = |perm(U[T, S])|^2 / (prod_k t_k!  *  prod_i s_i!)

where ``U[T, S]`` is the ``n x n`` matrix whose rows are the output modes (each
mode repeated by its occupancy) and whose columns are the input modes. By
unitarity these probabilities sum to 1 over all output configurations -- an
*independent* end-to-end correctness check (Layer 4) that needs no external
library. The permanents are evaluated in one batched call.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations_with_replacement
from typing import Iterable, Sequence

import numpy as np

import gbskernels

__all__ = ["output_configurations", "probabilities", "total_probability"]


def output_configurations(m: int, n: int) -> list[tuple[int, ...]]:
    """All output occupancy patterns: multisets of ``n`` modes drawn from ``m``."""
    return list(combinations_with_replacement(range(m), n))


def _occupancy_factor(modes: Sequence[int]) -> float:
    """``prod_k c_k!`` over mode multiplicities -- the bosonic normalization."""
    f = 1.0
    for c in Counter(modes).values():
        f *= math.factorial(c)
    return f


def _submatrix(U: np.ndarray, out_modes: Sequence[int], in_modes: Sequence[int]) -> np.ndarray:
    rows = np.asarray(out_modes, dtype=int)
    cols = np.asarray(in_modes, dtype=int)
    return U[np.ix_(rows, cols)]


def probabilities(
    U: np.ndarray,
    in_modes: Iterable[int] | None = None,
    precision: str = "fp64",
    backend: str = "cpu",
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Return ``(configs, probs)`` for every output configuration.

    ``in_modes`` defaults to ``(0, 1, ..., n-1)`` single photons; ``n`` is taken
    from its length. All submatrix permanents are computed in one batched call;
    ``backend="gpu"`` runs them through a :class:`gbskernels.Workspace`.
    """
    U = np.asarray(U, dtype=np.complex128)
    m = U.shape[0]
    in_modes = tuple(range(_default_n(U))) if in_modes is None else tuple(in_modes)
    n = len(in_modes)
    in_factor = _occupancy_factor(in_modes)

    configs = output_configurations(m, n)
    mats = [_submatrix(U, t, in_modes) for t in configs]
    if backend == "gpu":
        with gbskernels.Workspace(backend="gpu") as ws:
            perms = ws.perm_batched(mats, precision=precision)
    else:
        perms = gbskernels.perm_batched(mats, precision=precision, backend="cpu")

    probs = np.empty(len(configs), dtype=np.float64)
    for i, t in enumerate(configs):
        probs[i] = (abs(perms[i]) ** 2) / (_occupancy_factor(t) * in_factor)
    return configs, probs


def total_probability(U: np.ndarray, in_modes: Iterable[int] | None = None,
                      backend: str = "cpu") -> float:
    """Sum of output probabilities; equals 1 by unitarity (the Layer-4 check)."""
    _, probs = probabilities(U, in_modes, backend=backend)
    return float(probs.sum())


def _default_n(U: np.ndarray) -> int:
    # A sensible default photon number for a quick demo: half the modes.
    return max(1, U.shape[0] // 2)
