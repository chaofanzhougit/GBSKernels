"""Gaussian boson sampling probabilities via the (loop) hafnian — Layer-4 target.

A thin orchestration that exercises the hafnian kernels on the workload photonic
sampling actually produces (docs/DESIGN.md §2.2, sec.2.3): for a zero-displacement pure
Gaussian state prepared by squeezing ``r`` and an interferometer ``U``, the kernel
is the complex-symmetric matrix

    B = U @ diag(tanh r) @ U^T   (m x m)

and the probability of detecting photon pattern ``nbar = (n_1, ..., n_m)`` is

    P(nbar) = |haf(B_nbar)|^2 / (prod_i n_i!  *  prod_i cosh r_i)

where ``B_nbar`` repeats index ``i`` exactly ``n_i`` times. By construction these
probabilities sum to 1 over the full Fock space; truncating at a photon cutoff
recovers ~1 to truncation error -- an **independent** end-to-end check of the
hafnian (Layer 4, docs/DESIGN.md §8) that needs no external library. The whole batch
of pattern hafnians is computed in one :func:`gbskernels.haf_batched` call -- the
one batched call per distribution (docs/DESIGN.md §7).

That is the *number-resolving, zero-displacement* case (ordinary hafnian). The
other two physical regimes are handled here too and validated per-pattern against
The Walrus (docs/DESIGN.md §2.2): :func:`displaced_probabilities` is *displaced* GBS via
the **loop hafnian** (the displacement becomes diagonal loop weights), and
:func:`torontonian_threshold_probabilities` is the threshold click-pattern
distribution via the **torontonian** (computed directly, not by marginalizing the
hafnian PNR distribution as :func:`threshold_probabilities` does).
"""

from __future__ import annotations

from itertools import product
from math import cosh, factorial, prod, tanh
from typing import Iterable

import numpy as np

import gbskernels

__all__ = ["random_gbs_kernel", "fock_patterns", "probabilities", "total_probability",
           "displaced_probabilities", "displaced_total",
           "threshold_probabilities", "threshold_total",
           "torontonian_threshold_probabilities"]


def random_gbs_kernel(
    m: int, seed: int, r_min: float = 0.1, r_max: float = 0.45
) -> tuple[np.ndarray, np.ndarray]:
    """A pure-state GBS kernel ``B = U diag(tanh r) U^T`` with modest squeezing.

    Returns ``(B, r)``. Modest ``r`` keeps the Fock distribution light-tailed so a
    finite cutoff captures ~all the probability mass.
    """
    g = np.random.default_rng(seed)
    r = g.uniform(r_min, r_max, m)
    # Haar-random interferometer
    z = (g.standard_normal((m, m)) + 1j * g.standard_normal((m, m))) / np.sqrt(2.0)
    q, rr = np.linalg.qr(z)
    ph = np.diagonal(rr).copy()
    ph /= np.abs(ph)
    U = q * ph
    B = U @ np.diag(np.tanh(r)) @ U.T
    return B, r


def fock_patterns(m: int, cutoff: int) -> list[tuple[int, ...]]:
    """All photon patterns over ``m`` modes with each occupancy in ``[0, cutoff)``."""
    return list(product(range(cutoff), repeat=m))


def _submatrix(B: np.ndarray, nbar: tuple[int, ...]) -> np.ndarray:
    idx = [i for i, ni in enumerate(nbar) for _ in range(ni)]
    return B[np.ix_(idx, idx)] if idx else np.empty((0, 0), dtype=np.complex128)


def _eval_batched(func: str, mats, precision: str, backend: str) -> np.ndarray:
    """Evaluate a *ragged* list of ``func`` matrices on the chosen backend.

    The sampling workload is exactly the ragged batch the GPU path needs a
    :class:`gbskernels.Workspace` for: ``backend="gpu"`` routes through one, so the
    physics layer exercises real GPU **bucketing** (group by size) and **buffer
    residency**; ``backend="cpu"`` loops the reference. The GPU loop hafnian now
    accepts the odd-size submatrices displacement produces (odd photon patterns) via
    the augmentation identity, so displaced GBS runs on the GPU too. (Within per-kernel
    GPU size caps.)"""
    if backend == "gpu":
        with gbskernels.Workspace(backend="gpu") as ws:
            return getattr(ws, f"{func}_batched")(mats, precision=precision)
    return getattr(gbskernels, f"{func}_batched")(mats, precision=precision, backend="cpu")


def probabilities(
    B: np.ndarray,
    r: np.ndarray,
    cutoff: int,
    precision: str = "fp64",
    backend: str = "cpu",
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """GBS probabilities for every pattern up to ``cutoff`` photons per mode.

    All pattern hafnians are evaluated in a single batched call; ``backend="gpu"``
    runs them through a :class:`gbskernels.Workspace` (ragged bucketing + residency).
    """
    m = B.shape[0]
    norm = prod(cosh(float(x)) for x in r)
    patterns = fock_patterns(m, cutoff)

    # Odd total photon number has zero amplitude (haf of odd size = 0); only the
    # even-parity patterns need a hafnian. Batch those together.
    even = [p for p in patterns if sum(p) % 2 == 0]
    hafs = _eval_batched("haf", [_submatrix(B, p) for p in even], precision, backend)
    haf_of = dict(zip(even, hafs))

    probs = np.empty(len(patterns), dtype=np.float64)
    for k, nbar in enumerate(patterns):
        if sum(nbar) % 2 == 1:
            probs[k] = 0.0
            continue
        denom = prod(factorial(ni) for ni in nbar) * norm
        probs[k] = abs(haf_of[nbar]) ** 2 / denom
    return patterns, probs


def total_probability(B: np.ndarray, r: np.ndarray, cutoff: int, backend: str = "cpu") -> float:
    """Sum of GBS probabilities up to ``cutoff``; -> 1 as the cutoff grows."""
    _, probs = probabilities(B, r, cutoff, backend=backend)
    return float(probs.sum())


def _loop_submatrix(B: np.ndarray, gamma: np.ndarray, nbar: tuple[int, ...]) -> np.ndarray:
    """``B_nbar`` (repeat index ``i`` ``n_i`` times) with the loop weights ``gamma``
    on the diagonal -- the matrix whose loop hafnian is the displaced amplitude."""
    idx = [i for i, ni in enumerate(nbar) for _ in range(ni)]
    if not idx:
        return np.empty((0, 0), dtype=np.complex128)
    M = np.array(B[np.ix_(idx, idx)], dtype=np.complex128)
    np.fill_diagonal(M, np.asarray(gamma)[idx])
    return M


def displaced_probabilities(
    B: np.ndarray,
    r: np.ndarray,
    alpha: np.ndarray,
    cutoff: int,
    precision: str = "fp64",
    backend: str = "cpu",
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """*Displaced* GBS probabilities via the **loop hafnian** (nonzero displacement).

    For a displaced pure Gaussian state, the amplitude of pattern ``nbar`` is the
    loop hafnian of ``B_nbar`` with displacement-derived loop weights on the diagonal
    (Quesada; The Walrus ``pure_state_amplitude``):

        gamma = alpha - B conj(alpha)
        pref  = exp(-1/2 (||alpha||^2 - alpha* B alpha*))
        P(nbar) = |pref * lhaf(B_nbar; diag = gamma_nbar)|^2 / (prod_i n_i! * prod_i cosh r_i)

    where ``alpha`` is the per-mode displacement (the complex Husimi displacement,
    ``thewalrus.quantum.complex_to_real_displacements(mu)[:m]``) and
    ``prod cosh r_i = sqrt(det Q)`` for the pure state. With ``alpha = 0`` this is
    exactly :func:`probabilities` (zero diagonal -> the loop hafnian is the hafnian),
    so the loop-hafnian path is exercised end-to-end on the GBS workload and checked
    against The Walrus per pattern (tests/test_sampling_displaced.py). Unlike the
    zero-displacement case, *odd* photon patterns now carry amplitude.
    """
    B = np.asarray(B, dtype=np.complex128)
    alpha = np.asarray(alpha, dtype=np.complex128)
    m = B.shape[0]
    norm = prod(cosh(float(x)) for x in r)
    # The Husimi A-matrix block for the textbook kernel B = U tanh(r) U^T is its
    # negative (a squeezing-quadrature convention: phase-0 squeezing puts -tanh on
    # the A diagonal). The loop-hafnian formula is in terms of that A-block; with
    # alpha=0 the sign is immaterial (|haf(-M)| = |haf(M)|), so the zero-displacement
    # path is unaffected. (Verified against The Walrus per pattern.)
    A = -B
    gamma = alpha - A @ np.conj(alpha)
    pref = np.exp(-0.5 * (np.vdot(alpha, alpha) - alpha.conj() @ A @ alpha.conj()))

    patterns = fock_patterns(m, cutoff)
    # backend="gpu" routes the ragged loop-hafnian batch through a Workspace; the GPU
    # loop hafnian now accepts the odd-size submatrices displacement produces (odd
    # photon patterns) via the augmentation identity, so displaced GBS runs on the GPU.
    lhafs = _eval_batched("lhaf", [_loop_submatrix(A, gamma, p) for p in patterns],
                          precision, backend)
    probs = np.empty(len(patterns), dtype=np.float64)
    for k, nbar in enumerate(patterns):
        denom = prod(factorial(ni) for ni in nbar) * norm
        probs[k] = float(abs(pref * lhafs[k]) ** 2 / denom)
    return patterns, probs


def displaced_total(B: np.ndarray, r: np.ndarray, alpha: np.ndarray, cutoff: int,
                    backend: str = "cpu") -> float:
    """Sum of displaced GBS probabilities up to ``cutoff``; -> 1 as the cutoff grows."""
    _, probs = displaced_probabilities(B, r, alpha, cutoff, backend=backend)
    return float(probs.sum())


def threshold_probabilities(
    B: np.ndarray, r: np.ndarray, cutoff: int, precision: str = "fp64", backend: str = "cpu"
) -> dict[tuple[int, ...], float]:
    """Threshold (click / no-click) detection statistics from the GBS distribution.

    A threshold detector reports only whether each mode fired (>=1 photon) or not
    (0 photons). The probability of a click pattern ``c in {0,1}^m`` is the sum of
    the photon-number probabilities consistent with it -- so threshold statistics
    follow by marginalizing the (hafnian-derived) number-resolving distribution.
    Like that distribution they sum to 1 (over the truncated space). This
    demonstrates the threshold corner end-to-end; note it derives the threshold
    probabilities from the hafnian PNR distribution rather than evaluating the
    torontonian (the torontonian is validated separately, as a function and
    against The Walrus on physical covariances -- see tests/test_torontonian.py)."""
    patterns, probs = probabilities(B, r, cutoff, precision, backend=backend)
    out: dict[tuple[int, ...], float] = {}
    for nbar, p in zip(patterns, probs):
        click = tuple(1 if n > 0 else 0 for n in nbar)
        out[click] = out.get(click, 0.0) + float(p)
    return out


def threshold_total(B: np.ndarray, r: np.ndarray, cutoff: int, backend: str = "cpu") -> float:
    """Sum of threshold click-pattern probabilities; -> 1 as the cutoff grows."""
    return float(sum(threshold_probabilities(B, r, cutoff, backend=backend).values()))


def _qmat(cov: np.ndarray, hbar: float = 2.0) -> np.ndarray:
    """Husimi Q covariance from an xxpp Wigner covariance (transcribes
    thewalrus.quantum.Qmat; kept here so the sampler needs no runtime dependency)."""
    N = len(cov) // 2
    I = np.identity(N)
    x = cov[:N, :N] * 2 / hbar
    xp = cov[:N, N:] * 2 / hbar
    p = cov[N:, N:] * 2 / hbar
    aidaj = (x + p + 1j * (xp - xp.T) - 2 * I) / 4   # <a_i^dag a_j>
    aiaj = (x - p + 1j * (xp + xp.T)) / 4            # <a_i a_j>
    return np.block([[aidaj, aiaj.conj()], [aiaj, aidaj.conj()]]) + np.identity(2 * N)


def torontonian_threshold_probabilities(
    cov: np.ndarray, hbar: float = 2.0, precision: str = "fp64", backend: str = "cpu"
) -> dict[tuple[int, ...], float]:
    """Threshold click-pattern probabilities of a zero-displacement Gaussian state,
    computed **directly from the torontonian** (not by marginalizing the hafnian
    PNR distribution as :func:`threshold_probabilities` does).

    With ``O = I - Q^{-1}`` (``Q`` the Husimi covariance), the probability of click
    pattern ``c in {0,1}^m`` is ``tor(O_c) / sqrt(det Q)``, where ``O_c`` is ``O``
    reduced to the clicking modes in both the a and a-dagger blocks (Bulmer-Quesada-
    Paesani; The Walrus ``threshold_detection_prob``). Each torontonian is one batched
    :func:`gbskernels.tor_batched` element. Being a genuine distribution it sums to 1
    **exactly** (no photon cutoff). This exercises the torontonian end-to-end on the
    sampling workload; validated against The Walrus and against the marginalized
    hafnian distribution (tests/test_sampling_threshold_tor.py)."""
    m = len(cov) // 2
    Q = _qmat(cov, hbar)
    O = np.eye(2 * m) - np.linalg.inv(Q)
    sqrt_detQ = float(np.sqrt(np.linalg.det(Q)).real)

    patterns = list(product((0, 1), repeat=m))
    mats = []
    for c in patterns:
        S = [i for i in range(m) if c[i]]
        idx = S + [i + m for i in S]  # clicking modes in both blocks (xxpp/a-adag)
        mats.append(np.ascontiguousarray(O[np.ix_(idx, idx)]) if S
                    else np.empty((0, 0), dtype=np.complex128))
    tors = _eval_batched("tor", mats, precision, backend)
    return {c: float(np.real(t) / sqrt_detQ) for c, t in zip(patterns, tors)}
