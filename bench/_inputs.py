"""Input families for accuracy characterization and stress testing.

Single source of truth shared by the harness and the test suite, so the
cancellation family used to *measure* the FP64 boundary is exactly the one the
tests assert against.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "random_complex",
    "unit_modulus_complex",
    "pm1_matrix",
    "make_cancellation_matrix",
    "cancellation_hafnian",
    "cancellation_loop_hafnian",
    "cancellation_torontonian",
    "physical_permanent",
    "physical_hafnian",
    "physical_loop_hafnian",
    "physical_torontonian",
    "loss_permanent",
    "loss_hafnian",
    "loss_loop_hafnian",
    "loss_torontonian",
    "adversarial_permanent",
    "adversarial_hafnian",
    "adversarial_loop_hafnian",
    "adversarial_torontonian",
    "haar_unitary",
    "bench_batch",
]


def random_complex(n: int, seed: int, scale: float = 1.0) -> np.ndarray:
    g = np.random.default_rng(seed)
    return scale * (g.uniform(-1, 1, (n, n)) + 1j * g.uniform(-1, 1, (n, n)))


def unit_modulus_complex(n: int, seed: int) -> np.ndarray:
    """Entries ``e^{i theta}``, theta ~ U(0, 2pi).

    A naturally cancellation-heavy family: every entry has modulus 1, so the
    Glynn terms grow like ``n^{n/2}`` while the permanent stays near
    ``sqrt(n!)`` -- the summation condition number rises ~exponentially in ``n``.
    """
    g = np.random.default_rng(seed)
    theta = g.uniform(0.0, 2.0 * np.pi, (n, n))
    return np.exp(1j * theta)


def pm1_matrix(n: int, seed: int) -> np.ndarray:
    """+/-1 matrix. Integer-valued permanent -> exact ground truth at small n."""
    g = np.random.default_rng(seed)
    return np.where(g.random((n, n)) < 0.5, -1.0, 1.0)


def make_cancellation_matrix(n: int, delta: float, seed: int) -> np.ndarray:
    """A matrix with a *tunable* Glynn cancellation, kappa ~ O(1/delta).

    Direct-sum of a 2x2 near-singular-permanent block with a well-conditioned
    random remainder. The block ``B = [[2, 1], [-1, 0.5 + delta]]`` has
    ``perm(B) = 2*delta`` while its Glynn terms are O(1) (they cancel ~1.5 down
    to ``2*delta``), so the whole matrix has ``perm = 2*delta * perm(R)`` with a
    condition number that scales as ``1/delta`` -- the knob that drives FP64 off
    the accuracy cliff while the mpmath reference stays exact (docs/DESIGN.md §6/sec.8).
    Requires ``n >= 2``.
    """
    if n < 2:
        raise ValueError("cancellation matrix needs n >= 2")
    B = np.array([[2.0, 1.0], [-1.0, 0.5 + delta]], dtype=np.complex128)
    A = np.zeros((n, n), dtype=np.complex128)
    A[:2, :2] = B
    if n > 2:
        A[2:, 2:] = random_complex(n - 2, seed=seed)
    return A


def cancellation_hafnian(delta: float, seed: int) -> np.ndarray:
    """Symmetric matrix whose hafnian nearly cancels to ``delta`` (8x8).

    A 4x4 block with ``haf = a*f + b*e + c*d = 2 + (delta-2) = delta`` (three O(1)
    matchings cancel) direct-summed with a well-conditioned remainder, so
    ``haf = delta * haf(R)`` is tiny while the power-trace terms are O(1). Same
    family as ``core/check_hafnian_dd.cu``.
    """
    a = f = b = e = c = 1.0
    d = delta - 2.0
    B = np.array([[0, a, b, c], [a, 0, d, e], [b, d, 0, f], [c, e, f, 0]], dtype=np.complex128)
    g = np.random.default_rng(seed)
    G = g.standard_normal((4, 4))
    R = G + G.T
    np.fill_diagonal(R, 0.0)  # ordinary hafnian ignores the diagonal
    A = np.zeros((8, 8), dtype=np.complex128)
    A[:4, :4] = B
    A[4:, 4:] = R
    return A


def cancellation_loop_hafnian(delta: float, seed: int) -> np.ndarray:
    """Symmetric matrix whose loop hafnian nearly cancels to ``delta`` (6x6).

    A 2x2 block ``[[1,2],[2,delta-2]]`` has ``lhaf = h + g*k = 2 + (delta-2) =
    delta``, direct-summed with a well-conditioned remainder (nonzero diagonal =
    loops). Same family as ``core/check_loop_hafnian_dd.cu``.
    """
    B = np.array([[1.0, 2.0], [2.0, delta - 2.0]], dtype=np.complex128)
    g = np.random.default_rng(seed)
    G = g.standard_normal((4, 4))
    R = (G + G.T).astype(np.complex128)  # diagonal kept (loop weights)
    A = np.zeros((6, 6), dtype=np.complex128)
    A[:2, :2] = B
    A[2:, 2:] = R
    return A


def cancellation_torontonian(a: float) -> np.ndarray:
    """Single-mode O = diag(a, a): ``tor = a/(1-a)``, but the kernel computes
    ``1/sqrt(det(I-O)) - 1`` with a catastrophic ``(1+a+...) - 1`` cancellation as
    ``a -> 0``. Same family as ``core/check_torontonian_dd.cu``. Real-domain."""
    return np.array([[a, 0.0], [0.0, a]], dtype=np.complex128)


# --- physical (realistic, well-conditioned) inputs -------------------------
# Representative of how each function is actually used in photonic sampling, and
# well-conditioned (FP64 is accurate here) -- the complement of the adversarial
# cancellation families above.

def physical_permanent(n: int, seed: int) -> np.ndarray:
    """A Haar-random interferometer ``U`` (n x n): ``perm(U)`` is a standard
    boson-sampling amplitude. Well-conditioned (unit-modulus singular values)."""
    return haar_unitary(n, seed)


def physical_hafnian(n: int, seed: int) -> np.ndarray:
    """A Gaussian-boson-sampling kernel ``B = U diag(tanh r) U^T`` (n x n complex
    symmetric, modest squeezing) -- the matrix whose hafnian gives a GBS
    amplitude. ``n`` should be even (haf of odd size is 0)."""
    U = haar_unitary(n, seed)
    g = np.random.default_rng(seed + 99)
    r = g.uniform(0.1, 0.5, n)
    return (U @ np.diag(np.tanh(r)) @ U.T).astype(np.complex128)


def physical_loop_hafnian(n: int, seed: int) -> np.ndarray:
    """A GBS kernel with a displacement-like diagonal (the loop weights) -- the
    loop-hafnian (displaced GBS) input. ``n`` even."""
    B = physical_hafnian(n, seed).copy()
    g = np.random.default_rng(seed + 7)
    np.fill_diagonal(B, g.uniform(-0.3, 0.3, n) + 1j * g.uniform(-0.3, 0.3, n))
    return B


def physical_torontonian(n_modes: int, seed: int, scale: float = 0.1) -> np.ndarray:
    """A small-norm real symmetric ``O`` (2n x 2n) -- a physical threshold-detector
    matrix (``I - O_S`` stays positive definite). Real domain."""
    g = np.random.default_rng(seed)
    M = g.standard_normal((2 * n_modes, 2 * n_modes)) * scale
    return ((M + M.T) / 2).astype(np.complex128)


# --- loss / mixed-state inputs ---------------------------------------------
# A pure squeezed + interferometer state passed through a uniform loss channel
# (transmission eta < 1) becomes MIXED: its Husimi covariance has det(Q) > 1, and the
# matrices the kernels consume (the A-matrix block for the (loop) hafnian, O = I - Q^-1
# for the torontonian) are structurally different from the pure case -- the regime a
# real lossy GBS experiment produces. This is the third input regime alongside
# "physical" (pure, well-conditioned) and "adversarial" (tunable cancellation).

def _qmat(cov: np.ndarray, hbar: float = 2.0) -> np.ndarray:
    """Husimi Q covariance from an xxpp Wigner covariance (matches sampling.gbs._qmat)."""
    N = len(cov) // 2
    x = cov[:N, :N] * 2 / hbar
    xp = cov[:N, N:] * 2 / hbar
    p = cov[N:, N:] * 2 / hbar
    aidaj = (x + p + 1j * (xp - xp.T) - 2 * np.eye(N)) / 4
    aiaj = (x - p + 1j * (xp + xp.T)) / 4
    return np.block([[aidaj, aiaj.conj()], [aiaj, aidaj.conj()]]) + np.identity(2 * N)


def _lossy_cov(modes: int, seed: int, eta: float = 0.6, hbar: float = 2.0) -> np.ndarray:
    """xxpp Wigner covariance of a pure squeezed + interferometer state after a uniform
    loss channel of transmission ``eta`` (eta=1 is the pure state; eta<1 is mixed)."""
    g = np.random.default_rng(seed)
    r = g.uniform(0.3, 0.6, modes)
    z = (g.standard_normal((modes, modes)) + 1j * g.standard_normal((modes, modes))) / np.sqrt(2.0)
    U, rr = np.linalg.qr(z); ph = np.diagonal(rr).copy(); ph /= np.abs(ph); U = U * ph
    sq = np.block([[np.diag(np.exp(-r)), np.zeros((modes, modes))],
                   [np.zeros((modes, modes)), np.diag(np.exp(r))]])
    interf = np.block([[U.real, -U.imag], [U.imag, U.real]])
    S = interf @ sq
    pure = (hbar / 2.0) * S @ S.T
    return eta * pure + (1.0 - eta) * (hbar / 2.0) * np.eye(2 * modes)  # loss -> mixed


def _amat(cov: np.ndarray, hbar: float = 2.0) -> np.ndarray:
    """A-matrix ``X (I - Q^-1)`` of a Gaussian state (the (loop) hafnian's matrix)."""
    m = cov.shape[0] // 2
    Q = _qmat(cov, hbar)
    X = np.block([[np.zeros((m, m)), np.eye(m)], [np.eye(m), np.zeros((m, m))]])
    return X @ (np.eye(2 * m) - np.linalg.inv(Q))


def loss_hafnian(n: int, seed: int, eta: float = 0.6) -> np.ndarray:
    """The A-matrix block (n x n complex symmetric) of a lossy/mixed GBS state -- the
    matrix the hafnian consumes in the mixed regime. ``n`` even."""
    A = _amat(_lossy_cov(n, seed, eta))
    B = A[:n, :n]
    return np.ascontiguousarray(0.5 * (B + B.T)).astype(np.complex128)  # enforce symmetry


def loss_loop_hafnian(n: int, seed: int, eta: float = 0.6) -> np.ndarray:
    """The lossy A-block with displacement-derived loop weights on the diagonal -- the
    loop hafnian's input for a lossy displaced state. ``n`` even."""
    B = loss_hafnian(n, seed, eta).copy()
    g = np.random.default_rng(seed + 3)
    np.fill_diagonal(B, g.uniform(-0.3, 0.3, n) + 1j * g.uniform(-0.3, 0.3, n))
    return B


def loss_torontonian(n_modes: int, seed: int, eta: float = 0.6) -> np.ndarray:
    """``O = I - Q^-1`` (2n x 2n, real) of a lossy/mixed Gaussian state -- the threshold-
    detector matrix a real lossy GBS experiment produces."""
    Q = _qmat(_lossy_cov(n_modes, seed, eta))
    O = np.eye(2 * n_modes) - np.linalg.inv(Q)
    return np.ascontiguousarray(np.real(O)).astype(np.complex128)  # physical real domain


def haar_unitary(m: int, seed: int) -> np.ndarray:
    """Haar-random m x m unitary (QR of a complex Ginibre matrix, phase-fixed)."""
    g = np.random.default_rng(seed)
    z = (g.standard_normal((m, m)) + 1j * g.standard_normal((m, m))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    ph = np.diagonal(r).copy()
    ph /= np.abs(ph)
    return q * ph  # columns scaled by the phases of diag(R)


def loss_permanent(n: int, seed: int) -> np.ndarray:
    """The top-left ``n x n`` block of a Haar-random ``2n x 2n`` unitary -- a *sub-unitary*
    (a contraction, singular values <= 1): the linear-optical map of a LOSSY interferometer,
    the permanent's mixed-state analog (boson sampling with loss). Well-conditioned."""
    U = haar_unitary(2 * n, seed)
    return np.ascontiguousarray(U[:n, :n])


# --- adversarial (tunable cancellation) inputs, generalized to any dim -------
# The canonical near-cancelling block (+) a well-conditioned remainder. perm/haf/tor all
# factorize over a direct sum (perm(A(+)B)=perm A * perm B; same for haf; tor(O1(+)O2)=
# tor O1 * tor O2), so the small block sets the whole value's cancellation while the
# remainder fills out the requested size. A MODERATE strength (kappa ~ 1e6) -- enough to
# show FP64 shedding digits in the achieved-error column, but not so ill-conditioned that
# the GPU and CPU determinant/linear-algebra routines legitimately diverge (which would
# false-trip the public-path checksum gate). The accuracy STUDY (bench.accuracy) sweeps the
# strength all the way to the FP64 cliff; here we want a representative stressed throughput
# input on which GPU and CPU still agree.
_ADV_DELTA = 1e-6
_ADV_TOR_A = 1e-6


def adversarial_permanent(n: int, seed: int) -> np.ndarray:
    """``make_cancellation_matrix`` at the benchmark strength (2x2 block (+) remainder)."""
    return make_cancellation_matrix(n, _ADV_DELTA, seed)


def adversarial_hafnian(n: int, seed: int) -> np.ndarray:
    """4x4 near-cancelling hafnian block (+) a well-conditioned ``(n-4)`` symmetric
    remainder. ``n`` even, ``>= 4``. Generalizes ``cancellation_hafnian`` to any dim."""
    if n < 4 or n % 2:
        raise ValueError("adversarial hafnian needs even n >= 4")
    a = f = b = e = c = 1.0
    d = _ADV_DELTA - 2.0
    A = np.zeros((n, n), dtype=np.complex128)
    A[:4, :4] = np.array([[0, a, b, c], [a, 0, d, e], [b, d, 0, f], [c, e, f, 0]],
                         dtype=np.complex128)
    if n > 4:
        g = np.random.default_rng(seed)
        G = g.standard_normal((n - 4, n - 4))
        R = G + G.T
        np.fill_diagonal(R, 0.0)  # ordinary hafnian ignores the diagonal
        A[4:, 4:] = R
    return A


def adversarial_loop_hafnian(n: int, seed: int) -> np.ndarray:
    """2x2 near-cancelling loop-hafnian block (+) a ``(n-2)`` symmetric remainder (diagonal
    kept = loops). ``n`` even, ``>= 2``. Generalizes ``cancellation_loop_hafnian``."""
    if n < 2 or n % 2:
        raise ValueError("adversarial loop hafnian needs even n >= 2")
    A = np.zeros((n, n), dtype=np.complex128)
    A[:2, :2] = np.array([[1.0, 2.0], [2.0, _ADV_DELTA - 2.0]], dtype=np.complex128)
    if n > 2:
        g = np.random.default_rng(seed)
        G = g.standard_normal((n - 2, n - 2))
        A[2:, 2:] = (G + G.T).astype(np.complex128)
    return A


def adversarial_torontonian(n_modes: int, seed: int) -> np.ndarray:
    """One catastrophic-cancellation mode ``diag(a, a)`` (``tor_1 = 1/sqrt(1-a) - 1`` with a
    ``(1 + a/2 + ...) - 1`` cancellation as ``a -> 0``) (+) ``n_modes-1`` physical small-norm
    modes. Real, ``I - O_S`` positive definite for every S; ``dim = 2*n_modes``."""
    O = np.zeros((2 * n_modes, 2 * n_modes))
    if n_modes > 1:
        P = np.real(physical_torontonian(n_modes - 1, seed))   # 2(n-1) x 2(n-1)
        idx = list(range(1, n_modes)) + list(range(n_modes + 1, 2 * n_modes))
        O[np.ix_(idx, idx)] = P
    O[0, 0] = _ADV_TOR_A
    O[n_modes, n_modes] = _ADV_TOR_A                            # mode 0's a / a-dagger entries
    return np.ascontiguousarray(O).astype(np.complex128)


# --- the single shared benchmark workload generator ------------------------

_BENCH_FAMILIES = {
    "physical": {"perm": lambda d, s: physical_permanent(d, s),
                 "haf": lambda d, s: physical_hafnian(d, s),
                 "lhaf": lambda d, s: physical_loop_hafnian(d, s),
                 "tor": lambda d, s: physical_torontonian(d // 2, s)},
    "loss": {"perm": lambda d, s: loss_permanent(d, s),
             "haf": lambda d, s: loss_hafnian(d, s),
             "lhaf": lambda d, s: loss_loop_hafnian(d, s),
             "tor": lambda d, s: loss_torontonian(d // 2, s)},
    "adversarial": {"perm": lambda d, s: adversarial_permanent(d, s),
                    "haf": lambda d, s: adversarial_hafnian(d, s),
                    "lhaf": lambda d, s: adversarial_loop_hafnian(d, s),
                    "tor": lambda d, s: adversarial_torontonian(d // 2, s)},
}
BENCH_REGIMES = tuple(_BENCH_FAMILIES)


def bench_batch(func: str, dim: int, batch: int, regime: str = "physical",
                seed: int = 0) -> np.ndarray:
    """A uniform ``(batch, dim, dim)`` complex128 stack for ``func`` in ``regime`` -- THE
    single workload generator shared by the GPU, CPU, and The Walrus benchmarks, so a
    throughput comparison is *same-input* (docs/DESIGN.md §9). ``regime`` in
    ``{physical, loss, adversarial}``; for the torontonian ``dim = 2*modes``. Element ``k``
    uses ``seed + k`` so the batch is varied yet reproducible: the SAME
    ``(func, dim, regime, seed)`` yields the SAME matrices for every engine."""
    if regime not in _BENCH_FAMILIES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {BENCH_REGIMES}")
    fam = _BENCH_FAMILIES[regime]
    if func not in fam:
        raise ValueError(f"unknown func {func!r}; expected one of {tuple(fam)}")
    gen = fam[func]
    return np.ascontiguousarray(np.stack([gen(dim, seed + k) for k in range(batch)]))
