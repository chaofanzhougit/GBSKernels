"""Verbatim Q7-1076 state construction for Jiuzhang 1.0 (anchor §6).

Implements the ground-truth (TMSS-structured squeezed) and squashed-state
covariance constructions of Martinez-Cifuentes, Fonseca-Romero & Quesada,
Quantum 7, 1076 (2023) [arXiv:2207.10058v6], Eqs. (1)-(13), verbatim, in the
xxpp quadrature ordering with hbar = 2 (vacuum covariance = identity):

  sources   25 pairs of SMSS with squeezing parameters {-r_k, +r_k}: per
            Eqs. (3)-(4) the x-block diagonal of pair k is
            (e^{+2 r_k}, e^{-2 r_k}) and the p-block is swapped.  The
            squashed replacement e^{2r} -> 1 + 4*nbar, e^{-2r} -> 1 with
            nbar = sinh^2 r (Eq. 12) is applied in that SAME alternating
            basis, which preserves the mean photon number per source.
  pairing   B = direct sum of 25 real 50:50 beamsplitters
            H = [[1, -1], [1, 1]]/sqrt(2) acting identically on the x and
            p blocks (Eqs. 5-7).
  channel   sigma_OUT = (I - V V^T) + V sigma_IN V^T with
            V = [[Re T, -Im T], [Im T, Re T]] and T the (M x K) transfer
            matrix, output modes by input modes (Eqs. 8-9).

The authors' comparison data are archived under Zenodo DOI
10.5281/zenodo.7141021 (record 7194775). In moment language this construction
has cross-<aa> pair structure: per source pair, <a'_1 a'_2> = m with zero
self-<aa>, where m = sinh(r) cosh(r) for the squeezed hypothesis and
m = nbar = sinh(r)^2 for the squashed one.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "jiuzhang1"
ZEN = HERE.parents[1] / "data" / "q7_1076_zenodo"

# Conservative fp64 slogdet slack charged to every log sqrt(det Q).
LSDQ_SLACK = 1e-9

KINDS = ("squeezed", "squashed")


# ------------------------------------------------------------------ loaders
# Q7-1076 / Zenodo experiment IDs (their README table).  Every configuration
# is 25 TMSS sources (50 SMSS modes) into an M-mode lossy interferometer; the
# squeezing-parameter files carry one column per pump power.
CONFIGS = {
    0: {"label": "Jiuzhang 1.0",            "modes": 100, "sq": "sq_par.txt",      "col": None, "T": "t_matrix.mtx",      "nu": 0.786},
    1: {"label": "Jiuzhang 2.0 0.15W w65",  "modes": 144, "sq": "sq_par_w65.txt",  "col": 0,    "T": "t_matrix_w65.mtx",  "nu": 0.044},
    2: {"label": "Jiuzhang 2.0 0.3W w65",   "modes": 144, "sq": "sq_par_w65.txt",  "col": 1,    "T": "t_matrix_w65.mtx",  "nu": 0.093},
    3: {"label": "Jiuzhang 2.0 0.6W w65",   "modes": 144, "sq": "sq_par_w65.txt",  "col": 2,    "T": "t_matrix_w65.mtx",  "nu": 0.218},
    4: {"label": "Jiuzhang 2.0 1.0W w65",   "modes": 144, "sq": "sq_par_w65.txt",  "col": 3,    "T": "t_matrix_w65.mtx",  "nu": 0.442},
    5: {"label": "Jiuzhang 2.0 1.65W w65",  "modes": 144, "sq": "sq_par_w65.txt",  "col": 4,    "T": "t_matrix_w65.mtx",  "nu": 0.975},
    6: {"label": "Jiuzhang 2.0 1.412W w125", "modes": 144, "sq": "sq_par_w125.txt", "col": 0,    "T": "t_matrix_w125.mtx", "nu": 0.161},
    7: {"label": "Jiuzhang 2.0 0.5W w125",  "modes": 144, "sq": "sq_par_w125.txt", "col": 1,    "T": "t_matrix_w125.mtx", "nu": 0.055},
}


def load_r25() -> np.ndarray:
    """The 25 Jiuzhang 1.0 squeezing parameters (USTC release)."""
    return np.loadtxt(DATA / "squeezing parameters.txt")


def load_T_out_by_in() -> np.ndarray:
    """Transfer matrix as (M output modes) x (K input modes) = 100 x 50.

    The USTC copy stores T_full.npy as (K x M); Q7-1076's Eqs. (8)-(9) use the
    (M x K) orientation, so transpose without conjugation.
    """
    return np.load(DATA / "T_full.npy").T.copy()


def load_config(exp_id: int) -> tuple[np.ndarray, np.ndarray]:
    """(r25, T_out_by_in) for a Q7-1076 experiment ID, from the Zenodo files.

    Exp 0 (Jiuzhang 1.0) prefers the USTC copies in data/jiuzhang1; the 2.0
    configs read the Zenodo squeezing table (one column per pump power) and
    the per-waist transfer matrix.
    """
    cfg = CONFIGS[exp_id]
    if exp_id == 0 and (DATA / "T_full.npy").exists():
        return load_r25(), load_T_out_by_in()
    from scipy.io import mmread

    sq = np.loadtxt(ZEN / "sq_parameters" / cfg["sq"])
    r25 = sq if sq.ndim == 1 else sq[:, cfg["col"]]
    T = np.asarray(mmread(str(ZEN / "transfer_matrices" / cfg["T"])))
    if T.shape == (2 * len(r25), cfg["modes"]):
        # Their archive stores t_matrix_w125.mtx transposed relative to the
        # other two files; use a plain transpose without conjugation.
        T = T.T
    assert T.shape == (cfg["modes"], 2 * len(r25))
    return r25, np.ascontiguousarray(T)


# ------------------------------------------------------- Eqs. (3)-(7), (12)
def input_cov_xxpp(r25: np.ndarray, kind: str, sign_order: str = "paper") -> np.ndarray:
    """Input covariance of the 50 source modes before pairing, Eq. (4)/(12).

    sign_order="paper" is the published {-r_k, +r_k} alternation; "swapped"
    is the {+r_k, -r_k} variant retained for explicit construction audits.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    K = 2 * len(r25)
    dx, dp = np.empty(K), np.empty(K)
    for k, r in enumerate(r25):
        if kind == "squeezed":
            a, b = np.exp(2 * r), np.exp(-2 * r)
        else:  # squashed: e^{2r} -> 1 + 4 nbar, e^{-2r} -> 1  (Eq. 12)
            a, b = 1 + 4 * np.sinh(r) ** 2, 1.0
        if sign_order == "swapped":
            a, b = b, a
        dx[2 * k], dx[2 * k + 1] = a, b
        dp[2 * k], dp[2 * k + 1] = b, a
    return np.diag(np.concatenate([dx, dp]))


def pair_beamsplitter(K: int) -> np.ndarray:
    """B of Eqs. (5)-(6): 25 real 50:50 beamsplitters on consecutive pairs,
    acting identically on the x and p blocks (2K x 2K, xxpp)."""
    H = np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2.0)
    B2 = np.zeros((K, K))
    for k in range(K // 2):
        B2[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] = H
    Z = np.zeros((K, K))
    return np.block([[B2, Z], [Z, B2]])


# ------------------------------------------------------------ Eqs. (8)-(9)
def output_cov(T_out_by_in: np.ndarray, sigma_in: np.ndarray) -> np.ndarray:
    """Lossy-interferometer channel sigma_OUT = (I - V V^T) + V sigma_IN V^T."""
    M = T_out_by_in.shape[0]
    V = np.block(
        [[T_out_by_in.real, -T_out_by_in.imag], [T_out_by_in.imag, T_out_by_in.real]]
    )
    return (np.eye(2 * M) - V @ V.T) + V @ sigma_in @ V.T


def build_cov(kind: str, T_out_by_in: np.ndarray | None = None,
              r25: np.ndarray | None = None, sign_order: str = "paper") -> np.ndarray:
    """Output xxpp covariance (hbar=2) of one Q7-1076 hypothesis, Eqs. (3)-(13)."""
    if T_out_by_in is None:
        T_out_by_in = load_T_out_by_in()
    if r25 is None:
        r25 = load_r25()
    K = 2 * len(r25)
    B = pair_beamsplitter(K)
    sigma_in = B @ input_cov_xxpp(r25, kind, sign_order) @ B.T
    return output_cov(T_out_by_in, sigma_in)


# ------------------------------------------------- Husimi / torontonian side
def threshold_O_xxpp(cov: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact real torontonian matrix + log sqrt(det Q); see the library
    implementation :func:`sampling.gbs.threshold_O_xxpp` for the derivation
    (and for why the legacy ``Re(I - Q^{-1})`` cast is wrong)."""
    from sampling import gbs as gbs_mod

    return gbs_mod.threshold_O_xxpp(cov, hbar=2.0)


def build_state(kind: str, cov: np.ndarray | None = None) -> dict:
    """Q, exact real Ox, log sqrt(det Q), and diagnostics for one hypothesis.

    ``O`` is the exact quadrature-basis torontonian matrix from
    :func:`threshold_O_xxpp`.  The legacy complex-basis real cast
    ``Re(I - Q^{-1})`` is also returned (``O_legacy_recast``) for audits and
    regression tests only; never evaluate with it.
    """
    from sampling import gbs as gbs_mod

    if cov is None:
        cov = build_cov(kind)
    M = cov.shape[0] // 2
    O, log_sqrt_det = threshold_O_xxpp(cov)
    Q = gbs_mod._qmat(cov, hbar=2.0)
    O_legacy = np.eye(2 * M) - np.linalg.inv(Q)
    aidaj_tr = 0.25 * float(np.trace(cov).real - 2 * M)  # <n> total, hbar=2
    return {
        "kind": kind,
        "cov": cov,
        "Q": Q,
        "O": O,
        "log_sqrt_detQ": log_sqrt_det,
        "O_legacy_recast": np.ascontiguousarray(O_legacy.real),
        "max_imag_O_legacy": float(np.abs(O_legacy.imag).max()),
        "mean_photons_out": aidaj_tr,
    }


def click_statistics(Q: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Per-detector click probabilities, exact Cbar and sigma(C) (no MC).

    p_d       = 1 - det(Q_{d})^{-1/2}          (2x2 minor: modes {d, d+M})
    P(neither)= det(Q_{d,e})^{-1/2}            (4x4 minor)
    Var(C)    = sum p_d(1-p_d) + 2 sum_{d<e} [P(both) - p_d p_e].
    Directly comparable to Q7-1076 Table 2 (their values are MC estimates).
    """
    M = Q.shape[0] // 2
    idx2 = np.array([[d, d + M] for d in range(M)])
    q2 = Q[idx2[:, :, None], idx2[:, None, :]]
    p_off = 1.0 / np.sqrt(np.abs(np.linalg.det(q2).real))
    p = 1.0 - p_off
    pairs = np.array([(d, e) for d in range(M) for e in range(d + 1, M)])
    idx4 = np.concatenate([pairs, pairs + M], axis=1)
    q4 = Q[idx4[:, :, None], idx4[:, None, :]]
    p_nn = 1.0 / np.sqrt(np.abs(np.linalg.det(q4).real))
    p_both = 1.0 - p_off[pairs[:, 0]] - p_off[pairs[:, 1]] + p_nn
    var = float(np.sum(p * (1 - p)) + 2 * np.sum(p_both - p[pairs[:, 0]] * p[pairs[:, 1]]))
    return p, float(p.sum()), float(np.sqrt(var))
