"""Anomalous-coherence source family for physically meaningful GBS comparison.

For source-pair ``k``, ``n_k`` is the per-mode photon number and ``m_k`` is
``<a_1 a_2>``.  The Gaussian physical boundary is
``|m_k| <= sqrt(n_k (n_k + 1))``; the positive-P classical boundary is
``|m_k| <= n_k``.  ``coherence=0`` is thermal, ``coherence=1`` is the
photon-matched classical boundary, and ``coherence=inf`` is ideal squeezing.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def coherence_bounds(nbar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nbar = np.asarray(nbar, dtype=float)
    if np.any(nbar < 0) or not np.all(np.isfinite(nbar)):
        raise ValueError("nbar must be finite and nonnegative")
    return nbar.copy(), np.sqrt(nbar * (nbar + 1.0))


def anomalous_moment(nbar: np.ndarray, coherence: float | np.ndarray) -> np.ndarray:
    """Map a dimensionless coherence coordinate to ``m`` without leaving physics.

    The coordinate is the fraction of the physical boundary.  Consequently the
    classical region is source-dependent: ``coherence <= n/sqrt(n(n+1))``.
    This avoids treating the squeezed and classical endpoints as the only models.
    """
    nbar = np.asarray(nbar, dtype=float)
    _, physical = coherence_bounds(nbar)
    c = np.asarray(coherence, dtype=float)
    if np.any(~np.isfinite(c)) or np.any(c < 0) or np.any(c > 1):
        raise ValueError("coherence must lie in [0, 1]")
    return c * physical


def excess_moment(nbar: np.ndarray, eta: float | np.ndarray) -> np.ndarray:
    """Interpolate from the classical boundary (``eta=0``) to physics (``1``).

    This is the preferred scalar parameterization for inference: every source
    pair is classical at ``eta <= 0`` and crosses its positive-P boundary as
    ``eta`` becomes positive. Values outside ``[-1, 1]`` are rejected here so
    a registered grid cannot silently leave the physical family.
    """
    nbar = np.asarray(nbar, dtype=float)
    classical, physical = coherence_bounds(nbar)
    eta = np.asarray(eta, dtype=float)
    if np.any(~np.isfinite(eta)) or np.any(eta < -1) or np.any(eta > 1):
        raise ValueError("excess coherence must lie in [-1, 1]")
    return np.where(eta <= 0, (1.0 + eta) * classical,
                    classical + eta * (physical - classical))


def classify_moment(nbar: np.ndarray, moment: np.ndarray,
                    atol: float = 1e-12) -> dict[str, Any]:
    nbar = np.asarray(nbar, dtype=float)
    moment = np.asarray(moment, dtype=float)
    classical, physical = coherence_bounds(nbar)
    mag = np.abs(moment)
    if np.any(mag > physical + atol):
        raise ValueError("anomalous moment lies outside the Gaussian physical region")
    return {
        "physical": True,
        "classical_positive_p": bool(np.all(mag <= classical + atol)),
        "classical_margin": (classical - mag).tolist(),
        "physical_margin": (physical - mag).tolist(),
        "coherence_fraction": np.divide(
            mag, physical, out=np.zeros_like(mag), where=physical > 0).tolist(),
    }


def input_cov_xxpp_from_pair_moments(nbar: np.ndarray,
                                     moment: np.ndarray) -> np.ndarray:
    """50-mode input covariance after pairing, directly from ``n`` and ``m``.

    Each independent source pair has ``<a_i^dag a_i>=n`` and real
    ``<a_1 a_2>=m``.  In hbar=2 xxpp ordering its x block is
    ``[[2n+1, 2m], [2m, 2n+1]]`` and its p block has ``-2m`` off diagonal.
    """
    nbar = np.asarray(nbar, dtype=float)
    moment = np.asarray(moment, dtype=float)
    if nbar.shape != moment.shape or nbar.ndim != 1:
        raise ValueError("nbar and moment must be one-dimensional with equal shape")
    classify_moment(nbar, moment)
    modes = 2 * len(nbar)
    x = np.zeros((modes, modes), dtype=float)
    p = np.zeros_like(x)
    for k, (n, m) in enumerate(zip(nbar, moment)):
        sl = slice(2 * k, 2 * k + 2)
        x[sl, sl] = [[2 * n + 1, 2 * m], [2 * m, 2 * n + 1]]
        p[sl, sl] = [[2 * n + 1, -2 * m], [-2 * m, 2 * n + 1]]
    z = np.zeros_like(x)
    return np.block([[x, z], [z, p]])


def build_covariance(T_out_by_in: np.ndarray, nbar: np.ndarray,
                     moment: np.ndarray) -> np.ndarray:
    """Propagate the coherence-family input through the measured lossy channel."""
    import q7_construction as q7

    return q7.output_cov(np.asarray(T_out_by_in),
                         input_cov_xxpp_from_pair_moments(nbar, moment))


def jiuzhang_state(coherence: float, *, exp_id: int = 0,
                   parameterization: str = "physical_fraction",
                   calibration: dict[str, np.ndarray] | None = None) -> dict:
    """Construct one family member, optionally from a calibration/posterior draw."""
    import q7_construction as q7

    if calibration is None:
        r25, transfer = q7.load_config(exp_id)
    else:
        r25 = np.asarray(calibration["r25"], dtype=float)
        transfer = np.asarray(calibration["T_out_by_in"])
    nbar = np.sinh(r25) ** 2
    if parameterization == "physical_fraction":
        moment = anomalous_moment(nbar, coherence)
    elif parameterization == "classical_excess":
        moment = excess_moment(nbar, coherence)
    else:
        raise ValueError(f"unknown coherence parameterization {parameterization!r}")
    cov = build_covariance(transfer, nbar, moment)
    state = q7.build_state("squeezed", cov=cov)
    state.update({"coherence": float(coherence), "parameterization": parameterization,
                  "exp_id": int(exp_id),
                  "moment_classification": classify_moment(nbar, moment)})
    return state


def endpoint_coordinates(r25: np.ndarray) -> dict[str, np.ndarray]:
    """Coordinates of the thermal, squashed, and ideal-squeezed endpoints."""
    nbar = np.sinh(np.asarray(r25, dtype=float)) ** 2
    _, physical = coherence_bounds(nbar)
    return {"thermal": np.zeros_like(nbar), "classical_boundary": nbar / physical,
            "ideal_squeezed": np.ones_like(nbar)}
