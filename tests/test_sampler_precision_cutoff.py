"""Sampler precision plumbing (P0.2) + cutoff semantics & bias bound (P0.4).

CPU-only and fast (no thewalrus, not slow): the sampler validates precision/backend combos
early (never silently mis-serving a requested tier), and ``return_diagnostics`` quantifies
the per-mode-truncation bias with a per-pair geometric tail bound that genuinely
UPPER-bounds the measured truncation error.
"""

from __future__ import annotations

import numpy as np
import pytest

from sampling import gbs, sampler

pytestmark = pytest.mark.layer5


def _state(m: int, seed: int, r_lo: float = 0.15, r_hi: float = 0.35):
    """A zero-displacement pure GBS state -> (cov, r, B = U tanh(r) U^T)."""
    g = np.random.default_rng(seed)
    r = g.uniform(r_lo, r_hi, m)
    z = (g.standard_normal((m, m)) + 1j * g.standard_normal((m, m))) / np.sqrt(2.0)
    U, rr = np.linalg.qr(z); ph = np.diagonal(rr).copy(); ph /= np.abs(ph); U = U * ph
    S = (np.block([[U.real, -U.imag], [U.imag, U.real]])
         @ np.block([[np.diag(np.exp(-r)), np.zeros((m, m))],
                     [np.zeros((m, m)), np.diag(np.exp(r))]]))
    return S @ S.T, r, U @ np.diag(np.tanh(r)) @ U.T


# --- P0.2: precision is validated early; a requested tier is never silently mis-served ---

def test_cpu_backend_rejects_dd_precision():
    cov, _, _ = _state(3, 1)
    with pytest.raises(ValueError, match="dd"):
        sampler.sample(cov, 10, cutoff=4, backend="cpu", precision="dd")


def test_unknown_precision_rejected():
    cov, _, _ = _state(3, 1)
    with pytest.raises(ValueError, match="unknown precision"):
        sampler.sample(cov, 10, cutoff=4, precision="quad")


def test_cpu_fallback_mapping_never_downgrades_to_fp64():
    # The over-cap CPU fallback maps a requested tier to one AT LEAST as accurate: the CPU
    # has no double-double, so 'dd' -> mpmath 'ref' (more accurate), never silently 'fp64'.
    assert sampler._CPU_FALLBACK == {"fp64": "fp64", "auto": "auto", "dd": "ref", "ref": "ref"}
    # the DD split uses the (smaller) DD kernel cap, the rest the FP64 cap
    assert sampler._gpu_haf_cap("dd") == sampler._HAF_DD_CAP
    assert sampler._gpu_haf_cap("fp64") == sampler._HAF_CAP
    assert sampler._gpu_haf_cap("auto") == sampler._HAF_CAP


# --- P0.4: cutoff semantics are defined and the truncation bias is quantified -----------

def test_diagnostics_do_not_change_samples_and_carry_the_definition():
    cov, _, _ = _state(3, 7)
    s0 = sampler.sample(cov, 3000, cutoff=5, seed=3)
    s1, diag = sampler.sample(cov, 3000, cutoff=5, seed=3, return_diagnostics=True)
    assert np.array_equal(s0, s1), "diagnostics must not perturb the draws"
    assert {"definition", "tv_bias_estimate_mean", "tv_bias_estimate_max",
            "per_mode_discarded_mean", "tail_not_decaying"} <= set(diag)
    assert "per-mode truncation" in diag["definition"]


def test_tail_estimate_falls_with_cutoff():
    cov, _, _ = _state(3, 7)
    est = []
    for c in (4, 6, 8):
        _, d = sampler.sample(cov, 4000, cutoff=c, seed=3, return_diagnostics=True)
        est.append(d["tv_bias_estimate_mean"])
    assert est[0] > est[1] > est[2], f"mean estimate should fall with cutoff: {est}"
    assert est[-1] < 1e-3, f"a generous cutoff -> tiny bias estimate, got {est[-1]:.2e}"
    # Past the photon support every prefix's conditional has peaked, so nothing is flagged
    # as non-decaying (the per-prefix flag at a tight cutoff is honest, not a false alarm:
    # it marks a rare prefix whose mode wants more photons than the cutoff allows).
    _, d_big = sampler.sample(cov, 4000, cutoff=10, seed=3, return_diagnostics=True)
    assert not d_big["tail_not_decaying"]


def test_tail_estimate_covers_the_measured_truncation_bias():
    # The reported estimate (under the geometric-continuation assumption) should COVER the
    # actual truncation bias on a realistic GBS conditional. Measure the bias as TV(sampler-
    # at-cutoff, near-exact distribution) and assert it sits under the estimate plus sampling
    # noise. This is empirical validation of the heuristic, not a proof. 2 modes (cheap exact
    # enumeration), stronger squeezing + a tight cutoff so the bias is non-trivial.
    m, cutoff, N = 2, 3, 16000
    cov, r, B = _state(m, 5, r_lo=0.45, r_hi=0.6)
    samp, diag = sampler.sample(cov, N, cutoff=cutoff, seed=2, return_diagnostics=True)
    assert not diag["tail_not_decaying"]
    assert 1e-4 < diag["tv_bias_estimate_mean"] < 0.5, "bias should be real but finite here"

    pats, probs = gbs.probabilities(B, r, cutoff=11)        # near-exact (untruncated) target
    Z = float(probs.sum())
    exact = {p: float(pr) / Z for p, pr in zip(pats, probs)}
    emp: dict = {}
    for x in samp:
        t = tuple(int(v) for v in x); emp[t] = emp.get(t, 0) + 1
    emp = {k: v / N for k, v in emp.items()}
    tv = 0.5 * sum(abs(emp.get(k, 0.0) - exact.get(k, 0.0)) for k in set(emp) | set(exact))
    noise = 4.0 / np.sqrt(N)
    assert tv <= diag["tv_bias_estimate_mean"] + noise, \
        f"estimate {diag['tv_bias_estimate_mean']:.3e} should cover the bias (TV {tv:.3e}, noise {noise:.3e})"
