"""Calibration of precision="auto" -- kappa is a *measured* heuristic, not an assumed one.

On physical / loss / adversarial ensembles the kappa indicator must predict the FP64 error
(high log-log correlation), and the 1e8 trust threshold must deliver its promise: every case
``auto`` trusts to FP64 is accurate (no false trust), and the well-conditioned regimes sit far
below threshold. Calibrated on BOTH the CPU reference path and -- when the extension is
importable -- the **GPU auto path** (whose kappa comes from the on-device ``*_kappa`` kernels),
since that is the path the public claim is used for (docs/DESIGN.md §6).
"""

from __future__ import annotations

import pytest

import gbskernels
from bench import calibrate_auto


def _assert_well_calibrated(art):
    assert art["kind"] == "auto_calibration"
    assert "HEURISTIC" in art["indicator"], "kappa must be documented as a heuristic, not a certificate"
    assert art["backend"] in ("gpu", "cpu") and "gpu_backend" in art
    s = art["summary"]
    assert s["false_trust_count"] == 0, "auto trusted FP64 on a case where it was wrong"
    # worst trusted error is bounded by ~ kappa_threshold * eps (1e8 * 2.2e-16 ~ 2.2e-8)
    assert s["max_rel_err_when_trusted"] < 1e-6
    assert s["log_kappa_vs_log_relerr_corr"] > 0.8, "kappa should strongly predict the FP64 error"
    for regime in ("physical", "loss"):
        r = s["per_regime"][regime]
        assert r["n_trusted"] == r["n"], f"{regime}: a well-conditioned case was flagged risky"
        assert r["max_kappa"] < calibrate_auto.gbskernels._AUTO_KAPPA_MAX
        assert r["max_rel_err_when_trusted"] < 1e-9


def test_auto_calibration_cpu_path(tmp_path):
    art, path = calibrate_auto.run(dps=30, seeds=2, backend="cpu", out_dir=tmp_path)
    assert path.exists() and art["backend"] == "cpu"
    _assert_well_calibrated(art)


def test_auto_calibration_gpu_path(tmp_path):
    # The path the claim is actually used for: kappa from the on-device *_kappa kernels.
    if not gbskernels.gpu_available():
        pytest.skip("GPU extension not importable (needs the host-shim build)")
    art, _ = calibrate_auto.run(dps=30, seeds=2, backend="gpu", out_dir=tmp_path)
    assert art["backend"] == "gpu"
    _assert_well_calibrated(art)
