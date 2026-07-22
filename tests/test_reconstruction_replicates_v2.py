from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "examples" / "jiuzhang"
sys.path.insert(0, str(HERE))

import reconstruction_replicates as reconstruction  # noqa: E402


def _calibration_file(
    path: Path, *, include_drift: bool = True, vary_squeezing: bool = True,
    vary_transfer: bool = True, vary_drift: bool = True,
    detector_response_model: str = "folded_into_transfer_posterior",
    vary_efficiency: bool = True,
) -> None:
    draws, strata, modes, inputs = 2, 2, 2, 2
    r = np.full((draws, 1), 0.2)
    if vary_squeezing:
        r[1, 0] = 0.21
    transfer = np.zeros((draws, modes, inputs))
    transfer[:, 0, 0] = 0.4
    transfer[:, 1, 1] = 0.3
    if vary_transfer:
        transfer[1, 0, 0] = 0.41
    arrays = {
        "r25_draws": r,
        "T_real_draws": transfer,
        "T_imag_draws": np.zeros((draws, modes, inputs)),
    }
    if include_drift:
        drift = np.zeros((draws, strata, modes, inputs))
        drift[0, 1, 0, 0] = 0.001
        if vary_drift:
            drift[1, 1, 0, 0] = 0.002
        else:
            drift[1] = drift[0]
        arrays["block_drift_real_draws"] = drift
        arrays["block_drift_imag_draws"] = np.zeros(
            (draws, strata, modes, inputs))
    if detector_response_model == "explicit_efficiency_draws":
        efficiency = np.full((draws, modes), 0.8)
        if vary_efficiency:
            efficiency[1, 0] = 0.81
        arrays["detector_efficiency_draws"] = efficiency
    source_artifacts = [
        {"url": "https://calibration.example/source.npz", "sha256": "a" * 64},
        {"url": "https://calibration.example/dark-counts.json", "sha256": "b" * 64},
    ]
    if detector_response_model == "folded_into_transfer_posterior":
        response_semantics = {
            "transfer_draws_include_detector_efficiency": True,
            "efficiency_draws": "none",
        }
    else:
        response_semantics = {
            "transfer_draws_include_detector_efficiency": False,
            "efficiency_draws": "absolute_per_output_mode",
        }
    meta = {
        "schema": "gbskernels.calibration-posterior.v1", "bands": [27],
        "nuisance_families": ["squeezing", "transfer", "loss", "block_drift"],
        "source_artifacts": source_artifacts,
        "inference_method": "joint calibration bootstrap",
        "inference_code_sha256": "c" * 64,
        "created_utc": "2026-07-19T00:00:00Z",
        "independent_of_analysis_acquisition": True,
        "detector_response_model": detector_response_model,
        "detector_response_semantics": response_semantics,
        "dark_click_model": "explicitly_zero",
        "dark_click_evidence": {
            "source_url": "https://calibration.example/dark-counts.json",
            "source_sha256": "b" * 64,
            "justification": "No dark clicks in the registered detector window.",
        },
    }
    np.savez(path, meta=json.dumps(meta), **arrays)


def _rewrite_meta(path: Path, update) -> None:
    with np.load(path, allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files if name != "meta"}
        meta = json.loads(str(z["meta"]))
    update(meta)
    np.savez(path, meta=json.dumps(meta), **arrays)


def _rewrite_arrays(path: Path, update) -> None:
    with np.load(path, allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files if name != "meta"}
        meta = str(z["meta"])
    update(arrays)
    np.savez(path, meta=meta, **arrays)


def test_calibration_contract_requires_explicit_drift_and_detector_model(tmp_path):
    good = tmp_path / "good.npz"
    _calibration_file(good)
    out = reconstruction.load_calibration(good, [27], n_strata=2)
    assert out["block_drift"].shape == (2, 2, 2, 2)

    bad = tmp_path / "bad.npz"
    _calibration_file(bad, include_drift=False)
    with pytest.raises(ValueError, match="block drift"):
        reconstruction.load_calibration(bad, [27], n_strata=2)


@pytest.mark.parametrize("field", [
    "source_artifacts", "inference_method", "inference_code_sha256",
    "created_utc", "independent_of_analysis_acquisition", "dark_click_evidence",
])
def test_calibration_contract_requires_auditable_provenance(tmp_path, field):
    path = tmp_path / f"missing-{field}.npz"
    _calibration_file(path)
    _rewrite_meta(path, lambda meta: meta.pop(field))
    with pytest.raises(ValueError, match="calibration|dark|independent"):
        reconstruction.load_calibration(path, [27], n_strata=2)


def test_dark_click_evidence_must_bind_to_a_source_artifact(tmp_path):
    path = tmp_path / "unbound-dark-evidence.npz"
    _calibration_file(path)
    _rewrite_meta(path, lambda meta: meta["dark_click_evidence"].update(
        {"source_sha256": "d" * 64}))
    with pytest.raises(ValueError, match="dark-click evidence"):
        reconstruction.load_calibration(path, [27], n_strata=2)


@pytest.mark.parametrize(("field", "value"), [
    ("source_artifacts", [{"url": "http://calibration.example/source.npz",
                           "sha256": "a" * 64}]),
    ("inference_code_sha256", "not-a-hash"),
    ("created_utc", "2026-07-19T00:00:00"),
    ("independent_of_analysis_acquisition", False),
])
def test_calibration_contract_rejects_invalid_provenance(tmp_path, field, value):
    path = tmp_path / f"invalid-{field}.npz"
    _calibration_file(path)
    _rewrite_meta(path, lambda meta: meta.__setitem__(field, value))
    with pytest.raises(ValueError, match="calibration|UTC|independent"):
        reconstruction.load_calibration(path, [27], n_strata=2)


def test_calibration_contract_requires_multiple_joint_draws(tmp_path):
    path = tmp_path / "single-draw.npz"
    _calibration_file(path)
    _rewrite_arrays(path, lambda arrays: arrays.update(
        {name: values[:1] for name, values in arrays.items()}))
    with pytest.raises(ValueError, match="at least two distinct joint draws"):
        reconstruction.load_calibration(path, [27], n_strata=2)


@pytest.mark.parametrize(("kwargs", "family"), [
    ({"vary_squeezing": False}, "squeezing"),
    ({"vary_transfer": False}, "transfer"),
    ({"vary_drift": False}, "block_drift"),
    ({"detector_response_model": "explicit_efficiency_draws",
      "vary_efficiency": False}, "detector efficiency"),
])
def test_calibration_contract_rejects_pseudo_posterior_draws(
        tmp_path, kwargs, family):
    path = tmp_path / "repeated.npz"
    _calibration_file(path, **kwargs)
    with pytest.raises(ValueError, match=family):
        reconstruction.load_calibration(path, [27], n_strata=2)


def test_phase_only_transfer_variation_does_not_claim_loss_uncertainty(tmp_path):
    path = tmp_path / "phase-only-transfer.npz"
    _calibration_file(path)

    def phase_only(arrays):
        transfer = arrays["T_real_draws"][0] + 1j * arrays["T_imag_draws"][0]
        phases = np.diag(np.exp(1j * np.asarray([0.123, 0.456])))
        transformed = phases @ transfer
        arrays["T_real_draws"][1] = transformed.real
        arrays["T_imag_draws"][1] = transformed.imag

    _rewrite_arrays(path, phase_only)
    with pytest.raises(ValueError, match="loss/throughput"):
        reconstruction.load_calibration(path, [27], n_strata=2)


def test_explicit_detector_efficiency_semantics_and_variation(tmp_path):
    path = tmp_path / "explicit-efficiency.npz"
    _calibration_file(path, detector_response_model="explicit_efficiency_draws")
    out = reconstruction.load_calibration(path, [27], n_strata=2)
    assert out["efficiency"].shape == (2, 2)


def test_generate_propagates_every_model_and_keeps_normalizer_mean_shift(
        tmp_path, monkeypatch):
    plan = {
        "selection": {"bands": [27], "n_strata": 2},
        "models": {
            "exp_id": 0, "parameterization": "classical_excess",
            "reference_model": "reference", "alternative_model": "alternative",
            "coherence_points": {"reference": 0.0, "middle": 0.5,
                                 "alternative": 1.0},
        },
        "analysis": {
            "band_weights": {"27": 1.0},
            "normalizer_replicates": {"samples_per_replicate": 10},
            "calibration_draws": {
                "posterior_sha256": "a" * 64, "count": 2, "seed": 100},
        },
    }
    registration = {"plan": plan, "public": {"plan_sha256": ""}}
    rows = []
    for position, stratum in enumerate((0, 1)):
        rows.append({
            "band": 27, "position": position, "stratum": stratum,
            "manifest_id": "manifest", "event_id": f"event-{position}",
            "record_index": position, "input_sha256": f"input-{position}",
            "inclusion_probability": 0.5,
            "eligible_in_stratum": 10,
            "log_pattern_ratio_mid": 1.0,
            "model_log_probability_proxy": {
                "reference": {"mid": float(position)},
                "middle": {"mid": float(position) + 0.5},
                "alternative": {"mid": float(position) + 1.0},
            },
        })

    def fake_state(coherence, **kwargs):
        calibration = kwargs["calibration"]
        return {"value": float(coherence) + float(calibration["r25"][0])}

    def fake_event(manifest, band, position):
        return {"manifest_id": "manifest", "event_id": f"event-{position}",
                "band": band, "position": position, "record_index": position,
                "stratum": position, "input_sha256": f"input-{position}",
                "pattern": np.zeros(100, dtype=bool), "offset": float(position)}

    def fake_evaluator(states, event):
        proxy = {name: {"mid": state["value"] + event["offset"],
                        "halfwidth": 0.0}
                 for name, state in states.items()}
        names = list(states)
        return {"refused": False, "model_log_probability_proxy": proxy,
                "log_pattern_ratio_mid": proxy[names[1]]["mid"] - proxy[names[0]]["mid"]}

    monkeypatch.setattr(reconstruction.coherence_family, "jiuzhang_state", fake_state)
    monkeypatch.setattr(reconstruction, "_manifest_event", fake_event)
    monkeypatch.setattr(reconstruction, "_default_evaluator", fake_evaluator)

    calibration = {
        "r25": np.asarray([[0.2], [0.3]]),
        "T": np.zeros((2, 2, 2), dtype=np.complex128),
        "block_drift": np.zeros((2, 2, 2, 2), dtype=np.complex128),
    }
    draw_norm = tmp_path / "draw.npz"
    nominal_norm = tmp_path / "nominal.npz"
    draw_meta = json.dumps({"schema": "gbskernels.calibration-normalizer-draws.v1",
                       "bands": [27], "registration_id": "",
                       "n_strata": 2,
                       "calibration_posterior_sha256": "a" * 64,
                       "pairing": "calibration_draw_and_common_stratum",
                       "draws": 2, "samples_per_draw_stratum": 10,
                       "seed": 100,
                       "seed_rule": (
                           "seed + draw_index * n_strata + stratum; common across models"),
                       "calibration_draw_fingerprint_method":
                           reconstruction.CALIBRATION_FINGERPRINT_METHOD,
                       "paired_normalizer_fingerprint_method":
                           reconstruction.PAIRED_NORMALIZER_FINGERPRINT_METHOD,
                       "model_names": ["reference", "alternative", "middle"],
                       "coherence_points": [0.0, 1.0, 0.5], "exp_id": 0,
                       "parameterization": "classical_excess"})
    draw_probabilities = np.full((2, 2, 3, 1), 0.5)
    draw_probabilities[0, :, 0, 0] = 0.8
    draw_probabilities[1, :, 0, 0] = 0.9
    draw_probabilities[:, :, 1, 0] = 0.2
    calibration_fingerprints = reconstruction.calibration_draw_fingerprints(calibration)
    paired_fingerprints = reconstruction.paired_normalizer_draw_fingerprints(
        calibration_fingerprints, draw_probabilities)
    np.savez(
        draw_norm, meta=draw_meta, p_models=draw_probabilities,
        calibration_draw_sha256=np.asarray(calibration_fingerprints),
        paired_normalizer_draw_sha256=np.asarray(paired_fingerprints))
    nominal_meta = json.dumps({"schema": "gbskernels.joint-normalizer-replicates.v1",
                       "bands": [27], "registration_id": "",
                       "model_names": ["reference", "alternative", "middle"],
                       "coherence_points": [0.0, 1.0, 0.5], "exp_id": 0,
                       "parameterization": "classical_excess"})
    np.savez(nominal_norm, meta=nominal_meta, p_reference=np.ones((3, 1)),
             p_alternative=np.ones((3, 1)))

    out = reconstruction.generate(registration, {}, rows, calibration,
                                  draw_norm, nominal_norm)
    assert out["model_log_score_band_draws"].shape == (2, 3, 1)
    assert out["joint_log_score_band_draws"].shape == (2, 1)
    assert out["event_model_log_score_draws"].shape == (2, 2, 3)
    assert np.mean(out["normalizer_log_ratio_band_draws"]) > 0.9

    np.savez(
        draw_norm, meta=draw_meta, p_models=draw_probabilities[::-1],
        calibration_draw_sha256=np.asarray(calibration_fingerprints),
        paired_normalizer_draw_sha256=np.asarray(paired_fingerprints))
    with pytest.raises(ValueError, match="payload is not paired"):
        reconstruction.generate(
            registration, {}, rows, calibration, draw_norm, nominal_norm)

    changed_meta = json.loads(draw_meta)
    changed_meta["seed"] = 101
    np.savez(
        draw_norm, meta=json.dumps(changed_meta), p_models=draw_probabilities,
        calibration_draw_sha256=np.asarray(calibration_fingerprints),
        paired_normalizer_draw_sha256=np.asarray(paired_fingerprints))
    with pytest.raises(ValueError, match="effort or seed"):
        reconstruction.generate(
            registration, {}, rows, calibration, draw_norm, nominal_norm)


def test_bound_manifest_event_rejects_a_different_verified_identity(monkeypatch):
    monkeypatch.setattr(reconstruction, "_manifest_event", lambda *_: {
        "manifest_id": "manifest", "event_id": "wrong", "band": 27,
        "position": 0, "record_index": 1, "stratum": 0, "input_sha256": "input",
    })
    with pytest.raises(ValueError, match="event_id"):
        reconstruction._bound_manifest_event({}, {
            "manifest_id": "manifest", "event_id": "expected", "band": 27,
            "position": 0, "record_index": 1, "stratum": 0, "input_sha256": "input",
        })
