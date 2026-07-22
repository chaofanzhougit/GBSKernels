from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "examples" / "jiuzhang"
sys.path.insert(0, str(HERE))

from analyze_refusals import (REFUSAL_METHOD, analyze,
                              validate_refusal_analysis)  # noqa: E402
from confirmatory_common import hash_json  # noqa: E402


REGISTRATION_ID = "1" * 64
RUN_SHA256 = "2" * 64
RECOVERY_SHA256 = "3" * 64
MODELS = ["classical", "squeezed", "middle"]
CONFIG = {"method": REFUSAL_METHOD, "reps": 200, "seed": 2, "alpha": 0.05,
          "inferential_gate": False,
          "recovery_method": "independent_high_precision_interval_reevaluation",
          "minimum_precision_bits": 256,
          "recovery_source_sha256": "6" * 64,
          "recovery_container_digest": "recovery@sha256:" + "7" * 64}
PROVENANCE = {
    "analysis_commit": "deadbeef",
    "analysis_source_sha256": "4" * 64,
    "container_digest": "image@sha256:" + "5" * 64,
}


def _model_scores(midpoint: float) -> dict:
    return {
        "classical": {"mid": 0.0, "halfwidth": 0.01},
        "squeezed": {"mid": midpoint, "halfwidth": 0.01},
        "middle": {"mid": midpoint / 2, "halfwidth": 0.01},
    }


def _run():
    contract_body = {
        "schema": "gbskernels.run-contract.v2",
        "registration_id": REGISTRATION_ID,
        **PROVENANCE,
    }
    contract = {**contract_body, "run_id": hash_json(contract_body)}
    return {
        "schema": "gbskernels.verified-run.v2", "complete": True,
        "run_id": contract["run_id"], "contract": contract,
        "rows": [
            {"event_id": "a", "band": 27, "record_index": 1, "stratum": 0,
             "input_sha256": "a", "log_pattern_ratio_mid": 0.1,
             "replacement_for_refusal_event_id": None},
            {"event_id": "b", "band": 27, "record_index": 2, "stratum": 0,
             "input_sha256": "b", "log_pattern_ratio_mid": 0.2,
             "replacement_for_refusal_event_id": None},
        ],
        "refusals": [
            {"event_id": "r", "band": 27, "record_index": 3, "stratum": 0,
             "input_sha256": "r", "selected_primary": True}
        ],
    }


def _recovery(midpoint: float = 0.15) -> dict:
    scores = [{
        "event_id": "r", "band": 27, "record_index": 3, "stratum": 0,
        "input_sha256": "r", "selected_primary": True,
        "log_pattern_ratio_mid": midpoint, "log_pattern_ratio_halfwidth": 0.02,
        "model_log_probability_proxy": _model_scores(midpoint),
    }]
    run = _run()
    body = {
        "schema": "gbskernels.independent-refusal-recovery.v1",
        "run_id": run["run_id"], "registration_id": REGISTRATION_ID,
        "verified_run_sha256": RUN_SHA256,
        "method": CONFIG["recovery_method"], "precision_bits": 256,
        "independent_implementation": True,
        "recovery_source_sha256": CONFIG["recovery_source_sha256"],
        "recovery_container_digest": CONFIG["recovery_container_digest"],
        "scores": scores,
    }
    return {**body, "recovery_payload_sha256": hash_json(body)}


def _rehash_recovery(recovery: dict) -> None:
    body = {key: value for key, value in recovery.items()
            if key != "recovery_payload_sha256"}
    recovery["recovery_payload_sha256"] = hash_json(body)


def _analyze(run=None, recovered=None):
    return analyze(
        _run() if run is None else run,
        _recovery() if recovered is None else recovered,
        registration_id=REGISTRATION_ID, config=CONFIG, provenance=PROVENANCE,
        verified_run_sha256=RUN_SHA256, recovered_input_sha256=RECOVERY_SHA256,
        model_names=MODELS)


def test_refusal_recovery_is_exactly_bound():
    run = _run()
    out = _analyze(run)
    assert out["run_id"] == run["run_id"] and out["n_refused"] == 1
    assert out["registration_id"] == REGISTRATION_ID
    assert out["method"] == REFUSAL_METHOD and out["permutation_reps"] == 200
    assert out["pass"] is True and out["permutation_diagnostic_only"] is True
    assert out["inputs"] == {
        "verified_run_sha256": RUN_SHA256,
        "recovered_input_sha256": RECOVERY_SHA256,
        "recovery_source_sha256": CONFIG["recovery_source_sha256"],
    }
    assert 0 < out["permutation_pvalue"] <= 1
    assert out["recovered_scores"][0]["event_id"] == "r"
    assert validate_refusal_analysis(
        out, run=run, registration_id=REGISTRATION_ID, config=CONFIG,
        verified_run_sha256=RUN_SHA256, model_names=MODELS) == out


def test_missing_or_misbound_recovery_fails():
    with pytest.raises(ValueError, match="exactly"):
        empty = _recovery()
        empty["scores"] = []
        _rehash_recovery(empty)
        _analyze(recovered=empty)
    bad = _recovery()
    bad["scores"][0]["record_index"] = 99
    _rehash_recovery(bad)
    with pytest.raises(ValueError, match="record_index"):
        _analyze(recovered=bad)

    missing_width = _recovery()
    del missing_width["scores"][0]["log_pattern_ratio_halfwidth"]
    _rehash_recovery(missing_width)
    with pytest.raises(ValueError, match="half-width"):
        _analyze(recovered=missing_width)

    missing_grid = _recovery()
    del missing_grid["scores"][0]["model_log_probability_proxy"]
    _rehash_recovery(missing_grid)
    with pytest.raises(ValueError, match="model-score family"):
        _analyze(recovered=missing_grid)

    inconsistent = _recovery()
    inconsistent["scores"][0]["model_log_probability_proxy"]["squeezed"]["mid"] = 9.0
    _rehash_recovery(inconsistent)
    with pytest.raises(ValueError, match="enclosures are inconsistent"):
        _analyze(recovered=inconsistent)


@pytest.mark.parametrize("field,value,match", [
    ("precision_bits", 64, "precision"),
    ("verified_run_sha256", "8" * 64, "identity or provenance"),
    ("recovery_source_sha256", "8" * 64, "identity or provenance"),
    ("recovery_container_digest", "other@sha256:" + "8" * 64,
     "identity or provenance"),
])
def test_recovery_input_provenance_is_frozen(field, value, match):
    recovery = _recovery()
    recovery[field] = value
    _rehash_recovery(recovery)
    with pytest.raises(ValueError, match=match):
        _analyze(recovered=recovery)


def test_refusal_test_preserves_fixed_band_strata_and_excludes_replacements():
    run = _run()
    run["rows"].append({
        "event_id": "replacement", "band": 27, "record_index": 4, "stratum": 0,
        "input_sha256": "x", "log_pattern_ratio_mid": 100.0,
        "replacement_for_refusal_event_id": "r",
    })
    out = _analyze(run, _recovery(midpoint=1.0))
    assert out["observed_max_stratum_mean_difference"] == pytest.approx(0.85)

    second = {
        "event_id": "r2", "band": 28, "record_index": 5, "stratum": 0,
        "input_sha256": "r2", "selected_primary": True,
    }
    run["refusals"].append(second)
    run["rows"].append({
        "event_id": "c", "band": 28, "record_index": 6, "stratum": 0,
        "input_sha256": "c", "log_pattern_ratio_mid": 0.0,
        "replacement_for_refusal_event_id": None,
    })
    recovered = _recovery(midpoint=10.0)
    recovered["scores"].append({
        "event_id": "r2", "band": 28, "record_index": 5, "stratum": 0,
        "input_sha256": "r2", "selected_primary": True,
        "log_pattern_ratio_mid": -10.0, "log_pattern_ratio_halfwidth": 0.02,
        "model_log_probability_proxy": _model_scores(-10.0),
    })
    _rehash_recovery(recovered)
    out = _analyze(run, recovered)
    assert out["observed_max_stratum_mean_difference"] == pytest.approx(10.0)


@pytest.mark.parametrize("mutation", ["provenance", "config", "run_hash", "statistics"])
def test_refusal_validator_rejects_forged_artifacts(mutation):
    run = _run()
    artifact = _analyze(run)
    expected_hash = RUN_SHA256
    if mutation == "provenance":
        artifact["container_digest"] = "image@sha256:" + "6" * 64
    elif mutation == "config":
        artifact["seed"] += 1
    elif mutation == "run_hash":
        expected_hash = "7" * 64
    else:
        artifact["permutation_pvalue"] = 1.0 - artifact["permutation_pvalue"]
    with pytest.raises(ValueError):
        validate_refusal_analysis(
            artifact, run=run, registration_id=REGISTRATION_ID, config=CONFIG,
            verified_run_sha256=expected_hash, model_names=MODELS)


def test_refusal_validator_rejects_rehashed_recovered_score_forgery():
    run = _run()
    artifact = _analyze(run)
    forged = copy.deepcopy(artifact)
    forged["recovered_scores"][0]["log_pattern_ratio_mid"] = 99.0
    forged["recovered_scores"][0]["model_log_probability_proxy"][
        "squeezed"]["mid"] = 99.0
    forged["recovered_scores_sha256"] = hash_json(forged["recovered_scores"])
    with pytest.raises(ValueError, match="recovered-score payload"):
        validate_refusal_analysis(
            forged, run=run, registration_id=REGISTRATION_ID, config=CONFIG,
            verified_run_sha256=RUN_SHA256, model_names=MODELS)


def test_low_diagnostic_pvalue_does_not_gate_complete_recovery():
    run = _run()
    config = {**CONFIG, "alpha": 0.99}
    artifact = analyze(
        run, _recovery(midpoint=10.0), registration_id=REGISTRATION_ID,
        config=config, provenance=PROVENANCE, verified_run_sha256=RUN_SHA256,
        recovered_input_sha256=RECOVERY_SHA256, model_names=MODELS)
    assert artifact["permutation_pvalue"] < config["alpha"]
    assert artifact["pass"] is True
    validate_refusal_analysis(
        artifact, run=run, registration_id=REGISTRATION_ID, config=config,
        verified_run_sha256=RUN_SHA256, model_names=MODELS)


def test_complete_recovery_passes_when_diagnostic_cell_is_unavailable():
    run = _run()
    run["refusals"][0]["selected_primary"] = False
    recovery = _recovery()
    recovery["scores"][0]["selected_primary"] = False
    del recovery["scores"][0]["model_log_probability_proxy"]
    _rehash_recovery(recovery)
    artifact = _analyze(run, recovery)
    assert artifact["pass"] is True
    assert artifact["permutation_diagnostic_available"] is False
    assert artifact["permutation_pvalue"] is None
    validate_refusal_analysis(
        artifact, run=run, registration_id=REGISTRATION_ID, config=CONFIG,
        verified_run_sha256=RUN_SHA256, model_names=MODELS)
