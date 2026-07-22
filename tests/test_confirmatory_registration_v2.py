from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "examples" / "jiuzhang"
sys.path.insert(0, str(HERE))

import prepare_registration_v2 as prepare  # noqa: E402
from confirmatory_common import analysis_source_hash, write_json_exclusive  # noqa: E402
from confirmatory_contract import (ContractError, sha256_json,
                                   validate_registration,
                                   write_canonical_json)  # noqa: E402


def _plan() -> dict:
    return {
        "analysis_commit": "deadbeef",
        "randomness_beacon": {
            "provider": "test", "source": "test-beacon", "round": 17,
            "availability_utc": "2026-08-01T00:00:00Z",
        },
        "selection": {
            "bands": [27], "targets": {"27": 2}, "reserves": {"27": 1},
            "n_strata": 2, "n_records": 100, "record_bytes": 16,
            "source_raw_sha256": "a" * 64, "exclusion_sha256": "b" * 64,
            "exclusion_ledger_sha256": "c" * 64,
            "population_audit_sha256": "d" * 64,
        },
        "models": {
            "exp_id": 0, "parameterization": "classical_excess",
            "classical_boundary": 0.0, "reference_model": "classical",
            "alternative_model": "squeezed",
            "coherence_points": {"classical": 0.0, "middle": 0.25, "squeezed": 1.0},
        },
        "analysis": {
            "estimands": ["best_predictive_anomalous_coherence_grid_point"],
            "primary_estimand": "best_predictive_anomalous_coherence_grid_point",
            "design_report_sha256": "f" * 64,
            "minimum_relevant_coherence": 0.25,
            "target_power": 0.9,
            "type_i_error_max": 0.05,
            "monte_carlo_confidence": 0.99,
            "primary_decision_rule": {
                "method": "simultaneous_paired_model_score_max_t",
                "claim_if_confidence_set_above_classical_boundary": True,
                "require_predictive_pass_for_all_confidence_set_models": True,
                "report_failure_without_suppressing_analysis": True,
            },
            "band_weights": {"27": 1.0}, "bootstrap_reps": 100,
            "bootstrap_seed": 3, "alpha": 0.05,
            "refusal_analysis": {
                "method": "within_fixed_band_stratum_max_abs_mean_score_difference_permutation",
                "reps": 100, "seed": 5, "alpha": 0.05, "inferential_gate": False,
                "recovery_method": "independent_high_precision_interval_reevaluation",
                "minimum_precision_bits": 256, "recovery_source_sha256": "1" * 64,
                "recovery_container_digest": "recovery@sha256:" + "2" * 64,
            },
            "resampling_unit": "event_within_fixed_common_stratum",
            "resampling_method": "calibration_draw_conditioned_fixed_strata_srswor_fpc_max_t",
            "population_scope": "finite_registered_acquisition",
            "normalizer_replicates": {
                "count": 2, "samples_per_replicate": 1, "seed": 4},
            "predictive_checks": {
                "model_pass_policy": "all_plausible_best",
                "detector_pairs": [[0, 1]],
                "thresholds": {"click_count_tv_max": 0.1,
                               "marginal_rms_max": 0.1,
                               "pair_covariance_rms_max": 0.1},
            },
        },
        "numerical_contract": {"scope": "test", "state_fingerprints": {},
                               "analysis_source_sha256": "0" * 64},
        "prior_data_use": {
            "record_level_policy":
                "exclude_all_materialized_published_selected_or_scored_records",
            "outcome_blind_mechanical_scans_are_not_record_level_exposure": True,
            "full_acquisition_aggregates_previously_examined": True,
            "independent_acquisition": False,
            "claim_scope":
                "finite_registered_acquisition_conditional_on_prior_aggregate_use",
            "author_attestation_required": True,
        },
        "external_requirements": {
            "public_registration_required": True,
            "future_beacon_required": True,
            "joint_normalizer_replicates_required": True,
            "container_digest_required": True,
            "reconstruction_required": False,
            "absolute_predictive_checks_required": True,
        },
    }


def _resolve(plan: dict) -> dict:
    return prepare.resolve_registration(
        plan, public_url="https://example.invalid/frozen-plan",
        public_sha256=sha256_json(plan), published_at_utc="2026-07-20T00:00:00Z",
        timestamp_proof_url="https://example.invalid/archive/proof",
        timestamp_proof_sha256="e" * 64,
        beacon_source="test-beacon", beacon_round=17, beacon_value="value-17",
        beacon_proof_url="https://example.invalid/beacon/17",
        beacon_proof_sha256="c" * 64,
    )


def test_resolved_registration_binds_exact_plan_and_future_round():
    reg = _resolve(_plan())
    checked = validate_registration(reg)
    assert checked["public"]["sha256"] == sha256_json(_plan())
    assert checked["beacon"]["proof"]["record_sha256"] == "c" * 64


def test_resolve_rejects_cherry_picked_hash_or_beacon():
    plan = _plan()
    with pytest.raises(ValueError, match="exact canonical plan"):
        prepare.resolve_registration(
            plan, public_url="https://example.invalid/frozen-plan",
            public_sha256="d" * 64, published_at_utc="2026-07-20T00:00:00Z",
            timestamp_proof_url="https://example.invalid/archive/proof",
            timestamp_proof_sha256="e" * 64,
            beacon_source="test-beacon", beacon_round=17, beacon_value="value",
            beacon_proof_url="https://example.invalid/beacon/17",
            beacon_proof_sha256="c" * 64)
    with pytest.raises(ValueError, match="source/round"):
        prepare.resolve_registration(
            plan, public_url="https://example.invalid/frozen-plan",
            public_sha256=sha256_json(plan), published_at_utc="2026-07-20T00:00:00Z",
            timestamp_proof_url="https://example.invalid/archive/proof",
            timestamp_proof_sha256="e" * 64,
            beacon_source="test-beacon", beacon_round=18, beacon_value="value",
            beacon_proof_url="https://example.invalid/beacon/18",
            beacon_proof_sha256="c" * 64)


def test_external_beacon_record_is_mandatory():
    reg = _resolve(_plan())
    del reg["beacon"]["proof"]["url"]
    with pytest.raises(ContractError, match="proof URL"):
        validate_registration(reg)


def test_external_timestamp_record_is_mandatory():
    reg = _resolve(_plan())
    del reg["public"]["timestamp_proof_url"]
    with pytest.raises(ContractError, match="timestamp_proof_url"):
        validate_registration(reg)


def test_prepare_plan_refuses_unresolved_placeholders(monkeypatch):
    template = _plan()
    template["selection"]["source_raw_sha256"] = "REPLACE-WITH-HASH"
    monkeypatch.setattr(prepare, "current_commit", lambda: "deadbeef")
    monkeypatch.setattr(prepare, "analysis_sources_clean", lambda: True)
    monkeypatch.setattr(prepare, "build_readiness", lambda value, **kwargs: {
        "ready_for_public_timestamp": True,
        "candidate_plan": json.loads(json.dumps(value)),
        "blockers": [],
    })
    with pytest.raises(ValueError, match="unresolved registration placeholders"):
        prepare.prepare_plan(json.loads(json.dumps(template)))


def test_analysis_source_hash_is_content_addressed():
    digest = analysis_source_hash()
    assert len(digest) == 64 and int(digest, 16) >= 0


def test_prepare_plan_fills_state_and_source_fingerprints(monkeypatch):
    template = _plan()
    template["analysis_commit"] = None
    template["randomness_beacon"]["availability_utc"] = "2099-01-01T00:00:00Z"
    monkeypatch.setattr(prepare, "current_commit", lambda: "deadbeef")
    monkeypatch.setattr(prepare, "analysis_sources_clean", lambda: True)
    monkeypatch.setattr(prepare, "analysis_source_hash", lambda: "9" * 64)
    monkeypatch.setattr(prepare, "build_readiness", lambda value, **kwargs: {
        "ready_for_public_timestamp": True,
        "candidate_plan": json.loads(json.dumps(value)),
        "blockers": [],
    })
    monkeypatch.setattr(prepare.coherence_family, "jiuzhang_state",
                        lambda value, **kwargs: {"value": value})
    monkeypatch.setattr(prepare, "state_fingerprint",
                        lambda states: {name: state["value"] for name, state in states.items()})
    plan = prepare.prepare_plan(template)
    assert plan["analysis_commit"] == "deadbeef"
    assert plan["numerical_contract"]["analysis_source_sha256"] == "9" * 64
    assert set(plan["numerical_contract"]["state_fingerprints"]) == {
        "classical", "middle", "squeezed"}


def test_scientific_json_writers_are_create_only(tmp_path):
    pretty = tmp_path / "pretty.json"
    canonical = tmp_path / "canonical.json"
    write_json_exclusive(pretty, {"value": 1})
    write_canonical_json(canonical, {"value": 1})
    with pytest.raises(FileExistsError):
        write_json_exclusive(pretty, {"value": 2})
    with pytest.raises(FileExistsError):
        write_canonical_json(canonical, {"value": 2})
    assert json.loads(pretty.read_text())["value"] == 1
    assert canonical.read_text() == '{"value":1}'


def test_prepare_plan_refuses_dirty_analysis_sources(monkeypatch):
    monkeypatch.setattr(prepare, "current_commit", lambda: "deadbeef")
    monkeypatch.setattr(prepare, "analysis_sources_clean", lambda: False)
    with pytest.raises(ValueError, match="tracked and clean"):
        prepare.prepare_plan(_plan())


def test_prepare_plan_reports_unready_bound_inputs(monkeypatch):
    monkeypatch.setattr(prepare, "current_commit", lambda: "deadbeef")
    monkeypatch.setattr(prepare, "analysis_sources_clean", lambda: True)
    monkeypatch.setattr(prepare, "build_readiness", lambda value, **kwargs: {
        "ready_for_public_timestamp": False,
        "candidate_plan": value,
        "blockers": [{"code": "missing_calibration_posterior",
                      "detail": "independent posterior required"}],
    })
    with pytest.raises(ValueError, match="missing_calibration_posterior"):
        prepare.prepare_plan(_plan())


def test_grid_primary_cannot_use_any_unrelated_predictive_model():
    plan = _plan()
    plan["external_requirements"]["absolute_predictive_checks_required"] = True
    plan["analysis"]["predictive_checks"] = {
        "model_pass_policy": "any_registered",
        "detector_pairs": [[0, 1]],
        "thresholds": {"click_count_tv_max": 0.1,
                       "marginal_rms_max": 0.1,
                       "pair_covariance_rms_max": 0.1},
    }
    with pytest.raises(ContractError, match="every plausible best model"):
        _resolve(plan)


def test_v2_registration_cannot_switch_to_an_undesigned_primary():
    plan = _plan()
    plan["analysis"]["primary_estimand"] = "joint_window_log_score_difference"
    plan["analysis"]["estimands"].append("joint_window_log_score_difference")
    with pytest.raises(ContractError, match="coherence-grid primary"):
        _resolve(plan)


def test_registration_binds_type_i_grid_and_predictive_gate():
    plan = _plan()
    plan["analysis"]["alpha"] = 0.1
    with pytest.raises(ContractError, match="equal type_i_error_max"):
        _resolve(plan)

    plan = _plan()
    plan["models"]["coherence_points"]["middle"] = 0.5
    with pytest.raises(ContractError, match="registered model coordinate"):
        _resolve(plan)

    plan = _plan()
    plan["external_requirements"]["absolute_predictive_checks_required"] = False
    with pytest.raises(ContractError, match="absolute_predictive_checks_required"):
        _resolve(plan)


def test_registration_freezes_refusal_analysis_settings():
    plan = _plan()
    del plan["analysis"]["refusal_analysis"]["seed"]
    with pytest.raises(ContractError, match="refusal_analysis"):
        _resolve(plan)

    plan = _plan()
    plan["analysis"]["refusal_analysis"]["alpha"] = 0.1
    with pytest.raises(ContractError, match="alpha must equal"):
        _resolve(plan)

    plan = _plan()
    plan["analysis"]["refusal_analysis"]["inferential_gate"] = True
    with pytest.raises(ContractError, match="refusal_analysis"):
        _resolve(plan)
