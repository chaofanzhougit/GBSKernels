from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "examples" / "jiuzhang"
sys.path.insert(0, str(HERE))

import confirmatory_release as release_module  # noqa: E402
from build_exclusion_ledger import (ATTESTATION_SCOPE, ATTESTATION_STATEMENT,
                                    build_ledger)  # noqa: E402
from confirmatory_common import sha256_file  # noqa: E402
from confirmatory_contract import sha256_json  # noqa: E402
from confirmatory_design import (DESIGN_SCHEMA, SIMULATION_SCHEMA, build_report,
                                 design_id)  # noqa: E402
from confirmatory_inference import registered_nonclassical_decision  # noqa: E402
from confirmatory_release import (_REQUIRED_RELEASE_ARTIFACTS,
                                  _registered_decision_for_release,
                                  _validate_calibration_normalizer_meta,
                                  _require_analysis_inputs,
                                  _require_run_provenance,
                                  _validate_preregistration_artifacts,
                                  verify_release)  # noqa: E402
from select_confirmatory_v2 import DET_POSITIONS, RECORD_BYTES  # noqa: E402
from reconstruction_replicates import (CALIBRATION_FINGERPRINT_METHOD,
                                       PAIRED_NORMALIZER_FINGERPRINT_METHOD)  # noqa: E402


def _release_body(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    paths = {}
    artifacts = {}
    for role in sorted(_REQUIRED_RELEASE_ARTIFACTS):
        path = tmp_path / f"{role}.json"
        path.write_text("{}\n")
        paths[role] = path
        artifacts[role] = {
            "role": role, "path": path.name,
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
        }
    body = {
        "schema": "gbskernels.confirmatory-release.v2",
        "registration_id": "a" * 64, "manifest_id": "b" * 64,
        "run_id": "c" * 64, "analysis_commit": "deadbeef",
        "container_digest": "image@sha256:" + "d" * 64,
        "registered_primary_estimand": "coherence",
        "registered_decision": {"claim_supported": False},
        "population_scope": "finite", "numerical_scope": "test",
        "artifacts": artifacts,
    }
    return body, paths


def test_release_verifier_detects_artifact_tampering(tmp_path, monkeypatch):
    body, paths = _release_body(tmp_path)
    release = {**body, "release_payload_sha256": sha256_json(body)}
    path = tmp_path / "release.json"
    path.write_text(json.dumps(release))
    monkeypatch.setattr(release_module, "assemble", lambda **kwargs: release)
    assert verify_release(path, root=tmp_path)["run_id"] == "c" * 64
    paths["analysis"].write_text("{\"changed\":true}\n")
    with pytest.raises(ValueError, match="wrong size|hash mismatch"):
        verify_release(path, root=tmp_path)


def test_release_verifier_rejects_path_escape(tmp_path):
    body, _ = _release_body(tmp_path)
    body["artifacts"]["analysis"] = {
        "role": "analysis", "path": "../outside.json",
        "sha256": "0" * 64, "bytes": 1,
    }
    path = tmp_path / "release.json"
    path.write_text(json.dumps({**body, "release_payload_sha256": sha256_json(body)}))
    with pytest.raises(ValueError, match="escapes root"):
        verify_release(path, root=tmp_path)


def test_release_verifier_requires_complete_preregistration_bundle(tmp_path):
    body, _ = _release_body(tmp_path)
    del body["artifacts"]["design_simulation"]
    path = tmp_path / "release.json"
    path.write_text(json.dumps({**body, "release_payload_sha256": sha256_json(body)}))
    with pytest.raises(ValueError, match="missing required artifact roles.*design_simulation"):
        verify_release(path, root=tmp_path)


def test_release_verifier_rejects_rehashed_semantic_forgery(tmp_path, monkeypatch):
    body, _ = _release_body(tmp_path)
    honest = {**body, "release_payload_sha256": sha256_json(body)}
    forged_body = json.loads(json.dumps(body))
    forged_body["registered_decision"]["claim_supported"] = True
    forged = {**forged_body, "release_payload_sha256": sha256_json(forged_body)}
    path = tmp_path / "release.json"
    path.write_text(json.dumps(forged))
    monkeypatch.setattr(release_module, "assemble", lambda **kwargs: honest)
    with pytest.raises(ValueError, match="does not reproduce"):
        verify_release(path, root=tmp_path)


def test_release_verifier_reassembles_all_conditional_roles(tmp_path, monkeypatch):
    body, _ = _release_body(tmp_path)
    for role in ("reconstruction", "calibration", "calibration_normalizers",
                 "predictive_checks", "refusal_analysis", "refusal_recovery",
                 "refusal_recovery_source"):
        artifact = tmp_path / f"{role}.bin"
        artifact.write_bytes(role.encode("ascii"))
        body["artifacts"][role] = {
            "role": role, "path": artifact.name,
            "sha256": sha256_file(artifact), "bytes": artifact.stat().st_size,
        }
    release = {**body, "release_payload_sha256": sha256_json(body)}
    path = tmp_path / "release.json"
    path.write_text(json.dumps(release))
    called = {}

    def fake_assemble(**kwargs):
        called.update(kwargs)
        return release

    monkeypatch.setattr(release_module, "assemble", fake_assemble)
    verify_release(path, root=tmp_path)
    assert called["reconstruction_path"].name == "reconstruction.bin"
    assert called["predictive_checks_path"].name == "predictive_checks.bin"
    assert called["refusal_analysis_path"].name == "refusal_analysis.bin"
    assert called["refusal_recovery_path"].name == "refusal_recovery.bin"
    assert called["refusal_recovery_source_path"].name \
        == "refusal_recovery_source.bin"


def test_release_provenance_rejects_different_valid_container_digest():
    contract = {
        "analysis_commit": "deadbeef", "analysis_source_sha256": "a" * 64,
        "container_digest": "image@sha256:" + "b" * 64,
    }
    artifact = {**contract, "container_digest": "image@sha256:" + "c" * 64}
    with pytest.raises(ValueError, match="provenance differs"):
        _require_run_provenance(artifact, contract, "analysis")


def test_release_binds_calibration_normalizer_effort_seed_and_pairing_methods():
    plan = {"analysis": {
        "calibration_draws": {"count": 2, "seed": 9},
        "normalizer_replicates": {"samples_per_replicate": 1000},
    }}
    meta = {
        "schema": "gbskernels.calibration-normalizer-draws.v1",
        "registration_id": "r", "calibration_posterior_sha256": "c" * 64,
        "draws": 2, "samples_per_draw_stratum": 1000, "seed": 9,
        "seed_rule": "seed + draw_index * n_strata + stratum; common across models",
        "calibration_draw_fingerprint_method": CALIBRATION_FINGERPRINT_METHOD,
        "paired_normalizer_fingerprint_method": PAIRED_NORMALIZER_FINGERPRINT_METHOD,
    }
    _validate_calibration_normalizer_meta(
        meta, registration_id="r", calibration_sha256="c" * 64, plan=plan)
    for field in ("draws", "samples_per_draw_stratum", "seed", "seed_rule",
                  "calibration_draw_fingerprint_method",
                  "paired_normalizer_fingerprint_method"):
        forged = dict(meta)
        forged[field] = "changed"
        with pytest.raises(ValueError, match="binding is invalid"):
            _validate_calibration_normalizer_meta(
                forged, registration_id="r", calibration_sha256="c" * 64,
                plan=plan)


def test_release_rejects_same_run_input_artifact_substitution(tmp_path):
    paths = {}
    for role in ("verified_run", "normalizers", "reconstruction",
                 "predictive_checks", "refusal_analysis"):
        path = tmp_path / role
        path.write_bytes(role.encode("ascii"))
        paths[role] = path
    analysis = {"inputs": {
        "verified_run_sha256": sha256_file(paths["verified_run"]),
        "normalizer_replicates_sha256": sha256_file(paths["normalizers"]),
        "reconstruction_replicates_sha256": sha256_file(paths["reconstruction"]),
        "predictive_checks_sha256": sha256_file(paths["predictive_checks"]),
        "refusal_analysis_sha256": sha256_file(paths["refusal_analysis"]),
    }}
    _require_analysis_inputs(analysis, paths)
    paths["reconstruction"].write_bytes(b"different same-run reconstruction")
    with pytest.raises(ValueError, match="input hashes differ"):
        _require_analysis_inputs(analysis, paths)


def _raw_record(pattern: np.ndarray, timestamp: int) -> bytes:
    bits = np.zeros(RECORD_BYTES * 8, dtype=np.uint8)
    bits[:16] = np.unpackbits(np.asarray(
        [(timestamp >> 8) & 0xff, timestamp & 0xff], dtype=np.uint8))
    bits[DET_POSITIONS[::-1]] = np.asarray(pattern, dtype=np.uint8)
    return np.packbits(bits).tobytes()


def _preregistration_fixture(tmp_path: Path) -> tuple[dict, dict, dict[str, Path]]:
    raw_path = tmp_path / "data" / "raw.bin"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    retained = np.zeros(100, dtype=np.uint8)
    retained[:27] = 1
    unexposed = np.zeros(100, dtype=np.uint8)
    unexposed[27:54] = 1
    raw_path.write_bytes(b"".join(
        [_raw_record(retained, 0)]
        + [_raw_record(unexposed, position) for position in range(1, 200)]
    ))
    evidence_path = tmp_path / "evidence" / "retained.npy"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(evidence_path, retained[None, :])
    aggregate_path = tmp_path / "tools" / "aggregate.py"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text("# outcome-blind aggregate scan\n")
    catalog = {
        "schema": "gbskernels.exploratory-exposure-catalog.v1",
        "raw_source": {
            "path": "data/raw.bin", "sha256": sha256_file(raw_path),
            "record_bytes": 16, "n_records": 200,
        },
        "policy": {
            "record_level_exposure_rule": "exclude retained records",
            "aggregate_only_processing_rule": "disclose aggregate scans",
            "aggregate_scans_do_not_exclude_all_records": True,
        },
        "aggregate_exposures": [{
            "id": "scan", "scope": "synthetic acquisition",
            "operation": "aggregate count only",
            "record_level_outputs_retained": False,
            "artifact": {
                "path": "tools/aggregate.py", "sha256": sha256_file(aggregate_path),
            },
        }],
        "evidence": [{
            "id": "retained", "kind": "pattern_arrays",
            "expected_unique_patterns": 1,
            "artifact": {
                "path": "evidence/retained.npy", "sha256": sha256_file(evidence_path),
                "expected_rows": 1,
            },
        }],
        "author_attestation": {
            "scope": ATTESTATION_SCOPE, "statement": ATTESTATION_STATEMENT,
            "attested": True, "attestor": "Synthetic Author",
            "attested_utc": "2026-07-21T00:00:00Z",
        },
        "unresolved_provenance": [{
            "id": "review", "provenance_recovered": False,
            "exclusion_risk_resolved": True, "resolution": "reviewed",
        }],
    }
    ledger = build_ledger(tmp_path, catalog, require_complete=True)
    audit_body = {
        "schema": "gbskernels.selection-population-audit.v1",
        "source_raw_sha256": catalog["raw_source"]["sha256"],
        "n_records": 200, "record_bytes": 16,
        "exclusion_sha256": ledger["exclusion_sha256"],
        "exclusion_ledger_sha256": ledger["ledger_payload_sha256"],
        "exclusion_ledger_complete": True, "registration_eligible": True,
        "n_excluded": 1, "bands": [27], "n_strata": 2,
        "eligible_by_band_stratum": {"27": [100, 99]},
        "eligible_by_band": {"27": 199},
        "band_weights_within_window": {"27": 1.0},
        "band_weights_exact": {"27": "199/199"},
    }
    audit = {**audit_body, "audit_payload_sha256": sha256_json(audit_body)}
    analysis_contract = {
        "models": {
            "exp_id": 0,
            "parameterization": "classical_excess",
            "classical_boundary": 0.0,
            "reference_model": "classical-boundary",
            "alternative_model": "eta-mre",
            "coherence_points": {
                "classical-boundary": 0.0,
                "sub-mre": 0.1,
                "eta-mre": 0.25,
            },
        },
        "primary_estimand": "best_predictive_anomalous_coherence_grid_point",
        "primary_decision_rule": {
            "method": "simultaneous_paired_model_score_max_t",
            "claim_if_confidence_set_above_classical_boundary": True,
            "require_predictive_pass_for_all_confidence_set_models": True,
            "report_failure_without_suppressing_analysis": True,
        },
        "alpha": 0.05,
        "bootstrap_reps": 100,
        "bootstrap_seed": 7,
        "refusal_analysis": {
            "method": "within_fixed_band_stratum_max_abs_mean_score_difference_permutation",
            "reps": 100, "seed": 10, "alpha": 0.05, "inferential_gate": False,
            "recovery_method": "independent_high_precision_interval_reevaluation",
            "minimum_precision_bits": 256, "recovery_source_sha256": "d" * 64,
            "recovery_container_digest": "recovery@sha256:" + "e" * 64,
        },
        "normalizer_replicates": {
            "count": 2, "samples_per_replicate": 1, "seed": 8,
        },
        "reconstruction_required": True,
        "calibration_draws": {
            "count": 2, "seed": 9, "posterior_sha256": "c" * 64,
            "required_nuisances": ["squeezing", "transfer", "loss", "block_drift"],
            "dark_click_model": "explicitly_zero",
        },
        "predictive_model_pass_policy": "all_plausible_best",
        "predictive_detector_pairs": [[0, 1]],
        "resampling_unit": "event_within_fixed_common_stratum",
        "resampling_method": "calibration_draw_conditioned_fixed_strata_srswor_fpc_max_t",
        "population_scope": "finite_registered_acquisition",
    }
    simulator_source_path = tmp_path / "simulator_source.py"
    simulator_source_path.write_text("# frozen forward simulator\n")
    simulation_bank_path = tmp_path / "simulation_bank.bin"
    simulation_bank_path.write_bytes(b"frozen simulation bank")
    simulator_source_hash = sha256_file(simulator_source_path)
    simulation_bank_hash = sha256_file(simulation_bank_path)
    spec = {
        "schema": DESIGN_SCHEMA, "bands": [27],
        "calibration_posterior_sha256": "c" * 64,
        "simulator_source_sha256": simulator_source_hash,
        "analysis_contract": analysis_contract,
        "event_cost": {"27": 1.0},
        "candidate_targets": [{"name": "candidate", "targets": {"27": 20}}],
        "scenarios": {
            "classical": ["classical-boundary"], "alternative": ["eta-mre"],
            "predictive_valid": ["classical-boundary", "sub-mre", "eta-mre"],
            "predictive_misspecification": ["wrong-loss"],
        },
        "criteria": {
            "type_i_error_max": 0.05, "power_min": 0.8,
            "monte_carlo_confidence": 0.9, "minimum_relevant_coherence": 0.25,
            "predictive_family_false_failure_max": 0.3,
            "predictive_misspecification_rejection_min": 0.8,
            "reserve_family_failure_max": 0.05, "simulation_trials_min": 100,
            "predictive_draws_min": 80, "refusal_trials_min": 100,
        },
    }
    metrics = {
        "click_count_tv": [0.05] * 80,
        "marginal_rms": [0.025] * 80,
        "pair_covariance_rms": [0.0125] * 80,
    }
    simulation = {
        "schema": SIMULATION_SCHEMA, "design_id": design_id(spec, audit),
        "source": {
            "kind": "independent_calibration_forward_simulation",
            "uses_analysis_acquisition_outcomes": False,
            "input_roles": ["calibration_posterior", "forward_model"],
            "calibration_posterior_sha256": "c" * 64,
            "simulator_source_sha256": simulator_source_hash,
            "simulation_bank_sha256": simulation_bank_hash,
        },
        "candidates": [{
            "name": "candidate", "targets": {"27": 20},
            "classical_claims": [{
                "name": "classical-boundary", "successes": 0, "trials": 100}],
            "alternative_successes": [{
                "name": "eta-mre", "successes": 99, "trials": 100}],
            "predictive_valid_draws": [
                {"name": name, **metrics}
                for name in ("classical-boundary", "sub-mre", "eta-mre")
            ],
            "predictive_misspecification_draws": [{
                "name": "wrong-loss",
                **{key: [10 * value for value in values]
                   for key, values in metrics.items()},
            }],
            "refusal_studies": {"27": {"refusals": 0, "trials": 100}},
        }],
    }
    report = build_report(spec, audit, simulation)
    registration = {"plan": {
        "models": analysis_contract["models"],
        "selection": {
            "bands": [27], "targets": report["selected"]["targets"],
            "reserves": report["selected"]["reserves"], "n_records": 200,
            "record_bytes": 16,
            "source_raw_sha256": catalog["raw_source"]["sha256"],
            "exclusion_sha256": ledger["exclusion_sha256"],
            "exclusion_ledger_sha256": ledger["ledger_payload_sha256"],
        },
        "analysis": {
            "design_report_sha256": report["design_report_payload_sha256"],
            "minimum_relevant_coherence": 0.25, "target_power": 0.8,
            "type_i_error_max": 0.05, "monte_carlo_confidence": 0.9,
            "primary_estimand": analysis_contract["primary_estimand"],
            "primary_decision_rule": analysis_contract["primary_decision_rule"],
            "alpha": analysis_contract["alpha"],
            "bootstrap_reps": analysis_contract["bootstrap_reps"],
            "bootstrap_seed": analysis_contract["bootstrap_seed"],
            "refusal_analysis": analysis_contract["refusal_analysis"],
            "normalizer_replicates": analysis_contract["normalizer_replicates"],
            "resampling_unit": analysis_contract["resampling_unit"],
            "resampling_method": analysis_contract["resampling_method"],
            "population_scope": analysis_contract["population_scope"],
            "predictive_checks": {
                "model_pass_policy": analysis_contract["predictive_model_pass_policy"],
                "detector_pairs": analysis_contract["predictive_detector_pairs"],
                "thresholds": report["selected"]["predictive_thresholds"]["thresholds"],
            },
            "calibration_draws": analysis_contract["calibration_draws"],
        },
        "external_requirements": {"reconstruction_required": True},
    }}
    values = {
        "exposure_catalog": catalog, "exclusion_ledger": ledger,
        "design_spec": spec, "design_report": report, "design_simulation": simulation,
    }
    paths = {}
    for role, value in values.items():
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps(value))
        paths[role] = path
    paths["simulator_source"] = simulator_source_path
    paths["simulation_bank"] = simulation_bank_path
    paths["ledger_evidence"] = evidence_path
    return registration, audit, paths


def test_preregistration_release_inputs_reproduce_and_bind_all_artifacts(tmp_path):
    registration, audit, paths = _preregistration_fixture(tmp_path)
    identifiers = _validate_preregistration_artifacts(
        registration, audit,
        exposure_catalog_path=paths["exposure_catalog"],
        exclusion_ledger_path=paths["exclusion_ledger"],
        design_spec_path=paths["design_spec"],
        design_report_path=paths["design_report"],
        design_simulation_path=paths["design_simulation"],
        simulator_source_path=paths["simulator_source"],
        simulation_bank_path=paths["simulation_bank"],
        root=tmp_path,
    )
    assert identifiers["exclusion_ledger_payload_sha256"] \
        == registration["plan"]["selection"]["exclusion_ledger_sha256"]
    assert identifiers["design_report_payload_sha256"] \
        == registration["plan"]["analysis"]["design_report_sha256"]

    simulation = json.loads(paths["design_simulation"].read_text())
    simulation["candidates"][0]["alternative_successes"][0]["successes"] = 98
    paths["design_simulation"].write_text(json.dumps(simulation))
    with pytest.raises(ValueError, match="does not reproduce"):
        _validate_preregistration_artifacts(
            registration, audit,
            exposure_catalog_path=paths["exposure_catalog"],
            exclusion_ledger_path=paths["exclusion_ledger"],
            design_spec_path=paths["design_spec"],
            design_report_path=paths["design_report"],
            design_simulation_path=paths["design_simulation"],
            simulator_source_path=paths["simulator_source"],
            simulation_bank_path=paths["simulation_bank"],
            root=tmp_path,
        )


def test_preregistration_release_rebuilds_ledger_from_evidence(tmp_path):
    registration, audit, paths = _preregistration_fixture(tmp_path)
    paths["ledger_evidence"].write_bytes(b"changed evidence")
    with pytest.raises(ValueError, match="cannot be reconstructed.*evidence artifact changed"):
        _validate_preregistration_artifacts(
            registration, audit,
            exposure_catalog_path=paths["exposure_catalog"],
            exclusion_ledger_path=paths["exclusion_ledger"],
            design_spec_path=paths["design_spec"],
            design_report_path=paths["design_report"],
            design_simulation_path=paths["design_simulation"],
            simulator_source_path=paths["simulator_source"],
            simulation_bank_path=paths["simulation_bank"],
            root=tmp_path,
        )


def test_preregistration_release_binds_simulation_files_and_analysis_contract(tmp_path):
    registration, audit, paths = _preregistration_fixture(tmp_path)
    paths["simulator_source"].write_text("changed simulator\n")
    with pytest.raises(ValueError, match="simulator source differs"):
        _validate_preregistration_artifacts(
            registration, audit,
            exposure_catalog_path=paths["exposure_catalog"],
            exclusion_ledger_path=paths["exclusion_ledger"],
            design_spec_path=paths["design_spec"],
            design_report_path=paths["design_report"],
            design_simulation_path=paths["design_simulation"],
            simulator_source_path=paths["simulator_source"],
            simulation_bank_path=paths["simulation_bank"],
            root=tmp_path,
        )

    registration, audit, paths = _preregistration_fixture(tmp_path)
    paths["simulation_bank"].write_bytes(b"changed simulation bank")
    with pytest.raises(ValueError, match="simulation bank differs"):
        _validate_preregistration_artifacts(
            registration, audit,
            exposure_catalog_path=paths["exposure_catalog"],
            exclusion_ledger_path=paths["exclusion_ledger"],
            design_spec_path=paths["design_spec"],
            design_report_path=paths["design_report"],
            design_simulation_path=paths["design_simulation"],
            simulator_source_path=paths["simulator_source"],
            simulation_bank_path=paths["simulation_bank"],
            root=tmp_path,
        )

    registration, audit, paths = _preregistration_fixture(tmp_path)
    registration["plan"]["analysis"]["alpha"] = 0.1
    with pytest.raises(ValueError, match="analysis contract differs"):
        _validate_preregistration_artifacts(
            registration, audit,
            exposure_catalog_path=paths["exposure_catalog"],
            exclusion_ledger_path=paths["exclusion_ledger"],
            design_spec_path=paths["design_spec"],
            design_report_path=paths["design_report"],
            design_simulation_path=paths["design_simulation"],
            simulator_source_path=paths["simulator_source"],
            simulation_bank_path=paths["simulation_bank"],
            root=tmp_path,
        )


def test_failed_registered_decision_remains_releasable():
    rule = {
        "method": "simultaneous_paired_model_score_max_t",
        "claim_if_confidence_set_above_classical_boundary": True,
        "require_predictive_pass_for_all_confidence_set_models": True,
        "report_failure_without_suppressing_analysis": True,
    }
    registration = {"plan": {"analysis": {
        "primary_estimand": "best_predictive_anomalous_coherence_grid_point",
        "primary_decision_rule": rule, "minimum_relevant_coherence": 0.25,
        "predictive_checks": {"model_pass_policy": "all_plausible_best"},
    }}}
    result = {
        "registered_primary_estimand": "best_predictive_anomalous_coherence_grid_point",
        "coherence_grid": {
            "confidence_set_excludes_classical_region": False,
            "statistical_confidence_set_models": ["classical", "quantum"],
            "statistical_confidence_set_coordinate_interval": [0.0, 0.25],
        },
        "predictive_model_gate": {
            "policy": "all_plausible_best", "pass": False,
        },
    }
    result["registered_decision"] = registered_nonclassical_decision(
        result, registration["plan"]["analysis"])
    decision = _registered_decision_for_release(
        registration, {"result": result}, predictive_required=True)
    assert decision["claim_supported"] is False
    assert len(decision["failure_reasons"]) == 2
