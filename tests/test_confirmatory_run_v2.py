from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "examples" / "jiuzhang"
sys.path.insert(0, str(HERE))

import campaign_confirmatory_v2 as campaign  # noqa: E402
from confirmatory_contract import (beacon_payload_sha256, derive_seed,
                                   event_key, sha256_bytes, sha256_json,
                                   validate_registration)  # noqa: E402
from select_confirmatory_v2 import (DET_POSITIONS, INPUT_HASH_DOMAIN,
                                    population_audit_from_counts)  # noqa: E402


def _registration():
    source_hash = campaign.analysis_source_hash()
    population = population_audit_from_counts(
        source_hash="0" * 64, n_records=20, exclusions=[], bands=[27],
        n_strata=2, counts={27: [20, 0]})
    plan = {
        "analysis_commit": "deadbeef",
        "randomness_beacon": {"provider": "test", "source": "test-beacon",
                              "round": 9,
                              "availability_utc": "2026-01-02T00:00:00Z"},
        "selection": {"bands": [27], "targets": {"27": 1},
                      "reserves": {"27": 2}, "n_strata": 2,
                      "n_records": 20, "record_bytes": 16,
                      "source_raw_sha256": "0" * 64,
                      "exclusion_sha256": sha256_json([]),
                      "exclusion_ledger_sha256": "1" * 64,
                      "population_audit_sha256": population["audit_payload_sha256"]},
        "analysis": {
                     "estimands": ["best_predictive_anomalous_coherence_grid_point"],
                     "primary_estimand": "best_predictive_anomalous_coherence_grid_point",
                     "design_report_sha256": "2" * 64,
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
                     "bootstrap_seed": 1, "alpha": 0.05,
                     "refusal_analysis": {
                         "method": "within_fixed_band_stratum_max_abs_mean_score_difference_permutation",
                         "reps": 100, "seed": 3, "alpha": 0.05,
                         "inferential_gate": False,
                         "recovery_method": "independent_high_precision_interval_reevaluation",
                         "minimum_precision_bits": 256,
                         "recovery_source_sha256": "3" * 64,
                         "recovery_container_digest": "recovery@sha256:" + "4" * 64,
                     },
                     "resampling_unit": "event_within_fixed_common_stratum",
                     "resampling_method": "calibration_draw_conditioned_fixed_strata_srswor_fpc_max_t",
                     "population_scope": "finite_registered_acquisition",
                     "normalizer_replicates": {
                         "count": 2, "samples_per_replicate": 1, "seed": 2},
                     "predictive_checks": {
                         "model_pass_policy": "all_plausible_best",
                         "detector_pairs": [[0, 1]],
                         "thresholds": {"click_count_tv_max": 0.1,
                                        "marginal_rms_max": 0.1,
                                        "pair_covariance_rms_max": 0.1},
                     }},
        "numerical_contract": {"scope": "binary64-matrix torontonian enclosure",
                               "state_fingerprints": {},
                               "analysis_source_sha256": source_hash},
        "models": {"exp_id": 0, "parameterization": "classical_excess",
                   "classical_boundary": 0.0, "reference_model": "classical",
                   "alternative_model": "squeezed",
                   "coherence_points": {
                       "classical": 0.0, "middle": 0.25, "squeezed": 1.0}},
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
    ph = sha256_json(plan)
    source, rnd, value = "test-beacon", 9, "abcd"
    seed = derive_seed(public_url="https://example.invalid/plan", public_sha256=ph,
                       published_at_utc="2026-01-01T00:00:00Z",
                       beacon_source=source, beacon_round=rnd, beacon_value=value)
    return {
        "schema": "gbskernels.confirmatory.v2", "plan": plan,
        "public": {"url": "https://example.invalid/plan", "sha256": ph,
                   "plan_sha256": ph, "timestamp_utc": "2026-01-01T00:00:00Z",
                   "immutable": True,
                   "timestamp_proof_url": "https://example.invalid/archive/proof",
                   "timestamp_proof_sha256": "e" * 64},
        "beacon": {"source": source, "round": rnd, "value": value,
                   "availability_utc": "2026-01-02T00:00:00Z",
                   "proof": {"source": source, "round": rnd,
                             "payload_sha256": beacon_payload_sha256(source, rnd, value),
                             "url": "https://example.invalid/beacon/9",
                             "record_sha256": "f" * 64}},
        "seed_derivation": seed,
    }


def _fixture():
    registration = _registration()
    normalized = validate_registration(registration)
    source_hash = normalized["plan"]["numerical_contract"]["analysis_source_sha256"]
    population = population_audit_from_counts(
        source_hash="0" * 64, n_records=20, exclusions=[], bands=[27],
        n_strata=2, counts={27: [20, 0]})
    rows = []
    for pos, role in enumerate(("primary", "reserve", "reserve")):
        pattern = np.zeros(100, dtype=np.uint8); pattern[:27] = 1
        pattern[pos] = 0
        pattern[27 + pos] = 1
        packed = np.packbits(pattern).tobytes()
        bits = np.zeros(128, dtype=np.uint8)
        timestamp = pos + 1
        bits[:16] = np.unpackbits(np.asarray(
            [(timestamp >> 8) & 0xFF, timestamp & 0xFF], dtype=np.uint8))
        bits[DET_POSITIONS[::-1]] = pattern
        raw = np.packbits(bits).tobytes()
        rows.append({
            "role": role, "band": 27, "record_index": 10 + pos,
            "timestamp_bits": format(timestamp, "016b"), "timestamp_uint16": timestamp,
            "abnormal": False, "pattern": pattern.tolist(),
            "pattern_packed_hex": packed.hex(), "raw_record_hex": raw.hex(),
            "key": event_key(10 + pos, normalized["seed_derivation"]["seed_hex"]),
            "input_hash": sha256_bytes(INPUT_HASH_DOMAIN + (27).to_bytes(2, "big") + raw),
            "source_raw_hash": sha256_bytes(raw), "stratum": 0,
            "rank_in_stratum": pos, "eligible_in_stratum": 20,
            "primary_quota": 1, "reserve_quota": 2,
            "inclusion_probability": 0.05 if role == "primary" else 0.1,
            "inclusion_probability_exact": "1/20" if role == "primary" else "2/20",
            "primary_inclusion_probability": 0.05,
            "primary_inclusion_probability_exact": "1/20",
            "manifest_inclusion_probability": 0.15,
            "manifest_inclusion_probability_exact": "3/20",
        })
    body = {
        "kind": "jiuzhang1_confirmatory_selection_v2", "registered": True,
        "registration": normalized,
        "seed": normalized["seed_derivation"]["seed_hex"],
        "source": {"record_bytes": 16, "n_records": 20,
                   "source_raw_sha256": "0" * 64},
        "strata": {"count": 2, "definition": "test", "edges": [0, 10, 20]},
        "exclusions": {"record_indices": [], "count": 0,
                       "sha256": sha256_json([])},
        "population_audit": population,
        "design": {"27": {"eligible_by_stratum": [20, 0], "eligible_total": 20,
                           "primary_target": 1, "primary_by_stratum": [1, 0],
                           "reserve_target": 2, "reserve_by_stratum": [2, 0]}},
        "hash_contract": {}, "primary": rows[:1], "reserves": rows[1:],
    }
    manifest = {**body, "manifest_payload_sha256": sha256_json(body)}
    contract_body = {
        "schema": "gbskernels.run-contract.v2",
        "manifest_id": manifest["manifest_payload_sha256"],
        "registration_id": normalized["public"]["plan_sha256"],
        "analysis_commit": "deadbeef",
        "container_digest": "image@sha256:" + "a" * 64,
        "numerical_scope": normalized["plan"]["numerical_contract"]["scope"],
        "analysis_source_sha256": source_hash,
        "kernel_binary": {"filename": "gbskernels_ext.so", "sha256": "b" * 64,
                          "bytes": 123, "gbskernels_version": "test"},
        "states": {},
    }
    contract = {**contract_body, "run_id": sha256_json(contract_body)}
    return registration, manifest, contract


def _write_row(root: Path, manifest, pos: int, refused: bool):
    ev = campaign._manifest_event(manifest, 27, pos)
    run_id = json.loads((root / "contract.json").read_text())["run_id"]
    row = {"schema": "gbskernels.event-evaluation.v2", "run_id": run_id,
           **{k: v for k, v in ev.items() if k != "pattern"}, "refused": refused}
    if refused:
        row["refusal_reason"] = "test"
    else:
        row.update({"log_pattern_ratio_mid": 0.2,
                    "log_pattern_ratio_halfwidth": 0.01})
    (root / "events").mkdir(parents=True, exist_ok=True)
    (root / "events" / f"{ev['event_id']}.json").write_text(json.dumps(row))


def test_ranked_reserve_replaces_refusal(tmp_path):
    registration, manifest, contract = _fixture()
    (tmp_path / "contract.json").write_text(json.dumps(contract))
    _write_row(tmp_path, manifest, 0, True)
    _write_row(tmp_path, manifest, 1, False)
    out = campaign.reduce_run(registration, manifest, tmp_path)
    assert out["complete"] is True
    assert out["n_usable"] == 1 and out["n_refused"] == 1
    assert out["rows"][0]["record_index"] == 11
    assert out["rows"][0]["replacement_for_refusal"] is True
    assert out["rows"][0]["replacement_for_refusal_event_id"] \
        == out["refusals"][0]["event_id"]


def test_missing_rank_blocks_later_reserve(tmp_path):
    registration, manifest, contract = _fixture()
    (tmp_path / "contract.json").write_text(json.dumps(contract))
    _write_row(tmp_path, manifest, 1, False)
    out = campaign.reduce_run(registration, manifest, tmp_path)
    assert out["complete"] is False and out["n_usable"] == 0


def test_wrong_record_binding_fails(tmp_path):
    registration, manifest, contract = _fixture()
    (tmp_path / "contract.json").write_text(json.dumps(contract))
    _write_row(tmp_path, manifest, 0, False)
    path = next((tmp_path / "events").glob("*.json"))
    row = json.loads(path.read_text()); row["record_index"] = 99
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="record_index"):
        campaign.reduce_run(registration, manifest, tmp_path)


def test_unused_reserve_refusal_does_not_enter_missingness_analysis(tmp_path):
    registration, manifest, contract = _fixture()
    (tmp_path / "contract.json").write_text(json.dumps(contract))
    _write_row(tmp_path, manifest, 0, False)
    _write_row(tmp_path, manifest, 1, True)
    out = campaign.reduce_run(registration, manifest, tmp_path)
    assert out["complete"] is True
    assert out["n_refused"] == 0
    assert out["n_evaluated_refused_total"] == 1


def test_run_contract_binds_actual_source_bytes_and_container(monkeypatch):
    registration, manifest, _ = _fixture()
    monkeypatch.setattr(campaign, "current_commit", lambda: "deadbeef")
    source_hash = registration["plan"]["numerical_contract"]["analysis_source_sha256"]
    monkeypatch.setattr(campaign, "analysis_source_hash", lambda: source_hash)
    monkeypatch.setattr(campaign, "state_fingerprint", lambda states: {})
    monkeypatch.setattr(campaign, "kernel_binary_fingerprint", lambda: {
        "filename": "gbskernels_ext.so", "sha256": "b" * 64,
        "bytes": 123, "gbskernels_version": "test"})
    monkeypatch.setenv("GBS_CONTAINER_DIGEST", "image@sha256:" + "a" * 64)
    states = {"classical": {}, "squeezed": {}, "middle": {}}
    contract = campaign.make_run_contract(registration, manifest, states)
    assert contract["analysis_source_sha256"] == source_hash
    assert contract["container_digest"].startswith("image@sha256:")

    monkeypatch.setattr(campaign, "analysis_source_hash", lambda: "1" * 64)
    with pytest.raises(ValueError, match="source bytes"):
        campaign.make_run_contract(registration, manifest, states)


def test_event_probability_proxy_interval_is_outward(monkeypatch):
    def tor_single(matrix, **kwargs):
        value = 2.0 if matrix[0, 0] == 1.0 else 3.0
        return value, {"abs_error_bound": 0.1}

    monkeypatch.setitem(sys.modules, "gbskernels", types.SimpleNamespace(tor_single=tor_single))
    states = {
        "reference": {"O": np.eye(4), "log_sqrt_detQ": 0.2},
        "alternative": {"O": 2 * np.eye(4), "log_sqrt_detQ": 0.4},
    }
    event = {"pattern": np.asarray([True, False])}
    out = campaign._default_evaluator(states, event)
    exact = math.log(3.0) - 0.4 - (math.log(2.0) - 0.2)
    assert out["log_pattern_ratio_lo"] < exact < out["log_pattern_ratio_hi"]
    assert out["log_pattern_ratio_halfwidth"] >= (
        out["log_pattern_ratio_hi"] - out["log_pattern_ratio_lo"]) / 2


def test_state_fingerprint_binds_husimi_matrix():
    state = {"O": np.eye(4), "cov": 2 * np.eye(4), "Q": 3 * np.eye(4),
             "log_sqrt_detQ": 0.5}
    first = campaign.state_fingerprint({"model": state})
    changed = dict(state, Q=4 * np.eye(4))
    second = campaign.state_fingerprint({"model": changed})
    assert first["model"]["Q_sha256"] != second["model"]["Q_sha256"]


def test_kernel_exception_becomes_auditable_refusal(monkeypatch):
    def refuse(*args, **kwargs):
        raise ValueError("numerical boundary")

    monkeypatch.setitem(sys.modules, "gbskernels",
                        types.SimpleNamespace(tor_single=refuse))
    states = {"reference": {"O": np.eye(4), "log_sqrt_detQ": 0.0},
              "alternative": {"O": np.eye(4), "log_sqrt_detQ": 0.0}}
    out = campaign._default_evaluator(
        states, {"pattern": np.asarray([True, False])})
    assert out["refused"] is True
    assert "reference" in out["refusal_reason"]
