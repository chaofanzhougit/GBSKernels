from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from examples.jiuzhang.confirmatory_contract import (
    ContractError,
    beacon_payload_sha256,
    canonical_json,
    derive_seed,
    event_key,
    sha256_json,
    validate_registration,
)
from examples.jiuzhang.select_confirmatory_v2 import (
    DET_POSITIONS,
    RECORD_BYTES,
    SelectionError,
    common_stratum,
    exclusion_sha256,
    largest_remainder_allocation,
    load_exclusion_indices,
    select_from_raw,
)
from examples.jiuzhang.audit_selection_population import audit as audit_population


def _registration(plan: dict, *, published: str = "2026-01-01T00:00:00Z") -> dict:
    plan = json.loads(json.dumps(plan))
    selection = plan.setdefault("selection", {})
    targets = selection.setdefault("targets", {"2": 2})
    selection.setdefault("bands", [int(x) for x in targets])
    selection.setdefault("reserves", {str(x): 0 for x in targets})
    selection.setdefault("n_strata", 2)
    selection.setdefault("n_records", 1)
    selection.setdefault("record_bytes", 16)
    selection.setdefault("source_raw_sha256", "0" * 64)
    selection.setdefault("exclusion_sha256", "0" * 64)
    selection.setdefault("exclusion_ledger_sha256", "1" * 64)
    selection.setdefault("population_audit_sha256", "0" * 64)
    bands = selection["bands"]
    plan.setdefault("analysis_commit", "deadbeef")
    plan.setdefault("randomness_beacon", {
        "provider": "test-beacon", "source": "nist-randomness-beacon",
        "round": 41, "availability_utc": "2026-01-02T00:00:00Z",
    })
    plan.setdefault("models", {"exp_id": 0, "parameterization": "classical_excess",
                               "classical_boundary": 0.0,
                               "reference_model": "classical",
                               "alternative_model": "squeezed",
                               "coherence_points": {
                                   "classical": 0.0, "middle": 0.25,
                                   "squeezed": 1.0}})
    plan.setdefault("analysis", {
                                 "estimands": [
                                     "best_predictive_anomalous_coherence_grid_point"],
                                 "primary_estimand": (
                                     "best_predictive_anomalous_coherence_grid_point"),
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
                                 "band_weights": {str(x): 1 / len(bands) for x in bands},
                                 "bootstrap_reps": 100, "bootstrap_seed": 1,
                                 "alpha": 0.05,
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
                                     "count": 2, "samples_per_replicate": 1,
                                     "seed": 2},
                                 "predictive_checks": {
                                     "model_pass_policy": "all_plausible_best",
                                     "detector_pairs": [[0, 1]],
                                     "thresholds": {
                                         "click_count_tv_max": 0.1,
                                         "marginal_rms_max": 0.1,
                                         "pair_covariance_rms_max": 0.1,
                                     },
                                 }})
    plan.setdefault("numerical_contract", {"scope": "test",
                                            "state_fingerprints": {},
                                            "analysis_source_sha256": "0" * 64})
    plan.setdefault("prior_data_use", {
        "record_level_policy":
            "exclude_all_materialized_published_selected_or_scored_records",
        "outcome_blind_mechanical_scans_are_not_record_level_exposure": True,
        "full_acquisition_aggregates_previously_examined": True,
        "independent_acquisition": False,
        "claim_scope":
            "finite_registered_acquisition_conditional_on_prior_aggregate_use",
        "author_attestation_required": True,
    })
    plan.setdefault("external_requirements", {
        "public_registration_required": True,
        "future_beacon_required": True,
        "joint_normalizer_replicates_required": True,
        "container_digest_required": True,
        "absolute_predictive_checks_required": True,
    })
    public_hash = sha256_json(plan)
    source = "nist-randomness-beacon"
    round_number = 41
    value = "0123456789abcdef"
    proof_hash = beacon_payload_sha256(source, round_number, value)
    beacon = {
        "source": source,
        "round": round_number,
        "value": value,
        "availability_utc": "2026-01-02T00:00:00Z",
        "proof": {
            "source": source,
            "round": round_number,
            "payload_sha256": proof_hash,
            "url": "https://example.invalid/beacon/41",
            "record_sha256": "f" * 64,
        },
    }
    seed = derive_seed(
        public_url="https://example.invalid/registration/sha256-plan",
        public_sha256=public_hash,
        published_at_utc=published,
        beacon_source=source,
        beacon_round=round_number,
        beacon_value=value,
    )
    return {
        "schema": "gbskernels.confirmatory.v2",
        "plan": plan,
        "public": {
            "url": "https://example.invalid/registration/sha256-plan",
            "sha256": public_hash,
            "plan_sha256": public_hash,
            "timestamp_utc": published,
            "immutable": True,
            "timestamp_proof_url": "https://example.invalid/archive/proof",
            "timestamp_proof_sha256": "e" * 64,
        },
        "beacon": beacon,
        "seed_derivation": seed,
    }


def _raw_record(timestamp: int, clicks: tuple[int, ...], abnormal: bool = False) -> bytes:
    bits = np.zeros(128, dtype=np.uint8)
    bits[:16] = np.unpackbits(np.array([(timestamp >> 8) & 0xFF, timestamp & 0xFF], dtype=np.uint8))
    pattern = np.zeros(100, dtype=np.uint8)
    pattern[list(clicks)] = 1
    bits[DET_POSITIONS[::-1]] = pattern
    bits[127] = int(abnormal)
    return np.packbits(bits).tobytes()


def _synthetic_records() -> bytes:
    # Four common strata, with both target bands present in every stratum.
    records = []
    for h in range(4):
        for j in range(5):
            records.append(_raw_record(1000 + h * 10 + j, (h, 10 + j)))  # C=2
        for j in range(4):
            records.append(_raw_record(2000 + h * 10 + j, (20 + h, 30 + j, 40)))  # C=3
    return b"".join(records)


def test_canonical_json_and_registration_timing_are_strict():
    plan = {"selection": {"targets": {"2": 2}, "n_strata": 2}}
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    reg = _registration(plan)
    checked = validate_registration(reg)
    assert checked["seed"] == reg["seed_derivation"]["seed_uint64"]
    assert checked["seed_derivation"]["seed_hex"] == reg["seed_derivation"]["seed_hex"]

    equal_time = json.loads(json.dumps(reg))
    equal_time["beacon"]["availability_utc"] = equal_time["public"]["timestamp_utc"]
    with pytest.raises(ContractError, match="strictly"):
        validate_registration(equal_time)

    bad_proof = json.loads(json.dumps(reg))
    bad_proof["beacon"]["value"] = "different"
    with pytest.raises(ContractError, match="digest"):
        validate_registration(bad_proof)


def test_largest_remainder_is_exact_and_tie_broken_by_stratum():
    assert largest_remainder_allocation([1, 1], 1) == [1, 0]
    assert largest_remainder_allocation([1, 2, 6], 3) == [0, 1, 2]
    with pytest.raises(SelectionError):
        largest_remainder_allocation([1, 2], 4)


def test_selection_is_reproducible_and_auditable(tmp_path: Path):
    raw = _synthetic_records()
    out1 = tmp_path / "one.json"
    out2 = tmp_path / "two.json"
    kwargs = dict(
        targets={2: 8, 3: 6},
        reserves={2: 4, 3: 2},
        n_strata=4,
        exclude_record_indices=[0, 1],
        seed=123456,
        chunk_records=7,
    )
    first = select_from_raw(raw, output_path=out1, **kwargs)
    second = select_from_raw(raw, output_path=out2, **kwargs)
    assert first["manifest_payload_sha256"] == second["manifest_payload_sha256"]
    assert first["primary"] == second["primary"]
    assert first["reserves"] == second["reserves"]
    assert out1.read_bytes() == out2.read_bytes()

    assert len(first["primary"]) == 14
    assert len(first["reserves"]) == 6
    all_rows = first["primary"] + first["reserves"]
    assert all(row["record_index"] not in {0, 1} for row in all_rows)
    assert {row["stratum"] for row in all_rows} == {0, 1, 2, 3}
    for row in all_rows:
        assert len(row["pattern"]) == 100
        assert row["timestamp_bits"] == format(row["timestamp_uint16"], "016b")
        assert row["abnormal"] is False
        assert row["key"] == event_key(row["record_index"], 123456)
        assert len(row["input_hash"]) == 64
        assert len(row["source_raw_hash"]) == 64
        assert 0.0 < row["inclusion_probability"] <= 1.0
        assert row["manifest_inclusion_probability"] >= row["inclusion_probability"]

    assert first["exclusions"]["record_indices"] == [0, 1]
    assert first["exclusions"]["sha256"] == exclusion_sha256([0, 1])
    assert first["source"]["n_records"] == len(raw) // RECORD_BYTES
    assert first["source"]["source_raw_sha256"] == hashlib.sha256(raw).hexdigest()
    audit = audit_population(raw, bands=[2, 3], n_strata=4,
                             exclude_record_indices=[0, 1], chunk_records=7)
    assert first["population_audit"] == audit


def test_common_strata_are_global_and_allocation_uses_actual_counts():
    assert [common_stratum(i, 10, 3) for i in range(10)] == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    # In the synthetic source each band has equal counts per stratum; quotas are
    # consequently deterministic and never inferred from first-occurrence caps.
    manifest = select_from_raw(
        _synthetic_records(),
        targets={2: 8},
        reserves={2: 0},
        n_strata=4,
        exclude_record_indices=[],
        seed=9,
    )
    assert manifest["design"]["2"]["eligible_by_stratum"] == [5, 5, 5, 5]
    assert manifest["design"]["2"]["primary_by_stratum"] == [2, 2, 2, 2]


def test_exclusion_loader_requires_explicit_indices(tmp_path: Path):
    json_path = tmp_path / "exclude.json"
    json_path.write_text(json.dumps({"record_indices": [7, 2, 4]}))
    assert load_exclusion_indices(json_path) == [2, 4, 7]
    text_path = tmp_path / "exclude.txt"
    text_path.write_text("# audited holdout\n9\n3\n")
    assert load_exclusion_indices(text_path) == [3, 9]
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text("[1, 1]")
    with pytest.raises(SelectionError, match="duplicate"):
        load_exclusion_indices(duplicate)


def test_registered_plan_binds_source_and_design():
    raw = _synthetic_records()
    source_hash = hashlib.sha256(raw).hexdigest()
    exclusions = [0]
    ledger_hash = "1" * 64
    population = audit_population(raw, bands=[2], n_strata=4,
                                  exclude_record_indices=exclusions,
                                  exclusion_ledger_sha256=ledger_hash)
    plan = {
        "selection": {
            "targets": {"2": 8},
            "reserves": {"2": 4},
            "n_strata": 4,
            "n_records": len(raw) // RECORD_BYTES,
            "record_bytes": RECORD_BYTES,
            "source_raw_sha256": source_hash,
            "exclusion_sha256": exclusion_sha256(exclusions),
            "exclusion_ledger_sha256": ledger_hash,
            "population_audit_sha256": population["audit_payload_sha256"],
        }
    }
    reg = _registration(plan)
    manifest = select_from_raw(
        raw,
        targets={2: 8},
        reserves={2: 4},
        n_strata=4,
        exclude_record_indices=exclusions,
        exclusion_ledger_sha256=ledger_hash,
        registration=reg,
    )
    assert manifest["registered"] is True
    assert manifest["seed"] == validate_registration(reg)["seed_derivation"]["seed_hex"]
    assert all(row["key"] == event_key(row["record_index"], manifest["seed"])
               for row in manifest["primary"] + manifest["reserves"])
    bad = dict(plan)
    bad["selection"] = dict(plan["selection"], n_strata=2)
    with pytest.raises(SelectionError, match="mismatch"):
        select_from_raw(
            raw,
            targets={2: 8},
            reserves={2: 4},
            n_strata=4,
            exclude_record_indices=exclusions,
            exclusion_ledger_sha256=ledger_hash,
            registration=_registration(bad),
        )
