from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from examples.jiuzhang.confirmatory_contract import canonical_bytes, sha256_json
from examples.jiuzhang.confirmatory_design import (
    DESIGN_SCHEMA,
    SIMULATION_SCHEMA,
    DesignError,
    analysis_contract_from_plan,
    build_report,
    canonicalize_resolved_spec,
    design_id,
    load_canonical_spec,
    minimum_reserve_count,
    one_sided_binomial_bounds,
    predictive_thresholds,
    seed_for,
    upper_tolerance_limit,
)


REPO = Path(__file__).resolve().parents[1]


def _analysis_contract() -> dict:
    return {
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


def test_repository_templates_freeze_the_same_analysis_contract():
    plan = json.loads((REPO / "docs/confirmatory_v2_plan.template.json").read_text())
    spec = json.loads(
        (REPO / "docs/confirmatory_v2_design_spec.template.json").read_text())
    assert spec["analysis_contract"] == analysis_contract_from_plan(plan)


def test_resolved_design_spec_is_materialized_as_canonical_json(tmp_path):
    spec = _spec(("candidate-a", (10, 2)))
    source = tmp_path / "resolved-pretty.json"
    source.write_text(json.dumps(spec, indent=2))
    out = tmp_path / "confirmatory-design-spec.json"
    assert canonicalize_resolved_spec(source, out) == spec
    assert out.read_bytes() == canonical_bytes(spec)
    assert load_canonical_spec(out) == spec
    with pytest.raises(DesignError, match="must be a canonical JSON object"):
        load_canonical_spec(source)

    unresolved = tmp_path / "unresolved.json"
    unresolved.write_text(json.dumps({**spec, "simulator_source_sha256": "REPLACE-ME"}))
    with pytest.raises(DesignError, match="still contains placeholders"):
        canonicalize_resolved_spec(unresolved, tmp_path / "must-not-exist.json")


def _audit() -> dict:
    body = {
        "schema": "gbskernels.selection-population-audit.v1",
        "source_raw_sha256": "a" * 64,
        "n_records": 1000,
        "record_bytes": 16,
        "exclusion_sha256": "b" * 64,
        "exclusion_ledger_sha256": "9" * 64,
        "exclusion_ledger_complete": True,
        "registration_eligible": True,
        "n_excluded": 10,
        "bands": [27, 28],
        "n_strata": 2,
        "eligible_by_band_stratum": {"27": [100, 100], "28": [120, 80]},
        "eligible_by_band": {"27": 200, "28": 200},
        "band_weights_within_window": {"27": 0.5, "28": 0.5},
        "band_weights_exact": {"27": "200/400", "28": "200/400"},
    }
    return {**body, "audit_payload_sha256": sha256_json(body)}


def _spec(*candidate_rows: tuple[str, tuple[int, int]]) -> dict:
    if not candidate_rows:
        candidate_rows = (("candidate", (20, 20)),)
    return {
        "schema": DESIGN_SCHEMA,
        "bands": [27, 28],
        "calibration_posterior_sha256": "c" * 64,
        "simulator_source_sha256": "d" * 64,
        "analysis_contract": _analysis_contract(),
        "event_cost": {"27": 1.0, "28": 2.0},
        "candidate_targets": [
            {"name": name, "targets": {"27": targets[0], "28": targets[1]}}
            for name, targets in candidate_rows
        ],
        "scenarios": {
            "classical": ["classical-boundary"],
            "alternative": ["eta-mre"],
            "predictive_valid": ["classical-boundary", "sub-mre", "eta-mre"],
            "predictive_misspecification": ["wrong-loss"],
        },
        "criteria": {
            "type_i_error_max": 0.05,
            "power_min": 0.8,
            "monte_carlo_confidence": 0.9,
            "minimum_relevant_coherence": 0.25,
            "predictive_family_false_failure_max": 0.3,
            "predictive_misspecification_rejection_min": 0.8,
            "reserve_family_failure_max": 0.05,
            "simulation_trials_min": 100,
            "predictive_draws_min": 80,
            "refusal_trials_min": 100,
        },
    }


def _metrics(value: float, n: int = 80) -> dict:
    return {
        "click_count_tv": [value] * n,
        "marginal_rms": [value / 2] * n,
        "pair_covariance_rms": [value / 4] * n,
    }


def _candidate(name: str, targets: tuple[int, int], *, false_claims: int = 0) -> dict:
    return {
        "name": name,
        "targets": {"27": targets[0], "28": targets[1]},
        "classical_claims": [
            {"name": "classical-boundary", "successes": 10 * false_claims,
             "trials": 1000}
        ],
        "alternative_successes": [
            {"name": "eta-mre", "successes": 999, "trials": 1000}
        ],
        "predictive_valid_draws": [
            {"name": name, **_metrics(0.05)}
            for name in ("classical-boundary", "sub-mre", "eta-mre")
        ],
        "predictive_misspecification_draws": [
            {"name": "wrong-loss", **_metrics(0.5)}
        ],
        "refusal_studies": {
            "27": {"refusals": 0, "trials": 1000},
            "28": {"refusals": 0, "trials": 1000},
        },
    }


def _simulation(spec: dict, audit: dict, candidates: list[dict]) -> dict:
    return {
        "schema": SIMULATION_SCHEMA,
        "design_id": design_id(spec, audit),
        "source": {
            "kind": "independent_calibration_forward_simulation",
            "uses_analysis_acquisition_outcomes": False,
            "input_roles": ["calibration_posterior", "forward_model"],
            "calibration_posterior_sha256": "c" * 64,
            "simulator_source_sha256": "d" * 64,
            "simulation_bank_sha256": "e" * 64,
        },
        "candidates": candidates,
    }


def test_seed_derivation_is_deterministic_and_domain_separated():
    identifier = "1" * 64
    assert seed_for(identifier, "truth", 1) == seed_for(identifier, "truth", 1)
    assert seed_for(identifier, "truth", 1) != seed_for(identifier, "truth", 2)
    assert seed_for(identifier, "truth", 1) != seed_for(identifier, "normalizer", 1)


def test_exact_bounds_and_reserve_are_fail_closed():
    lower, upper = one_sided_binomial_bounds(0, 100, 0.95)
    assert lower == 0.0 and 0.0 < upper < 0.05
    reserve = minimum_reserve_count(20, upper, 0.01)
    assert reserve > 0
    assert minimum_reserve_count(20, 0.0, 0.01) == 0
    with pytest.raises(DesignError, match="no finite reserve"):
        minimum_reserve_count(20, 1.0, 0.01)


def test_distribution_free_predictive_tolerances():
    rng = np.random.default_rng(7)
    values = rng.uniform(0.0, 0.1, 1000)
    row = upper_tolerance_limit(values, content=0.95, confidence=0.95)
    assert 0.09 < row["threshold"] <= 0.1
    report = predictive_thresholds(
        [{key: values for key in ("click_count_tv", "marginal_rms",
                                  "pair_covariance_rms")}],
        family_false_failure=0.15,
        confidence=0.95,
    )
    assert set(report["thresholds"]) == {
        "click_count_tv_max", "marginal_rms_max", "pair_covariance_rms_max"
    }
    with pytest.raises(DesignError, match="too few"):
        upper_tolerance_limit([0.1, 0.2], content=0.999, confidence=0.99)


def test_report_selects_minimum_cost_feasible_candidate_and_keeps_failures():
    audit = _audit()
    spec = _spec(("expensive", (24, 24)), ("cheap", (20, 20)),
                 ("ambiguous-null", (18, 18)))
    expensive = _candidate("expensive", (24, 24))
    cheap = _candidate("cheap", (20, 20))
    failed = _candidate("ambiguous-null", (18, 18), false_claims=5)
    report = build_report(spec, audit, _simulation(spec, audit, [expensive, failed, cheap]))
    assert report["pass"] is True
    assert report["selected"]["name"] == "cheap"
    assurance = report["monte_carlo_assurance"]
    assert assurance["simultaneous_bound_families"] == 18
    assert assurance["per_family_confidence"] > assurance["global_confidence"]
    assert report["analysis_contract"] == spec["analysis_contract"]
    assert len(report["candidates"]) == 3
    assert next(row for row in report["candidates"]
                if row["name"] == "ambiguous-null")["pass"] is False
    body = {key: value for key, value in report.items()
            if key != "design_report_payload_sha256"}
    assert report["design_report_payload_sha256"] == sha256_json(body)


def test_no_feasible_design_is_an_explicit_result():
    audit, spec = _audit(), _spec(("failed", (20, 20)))
    failed = _candidate("failed", (20, 20), false_claims=5)
    report = build_report(spec, audit, _simulation(spec, audit, [failed]))
    assert report["pass"] is False
    assert report["selected"] is None
    assert report["status"] == "no_feasible_design"


def test_outcome_bearing_simulation_inputs_are_rejected():
    audit, spec = _audit(), _spec()
    simulation = _simulation(spec, audit, [_candidate("candidate", (20, 20))])
    simulation["source"]["input_roles"].append("legacy_holdout")
    with pytest.raises(DesignError, match="forbidden"):
        build_report(spec, audit, simulation)
    simulation["source"]["input_roles"] = "legacy_holdout"
    with pytest.raises(DesignError, match="string array"):
        build_report(spec, audit, simulation)
    simulation["source"]["input_roles"] = ["calibration_posterior"]
    simulation["source"]["uses_analysis_acquisition_outcomes"] = True
    with pytest.raises(DesignError, match="exclude"):
        build_report(spec, audit, simulation)


def test_design_id_changes_with_population_or_specification():
    audit, spec = _audit(), _spec()
    first = design_id(spec, audit)
    changed = json.loads(json.dumps(spec))
    changed["criteria"]["power_min"] = 0.9
    assert design_id(changed, audit) != first
    altered_audit = json.loads(json.dumps(audit))
    altered_audit["source_raw_sha256"] = "f" * 64
    body = {key: value for key, value in altered_audit.items()
            if key != "audit_payload_sha256"}
    altered_audit["audit_payload_sha256"] = sha256_json(body)
    assert design_id(spec, altered_audit) != first


def test_design_rejects_tampered_or_registration_ineligible_population_audit():
    audit, spec = _audit(), _spec()
    audit["eligible_by_band_stratum"]["27"][0] += 1
    with pytest.raises(DesignError, match="payload hash"):
        design_id(spec, audit)

    audit = _audit()
    audit["registration_eligible"] = False
    body = {key: value for key, value in audit.items()
            if key != "audit_payload_sha256"}
    audit["audit_payload_sha256"] = sha256_json(body)
    with pytest.raises(DesignError, match="registration-eligible"):
        design_id(spec, audit)


def test_design_binds_analysis_grid_alpha_and_bootstrap_configuration():
    audit, spec = _audit(), _spec()
    candidate = _candidate("candidate", (20, 20))
    simulation = _simulation(spec, audit, [candidate])
    spec["analysis_contract"]["alpha"] = 0.1
    with pytest.raises(DesignError, match="alpha"):
        build_report(spec, audit, simulation)

    spec = _spec()
    simulation = _simulation(spec, audit, [candidate])
    spec["analysis_contract"]["models"]["coherence_points"]["eta-mre"] = 0.2
    with pytest.raises(DesignError, match="minimum relevant coherence"):
        build_report(spec, audit, simulation)

    spec = _spec()
    simulation = _simulation(spec, audit, [candidate])
    spec["analysis_contract"]["refusal_analysis"]["seed"] += 1
    # A different frozen seed changes the design ID, so even a simulation
    # copied from the original contract is rejected before any outcome use.
    with pytest.raises(DesignError, match="different design specification"):
        build_report(spec, audit, simulation)


@pytest.mark.parametrize("metric", ["click_count_tv", "marginal_rms",
                                     "pair_covariance_rms"])
def test_predictive_draw_minimum_applies_to_every_metric(metric):
    audit, spec = _audit(), _spec()
    candidate = _candidate("candidate", (20, 20))
    candidate["predictive_valid_draws"][0][metric] = \
        candidate["predictive_valid_draws"][0][metric][:40]
    with pytest.raises(DesignError, match="registered minimum|lengths differ"):
        build_report(spec, audit, _simulation(spec, audit, [candidate]))
