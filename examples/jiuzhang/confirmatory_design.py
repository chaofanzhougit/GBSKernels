"""Outcome-blind operating-characteristic design for confirmatory v2.

This module does not simulate Jiuzhang data itself.  It consumes a canonical
summary produced by an independently specified forward simulator and applies
the frozen decision criteria used to choose targets, operational reserves, and
absolute-predictive tolerances.  Raw acquisition data, legacy holdout results,
selection manifests, and verified runs are forbidden inputs.

The separation is deliberate: design selection is cheap and deterministic,
whereas the physical simulation bank can be expensive.  Every candidate is
evaluated from raw success/failure counts and predictive-metric draws, with
one-sided exact binomial assurance bounds.  Ambiguous Monte Carlo evidence is
not rounded into a passing design.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import beta, binom

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from confirmatory_contract import (canonical_bytes, sha256_json,
                                   write_canonical_json)  # noqa: E402
from confirmatory_common import placeholder_paths  # noqa: E402
from select_confirmatory_v2 import largest_remainder_allocation  # noqa: E402


DESIGN_SCHEMA = "gbskernels.confirmatory-design-spec.v1"
SIMULATION_SCHEMA = "gbskernels.confirmatory-design-simulation.v1"
REPORT_SCHEMA = "gbskernels.confirmatory-design.v1"
DESIGN_SEED_DOMAIN = b"GBSKERNELS/CONFIRMATORY/V2/DESIGN"
METRICS = (
    "click_count_tv",
    "marginal_rms",
    "pair_covariance_rms",
)
_HEX64 = re.compile(r"\A[0-9a-fA-F]{64}\Z")
_CONTAINER_DIGEST = re.compile(r"\A[^\s@]+(?:/[^\s@]+)*@sha256:[0-9a-fA-F]{64}\Z")
_FORBIDDEN_ROLES = {
    "analysis_acquisition",
    "legacy_holdout",
    "private_holdout",
    "raw_acquisition_outcomes",
    "selection_manifest",
    "verified_run",
    "confirmatory_result",
}
PRIMARY_ESTIMAND = "best_predictive_anomalous_coherence_grid_point"
PRIMARY_DECISION_RULE = {
    "method": "simultaneous_paired_model_score_max_t",
    "claim_if_confidence_set_above_classical_boundary": True,
    "require_predictive_pass_for_all_confidence_set_models": True,
    "report_failure_without_suppressing_analysis": True,
}
PREDICTIVE_MODEL_PASS_POLICY = "all_plausible_best"
REFUSAL_ANALYSIS_METHOD = (
    "within_fixed_band_stratum_max_abs_mean_score_difference_permutation")


class DesignError(ValueError):
    """Raised when a design input is incomplete, outcome-leaking, or invalid."""


def canonicalize_resolved_spec(source: str | Path, out: str | Path) -> dict[str, Any]:
    """Create the canonical design-spec artifact after external fields are resolved."""
    try:
        value = json.loads(Path(source).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DesignError(f"cannot read resolved design spec: {exc}") from exc
    if not isinstance(value, dict):
        raise DesignError("resolved design spec must be a JSON object")
    unresolved = placeholder_paths(value, prefix="design_spec")
    if unresolved:
        raise DesignError(
            "resolved design spec still contains placeholders: " + ", ".join(unresolved))
    write_canonical_json(out, value)
    return value


def load_canonical_spec(path: str | Path) -> dict[str, Any]:
    """Load the exact design artifact accepted by report/readiness/release stages."""
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise DesignError(f"cannot read canonical design spec: {exc}") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise DesignError("design spec must be a canonical JSON object")
    return value


def _hex64(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise DesignError(f"{field} must be a 64-character SHA256 digest")
    return value.lower()


def _probability(value: Any, field: str, *, strict: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DesignError(f"{field} must be numeric")
    out = float(value)
    valid = 0.0 < out < 1.0 if strict else 0.0 <= out <= 1.0
    if not valid:
        interval = "(0,1)" if strict else "[0,1]"
        raise DesignError(f"{field} must lie in {interval}")
    return out


def _population_audit_hash(population_audit: Mapping[str, Any]) -> str:
    """Verify that design inputs are the complete, self-hashed audit bytes."""
    if population_audit.get("schema") != "gbskernels.selection-population-audit.v1":
        raise DesignError("unsupported population-audit schema")
    supplied = _hex64(
        population_audit.get("audit_payload_sha256"),
        "population_audit.audit_payload_sha256",
    )
    body = {key: value for key, value in population_audit.items()
            if key != "audit_payload_sha256"}
    if sha256_json(body) != supplied:
        raise DesignError("population audit payload hash does not match its content")
    if population_audit.get("registration_eligible") is not True:
        raise DesignError("design requires a registration-eligible population audit")
    return supplied


def analysis_contract_from_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Project the plan fields that the design simulation must execute."""
    try:
        models = plan["models"]
        analysis = plan["analysis"]
        predictive = analysis["predictive_checks"]
        normalizers = analysis["normalizer_replicates"]
        calibration = analysis["calibration_draws"]
        refusal = analysis["refusal_analysis"]
        external = plan["external_requirements"]
        if not all(isinstance(value, Mapping) for value in (
                models, analysis, predictive, normalizers, calibration, refusal, external)):
            raise TypeError
        contract = {
            "models": dict(models),
            "primary_estimand": analysis["primary_estimand"],
            "primary_decision_rule": analysis["primary_decision_rule"],
            "alpha": analysis["alpha"],
            "bootstrap_reps": analysis["bootstrap_reps"],
            "bootstrap_seed": analysis["bootstrap_seed"],
            "normalizer_replicates": dict(normalizers),
            "reconstruction_required": external["reconstruction_required"],
            "calibration_draws": dict(calibration),
            "refusal_analysis": dict(refusal),
            "predictive_model_pass_policy": predictive["model_pass_policy"],
            "predictive_detector_pairs": predictive["detector_pairs"],
            "resampling_unit": analysis["resampling_unit"],
            "resampling_method": analysis["resampling_method"],
            "population_scope": analysis["population_scope"],
        }
    except (KeyError, TypeError) as exc:
        raise DesignError("plan lacks the analysis fields required by the design") from exc
    return json.loads(canonical_bytes(contract))


def _validate_analysis_contract(value: Any, criteria: Mapping[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "models", "primary_estimand", "primary_decision_rule", "alpha",
        "bootstrap_reps", "bootstrap_seed", "normalizer_replicates",
        "reconstruction_required", "calibration_draws",
        "refusal_analysis",
        "predictive_model_pass_policy", "predictive_detector_pairs",
        "resampling_unit", "resampling_method", "population_scope",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise DesignError("analysis_contract must freeze the complete registered analysis")
    models = value["models"]
    if not isinstance(models, Mapping):
        raise DesignError("analysis_contract.models must be an object")
    if models.get("exp_id") != 0 or models.get("parameterization") != "classical_excess":
        raise DesignError("design requires the Jiuzhang classical-excess model family")
    points = models.get("coherence_points")
    if not isinstance(points, Mapping) or len(points) < 3 or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in points.values()):
        raise DesignError("analysis_contract must freeze at least three coherence points")
    coordinates = {str(name): float(item) for name, item in points.items()}
    if len(set(coordinates.values())) != len(coordinates) or any(
            not -1.0 <= item <= 1.0 for item in coordinates.values()):
        raise DesignError("analysis_contract coherence points are invalid")
    boundary = models.get("classical_boundary")
    if isinstance(boundary, bool) or not isinstance(boundary, (int, float)):
        raise DesignError("analysis_contract classical boundary is invalid")
    boundary = float(boundary)
    reference = models.get("reference_model")
    alternative = models.get("alternative_model")
    if (reference not in coordinates or alternative not in coordinates
            or coordinates[str(reference)] != boundary
            or coordinates[str(alternative)] <= boundary):
        raise DesignError("analysis_contract reference/alternative models are invalid")
    eta = float(criteria["minimum_relevant_coherence"])
    if eta not in coordinates.values():
        raise DesignError("minimum relevant coherence is absent from the registered model grid")
    if value["primary_estimand"] != PRIMARY_ESTIMAND:
        raise DesignError("analysis_contract has the wrong primary estimand")
    if value["primary_decision_rule"] != PRIMARY_DECISION_RULE:
        raise DesignError("analysis_contract has the wrong primary decision rule")
    alpha = _probability(value["alpha"], "analysis_contract.alpha")
    if alpha != float(criteria["type_i_error_max"]):
        raise DesignError("analysis alpha must equal the design type-I error maximum")
    reps, seed = value["bootstrap_reps"], value["bootstrap_seed"]
    if (isinstance(reps, bool) or not isinstance(reps, int) or reps < 100
            or isinstance(seed, bool) or not isinstance(seed, int)):
        raise DesignError("analysis_contract bootstrap configuration is invalid")
    normalizers = value["normalizer_replicates"]
    if (not isinstance(normalizers, Mapping)
            or isinstance(normalizers.get("count"), bool)
            or not isinstance(normalizers.get("count"), int)
            or normalizers["count"] < 2
            or isinstance(normalizers.get("samples_per_replicate"), bool)
            or not isinstance(normalizers.get("samples_per_replicate"), int)
            or normalizers["samples_per_replicate"] < 1
            or isinstance(normalizers.get("seed"), bool)
            or not isinstance(normalizers.get("seed"), int)):
        raise DesignError("analysis_contract normalizer configuration is invalid")
    if value["predictive_model_pass_policy"] != PREDICTIVE_MODEL_PASS_POLICY:
        raise DesignError("analysis_contract must gate every plausible best model")
    pairs = value["predictive_detector_pairs"]
    if (not isinstance(pairs, list) or not pairs or any(
            not isinstance(pair, list) or len(pair) != 2
            or any(isinstance(index, bool) or not isinstance(index, int) for index in pair)
            or not 0 <= pair[0] < pair[1] < 100 for pair in pairs)):
        raise DesignError("analysis_contract predictive detector pairs are invalid")
    if value["reconstruction_required"] is not True:
        raise DesignError("analysis_contract must require reconstruction marginalization")
    calibration = value["calibration_draws"]
    if not isinstance(calibration, Mapping):
        raise DesignError("analysis_contract calibration draws are required")
    _hex64(calibration.get("posterior_sha256"),
           "analysis_contract.calibration_draws.posterior_sha256")
    if (isinstance(calibration.get("count"), bool)
            or not isinstance(calibration.get("count"), int)
            or calibration["count"] < 2
            or isinstance(calibration.get("seed"), bool)
            or not isinstance(calibration.get("seed"), int)):
        raise DesignError("analysis_contract calibration draw count/seed are invalid")
    if set(calibration.get("required_nuisances", [])) \
            != {"squeezing", "transfer", "loss", "block_drift"}:
        raise DesignError("analysis_contract calibration nuisances are incomplete")
    if calibration.get("dark_click_model") != "explicitly_zero":
        raise DesignError("analysis_contract dark-click model is unsupported")
    refusal = value["refusal_analysis"]
    refusal_fields = {
        "method", "reps", "seed", "alpha", "inferential_gate",
        "recovery_method", "minimum_precision_bits",
        "recovery_source_sha256", "recovery_container_digest",
    }
    if (not isinstance(refusal, Mapping)
            or set(refusal) != refusal_fields
            or refusal.get("method") != REFUSAL_ANALYSIS_METHOD
            or refusal.get("inferential_gate") is not False
            or refusal.get("recovery_method")
            != "independent_high_precision_interval_reevaluation"
            or isinstance(refusal.get("minimum_precision_bits"), bool)
            or not isinstance(refusal.get("minimum_precision_bits"), int)
            or refusal["minimum_precision_bits"] < 128
            or isinstance(refusal.get("reps"), bool)
            or not isinstance(refusal.get("reps"), int)
            or refusal["reps"] < 100
            or isinstance(refusal.get("seed"), bool)
            or not isinstance(refusal.get("seed"), int)):
        raise DesignError("analysis_contract refusal-analysis configuration is invalid")
    _hex64(refusal.get("recovery_source_sha256"),
           "analysis_contract.refusal_analysis.recovery_source_sha256")
    if (not isinstance(refusal.get("recovery_container_digest"), str)
            or _CONTAINER_DIGEST.fullmatch(refusal["recovery_container_digest"]) is None):
        raise DesignError("analysis_contract refusal-recovery container is invalid")
    refusal_alpha = _probability(
        refusal.get("alpha"), "analysis_contract.refusal_analysis.alpha")
    if refusal_alpha != alpha:
        raise DesignError("refusal-analysis alpha must equal the registered analysis alpha")
    if (value["resampling_unit"] != "event_within_fixed_common_stratum"
            or value["resampling_method"]
            != "calibration_draw_conditioned_fixed_strata_srswor_fpc_max_t"
            or value["population_scope"] != "finite_registered_acquisition"):
        raise DesignError("analysis_contract does not use the registered finite-population method")
    return json.loads(canonical_bytes(value))


def design_id(spec: Mapping[str, Any], population_audit: Mapping[str, Any]) -> str:
    """Bind the design specification to the audited eligible population."""
    if spec.get("schema") != DESIGN_SCHEMA:
        raise DesignError("unsupported design specification schema")
    audit_hash = _population_audit_hash(population_audit)
    return sha256_json({"spec": spec, "population_audit_sha256": audit_hash})


def seed_for(identifier: str, label: str, *indices: int | str) -> int:
    """Derive one domain-separated 64-bit design seed."""
    identifier = _hex64(identifier, "design_id")
    if not isinstance(label, str) or not label:
        raise DesignError("seed label must be non-empty")
    material = [label, *indices]
    digest = hashlib.sha256(
        DESIGN_SEED_DOMAIN + b"\0" + identifier.encode("ascii") + b"\0"
        + canonical_bytes(material)
    ).digest()
    return int.from_bytes(digest[:8], "big")


def one_sided_binomial_bounds(successes: int, trials: int,
                              confidence: float) -> tuple[float, float]:
    """Clopper-Pearson one-sided lower and upper confidence bounds."""
    if (isinstance(successes, bool) or isinstance(trials, bool)
            or not isinstance(successes, (int, np.integer))
            or not isinstance(trials, (int, np.integer))
            or trials <= 0 or successes < 0 or successes > trials):
        raise DesignError("binomial successes/trials are invalid")
    confidence = _probability(confidence, "monte_carlo_confidence")
    k, n = int(successes), int(trials)
    lower = 0.0 if k == 0 else float(beta.ppf(1.0 - confidence, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(confidence, k + 1, n - k))
    return lower, upper


def minimum_reserve_count(primary: int, refusal_probability_upper: float,
                          failure_probability: float, *, max_reserve: int = 100_000) -> int:
    """Smallest reserve count whose binomial usable-event shortfall is bounded."""
    if (isinstance(primary, bool) or not isinstance(primary, (int, np.integer))
            or int(primary) <= 0):
        raise DesignError("primary count must be a positive integer")
    q = _probability(refusal_probability_upper, "refusal_probability_upper", strict=False)
    delta = _probability(failure_probability, "failure_probability")
    if q == 1.0:
        raise DesignError("a refusal upper bound of one admits no finite reserve")
    n = int(primary)
    for reserve in range(max_reserve + 1):
        failure = float(binom.cdf(n - 1, n + reserve, 1.0 - q))
        if failure <= delta:
            return reserve
    raise DesignError("reserve search exceeded max_reserve")


def reserve_total_for_band(populations: Sequence[int], primary_total: int,
                           refusal_probability_upper: float,
                           family_failure_probability: float) -> dict[str, Any]:
    """Choose a band reserve total that satisfies every registered cell."""
    counts = [int(x) for x in populations]
    if not counts or any(x <= 0 for x in counts):
        raise DesignError("every registered stratum needs a positive population")
    primary = largest_remainder_allocation(counts, int(primary_total))
    if any(x < 2 for x in primary):
        raise DesignError("finite-population inference needs at least two primary events per cell")
    delta_cell = _probability(
        family_failure_probability, "reserve_family_failure_probability") / len(counts)
    required = [minimum_reserve_count(n, refusal_probability_upper, delta_cell)
                for n in primary]
    remaining = [N - n for N, n in zip(counts, primary, strict=True)]
    if any(r > N for r, N in zip(required, remaining, strict=True)):
        raise DesignError("eligible population cannot supply the required reserves")
    for total in range(sum(required), sum(remaining) + 1):
        allocated = largest_remainder_allocation(remaining, total)
        if all(got >= need for got, need in zip(allocated, required, strict=True)):
            return {
                "primary_by_stratum": primary,
                "required_reserve_by_stratum": required,
                "reserve_by_stratum": allocated,
                "reserve_total": int(total),
                "cell_failure_probability": float(delta_cell),
            }
    raise DesignError("no proportional reserve total satisfies every cell")


def upper_tolerance_limit(values: Sequence[float], *, content: float,
                          confidence: float) -> dict[str, Any]:
    """Distribution-free one-sided upper tolerance limit from an order statistic."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or np.any(~np.isfinite(array)) or np.any(array < 0):
        raise DesignError("predictive metric draws must be finite nonnegative vectors")
    content = _probability(content, "tolerance_content")
    confidence = _probability(confidence, "tolerance_confidence")
    # P(F(X_(k)) >= content) = P(Binomial(n, content) <= k-1).
    k = int(binom.ppf(confidence, len(array), content)) + 1
    if k > len(array):
        raise DesignError(
            "too few predictive draws for the requested content/confidence tolerance"
        )
    ordered = np.sort(array)
    return {
        "threshold": float(ordered[k - 1]),
        "order_statistic": int(k),
        "draws": int(len(array)),
        "content": float(content),
        "confidence": float(confidence),
    }


def predictive_thresholds(valid_draw_groups: Sequence[Mapping[str, Sequence[float]]], *,
                          family_false_failure: float,
                          confidence: float) -> dict[str, Any]:
    """Calibrate simultaneous tolerances for every valid scenario/model group."""
    groups = list(valid_draw_groups)
    if not groups:
        raise DesignError("at least one valid predictive-draw group is required")
    alpha = _probability(family_false_failure, "predictive_family_false_failure")
    confidence = _probability(confidence, "monte_carlo_confidence")
    content = 1.0 - alpha / len(METRICS)
    simultaneous_confidence = 1.0 - (1.0 - confidence) / (len(groups) * len(METRICS))
    details: dict[str, list[dict[str, Any]]] = {metric: [] for metric in METRICS}
    thresholds: dict[str, float] = {}
    for metric in METRICS:
        for group in groups:
            if metric not in group:
                raise DesignError(f"predictive group lacks {metric}")
            details[metric].append(upper_tolerance_limit(
                group[metric], content=content, confidence=simultaneous_confidence))
        thresholds[f"{metric}_max"] = max(row["threshold"] for row in details[metric])
    return {
        "thresholds": thresholds,
        "family_false_failure": float(alpha),
        "familywise_method": "Bonferroni metrics and worst registered valid group",
        "per_metric_content": float(content),
        "simultaneous_tolerance_confidence": float(simultaneous_confidence),
        "groups": int(len(groups)),
        "details": details,
    }


def _rate_rows(rows: Any, *, confidence: float, label: str,
               expected_names: set[str], minimum_trials: int) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise DesignError(f"{label} must contain at least one scenario")
    out = []
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("name"), str):
            raise DesignError(f"{label} scenario is invalid")
        successes, trials = row.get("successes"), row.get("trials")
        if not isinstance(trials, int) or isinstance(trials, bool) or trials < minimum_trials:
            raise DesignError(f"{label} trials are below the registered minimum")
        lo, hi = one_sided_binomial_bounds(successes, trials, confidence)
        out.append({"name": row["name"], "successes": int(successes),
                    "trials": int(trials), "rate": int(successes) / int(trials),
                    "lower": lo, "upper": hi})
    if {row["name"] for row in out} != expected_names:
        raise DesignError(f"{label} scenarios differ from the registered specification")
    return out


def _predictive_groups(rows: Any, *, label: str, expected_names: set[str],
                       minimum_draws: int) -> list[Mapping[str, Sequence[float]]]:
    if not isinstance(rows, list) or len(rows) != len(expected_names):
        raise DesignError(f"{label} differ from the frozen scenarios/minimum")
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("name"), str):
            raise DesignError(f"{label} contain an invalid group")
        names.add(row["name"])
        lengths: list[int] = []
        for metric in METRICS:
            values = row.get(metric)
            try:
                length = len(values)  # type: ignore[arg-type]
            except TypeError as exc:
                raise DesignError(f"{label} lack metric {metric}") from exc
            if length < minimum_draws:
                raise DesignError(f"{label} metric {metric} is below the registered minimum")
            lengths.append(length)
        if len(set(lengths)) != 1:
            raise DesignError(f"{label} metric lengths differ")
    if names != expected_names:
        raise DesignError(f"{label} differ from the frozen scenarios/minimum")
    return rows


def _validate_spec(spec: Mapping[str, Any], population_audit: Mapping[str, Any]) -> dict[str, Any]:
    if spec.get("schema") != DESIGN_SCHEMA:
        raise DesignError("unsupported design specification schema")
    bands = [int(x) for x in spec.get("bands", [])]
    if not bands or len(set(bands)) != len(bands):
        raise DesignError("design bands must be unique and non-empty")
    audit_bands = [int(x) for x in population_audit.get("bands", [])]
    if bands != audit_bands:
        raise DesignError("design bands differ from the population audit")
    _hex64(spec.get("calibration_posterior_sha256"), "calibration_posterior_sha256")
    _hex64(spec.get("simulator_source_sha256"), "simulator_source_sha256")
    candidate_rows = spec.get("candidate_targets")
    if not isinstance(candidate_rows, list) or not candidate_rows:
        raise DesignError("candidate_targets must freeze every design candidate")
    candidates: dict[str, dict[str, int]] = {}
    for row in candidate_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("name"), str) \
                or not row["name"] or row["name"] in candidates:
            raise DesignError("candidate target names must be unique and non-empty")
        raw = row.get("targets")
        if not isinstance(raw, Mapping):
            raise DesignError("candidate targets must be mappings")
        targets = {str(band): int(raw.get(str(band), -1)) for band in bands}
        if any(value <= 0 for value in targets.values()):
            raise DesignError("every candidate must target every registered band")
        candidates[row["name"]] = targets
    scenarios = spec.get("scenarios")
    if not isinstance(scenarios, Mapping):
        raise DesignError("registered truth and misspecification scenarios are required")
    scenario_names: dict[str, set[str]] = {}
    for group in ("classical", "alternative", "predictive_valid",
                  "predictive_misspecification"):
        rows = scenarios.get(group)
        if not isinstance(rows, list) or not rows or any(
                not isinstance(name, str) or not name for name in rows):
            raise DesignError(f"scenarios.{group} must be a non-empty string list")
        if len(set(rows)) != len(rows):
            raise DesignError(f"scenarios.{group} contains duplicates")
        scenario_names[group] = set(rows)
    criteria = spec.get("criteria")
    if not isinstance(criteria, Mapping):
        raise DesignError("design criteria are required")
    alpha = _probability(criteria.get("type_i_error_max"), "type_i_error_max")
    power = _probability(criteria.get("power_min"), "power_min")
    if power <= 0.5:
        raise DesignError("power_min must exceed one half")
    confidence = _probability(
        criteria.get("monte_carlo_confidence"), "monte_carlo_confidence")
    eta = criteria.get("minimum_relevant_coherence")
    if isinstance(eta, bool) or not isinstance(eta, (int, float)) or not (0 < float(eta) <= 1):
        raise DesignError("minimum_relevant_coherence must lie in (0,1]")
    for field in ("simulation_trials_min", "predictive_draws_min", "refusal_trials_min"):
        value = criteria.get(field)
        if (isinstance(value, bool) or not isinstance(value, int) or value < 10):
            raise DesignError(f"criteria.{field} must be an integer of at least ten")
    analysis_contract = _validate_analysis_contract(spec.get("analysis_contract"), criteria)
    if analysis_contract["calibration_draws"]["posterior_sha256"] \
            != spec.get("calibration_posterior_sha256"):
        raise DesignError("analysis contract and design use different calibration posteriors")
    points = {str(name): float(value) for name, value in
              analysis_contract["models"]["coherence_points"].items()}
    boundary = float(analysis_contract["models"]["classical_boundary"])
    expected_classical = {name for name, value in points.items() if value <= boundary}
    expected_alternative = {name for name, value in points.items() if value >= float(eta)}
    if scenario_names["classical"] != expected_classical:
        raise DesignError("classical scenarios do not cover the registered classical grid")
    if scenario_names["alternative"] != expected_alternative:
        raise DesignError("alternative scenarios do not cover the minimum-relevant grid")
    if scenario_names["predictive_valid"] != set(points):
        raise DesignError("predictive-valid scenarios must cover every registered model")
    return {"bands": bands, "type_i": alpha, "power": power,
            "confidence": confidence, "criteria": criteria,
            "candidates": candidates, "scenarios": scenario_names,
            "analysis_contract": analysis_contract}


def _simulation_provenance(simulation: Mapping[str, Any], expected_id: str,
                           calibration_hash: str, simulator_hash: str) -> None:
    if simulation.get("schema") != SIMULATION_SCHEMA:
        raise DesignError("unsupported design simulation schema")
    if simulation.get("design_id") != expected_id:
        raise DesignError("simulation belongs to a different design specification")
    source = simulation.get("source")
    if not isinstance(source, Mapping):
        raise DesignError("simulation source provenance is required")
    if source.get("kind") != "independent_calibration_forward_simulation":
        raise DesignError("simulation is not an independent calibration forward simulation")
    if source.get("uses_analysis_acquisition_outcomes") is not False:
        raise DesignError("simulation must explicitly exclude analysis-acquisition outcomes")
    raw_roles = source.get("input_roles")
    if (not isinstance(raw_roles, list) or not raw_roles
            or any(not isinstance(role, str) or not role for role in raw_roles)
            or len(raw_roles) != len(set(raw_roles))):
        raise DesignError("simulation input_roles must be a non-empty unique string array")
    roles = set(raw_roles)
    forbidden = sorted(roles & _FORBIDDEN_ROLES)
    if forbidden:
        raise DesignError("simulation uses forbidden outcome-bearing inputs: " + ", ".join(forbidden))
    if _hex64(source.get("calibration_posterior_sha256"),
              "simulation calibration hash") != calibration_hash:
        raise DesignError("simulation calibration posterior differs from the design")
    if _hex64(source.get("simulator_source_sha256"),
              "simulator_source_sha256") != simulator_hash:
        raise DesignError("simulation source bytes differ from the frozen design")
    _hex64(source.get("simulation_bank_sha256"), "simulation_bank_sha256")


def _misspecification_rates(groups: Any, thresholds: Mapping[str, float],
                             confidence: float) -> list[dict[str, Any]]:
    if not isinstance(groups, list) or not groups:
        raise DesignError("misspecification predictive draws are required")
    out = []
    for group in groups:
        name = group.get("name") if isinstance(group, Mapping) else None
        if not isinstance(name, str) or not name:
            raise DesignError("misspecification group lacks a name")
        arrays = []
        for metric in METRICS:
            values = np.asarray(group.get(metric), dtype=float)
            if values.ndim != 1 or len(values) == 0 or np.any(~np.isfinite(values)):
                raise DesignError("misspecification metric draws are invalid")
            arrays.append(values)
        if len({len(values) for values in arrays}) != 1:
            raise DesignError("misspecification metric lengths differ")
        rejected = np.zeros(len(arrays[0]), dtype=bool)
        for metric, values in zip(METRICS, arrays, strict=True):
            rejected |= values > float(thresholds[f"{metric}_max"])
        lo, hi = one_sided_binomial_bounds(int(rejected.sum()), len(rejected), confidence)
        out.append({"name": name, "rejections": int(rejected.sum()),
                    "trials": int(len(rejected)), "rate": float(rejected.mean()),
                    "lower": lo, "upper": hi})
    return out


def evaluate_candidate(candidate: Mapping[str, Any], *, spec: Mapping[str, Any],
                       population_audit: Mapping[str, Any],
                       validated: Mapping[str, Any],
                       assurance_confidence: float) -> dict[str, Any]:
    bands = validated["bands"]
    criteria = validated["criteria"]
    confidence = _probability(assurance_confidence, "familywise assurance confidence")
    targets_raw = candidate.get("targets")
    if not isinstance(targets_raw, Mapping):
        raise DesignError("candidate targets are required")
    targets = {str(band): int(targets_raw.get(str(band), -1)) for band in bands}
    if any(value <= 0 for value in targets.values()):
        raise DesignError("candidate targets must be positive for every band")
    name = str(candidate.get("name", ""))
    if name not in validated["candidates"] or targets != validated["candidates"][name]:
        raise DesignError("simulation candidate differs from the frozen candidate lattice")

    null_rows = _rate_rows(
        candidate.get("classical_claims"), confidence=confidence,
        label="classical_claims", expected_names=validated["scenarios"]["classical"],
        minimum_trials=int(criteria["simulation_trials_min"]))
    alternative_rows = _rate_rows(candidate.get("alternative_successes"),
                                  confidence=confidence,
                                  label="alternative_successes",
                                  expected_names=validated["scenarios"]["alternative"],
                                  minimum_trials=int(criteria["simulation_trials_min"]))
    type_i_pass = all(row["upper"] <= validated["type_i"] for row in null_rows)
    power_pass = all(row["lower"] >= validated["power"] for row in alternative_rows)

    valid_draws = _predictive_groups(
        candidate.get("predictive_valid_draws"), label="valid predictive draws",
        expected_names=validated["scenarios"]["predictive_valid"],
        minimum_draws=int(criteria["predictive_draws_min"]))
    threshold_report = predictive_thresholds(
        valid_draws,
        family_false_failure=_probability(
            criteria.get("predictive_family_false_failure_max"),
            "predictive_family_false_failure_max"),
        confidence=confidence,
    )
    misspec_draws = _predictive_groups(
        candidate.get("predictive_misspecification_draws"),
        label="misspecification draws",
        expected_names=validated["scenarios"]["predictive_misspecification"],
        minimum_draws=int(criteria["predictive_draws_min"]))
    misspecification = _misspecification_rates(
        misspec_draws,
        threshold_report["thresholds"], confidence)
    misspec_min = _probability(
        criteria.get("predictive_misspecification_rejection_min"),
        "predictive_misspecification_rejection_min")
    misspecification_pass = all(row["lower"] >= misspec_min for row in misspecification)

    refusal = candidate.get("refusal_studies")
    if not isinstance(refusal, Mapping):
        raise DesignError("candidate refusal studies are required")
    family_delta = _probability(
        criteria.get("reserve_family_failure_max"), "reserve_family_failure_max")
    strata = int(population_audit.get("n_strata", 0))
    reserve_rows: dict[str, Any] = {}
    reserves: dict[str, int] = {}
    for band in bands:
        study = refusal.get(str(band))
        if not isinstance(study, Mapping):
            raise DesignError(f"refusal study missing band {band}")
        if (not isinstance(study.get("trials"), int)
                or isinstance(study.get("trials"), bool)
                or int(study["trials"]) < int(criteria["refusal_trials_min"])):
            raise DesignError(f"refusal study trials are insufficient for band {band}")
        _, q_upper = one_sided_binomial_bounds(
            study.get("refusals"), study.get("trials"), confidence)
        populations = population_audit["eligible_by_band_stratum"][str(band)]
        reserve = reserve_total_for_band(
            populations, targets[str(band)], q_upper,
            family_delta / len(bands),
        )
        reserve_rows[str(band)] = {
            "refusals": int(study["refusals"]), "trials": int(study["trials"]),
            "probability_upper": q_upper, **reserve,
        }
        reserves[str(band)] = int(reserve["reserve_total"])
    if strata < 2:
        raise DesignError("population audit needs at least two strata")

    costs = spec.get("event_cost", {})
    if not isinstance(costs, Mapping):
        raise DesignError("event_cost is required")
    total_cost = 0.0
    for band in bands:
        cost = costs.get(str(band))
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or float(cost) <= 0:
            raise DesignError(f"event cost missing/invalid for band {band}")
        total_cost += float(cost) * (targets[str(band)] + reserves[str(band)])
    pass_all = bool(type_i_pass and power_pass and misspecification_pass)
    return {
        "name": name,
        "targets": targets, "reserves": reserves,
        "estimated_compute_cost": float(total_cost),
        "classical_claims": null_rows, "alternative_successes": alternative_rows,
        "predictive_thresholds": threshold_report,
        "predictive_misspecification": misspecification,
        "reserve_design": reserve_rows,
        "checks": {"type_i": type_i_pass, "power": power_pass,
                   "predictive_misspecification": misspecification_pass},
        "pass": pass_all,
    }


def build_report(spec: Mapping[str, Any], population_audit: Mapping[str, Any],
                 simulation: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all candidates and choose the deterministic minimum-cost design."""
    validated = _validate_spec(spec, population_audit)
    identifier = design_id(spec, population_audit)
    calibration_hash = _hex64(
        spec.get("calibration_posterior_sha256"), "calibration_posterior_sha256")
    simulator_hash = _hex64(spec.get("simulator_source_sha256"),
                            "simulator_source_sha256")
    _simulation_provenance(simulation, identifier, calibration_hash, simulator_hash)
    candidates_raw = simulation.get("candidates")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise DesignError("simulation contains no candidate designs")
    simulated_names = [str(row.get("name")) for row in candidates_raw
                       if isinstance(row, Mapping)]
    if (len(simulated_names) != len(candidates_raw)
            or len(set(simulated_names)) != len(simulated_names)
            or set(simulated_names) != set(validated["candidates"])):
        raise DesignError("simulation must evaluate every frozen candidate exactly once")
    bound_families_per_candidate = (
        len(validated["scenarios"]["classical"])
        + len(validated["scenarios"]["alternative"])
        + 1  # predictive-valid thresholds are internally simultaneous
        + len(validated["scenarios"]["predictive_misspecification"])
        + len(validated["bands"])
    )
    simultaneous_bound_families = len(candidates_raw) * bound_families_per_candidate
    global_confidence = float(validated["confidence"])
    assurance_confidence = 1.0 - (1.0 - global_confidence) \
        / simultaneous_bound_families
    candidates = [evaluate_candidate(candidate, spec=spec,
                                     population_audit=population_audit,
                                     validated=validated,
                                     assurance_confidence=assurance_confidence)
                  for candidate in candidates_raw]
    feasible = [row for row in candidates if row["pass"]]
    bands = validated["bands"]
    feasible.sort(key=lambda row: (
        row["estimated_compute_cost"],
        tuple(row["targets"][str(band)] for band in bands),
        tuple(row["reserves"][str(band)] for band in bands),
        row["name"],
    ))
    selected = feasible[0] if feasible else None
    body = {
        "schema": REPORT_SCHEMA,
        "design_id": identifier,
        "design_spec_sha256": sha256_json(spec),
        "population_audit_sha256": population_audit["audit_payload_sha256"],
        "calibration_posterior_sha256": calibration_hash,
        "simulation_source": simulation["source"],
        "analysis_contract": validated["analysis_contract"],
        "criteria": spec["criteria"],
        "monte_carlo_assurance": {
            "method": "Bonferroni across all candidates and stochastic bound families",
            "global_confidence": global_confidence,
            "simultaneous_bound_families": simultaneous_bound_families,
            "per_family_confidence": assurance_confidence,
            "predictive_valid_family": (
                "additional Bonferroni across registered models and metrics"),
        },
        "candidates": candidates,
        "selection_rule": (
            "minimum estimated compute cost; then lexicographic targets, reserves, name"
        ),
        "selected": selected,
        "pass": selected is not None,
        "status": "feasible_design_selected" if selected is not None else "no_feasible_design",
    }
    return {**body, "design_report_payload_sha256": sha256_json(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonicalize-spec", action="store_true",
                        help="create canonical JSON from a fully resolved design spec")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--population-audit", type=Path)
    parser.add_argument("--simulation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.canonicalize_spec:
        if args.population_audit is not None or args.simulation is not None:
            parser.error("--canonicalize-spec accepts only --spec and --out")
        spec = canonicalize_resolved_spec(args.spec, args.out)
        print(f"SHA256 {sha256_json(spec)}  {args.out}")
        return 0
    if args.population_audit is None or args.simulation is None:
        parser.error("report generation requires --population-audit and --simulation")
    spec = load_canonical_spec(args.spec)
    audit = json.loads(args.population_audit.read_text())
    simulation = json.loads(args.simulation.read_text())
    report = build_report(spec, audit, simulation)
    write_canonical_json(args.out, report)
    print(f"pass={str(report['pass']).lower()}")
    print(f"SHA256 {sha256_json(report)}  {args.out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
