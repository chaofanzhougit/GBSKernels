"""Registration and hashing primitives for the v2 confirmatory selector.

The v2 selector separates two records that are often conflated:

* a public, immutable analysis plan that is frozen before the randomness beacon
  can publish its value; and
* a resolved registration containing the beacon proof and the seed derived from
  it.

This module deliberately performs no network access.  A caller obtains the
public registration and beacon proof through an independently auditable channel,
then passes the exact JSON objects here for deterministic validation.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse


SCHEMA = "gbskernels.confirmatory.v2"
SEED_DOMAIN = "GBSKERNELS/CONFIRMATORY/V2/SEED"
KEY_DOMAIN = "GBSKERNELS/CONFIRMATORY/V2/EVENT"
_HEX64 = re.compile(r"\A[0-9a-fA-F]{64}\Z")
_CONTAINER_DIGEST = re.compile(r"\A[^\s@]+(?:/[^\s@]+)*@sha256:[0-9a-fA-F]{64}\Z")


class ContractError(ValueError):
    """Raised when a registration is missing an auditable invariant."""


def canonical_json(value: Any) -> str:
    """Return the one canonical JSON representation used for all hashes.

    NaN/Infinity, implicit whitespace, and non-ASCII escaping are intentionally
    rejected or fixed so two implementations cannot hash different bytes for
    the same registration.
    """
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not canonicalizable JSON: {exc}") from exc


def canonical_bytes(value: Any) -> bytes:
    """UTF-8 bytes of :func:`canonical_json`."""
    return canonical_json(value).encode("utf-8")


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    """Lower-case SHA256 digest for bytes."""
    return hashlib.sha256(bytes(data)).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hash a file without loading it into memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    """SHA256 of canonical JSON bytes."""
    return sha256_bytes(canonical_bytes(value))


def _parse_utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{field} must be an ISO-8601 UTC string")
    text = value
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ContractError(f"{field} is not valid ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractError(f"{field} must include an explicit UTC offset")
    parsed = parsed.astimezone(timezone.utc)
    return parsed


def utc_timestamp(value: Any, field: str = "timestamp_utc") -> str:
    """Normalize an aware timestamp to ``...Z`` while validating UTC data."""
    parsed = _parse_utc(value, field)
    return parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _require_hex(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ContractError(f"{field} must be a 64-character SHA256 hex digest")
    return value.lower()


def _public_section(registration: Mapping[str, Any]) -> Mapping[str, Any]:
    public = registration.get("public")
    if not isinstance(public, Mapping):
        # Accept the flat spelling as an input convenience, but normalize it to
        # one schema before any hash is computed.
        public = {
            "url": registration.get("public_url"),
            "sha256": registration.get("public_sha256"),
            "timestamp_utc": registration.get("registered_at_utc"),
            "immutable": registration.get("public_immutable", False),
        }
    return public


def _beacon_section(registration: Mapping[str, Any]) -> Mapping[str, Any]:
    beacon = registration.get("beacon")
    if not isinstance(beacon, Mapping):
        raise ContractError("registration.beacon must be an object")
    return beacon


def beacon_payload(source: str, round_number: int, value: Any) -> dict[str, Any]:
    """Canonical payload whose digest is covered by a beacon proof."""
    if not isinstance(source, str) or not source:
        raise ContractError("beacon source must be a non-empty string")
    if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 0:
        raise ContractError("beacon round must be a non-negative integer")
    return {"source": source, "round": round_number, "value": value}


def beacon_payload_sha256(source: str, round_number: int, value: Any) -> str:
    """Digest used by the portable, source/round/value beacon proof."""
    return sha256_json(beacon_payload(source, round_number, value))


def verify_beacon_proof(beacon: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a self-contained beacon source/round/value proof.

    This verifies the binding supplied in the registration.  It does not claim
    to verify a provider's signature or contact a provider; a provider-specific
    signature can be retained under ``proof.signature`` and audited separately.
    """
    source = beacon.get("source")
    round_number = beacon.get("round")
    if not isinstance(source, str) or not source:
        raise ContractError("beacon.source must be non-empty")
    if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 0:
        raise ContractError("beacon.round must be a non-negative integer")
    if "value" not in beacon:
        raise ContractError("beacon.value is required after beacon resolution")
    expected = beacon_payload_sha256(source, round_number, beacon["value"])
    proof = beacon.get("proof")
    if not isinstance(proof, Mapping):
        raise ContractError("beacon.proof must bind source, round, and value")
    proof_source = proof.get("source", source)
    proof_round = proof.get("round", round_number)
    if proof_source != source or proof_round != round_number:
        raise ContractError("beacon proof source/round does not match beacon")
    supplied = proof.get("payload_sha256", proof.get("sha256"))
    if _require_hex(supplied, "beacon.proof.payload_sha256") != expected:
        raise ContractError("beacon proof digest does not match source/round/value")
    # A hash of source/round/value alone is not evidence of an external
    # beacon: it can be generated by whoever assembled this JSON.  Require an
    # independently retrievable provider record and bind its content hash.
    proof_url = proof.get("url", proof.get("proof_url"))
    if not isinstance(proof_url, str):
        raise ContractError("beacon proof URL is required")
    parsed = urlparse(proof_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ContractError("beacon proof URL must be HTTPS")
    record_sha256 = _require_hex(
        proof.get("record_sha256", proof.get("record_content_sha256")),
        "beacon.proof.record_sha256",
    )
    return {
        "source": source,
        "round": round_number,
        "value": beacon["value"],
        "payload_sha256": expected,
        "url": proof_url,
        "record_sha256": record_sha256,
    }


def derive_seed(
    *,
    public_url: str,
    public_sha256: str,
    published_at_utc: str,
    beacon_source: str,
    beacon_round: int,
    beacon_value: Any,
    domain: str = SEED_DOMAIN,
) -> dict[str, Any]:
    """Derive a reproducible 256-bit seed from the locked plan and beacon.

    The returned integer is the first 64 bits for APIs that need a bounded RNG
    seed; ``seed_hex`` preserves all 256 bits for auditability.
    """
    public_sha256 = _require_hex(public_sha256, "public_sha256")
    published = utc_timestamp(published_at_utc, "published_at_utc")
    if not isinstance(public_url, str) or not public_url:
        raise ContractError("public_url must be non-empty")
    if not isinstance(domain, str) or not domain:
        raise ContractError("seed domain must be non-empty")
    payload_hash = beacon_payload_sha256(beacon_source, beacon_round, beacon_value)
    material = {
        "domain": domain,
        "public_url": public_url,
        "public_sha256": public_sha256,
        "published_at_utc": published,
        "beacon_source": beacon_source,
        "beacon_round": beacon_round,
        "beacon_payload_sha256": payload_hash,
    }
    digest = hashlib.sha256(domain.encode("ascii") + b"\0" + canonical_bytes(material)).digest()
    return {
        "algorithm": "sha256",
        "domain": domain,
        "material": material,
        "seed_hex": digest.hex(),
        "seed_uint64": int.from_bytes(digest[:8], "big"),
    }


def registration_plan(registration: Mapping[str, Any]) -> Any:
    """Return the frozen plan object covered by the public hash."""
    plan = registration.get("plan")
    if plan is None:
        plan = registration.get("analysis_plan")
    if plan is None:
        raise ContractError("registration.plan is required")
    return plan


def validate_registration(
    registration: Mapping[str, Any],
    *,
    now_utc: str | datetime | None = None,
    require_beacon: bool = True,
) -> dict[str, Any]:
    """Validate and normalize a resolved v2 registration.

    ``public.timestamp_utc`` must precede the beacon's availability timestamp;
    equality is rejected.  ``require_beacon=False`` validates a pre-beacon plan
    and requires that the advertised beacon availability is still in the future
    relative to the public registration timestamp.
    """
    if not isinstance(registration, Mapping):
        raise ContractError("registration must be a JSON object")
    if registration.get("schema") not in (None, SCHEMA):
        raise ContractError(f"unsupported registration schema: {registration.get('schema')!r}")
    public = _public_section(registration)
    url = public.get("url")
    parsed = urlparse(str(url))
    if parsed.scheme != "https" or not parsed.netloc:
        raise ContractError("public.url must be an HTTPS URL")
    if public.get("immutable") is not True:
        raise ContractError("public.immutable must be true for registration")
    public_hash = _require_hex(public.get("sha256"), "public.sha256")
    published = utc_timestamp(public.get("timestamp_utc"), "public.timestamp_utc")
    timestamp_proof_url = public.get("timestamp_proof_url")
    if not isinstance(timestamp_proof_url, str):
        raise ContractError("public.timestamp_proof_url is required")
    timestamp_url = urlparse(timestamp_proof_url)
    if timestamp_url.scheme != "https" or not timestamp_url.netloc:
        raise ContractError("public.timestamp_proof_url must be HTTPS")
    timestamp_proof_sha256 = _require_hex(
        public.get("timestamp_proof_sha256"), "public.timestamp_proof_sha256")

    plan = registration_plan(registration)
    if not isinstance(plan, Mapping):
        raise ContractError("registration.plan must be an object")
    for field in ("analysis_commit", "selection", "models", "analysis",
                  "numerical_contract", "prior_data_use", "external_requirements"):
        if field not in plan:
            raise ContractError(f"registration.plan lacks {field}")
    if (not isinstance(plan["analysis_commit"], str)
            or re.fullmatch(r"[0-9a-fA-F]{7,64}", plan["analysis_commit"]) is None):
        raise ContractError("analysis_commit must be a hexadecimal commit identifier")
    selection = plan["selection"]
    if not isinstance(selection, Mapping):
        raise ContractError("plan.selection must be an object")
    randomness = plan.get("randomness_beacon")
    if not isinstance(randomness, Mapping):
        raise ContractError("plan.randomness_beacon is required")
    if not isinstance(randomness.get("provider"), str) or not randomness["provider"]:
        raise ContractError("plan.randomness_beacon.provider is required")
    beacon_source = randomness.get("source")
    beacon_round = randomness.get("round")
    if not isinstance(beacon_source, str) or not beacon_source:
        raise ContractError("plan.randomness_beacon.source is required")
    if isinstance(beacon_round, bool) or not isinstance(beacon_round, int) or beacon_round < 0:
        raise ContractError("plan.randomness_beacon.round must be non-negative")
    bands = selection.get("bands")
    if not isinstance(bands, list) or not bands:
        raise ContractError("plan.selection.bands must be a non-empty list")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in bands):
        raise ContractError("selection bands must be integers")
    try:
        band_values = [int(x) for x in bands]
    except (TypeError, ValueError) as exc:
        raise ContractError("selection bands must be integers") from exc
    if (len(set(band_values)) != len(band_values)
            or any(x <= 0 or x > 100 for x in band_values)):
        raise ContractError("selection bands must be unique integers in [1,100]")
    for field in ("n_records", "record_bytes", "n_strata"):
        value = selection.get(field)
        if (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ContractError(f"selection.{field} must be a positive integer")
    if int(selection["record_bytes"]) != 16:
        raise ContractError("selection.record_bytes must be exactly 16")
    try:
        raw_targets = selection["targets"]
        raw_reserves = selection["reserves"]
        if not isinstance(raw_targets, Mapping) or not isinstance(raw_reserves, Mapping):
            raise TypeError
        if any(isinstance(v, bool) or not isinstance(v, int)
               for v in list(raw_targets.values()) + list(raw_reserves.values())):
            raise TypeError
        targets = {str(int(k)): int(v) for k, v in raw_targets.items()}
        reserves = {str(int(k)): int(v) for k, v in raw_reserves.items()}
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("selection targets/reserves are invalid") from exc
    if set(targets) != {str(x) for x in band_values} or set(reserves) != set(targets):
        raise ContractError("selection band/target/reserve keys do not agree")
    if any(v <= 0 for v in targets.values()) or any(v < 0 for v in reserves.values()):
        raise ContractError("selection targets must be positive and reserves nonnegative")
    if int(selection["n_strata"]) < 2:
        raise ContractError("at least two common acquisition strata are required")
    for field in ("source_raw_sha256", "exclusion_sha256",
                  "exclusion_ledger_sha256", "population_audit_sha256"):
        _require_hex(selection.get(field), f"selection.{field}")
    models = plan["models"]
    if not isinstance(models, Mapping):
        raise ContractError("plan.models must be an object")
    if (isinstance(models.get("exp_id"), bool)
            or not isinstance(models.get("exp_id"), int)
            or int(models["exp_id"]) != 0):
        raise ContractError("Jiuzhang 1.0 confirmatory v2 requires models.exp_id = 0")
    points = models.get("coherence_points")
    if not isinstance(points, Mapping) or len(points) < 2:
        raise ContractError("at least two registered coherence points are required")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           for value in points.values()):
        raise ContractError("coherence point coordinates must be numeric")
    reference, alternative = models.get("reference_model"), models.get("alternative_model")
    if (not isinstance(reference, str) or not reference
            or not isinstance(alternative, str) or not alternative):
        raise ContractError("model reference/alternative names must be non-empty strings")
    if reference == alternative or reference not in points or alternative not in points:
        raise ContractError("model reference/alternative is not in coherence_points")
    try:
        coordinates = [float(value) for value in points.values()]
    except (TypeError, ValueError) as exc:
        raise ContractError("coherence point coordinates must be numeric") from exc
    if len(set(coordinates)) != len(coordinates) or any(not (-1 <= x <= 1) for x in coordinates):
        raise ContractError("coherence coordinates must be unique and lie in [-1,1]")
    parameterization = models.get("parameterization", "classical_excess")
    if parameterization not in {"classical_excess", "physical_fraction"}:
        raise ContractError("unsupported coherence parameterization")
    if parameterization == "physical_fraction" and any(x < 0 for x in coordinates):
        raise ContractError("physical_fraction coherence coordinates must be nonnegative")
    if "classical_boundary" in models:
        boundary = float(models["classical_boundary"])
        if not (-1 <= boundary <= 1):
            raise ContractError("classical_boundary must lie in [-1,1]")
        if parameterization == "classical_excess" and boundary != 0.0:
            raise ContractError(
                "classical_excess parameterization has a fixed classical boundary at zero"
            )
    analysis = plan["analysis"]
    if not isinstance(analysis, Mapping):
        raise ContractError("plan.analysis must be an object")
    estimands = analysis.get("estimands")
    if not isinstance(estimands, list) or analysis.get("primary_estimand") not in estimands:
        raise ContractError("analysis primary_estimand is not registered")
    if analysis.get("primary_estimand") \
            != "best_predictive_anomalous_coherence_grid_point":
        raise ContractError("confirmatory v2 requires the coherence-grid primary estimand")
    if (analysis.get("primary_estimand")
            == "best_predictive_anomalous_coherence_grid_point" and len(points) < 3):
        raise ContractError("coherence-grid primary estimand requires at least three points")
    if (analysis.get("primary_estimand")
            == "best_predictive_anomalous_coherence_grid_point"
            and parameterization != "classical_excess"):
        raise ContractError(
            "coherence-grid primary inference requires classical_excess coordinates"
        )
    _require_hex(analysis.get("design_report_sha256"),
                 "analysis.design_report_sha256")
    minimum_relevant = analysis.get("minimum_relevant_coherence")
    if (isinstance(minimum_relevant, bool)
            or not isinstance(minimum_relevant, (int, float))
            or not (float(models.get("classical_boundary", 0.0))
                    < float(minimum_relevant) <= 1.0)):
        raise ContractError("minimum_relevant_coherence must be above the classical boundary")
    if (analysis.get("primary_estimand")
            == "best_predictive_anomalous_coherence_grid_point"
            and float(minimum_relevant) not in coordinates):
        raise ContractError("minimum_relevant_coherence must be a registered model coordinate")
    boundary = float(models.get("classical_boundary", 0.0))
    if float(points[reference]) != boundary or float(points[alternative]) <= boundary:
        raise ContractError(
            "reference_model must be the classical boundary and alternative_model nonclassical")
    target_power = analysis.get("target_power")
    if (isinstance(target_power, bool) or not isinstance(target_power, (int, float))
            or not (0.5 < float(target_power) < 1.0)):
        raise ContractError("target_power must lie in (0.5,1)")
    type_i_error = analysis.get("type_i_error_max")
    if (isinstance(type_i_error, bool)
            or not isinstance(type_i_error, (int, float))
            or not (0.0 < float(type_i_error) < 1.0)):
        raise ContractError("type_i_error_max must lie in (0,1)")
    mc_confidence = analysis.get("monte_carlo_confidence")
    if (isinstance(mc_confidence, bool)
            or not isinstance(mc_confidence, (int, float))
            or not (0.9 <= float(mc_confidence) < 1.0)):
        raise ContractError("monte_carlo_confidence must lie in [0.9,1)")
    decision = analysis.get("primary_decision_rule")
    expected_decision = {
        "method": "simultaneous_paired_model_score_max_t",
        "claim_if_confidence_set_above_classical_boundary": True,
        "require_predictive_pass_for_all_confidence_set_models": True,
        "report_failure_without_suppressing_analysis": True,
    }
    if decision != expected_decision:
        raise ContractError("primary_decision_rule is not the fail-closed v2 rule")
    weights = analysis.get("band_weights")
    if not isinstance(weights, Mapping) or set(map(str, band_values)) != set(weights):
        raise ContractError("analysis band weights do not match bands")
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           for x in weights.values()):
        raise ContractError("analysis band weights must be numeric")
    numeric_weights = [float(x) for x in weights.values()]
    if any(not (x > 0) for x in numeric_weights):
        raise ContractError("analysis band weights must be positive")
    if abs(sum(numeric_weights) - 1.0) > 1e-12:
        raise ContractError("analysis band weights must sum to one")
    if (isinstance(analysis.get("bootstrap_reps"), bool)
            or not isinstance(analysis.get("bootstrap_reps"), int)
            or int(analysis["bootstrap_reps"]) < 100):
        raise ContractError("bootstrap_reps must be at least 100")
    if (isinstance(analysis.get("bootstrap_seed"), bool)
            or not isinstance(analysis.get("bootstrap_seed"), int)):
        raise ContractError("bootstrap_seed must be an integer")
    if not (0 < float(analysis.get("alpha", 0)) < 1):
        raise ContractError("analysis alpha must lie in (0,1)")
    if float(analysis["alpha"]) != float(type_i_error):
        raise ContractError("analysis alpha must equal type_i_error_max")
    refusal = analysis.get("refusal_analysis")
    refusal_fields = {
        "method", "reps", "seed", "alpha", "inferential_gate",
        "recovery_method", "minimum_precision_bits",
        "recovery_source_sha256", "recovery_container_digest",
    }
    if (not isinstance(refusal, Mapping)
            or set(refusal) != refusal_fields
            or refusal.get("method")
            != "within_fixed_band_stratum_max_abs_mean_score_difference_permutation"
            or refusal.get("inferential_gate") is not False
            or refusal.get("recovery_method")
            != "independent_high_precision_interval_reevaluation"
            or isinstance(refusal.get("minimum_precision_bits"), bool)
            or not isinstance(refusal.get("minimum_precision_bits"), int)
            or int(refusal["minimum_precision_bits"]) < 128
            or isinstance(refusal.get("reps"), bool)
            or not isinstance(refusal.get("reps"), int)
            or int(refusal["reps"]) < 100
            or isinstance(refusal.get("seed"), bool)
            or not isinstance(refusal.get("seed"), int)):
        raise ContractError("analysis.refusal_analysis is not a valid frozen v2 contract")
    _require_hex(refusal.get("recovery_source_sha256"),
                 "analysis.refusal_analysis.recovery_source_sha256")
    if (not isinstance(refusal.get("recovery_container_digest"), str)
            or _CONTAINER_DIGEST.fullmatch(refusal["recovery_container_digest"]) is None):
        raise ContractError("refusal recovery requires a pinned container digest")
    refusal_alpha = refusal.get("alpha")
    if (isinstance(refusal_alpha, bool)
            or not isinstance(refusal_alpha, (int, float))
            or float(refusal_alpha) != float(analysis["alpha"])):
        raise ContractError("refusal-analysis alpha must equal analysis alpha")
    if analysis.get("resampling_unit") != "event_within_fixed_common_stratum":
        raise ContractError(
            "analysis.resampling_unit must be event_within_fixed_common_stratum")
    if analysis.get("resampling_method") \
            != "calibration_draw_conditioned_fixed_strata_srswor_fpc_max_t":
        raise ContractError("analysis.resampling_method is not the registered v2 method")
    if analysis.get("population_scope") != "finite_registered_acquisition":
        raise ContractError(
            "v2 currently supports only the finite_registered_acquisition scope"
        )
    prior = plan["prior_data_use"]
    expected_prior = {
        "record_level_policy":
            "exclude_all_materialized_published_selected_or_scored_records",
        "outcome_blind_mechanical_scans_are_not_record_level_exposure": True,
        "full_acquisition_aggregates_previously_examined": True,
        "independent_acquisition": False,
        "claim_scope":
            "finite_registered_acquisition_conditional_on_prior_aggregate_use",
        "author_attestation_required": True,
    }
    if prior != expected_prior:
        raise ContractError("prior_data_use must disclose the finite-acquisition limitations")
    external = plan["external_requirements"]
    if not isinstance(external, Mapping):
        raise ContractError("plan.external_requirements must be an object")
    for required_flag in ("public_registration_required", "future_beacon_required",
                          "joint_normalizer_replicates_required",
                          "container_digest_required",
                          "absolute_predictive_checks_required"):
        if external.get(required_flag) is not True:
            raise ContractError(f"external_requirements.{required_flag} must be true")
    normalizer = analysis.get("normalizer_replicates")
    if not isinstance(normalizer, Mapping):
        raise ContractError("analysis.normalizer_replicates is required")
    if (isinstance(normalizer.get("count"), bool)
            or not isinstance(normalizer.get("count"), int)
            or int(normalizer["count"]) < 2
            or isinstance(normalizer.get("samples_per_replicate"), bool)
            or not isinstance(normalizer.get("samples_per_replicate"), int)
            or int(normalizer["samples_per_replicate"]) < 1
            or isinstance(normalizer.get("seed"), bool)
            or not isinstance(normalizer.get("seed"), int)):
        raise ContractError("normalizer replicate count/samples/seed are invalid")
    if external.get("reconstruction_required", False):
        calibration = analysis.get("calibration_draws")
        if not isinstance(calibration, Mapping):
            raise ContractError("required reconstruction needs analysis.calibration_draws")
        _require_hex(calibration.get("posterior_sha256"),
                     "analysis.calibration_draws.posterior_sha256")
        if (isinstance(calibration.get("count"), bool)
                or not isinstance(calibration.get("count"), int)
                or int(calibration["count"]) < 2):
            raise ContractError("calibration_draws.count must be at least two")
        if (isinstance(calibration.get("seed"), bool)
                or not isinstance(calibration.get("seed"), int)):
            raise ContractError("calibration_draws.seed must be an integer")
        nuisances = set(calibration.get("required_nuisances", []))
        if not {"squeezing", "transfer", "loss", "block_drift"}.issubset(nuisances):
            raise ContractError("calibration_draws must register all physical nuisance families")
        if calibration.get("dark_click_model") != "explicitly_zero":
            raise ContractError("only an explicitly zero dark-click model is currently supported")
    if external.get("absolute_predictive_checks_required", False):
        predictive = analysis.get("predictive_checks")
        if not isinstance(predictive, Mapping):
            raise ContractError("analysis.predictive_checks is required")
        pairs = predictive.get("detector_pairs")
        if not isinstance(pairs, list) or not pairs:
            raise ContractError("predictive detector_pairs must be non-empty")
        for pair in pairs:
            if (not isinstance(pair, list) or len(pair) != 2
                    or any(isinstance(x, bool) or not isinstance(x, int) for x in pair)
                    or not (0 <= pair[0] < pair[1] < 100)):
                raise ContractError("predictive detector_pairs contain an invalid pair")
        thresholds = predictive.get("thresholds")
        if not isinstance(thresholds, Mapping):
            raise ContractError("predictive thresholds are required")
        policy = predictive.get("model_pass_policy")
        if policy != "all_plausible_best":
            raise ContractError(
                "predictive model-pass policy must cover every plausible best model")
        for key in ("click_count_tv_max", "marginal_rms_max", "pair_covariance_rms_max"):
            if not (float(thresholds.get(key, -1)) > 0):
                raise ContractError(f"predictive threshold {key} must be positive")
    numerical = plan["numerical_contract"]
    if not isinstance(numerical, Mapping) or not numerical.get("scope"):
        raise ContractError("numerical_contract.scope is required")
    if not isinstance(numerical.get("state_fingerprints"), Mapping):
        raise ContractError("numerical_contract.state_fingerprints is required")
    _require_hex(numerical.get("analysis_source_sha256"),
                 "numerical_contract.analysis_source_sha256")
    plan_hash = sha256_json(plan)
    # The public digest must be the digest of the exact frozen plan.  Allowing
    # an unrelated archive-record digest would let the seed be changed after
    # observing the beacon.
    if public_hash != plan_hash:
        raise ContractError("public.sha256 must equal the canonical plan hash")
    declared_plan_hash = public.get("plan_sha256", public_hash)
    if _require_hex(declared_plan_hash, "public.plan_sha256") != plan_hash:
        raise ContractError("public plan hash does not match registration.plan")

    beacon = _beacon_section(registration)
    availability_value = beacon.get("availability_utc", beacon.get("expected_availability_utc"))
    availability = utc_timestamp(availability_value, "beacon.availability_utc")
    if beacon.get("source") != beacon_source or beacon.get("round") != beacon_round:
        raise ContractError("resolved beacon does not match the frozen beacon source/round")
    frozen_availability = utc_timestamp(
        randomness.get("availability_utc", availability),
        "plan.randomness_beacon.availability_utc",
    )
    if availability != frozen_availability:
        raise ContractError(
            "resolved beacon availability differs from frozen plan; "
            "registration must precede beacon availability strictly"
        )
    published_dt = _parse_utc(published, "public.timestamp_utc")
    availability_dt = _parse_utc(availability, "beacon.availability_utc")
    if not published_dt < availability_dt:
        raise ContractError("public registration must precede beacon availability strictly")

    if now_utc is not None:
        now_dt = _parse_utc(now_utc, "now_utc") if isinstance(now_utc, str) else now_utc
        if now_dt.tzinfo is None or now_dt.utcoffset() is None:
            raise ContractError("now_utc must be timezone-aware")
        now_dt = now_dt.astimezone(timezone.utc)
    else:
        now_dt = None

    if not require_beacon:
        if now_dt is not None and not now_dt < availability_dt:
            raise ContractError("pre-beacon validation requires availability in the future")
        return {
            "schema": SCHEMA,
            "plan": plan,
            "public": {
                "url": str(url),
                "sha256": public_hash,
                "plan_sha256": sha256_json(plan),
                "timestamp_utc": published,
                "immutable": True,
                "timestamp_proof_url": timestamp_proof_url,
                "timestamp_proof_sha256": timestamp_proof_sha256,
            },
            "beacon": {**dict(beacon), "availability_utc": availability},
        }

    proof = verify_beacon_proof(beacon)
    derived = derive_seed(
        public_url=str(url),
        public_sha256=public_hash,
        published_at_utc=published,
        beacon_source=proof["source"],
        beacon_round=proof["round"],
        beacon_value=proof["value"],
        domain=str(registration.get("seed_derivation", {}).get("domain", SEED_DOMAIN)),
    )
    declared = registration.get("seed_derivation")
    if not isinstance(declared, Mapping):
        raise ContractError("seed_derivation is required")
    if declared.get("algorithm") != "sha256" or declared.get("domain") != derived["domain"]:
        raise ContractError("seed_derivation algorithm/domain mismatch")
    if declared.get("seed_hex") != derived["seed_hex"]:
        raise ContractError("declared seed_hex does not match deterministic derivation")
    if int(declared.get("seed_uint64", -1)) != derived["seed_uint64"]:
        raise ContractError("declared seed_uint64 does not match deterministic derivation")
    return {
        "schema": SCHEMA,
        "plan": plan,
        "public": {
            "url": str(url),
            "sha256": public_hash,
            "plan_sha256": sha256_json(plan),
            "timestamp_utc": published,
            "immutable": True,
            "timestamp_proof_url": timestamp_proof_url,
            "timestamp_proof_sha256": timestamp_proof_sha256,
        },
        "beacon": {**dict(beacon), "availability_utc": availability, "proof": proof},
        "seed_derivation": derived,
        "seed": derived["seed_uint64"],
    }


def load_registration(path: str | Path, *, require_beacon: bool = True) -> dict[str, Any]:
    """Load canonical JSON and validate it."""
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read registration {path}: {exc}") from exc
    if raw != canonical_bytes(value):
        raise ContractError("registration file is not canonical JSON")
    return validate_registration(value, require_beacon=require_beacon)


def write_canonical_json(path: str | Path, value: Any) -> str:
    """Atomically create canonical JSON without replacing an existing record."""
    data = canonical_bytes(value)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.link(temporary, target)
        temporary.unlink()
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return sha256_bytes(data)


def event_key(record_index: int, seed: int | str, *, domain: str = KEY_DOMAIN) -> str:
    """Domain-separated full-width event ranking key."""
    if isinstance(record_index, bool) or not isinstance(record_index, int) or record_index < 0:
        raise ContractError("record_index must be a non-negative integer")
    if not isinstance(seed, (int, str)):
        raise ContractError("seed must be an integer or string")
    material = {"domain": domain, "record_index": record_index, "seed": str(seed)}
    return sha256_bytes(domain.encode("ascii") + b"\0" + canonical_bytes(material))


# Explicit aliases make the small utility surface easy to discover in scripts.
canonical_sha256 = sha256_json
derive_selection_seed = derive_seed
validate_contract = validate_registration
