"""Design-based confirmatory inference for content-addressed Jiuzhang runs.

The primary analysis consumes immutable per-event log likelihoods, joint raw
normalizer replicates, and (when the registered claim is reconstruction
marginal) coherent reconstruction replicates.  Deterministic arithmetic
half-widths widen the final confidence set; they are never treated as noise.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from confirmatory_common import (analysis_source_hash, current_commit, hash_json, load_json,
                                 sha256_file, valid_container_digest,
                                 write_json_exclusive)  # noqa: E402
from confirmatory_contract import validate_registration  # noqa: E402
from analyze_refusals import validate_refusal_analysis  # noqa: E402


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def horvitz_thompson_band(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Estimate a finite-population band mean with registered inclusion weights.

    Because the eligible population size is fixed by the manifest and every
    selected unit in a cell has the same inclusion probability, this is the
    ratio form of the design-weighted (Hájek) estimator.  The name is retained
    for compatibility with the first v2 draft.
    """
    if not rows:
        raise ValueError("band has no rows")
    x = np.asarray([r["log_pattern_ratio_mid"] for r in rows], dtype=float)
    h = np.asarray([r["log_pattern_ratio_halfwidth"] for r in rows], dtype=float)
    pi = np.asarray([r["inclusion_probability"] for r in rows], dtype=float)
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(h)) or np.any(h < 0):
        raise ValueError("non-finite event score or invalid half-width")
    if np.any(~np.isfinite(pi)) or np.any((pi <= 0) | (pi > 1)):
        raise ValueError("inclusion probabilities must lie in (0, 1]")
    weights = 1.0 / pi
    return {"pattern_log_ratio": _weighted_mean(x, weights),
            "arith_halfwidth": _weighted_mean(h, weights),
            "n_usable": int(len(rows))}


def _validated_cells(rows: list[dict[str, Any]], bands: list[int]) \
        -> tuple[list[int], dict[tuple[int, int], list[dict[str, Any]]], np.ndarray]:
    """Validate the stratified SRSWOR design carried by the analysis rows.

    The finite population consists of fixed acquisition strata.  Within each
    band/stratum cell the beacon ranking selects ``n`` primary records from ``N``
    eligible records, so all analysis rows must carry the same primary inclusion
    probability ``n/N``.  At least two sampled records are needed to estimate a
    within-cell sampling distribution.
    """
    if not rows:
        raise ValueError("analysis has no rows")
    if len(set(bands)) != len(bands) or not bands:
        raise ValueError("analysis bands must be non-empty and unique")
    if {int(r["band"]) for r in rows} != set(bands):
        raise ValueError("analysis rows do not match the registered bands")
    strata = sorted({int(r["stratum"]) for r in rows})
    if len(strata) < 2:
        raise ValueError("at least two fixed acquisition strata are required")
    cells: dict[tuple[int, int], list[dict[str, Any]]] = {}
    population = np.empty((len(strata), len(bands)), dtype=float)
    for hi, stratum in enumerate(strata):
        for bi, band in enumerate(bands):
            cell = [r for r in rows if int(r["stratum"]) == stratum
                    and int(r["band"]) == band]
            if len(cell) < 2:
                raise ValueError(
                    f"band {band}, stratum {stratum} needs at least two sampled events"
                )
            sizes = {int(r.get("eligible_in_stratum", -1)) for r in cell}
            if len(sizes) != 1 or next(iter(sizes)) <= 0:
                raise ValueError("v2 rows must carry one positive eligible_in_stratum")
            eligible = next(iter(sizes))
            if len(cell) > eligible:
                raise ValueError("sampled cell exceeds its eligible finite population")
            probabilities = np.asarray(
                [float(r.get("inclusion_probability", float("nan"))) for r in cell],
                dtype=float,
            )
            expected = len(cell) / eligible
            if (np.any(~np.isfinite(probabilities))
                    or np.any((probabilities <= 0) | (probabilities > 1))
                    or not np.allclose(probabilities, probabilities[0], rtol=0, atol=1e-15)
                    or not math.isclose(float(probabilities[0]), expected,
                                        rel_tol=0, abs_tol=1e-12)):
                raise ValueError(
                    "primary inclusion probability must be equal within each cell and equal n/N"
                )
            cells[(stratum, band)] = cell
            population[hi, bi] = eligible
    if len(cells) != len(strata) * len(bands):
        raise ValueError("analysis rows do not form a complete band/stratum design")
    return strata, cells, population


def _common_block_cells(rows: list[dict[str, Any]], bands: list[int]) \
        -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    strata, cells, population = _validated_cells(rows, bands)
    score = np.empty((len(strata), len(bands)), dtype=float)
    halfwidth = np.empty_like(score)
    for hi, stratum in enumerate(strata):
        for bi, band in enumerate(bands):
            cell = cells[(stratum, band)]
            estimate = horvitz_thompson_band(cell)
            score[hi, bi] = estimate["pattern_log_ratio"]
            halfwidth[hi, bi] = estimate["arith_halfwidth"]
    return strata, score, halfwidth, population


def common_block_estimates(rows: list[dict[str, Any]], band_weights: dict[int, float],
                           normalizer_shift: dict[int, float] | None = None) -> np.ndarray:
    """Descriptive population-weighted contributions by acquisition stratum.

    A block is a common record-index stratum.  Its contribution is multiplied
    by ``H * N_(C,h) / N_C`` so that the arithmetic mean of the block values is
    exactly the full finite-population design estimate.  These fixed strata are
    never resampled for finite-acquisition inference.
    """
    normalizer_shift = normalizer_shift or {c: 0.0 for c in band_weights}
    bands = list(band_weights)
    strata, score, _, population = _common_block_cells(rows, bands)
    totals = population.sum(axis=0)
    shifts = np.asarray([normalizer_shift.get(band, 0.0) for band in bands])
    mixture = np.asarray([band_weights[band] for band in bands])
    return len(strata) * np.sum(
        mixture[None, :] * population / totals[None, :] * (score + shifts[None, :]),
        axis=1)


def load_joint_normalizer_replicates(path: str | Path, bands: list[int], *,
                                     registration_id: str | None = None,
                                     expected_replicates: int | None = None,
                                     expected_samples_per_replicate: int | None = None,
                                     expected_seed: int | None = None) \
        -> dict[str, np.ndarray]:
    """Load raw paired probability replicates without applying a log transform.

    Required NPZ members are ``p_reference`` and ``p_alternative`` with identical
    shape ``(replicate, band)`` plus JSON ``meta`` containing the exact band order.
    Keeping raw paired vectors preserves all cross-hypothesis and cross-band
    covariance; marginal standard deviations are deliberately insufficient.
    """
    with np.load(path, allow_pickle=False) as z:
        required = {"p_reference", "p_alternative", "meta"}
        if not required.issubset(z.files):
            raise ValueError(f"normalizer artifact requires {sorted(required)}")
        meta = json.loads(str(z["meta"]))
        if meta.get("schema") != "gbskernels.joint-normalizer-replicates.v1":
            raise ValueError("unsupported normalizer replicate schema")
        if [int(x) for x in meta.get("bands", [])] != bands:
            raise ValueError("normalizer band order does not match registration")
        if registration_id is not None and meta.get("registration_id") != registration_id:
            raise ValueError("normalizer artifact is bound to a different registration")
        if (expected_samples_per_replicate is not None
                and meta.get("samples_per_replicate") != expected_samples_per_replicate):
            raise ValueError("normalizer sample count differs from registration")
        if expected_seed is not None and meta.get("seed") != expected_seed:
            raise ValueError("normalizer seed differs from registration")
        p0 = np.asarray(z["p_reference"], dtype=float)
        p1 = np.asarray(z["p_alternative"], dtype=float)
    if p0.shape != p1.shape or p0.ndim != 2 or p0.shape[1] != len(bands):
        raise ValueError("normalizer replicate arrays have invalid shape")
    if (p0.shape[0] < 2 or np.any(~np.isfinite(p0))
            or np.any(~np.isfinite(p1)) or np.any(p0 <= 0) or np.any(p1 <= 0)
            or np.any(p0 > 1) or np.any(p1 > 1)):
        raise ValueError(
            "normalizer probabilities must lie in (0,1] with >=2 replicates")
    if expected_replicates is not None and p0.shape[0] != expected_replicates:
        raise ValueError("normalizer replicate count differs from registration")
    return {"p_reference": p0, "p_alternative": p1}


def _normalizer_estimate_and_errors(
    replicates: np.ndarray | dict[str, np.ndarray],
    *,
    bootstrap_reps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Pool all equal-effort replicates and bootstrap the pooled estimator."""
    if isinstance(replicates, dict):
        if set(replicates) != {"p_reference", "p_alternative"}:
            raise ValueError("raw normalizer replicates have invalid members")
        p0 = np.asarray(replicates["p_reference"], dtype=float)
        p1 = np.asarray(replicates["p_alternative"], dtype=float)
        if (p0.shape != p1.shape or p0.ndim != 2 or p0.shape[0] < 2
                or np.any(~np.isfinite(p0)) or np.any(~np.isfinite(p1))
                or np.any(p0 <= 0) or np.any(p1 <= 0)):
            raise ValueError("raw normalizer probability replicates are invalid")
        point = np.log(p0.mean(axis=0)) - np.log(p1.mean(axis=0))

        def estimate(draw: np.ndarray) -> np.ndarray:
            return (np.log(p0[draw].mean(axis=1))
                    - np.log(p1[draw].mean(axis=1)))

        method = "paired bootstrap of log ratio of pooled probability means"
        count = len(p0)
    else:
        values = np.asarray(replicates, dtype=float)
        if (values.ndim != 2 or values.shape[0] < 2
                or np.any(~np.isfinite(values))):
            raise ValueError("normalizer log-ratio replicates are invalid")
        point = values.mean(axis=0)

        def estimate(draw: np.ndarray) -> np.ndarray:
            return values[draw].mean(axis=1)

        method = "paired bootstrap of pooled log-ratio replicate mean"
        count = len(values)
    if bootstrap_reps < 2:
        raise ValueError("normalizer bootstrap requires at least two draws")
    errors = np.empty((bootstrap_reps, len(point)), dtype=float)
    chunk_size = 2048
    for start in range(0, bootstrap_reps, chunk_size):
        stop = min(start + chunk_size, bootstrap_reps)
        draw = rng.integers(0, count, size=(stop - start, count))
        errors[start:stop] = estimate(draw) - point
    return point, errors, count, method


def load_reconstruction_replicates(path: str | Path, bands: list[int], *,
                                   run_id: str | None = None,
                                   registration_id: str | None = None,
                                   manifest_id: str | None = None,
                                   expected_event_rows: list[dict[str, Any]] | None = None,
                                   expected_draws: int | None = None,
                                   expected_model_names: list[str] | None = None,
                                   expected_coherence_points: list[float] | None = None,
                                   calibration_sha256: str | None = None,
                                   nominal_normalizers_sha256: str | None = None,
                                   refusal_analysis_sha256: str | None = None) \
        -> dict[str, np.ndarray]:
    """Load draw-by-event reconstruction scores and their paired normalizers."""
    z = np.load(path, allow_pickle=False)
    required = {
        "joint_log_score_band_draws", "normalizer_log_ratio_band_draws",
        "model_log_score_band_draws", "model_arith_halfwidth_band_draws",
        "joint_arith_halfwidth_band_draws", "event_model_log_score_draws",
        "event_model_arith_halfwidth_draws", "event_ids", "event_bands",
        "event_strata", "event_positions", "event_record_indices",
        "event_input_sha256", "event_inclusion_probability",
        "event_eligible_in_stratum", "meta",
    }
    if not required.issubset(z.files):
        raise ValueError(f"reconstruction artifact requires {sorted(required)}")
    meta = json.loads(str(z["meta"]))
    if meta.get("schema") != "gbskernels.reconstruction-replicates.v3":
        raise ValueError("unsupported reconstruction replicate schema")
    if [int(x) for x in meta.get("bands", [])] != bands:
        raise ValueError("reconstruction band order does not match registration")
    if (meta.get("loss_variation_rtol") != 1e-10
            or meta.get("loss_variation_atol") != 1e-12):
        raise ValueError("reconstruction loss-variation tolerance is not registered")
    if run_id is not None and meta.get("run_id") != run_id:
        raise ValueError("reconstruction artifact is bound to a different run")
    if registration_id is not None and meta.get("registration_id") != registration_id:
        raise ValueError("reconstruction artifact is bound to a different registration")
    if manifest_id is not None and meta.get("manifest_id") != manifest_id:
        raise ValueError("reconstruction artifact is bound to a different manifest")
    joint = np.asarray(z["joint_log_score_band_draws"], dtype=float)
    norm = np.asarray(z["normalizer_log_ratio_band_draws"], dtype=float)
    models = np.asarray(z["model_log_score_band_draws"], dtype=float)
    model_arith = np.asarray(z["model_arith_halfwidth_band_draws"], dtype=float)
    joint_arith = np.asarray(z["joint_arith_halfwidth_band_draws"], dtype=float)
    event_models = np.asarray(z["event_model_log_score_draws"], dtype=float)
    event_arith = np.asarray(z["event_model_arith_halfwidth_draws"], dtype=float)
    event_ids = np.asarray(z["event_ids"]).astype(str)
    event_bands = np.asarray(z["event_bands"], dtype=int)
    event_strata = np.asarray(z["event_strata"], dtype=int)
    event_positions = np.asarray(z["event_positions"], dtype=int)
    event_records = np.asarray(z["event_record_indices"], dtype=int)
    event_inputs = np.asarray(z["event_input_sha256"]).astype(str)
    event_probability = np.asarray(z["event_inclusion_probability"], dtype=float)
    event_eligible = np.asarray(z["event_eligible_in_stratum"], dtype=int)
    if (joint.shape != norm.shape or joint.ndim != 2
            or joint.shape[1] != len(bands) or joint.shape[0] < 2):
        raise ValueError("invalid reconstruction replicate shape")
    if not np.all(np.isfinite(joint)) or not np.all(np.isfinite(norm)):
        raise ValueError("non-finite reconstruction displacement")
    model_names = [str(x) for x in meta.get("model_names", [])]
    if expected_model_names is not None and model_names != expected_model_names:
        raise ValueError("reconstruction model order differs from registration")
    if expected_coherence_points is not None and [float(x) for x in meta.get(
            "coherence_points", [])] != [float(x) for x in expected_coherence_points]:
        raise ValueError("reconstruction coherence coordinates differ from registration")
    if (models.ndim != 3 or models.shape[0] != joint.shape[0]
            or models.shape[2] != len(bands) or models.shape[1] != len(model_names)
            or not np.all(np.isfinite(models))):
        raise ValueError("invalid coherence-model reconstruction displacements")
    if (model_arith.shape != models.shape or joint_arith.shape != joint.shape
            or np.any(~np.isfinite(model_arith)) or np.any(model_arith < 0)
            or np.any(~np.isfinite(joint_arith)) or np.any(joint_arith < 0)):
        raise ValueError("invalid reconstruction arithmetic widths")
    if meta.get("draws") != joint.shape[0]:
        raise ValueError("reconstruction metadata draw count is inconsistent")
    if expected_draws is not None and joint.shape[0] != expected_draws:
        raise ValueError("reconstruction draw count differs from registration")
    event_count = len(event_ids)
    identity_arrays = (event_bands, event_strata, event_positions, event_records,
                       event_inputs, event_probability, event_eligible)
    if (event_ids.ndim != 1 or event_count == 0
            or any(array.shape != (event_count,) for array in identity_arrays)
            or len(set(event_ids.tolist())) != event_count
            or event_models.shape != (joint.shape[0], event_count, len(model_names))
            or event_arith.shape != event_models.shape
            or np.any(~np.isfinite(event_models))
            or np.any(~np.isfinite(event_arith)) or np.any(event_arith < 0)
            or np.any(~np.isfinite(event_probability))
            or np.any(event_probability <= 0) or np.any(event_probability > 1)
            or np.any(event_eligible <= 0)):
        raise ValueError("invalid reconstruction event-level tensor or identity")
    identity = [{
        "event_id": event_ids[index], "band": int(event_bands[index]),
        "stratum": int(event_strata[index]), "position": int(event_positions[index]),
        "record_index": int(event_records[index]),
        "input_sha256": event_inputs[index],
        "inclusion_probability": float(event_probability[index]),
        "eligible_in_stratum": int(event_eligible[index]),
    } for index in range(event_count)]
    if (meta.get("event_count") != event_count
            or meta.get("event_identity_sha256") != hash_json(identity)):
        raise ValueError("reconstruction event identity metadata is inconsistent")
    if expected_event_rows is not None:
        expected_identity = [{
            "event_id": str(row["event_id"]), "band": int(row["band"]),
            "stratum": int(row["stratum"]), "position": int(row["position"]),
            "record_index": int(row["record_index"]),
            "input_sha256": str(row["input_sha256"]),
            "inclusion_probability": float(row["inclusion_probability"]),
            "eligible_in_stratum": int(row["eligible_in_stratum"]),
        } for row in expected_event_rows]
        if identity != expected_identity:
            raise ValueError("reconstruction events differ from the verified analysis rows")
    if calibration_sha256 is not None and meta.get(
            "calibration_sha256") != calibration_sha256:
        raise ValueError("reconstruction calibration differs from registration")
    if nominal_normalizers_sha256 is not None and meta.get(
            "nominal_normalizers_sha256") != nominal_normalizers_sha256:
        raise ValueError("reconstruction used different nominal normalizers")
    if refusal_analysis_sha256 is None:
        if "refusal_analysis_sha256" in meta:
            raise ValueError("reconstruction unexpectedly consumed a refusal analysis")
    elif meta.get("refusal_analysis_sha256") != refusal_analysis_sha256:
        raise ValueError("reconstruction used a different refusal analysis")
    for field in ("calibration_sha256", "normalizer_draws_sha256",
                  "nominal_normalizers_sha256"):
        value = meta.get(field)
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError(f"reconstruction metadata has invalid {field}")
    recomputed_models = np.empty_like(models)
    recomputed_arith = np.empty_like(model_arith)
    for bi, band in enumerate(bands):
        indices = np.flatnonzero(event_bands == band)
        if len(indices) == 0:
            raise ValueError("reconstruction events do not cover every registered band")
        weights = 1.0 / event_probability[indices]
        recomputed_models[:, :, bi] = np.sum(
            event_models[:, indices, :] * weights[None, :, None], axis=1
        ) / np.sum(weights)
        recomputed_arith[:, :, bi] = np.sum(
            event_arith[:, indices, :] * weights[None, :, None], axis=1
        ) / np.sum(weights)
    if (set(event_bands.tolist()) != set(bands)
            or len(model_names) < 2
            or not np.allclose(models, recomputed_models, rtol=1e-13, atol=1e-13)
            or not np.allclose(model_arith, recomputed_arith, rtol=1e-13, atol=1e-13)
            or not np.array_equal(joint, models[:, 1, :] - models[:, 0, :])
            or not np.array_equal(joint_arith,
                                  model_arith[:, 0, :] + model_arith[:, 1, :])):
        raise ValueError("reconstruction endpoint arrays contradict the model tensor")
    return {
        "joint": joint, "normalizer": norm,
        "model_scores": models, "model_names": model_names,
        "model_arith": model_arith, "joint_arith": joint_arith,
        "event_model_scores": event_models, "event_model_arith": event_arith,
        "event_ids": event_ids, "event_bands": event_bands,
        "event_strata": event_strata, "event_positions": event_positions,
        "event_record_indices": event_records, "event_input_sha256": event_inputs,
        "event_inclusion_probability": event_probability,
        "event_eligible_in_stratum": event_eligible,
    }


def restore_recovered_primary_rows(run: dict[str, Any], refusal: dict[str, Any]) \
        -> list[dict[str, Any]]:
    """Replace operational reserve rows with recovered refused primaries.

    The registered primary rank remains the inferential unit. Reserves only
    keep the evaluator moving; once an independent recovery score exists, the
    primary event is restored and its reserve replacement is removed.
    """
    rows = {str(row["event_id"]): dict(row) for row in run.get("rows", [])}
    refused = {str(row["event_id"]): row for row in run.get("refusals", [])}
    recovered = refusal.get("recovered_scores")
    if not isinstance(recovered, list) or {str(row.get("event_id")) for row in recovered} \
            != set(refused):
        raise ValueError("refusal artifact lacks an exact recovered score for every refusal")
    for recovered_row in recovered:
        event_id = str(recovered_row["event_id"])
        refusal_row = refused[event_id]
        for key in ("band", "record_index", "stratum", "input_sha256"):
            if recovered_row.get(key) != refusal_row.get(key):
                raise ValueError(f"recovered primary {event_id} mismatches {key}")
        selected_primary = refusal_row.get("selected_primary")
        if (not isinstance(selected_primary, bool)
                or recovered_row.get("selected_primary") is not selected_primary):
            raise ValueError(f"recovered event {event_id} mismatches its registered role")
        if not selected_primary:
            continue
        replacements = [
            row for row in rows.values()
            if row.get("replacement_for_refusal_event_id") == event_id
        ]
        if len(replacements) != 1:
            raise ValueError(
                f"recovered primary {event_id} does not have exactly one reserve replacement"
            )
        del rows[str(replacements[0]["event_id"])]
        midpoint = recovered_row.get("log_pattern_ratio_mid")
        halfwidth = recovered_row.get("log_pattern_ratio_halfwidth")
        if (isinstance(midpoint, bool) or not isinstance(midpoint, (int, float))
                or not math.isfinite(float(midpoint))
                or isinstance(halfwidth, bool) or not isinstance(halfwidth, (int, float))
                or not math.isfinite(float(halfwidth)) or float(halfwidth) < 0):
            raise ValueError(f"recovered primary {event_id} lacks a valid score enclosure")
        restored = dict(refusal_row)
        restored.update({
            "refused": False,
            "recovered_from_refusal": True,
            "log_pattern_ratio_mid": float(midpoint),
            "log_pattern_ratio_halfwidth": float(halfwidth),
            "replacement_for_refusal": False,
            "replacement_for_refusal_event_id": None,
        })
        if "model_log_probability_proxy" in recovered_row:
            restored["model_log_probability_proxy"] = recovered_row[
                "model_log_probability_proxy"]
        rows[event_id] = restored
    return list(rows.values())


_PREDICTIVE_METRIC_THRESHOLDS = {
    "click_count_tv": "click_count_tv_max",
    "marginal_rms": "marginal_rms_max",
    "pair_covariance_rms": "pair_covariance_rms_max",
}


def _finite_nonnegative(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)) and float(value) >= 0.0)


def validate_absolute_predictive_checks(
        checks: dict[str, Any], *, run_id: str, registration_id: str,
        selection_cfg: dict[str, Any], predictive_cfg: dict[str, Any],
        model_names: list[str], expected_provenance: dict[str, Any]) -> dict[str, bool]:
    """Validate a predictive-check artifact and recompute every model decision."""
    if (not isinstance(checks, dict)
            or checks.get("schema") != "gbskernels.absolute-predictive-checks.v1"):
        raise ValueError("registration requires absolute predictive checks")
    if checks.get("run_id") != run_id:
        raise ValueError("predictive checks are bound to a different run")
    if checks.get("registration_id") != registration_id:
        raise ValueError("predictive checks are bound to a different registration")
    if checks.get("data_sha256") != selection_cfg.get("source_raw_sha256"):
        raise ValueError("predictive checks used a different acquisition")
    if checks.get("exclusion_sha256") != selection_cfg.get("exclusion_sha256"):
        raise ValueError("predictive checks used different registered exclusions")
    if any(checks.get(key) != value for key, value in expected_provenance.items()):
        raise ValueError("predictive-check provenance differs from the verified run")

    artifact_pairs = checks.get("detector_pairs")
    if (not isinstance(artifact_pairs, list)
            or any(not isinstance(pair, list) or len(pair) != 2
                   or any(isinstance(index, bool) or not isinstance(index, int)
                          for index in pair)
                   for pair in artifact_pairs)
            or artifact_pairs != predictive_cfg.get("detector_pairs")):
        raise ValueError("predictive-check detector pairs differ from registration")
    registered_thresholds = predictive_cfg.get("thresholds")
    artifact_thresholds = checks.get("thresholds")
    if (not isinstance(registered_thresholds, dict)
            or not isinstance(artifact_thresholds, dict)
            or artifact_thresholds != registered_thresholds):
        raise ValueError("predictive-check thresholds differ from registration")
    for threshold_key in _PREDICTIVE_METRIC_THRESHOLDS.values():
        if (not _finite_nonnegative(registered_thresholds.get(threshold_key))
                or not _finite_nonnegative(artifact_thresholds.get(threshold_key))
                or float(registered_thresholds[threshold_key]) <= 0.0):
            raise ValueError("registered predictive-check thresholds are invalid")

    model_checks = checks.get("models")
    expected_models = {str(name) for name in model_names}
    if (not isinstance(model_checks, dict)
            or set(model_checks) != expected_models
            or len(expected_models) != len(model_names)):
        raise ValueError("predictive-check model set differs from registration")
    model_passes: dict[str, bool] = {}
    for name in model_names:
        row = model_checks[name]
        if not isinstance(row, dict):
            raise ValueError(f"predictive-check model {name!r} lacks metric results")
        computed_pass = True
        for metric_key, threshold_key in _PREDICTIVE_METRIC_THRESHOLDS.items():
            value = row.get(metric_key)
            if not _finite_nonnegative(value):
                raise ValueError(
                    f"predictive-check model {name!r} has invalid {metric_key}")
            computed_pass = computed_pass and (
                float(value) <= float(registered_thresholds[threshold_key]))
        if not isinstance(row.get("pass"), bool) or row["pass"] != computed_pass:
            raise ValueError(
                f"predictive-check model {name!r} pass flag disagrees with its metrics")
        model_passes[name] = bool(computed_pass)
    any_passes = any(model_passes.values())
    if (not isinstance(checks.get("any_model_passes"), bool)
            or checks["any_model_passes"] != any_passes):
        raise ValueError("predictive-check aggregate pass flag disagrees with model metrics")
    return model_passes


def predictive_model_gate(result: dict[str, Any], model_passes: dict[str, bool], *,
                          policy: str, model_cfg: dict[str, Any]) -> dict[str, Any]:
    registered_models = {str(name) for name in model_cfg.get("coherence_points", {})}
    if (not isinstance(model_passes, dict)
            or set(model_passes) != registered_models
            or any(not isinstance(value, bool) for value in model_passes.values())):
        raise ValueError("predictive gate requires validated results for every registered model")
    passing = {str(name) for name, passed in model_passes.items() if passed}
    if policy == "all_plausible_best":
        required = set(result.get("coherence_grid", {}).get(
            "statistical_confidence_set_models", []))
        passed = bool(required) and required.issubset(passing)
    elif policy == "estimated_best":
        required = {str(result.get("coherence_grid", {}).get("estimate_model", ""))}
        passed = "" not in required and required.issubset(passing)
    elif policy == "reference_and_alternative":
        required = {str(model_cfg["reference_model"]),
                    str(model_cfg["alternative_model"])}
        passed = required.issubset(passing)
    elif policy == "any_registered":
        required = registered_models
        passed = bool(passing)
    else:
        raise ValueError(f"unsupported predictive model-pass policy {policy!r}")
    return {"policy": policy, "required_models": sorted(required),
            "passing_models": sorted(passing), "pass": bool(passed)}


def registered_nonclassical_decision(result: dict[str, Any],
                                     analysis_config: dict[str, Any]) -> dict[str, Any]:
    """Apply the exact registered finite-acquisition claim rule."""
    rule = analysis_config.get("primary_decision_rule")
    expected = {
        "method": "simultaneous_paired_model_score_max_t",
        "claim_if_confidence_set_above_classical_boundary": True,
        "require_predictive_pass_for_all_confidence_set_models": True,
        "report_failure_without_suppressing_analysis": True,
    }
    if rule != expected:
        raise ValueError("analysis configuration lacks the registered v2 decision rule")
    grid = result.get("coherence_grid")
    if not isinstance(grid, dict):
        return {
            "rule": rule,
            "claim": "nonclassical_anomalous_coherence",
            "claim_scope": "finite_registered_acquisition",
            "confidence_set_pass": False,
            "predictive_gate_pass": False,
            "claim_supported": False,
            "failure_reasons": ["registered coherence-grid result is unavailable"],
        }
    confidence_pass = grid.get("confidence_set_excludes_classical_region") is True
    predictive = result.get("predictive_model_gate")
    predictive_pass = isinstance(predictive, dict) and predictive.get("pass") is True
    reasons = []
    if not confidence_pass:
        reasons.append("simultaneous confidence set intersects the classical region")
    if not predictive_pass:
        reasons.append(
            "not every model in the simultaneous confidence set passes prediction")
    return {
        "rule": rule,
        "claim": "nonclassical_anomalous_coherence",
        "claim_scope": "finite_registered_acquisition",
        "minimum_relevant_coherence_used_for_design": float(
            analysis_config["minimum_relevant_coherence"]),
        "statistical_confidence_set_models":
            list(grid["statistical_confidence_set_models"]),
        "statistical_confidence_set_coordinate_interval":
            list(grid["statistical_confidence_set_coordinate_interval"]),
        "confidence_set_pass": confidence_pass,
        "predictive_gate_pass": predictive_pass,
        "claim_supported": bool(confidence_pass and predictive_pass),
        "failure_reasons": reasons,
    }


def _confidence_interval_from_errors(
    point: float,
    estimation_errors: np.ndarray,
    *,
    alpha: float,
    deterministic: float = 0.0,
    model_displacements: np.ndarray | None = None,
) -> dict[str, Any]:
    """Invert estimator errors and optionally add model-uncertainty shifts."""
    errors = np.asarray(estimation_errors, dtype=float)
    if (errors.ndim != 1 or len(errors) < 2 or np.any(~np.isfinite(errors))
            or not (0 < alpha < 1) or not math.isfinite(deterministic)
            or deterministic < 0):
        raise ValueError("invalid confidence-interval inputs")
    displacement = -errors
    if model_displacements is not None:
        model_shift = np.asarray(model_displacements, dtype=float)
        if model_shift.shape != errors.shape or np.any(~np.isfinite(model_shift)):
            raise ValueError("model displacements must match estimation errors")
        displacement = displacement + model_shift
    lo_q, hi_q = np.quantile(displacement, [alpha / 2, 1 - alpha / 2])
    random_ci = [float(point + lo_q), float(point + hi_q)]
    return {
        "estimate": float(point),
        "confidence_interval_random": random_ci,
        "confidence_interval_with_arithmetic": [
            random_ci[0] - deterministic, random_ci[1] + deterministic],
        "arithmetic_worstcase_displacement": float(deterministic),
    }


def _centered_srswor_bootstrap(values: np.ndarray, eligible: int, reps: int,
                               rng: np.random.Generator) -> np.ndarray:
    """Bootstrap centered sample-mean errors with exact estimated SRSWOR variance.

    Resampling ``n`` centered observations with replacement has conditional
    variance ``(n-1) s^2 / n^2``. Multiplication by
    ``sqrt((1-n/N) n/(n-1))`` therefore gives exactly the usual SRSWOR variance
    estimate ``(1-n/N) s^2/n``. A census cell has identically zero sampling
    error and consumes no random numbers.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim not in (1, 2):
        raise ValueError("bootstrap values must be a vector or matrix")
    n = len(values)
    if n < 2 or eligible < n or reps < 1:
        raise ValueError("invalid SRSWOR bootstrap dimensions")
    if np.any(~np.isfinite(values)):
        raise ValueError("non-finite values in SRSWOR bootstrap cell")
    shape = (reps,) + values.shape[1:]
    if n == eligible:
        return np.zeros(shape, dtype=float)
    centered = values - values.mean(axis=0)
    scale = math.sqrt((1.0 - n / eligible) * n / (n - 1))
    out = np.empty(shape, dtype=float)
    chunk_size = 2048
    for start in range(0, reps, chunk_size):
        stop = min(start + chunk_size, reps)
        draw = rng.integers(0, n, size=(stop - start, n))
        out[start:stop] = centered[draw].mean(axis=1) * scale
    return out


def _resample_scalar_cells(rows: list[dict[str, Any]], bands: list[int],
                           strata: list[int], reps: int,
                           rng: np.random.Generator) -> np.ndarray:
    """Centered sampling errors for every fixed band/stratum cell."""
    checked_strata, cells, population = _validated_cells(rows, bands)
    if checked_strata != strata:
        raise ValueError("fixed-stratum order differs from validated design")
    errors = np.empty((reps, len(strata), len(bands)), dtype=float)
    for hi, stratum in enumerate(strata):
        for bi, band in enumerate(bands):
            cell = cells[(stratum, band)]
            values = np.asarray([r["log_pattern_ratio_mid"] for r in cell], dtype=float)
            half = np.asarray([r["log_pattern_ratio_halfwidth"] for r in cell], dtype=float)
            if np.any(~np.isfinite(half)) or np.any(half < 0):
                raise ValueError("invalid event score interval in bootstrap cell")
            errors[:, hi, bi] = _centered_srswor_bootstrap(
                values, int(population[hi, bi]), reps, rng)
    return errors


def _resample_model_cells(rows: list[dict[str, Any]], bands: list[int],
                          strata: list[int], names: list[str], reps: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Paired model-score sampling errors for every fixed finite cell."""
    checked_strata, cells, population = _validated_cells(rows, bands)
    if checked_strata != strata:
        raise ValueError("fixed-stratum order differs from validated design")
    errors = np.empty((reps, len(strata), len(bands), len(names)), dtype=float)
    for hi, stratum in enumerate(strata):
        for bi, band in enumerate(bands):
            cell = cells[(stratum, band)]
            values = np.asarray([
                [r["model_log_probability_proxy"][name]["mid"] for name in names]
                for r in cell], dtype=float)
            half = np.asarray([
                [r["model_log_probability_proxy"][name]["halfwidth"] for name in names]
                for r in cell], dtype=float)
            if np.any(~np.isfinite(half)) or np.any(half < 0):
                raise ValueError("invalid coherence score interval in bootstrap cell")
            errors[:, hi, bi, :] = _centered_srswor_bootstrap(
                values, int(population[hi, bi]), reps, rng)
    return errors


def _aligned_reconstruction_event_models(
    rows: list[dict[str, Any]], reconstruction: dict[str, np.ndarray],
    names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Align draw-by-event scores to verified rows and a requested model order."""
    required = {
        "event_model_scores", "event_model_arith", "event_ids", "event_bands",
        "event_strata", "event_inclusion_probability", "event_eligible_in_stratum",
        "model_names",
    }
    if not required.issubset(reconstruction):
        raise ValueError("reconstruction requires draw-by-event model scores")
    row_ids = [str(row.get("event_id")) for row in rows]
    artifact_ids = [str(value) for value in reconstruction["event_ids"]]
    if (len(set(row_ids)) != len(row_ids) or len(set(artifact_ids)) != len(artifact_ids)
            or set(row_ids) != set(artifact_ids)):
        raise ValueError("reconstruction event IDs differ from analysis rows")
    event_index = {event_id: index for index, event_id in enumerate(artifact_ids)}
    reorder_events = np.asarray([event_index[event_id] for event_id in row_ids], dtype=int)
    artifact_names = [str(name) for name in reconstruction["model_names"]]
    requested_names = artifact_names if names is None else names
    try:
        reorder_models = np.asarray(
            [artifact_names.index(name) for name in requested_names], dtype=int)
    except ValueError as exc:
        raise ValueError("reconstruction artifact lacks a registered grid model") from exc
    scores = np.asarray(reconstruction["event_model_scores"], dtype=float)
    arith = np.asarray(reconstruction["event_model_arith"], dtype=float)
    if (scores.ndim != 3 or arith.shape != scores.shape
            or scores.shape[1] != len(artifact_ids)
            or scores.shape[2] != len(artifact_names)
            or scores.shape[0] < 2 or np.any(~np.isfinite(scores))
            or np.any(~np.isfinite(arith)) or np.any(arith < 0)):
        raise ValueError("invalid reconstruction event score tensor")
    for row_index, artifact_index in enumerate(reorder_events):
        row = rows[row_index]
        comparisons = (
            (int(reconstruction["event_bands"][artifact_index]), int(row["band"])),
            (int(reconstruction["event_strata"][artifact_index]), int(row["stratum"])),
            (float(reconstruction["event_inclusion_probability"][artifact_index]),
             float(row["inclusion_probability"])),
            (int(reconstruction["event_eligible_in_stratum"][artifact_index]),
             int(row["eligible_in_stratum"])),
        )
        if any(actual != expected for actual, expected in comparisons):
            raise ValueError("reconstruction event design metadata differs from analysis rows")
    return (scores[:, reorder_events][:, :, reorder_models],
            arith[:, reorder_events][:, :, reorder_models])


def _event_draw_band_means(
    rows: list[dict[str, Any]], bands: list[int], values: np.ndarray,
) -> np.ndarray:
    """Compute the registered inverse-inclusion weighted band mean for each draw."""
    values = np.asarray(values, dtype=float)
    if (values.ndim != 3 or values.shape[1] != len(rows)
            or np.any(~np.isfinite(values))):
        raise ValueError("invalid draw-by-event score tensor")
    out = np.empty((values.shape[0], len(bands), values.shape[2]), dtype=float)
    for bi, band in enumerate(bands):
        indices = [index for index, row in enumerate(rows) if int(row["band"]) == band]
        if not indices:
            raise ValueError("draw-by-event scores do not cover every registered band")
        probability = np.asarray(
            [float(rows[index]["inclusion_probability"]) for index in indices])
        if (np.any(~np.isfinite(probability)) or np.any(probability <= 0)
                or np.any(probability > 1)):
            raise ValueError("invalid inclusion probability in reconstruction rows")
        weights = 1.0 / probability
        out[:, bi, :] = np.sum(
            values[:, indices, :] * weights[None, :, None], axis=1
        ) / np.sum(weights)
    return out


def _resample_draw_specific_model_cells(
    rows: list[dict[str, Any]], bands: list[int], strata: list[int],
    values: np.ndarray, draw_indices: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """SRSWOR sampling errors conditional on each replicate's calibration draw."""
    checked_strata, cells, population = _validated_cells(rows, bands)
    if checked_strata != strata:
        raise ValueError("fixed-stratum order differs from validated design")
    values = np.asarray(values, dtype=float)
    draw_indices = np.asarray(draw_indices, dtype=int)
    if (values.ndim != 3 or values.shape[1] != len(rows)
            or draw_indices.ndim != 1 or np.any(draw_indices < 0)
            or np.any(draw_indices >= values.shape[0]) or np.any(~np.isfinite(values))):
        raise ValueError("invalid conditional reconstruction bootstrap inputs")
    row_index = {id(row): index for index, row in enumerate(rows)}
    errors = np.empty(
        (len(draw_indices), len(strata), len(bands), values.shape[2]), dtype=float)
    for hi, stratum in enumerate(strata):
        for bi, band in enumerate(bands):
            cell = cells[(stratum, band)]
            indices = np.asarray([row_index[id(row)] for row in cell], dtype=int)
            n = len(indices)
            eligible = int(population[hi, bi])
            if n == eligible:
                errors[:, hi, bi, :] = 0.0
                continue
            cell_values = values[:, indices, :]
            centered = cell_values - cell_values.mean(axis=1, keepdims=True)
            scale = math.sqrt((1.0 - n / eligible) * n / (n - 1))
            chunk_size = 2048
            for start in range(0, len(draw_indices), chunk_size):
                stop = min(start + chunk_size, len(draw_indices))
                selected_draws = draw_indices[start:stop]
                sampled_indices = rng.integers(0, n, size=(stop - start, n))
                sampled = centered[selected_draws[:, None], sampled_indices]
                errors[start:stop, hi, bi, :] = sampled.mean(axis=1) * scale
    return errors


def analyze(rows: list[dict[str, Any]], *, band_weights: dict[int, float],
            normalizer_replicates: np.ndarray | dict[str, np.ndarray],
            reconstruction_replicates: dict[str, np.ndarray] | None,
            bootstrap_reps: int, bootstrap_seed: int, alpha: float = 0.05) -> dict[str, Any]:
    bands = list(band_weights)
    if any(not (value > 0) for value in band_weights.values()):
        raise ValueError("band weights must be positive")
    if not math.isclose(sum(band_weights.values()), 1.0, abs_tol=1e-12):
        raise ValueError("band weights must sum to one")
    strata, _, block_halfwidths, block_population = _common_block_cells(rows, bands)
    by_band = {c: horvitz_thompson_band([r for r in rows if int(r["band"]) == c])
               for c in bands}
    pattern_point = sum(band_weights[c] * by_band[c]["pattern_log_ratio"] for c in bands)
    arithmetic_bound = 0.0
    for bi, band in enumerate(bands):
        population_share = block_population[:, bi] / block_population[:, bi].sum()
        arithmetic_bound += band_weights[band] * float(
            np.dot(population_share, block_halfwidths[:, bi]))
    point_blocks = common_block_estimates(rows, band_weights)

    rng = np.random.default_rng(bootstrap_seed)
    cell_errors = _resample_scalar_cells(rows, bands, strata, bootstrap_reps, rng)
    event_draw = np.zeros(bootstrap_reps, dtype=float)
    for bi, band in enumerate(bands):
        population_share = block_population[:, bi] / block_population[:, bi].sum()
        event_draw += band_weights[band] * (cell_errors[:, :, bi] @ population_share)
    w = np.asarray(list(band_weights.values()), dtype=float)
    normalizer_center, norm_draw, normalizer_count, normalizer_method = \
        _normalizer_estimate_and_errors(
            normalizer_replicates, bootstrap_reps=bootstrap_reps, rng=rng)
    if normalizer_center.shape != (len(bands),):
        raise ValueError("normalizer replicate width mismatch")
    normalizer_point = float(np.dot(w, normalizer_center))
    norm_weighted_draw = norm_draw @ w

    joint_point = pattern_point
    click_point = -normalizer_point
    conditional_point = pattern_point + normalizer_point
    joint_result = _confidence_interval_from_errors(
        joint_point, event_draw, alpha=alpha, deterministic=arithmetic_bound)
    click_result = _confidence_interval_from_errors(
        click_point, -norm_weighted_draw, alpha=alpha)
    conditional_result = _confidence_interval_from_errors(
        conditional_point, event_draw + norm_weighted_draw,
        alpha=alpha, deterministic=arithmetic_bound)

    result: dict[str, Any] = {
        "estimands": {
            "joint_window_log_score_difference": joint_result,
            "click_count_log_score_difference": click_result,
            "conditional_pattern_log_score_difference": conditional_result,
        },
        "decomposition_identity": "joint = click_count + conditional_pattern",
        "positive_direction": "alternative assigns higher log score",
        "point_model": {
            "joint": joint_result, "click_count": click_result,
            "conditional_pattern": conditional_result,
            "common_block_estimates_pattern_only": point_blocks.tolist(),
        },
        "bands": {str(c): by_band[c] for c in bands},
        "normalizer_covariance": np.atleast_2d(
            np.cov(norm_draw, rowvar=False, ddof=1)).tolist(),
        "normalizer_estimator": {
            "replicates": int(normalizer_count),
            "method": normalizer_method,
            "pooled_band_log_ratio": normalizer_center.tolist(),
        },
        "bootstrap": {"reps": int(bootstrap_reps), "seed": int(bootstrap_seed),
                      "alpha": float(alpha),
                      "unit": "selected event within fixed band/stratum cell",
                      "strata": "fixed; never resampled",
                      "estimator": "fixed-stratum finite-population weighted mean",
                      "within_cell_bootstrap": (
                          "centered resampling with exact estimated SRSWOR "
                          "finite-population correction"),
                      "arithmetic_bound": arithmetic_bound},
    }
    if reconstruction_replicates is None:
        result["reconstruction_marginal"] = None
        result["scope"] = "frozen point models only"
    else:
        event_models, event_arith = _aligned_reconstruction_event_models(
            rows, reconstruction_replicates)
        if event_models.shape[2] < 2:
            raise ValueError("reconstruction requires reference and alternative models")
        joint_event = event_models[:, :, 1:2] - event_models[:, :, 0:1]
        joint_event_arith = event_arith[:, :, 0:1] + event_arith[:, :, 1:2]
        joint_by_band = _event_draw_band_means(rows, bands, joint_event)[:, :, 0]
        arith_by_band = _event_draw_band_means(
            rows, bands, joint_event_arith)[:, :, 0]
        joint_recon = np.asarray(reconstruction_replicates["joint"], dtype=float)
        norm_recon = np.asarray(reconstruction_replicates["normalizer"], dtype=float)
        if (joint_recon.shape != joint_by_band.shape
                or norm_recon.shape != joint_by_band.shape
                or not np.allclose(joint_recon, joint_by_band, rtol=1e-13, atol=1e-13)
                or np.any(~np.isfinite(norm_recon))):
            raise ValueError("reconstruction band summaries contradict event-level draws")
        ri = rng.integers(0, len(joint_recon), bootstrap_reps)
        joint_center = joint_recon.mean(axis=0)
        norm_center = norm_recon.mean(axis=0)
        joint_recon_draw = (joint_recon[ri] - joint_center) @ w
        norm_recon_draw = (norm_recon[ri] - norm_center) @ w
        conditional_cell_errors = _resample_draw_specific_model_cells(
            rows, bands, strata, joint_event, ri, rng)[:, :, :, 0]
        conditional_event_draw = np.zeros(bootstrap_reps, dtype=float)
        for bi, band in enumerate(bands):
            population_share = block_population[:, bi] / block_population[:, bi].sum()
            conditional_event_draw += band_weights[band] * (
                conditional_cell_errors[:, :, bi] @ population_share)
        marginal_joint = float(joint_center @ w)
        marginal_click = -float(norm_center @ w)
        marginal_conditional = float((joint_center + norm_center) @ w)
        reconstruction_arith = float(np.max(arith_by_band @ w))
        result["reconstruction_marginal"] = {
            "joint": _confidence_interval_from_errors(
                marginal_joint, conditional_event_draw, alpha=alpha,
                deterministic=reconstruction_arith,
                model_displacements=joint_recon_draw),
            "click_count": _confidence_interval_from_errors(
                marginal_click, np.zeros_like(norm_weighted_draw), alpha=alpha,
                model_displacements=-norm_recon_draw),
            "conditional_pattern": _confidence_interval_from_errors(
                marginal_conditional, conditional_event_draw,
                alpha=alpha, deterministic=reconstruction_arith,
                model_displacements=joint_recon_draw + norm_recon_draw),
            "n_draws": int(len(joint_recon)),
            "sampling_calibration_pairing": (
                "event resampling is conditional on the same calibration draw as "
                "the model and normalizer displacement"),
        }
        result["scope"] = "reconstruction-marginal model family"
    return result


def _simultaneous_best_confidence_set(
    point_scores: np.ndarray, bootstrap_errors: np.ndarray,
    arithmetic_halfwidths: np.ndarray, names: list[str], *, alpha: float,
) -> dict[str, Any]:
    """Invert simultaneous paired score-difference max-t intervals.

    Model ``g`` remains in the confidence set unless some model ``j`` has a
    strictly positive lower confidence bound for ``mu_j - mu_g``. Arithmetic
    widths are combined by Minkowski addition after the random max-t interval.
    """
    point_scores = np.asarray(point_scores, dtype=float)
    errors = np.asarray(bootstrap_errors, dtype=float)
    arithmetic = np.asarray(arithmetic_halfwidths, dtype=float)
    m = len(names)
    if (point_scores.shape != (m,) or arithmetic.shape != (m,)
            or errors.ndim != 2 or errors.shape[1] != m or errors.shape[0] < 2
            or np.any(~np.isfinite(point_scores)) or np.any(~np.isfinite(errors))
            or np.any(~np.isfinite(arithmetic)) or np.any(arithmetic < 0)
            or not (0 < alpha < 1)):
        raise ValueError("invalid simultaneous confidence-set inputs")
    # Axis convention is [competitor j, candidate g]. All model coordinates are
    # evaluated on the same sampled events, so these differences remain paired.
    pair_errors = errors[:, :, None] - errors[:, None, :]
    pair_se = np.std(pair_errors, axis=0, ddof=1)
    standardized = np.zeros_like(pair_errors)
    np.divide(np.abs(pair_errors), pair_se[None, :, :], out=standardized,
              where=pair_se[None, :, :] > 0)
    max_t = np.max(standardized, axis=(1, 2))
    critical = float(np.quantile(max_t, 1 - alpha, method="higher"))
    random_halfwidth = critical * pair_se
    arithmetic_pair_halfwidth = arithmetic[:, None] + arithmetic[None, :]
    total_halfwidth = random_halfwidth + arithmetic_pair_halfwidth
    observed_difference = point_scores[:, None] - point_scores[None, :]

    random_plausible = np.all(
        observed_difference - random_halfwidth <= 0, axis=0)
    expanded_plausible = np.all(
        observed_difference - total_halfwidth <= 0, axis=0)
    return {
        "method": "simultaneous paired score-difference max-t",
        "confidence_level": float(1 - alpha),
        "critical_value": critical,
        "pairwise_axis": "row competitor minus column candidate",
        "pairwise_score_differences": observed_difference.tolist(),
        "pairwise_standard_errors": pair_se.tolist(),
        "pairwise_random_halfwidths": random_halfwidth.tolist(),
        "pairwise_arithmetic_halfwidths": arithmetic_pair_halfwidth.tolist(),
        "pairwise_total_halfwidths": total_halfwidth.tolist(),
        "models_random_only": [names[i] for i in np.flatnonzero(random_plausible)],
        "models": [names[i] for i in np.flatnonzero(expanded_plausible)],
    }


def analyze_coherence_grid(
    rows: list[dict[str, Any]], model_coordinates: dict[str, float], *,
    band_weights: dict[int, float],
    classical_boundary: float = 0.0, bootstrap_reps: int,
    bootstrap_seed: int, alpha: float = 0.05,
    reconstruction_replicates: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Infer a scalar coherence coordinate from registered grid evaluations.

    Each event row must contain ``model_log_probability_proxy`` for every grid
    point. The point estimate maximizes the finite-population weighted mean log
    score. Fixed strata are never resampled: centered event vectors are
    resampled within each cell with the SRSWOR finite-population correction.
    Inference inverts simultaneous paired model-score-difference max-t intervals;
    the argmax bootstrap distribution is descriptive only. The proxy includes
    both the torontonian and frozen determinant normalizer.
    """
    if len(model_coordinates) < 3:
        raise ValueError("coherence inference requires at least three grid points")
    names = sorted(model_coordinates, key=model_coordinates.get)
    coords = np.asarray([model_coordinates[n] for n in names], dtype=float)
    if np.any(np.diff(coords) <= 0):
        raise ValueError("coherence grid coordinates must be strictly increasing")
    bands = list(band_weights)
    if (any(not (value > 0) for value in band_weights.values())
            or not math.isclose(sum(band_weights.values()), 1.0, abs_tol=1e-12)):
        raise ValueError("band weights must be positive and sum to one")
    if bootstrap_reps < 2 or not (0 < alpha < 1):
        raise ValueError("bootstrap_reps/alpha are invalid")
    strata, cells, population = _validated_cells(rows, bands)
    cell_scores = np.empty((len(strata), len(bands), len(names)), dtype=float)
    cell_halfwidths = np.empty_like(cell_scores)
    for hi, stratum in enumerate(strata):
        for bi, band in enumerate(bands):
            cell = cells[(stratum, band)]
            for mi, name in enumerate(names):
                values = np.asarray([r["model_log_probability_proxy"][name]["mid"]
                                     for r in cell], dtype=float)
                halfwidths = np.asarray([
                    r["model_log_probability_proxy"][name]["halfwidth"] for r in cell
                ], dtype=float)
                if (np.any(~np.isfinite(values)) or np.any(~np.isfinite(halfwidths))
                        or np.any(halfwidths < 0)):
                    raise ValueError("invalid coherence-grid score interval")
                cell_scores[hi, bi, mi] = float(values.mean())
                cell_halfwidths[hi, bi, mi] = float(halfwidths.mean())
    mixture = np.asarray([band_weights[band] for band in bands], dtype=float)
    nominal_scores = np.zeros(len(names), dtype=float)
    point_halfwidths = np.zeros(len(names), dtype=float)
    for bi in range(len(bands)):
        normalized_population = population[:, bi] / population[:, bi].sum()
        nominal_scores += mixture[bi] * np.sum(
            normalized_population[:, None] * cell_scores[:, bi, :], axis=0)
        point_halfwidths += mixture[bi] * np.sum(
            normalized_population[:, None] * cell_halfwidths[:, bi, :], axis=0)
    reconstruction_draws = None
    reconstruction_event_scores = None
    reconstruction_center = np.zeros(len(names), dtype=float)
    if reconstruction_replicates is not None:
        reconstruction_event_scores, reconstruction_event_arith = \
            _aligned_reconstruction_event_models(rows, reconstruction_replicates, names)
        score_bands = _event_draw_band_means(
            rows, bands, reconstruction_event_scores)
        arithmetic_bands = _event_draw_band_means(
            rows, bands, reconstruction_event_arith)
        mixture = np.asarray([band_weights[band] for band in bands], dtype=float)
        reconstruction_draws = np.sum(
            score_bands * mixture[None, :, None], axis=1)
        reconstruction_center = reconstruction_draws.mean(axis=0)
        point_halfwidths = np.max(
            np.sum(arithmetic_bands * mixture[None, :, None], axis=1), axis=0)
    point_scores = (nominal_scores if reconstruction_draws is None
                    else reconstruction_center)
    best = int(np.argmax(point_scores))
    rng = np.random.default_rng(bootstrap_seed)
    if reconstruction_draws is None:
        cell_errors = _resample_model_cells(
            rows, bands, strata, names, bootstrap_reps, rng)
        reconstruction_indices = None
    else:
        reconstruction_indices = rng.integers(
            0, len(reconstruction_draws), bootstrap_reps)
        cell_errors = _resample_draw_specific_model_cells(
            rows, bands, strata, reconstruction_event_scores,
            reconstruction_indices, rng)
    bootstrap_errors = np.zeros((bootstrap_reps, len(names)), dtype=float)
    for bi in range(len(bands)):
        population_share = population[:, bi] / population[:, bi].sum()
        bootstrap_errors += mixture[bi] * np.sum(
            population_share[None, :, None] * cell_errors[:, :, bi, :], axis=1)
    if reconstruction_draws is not None:
        bootstrap_errors += (
            reconstruction_draws[reconstruction_indices] - reconstruction_center)
    confidence_set = _simultaneous_best_confidence_set(
        point_scores, bootstrap_errors, point_halfwidths, names, alpha=alpha)
    statistical_models = confidence_set["models"]
    statistical_indices = [names.index(name) for name in statistical_models]
    statistical_interval = [float(np.min(coords[statistical_indices])),
                            float(np.max(coords[statistical_indices]))]

    draw_scores = point_scores + bootstrap_errors
    draw_best = np.argmax(draw_scores, axis=1)
    coordinate_draws = coords[draw_best]
    point_lower = point_scores - point_halfwidths
    point_upper = point_scores + point_halfwidths
    argmax_counts = np.bincount(draw_best, minlength=len(names))
    return {
        "estimand": "best_predictive_anomalous_coherence_grid_point",
        "scope": ("reconstruction-marginal registered joint pattern-probability proxy"
                  if reconstruction_draws is not None else
                  "registered joint pattern-probability proxy on frozen binary64 states"),
        "model_order": names, "coordinates": coords.tolist(),
        "mean_log_scores": point_scores.tolist(),
        "arithmetic_score_halfwidths": point_halfwidths.tolist(),
        "score_intervals_with_arithmetic": [
            [float(point_lower[i]), float(point_upper[i])] for i in range(len(names))
        ],
        "estimate": float(coords[best]), "estimate_model": names[best],
        "statistical_confidence_set_models": statistical_models,
        "statistical_confidence_set_coordinate_interval": statistical_interval,
        "statistical_confidence_set": confidence_set,
        "confidence_set_excludes_classical_region": bool(
            statistical_interval[0] > classical_boundary),
        "descriptive_argmax_bootstrap": {
            "model_frequencies": {
                name: float(argmax_counts[i] / bootstrap_reps)
                for i, name in enumerate(names)
            },
            "coordinate_quantile_interval": [float(x) for x in np.quantile(
                coordinate_draws, [alpha / 2, 1 - alpha / 2])],
            "probability_at_or_below_classical_boundary": float(
                np.mean(coordinate_draws <= classical_boundary)),
            "inferential": False,
        },
        "classical_boundary": float(classical_boundary),
        "reconstruction_draws": (0 if reconstruction_draws is None
                                 else int(len(reconstruction_draws))),
        "bootstrap": {"reps": int(bootstrap_reps), "seed": int(bootstrap_seed),
                      "alpha": float(alpha),
                      "unit": "paired model-score vector within fixed band/stratum cell",
                      "strata": "fixed; never resampled",
                      "estimator": "fixed-stratum finite-population weighted mean",
                      "within_cell_bootstrap": (
                          "centered resampling with exact estimated SRSWOR "
                          "finite-population correction"),
                      "confidence_set_method": (
                          "simultaneous paired score-difference max-t with "
                          "Minkowski arithmetic expansion")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verified-run", required=True,
                    help="fail-closed reducer output containing immutable usable rows")
    ap.add_argument("--normalizer-replicates", required=True)
    ap.add_argument("--reconstruction-replicates")
    ap.add_argument("--predictive-checks")
    ap.add_argument("--refusal-analysis")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = load_json(args.verified_run)
    if run.get("schema") != "gbskernels.verified-run.v2" or not run.get("complete"):
        raise SystemExit("verified run is not complete")
    contract = run.get("contract", {})
    contract_body = {key: value for key, value in contract.items() if key != "run_id"}
    if (run.get("run_id") != contract.get("run_id")
            or contract.get("run_id") != hash_json(contract_body)):
        raise SystemExit("verified run contract hash is invalid")
    registration = run["registration"]
    normalized = validate_registration(registration, require_beacon=True)
    plan = normalized["plan"]
    commit = current_commit()
    if commit != plan["analysis_commit"]:
        raise SystemExit("current analysis commit differs from the registration")
    analysis_container = os.environ.get("GBS_CONTAINER_DIGEST")
    if (plan["external_requirements"].get("container_digest_required", True)
            and not valid_container_digest(analysis_container)):
        raise SystemExit("analysis requires a pinned GBS_CONTAINER_DIGEST")
    if analysis_container != contract.get("container_digest"):
        raise SystemExit("analysis container differs from the verified run")
    if (run.get("contract", {}).get("analysis_source_sha256")
            != plan["numerical_contract"]["analysis_source_sha256"]
            or analysis_source_hash() != run.get("contract", {}).get("analysis_source_sha256")):
        raise SystemExit("current analysis source differs from the registered run")
    bands = [int(x) for x in plan["selection"]["bands"]]
    if int(run.get("strata_count", 0)) != int(plan["selection"]["n_strata"]):
        raise SystemExit("verified run strata count differs from registration")
    for band in bands:
        registered = run.get("population", {}).get(str(band), {})
        if not registered or int(registered.get("eligible_total", -1)) <= 0:
            raise SystemExit(f"verified run lacks population audit for C={band}")
        design_pop = registered.get("eligible_by_stratum", [])
        if sum(int(x) for x in design_pop) != int(registered["eligible_total"]):
            raise SystemExit(f"population audit is inconsistent for C={band}")
        for stratum, size in enumerate(design_pop):
            observed = {int(row.get("eligible_in_stratum", -1))
                        for row in run["rows"]
                        if int(row["band"]) == band and int(row["stratum"]) == stratum}
            if observed != {int(size)}:
                raise SystemExit(f"row population audit differs for C={band}, h={stratum}")
    weights = {band: float(plan["analysis"]["band_weights"][str(band)])
               for band in bands}
    registration_id = normalized["public"]["plan_sha256"]
    with np.load(args.normalizer_replicates, allow_pickle=False) as normalizer_file:
        normalizer_meta = json.loads(str(normalizer_file["meta"]))
    expected_provenance = {
        "analysis_commit": run["contract"]["analysis_commit"],
        "analysis_source_sha256": run["contract"]["analysis_source_sha256"],
        "container_digest": run["contract"]["container_digest"],
    }
    if any(normalizer_meta.get(key) != value
           for key, value in expected_provenance.items()):
        raise SystemExit("normalizer provenance differs from the verified run")
    model_cfg = plan["models"]
    expected_model_names = [str(model_cfg["reference_model"]),
                            str(model_cfg["alternative_model"])] + sorted(
                                set(model_cfg["coherence_points"])
                                - {model_cfg["reference_model"],
                                   model_cfg["alternative_model"]})
    if [str(x) for x in normalizer_meta.get("model_names", [])] != expected_model_names:
        raise SystemExit("normalizer model order differs from registration")
    if [float(x) for x in normalizer_meta.get("coherence_points", [])] != [
            float(model_cfg["coherence_points"][name]) for name in expected_model_names]:
        raise SystemExit("normalizer coherence coordinates differ from registration")
    if (int(normalizer_meta.get("exp_id", -1)) != int(model_cfg["exp_id"])
            or normalizer_meta.get("parameterization", "classical_excess")
            != model_cfg.get("parameterization", "classical_excess")):
        raise SystemExit("normalizer state construction differs from registration")
    normalizers = load_joint_normalizer_replicates(
        args.normalizer_replicates, bands, registration_id=registration_id,
        expected_replicates=int(plan["analysis"]["normalizer_replicates"]["count"]),
        expected_samples_per_replicate=int(
            plan["analysis"]["normalizer_replicates"]["samples_per_replicate"]),
        expected_seed=int(plan["analysis"]["normalizer_replicates"]["seed"]))
    if run.get("n_refused", 0):
        if not args.refusal_analysis:
            raise SystemExit("refused events require an independent refusal-analysis artifact")
        refusal = load_json(args.refusal_analysis)
        try:
            validate_refusal_analysis(
                refusal, run=run, registration_id=registration_id,
                config=plan["analysis"]["refusal_analysis"],
                verified_run_sha256=sha256_file(args.verified_run),
                model_names=expected_model_names)
            analysis_rows = restore_recovered_primary_rows(run, refusal)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        refusal_analysis_sha256 = sha256_file(args.refusal_analysis)
    else:
        if args.refusal_analysis:
            raise SystemExit("a refusal analysis is forbidden when the run has no refusals")
        analysis_rows = list(run["rows"])
        refusal_analysis_sha256 = None
    if args.reconstruction_replicates:
        with np.load(args.reconstruction_replicates, allow_pickle=False) as reconstruction_file:
            reconstruction_meta = json.loads(str(reconstruction_file["meta"]))
        if any(reconstruction_meta.get(key) != value
               for key, value in expected_provenance.items()):
            raise SystemExit("reconstruction provenance differs from the verified run")
        reconstruction = load_reconstruction_replicates(
            args.reconstruction_replicates, bands,
            run_id=run["run_id"], registration_id=registration_id,
            manifest_id=run["contract"]["manifest_id"],
            expected_event_rows=analysis_rows,
            expected_draws=int(plan["analysis"]["calibration_draws"]["count"]),
            expected_model_names=expected_model_names,
            expected_coherence_points=[float(model_cfg["coherence_points"][name])
                                       for name in expected_model_names],
            calibration_sha256=plan["analysis"]["calibration_draws"][
                "posterior_sha256"],
            nominal_normalizers_sha256=sha256_file(args.normalizer_replicates),
            refusal_analysis_sha256=refusal_analysis_sha256)
    else:
        reconstruction = None
    if plan["external_requirements"].get("reconstruction_required", False) \
            and reconstruction is None:
        raise SystemExit("registration requires reconstruction replicates")
    cfg = plan["analysis"]
    scope = cfg.get("population_scope", "finite_registered_acquisition")
    if scope != "finite_registered_acquisition":
        raise SystemExit("process-level claims require a registered multi-acquisition analyzer")
    checks = load_json(args.predictive_checks) if args.predictive_checks else None
    predictive_passes = None
    if plan["external_requirements"].get("absolute_predictive_checks_required", False):
        try:
            predictive_passes = validate_absolute_predictive_checks(
                checks, run_id=run["run_id"], registration_id=registration_id,
                selection_cfg=plan["selection"],
                predictive_cfg=plan["analysis"]["predictive_checks"],
                model_names=expected_model_names,
                expected_provenance=expected_provenance)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    result = analyze(analysis_rows, band_weights=weights,
                     normalizer_replicates=normalizers,
                     reconstruction_replicates=reconstruction,
                     bootstrap_reps=int(cfg["bootstrap_reps"]),
                     bootstrap_seed=int(cfg["bootstrap_seed"]),
                     alpha=float(cfg["alpha"]))
    result["absolute_predictive_checks"] = checks
    coordinates = {str(k): float(v) for k, v in plan["models"]["coherence_points"].items()}
    if len(coordinates) >= 3:
        if any(set(row.get("model_log_probability_proxy", {})) != set(coordinates)
               for row in analysis_rows):
            raise SystemExit(
                "coherence-grid inference requires recovered per-model scores "
                "for every refused primary event"
            )
        result["coherence_grid"] = analyze_coherence_grid(
            analysis_rows, coordinates, band_weights=weights,
            classical_boundary=float(plan["models"].get("classical_boundary", 0.0)),
            bootstrap_reps=int(cfg["bootstrap_reps"]),
            bootstrap_seed=int(cfg["bootstrap_seed"]), alpha=float(cfg["alpha"]),
            reconstruction_replicates=reconstruction)
    if checks is not None and plan["external_requirements"].get(
            "absolute_predictive_checks_required", False):
        policy = plan["analysis"]["predictive_checks"].get(
            "model_pass_policy", "any_registered")
        result["predictive_model_gate"] = predictive_model_gate(
            result, predictive_passes, policy=policy, model_cfg=model_cfg)
    primary = str(cfg["primary_estimand"])
    if primary in result["estimands"]:
        primary_result = result["estimands"][primary]
    elif (result.get("coherence_grid", {}).get("estimand") == primary):
        primary_result = result["coherence_grid"]
    else:
        raise SystemExit(f"unknown registered primary estimand {primary!r}")
    result["registered_primary_estimand"] = primary
    result["registered_primary_result"] = primary_result
    result["registered_decision"] = registered_nonclassical_decision(result, cfg)
    inputs = {
        "verified_run_sha256": sha256_file(args.verified_run),
        "normalizer_replicates_sha256": sha256_file(args.normalizer_replicates),
    }
    if plan["external_requirements"].get("reconstruction_required", False):
        inputs["reconstruction_replicates_sha256"] = sha256_file(
            args.reconstruction_replicates)
    if plan["external_requirements"].get("absolute_predictive_checks_required", False):
        inputs["predictive_checks_sha256"] = sha256_file(args.predictive_checks)
    if run.get("n_refused", 0):
        inputs["refusal_analysis_sha256"] = sha256_file(args.refusal_analysis)
    artifact = {"schema": "gbskernels.confirmatory-analysis.v2",
                "run_id": run["run_id"],
                "registration_id": normalized["public"]["plan_sha256"],
                "analysis_commit": commit,
                "analysis_source_sha256": run["contract"]["analysis_source_sha256"],
                "container_digest": analysis_container,
                "inputs": inputs,
                "result": result}
    write_json_exclusive(args.out, artifact)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
