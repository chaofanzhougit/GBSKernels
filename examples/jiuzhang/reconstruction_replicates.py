"""Propagate registered calibration/posterior draws through every coherence model.

The calibration file is an external scientific input. This program makes its
use executable and auditable rather than silently treating one reconstructed
transfer matrix as ground truth. Every draw rebuilds the full registered family,
re-evaluates the same immutable event set, and pairs its normalizer draw with
the same draw index.

Calibration NPZ contract::

    meta = {"schema":"gbskernels.calibration-posterior.v1", "bands":[...],
            "n_modes":100, "n_sources":25,
            "source_artifacts":[{"url":"https://...", "sha256":"..."}],
            "inference_method":"...", "inference_code_sha256":"...",
            "created_utc":"...Z",
            "independent_of_analysis_acquisition":true}
    r25_draws       shape (draw, 25)
    T_real_draws    shape (draw, 100, 50)
    T_imag_draws    shape (draw, 100, 50)
    block_drift_real_draws / block_drift_imag_draws
                    shape (draw, n_strata, 100, 50)
    detector_efficiency_draws (optional when response is not folded into T)
                    shape (draw, 100)

The calibration-normalizer NPZ contains ``p_models`` with shape
``(draw, common_stratum, model, band)``. It is generated from the same
calibration draw and block-drift state by
``calibration_normalizer_replicates.py``. Nonzero dark clicks are refused
because the exact threshold-pattern convolution is not implemented at the
registered C=27--30 sizes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from campaign_confirmatory_v2 import (_default_evaluator, _manifest_event,
                                      load_manifest,
                                      validate_manifest_registration)  # noqa: E402
from analyze_refusals import validate_refusal_analysis  # noqa: E402
from confirmatory_common import (analysis_source_hash, current_commit, hash_json,
                                 load_json, sha256_file, valid_container_digest,
                                 write_npz_exclusive)  # noqa: E402
from confirmatory_contract import load_registration  # noqa: E402
from confirmatory_inference import restore_recovered_primary_rows  # noqa: E402
import coherence_family  # noqa: E402


CALIBRATION_FINGERPRINT_METHOD = (
    "sha256-v1 over labeled little-endian float64/complex128 draw arrays")
PAIRED_NORMALIZER_FINGERPRINT_METHOD = (
    "sha256-v1 over calibration draw fingerprint and selected p_models slice")
LOSS_VARIATION_RTOL = 1e-10
LOSS_VARIATION_ATOL = 1e-12


def _update_array_digest(digest: Any, name: str, values: np.ndarray) -> None:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        canonical = np.asarray(array, dtype="<c16", order="C")
        dtype_name = "complex128-le"
    else:
        canonical = np.asarray(array, dtype="<f8", order="C")
        dtype_name = "float64-le"
    header = json.dumps({"name": name, "dtype": dtype_name,
                         "shape": list(canonical.shape)},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(canonical.tobytes(order="C"))


def calibration_draw_fingerprints(calibration: dict[str, np.ndarray]) -> list[str]:
    """Return a stable physical-state fingerprint for each calibration draw."""
    required = ("r25", "T", "block_drift")
    if not all(name in calibration for name in required):
        raise ValueError("calibration fingerprints require squeezing, transfer, and drift")
    count = len(np.asarray(calibration["r25"]))
    if count < 1 or any(len(np.asarray(calibration[name])) != count for name in required):
        raise ValueError("calibration fingerprint draw axes are inconsistent")
    if "efficiency" in calibration and len(np.asarray(calibration["efficiency"])) != count:
        raise ValueError("calibration efficiency fingerprint axis is inconsistent")
    out = []
    for index in range(count):
        digest = hashlib.sha256(b"gbskernels.calibration-draw.v1\0")
        for name in required:
            _update_array_digest(digest, name, np.asarray(calibration[name])[index])
        if "efficiency" in calibration:
            _update_array_digest(
                digest, "efficiency", np.asarray(calibration["efficiency"])[index])
        else:
            digest.update(b"efficiency\0absent")
        out.append(digest.hexdigest())
    return out


def paired_normalizer_draw_fingerprints(
    calibration_fingerprints: list[str] | np.ndarray,
    p_models: np.ndarray,
) -> list[str]:
    """Bind each normalizer slice to its corresponding physical calibration draw."""
    fingerprints = [str(value) for value in calibration_fingerprints]
    probabilities = np.asarray(p_models, dtype=float)
    if probabilities.ndim != 4 or len(probabilities) != len(fingerprints):
        raise ValueError("normalizer fingerprint draw axes are inconsistent")
    out = []
    for index, fingerprint in enumerate(fingerprints):
        if not _valid_sha256(fingerprint):
            raise ValueError("invalid calibration draw fingerprint")
        digest = hashlib.sha256(b"gbskernels.paired-normalizer-draw.v1\0")
        digest.update(fingerprint.lower().encode("ascii"))
        _update_array_digest(digest, "p_models", probabilities[index])
        out.append(digest.hexdigest())
    return out


def _validate_transfer(transfer: np.ndarray) -> None:
    if transfer.ndim != 2 or not np.all(np.isfinite(transfer)):
        raise ValueError("transfer matrix is not finite and two-dimensional")
    singular = np.linalg.svd(transfer, compute_uv=False)
    if np.max(singular, initial=0.0) > 1.0 + 1e-10:
        raise ValueError("calibration transfer draw is not a contractive loss channel")


def _valid_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdefABCDEF" for char in value))


def _bound_manifest_event(manifest: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Rebuild an event and prove it is the verified row being propagated."""
    event = _manifest_event(manifest, int(row["band"]), int(row["position"]))
    for field in ("manifest_id", "event_id", "band", "position", "record_index",
                  "stratum", "input_sha256"):
        if event.get(field) != row.get(field):
            raise ValueError(
                f"reconstructed manifest event differs from verified row on {field}")
    return event


def _valid_https_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def _validate_calibration_provenance(meta: dict) -> None:
    sources = meta.get("source_artifacts")
    if not isinstance(sources, list) or not sources:
        raise ValueError("calibration provenance requires source artifacts")
    source_pairs = set()
    for source in sources:
        if (not isinstance(source, dict)
                or not _valid_https_url(source.get("url"))
                or not _valid_sha256(source.get("sha256"))):
            raise ValueError("calibration source artifacts require HTTPS URLs and SHA256 hashes")
        source_pairs.add((source["url"], source["sha256"].lower()))
    method = meta.get("inference_method")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("calibration provenance requires a nonempty inference method")
    if not _valid_sha256(meta.get("inference_code_sha256")):
        raise ValueError("calibration provenance requires an inference code SHA256")
    created = meta.get("created_utc")
    try:
        parsed = datetime.fromisoformat(
            created[:-1] + "+00:00" if isinstance(created, str) and created.endswith("Z")
            else created)
    except (TypeError, ValueError):
        parsed = None
    if parsed is None or parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("calibration provenance requires an aware UTC creation time")
    if meta.get("independent_of_analysis_acquisition") is not True:
        raise ValueError(
            "calibration posterior must be independent of the analysis acquisition")
    evidence = meta.get("dark_click_evidence")
    if (not isinstance(evidence, dict)
            or not _valid_https_url(evidence.get("source_url"))
            or not _valid_sha256(evidence.get("source_sha256"))
            or not isinstance(evidence.get("justification"), str)
            or not evidence["justification"].strip()):
        raise ValueError("explicitly zero dark clicks require hashed source evidence")
    evidence_pair = (evidence["source_url"], evidence["source_sha256"].lower())
    if evidence_pair not in source_pairs:
        raise ValueError("dark-click evidence must identify a calibration source artifact")


def _has_draw_variation(values: np.ndarray) -> bool:
    return (len(values) >= 2
            and any(not np.array_equal(values[index], values[0])
                    for index in range(1, len(values))))


def _require_draw_variation(name: str, values: np.ndarray) -> None:
    if not _has_draw_variation(values):
        raise ValueError(f"calibration posterior has no draw variation for {name}")


def _has_loss_variation(transfers: np.ndarray) -> bool:
    """Distinguish throughput/loss uncertainty from phase-only transfer changes."""
    reference_singular = np.linalg.svd(transfers[0], compute_uv=False)
    reference_throughput = np.sum(np.abs(transfers[0]) ** 2, axis=0)
    return any(
        not np.allclose(np.linalg.svd(transfers[index], compute_uv=False),
                        reference_singular, rtol=LOSS_VARIATION_RTOL,
                        atol=LOSS_VARIATION_ATOL)
        or not np.allclose(np.sum(np.abs(transfers[index]) ** 2, axis=0),
                           reference_throughput, rtol=LOSS_VARIATION_RTOL,
                           atol=LOSS_VARIATION_ATOL)
        for index in range(1, len(transfers)))


def load_calibration(path: str | Path, bands: list[int], *, n_strata: int) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    if not {"r25_draws", "T_real_draws", "T_imag_draws", "meta"}.issubset(z.files):
        raise ValueError("calibration posterior lacks required arrays")
    meta = json.loads(str(z["meta"]))
    if not isinstance(meta, dict) or meta.get("schema") != "gbskernels.calibration-posterior.v1":
        raise ValueError("unsupported calibration posterior schema")
    _validate_calibration_provenance(meta)
    if [int(x) for x in meta.get("bands", [])] != bands:
        raise ValueError("calibration band order mismatch")
    r = np.asarray(z["r25_draws"], dtype=float)
    re = np.asarray(z["T_real_draws"], dtype=float)
    im = np.asarray(z["T_imag_draws"], dtype=float)
    if r.ndim != 2 or re.ndim != 3 or im.shape != re.shape \
            or re.shape[0] != r.shape[0] or re.shape[2] != 2 * r.shape[1]:
        raise ValueError("calibration posterior array shapes are inconsistent")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(re)) or not np.all(np.isfinite(im)):
        raise ValueError("calibration posterior contains non-finite values")
    if len(r) < 2:
        raise ValueError("calibration posterior requires at least two distinct joint draws")
    if (int(meta.get("n_sources", r.shape[1])) != r.shape[1]
            or int(meta.get("n_modes", re.shape[1])) != re.shape[1]):
        raise ValueError("calibration metadata dimensions do not match arrays")
    out = {"r25": r, "T": re + 1j * im}
    for transfer in out["T"]:
        _validate_transfer(transfer)
    _require_draw_variation("squeezing", out["r25"])
    _require_draw_variation("transfer", out["T"])
    if not _has_loss_variation(out["T"]):
        raise ValueError("calibration posterior has no draw variation for loss/throughput")
    if "detector_efficiency_draws" in z.files:
        efficiency = np.asarray(z["detector_efficiency_draws"], dtype=float)
        if (efficiency.shape != (r.shape[0], re.shape[1])
                or np.any(~np.isfinite(efficiency))
                or np.any((efficiency <= 0) | (efficiency > 1))):
            raise ValueError("detector efficiency draws have invalid shape or range")
        out["efficiency"] = efficiency
    if "block_drift_real_draws" in z.files or "block_drift_imag_draws" in z.files:
        if not {"block_drift_real_draws", "block_drift_imag_draws"}.issubset(z.files):
            raise ValueError("both real and imaginary block drift draws are required")
        dr = np.asarray(z["block_drift_real_draws"], dtype=float)
        di = np.asarray(z["block_drift_imag_draws"], dtype=float)
        expected = (r.shape[0], n_strata, re.shape[1], re.shape[2])
        if dr.shape != expected or di.shape != expected or not np.all(np.isfinite(dr + di)):
            raise ValueError("block drift draws have invalid shape")
        out["block_drift"] = dr + 1j * di
    meta_nuisances = set(meta.get("nuisance_families", []))
    if not {"squeezing", "transfer", "loss", "block_drift"}.issubset(meta_nuisances):
        raise ValueError("calibration metadata must name squeezing, transfer, loss, and block_drift")
    if "block_drift" not in out:
        raise ValueError("registered calibration posterior requires block drift draws")
    _require_draw_variation("block_drift", out["block_drift"])
    if (n_strata >= 2 and not any(
            not np.array_equal(draw[stratum], draw[0])
            for draw in out["block_drift"] for stratum in range(1, n_strata))):
        raise ValueError("block drift posterior has no variation across acquisition strata")
    response_model = meta.get("detector_response_model")
    if response_model not in {
            "folded_into_transfer_posterior", "explicit_efficiency_draws"}:
        raise ValueError("calibration metadata must state the detector response model")
    semantics = meta.get("detector_response_semantics")
    if not isinstance(semantics, dict):
        raise ValueError("calibration metadata must state detector-response semantics")
    if response_model == "folded_into_transfer_posterior":
        if (semantics.get("transfer_draws_include_detector_efficiency") is not True
                or semantics.get("efficiency_draws") != "none"):
            raise ValueError("folded detector response semantics are inconsistent")
        if "efficiency" in out:
            raise ValueError("folded detector response must not include separate efficiency draws")
    else:
        if (semantics.get("transfer_draws_include_detector_efficiency") is not False
                or semantics.get("efficiency_draws") != "absolute_per_output_mode"):
            raise ValueError("explicit detector response semantics are inconsistent")
        if "efficiency" not in out:
            raise ValueError("explicit detector response requires efficiency draws")
        _require_draw_variation("detector efficiency", out["efficiency"])
    dark_model = meta.get("dark_click_model")
    if dark_model != "explicitly_zero":
        raise ValueError("unsupported dark-click model; register an explicit convolution")
    return out


def generate(registration: dict, manifest: dict, rows: list[dict], calibration: dict,
             normalizer_path: str | Path,
             nominal_normalizer_path: str | Path) -> dict[str, np.ndarray]:
    plan = registration["plan"]
    bands = [int(x) for x in plan["selection"]["bands"]]
    models = plan["models"]
    reference = str(models["reference_model"])
    alternative = str(models["alternative_model"])
    points = models["coherence_points"]
    order = [reference, alternative] + sorted(set(points) - {reference, alternative})
    parameterization = str(models.get("parameterization", "classical_excess"))
    strata = sorted({int(row["stratum"]) for row in rows})
    if strata != list(range(int(plan["selection"]["n_strata"]))):
        raise ValueError("verified rows do not cover every registered common stratum")
    nz = np.load(normalizer_path, allow_pickle=False)
    if "meta" not in nz.files:
        raise ValueError("draw-level normalizer artifact lacks metadata")
    draw_meta = json.loads(str(nz["meta"]))
    if draw_meta.get("schema") != "gbskernels.calibration-normalizer-draws.v1":
        raise ValueError("unsupported draw-level normalizer schema")
    if [int(x) for x in draw_meta.get("bands", [])] != bands:
        raise ValueError("draw-level normalizer band order mismatch")
    if int(draw_meta.get("n_strata", -1)) != len(strata):
        raise ValueError("draw-level normalizer stratum count mismatch")
    if (not isinstance(draw_meta.get("calibration_posterior_sha256"), str)
            or len(draw_meta["calibration_posterior_sha256"]) != 64
            or draw_meta.get("pairing") != "calibration_draw_and_common_stratum"):
        raise ValueError("draw-level normalizer calibration pairing is not auditable")
    if draw_meta.get("registration_id") != registration["public"]["plan_sha256"]:
        raise ValueError("draw-level normalizer is bound to a different registration")
    expected_names = [reference, alternative] + sorted(
        set(points) - {reference, alternative})
    if [str(x) for x in draw_meta.get("model_names", [])] != expected_names:
        raise ValueError("draw-level normalizer model order differs from registration")
    if [float(x) for x in draw_meta.get("coherence_points", [])] != [
            float(points[name]) for name in expected_names]:
        raise ValueError("draw-level normalizer coherence coordinates differ from registration")
    if (int(draw_meta.get("exp_id", -1)) != int(models["exp_id"])
            or draw_meta.get("parameterization", "classical_excess")
            != parameterization):
        raise ValueError("draw-level normalizer state construction differs from registration")
    if "p_models" not in nz.files:
        raise ValueError("draw-level normalizer artifact lacks the model tensor")
    draw_probabilities = np.asarray(nz["p_models"], dtype=float)
    expected_shape = (len(calibration["r25"]), len(strata), len(order), len(bands))
    if draw_probabilities.shape != expected_shape:
        raise ValueError("draw-level normalizer tensor does not match draw/stratum/model/band")
    if (np.any(~np.isfinite(draw_probabilities))
            or np.any(draw_probabilities <= 0)
            or np.any(draw_probabilities > 1)):
        raise ValueError("draw-level normalizer probabilities must lie in (0, 1]")
    normalizer_cfg = plan["analysis"]["normalizer_replicates"]
    calibration_cfg = plan["analysis"]["calibration_draws"]
    expected_seed_rule = "seed + draw_index * n_strata + stratum; common across models"
    if (draw_meta.get("draws") != len(calibration["r25"])
            or draw_meta.get("samples_per_draw_stratum")
            != int(normalizer_cfg["samples_per_replicate"])
            or draw_meta.get("seed") != int(calibration_cfg["seed"])
            or draw_meta.get("seed_rule") != expected_seed_rule
            or draw_meta.get("calibration_posterior_sha256")
            != calibration_cfg["posterior_sha256"]):
        raise ValueError("calibration normalizer effort or seed differs from registration")
    if (draw_meta.get("calibration_draw_fingerprint_method")
            != CALIBRATION_FINGERPRINT_METHOD
            or draw_meta.get("paired_normalizer_fingerprint_method")
            != PAIRED_NORMALIZER_FINGERPRINT_METHOD
            or not {"calibration_draw_sha256",
                    "paired_normalizer_draw_sha256"}.issubset(nz.files)):
        raise ValueError("calibration normalizer lacks draw-level pairing fingerprints")
    expected_calibration_fingerprints = calibration_draw_fingerprints(calibration)
    stored_calibration_fingerprints = [
        str(value) for value in np.asarray(nz["calibration_draw_sha256"])]
    if stored_calibration_fingerprints != expected_calibration_fingerprints:
        raise ValueError("calibration normalizer draw order differs from the posterior")
    expected_paired_fingerprints = paired_normalizer_draw_fingerprints(
        expected_calibration_fingerprints, draw_probabilities)
    stored_paired_fingerprints = [
        str(value) for value in np.asarray(nz["paired_normalizer_draw_sha256"])]
    if stored_paired_fingerprints != expected_paired_fingerprints:
        raise ValueError("calibration normalizer payload is not paired to its draw labels")
    p0 = draw_probabilities[:, :, 0, :]
    p1 = draw_probabilities[:, :, 1, :]
    by_band = {band: [r for r in rows if int(r["band"]) == band] for band in bands}
    population = np.empty((len(strata), len(bands)), dtype=float)
    for hi, stratum in enumerate(strata):
        for bi, band in enumerate(bands):
            sizes = {int(row["eligible_in_stratum"])
                     for row in by_band[band] if int(row["stratum"]) == stratum}
            if len(sizes) != 1 or next(iter(sizes)) <= 0:
                raise ValueError("verified rows have invalid stratum population metadata")
            population[hi, bi] = next(iter(sizes))
    population /= population.sum(axis=0, keepdims=True)
    normalizer_log_ratio = np.sum(
        population[None, :, :] * (np.log(p0) - np.log(p1)), axis=1)
    event_rows = list(rows)
    event_ids = [str(row["event_id"]) for row in event_rows]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("verified reconstruction rows require unique event IDs")
    event_model_mid = np.empty(
        (len(draw_probabilities), len(event_rows), len(order)), dtype=float)
    event_model_arith = np.empty_like(event_model_mid)
    nominal_z = np.load(nominal_normalizer_path, allow_pickle=False)
    if "meta" not in nominal_z.files:
        raise ValueError("nominal normalizer artifact lacks metadata")
    nominal_meta = json.loads(str(nominal_z["meta"]))
    if (nominal_meta.get("schema") != "gbskernels.joint-normalizer-replicates.v1"
            or [int(x) for x in nominal_meta.get("bands", [])] != bands
            or nominal_meta.get("registration_id") != registration["public"]["plan_sha256"]):
        raise ValueError("nominal normalizer artifact is bound incorrectly")
    if [str(x) for x in nominal_meta.get("model_names", [])] != expected_names:
        raise ValueError("nominal normalizer model order differs from registration")
    if [float(x) for x in nominal_meta.get("coherence_points", [])] != [
            float(points[name]) for name in expected_names]:
        raise ValueError("nominal normalizer coherence coordinates differ from registration")
    if (int(nominal_meta.get("exp_id", -1)) != int(models["exp_id"])
            or nominal_meta.get("parameterization", "classical_excess")
            != parameterization):
        raise ValueError("nominal normalizer state construction differs from registration")
    nominal_p0 = np.asarray(nominal_z["p_reference"], dtype=float)
    nominal_p1 = np.asarray(nominal_z["p_alternative"], dtype=float)
    if (nominal_p0.ndim != 2 or nominal_p1.shape != nominal_p0.shape
            or nominal_p0.shape[1] != len(bands)
            or np.any(~np.isfinite(nominal_p0)) or np.any(~np.isfinite(nominal_p1))
            or np.any(nominal_p0 <= 0) or np.any(nominal_p1 <= 0)):
        raise ValueError("nominal normalizer artifact has invalid shape")
    if np.any(nominal_p0 > 1) or np.any(nominal_p1 > 1):
        raise ValueError("nominal normalizer probabilities must lie in (0, 1]")
    band_weights = {int(k): float(v) for k, v in plan["analysis"]["band_weights"].items()}
    if not np.isclose(sum(band_weights.values()), 1.0):
        raise ValueError("registered band weights must sum to one")
    for d, (r25, transfer) in enumerate(zip(calibration["r25"], calibration["T"], strict=True)):
        states_by_stratum = {}
        for stratum in strata:
            transfer_s = np.asarray(transfer, dtype=np.complex128).copy()
            if "efficiency" in calibration:
                transfer_s *= np.sqrt(calibration["efficiency"][d])[:, None]
            if "block_drift" in calibration:
                transfer_s += calibration["block_drift"][d, stratum]
            _validate_transfer(transfer_s)
            states = {}
            for name in order:
                states[name] = coherence_family.jiuzhang_state(
                    float(points[name]), exp_id=int(models["exp_id"]),
                    parameterization=parameterization,
                    calibration={"r25": r25, "T_out_by_in": transfer_s})
            states_by_stratum[stratum] = states
        for ei, row in enumerate(event_rows):
            # Reconstruct the manifest event from its stable position. The
            # verified row retains position and band; no row-order guess is used.
            ev = _bound_manifest_event(manifest, row)
            states = states_by_stratum[int(row["stratum"])]
            result = _default_evaluator(states, ev)
            if result.get("refused"):
                raise ValueError(f"calibration draw {d} refused event {row['event_id']}")
            proxy = result.get("model_log_probability_proxy", {})
            if set(proxy) != set(order):
                raise ValueError("calibration evaluator did not return the registered model family")
            event_model_mid[d, ei] = [float(proxy[name]["mid"]) for name in order]
            event_model_arith[d, ei] = [
                float(proxy[name]["halfwidth"]) for name in order]
            if (np.any(~np.isfinite(event_model_mid[d, ei]))
                    or np.any(~np.isfinite(event_model_arith[d, ei]))
                    or np.any(event_model_arith[d, ei] < 0)):
                raise ValueError("calibration evaluator returned an invalid score enclosure")

    model_scores = np.empty(
        (len(draw_probabilities), len(order), len(bands)), dtype=float)
    model_arith = np.empty_like(model_scores)
    for bi, band in enumerate(bands):
        indices = [index for index, row in enumerate(event_rows)
                   if int(row["band"]) == band]
        if not indices:
            raise ValueError(f"band {band} has no rows")
        probabilities = np.asarray(
            [float(event_rows[index]["inclusion_probability"]) for index in indices])
        if (np.any(~np.isfinite(probabilities)) or np.any(probabilities <= 0)
                or np.any(probabilities > 1)):
            raise ValueError("verified rows contain invalid inclusion probabilities")
        weights = 1.0 / probabilities
        model_scores[:, :, bi] = np.sum(
            event_model_mid[:, indices, :] * weights[None, :, None], axis=1
        ) / np.sum(weights)
        model_arith[:, :, bi] = np.sum(
            event_model_arith[:, indices, :] * weights[None, :, None], axis=1
        ) / np.sum(weights)
    joint = model_scores[:, 1, :] - model_scores[:, 0, :]
    return {
        "joint_log_score_band_draws": joint,
        "normalizer_log_ratio_band_draws": normalizer_log_ratio,
        "model_log_score_band_draws": model_scores,
        "model_arith_halfwidth_band_draws": model_arith,
        "joint_arith_halfwidth_band_draws": (
            model_arith[:, 0, :] + model_arith[:, 1, :]),
        "event_model_log_score_draws": event_model_mid,
        "event_model_arith_halfwidth_draws": event_model_arith,
        "event_ids": np.asarray(event_ids),
        "event_bands": np.asarray([int(row["band"]) for row in event_rows], dtype=int),
        "event_strata": np.asarray(
            [int(row["stratum"]) for row in event_rows], dtype=int),
        "event_positions": np.asarray(
            [int(row["position"]) for row in event_rows], dtype=int),
        "event_record_indices": np.asarray(
            [int(row["record_index"]) for row in event_rows], dtype=int),
        "event_input_sha256": np.asarray(
            [str(row["input_sha256"]) for row in event_rows]),
        "event_inclusion_probability": np.asarray(
            [float(row["inclusion_probability"]) for row in event_rows], dtype=float),
        "event_eligible_in_stratum": np.asarray(
            [int(row["eligible_in_stratum"]) for row in event_rows], dtype=int),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registration", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--verified-run", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--normalizer-draws", type=Path, required=True)
    ap.add_argument("--nominal-normalizers", type=Path, required=True)
    ap.add_argument("--refusal-analysis", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    registration = load_registration(args.registration)
    plan = registration["plan"]
    commit = current_commit()
    if commit != plan["analysis_commit"]:
        raise SystemExit("reconstruction generator commit differs from registration")
    source_hash = analysis_source_hash()
    if source_hash != plan["numerical_contract"]["analysis_source_sha256"]:
        raise SystemExit("reconstruction generator source bytes differ from registration")
    container_digest = os.environ.get("GBS_CONTAINER_DIGEST")
    if (plan["external_requirements"].get("container_digest_required", True)
            and not valid_container_digest(container_digest)):
        raise SystemExit("reconstruction generation requires GBS_CONTAINER_DIGEST")
    try:
        manifest, manifest_id = load_manifest(args.manifest)
        validate_manifest_registration(manifest, registration)
    except ValueError as exc:
        raise SystemExit(f"selection manifest is invalid: {exc}") from exc
    run = load_json(args.verified_run)
    if not run.get("complete"):
        raise SystemExit("reconstruction propagation requires a complete verified run")
    contract = run.get("contract", {})
    if (run.get("run_id") != contract.get("run_id")
            or contract.get("run_id") != hash_json(
                {key: value for key, value in contract.items() if key != "run_id"})):
        raise SystemExit("verified run contract hash is invalid")
    if contract.get("manifest_id") != manifest_id:
        raise SystemExit("verified run is bound to a different selection manifest")
    if run.get("registration", {}).get("public") != registration["public"]:
        raise SystemExit("run and registration do not match")
    model_cfg = plan["models"]
    reference = str(model_cfg["reference_model"])
    alternative = str(model_cfg["alternative_model"])
    model_names = [reference, alternative] + sorted(
        set(map(str, model_cfg["coherence_points"])) - {reference, alternative})
    refusal_analysis_sha256 = None
    if run.get("n_refused", 0):
        if args.refusal_analysis is None:
            raise SystemExit("refused primaries require --refusal-analysis for reconstruction")
        refusal = load_json(args.refusal_analysis)
        try:
            validate_refusal_analysis(
                refusal, run=run,
                registration_id=registration["public"]["plan_sha256"],
                config=plan["analysis"]["refusal_analysis"],
                verified_run_sha256=sha256_file(args.verified_run),
                model_names=model_names)
            analysis_rows = restore_recovered_primary_rows(run, refusal)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        refusal_analysis_sha256 = sha256_file(args.refusal_analysis)
    else:
        if args.refusal_analysis is not None:
            raise SystemExit("a refusal analysis is forbidden when the run has no refusals")
        analysis_rows = list(run["rows"])
    bands = [int(x) for x in registration["plan"]["selection"]["bands"]]
    expected_calibration = registration["plan"]["analysis"]["calibration_draws"][
        "posterior_sha256"]
    if sha256_file(args.calibration) != expected_calibration:
        raise SystemExit("calibration posterior hash differs from registration")
    with np.load(args.normalizer_draws, allow_pickle=False) as draw_file:
        draw_meta = json.loads(str(draw_file["meta"]))
    if draw_meta.get("calibration_posterior_sha256") != expected_calibration:
        raise SystemExit("normalizer draws are not paired to the registered calibration posterior")
    if draw_meta.get("pairing") != "calibration_draw_and_common_stratum":
        raise SystemExit("normalizer draw pairing rule is not registered")
    if any(draw_meta.get(key) != contract.get(key) for key in (
            "analysis_commit", "analysis_source_sha256", "container_digest")):
        raise SystemExit("calibration-normalizer provenance differs from the verified run")
    calibration = load_calibration(args.calibration, bands,
                                   n_strata=int(registration["plan"]["selection"]["n_strata"]))
    expected_draws = int(registration["plan"]["analysis"]["calibration_draws"]["count"])
    if len(calibration["r25"]) != expected_draws:
        raise SystemExit("calibration draw count differs from registration")
    output = generate(registration, manifest, analysis_rows, calibration,
                      args.normalizer_draws, args.nominal_normalizers)
    event_identity = [{
        "event_id": str(row["event_id"]), "band": int(row["band"]),
        "stratum": int(row["stratum"]), "position": int(row["position"]),
        "record_index": int(row["record_index"]),
        "input_sha256": str(row["input_sha256"]),
        "inclusion_probability": float(row["inclusion_probability"]),
        "eligible_in_stratum": int(row["eligible_in_stratum"]),
    } for row in analysis_rows]
    meta = {"schema": "gbskernels.reconstruction-replicates.v3", "bands": bands,
            "draws": len(output["joint_log_score_band_draws"]),
            "run_id": run["run_id"],
            "registration_id": registration["public"]["plan_sha256"],
            "manifest_id": manifest_id,
            "event_count": len(analysis_rows),
            "event_identity_sha256": hash_json(event_identity),
            "analysis_commit": commit, "analysis_source_sha256": source_hash,
            "container_digest": container_digest,
            "calibration_sha256": sha256_file(args.calibration),
            "normalizer_draws_sha256": sha256_file(args.normalizer_draws),
            "nominal_normalizers_sha256": sha256_file(args.nominal_normalizers),
            "model_names": model_names,
            "coherence_points": [
                float(registration["plan"]["models"]["coherence_points"][name])
                for name in ([registration["plan"]["models"]["reference_model"],
                              registration["plan"]["models"]["alternative_model"]]
                             + sorted(set(registration["plan"]["models"]["coherence_points"])
                                      - {registration["plan"]["models"]["reference_model"],
                                         registration["plan"]["models"]["alternative_model"]}))
            ],
            "reference_model": registration["plan"]["models"]["reference_model"],
            "alternative_model": registration["plan"]["models"]["alternative_model"],
            "normalizer_pairing": (
                "same calibration draw and common stratum; coherent hypotheses"),
            "loss_variation_rtol": LOSS_VARIATION_RTOL,
            "loss_variation_atol": LOSS_VARIATION_ATOL,
            "nuisance_families": ["squeezing", "transfer", "loss", "block_drift"]}
    if refusal_analysis_sha256 is not None:
        meta["refusal_analysis_sha256"] = refusal_analysis_sha256
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    write_npz_exclusive(args.out, meta=json.dumps(meta, sort_keys=True), **output)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
