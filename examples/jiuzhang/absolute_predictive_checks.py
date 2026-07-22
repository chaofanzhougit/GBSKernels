"""Absolute held-out checks for every registered anomalous-coherence model.

This scans the complete registered acquisition and compares click-count mass,
detector marginals, and pre-registered detector-pair correlations with model
predictions. It is deliberately separate from relative log-score inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from campaign_confirmatory_v2 import (_load_registered_states, load_manifest,
                                      validate_manifest_registration)  # noqa: E402
from confirmatory_common import (analysis_source_hash, current_commit, hash_json, load_json,
                                 valid_container_digest,
                                 write_json_exclusive)  # noqa: E402
from confirmatory_contract import load_registration  # noqa: E402
from select_confirmatory_v2 import RECORD_BYTES, decode_records  # noqa: E402


def model_observables(state: dict, pairs: list[tuple[int, int]]) -> dict:
    Q = np.asarray(state["Q"])
    modes = len(Q) // 2
    idx2 = np.asarray([[i, i + modes] for i in range(modes)])
    minors = Q[idx2[:, :, None], idx2[:, None, :]]
    det2 = np.linalg.det(minors).real
    if np.any(~np.isfinite(det2)) or np.any(det2 <= 0):
        raise ValueError("model Husimi minors are not positive")
    p_off = 1.0 / np.sqrt(det2)
    p_click = 1.0 - p_off
    if np.any(p_click < -1e-12) or np.any(p_click > 1 + 1e-12):
        raise ValueError("model detector marginals lie outside [0,1]")
    pair_joint = []
    pair_cov = []
    for i, j in pairs:
        if not (0 <= i < j < modes):
            raise ValueError(f"detector pair {(i, j)} outside model modes")
        idx = [i, j, i + modes, j + modes]
        det4 = float(np.linalg.det(Q[np.ix_(idx, idx)]).real)
        if not np.isfinite(det4) or det4 <= 0:
            raise ValueError("model pair Husimi minor is not positive")
        neither = 1.0 / np.sqrt(det4)
        both = 1.0 - p_off[i] - p_off[j] + neither
        pair_joint.append(float(both))
        pair_cov.append(float(both - p_click[i] * p_click[j]))
    return {"click_marginals": p_click, "pair_joint": np.asarray(pair_joint),
            "pair_covariance": np.asarray(pair_cov)}


def empirical_observables(patterns: np.ndarray, pairs: list[tuple[int, int]]) -> dict:
    patterns = np.asarray(patterns, dtype=bool)
    if patterns.ndim != 2 or len(patterns) == 0:
        raise ValueError("patterns must be a non-empty 2D array")
    if any(not (0 <= i < j < patterns.shape[1]) for i, j in pairs):
        raise ValueError("detector pair outside empirical pattern width")
    marginal = patterns.mean(axis=0)
    joint = np.asarray([(patterns[:, i] & patterns[:, j]).mean() for i, j in pairs])
    cov = np.asarray([joint[k] - marginal[i] * marginal[j]
                      for k, (i, j) in enumerate(pairs)])
    hist = np.bincount(patterns.sum(axis=1), minlength=patterns.shape[1] + 1)
    return {"n": len(patterns), "click_marginals": marginal,
            "pair_joint": joint, "pair_covariance": cov, "click_histogram": hist}


def compare(empirical: dict, model: dict, click_probability: np.ndarray) -> dict:
    hist = np.asarray(empirical["click_histogram"], dtype=float)
    emp_p = hist / hist.sum()
    mod_p = np.clip(np.asarray(click_probability, dtype=float)[: len(emp_p)], 0, None)
    if mod_p.sum() <= 0:
        raise ValueError("model click-count probabilities have no positive mass")
    mod_p /= mod_p.sum()
    dm = np.asarray(empirical["click_marginals"]) - np.asarray(model["click_marginals"])
    dc = np.asarray(empirical["pair_covariance"]) - np.asarray(model["pair_covariance"])
    return {
        "click_count_tv": float(0.5 * np.abs(emp_p - mod_p).sum()),
        "click_count_ks": float(np.max(np.abs(np.cumsum(emp_p) - np.cumsum(mod_p)))),
        "empirical_mean_clicks": float(np.dot(np.arange(len(emp_p)), emp_p)),
        "model_mean_clicks": float(np.dot(np.arange(len(mod_p)), mod_p)),
        "marginal_rms": float(np.sqrt(np.mean(dm * dm))),
        "marginal_max_abs": float(np.max(np.abs(dm))),
        "pair_covariance_rms": float(np.sqrt(np.mean(dc * dc))) if len(dc) else None,
        "pair_covariance_max_abs": float(np.max(np.abs(dc))) if len(dc) else None,
    }


def scan_data(path: Path, *, chunk_records: int,
              pairs: list[tuple[int, int]],
              exclusions: list[int] | tuple[int, ...] = ()) -> tuple[dict, str]:
    exclusions = np.asarray(sorted(int(x) for x in exclusions), dtype=np.int64)
    if len(exclusions) != len(np.unique(exclusions)):
        raise ValueError("exclusion indices contain duplicates")
    marg_sum = None
    pair_sum = np.zeros(len(pairs), dtype=np.int64)
    hist = None
    n = 0
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        record_index = 0
        while block := fh.read(chunk_records * RECORD_BYTES):
            digest.update(block)
            raw = np.frombuffer(block, dtype=np.uint8).reshape(-1, RECORD_BYTES)
            _, patterns, abnormal = decode_records(raw)
            indices = record_index + np.arange(len(patterns), dtype=np.int64)
            excluded = np.isin(indices, exclusions, assume_unique=True)
            patterns = patterns[(~abnormal) & (~excluded)]
            record_index += len(raw)
            if marg_sum is None:
                marg_sum = np.zeros(patterns.shape[1], dtype=np.int64)
                hist = np.zeros(patterns.shape[1] + 1, dtype=np.int64)
            marg_sum += patterns.sum(axis=0)
            for k, (i, j) in enumerate(pairs):
                pair_sum[k] += int(np.sum(patterns[:, i] & patterns[:, j]))
            hist += np.bincount(patterns.sum(axis=1), minlength=len(hist))
            n += len(patterns)
    if marg_sum is None or hist is None or n == 0:
        raise ValueError("registered acquisition contains no normal records")
    marginal = marg_sum / n
    pair_joint = pair_sum / n
    pair_cov = np.asarray([pair_joint[k] - marginal[i] * marginal[j]
                           for k, (i, j) in enumerate(pairs)])
    return {"n": n, "click_marginals": marginal, "pair_joint": pair_joint,
            "pair_covariance": pair_cov, "click_histogram": hist}, digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registration", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="canonical v2 selection manifest (binds exclusions)")
    ap.add_argument("--verified-run", type=Path, required=True)
    ap.add_argument("--normalizer-replicates", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk-records", type=int, default=2_000_000)
    args = ap.parse_args()
    reg = load_registration(args.registration)
    plan = reg["plan"]
    commit = current_commit()
    if commit != plan["analysis_commit"]:
        raise SystemExit("predictive-check commit differs from registration")
    source_hash = analysis_source_hash()
    if source_hash != plan["numerical_contract"]["analysis_source_sha256"]:
        raise SystemExit("predictive-check source bytes differ from registration")
    container_digest = os.environ.get("GBS_CONTAINER_DIGEST")
    if (plan["external_requirements"].get("container_digest_required", True)
            and not valid_container_digest(container_digest)):
        raise SystemExit("predictive checks require GBS_CONTAINER_DIGEST")
    manifest, _ = load_manifest(args.manifest)
    validate_manifest_registration(manifest, reg)
    run = load_json(args.verified_run)
    if not run.get("complete") or run.get("registration", {}).get("public") != reg["public"]:
        raise SystemExit("verified run and registration do not match")
    contract = run.get("contract", {})
    if (run.get("run_id") != contract.get("run_id")
            or contract.get("run_id") != hash_json(
                {key: value for key, value in contract.items() if key != "run_id"})):
        raise SystemExit("verified run contract hash is invalid")
    cfg = plan["analysis"]["predictive_checks"]
    pairs = [tuple(map(int, pair)) for pair in cfg["detector_pairs"]]
    exclusions = [int(x) for x in manifest["exclusions"]["record_indices"]]
    empirical, data_hash = scan_data(
        args.data, chunk_records=args.chunk_records, pairs=pairs, exclusions=exclusions)
    if data_hash != plan["selection"]["source_raw_sha256"]:
        raise SystemExit("raw acquisition hash differs from registration")
    z = np.load(args.normalizer_replicates, allow_pickle=False)
    if not {"p_models_full", "meta"}.issubset(z.files):
        raise SystemExit("normalizer artifact lacks full coherence-grid probabilities")
    meta = json.loads(str(z["meta"]))
    if meta.get("registration_id") != reg["public"]["plan_sha256"]:
        raise SystemExit("normalizer artifact belongs to a different registration")
    if (int(meta.get("exp_id", -1)) != int(plan["models"]["exp_id"])
            or meta.get("parameterization", "classical_excess")
            != plan["models"].get("parameterization", "classical_excess")):
        raise SystemExit("normalizer state construction differs from registration")
    names = [str(x) for x in meta["model_names"]]
    states = _load_registered_states(load_json(args.registration))
    if names != list(states):
        raise SystemExit("normalizer model order differs from registered states")
    registered_points = plan["models"]["coherence_points"]
    expected_points = [float(registered_points[name]) for name in names]
    if [float(x) for x in meta.get("coherence_points", [])] != expected_points:
        raise SystemExit("normalizer coherence coordinates differ from registration")
    model_draws = np.asarray(z["p_models_full"], dtype=float)
    if (model_draws.ndim != 3 or model_draws.shape[1] != len(names)
            or model_draws.shape[0] != int(plan["analysis"][
                "normalizer_replicates"]["count"])
            or np.any(~np.isfinite(model_draws)) or np.any(model_draws < 0)):
        raise SystemExit("normalizer coherence-grid probabilities have invalid shape")
    probability = model_draws.mean(axis=0)
    thresholds = cfg["thresholds"]
    rows = {}
    for i, name in enumerate(names):
        metrics = compare(empirical, model_observables(states[name], pairs), probability[i])
        metrics["pass"] = bool(
            metrics["click_count_tv"] <= float(thresholds["click_count_tv_max"])
            and metrics["marginal_rms"] <= float(thresholds["marginal_rms_max"])
            and (metrics["pair_covariance_rms"] is None
                 or metrics["pair_covariance_rms"] <= float(thresholds["pair_covariance_rms_max"])))
        rows[name] = metrics
    artifact = {"schema": "gbskernels.absolute-predictive-checks.v1",
                "run_id": run["run_id"], "registration_id": reg["public"]["plan_sha256"],
                "analysis_commit": commit, "analysis_source_sha256": source_hash,
                "container_digest": container_digest,
                "data_sha256": data_hash, "n_normal_records": empirical["n"],
                "exclusion_sha256": manifest["exclusions"]["sha256"],
                "n_excluded": len(exclusions),
                "scope": "frozen nominal state diagnostics; reconstruction uncertainty is separate",
                "detector_pairs": pairs, "thresholds": dict(thresholds),
                "models": rows,
                "any_model_passes": any(row["pass"] for row in rows.values())}
    write_json_exclusive(args.out, artifact)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
