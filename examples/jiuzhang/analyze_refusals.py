"""Recover refused event scores and test refusal dependence on the score.

The DD evaluator may refuse a matrix near a numerical boundary. This artifact
requires an independent high-precision recovery score for every refused event,
so no selected primary remains missing. A registered within-cell permutation
summary is retained as a reproducible diagnostic; failure to reject exchangeable
refusal is not treated as evidence and does not gate inference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from confirmatory_common import (analysis_source_hash, current_commit, hash_json, load_json,
                                 sha256_file, valid_container_digest,
                                 write_json_exclusive)  # noqa: E402
from confirmatory_contract import load_registration  # noqa: E402


REFUSAL_SCHEMA = "gbskernels.refusal-analysis.v2"
REFUSAL_METHOD = "within_fixed_band_stratum_max_abs_mean_score_difference_permutation"
RECOVERY_SCHEMA = "gbskernels.independent-refusal-recovery.v1"
RECOVERY_METHOD = "independent_high_precision_interval_reevaluation"


def _sha256(value: Any, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return value


def validate_refusal_config(value: Any) -> dict[str, Any]:
    """Validate the complete refusal-analysis portion of the frozen plan."""
    expected_fields = {
        "method", "reps", "seed", "alpha", "inferential_gate",
        "recovery_method", "minimum_precision_bits",
        "recovery_source_sha256", "recovery_container_digest",
    }
    if (not isinstance(value, Mapping)
            or set(value) != expected_fields
            or value.get("method") != REFUSAL_METHOD
            or value.get("inferential_gate") is not False
            or value.get("recovery_method") != RECOVERY_METHOD
            or isinstance(value.get("minimum_precision_bits"), bool)
            or not isinstance(value.get("minimum_precision_bits"), int)
            or value["minimum_precision_bits"] < 128
            or isinstance(value.get("reps"), bool)
            or not isinstance(value.get("reps"), int)
            or value["reps"] < 100
            or isinstance(value.get("seed"), bool)
            or not isinstance(value.get("seed"), int)
            or isinstance(value.get("alpha"), bool)
            or not isinstance(value.get("alpha"), (int, float))
            or not 0 < float(value["alpha"]) < 1):
        raise ValueError("invalid registered refusal-analysis contract")
    source_hash = _sha256(
        value["recovery_source_sha256"], "refusal recovery source")
    if not valid_container_digest(value["recovery_container_digest"]):
        raise ValueError("refusal recovery container must be pinned by SHA256")
    return {"method": REFUSAL_METHOD, "reps": int(value["reps"]),
            "seed": int(value["seed"]), "alpha": float(value["alpha"]),
            "inferential_gate": False, "recovery_method": RECOVERY_METHOD,
            "minimum_precision_bits": int(value["minimum_precision_bits"]),
            "recovery_source_sha256": source_hash,
            "recovery_container_digest": value["recovery_container_digest"]}


def _finite_number(value: Any, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    out = float(value)
    if not np.isfinite(out) or (nonnegative and out < 0):
        raise ValueError(f"{field} is invalid")
    return out


def _normalise_recovered_scores(run: Mapping[str, Any], recovered: Any, *,
                                model_names: list[str]) -> list[dict]:
    accepted_rows = run.get("rows")
    refusal_rows = run.get("refusals")
    if not isinstance(accepted_rows, list) or not isinstance(refusal_rows, list):
        raise ValueError("verified run lacks accepted/refused event rows")
    accepted_ids = [str(row.get("event_id")) for row in accepted_rows]
    refusal_ids = [str(row.get("event_id")) for row in refusal_rows]
    if (len(accepted_ids) != len(set(accepted_ids))
            or len(refusal_ids) != len(set(refusal_ids))
            or set(accepted_ids) & set(refusal_ids)):
        raise ValueError("verified run has duplicate or conflicting event IDs")
    if not isinstance(recovered, list) or any(not isinstance(row, Mapping)
                                               for row in recovered):
        raise ValueError("recovery input must be a JSON list of event objects")
    if (len(model_names) < 2 or len(model_names) != len(set(model_names))
            or any(not isinstance(name, str) or not name for name in model_names)):
        raise ValueError("registered refusal-analysis model order is invalid")
    recovered_ids = [str(row.get("event_id")) for row in recovered]
    if len(recovered_ids) != len(set(recovered_ids)) or set(recovered_ids) != set(refusal_ids):
        raise ValueError("recovery scores must cover exactly all refused event IDs")

    refused = {str(row["event_id"]): row for row in refusal_rows}
    output: list[dict] = []
    for source in recovered:
        event_id = str(source["event_id"])
        refusal = refused[event_id]
        for key in ("band", "record_index", "stratum", "input_sha256"):
            if source.get(key) != refusal.get(key):
                raise ValueError(f"recovered refusal {event_id} mismatches {key}")
        selected_primary = refusal.get("selected_primary")
        if not isinstance(selected_primary, bool):
            raise ValueError(f"refused event {event_id} lacks its registered role")
        if ("selected_primary" in source
                and source.get("selected_primary") is not selected_primary):
            raise ValueError(f"recovered refusal {event_id} mismatches selected_primary")
        score = _finite_number(
            source.get("log_pattern_ratio_mid"),
            f"recovered refusal {event_id} midpoint")
        halfwidth = _finite_number(
            source.get("log_pattern_ratio_halfwidth"),
            f"recovered refusal {event_id} half-width", nonnegative=True)
        row = {
            "event_id": event_id,
            "band": int(refusal["band"]),
            "record_index": int(refusal["record_index"]),
            "stratum": int(refusal["stratum"]),
            "input_sha256": refusal["input_sha256"],
            "selected_primary": selected_primary,
            "log_pattern_ratio_mid": score,
            "log_pattern_ratio_halfwidth": halfwidth,
        }
        if selected_primary and "model_log_probability_proxy" not in source:
            raise ValueError(
                f"recovered primary {event_id} lacks the registered model-score family")
        if "model_log_probability_proxy" in source:
            proxy = source["model_log_probability_proxy"]
            if (not isinstance(proxy, Mapping)
                    or (selected_primary and set(proxy) != set(model_names))):
                raise ValueError(f"recovered refusal {event_id} has invalid model scores")
            checked_proxy = {}
            for name, interval in proxy.items():
                if not isinstance(name, str) or not isinstance(interval, Mapping):
                    raise ValueError(f"recovered refusal {event_id} has invalid model scores")
                checked_proxy[name] = {
                    "mid": _finite_number(
                        interval.get("mid"), f"recovered refusal {event_id} model midpoint"),
                    "halfwidth": _finite_number(
                        interval.get("halfwidth"),
                        f"recovered refusal {event_id} model half-width",
                        nonnegative=True),
                }
            row["model_log_probability_proxy"] = checked_proxy
            if selected_primary:
                reference = checked_proxy[model_names[0]]
                alternative = checked_proxy[model_names[1]]
                expected_midpoint = alternative["mid"] - reference["mid"]
                expected_halfwidth = (
                    alternative["halfwidth"] + reference["halfwidth"])
                scale = max(1.0, abs(score), abs(expected_midpoint),
                            halfwidth, expected_halfwidth)
                tolerance = 32.0 * np.finfo(float).eps * scale
                if (abs(score - expected_midpoint) > tolerance
                        or abs(halfwidth - expected_halfwidth) > tolerance):
                    raise ValueError(
                        f"recovered primary {event_id} scalar and model-score "
                        "enclosures are inconsistent")
        output.append(row)
    return sorted(output, key=lambda row: row["event_id"])


def _validate_recovery_input(
    value: Any, *, run: Mapping[str, Any], registration_id: str,
    config: Mapping[str, Any], model_names: list[str], verified_run_sha256: str,
) -> tuple[dict[str, Any], list[dict]]:
    cfg = validate_refusal_config(config)
    expected_fields = {
        "schema", "run_id", "registration_id", "verified_run_sha256",
        "method", "precision_bits",
        "independent_implementation", "recovery_source_sha256",
        "recovery_container_digest", "scores", "recovery_payload_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("recovery input is not a complete independent recovery artifact")
    body = {key: item for key, item in value.items()
            if key != "recovery_payload_sha256"}
    if (value.get("schema") != RECOVERY_SCHEMA
            or value.get("recovery_payload_sha256") != hash_json(body)
            or value.get("run_id") != run.get("run_id")
            or value.get("registration_id") != registration_id
            or value.get("verified_run_sha256") != verified_run_sha256
            or value.get("method") != cfg["recovery_method"]
            or value.get("independent_implementation") is not True
            or value.get("recovery_source_sha256") != cfg["recovery_source_sha256"]
            or value.get("recovery_container_digest")
            != cfg["recovery_container_digest"]):
        raise ValueError("independent recovery identity or provenance is invalid")
    precision = value.get("precision_bits")
    if (isinstance(precision, bool) or not isinstance(precision, int)
            or precision < cfg["minimum_precision_bits"]):
        raise ValueError("independent recovery precision is below the registered minimum")
    if value["recovery_source_sha256"] == run.get("contract", {}).get(
            "analysis_source_sha256"):
        raise ValueError("refusal recovery must use a source-distinct implementation")
    scores = _normalise_recovered_scores(
        run, value.get("scores"), model_names=model_names)
    return dict(value), scores


def _permutation_statistics(run: Mapping[str, Any], recovered: list[dict], *,
                            reps: int, seed: int) -> tuple[float | None, float | None]:
    all_rows = []
    for row in run["rows"]:
        # A reserve replacement is operationally useful but is removed once
        # its refused registered primary has been independently recovered. It
        # therefore cannot also serve as a non-refused inferential unit here.
        if row.get("replacement_for_refusal_event_id") is not None:
            continue
        score = _finite_number(
            row.get("log_pattern_ratio_mid"), "accepted refusal-analysis score")
        all_rows.append({"band": int(row["band"]), "stratum": int(row["stratum"]),
                         "refused": False, "score": score})
    for row in recovered:
        if not row["selected_primary"]:
            continue
        all_rows.append({"band": int(row["band"]), "stratum": int(row["stratum"]),
                         "refused": True, "score": float(row["log_pattern_ratio_mid"])})
    mixed_cells: list[tuple[np.ndarray, int]] = []
    observed = []
    keys = sorted({(row["band"], row["stratum"]) for row in all_rows})
    for band, stratum in keys:
        cell = [row for row in all_rows
                if row["band"] == band and row["stratum"] == stratum]
        yes = np.asarray([row["score"] for row in cell if row["refused"]])
        no = np.asarray([row["score"] for row in cell if not row["refused"]])
        if len(yes) and len(no):
            observed.append(float(yes.mean() - no.mean()))
            mixed_cells.append((np.asarray([row["score"] for row in cell]), len(yes)))
    if not mixed_cells:
        return None, None
    observed_max = float(np.max(np.abs(observed)))
    rng = np.random.default_rng(seed)
    null = np.empty(reps, dtype=float)
    for index in range(reps):
        effects = []
        for values, n_yes in mixed_cells:
            labels = np.zeros(len(values), dtype=bool)
            labels[rng.choice(len(values), size=n_yes, replace=False)] = True
            effects.append(float(values[labels].mean() - values[~labels].mean()))
        null[index] = max(np.abs(effects))
    pvalue = float((1 + np.sum(null >= observed_max)) / (reps + 1))
    return observed_max, pvalue


def analyze(run: dict, recovered: Mapping[str, Any], *, registration_id: str,
            config: Mapping[str, Any], provenance: Mapping[str, Any],
            verified_run_sha256: str, recovered_input_sha256: str,
            model_names: list[str]) -> dict:
    if run.get("schema") != "gbskernels.verified-run.v2" or not run.get("complete"):
        raise ValueError("refusal analysis requires a complete verified run")
    cfg = validate_refusal_config(config)
    recovery, recovered_scores = _validate_recovery_input(
        recovered, run=run, registration_id=registration_id, config=cfg,
        model_names=model_names, verified_run_sha256=verified_run_sha256)
    observed, pvalue = _permutation_statistics(
        run, recovered_scores, reps=cfg["reps"], seed=cfg["seed"])
    artifact = {
        "schema": REFUSAL_SCHEMA,
        "run_id": run["run_id"],
        "registration_id": registration_id,
        "analysis_commit": provenance.get("analysis_commit"),
        "analysis_source_sha256": provenance.get("analysis_source_sha256"),
        "container_digest": provenance.get("container_digest"),
        "method": cfg["method"],
        "permutation_reps": cfg["reps"],
        "seed": cfg["seed"],
        "alpha": cfg["alpha"],
        "n_refused": len(run["refusals"]),
        "observed_max_stratum_mean_difference": observed,
        "permutation_pvalue": pvalue,
        "permutation_diagnostic_available": pvalue is not None,
        "permutation_diagnostic_only": True,
        "pass": True,
        "inputs": {
            "verified_run_sha256": verified_run_sha256,
            "recovered_input_sha256": recovered_input_sha256,
            "recovery_source_sha256": cfg["recovery_source_sha256"],
        },
        "recovery": recovery,
        "recovered_scores": recovered_scores,
        "recovered_scores_sha256": hash_json(recovered_scores),
    }
    return artifact


def validate_refusal_analysis(
    artifact: Any, *, run: Mapping[str, Any], registration_id: str,
    config: Mapping[str, Any], verified_run_sha256: str,
    model_names: list[str],
) -> dict[str, Any]:
    """Recompute and validate one refusal artifact against its exact run."""
    expected_fields = {
        "schema", "run_id", "registration_id", "analysis_commit",
        "analysis_source_sha256", "container_digest", "method",
        "permutation_reps", "seed", "alpha", "n_refused",
        "observed_max_stratum_mean_difference", "permutation_pvalue",
        "permutation_diagnostic_available", "permutation_diagnostic_only", "pass",
        "inputs", "recovery", "recovered_scores", "recovered_scores_sha256",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != expected_fields \
            or artifact.get("schema") != REFUSAL_SCHEMA:
        raise ValueError("unsupported or malformed refusal-analysis artifact")
    cfg = validate_refusal_config(config)
    contract = run.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("verified run lacks its content-addressed contract")
    contract_body = {key: value for key, value in contract.items() if key != "run_id"}
    if (run.get("schema") != "gbskernels.verified-run.v2"
            or not run.get("complete")
            or run.get("run_id") != contract.get("run_id")
            or contract.get("run_id") != hash_json(contract_body)):
        raise ValueError("verified run contract is invalid")
    expected_provenance = {
        "analysis_commit": contract.get("analysis_commit"),
        "analysis_source_sha256": contract.get("analysis_source_sha256"),
        "container_digest": contract.get("container_digest"),
    }
    if (artifact.get("run_id") != run.get("run_id")
            or artifact.get("registration_id") != registration_id
            or contract.get("registration_id") != registration_id
            or any(artifact.get(key) != value
                   for key, value in expected_provenance.items())):
        raise ValueError("refusal-analysis run, registration, or provenance differs")
    if (artifact.get("method") != cfg["method"]
            or artifact.get("permutation_reps") != cfg["reps"]
            or artifact.get("seed") != cfg["seed"]
            or artifact.get("alpha") != cfg["alpha"]
            or artifact.get("permutation_diagnostic_only") is not True):
        raise ValueError("refusal analysis did not execute the registered contract")
    inputs = artifact.get("inputs")
    if (not isinstance(inputs, Mapping)
            or set(inputs) != {"verified_run_sha256", "recovered_input_sha256",
                               "recovery_source_sha256"}
            or inputs.get("verified_run_sha256") != verified_run_sha256
            or inputs.get("recovery_source_sha256") != cfg["recovery_source_sha256"]
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value)
                   for value in inputs.values())):
        raise ValueError("refusal-analysis input hashes are invalid or misbound")
    recovery, recovered = _validate_recovery_input(
        artifact.get("recovery"), run=run, registration_id=registration_id,
        config=cfg, model_names=model_names,
        verified_run_sha256=verified_run_sha256)
    if (artifact.get("recovered_scores") != recovered
            or artifact.get("recovery") != recovery
            or artifact.get("recovered_scores_sha256") != hash_json(recovered)):
        raise ValueError("refusal-analysis recovered-score payload is invalid")
    observed, pvalue = _permutation_statistics(
        run, recovered, reps=cfg["reps"], seed=cfg["seed"])
    if (artifact.get("n_refused") != len(run["refusals"])
            or artifact.get("observed_max_stratum_mean_difference") != observed
            or artifact.get("permutation_pvalue") != pvalue
            or artifact.get("permutation_diagnostic_available") is not (pvalue is not None)
            or artifact.get("pass") is not True):
        raise ValueError("refusal-analysis statistics do not reproduce")
    return dict(artifact)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registration", type=Path, required=True)
    ap.add_argument("--verified-run", type=Path, required=True)
    ap.add_argument("--recovered", type=Path, required=True,
                    help="self-hashed independent high-precision recovery artifact")
    ap.add_argument("--recovery-source", type=Path, required=True,
                    help="frozen independent recovery implementation source")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    registration = load_registration(args.registration)
    plan = registration["plan"]
    run = load_json(args.verified_run)
    recovered = json.loads(args.recovered.read_text())
    contract = run.get("contract", {})
    contract_body = {key: value for key, value in contract.items() if key != "run_id"}
    registration_id = registration["public"]["plan_sha256"]
    if (run.get("schema") != "gbskernels.verified-run.v2" or not run.get("complete")
            or run.get("run_id") != contract.get("run_id")
            or contract.get("run_id") != hash_json(contract_body)
            or contract.get("registration_id") != registration_id
            or run.get("registration", {}).get("public") != registration["public"]):
        raise SystemExit("verified run does not match the registered run contract")
    commit = current_commit()
    source_hash = analysis_source_hash()
    container_digest = os.environ.get("GBS_CONTAINER_DIGEST")
    if (commit != plan["analysis_commit"] or commit != contract.get("analysis_commit")):
        raise SystemExit("refusal-analysis commit differs from the registered run")
    if (source_hash != plan["numerical_contract"]["analysis_source_sha256"]
            or source_hash != contract.get("analysis_source_sha256")):
        raise SystemExit("refusal-analysis source bytes differ from the registered run")
    if (not valid_container_digest(container_digest)
            or container_digest != contract.get("container_digest")):
        raise SystemExit("refusal-analysis container differs from the registered run")
    refusal_cfg = validate_refusal_config(plan["analysis"]["refusal_analysis"])
    if sha256_file(args.recovery_source) != refusal_cfg["recovery_source_sha256"]:
        raise SystemExit("independent recovery source differs from the registration")
    models = plan["models"]
    reference = str(models["reference_model"])
    alternative = str(models["alternative_model"])
    model_names = [reference, alternative] + sorted(
        set(map(str, models["coherence_points"])) - {reference, alternative})
    result = analyze(
        run, recovered, registration_id=registration_id,
        config=refusal_cfg, provenance=contract,
        verified_run_sha256=sha256_file(args.verified_run),
        recovered_input_sha256=sha256_file(args.recovered),
        model_names=model_names)
    validate_refusal_analysis(
        result, run=run, registration_id=registration_id,
        config=plan["analysis"]["refusal_analysis"],
        verified_run_sha256=sha256_file(args.verified_run),
        model_names=model_names)
    write_json_exclusive(args.out, result)
    if (result["permutation_pvalue"] is not None
            and result["permutation_pvalue"] < result["alpha"]):
        print("diagnostic refusal permutation p-value is below registered alpha; "
              "complete primary recovery remains valid", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
