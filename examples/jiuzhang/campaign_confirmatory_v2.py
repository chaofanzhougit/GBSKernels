"""Immutable evaluator and fail-closed reducer for confirmatory manifests v2.

This module deliberately separates expensive event evaluation from inference.
Every event is an immutable JSON object in a content-addressed run directory.
The reducer validates every registered hash and exact event identity, resolves
refusals through the pre-ranked reserve list, and emits no scientific interval.
Inference belongs to :mod:`confirmatory_inference`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from confirmatory_common import (analysis_source_hash, current_commit, event_id,
                                 hash_array, hash_json, load_json, pattern_hash,
                                 sha256_file,
                                 valid_container_digest,
                                 write_json_exclusive)  # noqa: E402
from confirmatory_contract import (canonical_bytes, event_key, sha256_bytes,
                                   sha256_json, validate_registration)  # noqa: E402


def _valid_container_digest(value: Any) -> bool:
    return valid_container_digest(value)


def load_manifest(path: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(path)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("kind") != "jiuzhang1_confirmatory_selection_v2":
        raise ValueError("unsupported manifest schema")
    declared = manifest.get("manifest_payload_sha256")
    body = {k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}
    expected = sha256_json(body)
    if declared != expected:
        raise ValueError("manifest content hash mismatch")
    if raw != canonical_bytes(manifest):
        raise ValueError("manifest file is not canonical JSON")
    return manifest, expected


def _band_candidates(manifest: dict[str, Any], band: int) -> list[dict[str, Any]]:
    rows = [dict(r) for r in manifest["primary"] + manifest["reserves"]
            if int(r["band"]) == band]
    rows.sort(key=lambda r: (int(r["stratum"]), int(r["rank_in_stratum"])))
    return rows


def _manifest_event(manifest: dict[str, Any], band: int, position: int) -> dict[str, Any]:
    rows = _band_candidates(manifest, band)
    if not 0 <= position < len(rows):
        raise IndexError(position)
    row = rows[position]
    pattern = np.asarray(row["pattern"], dtype=bool)
    if pattern.shape != (100,) or int(pattern.sum()) != int(band):
        raise ValueError(f"invalid registered pattern at C={band} position={position}")
    packed = np.packbits(pattern.astype(np.uint8)).tobytes().hex()
    if packed != row["pattern_packed_hex"]:
        raise ValueError(f"packed pattern mismatch at C={band} position={position}")
    if row.get("pattern_sha256") and row["pattern_sha256"] != pattern_hash(pattern):
        raise ValueError(f"pattern hash mismatch at C={band} position={position}")
    mid = manifest["manifest_payload_sha256"]
    ridx = int(row["record_index"])
    design = manifest["design"][str(band)]
    input_hash = str(row["input_hash"])
    seed = str(manifest["seed"])
    if row["key"] != event_key(ridx, seed):
        raise ValueError(f"selection key mismatch at C={band} position={position}")
    raw = bytes.fromhex(str(row["raw_record_hex"]))
    if len(raw) != int(manifest["source"]["record_bytes"]):
        raise ValueError(f"raw record length mismatch at C={band} position={position}")
    if sha256_bytes(raw) != row["source_raw_hash"]:
        raise ValueError(f"raw record hash mismatch at C={band} position={position}")
    from select_confirmatory_v2 import INPUT_HASH_DOMAIN, decode_records

    timestamps, decoded, abnormal = decode_records(np.frombuffer(raw, dtype=np.uint8)[None, :])
    if (not np.array_equal(decoded[0], pattern)
            or int(timestamps[0]) != int(row["timestamp_uint16"])
            or str(row["timestamp_bits"]) != format(int(timestamps[0]), "016b")
            or bool(abnormal[0]) != bool(row["abnormal"])):
        raise ValueError(f"raw record audit fields mismatch at C={band} position={position}")
    expected_input = sha256_bytes(INPUT_HASH_DOMAIN + int(band).to_bytes(2, "big") + raw)
    if input_hash != expected_input:
        raise ValueError(f"input hash mismatch at C={band} position={position}")
    return {
        "manifest_id": mid, "event_id": event_id(mid, band, ridx, input_hash),
        "band": int(band), "position": int(position), "record_index": ridx,
        "timestamp_bits": str(row["timestamp_bits"]),
        "abnormal": bool(row["abnormal"]),
        "stratum": int(row["stratum"]),
        "rank_within_stratum": int(row["rank_in_stratum"]),
        "selected_primary": row["role"] == "primary",
        "inclusion_probability": float(row["primary_inclusion_probability"]),
        "primary_quota": int(row["primary_quota"]),
        "reserve_quota": int(row["reserve_quota"]),
        "eligible_in_stratum": int(row["eligible_in_stratum"]),
        "eligible_in_band": int(design["eligible_total"]),
        "selection_key": str(row["key"]),
        "input_sha256": input_hash, "source_raw_sha256": str(row["source_raw_hash"]),
        "pattern_sha256": pattern_hash(pattern), "pattern": pattern,
    }


def state_fingerprint(states: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for name, state in sorted(states.items()):
        O = np.asarray(state["O"])
        cov = np.asarray(state["cov"])
        Q = np.asarray(state["Q"])
        log_norm = float(state["log_sqrt_detQ"])
        if (O.ndim != 2 or O.shape[0] != O.shape[1] or cov.ndim != 2
                or cov.shape != O.shape or Q.shape != O.shape
                or not np.all(np.isfinite(O)) or not np.all(np.isfinite(cov))
                or not np.all(np.isfinite(Q)) or not math.isfinite(log_norm)):
            raise ValueError(f"state {name} contains invalid numerical arrays")
        out[name] = {"O_sha256": hash_array(O), "cov_sha256": hash_array(cov),
                     "Q_sha256": hash_array(Q),
                     "log_sqrt_detQ": log_norm}
    return out


def kernel_binary_fingerprint() -> dict[str, Any]:
    """Hash the exact compiled extension loaded for the expensive evaluator."""
    import gbskernels

    loader = getattr(gbskernels, "_load_gpu_ext", None)
    extension = loader() if callable(loader) else None
    path_value = getattr(extension, "__file__", None)
    if extension is None or not isinstance(path_value, str):
        raise ValueError("the compiled gbskernels GPU extension is not loaded")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise ValueError("the loaded gbskernels extension has no hashable binary")
    return {"filename": path.name, "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "gbskernels_version": str(getattr(gbskernels, "__version__", "unknown"))}


def validate_manifest_registration(manifest: dict[str, Any],
                                   normalized: dict[str, Any]) -> None:
    if manifest.get("registration") != normalized or not manifest.get("registered"):
        raise ValueError("manifest registration record does not match")
    if str(manifest.get("seed")) != str(normalized["seed_derivation"]["seed_hex"]):
        raise ValueError("manifest seed does not match future-beacon derivation")
    selection = normalized["plan"]["selection"]
    source = manifest.get("source", {})
    for field in ("n_records", "record_bytes"):
        if int(source.get(field, -1)) != int(selection[field]):
            raise ValueError(f"manifest source {field} does not match registration")
    if source.get("source_raw_sha256") != selection["source_raw_sha256"]:
        raise ValueError("manifest raw-source hash does not match registration")
    exclusions = manifest.get("exclusions", {}).get("record_indices")
    if (not isinstance(exclusions, list)
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 0
                   or x >= int(selection["n_records"]) for x in exclusions)
            or exclusions != sorted(set(exclusions))
            or int(manifest.get("exclusions", {}).get("count", -1)) != len(exclusions)
            or sha256_json(exclusions) != manifest.get("exclusions", {}).get("sha256")
            or manifest.get("exclusions", {}).get("sha256") != selection["exclusion_sha256"]):
        raise ValueError("manifest exclusion hash does not match registration")
    population_audit = manifest.get("population_audit", {})
    audit_body = {k: v for k, v in population_audit.items()
                  if k != "audit_payload_sha256"}
    if population_audit.get("audit_payload_sha256") != sha256_json(audit_body):
        raise ValueError("manifest population audit content hash is invalid")
    if population_audit.get("audit_payload_sha256") != selection["population_audit_sha256"]:
        raise ValueError("manifest population audit does not match registration")
    if (population_audit.get("source_raw_sha256") != selection["source_raw_sha256"]
            or int(population_audit.get("n_records", -1)) != int(selection["n_records"])):
        raise ValueError("manifest population audit source does not match registration")
    if int(population_audit.get("n_excluded", -1)) != len(exclusions):
        raise ValueError("manifest population audit exclusion count is inconsistent")
    if (population_audit.get("exclusion_sha256") != selection["exclusion_sha256"]
            or int(population_audit.get("n_strata", -1)) != int(selection["n_strata"])
            or [int(x) for x in population_audit.get("bands", [])]
            != [int(x) for x in selection["bands"]]):
        raise ValueError("manifest population audit design does not match registration")
    audited_weights = {str(k): float(v) for k, v in population_audit.get(
        "band_weights_within_window", {}).items()}
    registered_weights = {str(k): float(v) for k, v in normalized["plan"][
        "analysis"]["band_weights"].items()}
    if audited_weights.keys() != registered_weights.keys() or any(
            not math.isclose(audited_weights[key], registered_weights[key], abs_tol=1e-12)
            for key in audited_weights):
        raise ValueError("registered band weights differ from population audit")
    n_strata = int(selection["n_strata"])
    if int(manifest.get("strata", {}).get("count", 0)) != n_strata:
        raise ValueError("manifest stratum count does not match registration")
    primary_rows = manifest.get("primary")
    reserve_rows = manifest.get("reserves")
    if not isinstance(primary_rows, list) or not isinstance(reserve_rows, list):
        raise ValueError("manifest primary/reserve rows are required")
    if any(row.get("role") != "primary" for row in primary_rows):
        raise ValueError("manifest primary list contains a non-primary row")
    if any(row.get("role") != "reserve" for row in reserve_rows):
        raise ValueError("manifest reserve list contains a non-reserve row")
    all_rows = primary_rows + reserve_rows
    identities = [(int(row["band"]), int(row["record_index"])) for row in all_rows]
    if len(set(identities)) != len(identities):
        raise ValueError("manifest contains duplicate band/record identities")
    for band in [int(x) for x in selection["bands"]]:
        row = manifest.get("design", {}).get(str(band), {})
        primary = [int(x) for x in row.get("primary_by_stratum", [])]
        reserve = [int(x) for x in row.get("reserve_by_stratum", [])]
        eligible = [int(x) for x in row.get("eligible_by_stratum", [])]
        if not (len(primary) == len(reserve) == len(eligible) == n_strata):
            raise ValueError(f"manifest C={band} stratum vectors have wrong length")
        if sum(primary) != int(selection["targets"][str(band)]):
            raise ValueError(f"manifest C={band} primary target mismatch")
        if sum(reserve) != int(selection["reserves"][str(band)]):
            raise ValueError(f"manifest C={band} reserve target mismatch")
        if any(p + r > n for p, r, n in zip(primary, reserve, eligible)):
            raise ValueError(f"manifest C={band} quota exceeds eligible population")
        if [int(x) for x in population_audit.get(
                "eligible_by_band_stratum", {}).get(str(band), [])] != eligible:
            raise ValueError(f"manifest C={band} population counts differ from audit")
        for stratum in range(n_strata):
            cell = sorted(
                (item for item in all_rows if int(item["band"]) == band
                 and int(item["stratum"]) == stratum),
                key=lambda item: int(item["rank_in_stratum"]),
            )
            capacity = primary[stratum] + reserve[stratum]
            if len(cell) != capacity:
                raise ValueError(f"manifest C={band}, h={stratum} capacity mismatch")
            if [int(item["rank_in_stratum"]) for item in cell] != list(range(capacity)):
                raise ValueError(f"manifest C={band}, h={stratum} ranks are not contiguous")
            expected_roles = (["primary"] * primary[stratum]
                              + ["reserve"] * reserve[stratum])
            if [item["role"] for item in cell] != expected_roles:
                raise ValueError(f"manifest C={band}, h={stratum} role order is invalid")
            for item in cell:
                if (int(item["eligible_in_stratum"]) != eligible[stratum]
                        or int(item["primary_quota"]) != primary[stratum]
                        or int(item["reserve_quota"]) != reserve[stratum]):
                    raise ValueError(f"manifest C={band}, h={stratum} audit quota mismatch")
                primary_pi = primary[stratum] / eligible[stratum]
                manifest_pi = capacity / eligible[stratum]
                role_pi = ((primary[stratum] if item["role"] == "primary"
                            else reserve[stratum]) / eligible[stratum])
                if (not math.isclose(float(item["primary_inclusion_probability"]),
                                     primary_pi, abs_tol=1e-15)
                        or not math.isclose(float(item["manifest_inclusion_probability"]),
                                            manifest_pi, abs_tol=1e-15)
                        or not math.isclose(float(item["inclusion_probability"]),
                                            role_pi, abs_tol=1e-15)):
                    raise ValueError(f"manifest C={band}, h={stratum} probability mismatch")


def make_run_contract(registration: dict[str, Any], manifest: dict[str, Any],
                      states: dict[str, dict[str, Any]]) -> dict[str, Any]:
    normalized = validate_registration(registration, require_beacon=True)
    registration_id = normalized["public"]["plan_sha256"]
    validate_manifest_registration(manifest, normalized)
    manifest_registration = manifest.get("registration") or {}
    manifest_id = manifest["manifest_payload_sha256"]
    if (manifest_registration.get("public", {}).get("plan_sha256")
            != registration_id):
        raise ValueError("manifest is bound to a different registration")
    if manifest_registration != normalized:
        raise ValueError("manifest beacon/seed registration does not match supplied registration")
    plan = normalized["plan"]
    model_cfg = plan["models"]
    expected_order = [str(model_cfg["reference_model"]),
                      str(model_cfg["alternative_model"])] + sorted(
                          set(model_cfg["coherence_points"])
                          - {model_cfg["reference_model"],
                             model_cfg["alternative_model"]})
    if list(states) != expected_order:
        raise ValueError("state dictionary order does not match registered model order")
    commit = current_commit()
    if commit != plan["analysis_commit"]:
        raise ValueError(f"analysis commit {commit!r} does not match registration")
    sf = state_fingerprint(states)
    expected_states = plan["numerical_contract"]["state_fingerprints"]
    if sf != expected_states:
        raise ValueError("constructed states do not match registered fingerprints")
    source_hash = analysis_source_hash()
    if source_hash != plan["numerical_contract"]["analysis_source_sha256"]:
        raise ValueError("analysis source bytes do not match the registered hash")
    container_digest = os.environ.get("GBS_CONTAINER_DIGEST")
    if (plan["external_requirements"].get("container_digest_required", True)
            and not _valid_container_digest(container_digest)):
        raise ValueError("a pinned GBS_CONTAINER_DIGEST is required by registration")
    kernel_binary = kernel_binary_fingerprint()
    contract = {
        "schema": "gbskernels.run-contract.v2",
        "registration_id": registration_id,
        "manifest_id": manifest_id,
        "analysis_commit": commit,
        "container_digest": container_digest,
        "numerical_scope": plan["numerical_contract"]["scope"],
        "analysis_source_sha256": source_hash,
        "kernel_binary": kernel_binary,
        "states": sf,
    }
    contract["run_id"] = hash_json(contract)
    return contract


def _default_evaluator(states: dict[str, dict[str, Any]], event: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the registered binary64 matrices and return a kernel enclosure.

    This does not call the legacy probability evaluator.  It returns a
    probability *proxy* for the frozen binary64 state matrices: the torontonian
    interval is propagated through an 80-digit logarithm and the registered
    binary64 normalizer.  It is not an end-to-end interval certificate for
    state construction, inversion, determinant, or detector reconstruction.
    """
    import gbskernels

    names = list(states)
    if len(names) < 2:
        raise ValueError("event evaluator requires at least two registered models")
    import mpmath as mp

    mp.mp.dps = 80
    logs: dict[str, tuple[float, float]] = {}
    log_tors: dict[str, tuple[float, float]] = {}
    elapsed = 0.0
    clicked = np.flatnonzero(event["pattern"])
    for name in names:
        st = states[name]
        modes = st["O"].shape[0] // 2
        idx = list(clicked) + [int(j) + modes for j in clicked]
        sub = np.ascontiguousarray(st["O"][np.ix_(idx, idx)])
        t0 = time.time()
        try:
            value, diag = gbskernels.tor_single(
                sub, groups=min(len(clicked), 14), dd=True)
        except (ValueError, FloatingPointError) as exc:
            elapsed += time.time() - t0
            return {"refused": True,
                    "refusal_reason": f"kernel refused {name}: {type(exc).__name__}",
                    "seconds": elapsed}
        elapsed += time.time() - t0
        bound = float(diag["abs_error_bound"])
        if not (math.isfinite(value) and math.isfinite(bound) and bound >= 0
                and value - bound > 0):
            return {"refused": True, "refusal_reason": "kernel interval contains zero",
                    "seconds": elapsed}
        try:
            lo_mp = mp.log(mp.mpf(value) - mp.mpf(bound))
            hi_mp = mp.log(mp.mpf(value) + mp.mpf(bound))
        except (ValueError, OverflowError):
            return {"refused": True, "refusal_reason": "logarithm propagation failed",
                    "seconds": elapsed}
        log_lo = float(np.nextafter(float(lo_mp), -np.inf))
        log_hi = float(np.nextafter(float(hi_mp), np.inf))
        log_tors[name] = (log_lo, log_hi)
        frozen_log_norm = float(st["log_sqrt_detQ"])
        logs[name] = (float(np.nextafter(log_lo - frozen_log_norm, -np.inf)),
                      float(np.nextafter(log_hi - frozen_log_norm, np.inf)))
    reference, alternative = names[:2]
    lo = float(np.nextafter(logs[alternative][0] - logs[reference][1], -np.inf))
    hi = float(np.nextafter(logs[alternative][1] - logs[reference][0], np.inf))

    def _interval(bounds: tuple[float, float]) -> dict[str, float]:
        return {"mid": (bounds[0] + bounds[1]) / 2,
                "halfwidth": float(np.nextafter((bounds[1] - bounds[0]) / 2, np.inf)),
                "lo": float(bounds[0]), "hi": float(bounds[1])}

    return {"refused": False, "log_pattern_ratio_lo": lo,
            "log_pattern_ratio_hi": hi,
            "log_pattern_ratio_mid": (lo + hi) / 2,
            "log_pattern_ratio_halfwidth": float(np.nextafter((hi - lo) / 2, np.inf)),
            "reference_model": reference, "alternative_model": alternative,
            "model_log_probability_proxy": {
                name: _interval(bounds)
                for name, bounds in logs.items()
            },
            "model_log_torontonian": {
                name: _interval(bounds)
                for name, bounds in log_tors.items()
            },
            "seconds": elapsed}


def evaluate_positions(registration: dict[str, Any], manifest: dict[str, Any],
                       states: dict[str, dict[str, Any]],
                       run_root: str | Path, bands: list[int], positions: dict[int, range],
                       evaluator: Callable[[dict[str, dict[str, Any]], dict[str, Any]],
                                           dict[str, Any]] = _default_evaluator) -> str:
    contract = make_run_contract(registration, manifest, states)
    root = Path(run_root) / contract["run_id"]
    (root / "events").mkdir(parents=True, exist_ok=True)
    contract_path = root / "contract.json"
    if contract_path.exists():
        if load_json(contract_path) != contract:
            raise ValueError("run contract collision")
    else:
        write_json_exclusive(contract_path, contract)
    for band in bands:
        for position in positions[band]:
            event = _manifest_event(manifest, band, position)
            out = root / "events" / f"{event['event_id']}.json"
            if out.exists():
                prior = load_json(out)
                for key in ("schema", "run_id", "event_id", "manifest_id", "band", "position",
                            "record_index", "stratum", "rank_within_stratum",
                            "input_sha256", "selection_key", "pattern_sha256",
                            "source_raw_sha256", "timestamp_bits", "abnormal",
                            "eligible_in_stratum", "eligible_in_band",
                            "primary_quota", "reserve_quota", "selected_primary",
                            "inclusion_probability"):
                    expected = ("gbskernels.event-evaluation.v2" if key == "schema"
                                else contract["run_id"] if key == "run_id"
                                else event.get(key))
                    if prior.get(key) != expected:
                        raise ValueError(f"immutable event conflict: {out} ({key})")
                # The run contract already binds code, container, matrices, and
                # registration. Do not rerun an immutable expensive event merely
                # to compare wall-clock timing fields.
                continue
            base = {k: v for k, v in event.items() if k != "pattern"}
            result = {"schema": "gbskernels.event-evaluation.v2", "run_id": contract["run_id"],
                      **base, **evaluator(states, event)}
            write_json_exclusive(out, result)
    return contract["run_id"]


def _expected_by_cell(manifest: dict[str, Any], band: int) \
        -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for pos in range(len(_band_candidates(manifest, band))):
        ev = _manifest_event(manifest, band, pos)
        out.setdefault(ev["stratum"], []).append(ev)
    for values in out.values():
        values.sort(key=lambda e: e["rank_within_stratum"])
    return out


def reduce_run(registration: dict[str, Any], manifest: dict[str, Any],
               run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    contract = load_json(run_dir / "contract.json")
    if contract.get("schema") != "gbskernels.run-contract.v2":
        raise ValueError("invalid run contract")
    contract_body = {key: value for key, value in contract.items() if key != "run_id"}
    if contract.get("run_id") != hash_json(contract_body):
        raise ValueError("run contract content hash is invalid")
    if analysis_source_hash() != contract.get("analysis_source_sha256"):
        raise ValueError("current analysis source differs from the run contract")
    if contract["manifest_id"] != manifest["manifest_payload_sha256"]:
        raise ValueError("run and manifest IDs differ")
    normalized = validate_registration(registration, require_beacon=True)
    validate_manifest_registration(manifest, normalized)
    registration_id = normalized["public"]["plan_sha256"]
    if contract["registration_id"] != registration_id:
        raise ValueError("run and registration IDs differ")
    plan = normalized["plan"]
    if (contract.get("analysis_commit") != plan["analysis_commit"]
            or contract.get("numerical_scope") != plan["numerical_contract"]["scope"]
            or contract.get("analysis_source_sha256")
            != plan["numerical_contract"]["analysis_source_sha256"]
            or contract.get("states") != plan["numerical_contract"]["state_fingerprints"]):
        raise ValueError("run contract does not match the registered numerical/code contract")
    if (plan["external_requirements"].get("container_digest_required", True)
            and not _valid_container_digest(contract.get("container_digest"))):
        raise ValueError("run contract lacks a pinned container digest")
    kernel_binary = contract.get("kernel_binary")
    if (not isinstance(kernel_binary, dict)
            or not isinstance(kernel_binary.get("filename"), str)
            or not isinstance(kernel_binary.get("sha256"), str)
            or len(kernel_binary["sha256"]) != 64
            or not isinstance(kernel_binary.get("bytes"), int)
            or kernel_binary["bytes"] <= 0):
        raise ValueError("run contract lacks a compiled-kernel binary fingerprint")
    records: dict[str, dict[str, Any]] = {}
    for path in sorted((run_dir / "events").glob("*.json")):
        row = load_json(path)
        eid = row.get("event_id")
        if path.stem != eid or row.get("run_id") != contract["run_id"]:
            raise ValueError(f"misbound event artifact {path}")
        if eid in records and records[eid] != row:
            raise ValueError(f"conflicting duplicate event {eid}")
        records[eid] = row

    design = plan["selection"]
    expected_events: dict[str, dict[str, Any]] = {}
    cells_by_band: dict[int, dict[int, list[dict[str, Any]]]] = {}
    for band in [int(x) for x in design["bands"]]:
        cells_by_band[band] = _expected_by_cell(manifest, band)
        for cell in cells_by_band[band].values():
            for ev in cell:
                expected_events[ev["event_id"]] = ev
    unexpected = set(records) - set(expected_events)
    if unexpected:
        raise ValueError(f"run contains unexpected event IDs: {sorted(unexpected)[:3]}")
    for eid, row in records.items():
        ev = expected_events[eid]
        invariant = {
            "schema": "gbskernels.event-evaluation.v2",
            "event_id": ev["event_id"], "manifest_id": ev["manifest_id"],
            "band": ev["band"], "position": ev["position"],
            "record_index": ev["record_index"], "stratum": ev["stratum"],
            "rank_within_stratum": ev["rank_within_stratum"],
            "input_sha256": ev["input_sha256"], "selection_key": ev["selection_key"],
            "pattern_sha256": ev["pattern_sha256"],
            "source_raw_sha256": ev["source_raw_sha256"],
            "timestamp_bits": ev["timestamp_bits"],
            "abnormal": ev["abnormal"],
            "eligible_in_stratum": ev["eligible_in_stratum"],
            "eligible_in_band": ev["eligible_in_band"],
            "primary_quota": ev["primary_quota"],
            "reserve_quota": ev["reserve_quota"],
        }
        for key, value in invariant.items():
            if row.get(key) != value:
                raise ValueError(f"event {eid} has mismatched {key}")

    usable: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    missing: list[str] = []
    for band in [int(x) for x in design["bands"]]:
        cells = cells_by_band[band]
        targets = {s: int(n) for s, n in enumerate(
            manifest["design"][str(band)]["primary_by_stratum"])}
        for stratum, ordered in sorted(cells.items()):
            accepted = 0
            pending_refused_primary: list[str] = []
            if targets[stratum] == 0:
                continue
            for ev in ordered:
                row = records.get(ev["event_id"])
                if row is None:
                    missing.append(ev["event_id"])
                    break  # reserve order makes later rows ineligible until this is known
                if row.get("refused"):
                    if not isinstance(row.get("refusal_reason"), str) or not row["refusal_reason"]:
                        raise ValueError(f"refused event {ev['event_id']} lacks a reason")
                    refusals.append(row)
                    if ev["selected_primary"]:
                        pending_refused_primary.append(ev["event_id"])
                    continue
                if accepted < targets[stratum]:
                    for field in ("log_pattern_ratio_mid", "log_pattern_ratio_halfwidth"):
                        if not isinstance(row.get(field), (int, float)) \
                                or not math.isfinite(float(row[field])):
                            raise ValueError(f"usable event {ev['event_id']} has invalid {field}")
                    row = dict(row)
                    row["inclusion_probability"] = ev["inclusion_probability"]
                    row["manifest_role"] = "primary" if ev["selected_primary"] else "reserve"
                    row["selected_primary"] = True
                    replacement_id = (
                        pending_refused_primary.pop(0)
                        if not ev["selected_primary"] and pending_refused_primary
                        else None
                    )
                    row["replacement_for_refusal"] = replacement_id is not None
                    row["replacement_for_refusal_event_id"] = replacement_id
                    usable.append(row)
                    accepted += 1
                if accepted == targets[stratum]:
                    break
            if accepted != targets[stratum]:
                continue
    required = sum(int(v) for v in design["targets"].values())
    complete = len(usable) == required and not missing
    population = {
        str(band): {
            "eligible_by_stratum": [int(x) for x in manifest["design"][str(band)][
                "eligible_by_stratum"]],
            "eligible_total": int(manifest["design"][str(band)]["eligible_total"]),
        }
        for band in [int(x) for x in design["bands"]]
    }
    return {"schema": "gbskernels.verified-run.v2", "run_id": contract["run_id"],
            "complete": complete, "required_usable": required, "n_usable": len(usable),
            "n_refused": len(refusals),
            "n_evaluated_refused_total": sum(bool(r.get("refused")) for r in records.values()),
            "missing_event_ids": missing,
            "strata_count": int(design["n_strata"]), "population": population,
            "registration": normalized, "contract": contract,
            "rows": usable, "refusals": refusals}


def _load_registered_states(reg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    import coherence_family

    model = validate_registration(reg, require_beacon=True)["plan"]["models"]
    exp_id = int(model["exp_id"])
    parameterization = str(model.get("parameterization", "classical_excess"))
    points = model["coherence_points"]
    reference = str(model["reference_model"])
    alternative = str(model["alternative_model"])
    if reference == alternative or reference not in points or alternative not in points:
        raise ValueError("registered reference/alternative model names are invalid")
    order = [reference, alternative] + sorted(set(points) - {reference, alternative})
    return {name: coherence_family.jiuzhang_state(float(points[name]), exp_id=exp_id,
                                                   parameterization=parameterization)
            for name in order}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--run-root", default="results/confirmatory_v2/runs")
    ap.add_argument("--bands", default="")
    ap.add_argument("--slice", default="")
    ap.add_argument("--reduce", action="store_true")
    ap.add_argument("--run-id")
    ap.add_argument("--out")
    args = ap.parse_args()
    registration = load_json(args.registration)
    normalized = validate_registration(registration, require_beacon=True)
    manifest, _ = load_manifest(args.manifest)
    selection = normalized["plan"]["selection"]
    registered_bands = [int(x) for x in selection["bands"]]
    try:
        bands = ([int(x) for x in args.bands.split(",")] if args.bands
                 else registered_bands)
    except ValueError:
        ap.error("--bands must be a comma-separated integer list")
    if len(set(bands)) != len(bands) or any(band not in registered_bands for band in bands):
        ap.error("--bands must be a unique subset of the registered bands")
    if args.reduce:
        if not args.run_id or not args.out:
            ap.error("--reduce requires --run-id and --out")
        artifact = reduce_run(registration, manifest, Path(args.run_root) / args.run_id)
        write_json_exclusive(args.out, artifact)
        if not artifact["complete"]:
            print(f"incomplete: {artifact['n_usable']}/{artifact['required_usable']}", file=sys.stderr)
            return 2
        return 0
    states = _load_registered_states(registration)
    lo, hi = (0, 10**12)
    if args.slice:
        try:
            lo, hi = (int(x) for x in args.slice.split(":"))
        except ValueError:
            ap.error("--slice must have the form start:end")
        if lo < 0 or hi < lo:
            ap.error("--slice must be nonnegative start:end with end >= start")
    positions = {c: range(lo, min(hi, len(_band_candidates(manifest, c)))) for c in bands}
    run_id = evaluate_positions(registration, manifest, states, args.run_root,
                                bands, positions)
    print(run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
