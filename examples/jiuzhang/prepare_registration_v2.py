"""Prepare and resolve the machine-readable v2 public registration.

Two phases are intentionally separate:

``prepare-plan`` finalizes commit and state fingerprints and writes the exact
canonical plan that must be publicly archived before the beacon round.
``resolve`` consumes that unchanged public plan plus the later beacon value and
writes the registration accepted by selection/evaluation.

This utility performs no upload and no network request.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from campaign_confirmatory_v2 import state_fingerprint  # noqa: E402
from confirmatory_contract import (beacon_payload_sha256, derive_seed,
                                   sha256_json, validate_registration,
                                   write_canonical_json)  # noqa: E402
from confirmatory_common import (analysis_source_hash, analysis_sources_clean,
                                 current_commit, placeholder_paths)  # noqa: E402
from registration_readiness_v2 import build_readiness  # noqa: E402
import coherence_family  # noqa: E402


def prepare_plan(template: dict, *, exposure_catalog_path: str | Path | None = None,
                 exclusion_ledger_path: str | Path | None = None,
                 population_audit_path: str | Path | None = None,
                 design_spec_path: str | Path | None = None,
                 design_report_path: str | Path | None = None,
                 design_simulation_path: str | Path | None = None,
                 simulator_source_path: str | Path | None = None,
                 simulation_bank_path: str | Path | None = None,
                 refusal_recovery_source_path: str | Path | None = None,
                 calibration_path: str | Path | None = None,
                 evidence_root: str | Path = HERE.parents[1]) -> dict:
    plan = json.loads(json.dumps(template, sort_keys=True))
    commit = current_commit()
    if not commit:
        raise ValueError("cannot bind a registration without an analysis commit")
    if not analysis_sources_clean():
        raise ValueError("analysis sources must be tracked and clean before public registration")
    if plan.get("analysis_commit") not in (None, commit):
        raise ValueError("template analysis_commit differs from the current checkout")
    readiness = build_readiness(
        plan,
        exposure_catalog_path=exposure_catalog_path,
        exclusion_ledger_path=exclusion_ledger_path,
        population_audit_path=population_audit_path,
        design_spec_path=design_spec_path,
        design_report_path=design_report_path,
        design_simulation_path=design_simulation_path,
        simulator_source_path=simulator_source_path,
        simulation_bank_path=simulation_bank_path,
        refusal_recovery_source_path=refusal_recovery_source_path,
        calibration_path=calibration_path,
        evidence_root=evidence_root,
    )
    if not readiness["ready_for_public_timestamp"]:
        detail = "; ".join(
            f"{row['code']}: {row['detail']}" for row in readiness["blockers"])
        raise ValueError("registration inputs are not ready: " + detail)
    plan = readiness["candidate_plan"]
    plan["analysis_commit"] = commit
    placeholders = placeholder_paths(plan)
    if placeholders:
        raise ValueError("unresolved registration placeholders: " + ", ".join(placeholders))
    models = plan["models"]
    exp_id = int(models["exp_id"])
    parameterization = str(models.get("parameterization", "classical_excess"))
    states = {name: coherence_family.jiuzhang_state(
                  float(value), exp_id=exp_id, parameterization=parameterization)
              for name, value in models["coherence_points"].items()}
    plan.setdefault("numerical_contract", {})["state_fingerprints"] = state_fingerprint(states)
    plan["numerical_contract"]["analysis_source_sha256"] = analysis_source_hash()
    now = datetime.now(timezone.utc)
    preflight = {
        "schema": "gbskernels.confirmatory.v2", "plan": plan,
        "public": {
            "url": "https://example.invalid/preflight-plan",
            "sha256": sha256_json(plan), "plan_sha256": sha256_json(plan),
            "timestamp_utc": now.isoformat(), "immutable": True,
            "timestamp_proof_url": "https://example.invalid/preflight-timestamp",
            "timestamp_proof_sha256": "0" * 64,
        },
        "beacon": {
            "source": plan["randomness_beacon"]["source"],
            "round": int(plan["randomness_beacon"]["round"]),
            "availability_utc": plan["randomness_beacon"]["availability_utc"],
        },
    }
    validate_registration(preflight, require_beacon=False, now_utc=now)
    return plan


def resolve_registration(plan: dict, *, public_url: str, public_sha256: str,
                         published_at_utc: str, timestamp_proof_url: str,
                         timestamp_proof_sha256: str, beacon_source: str,
                         beacon_round: int, beacon_value: str,
                         beacon_proof_url: str,
                         beacon_proof_sha256: str) -> dict:
    plan = json.loads(json.dumps(plan, sort_keys=True))
    plan_hash = sha256_json(plan)
    frozen_beacon = plan["randomness_beacon"]
    if (beacon_source != frozen_beacon["source"]
            or int(beacon_round) != int(frozen_beacon["round"])):
        raise ValueError("beacon source/round differs from the publicly frozen plan")
    if public_sha256.lower() != plan_hash:
        raise ValueError("public SHA256 must hash the exact canonical plan")
    registration = {
        "schema": "gbskernels.confirmatory.v2", "plan": plan,
        "public": {"url": public_url, "sha256": public_sha256,
                   "plan_sha256": plan_hash, "timestamp_utc": published_at_utc,
                   "immutable": True,
                   "timestamp_proof_url": timestamp_proof_url,
                   "timestamp_proof_sha256": timestamp_proof_sha256},
        "beacon": {
            "source": beacon_source, "round": int(beacon_round), "value": beacon_value,
            "availability_utc": frozen_beacon["availability_utc"],
            "proof": {"source": beacon_source, "round": int(beacon_round),
                      "payload_sha256": beacon_payload_sha256(
                          beacon_source, int(beacon_round), beacon_value),
                      "url": beacon_proof_url,
                      "record_sha256": beacon_proof_sha256},
        },
    }
    registration["seed_derivation"] = derive_seed(
        public_url=public_url, public_sha256=public_sha256,
        published_at_utc=published_at_utc, beacon_source=beacon_source,
        beacon_round=int(beacon_round), beacon_value=beacon_value)
    validate_registration(registration, require_beacon=True)
    return registration


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-plan")
    prep.add_argument("--template", type=Path, required=True)
    prep.add_argument("--exposure-catalog", type=Path, required=True)
    prep.add_argument("--exclusion-ledger", type=Path, required=True)
    prep.add_argument("--evidence-root", type=Path, default=HERE.parents[1])
    prep.add_argument("--population-audit", type=Path, required=True)
    prep.add_argument("--design-spec", type=Path, required=True)
    prep.add_argument("--design-report", type=Path, required=True)
    prep.add_argument("--design-simulation", type=Path, required=True)
    prep.add_argument("--simulator-source", type=Path, required=True)
    prep.add_argument("--simulation-bank", type=Path, required=True)
    prep.add_argument("--refusal-recovery-source", type=Path, required=True)
    prep.add_argument("--calibration", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    resolve = sub.add_parser("resolve")
    resolve.add_argument("--plan", type=Path, required=True)
    resolve.add_argument("--public-url", required=True)
    resolve.add_argument("--public-sha256", required=True)
    resolve.add_argument("--published-at", required=True)
    resolve.add_argument("--timestamp-proof-url", required=True)
    resolve.add_argument("--timestamp-proof-sha256", required=True)
    resolve.add_argument("--beacon-source", required=True)
    resolve.add_argument("--beacon-round", type=int, required=True)
    resolve.add_argument("--beacon-value", required=True)
    resolve.add_argument("--beacon-proof-url", required=True)
    resolve.add_argument("--beacon-proof-sha256", required=True)
    resolve.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.command == "prepare-plan":
        value = prepare_plan(
            json.loads(args.template.read_text()),
            exposure_catalog_path=args.exposure_catalog,
            exclusion_ledger_path=args.exclusion_ledger,
            population_audit_path=args.population_audit,
            design_spec_path=args.design_spec,
            design_report_path=args.design_report,
            design_simulation_path=args.design_simulation,
            simulator_source_path=args.simulator_source,
            simulation_bank_path=args.simulation_bank,
            refusal_recovery_source_path=args.refusal_recovery_source,
            calibration_path=args.calibration,
            evidence_root=args.evidence_root,
        )
    else:
        value = resolve_registration(
            json.loads(args.plan.read_text()), public_url=args.public_url,
            public_sha256=args.public_sha256, published_at_utc=args.published_at,
            timestamp_proof_url=args.timestamp_proof_url,
            timestamp_proof_sha256=args.timestamp_proof_sha256,
            beacon_source=args.beacon_source, beacon_round=args.beacon_round,
            beacon_value=args.beacon_value,
            beacon_proof_url=args.beacon_proof_url,
            beacon_proof_sha256=args.beacon_proof_sha256)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    write_canonical_json(args.out, value)
    print(f"SHA256 {sha256_json(value)}  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
