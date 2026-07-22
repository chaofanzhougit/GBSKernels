"""Assemble a fail-closed confirmatory-v2 registration-readiness report.

Verified local artifacts fill a candidate plan, while missing or incomplete
external inputs remain blockers.  This command never creates a public
registration and never converts placeholders into invented values.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_exclusion_ledger import build_ledger  # noqa: E402
from confirmatory_common import (analysis_source_hash, analysis_sources_clean,
                                 current_commit, placeholder_paths,
                                 sha256_file)  # noqa: E402
from confirmatory_contract import (canonical_bytes, sha256_json,
                                   write_canonical_json)  # noqa: E402
from confirmatory_design import (DesignError, analysis_contract_from_plan,
                                 build_report as build_design_report)  # noqa: E402
from reconstruction_replicates import load_calibration  # noqa: E402
from select_confirmatory_v2 import _validated_ledger  # noqa: E402


READINESS_SCHEMA = "gbskernels.confirmatory-registration-readiness.v1"


class ReadinessError(ValueError):
    """Raised when a supplied artifact is malformed rather than merely absent."""


def _load_canonical(path: str | Path, field: str) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReadinessError(f"{field} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ReadinessError(f"{field} must be a JSON object")
    if raw != canonical_bytes(value):
        raise ReadinessError(f"{field} is not canonical JSON")
    return value


def _self_hash(value: Mapping[str, Any], hash_field: str, field: str) -> str:
    supplied = value.get(hash_field)
    if not isinstance(supplied, str) or len(supplied) != 64:
        raise ReadinessError(f"{field}.{hash_field} is invalid")
    body = {key: item for key, item in value.items() if key != hash_field}
    if sha256_json(body) != supplied:
        raise ReadinessError(f"{field} payload hash does not match its content")
    return supplied


def _set(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = target
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[path[-1]] = value


def build_readiness(
    template: Mapping[str, Any], *,
    exposure_catalog_path: str | Path | None = None,
    exclusion_ledger_path: str | Path | None = None,
    population_audit_path: str | Path | None = None,
    design_spec_path: str | Path | None = None,
    design_report_path: str | Path | None = None,
    design_simulation_path: str | Path | None = None,
    simulator_source_path: str | Path | None = None,
    simulation_bank_path: str | Path | None = None,
    refusal_recovery_source_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
    evidence_root: str | Path = HERE.parents[1],
) -> dict[str, Any]:
    plan = deepcopy(dict(template))
    blockers: list[dict[str, str]] = []
    verified: dict[str, Any] = {}

    if (refusal_recovery_source_path is None
            or not Path(refusal_recovery_source_path).is_file()):
        blockers.append({
            "code": "missing_refusal_recovery_source",
            "detail": "frozen independent refusal-recovery source is required",
        })
    else:
        recovery_source_hash = sha256_file(refusal_recovery_source_path)
        declared = plan.get("analysis", {}).get("refusal_analysis", {}).get(
            "recovery_source_sha256")
        if (isinstance(declared, str) and not declared.startswith("REPLACE-")
                and declared.lower() != recovery_source_hash):
            blockers.append({
                "code": "refusal_recovery_source_mismatch",
                "detail": "recovery source bytes differ from the plan declaration",
            })
        _set(plan, ("analysis", "refusal_analysis", "recovery_source_sha256"),
             recovery_source_hash)
        verified["refusal_recovery_source"] = {
            "path": str(refusal_recovery_source_path),
            "sha256": recovery_source_hash,
            "container_digest": plan["analysis"]["refusal_analysis"].get(
                "recovery_container_digest"),
            "container_note": "digest syntax is validated; image availability is external",
        }

    catalog = None
    if exposure_catalog_path is None:
        blockers.append({"code": "missing_exposure_catalog",
                         "detail": "canonical exploratory exposure catalog is required"})
    else:
        catalog = _load_canonical(exposure_catalog_path, "exposure_catalog")
        if catalog.get("schema") != "gbskernels.exploratory-exposure-catalog.v1":
            raise ReadinessError("unsupported exposure-catalog schema")
        verified["exposure_catalog"] = {
            "path": str(exposure_catalog_path),
            "payload_sha256": sha256_json(catalog),
        }

    ledger = None
    if exclusion_ledger_path is None:
        blockers.append({"code": "missing_exclusion_ledger",
                         "detail": "complete author-attested exposure ledger is required"})
    else:
        ledger = _load_canonical(exclusion_ledger_path, "exclusion_ledger")
        if catalog is not None:
            try:
                rebuilt = build_ledger(evidence_root, catalog, require_complete=False)
            except ValueError as exc:
                raise ReadinessError(
                    f"exclusion ledger cannot be reconstructed from its catalog: {exc}") from exc
            if ledger != rebuilt:
                raise ReadinessError(
                    "exclusion ledger does not reproduce from the exposure catalog")
        indices, ledger_hash = _validated_ledger(ledger, require_complete=False)
        source = ledger.get("source")
        if not isinstance(source, Mapping):
            raise ReadinessError("exclusion ledger source is invalid")
        _set(plan, ("selection", "n_records"), int(source["n_records"]))
        _set(plan, ("selection", "record_bytes"), int(source["record_bytes"]))
        _set(plan, ("selection", "source_raw_sha256"), str(source["sha256"]))
        _set(plan, ("selection", "exclusion_sha256"), str(ledger["exclusion_sha256"]))
        _set(plan, ("selection", "exclusion_ledger_sha256"), ledger_hash)
        verified["exclusion_ledger"] = {
            "path": str(exclusion_ledger_path), "payload_sha256": ledger_hash,
            "record_indices": len(indices), "complete": ledger.get("complete") is True,
        }
        if ledger.get("complete") is not True:
            blockers.append({"code": "incomplete_exclusion_attestation",
                             "detail": "ledger evidence verifies, but author attestation is incomplete"})

    audit = None
    if population_audit_path is None:
        blockers.append({"code": "missing_population_audit",
                         "detail": "full eligible-population audit is required"})
    else:
        audit = _load_canonical(population_audit_path, "population_audit")
        audit_hash = _self_hash(audit, "audit_payload_sha256", "population_audit")
        if audit.get("schema") != "gbskernels.selection-population-audit.v1":
            raise ReadinessError("unsupported population-audit schema")
        if ledger is not None:
            expected = {
                "source_raw_sha256": ledger["source"]["sha256"],
                "exclusion_sha256": ledger["exclusion_sha256"],
                "exclusion_ledger_sha256": ledger["ledger_payload_sha256"],
                "n_records": ledger["source"]["n_records"],
                "record_bytes": ledger["source"]["record_bytes"],
            }
            for field, value in expected.items():
                if audit.get(field) != value:
                    raise ReadinessError(
                        f"population audit {field} differs from the exclusion ledger")
        _set(plan, ("selection", "population_audit_sha256"), audit_hash)
        _set(plan, ("selection", "bands"), [int(x) for x in audit["bands"]])
        _set(plan, ("selection", "n_strata"), int(audit["n_strata"]))
        _set(plan, ("analysis", "band_weights"), {
            str(key): float(value)
            for key, value in audit["band_weights_within_window"].items()})
        verified["population_audit"] = {
            "path": str(population_audit_path), "payload_sha256": audit_hash,
            "registration_eligible": audit.get("registration_eligible") is True,
        }
        if audit.get("registration_eligible") is not True:
            blockers.append({"code": "draft_population_audit",
                             "detail": "audit was generated from an incomplete ledger"})

    design_paths = {
        "design_spec": design_spec_path,
        "design_report": design_report_path,
        "design_simulation": design_simulation_path,
        "simulator_source": simulator_source_path,
        "simulation_bank": simulation_bank_path,
    }
    for role, path in design_paths.items():
        if path is None:
            blockers.append({
                "code": f"missing_{role}",
                "detail": f"{role.replace('_', ' ')} is required to reproduce the design",
            })

    design = None
    if all(path is not None for path in design_paths.values()) and audit is None:
        blockers.append({
            "code": "design_not_reproduced",
            "detail": "design inputs cannot be reproduced without the population audit",
        })
    elif all(path is not None for path in design_paths.values()):
        spec = _load_canonical(design_spec_path, "design_spec")
        simulation = _load_canonical(design_simulation_path, "design_simulation")
        design = _load_canonical(design_report_path, "design_report")
        simulator_source_hash = sha256_file(simulator_source_path)
        simulation_bank_hash = sha256_file(simulation_bank_path)
        source = simulation.get("source")
        if not isinstance(source, Mapping):
            raise ReadinessError("design simulation lacks source provenance")
        if (simulator_source_hash != spec.get("simulator_source_sha256")
                or simulator_source_hash != source.get("simulator_source_sha256")):
            raise ReadinessError("simulator source bytes differ from the design declarations")
        if simulation_bank_hash != source.get("simulation_bank_sha256"):
            raise ReadinessError("simulation bank bytes differ from the design declaration")
        try:
            reproduced = build_design_report(spec, audit, simulation)
        except DesignError as exc:
            raise ReadinessError(f"design inputs do not validate: {exc}") from exc
        if design != reproduced:
            raise ReadinessError(
                "design report does not reproduce from the supplied spec, audit, and simulation")
        design_hash = _self_hash(
            design, "design_report_payload_sha256", "design_report")
        if design.get("schema") != "gbskernels.confirmatory-design.v1":
            raise ReadinessError("unsupported design-report schema")
        try:
            expected_analysis = analysis_contract_from_plan(plan)
        except DesignError as exc:
            raise ReadinessError(str(exc)) from exc
        if design.get("analysis_contract") != expected_analysis:
            raise ReadinessError(
                "design report analysis contract differs from the registration plan")
        if design.get("pass") is not True or not isinstance(design.get("selected"), Mapping):
            blockers.append({"code": "no_feasible_design",
                             "detail": "design report did not select a feasible candidate"})
        else:
            selected = design["selected"]
            _set(plan, ("selection", "targets"), dict(selected["targets"]))
            _set(plan, ("selection", "reserves"), dict(selected["reserves"]))
            thresholds = selected["predictive_thresholds"]["thresholds"]
            _set(plan, ("analysis", "predictive_checks", "thresholds"), dict(thresholds))
        _set(plan, ("analysis", "design_report_sha256"), design_hash)
        criteria = design.get("criteria", {})
        for source_key, plan_key in (
            ("minimum_relevant_coherence", "minimum_relevant_coherence"),
            ("power_min", "target_power"),
            ("type_i_error_max", "type_i_error_max"),
            ("monte_carlo_confidence", "monte_carlo_confidence"),
        ):
            if source_key in criteria:
                _set(plan, ("analysis", plan_key), criteria[source_key])
        if audit is not None and design.get("population_audit_sha256") != audit.get(
                "audit_payload_sha256"):
            raise ReadinessError("design report belongs to a different population audit")
        verified["design_report"] = {
            "path": str(design_report_path), "payload_sha256": design_hash,
            "pass": design.get("pass") is True,
        }
        verified["design_spec"] = {
            "path": str(design_spec_path), "payload_sha256": sha256_json(spec),
        }
        verified["design_simulation"] = {
            "path": str(design_simulation_path),
            "payload_sha256": sha256_json(simulation),
        }
        verified["simulator_source"] = {
            "path": str(simulator_source_path), "sha256": simulator_source_hash,
        }
        verified["simulation_bank"] = {
            "path": str(simulation_bank_path), "sha256": simulation_bank_hash,
        }

    if calibration_path is None:
        blockers.append({"code": "missing_calibration_posterior",
                         "detail": "independent physical calibration posterior is required"})
    else:
        calibration_hash = sha256_file(calibration_path)
        bands = [int(x) for x in plan["selection"]["bands"]]
        n_strata = int(plan["selection"]["n_strata"])
        calibration = load_calibration(calibration_path, bands, n_strata=n_strata)
        _set(plan, ("analysis", "calibration_draws", "posterior_sha256"), calibration_hash)
        _set(plan, ("analysis", "calibration_draws", "count"), len(calibration["r25"]))
        if design is not None and design.get("calibration_posterior_sha256") != calibration_hash:
            raise ReadinessError("design report used a different calibration posterior")
        verified["calibration_posterior"] = {
            "path": str(calibration_path), "sha256": calibration_hash,
            "draws": len(calibration["r25"]),
        }

    commit = current_commit()
    source_hash = analysis_source_hash()
    clean = analysis_sources_clean()
    verified["analysis_source"] = {
        "commit": commit, "sha256": source_hash, "tracked_and_clean": clean}
    if not clean:
        blockers.append({"code": "dirty_analysis_source",
                         "detail": "analysis source must be committed and clean"})
    else:
        _set(plan, ("analysis_commit",), commit)
        _set(plan, ("numerical_contract", "analysis_source_sha256"), source_hash)

    placeholders = placeholder_paths(plan)
    if placeholders:
        blockers.append({"code": "unresolved_plan_fields",
                         "detail": ", ".join(placeholders)})
    body = {
        "schema": READINESS_SCHEMA,
        "ready_for_public_timestamp": not blockers,
        "verified_inputs": verified,
        "blockers": blockers,
        "candidate_plan": plan,
        "unresolved_placeholder_paths": placeholders,
    }
    return {**body, "readiness_payload_sha256": sha256_json(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--exposure-catalog", type=Path)
    parser.add_argument("--exclusion-ledger", type=Path)
    parser.add_argument("--evidence-root", type=Path, default=HERE.parents[1])
    parser.add_argument("--population-audit", type=Path)
    parser.add_argument("--design-spec", type=Path)
    parser.add_argument("--design-report", type=Path)
    parser.add_argument("--design-simulation", type=Path)
    parser.add_argument("--simulator-source", type=Path)
    parser.add_argument("--simulation-bank", type=Path)
    parser.add_argument("--refusal-recovery-source", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    template = json.loads(args.template.read_text())
    report = build_readiness(
        template,
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
    write_canonical_json(args.out, report)
    print(f"ready={str(report['ready_for_public_timestamp']).lower()}")
    for blocker in report["blockers"]:
        print(f"BLOCKER {blocker['code']}: {blocker['detail']}")
    print(args.out)
    return 0 if report["ready_for_public_timestamp"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
