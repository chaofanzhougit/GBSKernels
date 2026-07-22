"""Assemble and verify the hash-pinned confirmatory v2 release bundle.

The release manifest is the only supported input to a paper or figure build.
It refuses missing scientific layers, binds every artifact to one registration
and run, and records byte-level SHA256 digests. ``verify`` fails after any file
is replaced, truncated, or silently regenerated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from campaign_confirmatory_v2 import (load_manifest,
                                      validate_manifest_registration)  # noqa: E402
from analyze_refusals import validate_refusal_analysis  # noqa: E402
from build_exclusion_ledger import build_ledger  # noqa: E402
from confirmatory_common import (hash_json, load_json, sha256_file, valid_container_digest,
                                 write_json_exclusive)  # noqa: E402
from confirmatory_contract import load_registration, sha256_json  # noqa: E402
from confirmatory_design import (analysis_contract_from_plan,
                                 build_report as build_design_report)  # noqa: E402
from confirmatory_inference import (predictive_model_gate, registered_nonclassical_decision,
                                    validate_absolute_predictive_checks)  # noqa: E402
from reconstruction_replicates import (CALIBRATION_FINGERPRINT_METHOD,
                                       PAIRED_NORMALIZER_FINGERPRINT_METHOD)  # noqa: E402
from select_confirmatory_v2 import _validated_ledger  # noqa: E402


_REQUIRED_RELEASE_ARTIFACTS = frozenset({
    "registration",
    "exposure_catalog",
    "exclusion_ledger",
    "population_audit",
    "design_spec",
    "design_report",
    "design_simulation",
    "simulator_source",
    "simulation_bank",
    "manifest",
    "verified_run",
    "normalizers",
    "analysis",
})
_CONDITIONAL_RELEASE_ARTIFACTS = frozenset({
    "reconstruction",
    "calibration",
    "calibration_normalizers",
    "predictive_checks",
    "refusal_analysis",
    "refusal_recovery",
    "refusal_recovery_source",
})


def _portable_path(path: Path, root: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(root.resolve()))
    except ValueError:
        raise ValueError(f"release artifact must live under --root: {path}")


def _entry(path: str | Path, role: str, root: Path) -> dict[str, Any]:
    value = Path(path)
    if not value.is_file():
        raise ValueError(f"required {role} artifact is missing: {value}")
    return {"role": role, "path": _portable_path(value, root),
            "sha256": sha256_file(value), "bytes": value.stat().st_size}


def _npz_meta(path: str | Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    if "meta" not in z.files:
        raise ValueError(f"NPZ artifact lacks meta: {path}")
    return json.loads(str(z["meta"]))


def _json_object(path: str | Path, role: str) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{role} artifact must be a JSON object")
    return value


def _normalized_counts(value: Any, field: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    try:
        return {str(key): int(count) for key, count in value.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} contains a non-integer count") from exc


def _require_run_provenance(value: dict[str, Any], contract: dict[str, Any],
                            role: str) -> None:
    expected = {
        "analysis_commit": contract.get("analysis_commit"),
        "analysis_source_sha256": contract.get("analysis_source_sha256"),
        "container_digest": contract.get("container_digest"),
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError(f"{role} artifact provenance differs from the run contract")


def _require_analysis_inputs(analysis: dict[str, Any],
                             paths: dict[str, str | Path]) -> None:
    fields = {
        "verified_run": "verified_run_sha256",
        "normalizers": "normalizer_replicates_sha256",
        "reconstruction": "reconstruction_replicates_sha256",
        "predictive_checks": "predictive_checks_sha256",
        "refusal_analysis": "refusal_analysis_sha256",
    }
    expected = {
        field: sha256_file(paths[role])
        for role, field in fields.items()
        if role in paths
    }
    if analysis.get("inputs") != expected:
        raise ValueError("analysis input hashes differ from the released artifacts")


def _validate_calibration_normalizer_meta(
    meta: dict[str, Any], *, registration_id: str, calibration_sha256: str,
    plan: dict[str, Any],
) -> None:
    calibration_cfg = plan["analysis"]["calibration_draws"]
    normalizer_cfg = plan["analysis"]["normalizer_replicates"]
    expected_seed_rule = "seed + draw_index * n_strata + stratum; common across models"
    if (meta.get("schema") != "gbskernels.calibration-normalizer-draws.v1"
            or meta.get("registration_id") != registration_id
            or meta.get("calibration_posterior_sha256") != calibration_sha256
            or meta.get("draws") != int(calibration_cfg["count"])
            or meta.get("samples_per_draw_stratum")
            != int(normalizer_cfg["samples_per_replicate"])
            or meta.get("seed") != int(calibration_cfg["seed"])
            or meta.get("seed_rule") != expected_seed_rule
            or meta.get("calibration_draw_fingerprint_method")
            != CALIBRATION_FINGERPRINT_METHOD
            or meta.get("paired_normalizer_fingerprint_method")
            != PAIRED_NORMALIZER_FINGERPRINT_METHOD):
        raise ValueError("calibration-normalizer artifact binding is invalid")


def _validate_preregistration_artifacts(
    registration: dict[str, Any], population_audit: dict[str, Any], *,
    exposure_catalog_path: str | Path, exclusion_ledger_path: str | Path,
    design_spec_path: str | Path, design_report_path: str | Path,
    design_simulation_path: str | Path, simulator_source_path: str | Path,
    simulation_bank_path: str | Path, root: str | Path,
) -> dict[str, str]:
    """Validate and return the semantic identifiers of every design input."""
    plan = registration["plan"]
    selection = plan["selection"]
    analysis = plan["analysis"]

    catalog = _json_object(exposure_catalog_path, "exposure catalog")
    if catalog.get("schema") != "gbskernels.exploratory-exposure-catalog.v1":
        raise ValueError("unsupported exposure-catalog schema")
    ledger = _json_object(exclusion_ledger_path, "exclusion ledger")
    try:
        rebuilt_ledger = build_ledger(root, catalog, require_complete=True)
    except ValueError as exc:
        raise ValueError(f"released exclusion ledger cannot be reconstructed: {exc}") from exc
    if ledger != rebuilt_ledger:
        raise ValueError("exclusion ledger does not reproduce from the released catalog")
    _, ledger_hash = _validated_ledger(ledger, require_complete=True)
    catalog_hash = sha256_json(catalog)
    if ledger.get("catalog_payload_sha256") != catalog_hash:
        raise ValueError("exclusion ledger belongs to a different exposure catalog")
    catalog_source = catalog.get("raw_source")
    ledger_source = ledger.get("source")
    if not isinstance(catalog_source, dict) or not isinstance(ledger_source, dict):
        raise ValueError("exposure catalog or exclusion ledger lacks raw-source metadata")
    for field in ("path", "sha256", "record_bytes", "n_records"):
        if catalog_source.get(field) != ledger_source.get(field):
            raise ValueError(f"exposure catalog and exclusion ledger differ on {field}")
    expected_selection = {
        "source_raw_sha256": ledger_source["sha256"],
        "n_records": ledger_source["n_records"],
        "record_bytes": ledger_source["record_bytes"],
        "exclusion_sha256": ledger["exclusion_sha256"],
        "exclusion_ledger_sha256": ledger_hash,
    }
    if any(selection.get(field) != expected
           for field, expected in expected_selection.items()):
        raise ValueError("exclusion ledger differs from the registered selection")
    if (population_audit.get("exclusion_ledger_sha256") != ledger_hash
            or population_audit.get("exclusion_sha256") != ledger["exclusion_sha256"]
            or population_audit.get("registration_eligible") is not True):
        raise ValueError("population audit is not bound to the complete exclusion ledger")

    spec = _json_object(design_spec_path, "design specification")
    simulation = _json_object(design_simulation_path, "design simulation")
    report = _json_object(design_report_path, "design report")
    try:
        recomputed_report = build_design_report(spec, population_audit, simulation)
    except ValueError as exc:
        raise ValueError(f"released design inputs do not validate: {exc}") from exc
    if report != recomputed_report:
        raise ValueError("design report does not reproduce from the released inputs")
    try:
        registered_analysis_contract = analysis_contract_from_plan(registration["plan"])
    except ValueError as exc:
        raise ValueError(f"registration lacks a complete design analysis contract: {exc}") from exc
    if report.get("analysis_contract") != registered_analysis_contract:
        raise ValueError("design analysis contract differs from the registration")
    simulation_source = simulation.get("source", {})
    simulator_source_hash = sha256_file(simulator_source_path)
    simulation_bank_hash = sha256_file(simulation_bank_path)
    if (simulator_source_hash != str(spec.get("simulator_source_sha256", "")).lower()
            or simulator_source_hash
            != str(simulation_source.get("simulator_source_sha256", "")).lower()):
        raise ValueError("released simulator source differs from the design declarations")
    if simulation_bank_hash \
            != str(simulation_source.get("simulation_bank_sha256", "")).lower():
        raise ValueError("released simulation bank differs from the design declaration")
    report_hash = report.get("design_report_payload_sha256")
    if report.get("pass") is not True or report_hash != analysis.get("design_report_sha256"):
        raise ValueError("design report is not passing or differs from the registration")
    selected = report.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("design report lacks its selected design")
    if (_normalized_counts(selected.get("targets"), "design targets")
            != _normalized_counts(selection.get("targets"), "registered targets")
            or _normalized_counts(selected.get("reserves"), "design reserves")
            != _normalized_counts(selection.get("reserves"), "registered reserves")):
        raise ValueError("registered targets or reserves differ from the design report")
    thresholds = selected.get("predictive_thresholds", {}).get("thresholds")
    registered_thresholds = analysis.get("predictive_checks", {}).get("thresholds")
    if thresholds != registered_thresholds:
        raise ValueError("registered predictive thresholds differ from the design report")
    criteria = report.get("criteria", {})
    criteria_bindings = {
        "minimum_relevant_coherence": "minimum_relevant_coherence",
        "power_min": "target_power",
        "type_i_error_max": "type_i_error_max",
        "monte_carlo_confidence": "monte_carlo_confidence",
    }
    if any(criteria.get(source) != analysis.get(target)
           for source, target in criteria_bindings.items()):
        raise ValueError("registered operating criteria differ from the design report")
    calibration = analysis.get("calibration_draws")
    if (not isinstance(calibration, dict)
            or report.get("calibration_posterior_sha256")
            != calibration.get("posterior_sha256")):
        raise ValueError("design report used a different calibration posterior")
    return {
        "exposure_catalog_payload_sha256": catalog_hash,
        "exclusion_ledger_payload_sha256": ledger_hash,
        "population_audit_payload_sha256": str(population_audit["audit_payload_sha256"]),
        "design_spec_payload_sha256": sha256_json(spec),
        "design_report_payload_sha256": str(report_hash),
        "design_simulation_payload_sha256": sha256_json(simulation),
        "simulator_source_sha256": simulator_source_hash,
        "simulation_bank_sha256": simulation_bank_hash,
    }


def _registered_decision_for_release(
    registration: dict[str, Any], analysis: dict[str, Any], *,
    predictive_required: bool,
) -> dict[str, Any]:
    """Recompute the registered decision without requiring a positive claim."""
    result = analysis.get("result")
    if not isinstance(result, dict):
        raise ValueError("analysis artifact lacks its result")
    registered_primary = registration["plan"]["analysis"]["primary_estimand"]
    if result.get("registered_primary_estimand") != registered_primary:
        raise ValueError("analysis did not execute the registered primary estimand")
    if predictive_required:
        expected_policy = registration["plan"]["analysis"]["predictive_checks"].get(
            "model_pass_policy", "any_registered")
        gate = result.get("predictive_model_gate")
        if not isinstance(gate, dict) or gate.get("policy") != expected_policy:
            raise ValueError("analysis did not execute the registered predictive model gate")
    expected = registered_nonclassical_decision(
        result, registration["plan"]["analysis"])
    if result.get("registered_decision") != expected:
        raise ValueError("analysis registered decision is missing or inconsistent")
    return expected


def assemble(*, registration_path: str | Path, manifest_path: str | Path,
             exposure_catalog_path: str | Path, exclusion_ledger_path: str | Path,
             population_audit_path: str | Path, design_spec_path: str | Path,
             design_report_path: str | Path, design_simulation_path: str | Path,
             simulator_source_path: str | Path, simulation_bank_path: str | Path,
             verified_run_path: str | Path, normalizers_path: str | Path,
             reconstruction_path: str | Path | None,
             calibration_path: str | Path | None,
             calibration_normalizers_path: str | Path | None,
             predictive_checks_path: str | Path | None,
             analysis_path: str | Path, refusal_analysis_path: str | Path | None,
             refusal_recovery_path: str | Path | None,
             refusal_recovery_source_path: str | Path | None,
             root: str | Path) -> dict[str, Any]:
    root = Path(root)
    registration = load_registration(registration_path)
    registration_id = registration["public"]["plan_sha256"]
    population_audit = load_json(population_audit_path)
    population_body = {k: v for k, v in population_audit.items()
                       if k != "audit_payload_sha256"}
    if (population_audit.get("schema") != "gbskernels.selection-population-audit.v1"
            or population_audit.get("audit_payload_sha256") != sha256_json(population_body)
            or population_audit.get("audit_payload_sha256")
            != registration["plan"]["selection"]["population_audit_sha256"]):
        raise ValueError("population-audit artifact binding is invalid")
    preregistration = _validate_preregistration_artifacts(
        registration,
        population_audit,
        exposure_catalog_path=exposure_catalog_path,
        exclusion_ledger_path=exclusion_ledger_path,
        design_spec_path=design_spec_path,
        design_report_path=design_report_path,
        design_simulation_path=design_simulation_path,
        simulator_source_path=simulator_source_path,
        simulation_bank_path=simulation_bank_path,
        root=root,
    )
    manifest, manifest_id = load_manifest(manifest_path)
    validate_manifest_registration(manifest, registration)
    if manifest.get("population_audit") != population_audit:
        raise ValueError("standalone population audit differs from the manifest")
    run = load_json(verified_run_path)
    if run.get("schema") != "gbskernels.verified-run.v2" or not run.get("complete"):
        raise ValueError("release requires a complete verified run")
    if run.get("registration", {}).get("public", {}).get("plan_sha256") != registration_id:
        raise ValueError("verified run belongs to a different registration")
    contract = run.get("contract", {})
    contract_body = {key: value for key, value in contract.items() if key != "run_id"}
    if (run.get("run_id") != contract.get("run_id")
            or contract.get("run_id") != hash_json(contract_body)):
        raise ValueError("verified run contract hash is invalid")
    if run.get("contract", {}).get("manifest_id") != manifest_id:
        raise ValueError("verified run belongs to a different manifest")
    if run.get("contract", {}).get("analysis_source_sha256") \
            != registration["plan"]["numerical_contract"]["analysis_source_sha256"]:
        raise ValueError("verified run source hash differs from registration")
    if not valid_container_digest(contract.get("container_digest")):
        raise ValueError("release requires a pinned container digest")
    run_id = str(run["run_id"])
    def require_provenance(value: dict[str, Any], role: str) -> None:
        _require_run_provenance(value, contract, role)

    normalizer_meta = _npz_meta(normalizers_path)
    if (normalizer_meta.get("schema") != "gbskernels.joint-normalizer-replicates.v1"
            or normalizer_meta.get("registration_id") != registration_id):
        raise ValueError("normalizer artifact binding is invalid")
    require_provenance(normalizer_meta, "normalizer")
    reconstruction_required = bool(
        registration["plan"]["external_requirements"].get("reconstruction_required", False))
    if reconstruction_required:
        if (reconstruction_path is None or calibration_path is None
                or calibration_normalizers_path is None):
            raise ValueError(
                "registration requires reconstruction, calibration, and calibration normalizers"
            )
        reconstruction_meta = _npz_meta(reconstruction_path)
        if (reconstruction_meta.get("schema") != "gbskernels.reconstruction-replicates.v3"
                or reconstruction_meta.get("registration_id") != registration_id
                or reconstruction_meta.get("run_id") != run_id):
            raise ValueError("reconstruction artifact binding is invalid")
        require_provenance(reconstruction_meta, "reconstruction")
        calibration_sha256 = sha256_file(calibration_path)
        calibration_normalizers_sha256 = sha256_file(calibration_normalizers_path)
        if (calibration_sha256 != registration["plan"]["analysis"][
                "calibration_draws"]["posterior_sha256"]
                or reconstruction_meta.get("calibration_sha256") != calibration_sha256
                or reconstruction_meta.get("normalizer_draws_sha256")
                != calibration_normalizers_sha256
                or reconstruction_meta.get("nominal_normalizers_sha256")
                != sha256_file(normalizers_path)):
            raise ValueError("reconstruction input hashes are not release-complete")
        calibration_normalizer_meta = _npz_meta(calibration_normalizers_path)
        _validate_calibration_normalizer_meta(
            calibration_normalizer_meta, registration_id=registration_id,
            calibration_sha256=calibration_sha256, plan=registration["plan"])
        require_provenance(calibration_normalizer_meta, "calibration-normalizer")
    predictive_required = bool(
        registration["plan"]["external_requirements"].get(
            "absolute_predictive_checks_required", False))
    checks = None
    if predictive_required:
        if predictive_checks_path is None:
            raise ValueError("registration requires absolute predictive checks")
        checks = load_json(predictive_checks_path)
        if (checks.get("schema") != "gbskernels.absolute-predictive-checks.v1"
                or checks.get("registration_id") != registration_id
                or checks.get("run_id") != run_id):
            raise ValueError("predictive-check artifact binding is invalid")
        require_provenance(checks, "predictive-check")
    analysis = load_json(analysis_path)
    if (analysis.get("schema") != "gbskernels.confirmatory-analysis.v2"
            or analysis.get("registration_id") != registration_id
            or analysis.get("run_id") != run_id):
        raise ValueError("analysis artifact binding is invalid")
    require_provenance(analysis, "analysis")
    model_cfg = registration["plan"]["models"]
    model_names = [str(model_cfg["reference_model"]),
                   str(model_cfg["alternative_model"])] + sorted(
                       set(model_cfg["coherence_points"])
                       - {model_cfg["reference_model"],
                          model_cfg["alternative_model"]})
    if predictive_required:
        predictive_cfg = registration["plan"]["analysis"]["predictive_checks"]
        model_passes = validate_absolute_predictive_checks(
            checks, run_id=run_id, registration_id=registration_id,
            selection_cfg=registration["plan"]["selection"],
            predictive_cfg=predictive_cfg, model_names=model_names,
            expected_provenance={
                "analysis_commit": contract["analysis_commit"],
                "analysis_source_sha256": contract["analysis_source_sha256"],
                "container_digest": contract["container_digest"],
            })
        result = analysis.get("result")
        if (not isinstance(result, dict)
                or result.get("absolute_predictive_checks") != checks):
            raise ValueError("analysis used different absolute predictive checks")
        expected_gate = predictive_model_gate(
            result, model_passes,
            policy=predictive_cfg.get("model_pass_policy", "any_registered"),
            model_cfg=model_cfg)
        if result.get("predictive_model_gate") != expected_gate:
            raise ValueError("analysis predictive model gate is inconsistent")
    registered_primary = registration["plan"]["analysis"]["primary_estimand"]
    registered_decision = _registered_decision_for_release(
        registration, analysis, predictive_required=predictive_required)

    paths = {
        "registration": registration_path,
        "exposure_catalog": exposure_catalog_path,
        "exclusion_ledger": exclusion_ledger_path,
        "population_audit": population_audit_path,
        "design_spec": design_spec_path,
        "design_report": design_report_path,
        "design_simulation": design_simulation_path,
        "simulator_source": simulator_source_path,
        "simulation_bank": simulation_bank_path,
        "manifest": manifest_path,
        "verified_run": verified_run_path, "normalizers": normalizers_path,
        "analysis": analysis_path,
    }
    if reconstruction_required:
        paths["reconstruction"] = reconstruction_path
        paths["calibration"] = calibration_path
        paths["calibration_normalizers"] = calibration_normalizers_path
    if predictive_required:
        paths["predictive_checks"] = predictive_checks_path
    if int(run.get("n_refused", 0)):
        if (refusal_analysis_path is None or refusal_recovery_path is None
                or refusal_recovery_source_path is None):
            raise ValueError(
                "refused events require release-bound analysis, recovery, and recovery source")
        refusal = load_json(refusal_analysis_path)
        validate_refusal_analysis(
            refusal, run=run, registration_id=registration_id,
            config=registration["plan"]["analysis"]["refusal_analysis"],
            verified_run_sha256=sha256_file(verified_run_path),
            model_names=model_names)
        recovery = _json_object(refusal_recovery_path, "refusal recovery")
        refusal_inputs = refusal["inputs"]
        if (recovery != refusal["recovery"]
                or sha256_file(refusal_recovery_path)
                != refusal_inputs["recovered_input_sha256"]
                or sha256_file(refusal_recovery_source_path)
                != refusal_inputs["recovery_source_sha256"]):
            raise ValueError("released refusal recovery or source differs from the analysis")
        if (reconstruction_required
                and reconstruction_meta.get("refusal_analysis_sha256")
                != sha256_file(refusal_analysis_path)):
            raise ValueError("reconstruction used a different refusal analysis")
        paths["refusal_analysis"] = refusal_analysis_path
        paths["refusal_recovery"] = refusal_recovery_path
        paths["refusal_recovery_source"] = refusal_recovery_source_path
    _require_analysis_inputs(analysis, paths)
    entries = {name: _entry(path, name, root) for name, path in paths.items()}
    body = {
        "schema": "gbskernels.confirmatory-release.v2",
        "registration_id": registration_id, "manifest_id": manifest_id,
        "run_id": run_id, "analysis_commit": run["contract"]["analysis_commit"],
        "container_digest": run["contract"]["container_digest"],
        "kernel_binary": run["contract"]["kernel_binary"],
        "analysis_source_sha256": run["contract"]["analysis_source_sha256"],
        "registered_primary_estimand": registered_primary,
        "registered_decision": registered_decision,
        "population_scope": registration["plan"]["analysis"]["population_scope"],
        "numerical_scope": registration["plan"]["numerical_contract"]["scope"],
        "preregistration": preregistration,
        "artifacts": entries,
    }
    return {**body, "release_payload_sha256": sha256_json(body)}


def verify_release(path: str | Path, *, root: str | Path) -> dict[str, Any]:
    release = load_json(path)
    if release.get("schema") != "gbskernels.confirmatory-release.v2":
        raise ValueError("unsupported release schema")
    body = {k: v for k, v in release.items() if k != "release_payload_sha256"}
    if release.get("release_payload_sha256") != sha256_json(body):
        raise ValueError("release manifest content hash mismatch")
    artifacts = release.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("release artifact map is missing")
    missing = sorted(_REQUIRED_RELEASE_ARTIFACTS - set(artifacts))
    if missing:
        raise ValueError("release is missing required artifact roles: " + ", ".join(missing))
    unexpected = sorted(
        set(artifacts) - _REQUIRED_RELEASE_ARTIFACTS - _CONDITIONAL_RELEASE_ARTIFACTS)
    if unexpected:
        raise ValueError("release has unsupported artifact roles: " + ", ".join(unexpected))
    decision = release.get("registered_decision")
    if (not isinstance(decision, dict)
            or not isinstance(decision.get("claim_supported"), bool)):
        raise ValueError("release lacks a valid registered decision")
    root = Path(root).resolve()
    resolved: dict[str, Path] = {}
    for name, entry in artifacts.items():
        if (not isinstance(entry, dict) or entry.get("role") != name
                or not isinstance(entry.get("path"), str)):
            raise ValueError(f"release artifact entry is malformed: {name}")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"release artifact path escapes root: {name}")
        artifact = (root / relative).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"release artifact path escapes root: {name}") from exc
        if (not artifact.is_file()
                or artifact.stat().st_size != int(entry.get("bytes", -1))):
            raise ValueError(f"release artifact missing or wrong size: {name}")
        if sha256_file(artifact) != entry.get("sha256"):
            raise ValueError(f"release artifact hash mismatch: {name}")
        resolved[name] = artifact

    reconstructed = assemble(
        registration_path=resolved["registration"],
        manifest_path=resolved["manifest"],
        exposure_catalog_path=resolved["exposure_catalog"],
        exclusion_ledger_path=resolved["exclusion_ledger"],
        population_audit_path=resolved["population_audit"],
        design_spec_path=resolved["design_spec"],
        design_report_path=resolved["design_report"],
        design_simulation_path=resolved["design_simulation"],
        simulator_source_path=resolved["simulator_source"],
        simulation_bank_path=resolved["simulation_bank"],
        verified_run_path=resolved["verified_run"],
        normalizers_path=resolved["normalizers"],
        reconstruction_path=resolved.get("reconstruction"),
        calibration_path=resolved.get("calibration"),
        calibration_normalizers_path=resolved.get("calibration_normalizers"),
        predictive_checks_path=resolved.get("predictive_checks"),
        analysis_path=resolved["analysis"],
        refusal_analysis_path=resolved.get("refusal_analysis"),
        refusal_recovery_path=resolved.get("refusal_recovery"),
        refusal_recovery_source_path=resolved.get("refusal_recovery_source"),
        root=root,
    )
    if reconstructed != release:
        raise ValueError("release manifest does not reproduce from its registered artifacts")
    return release


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    build = sub.add_parser("assemble")
    for name in ("registration", "exposure-catalog", "exclusion-ledger",
                 "population-audit", "design-spec", "design-report",
                 "design-simulation", "simulator-source", "simulation-bank",
                 "manifest", "verified-run", "normalizers", "analysis"):
        build.add_argument(f"--{name}", required=True)
    build.add_argument("--reconstruction")
    build.add_argument("--calibration")
    build.add_argument("--calibration-normalizers")
    build.add_argument("--predictive-checks")
    build.add_argument("--refusal-analysis")
    build.add_argument("--refusal-recovery")
    build.add_argument("--refusal-recovery-source")
    build.add_argument("--root", default=".")
    build.add_argument("--out", required=True)
    check = sub.add_parser("verify")
    check.add_argument("--release", required=True)
    check.add_argument("--root", default=".")
    args = ap.parse_args()
    if args.command == "verify":
        verify_release(args.release, root=args.root)
        print("release verified")
        return 0
    value = assemble(
        registration_path=args.registration, manifest_path=args.manifest,
        exposure_catalog_path=args.exposure_catalog,
        exclusion_ledger_path=args.exclusion_ledger,
        population_audit_path=args.population_audit,
        design_spec_path=args.design_spec, design_report_path=args.design_report,
        design_simulation_path=args.design_simulation,
        simulator_source_path=args.simulator_source,
        simulation_bank_path=args.simulation_bank,
        verified_run_path=args.verified_run, normalizers_path=args.normalizers,
        reconstruction_path=args.reconstruction,
        calibration_path=args.calibration,
        calibration_normalizers_path=args.calibration_normalizers,
        predictive_checks_path=args.predictive_checks, analysis_path=args.analysis,
        refusal_analysis_path=args.refusal_analysis,
        refusal_recovery_path=args.refusal_recovery,
        refusal_recovery_source_path=args.refusal_recovery_source,
        root=args.root)
    write_json_exclusive(args.out, value)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
