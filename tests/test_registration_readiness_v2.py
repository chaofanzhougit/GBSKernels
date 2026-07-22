from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.jiuzhang.build_exclusion_ledger import (
    ATTESTATION_SCOPE,
    ATTESTATION_STATEMENT,
)
from examples.jiuzhang.confirmatory_contract import sha256_json, write_canonical_json
from examples.jiuzhang.confirmatory_common import sha256_file
import examples.jiuzhang.registration_readiness_v2 as readiness
from examples.jiuzhang.registration_readiness_v2 import (
    READINESS_SCHEMA,
    ReadinessError,
    build_readiness,
)
from examples.jiuzhang.select_confirmatory_v2 import exclusion_sha256


def _ledger(*, complete: bool = False) -> dict:
    body = {
        "schema": "gbskernels.exploratory-exclusion-ledger.v1",
        "complete": complete,
        "catalog_payload_sha256": "8" * 64,
        "source": {"path": "data/raw.bin", "sha256": "a" * 64,
                   "bytes": 1600, "record_bytes": 16, "n_records": 100},
        "policy": {
            "record_level_exposure_rule": "Exclude retained records.",
            "aggregate_only_processing_rule": "Disclose aggregate scans.",
            "aggregate_scans_do_not_exclude_all_records": True,
        },
        "aggregate_exposures": [{"id": "aggregate", "artifacts": []}],
        "author_attestation": {
            "scope": ATTESTATION_SCOPE,
            "statement": ATTESTATION_STATEMENT,
            "attested": complete,
            "attestor": "Synthetic Author" if complete else "REPLACE-WITH-ATTESTOR",
            "attested_utc": ("2026-07-21T00:00:00Z" if complete
                             else "REPLACE-WITH-UTC"),
        },
        "unresolved_provenance": [{
            "id": "off_repository_access",
            "provenance_recovered": False,
            "exclusion_risk_resolved": complete,
            "resolution": ("Author review completed." if complete
                           else "REPLACE-WITH-AUTHOR-REVIEW"),
        }],
        "verification": {
            "all_evidence_verified": True,
            "one_streaming_raw_pass": True,
            "artifact_count": 2,
            "contributing_evidence_count": 1,
        },
        "evidence": [{
            "id": "synthetic", "verified": True,
            "contributes_exclusions": True,
        }],
        "overlaps": [],
        "record_indices": [1, 4, 9],
        "record_indices_count": 3,
        "exclusion_sha256": exclusion_sha256([1, 4, 9]),
    }
    return {**body, "ledger_payload_sha256": sha256_json(body)}


def _audit(ledger: dict, *, eligible: bool = False) -> dict:
    body = {
        "schema": "gbskernels.selection-population-audit.v1",
        "source_raw_sha256": ledger["source"]["sha256"],
        "n_records": ledger["source"]["n_records"],
        "record_bytes": 16,
        "exclusion_sha256": ledger["exclusion_sha256"],
        "exclusion_ledger_sha256": ledger["ledger_payload_sha256"],
        "exclusion_ledger_complete": eligible,
        "registration_eligible": eligible,
        "n_excluded": 3,
        "bands": [27, 28, 29, 30],
        "n_strata": 20,
        "eligible_by_band_stratum": {str(c): [5] * 20 for c in (27, 28, 29, 30)},
        "eligible_by_band": {str(c): 100 for c in (27, 28, 29, 30)},
        "band_weights_within_window": {str(c): 0.25 for c in (27, 28, 29, 30)},
        "band_weights_exact": {str(c): "100/400" for c in (27, 28, 29, 30)},
    }
    return {**body, "audit_payload_sha256": sha256_json(body)}


def _write(path: Path, value: dict) -> Path:
    write_canonical_json(path, value)
    return path


def test_readiness_fills_only_verified_draft_fields_and_lists_blockers(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    template = json.loads((repo / "docs" / "confirmatory_v2_plan.template.json").read_text())
    ledger = _ledger()
    audit = _audit(ledger)
    report = build_readiness(
        template,
        exclusion_ledger_path=_write(tmp_path / "ledger.json", ledger),
        population_audit_path=_write(tmp_path / "audit.json", audit),
    )
    assert report["schema"] == READINESS_SCHEMA
    assert report["ready_for_public_timestamp"] is False
    codes = {row["code"] for row in report["blockers"]}
    assert {"missing_exposure_catalog", "incomplete_exclusion_attestation",
            "draft_population_audit",
            "missing_design_spec", "missing_design_report",
            "missing_design_simulation", "missing_simulator_source",
            "missing_simulation_bank", "missing_calibration_posterior",
            "missing_refusal_recovery_source",
            "unresolved_plan_fields"}.issubset(codes)
    plan = report["candidate_plan"]
    assert plan["selection"]["source_raw_sha256"] == "a" * 64
    assert plan["selection"]["exclusion_sha256"] == ledger["exclusion_sha256"]
    assert plan["selection"]["population_audit_sha256"] == audit["audit_payload_sha256"]
    assert plan["analysis"]["band_weights"] == {str(c): 0.25 for c in (27, 28, 29, 30)}
    body = {key: value for key, value in report.items()
            if key != "readiness_payload_sha256"}
    assert report["readiness_payload_sha256"] == sha256_json(body)


def test_readiness_binds_refusal_recovery_source_bytes(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    template = json.loads((repo / "docs" / "confirmatory_v2_plan.template.json").read_text())
    source = tmp_path / "independent_recovery.py"
    source.write_text("# independent interval recovery\n")
    report = build_readiness(template, refusal_recovery_source_path=source)
    digest = sha256_file(source)
    assert report["candidate_plan"]["analysis"]["refusal_analysis"][
        "recovery_source_sha256"] == digest
    assert report["verified_inputs"]["refusal_recovery_source"]["sha256"] == digest
    assert "missing_refusal_recovery_source" not in {
        row["code"] for row in report["blockers"]}

    template["analysis"]["refusal_analysis"]["recovery_source_sha256"] = "f" * 64
    report = build_readiness(template, refusal_recovery_source_path=source)
    assert "refusal_recovery_source_mismatch" in {
        row["code"] for row in report["blockers"]}


def test_readiness_rejects_cross_artifact_hash_mismatch(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    template = json.loads((repo / "docs" / "confirmatory_v2_plan.template.json").read_text())
    ledger = _ledger()
    audit = _audit(ledger)
    audit["source_raw_sha256"] = "f" * 64
    audit_body = {key: value for key, value in audit.items()
                  if key != "audit_payload_sha256"}
    audit["audit_payload_sha256"] = sha256_json(audit_body)
    with pytest.raises(ReadinessError, match="differs"):
        build_readiness(
            template,
            exclusion_ledger_path=_write(tmp_path / "ledger.json", ledger),
            population_audit_path=_write(tmp_path / "audit.json", audit),
        )


def test_noncanonical_input_is_rejected(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    template = json.loads((repo / "docs" / "confirmatory_v2_plan.template.json").read_text())
    path = tmp_path / "pretty-ledger.json"
    path.write_text(json.dumps(_ledger(), indent=2))
    with pytest.raises(ReadinessError, match="canonical"):
        build_readiness(template, exclusion_ledger_path=path)


def test_readiness_reproduces_design_report_from_released_inputs(
        tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    template = json.loads((repo / "docs" / "confirmatory_v2_plan.template.json").read_text())
    ledger = _ledger(complete=True)
    audit = _audit(ledger, eligible=True)
    spec = {"schema": "synthetic-design-spec"}
    simulator_source = tmp_path / "simulator.py"
    simulator_source.write_text("# frozen simulator\n")
    simulation_bank = tmp_path / "simulation-bank.bin"
    simulation_bank.write_bytes(b"frozen simulation bank")
    recovery_source = tmp_path / "independent_recovery.py"
    recovery_source.write_text("# independent interval recovery\n")
    template["analysis"]["refusal_analysis"]["recovery_source_sha256"] = \
        sha256_file(recovery_source)
    spec["simulator_source_sha256"] = sha256_file(simulator_source)
    simulation = {
        "schema": "synthetic-design-simulation",
        "source": {
            "simulator_source_sha256": sha256_file(simulator_source),
            "simulation_bank_sha256": sha256_file(simulation_bank),
        },
    }
    body = {
        "schema": "gbskernels.confirmatory-design.v1",
        "pass": False,
        "selected": None,
        "analysis_contract": readiness.analysis_contract_from_plan(template),
        "population_audit_sha256": audit["audit_payload_sha256"],
        "calibration_posterior_sha256": "c" * 64,
        "criteria": {},
    }
    reproduced = {**body, "design_report_payload_sha256": sha256_json(body)}
    supplied = json.loads(json.dumps(reproduced))
    supplied["status"] = "forged"
    forged_body = {key: value for key, value in supplied.items()
                   if key != "design_report_payload_sha256"}
    supplied["design_report_payload_sha256"] = sha256_json(forged_body)
    monkeypatch.setattr(readiness, "build_design_report",
                        lambda actual_spec, actual_audit, actual_simulation: reproduced)

    with pytest.raises(ReadinessError, match="does not reproduce"):
        build_readiness(
            template,
            exclusion_ledger_path=_write(tmp_path / "ledger.json", ledger),
            population_audit_path=_write(tmp_path / "audit.json", audit),
            design_spec_path=_write(tmp_path / "spec.json", spec),
            design_report_path=_write(tmp_path / "report.json", supplied),
            design_simulation_path=_write(tmp_path / "simulation.json", simulation),
            simulator_source_path=simulator_source,
            simulation_bank_path=simulation_bank,
            refusal_recovery_source_path=recovery_source,
        )


def test_readiness_rebuilds_ledger_from_the_catalog(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    template = json.loads((repo / "docs" / "confirmatory_v2_plan.template.json").read_text())
    ledger = _ledger(complete=True)
    catalog = {"schema": "gbskernels.exploratory-exposure-catalog.v1"}
    rebuilt = json.loads(json.dumps(ledger))
    rebuilt["record_indices"] = [1]
    monkeypatch.setattr(readiness, "build_ledger", lambda *args, **kwargs: rebuilt)
    with pytest.raises(ReadinessError, match="does not reproduce"):
        build_readiness(
            template,
            exposure_catalog_path=_write(tmp_path / "catalog.json", catalog),
            exclusion_ledger_path=_write(tmp_path / "ledger.json", ledger),
        )
