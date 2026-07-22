from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from examples.jiuzhang.build_exclusion_ledger import (
    ATTESTATION_SCOPE,
    ATTESTATION_STATEMENT,
    CATALOG_SCHEMA,
    LedgerError,
    build_ledger,
    pin_catalog,
)
from examples.jiuzhang.confirmatory_contract import canonical_bytes, sha256_json
from examples.jiuzhang.select_confirmatory_v2 import (
    ABNORMAL_BIT,
    DET_POSITIONS,
    RECORD_BYTES,
    decode_records,
)


def _pattern(*clicked: int) -> np.ndarray:
    value = np.zeros(100, dtype=bool)
    value[list(clicked)] = True
    return value


def _record(pattern: np.ndarray, *, abnormal: bool = False, timestamp: int = 0) -> bytes:
    bits = np.zeros(RECORD_BYTES * 8, dtype=np.uint8)
    bits[:16] = np.unpackbits(np.asarray([timestamp >> 8, timestamp & 0xff], dtype=np.uint8))
    bits[DET_POSITIONS[::-1]] = np.asarray(pattern, dtype=np.uint8)
    bits[ABNORMAL_BIT] = int(abnormal)
    return np.packbits(bits).tobytes()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(root: Path, relative: str, **extra) -> dict:
    return {"path": relative, "sha256": _sha(root / relative), **extra}


def _fixture(root: Path) -> tuple[dict, list[np.ndarray]]:
    patterns = [
        _pattern(0),          # 0: first normal C1
        _pattern(0, 1),       # 1: first normal C2
        _pattern(2),          # 2: abnormal; never enters the first cap
        _pattern(3),          # 3: second normal C1
        _pattern(3, 4),       # 4: second normal C2
        _pattern(5),          # 5: indexed C1
        _pattern(5, 6),       # 6: indexed C2
        _pattern(7, 8, 9),    # 7: content-only pattern, occurrence 1
        _pattern(10, 11, 12), # 8: content-only unique pattern
        _pattern(7, 8, 9),    # 9: same content, occurrence 2
    ]
    records = [
        _record(pattern, abnormal=(position == 2), timestamp=position)
        for position, pattern in enumerate(patterns)
    ]
    raw_path = root / "data" / "raw.bin"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(b"".join(records))

    decoded = np.frombuffer(raw_path.read_bytes(), dtype=np.uint8).reshape(-1, RECORD_BYTES)
    _, round_trip, abnormal = decode_records(decoded)
    assert np.array_equal(round_trip, np.asarray(patterns))
    assert np.flatnonzero(abnormal).tolist() == [2]

    cap_path = root / "evidence" / "first_cap.npz"
    cap_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cap_path,
        meta=json.dumps({"kind": "test-first-cap", "cap_per_band": 2}),
        C1=np.asarray([patterns[0], patterns[3]]),
        C2=np.asarray([patterns[1], patterns[4]]),
    )
    indexed_path = root / "evidence" / "indexed.npz"
    np.savez_compressed(
        indexed_path,
        meta=json.dumps({"kind": "test-indexed"}),
        ridx_C1=np.asarray([5], dtype=np.int64),
        pats_C1=np.asarray([patterns[5]]),
        ridx_C2=np.asarray([6], dtype=np.int64),
        pats_C2=np.asarray([patterns[6]]),
    )
    content_path = root / "evidence" / "content.npy"
    np.save(content_path, np.asarray([patterns[7], patterns[8]], dtype=np.int8))

    # The non-index values are deliberately not valid JSON.  A successful
    # build proves the cross-check extracts only ridx and never parses scores.
    log_path = root / "evidence" / "indices.jsonl"
    log_path.write_bytes(
        b'{"ridx":5,"x_mid":THIS_MUST_NOT_BE_PARSED}\n'
        b'{"x_half":NOR_THIS,"ridx":6}\n'
    )
    aggregate_path = root / "tools" / "aggregate_scan.py"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text("# aggregate-only synthetic fixture\n")

    catalog = {
        "schema": CATALOG_SCHEMA,
        "policy": {
            "record_level_exposure_rule": "Exclude retained record-level evidence.",
            "aggregate_only_processing_rule": "Disclose aggregate-only scans separately.",
            "aggregate_scans_do_not_exclude_all_records": True,
        },
        "raw_source": {
            "path": "data/raw.bin",
            "sha256": _sha(raw_path),
            "record_bytes": RECORD_BYTES,
            "n_records": len(patterns),
        },
        "aggregate_exposures": [{
            "id": "aggregate_scan",
            "scope": "all synthetic records",
            "operation": "aggregate count only",
            "record_level_outputs_retained": False,
            "artifact": _artifact(root, "tools/aggregate_scan.py"),
        }],
        "evidence": [
            {
                "id": "first_caps",
                "kind": "first_cap_npz",
                "artifact": _artifact(root, "evidence/first_cap.npz"),
                "expected_metadata": {"kind": "test-first-cap", "cap_per_band": 2},
                "bands": [1, 2],
                "cap_per_band": 2,
                "pattern_key_template": "C{band}",
            },
            {
                "id": "indexed_selection",
                "kind": "indexed_selection_npz",
                "artifact": _artifact(root, "evidence/indexed.npz"),
                "expected_metadata": {"kind": "test-indexed"},
                "expected_counts": {"1": 1, "2": 1},
                "index_key_template": "ridx_C{band}",
                "pattern_key_template": "pats_C{band}",
            },
            {
                "id": "index_log",
                "kind": "jsonl_index_crosscheck",
                "reference_evidence_id": "indexed_selection",
                "record_index_field": "ridx",
                "artifact": _artifact(root, "evidence/indices.jsonl"),
            },
            {
                "id": "content_patterns",
                "kind": "pattern_arrays",
                "expected_unique_patterns": 2,
                "artifact": _artifact(root, "evidence/content.npy", expected_rows=2),
            },
        ],
        "unresolved_provenance": [{
            "id": "off_repository_access",
            "provenance_recovered": False,
            "exclusion_risk_resolved": True,
            "resolution": "The author attestation covers off-repository access.",
        }],
        "author_attestation": {
            "scope": ATTESTATION_SCOPE,
            "statement": ATTESTATION_STATEMENT,
            "attested": True,
            "attestor": "Synthetic Test Author",
            "attested_utc": "2026-07-21T00:00:00Z",
        },
    }
    return catalog, patterns


def _evidence(catalog: dict, evidence_id: str) -> dict:
    return next(item for item in catalog["evidence"] if item["id"] == evidence_id)


def test_builds_complete_deterministic_union_and_excludes_all_content_matches(tmp_path):
    catalog, _ = _fixture(tmp_path)
    first = build_ledger(tmp_path, catalog, chunk_records=1, require_complete=True)
    second = build_ledger(tmp_path, catalog, chunk_records=4, require_complete=True)

    assert first == second
    assert first["complete"] is True
    assert first["record_indices"] == [0, 1, 3, 4, 5, 6, 7, 8, 9]
    assert first["record_indices_count"] == 9
    assert first["exclusion_sha256"] == sha256_json(first["record_indices"])
    assert first["verification"]["one_streaming_raw_pass"] is True
    content = next(row for row in first["evidence"] if row["id"] == "content_patterns")
    assert content["source_rows"] == 2
    assert content["unique_patterns"] == 2
    assert content["record_indices_count"] == 3


def test_output_is_canonical_and_create_only(tmp_path):
    catalog, _ = _fixture(tmp_path)
    output = tmp_path / "ledger.json"
    ledger = build_ledger(tmp_path, catalog, chunk_records=3, output_path=output)
    assert output.read_bytes() == canonical_bytes(ledger)
    with pytest.raises(FileExistsError):
        build_ledger(tmp_path, catalog, chunk_records=3, output_path=output)


def test_pin_catalog_resolves_only_file_hashes_and_is_create_only(tmp_path):
    catalog, _ = _fixture(tmp_path)
    for row in catalog["aggregate_exposures"] + catalog["evidence"]:
        artifacts = ([row["artifact"]] if "artifact" in row else row.get("artifacts", []))
        for artifact in artifacts:
            artifact["sha256"] = "REPLACE-WITH-SHA256"
    catalog["raw_source"]["sha256"] = "REPLACE-WITH-RAW-SHA256"
    catalog["author_attestation"].update({
        "attested": False,
        "attestor": "REPLACE-WITH-ATTESTOR",
        "attested_utc": "REPLACE-WITH-UTC",
    })
    output = tmp_path / "pinned.json"
    pinned = pin_catalog(tmp_path, catalog, output_path=output)
    assert output.read_bytes() == canonical_bytes(pinned)
    assert pinned["raw_source"]["sha256"] == _sha(tmp_path / "data" / "raw.bin")
    assert pinned["author_attestation"]["attested"] is False
    assert pinned["author_attestation"]["attestor"] == "REPLACE-WITH-ATTESTOR"
    with pytest.raises(FileExistsError):
        pin_catalog(tmp_path, catalog, output_path=output)


def test_attestation_and_provenance_control_completeness(tmp_path):
    catalog, _ = _fixture(tmp_path)
    catalog["author_attestation"]["attested"] = False
    catalog["author_attestation"]["attestor"] = "REPLACE-WITH-ATTESTOR"
    catalog["author_attestation"]["attested_utc"] = "REPLACE-WITH-UTC"
    catalog["unresolved_provenance"][0]["exclusion_risk_resolved"] = False
    catalog["unresolved_provenance"][0]["resolution"] = "REPLACE-WITH-REVIEW"

    assert build_ledger(tmp_path, catalog, chunk_records=2)["complete"] is False
    with pytest.raises(LedgerError, match="attestation or provenance-risk closure"):
        build_ledger(tmp_path, catalog, chunk_records=2, require_complete=True)


def test_changed_artifact_and_jsonl_disagreement_fail_closed(tmp_path):
    catalog, _ = _fixture(tmp_path)
    aggregate = tmp_path / "tools" / "aggregate_scan.py"
    aggregate.write_text("changed\n")
    with pytest.raises(LedgerError, match="evidence artifact changed"):
        build_ledger(tmp_path, catalog)

    catalog, _ = _fixture(tmp_path)
    log = tmp_path / "evidence" / "indices.jsonl"
    log.write_bytes(b'{"ridx":5,"score":IGNORED}\n{"ridx":7,"score":IGNORED}\n')
    _evidence(catalog, "index_log")["artifact"]["sha256"] = _sha(log)
    with pytest.raises(LedgerError, match="disagrees"):
        build_ledger(tmp_path, catalog)


def test_first_cap_and_indexed_patterns_are_verified_against_raw(tmp_path):
    catalog, patterns = _fixture(tmp_path)
    cap = tmp_path / "evidence" / "first_cap.npz"
    np.savez_compressed(
        cap,
        meta=json.dumps({"kind": "test-first-cap", "cap_per_band": 2}),
        C1=np.asarray([patterns[0], _pattern(99)]),
        C2=np.asarray([patterns[1], patterns[4]]),
    )
    _evidence(catalog, "first_caps")["artifact"]["sha256"] = _sha(cap)
    with pytest.raises(LedgerError, match="first-cap pattern mismatch"):
        build_ledger(tmp_path, catalog, chunk_records=2)

    catalog, patterns = _fixture(tmp_path)
    indexed = tmp_path / "evidence" / "indexed.npz"
    np.savez_compressed(
        indexed,
        meta=json.dumps({"kind": "test-indexed"}),
        ridx_C1=np.asarray([5]),
        pats_C1=np.asarray([_pattern(98)]),
        ridx_C2=np.asarray([6]),
        pats_C2=np.asarray([patterns[6]]),
    )
    _evidence(catalog, "indexed_selection")["artifact"]["sha256"] = _sha(indexed)
    with pytest.raises(LedgerError, match="indexed evidence.*pattern mismatch"):
        build_ledger(tmp_path, catalog, chunk_records=2)


def test_unmatched_content_and_raw_hash_change_fail_closed(tmp_path):
    catalog, _ = _fixture(tmp_path)
    content = tmp_path / "evidence" / "content.npy"
    np.save(content, np.asarray([_pattern(90, 91), _pattern(92, 93)], dtype=np.int8))
    evidence = _evidence(catalog, "content_patterns")
    evidence["artifact"]["sha256"] = _sha(content)
    with pytest.raises(LedgerError, match="no exact raw-source match"):
        build_ledger(tmp_path, catalog, chunk_records=4)

    catalog, _ = _fixture(tmp_path)
    catalog["raw_source"]["sha256"] = "0" * 64
    with pytest.raises(LedgerError, match="raw source changed"):
        build_ledger(tmp_path, catalog, chunk_records=4)


def test_template_is_private_safe_and_requires_resolution():
    template_path = (Path(__file__).resolve().parents[1] / "docs" /
                     "exploratory_exposure_manifest.template.json")
    template = json.loads(template_path.read_text())
    serialized = template_path.read_text().lower()
    assert template["schema"] == CATALOG_SCHEMA
    assert template["author_attestation"]["attested"] is False
    assert "x_mid" not in serialized and "delta_h" not in serialized
    assert all(not Path(item["path"]).is_absolute()
               for evidence in template["evidence"]
               for item in ([evidence["artifact"]] if "artifact" in evidence
                            else evidence["artifacts"]))
