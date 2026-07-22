"""Build an audited Jiuzhang 1.0 exploratory-exposure exclusion ledger.

The input catalog names every evidence artifact explicitly and pins its SHA-256.
One streaming pass over the raw acquisition then:

* reconstructs and verifies first-occurrence cap arrays;
* verifies record-indexed selection manifests against the exact raw records;
* finds every raw occurrence of content-only pattern evidence; and
* hashes the raw source itself.

Aggregate-only scans are disclosed separately.  They do not make every record
an exclusion under the catalog's declared policy.  A ledger is marked complete
only when every byte-pinned artifact verifies, every provenance risk is closed,
and the author supplies the exact attestation defined below.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from confirmatory_contract import (  # noqa: E402
    canonical_bytes,
    sha256_bytes,
    sha256_json,
    write_canonical_json,
)
from select_confirmatory_v2 import (  # noqa: E402
    RECORD_BYTES,
    decode_records,
)


CATALOG_SCHEMA = "gbskernels.exploratory-exposure-catalog.v1"
LEDGER_SCHEMA = "gbskernels.exploratory-exclusion-ledger.v1"
ATTESTATION_SCOPE = "all_known_pre_registration_record_level_exposures"
ATTESTATION_STATEMENT = (
    "I attest that this catalog identifies every known Jiuzhang 1.0 raw record "
    "whose detector pattern or record-level result was deliberately retained, "
    "published, selected, inspected, or evaluated by this project before the "
    "v2 registration, including any such exposure outside this repository."
)

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_PLACEHOLDER = re.compile(r"REPLACE|PLACEHOLDER|TODO", re.IGNORECASE)


class LedgerError(ValueError):
    """Raised when an exposure ledger cannot be verified exactly."""


@dataclass(frozen=True)
class _Artifact:
    path: str
    sha256: str
    size: int
    data: bytes

    def public(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.size}


@dataclass
class _Evidence:
    evidence_id: str
    kind: str
    artifacts: list[_Artifact]
    contributes: bool
    indices: set[int] = field(default_factory=set)
    source_rows: int = 0
    unique_patterns: set[bytes] = field(default_factory=set)
    matched_patterns: set[bytes] = field(default_factory=set)
    declared_indices: set[int] = field(default_factory=set)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FirstCap:
    evidence_id: str
    cap: int
    patterns: dict[int, np.ndarray]
    consumed: dict[int, int]


@dataclass(frozen=True)
class _IndexedPattern:
    evidence_id: str
    band: int
    pattern: bytes


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LedgerError(f"{field_name} must be a JSON object")
    return value


def _integer(value: Any, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise LedgerError(f"{field_name} must be an integer")
    out = int(value)
    if out < minimum:
        raise LedgerError(f"{field_name} must be at least {minimum}")
    return out


def _integer_key(value: Any, field_name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, str) or re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise LedgerError(f"{field_name} must be a canonical non-negative integer key")
    return _integer(int(value), field_name, minimum=minimum)


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or _PLACEHOLDER.search(value):
        raise LedgerError(f"{field_name} must be a resolved non-placeholder string")
    return value


def _relative_path(root: Path, value: Any, field_name: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise LedgerError(f"{field_name} must be a repository-relative POSIX path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise LedgerError(f"{field_name} must not be absolute or contain dot segments")
    # Do not resolve the final path: large public datasets may be mounted by a
    # repository-relative symlink.  Dot-segment rejection above prevents
    # lexical traversal, and the catalog pins the exact target bytes.
    return pure.as_posix(), root.resolve().joinpath(*pure.parts)


def _load_artifact(root: Path, spec: Any, cache: dict[str, _Artifact],
                   field_name: str) -> _Artifact:
    item = _mapping(spec, field_name)
    relative, path = _relative_path(root, item.get("path"), f"{field_name}.path")
    expected = item.get("sha256")
    if not isinstance(expected, str) or _HEX64.fullmatch(expected) is None:
        raise LedgerError(f"{field_name}.sha256 must be a resolved lower-case SHA-256")
    prior = cache.get(relative)
    if prior is not None:
        if prior.sha256 != expected:
            raise LedgerError(f"conflicting hashes for artifact {relative}")
        return prior
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise LedgerError(f"cannot read evidence artifact {relative}: {exc}") from exc
    actual = sha256_bytes(data)
    if actual != expected:
        raise LedgerError(
            f"evidence artifact changed: {relative} has SHA-256 {actual}, expected {expected}"
        )
    artifact = _Artifact(relative, actual, len(data), data)
    cache[relative] = artifact
    return artifact


def _load_catalog(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        # Round-trip to detach callers from the normalized object and reject
        # values outside canonical JSON (including NaN and NumPy scalars).
        return json.loads(canonical_bytes(value))
    path = Path(value)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LedgerError(f"cannot load exposure catalog {path}: {exc}") from exc
    return dict(_mapping(loaded, "catalog"))


def _stream_sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def pin_catalog(root: str | Path, template: str | Path | Mapping[str, Any], *,
                output_path: str | Path | None = None) -> dict[str, Any]:
    """Resolve only path-bound SHA-256 placeholders in an exposure catalog.

    Author attestation and provenance-risk fields are deliberately untouched.
    The resulting catalog can therefore be verified as a draft, but it cannot
    become registration-complete without a real author review.
    """
    root_path = Path(root)
    value = _load_catalog(template)
    if value.get("schema") != CATALOG_SCHEMA:
        raise LedgerError(f"catalog.schema must be {CATALOG_SCHEMA!r}")
    artifacts = 0
    resolved = 0

    def visit(item: Any, field_name: str) -> None:
        nonlocal artifacts, resolved
        if isinstance(item, dict):
            if "path" in item and "sha256" in item:
                artifacts += 1
                relative, path = _relative_path(root_path, item["path"],
                                                f"{field_name}.path")
                try:
                    actual = _stream_sha256(path)
                except OSError as exc:
                    raise LedgerError(f"cannot hash catalog artifact {relative}: {exc}") from exc
                expected = item["sha256"]
                if isinstance(expected, str) and _PLACEHOLDER.search(expected):
                    item["sha256"] = actual
                    resolved += 1
                elif expected != actual:
                    raise LedgerError(
                        f"resolved catalog hash mismatch for {relative}: "
                        f"got {actual}, expected {expected}"
                    )
            for key, child in item.items():
                visit(child, f"{field_name}.{key}")
        elif isinstance(item, list):
            for position, child in enumerate(item):
                visit(child, f"{field_name}[{position}]")

    visit(value, "catalog")
    if artifacts == 0:
        raise LedgerError("catalog template contains no path-bound artifacts")
    if output_path is not None:
        write_canonical_json(output_path, value)
    return value


def _npz(artifact: _Artifact) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(artifact.data), allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise LedgerError(f"cannot load NPZ evidence {artifact.path}: {exc}") from exc


def _npy(artifact: _Artifact) -> np.ndarray:
    try:
        return np.array(np.load(io.BytesIO(artifact.data), allow_pickle=False), copy=True)
    except (OSError, ValueError) as exc:
        raise LedgerError(f"cannot load NPY evidence {artifact.path}: {exc}") from exc


def _metadata(archive: Mapping[str, np.ndarray], key: str, artifact: _Artifact) -> dict[str, Any]:
    if key not in archive:
        raise LedgerError(f"{artifact.path} lacks metadata key {key!r}")
    raw = archive[key]
    if raw.shape != ():
        raise LedgerError(f"{artifact.path}:{key} must be a scalar JSON string")
    value = raw.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise LedgerError(f"{artifact.path}:{key} must contain JSON text")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise LedgerError(f"invalid metadata JSON in {artifact.path}:{key}") from exc
    return dict(_mapping(parsed, f"{artifact.path}:{key}"))


def _expected_metadata(actual: Mapping[str, Any], expected: Any, field_name: str) -> None:
    if expected is None:
        return
    for key, value in _mapping(expected, field_name).items():
        if actual.get(key) != value:
            raise LedgerError(
                f"metadata mismatch for {field_name}.{key}: got {actual.get(key)!r}, "
                f"expected {value!r}"
            )


def _pattern_array(value: np.ndarray, field_name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 100:
        raise LedgerError(f"{field_name} must have shape (n,100)")
    if not (np.issubdtype(array.dtype, np.bool_) or np.issubdtype(array.dtype, np.integer)):
        raise LedgerError(f"{field_name} must be a boolean or integer array")
    if np.any((array != 0) & (array != 1)):
        raise LedgerError(f"{field_name} contains values outside {{0,1}}")
    return np.ascontiguousarray(array, dtype=bool)


def _packed_rows(patterns: np.ndarray) -> list[bytes]:
    packed = np.ascontiguousarray(np.packbits(patterns, axis=1))
    return [row.tobytes() for row in packed]


def _format_key(template: Any, band: int, field_name: str) -> str:
    if not isinstance(template, str) or "{band}" not in template:
        raise LedgerError(f"{field_name} must contain '{{band}}'")
    try:
        return template.format(band=band)
    except (KeyError, ValueError) as exc:
        raise LedgerError(f"invalid key template in {field_name}") from exc


def _jsonl_record_indices(data: bytes, field_name: str, artifact: _Artifact) -> list[int]:
    """Extract one integer field per JSONL row without parsing any other value.

    Historical rows also contain likelihood scores.  This deliberately narrow
    lexical check neither deserializes nor exposes those score fields; the whole
    file is nevertheless integrity-bound by the artifact SHA-256.
    """
    if not isinstance(field_name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field_name):
        raise LedgerError("JSONL record_index_field is invalid")
    key = re.escape(field_name.encode("ascii"))
    pattern = re.compile(
        rb'(?<!\\)"' + key + rb'"\s*:\s*(-?(?:0|[1-9][0-9]*))(?=\s*[,}])'
    )
    result: list[int] = []
    for line_number, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            continue
        matches = pattern.findall(line)
        if len(matches) != 1:
            raise LedgerError(
                f"{artifact.path}:{line_number} must contain exactly one integer "
                f"{field_name!r} field"
            )
        result.append(int(matches[0]))
    return result


def _attestation(value: Any) -> tuple[dict[str, Any], bool]:
    item = _mapping(value, "author_attestation")
    attested = item.get("attested")
    if not isinstance(attested, bool):
        raise LedgerError("author_attestation.attested must be boolean")
    normalized = {
        "scope": item.get("scope"),
        "statement": item.get("statement"),
        "attested": attested,
        "attestor": item.get("attestor"),
        "attested_utc": item.get("attested_utc"),
    }
    if normalized["scope"] != ATTESTATION_SCOPE:
        raise LedgerError(f"author_attestation.scope must be {ATTESTATION_SCOPE!r}")
    if normalized["statement"] != ATTESTATION_STATEMENT:
        raise LedgerError("author_attestation.statement does not match the required attestation")
    if attested:
        _identifier(normalized["attestor"], "author_attestation.attestor")
        timestamp = normalized["attested_utc"]
        if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
            raise LedgerError("author_attestation.attested_utc must be an ISO-8601 UTC timestamp")
        try:
            datetime.fromisoformat(timestamp[:-1] + "+00:00")
        except ValueError as exc:
            raise LedgerError("author_attestation.attested_utc is invalid") from exc
    return normalized, attested


def _provenance_gaps(value: Any) -> tuple[list[dict[str, Any]], bool]:
    if not isinstance(value, list) or not value:
        raise LedgerError("unresolved_provenance must be a non-empty list")
    normalized = []
    ids: set[str] = set()
    all_risks_resolved = True
    for position, raw in enumerate(value):
        item = _mapping(raw, f"unresolved_provenance[{position}]")
        gap_id = _identifier(item.get("id"), f"unresolved_provenance[{position}].id")
        if gap_id in ids:
            raise LedgerError(f"duplicate provenance gap id {gap_id!r}")
        ids.add(gap_id)
        recovered = item.get("provenance_recovered")
        resolved = item.get("exclusion_risk_resolved")
        if not isinstance(recovered, bool) or not isinstance(resolved, bool):
            raise LedgerError("provenance gap status fields must be boolean")
        resolution = item.get("resolution")
        if not isinstance(resolution, str) or not resolution:
            raise LedgerError(f"provenance gap {gap_id!r} needs a resolution statement")
        if resolved and _PLACEHOLDER.search(resolution):
            raise LedgerError(f"resolved provenance gap {gap_id!r} has a placeholder resolution")
        normalized.append({
            "id": gap_id,
            "provenance_recovered": recovered,
            "exclusion_risk_resolved": resolved,
            "resolution": resolution,
        })
        all_risks_resolved &= resolved
    return sorted(normalized, key=lambda row: row["id"]), all_risks_resolved


def _policy(value: Any) -> dict[str, Any]:
    item = _mapping(value, "policy")
    record_rule = item.get("record_level_exposure_rule")
    aggregate_rule = item.get("aggregate_only_processing_rule")
    if not isinstance(record_rule, str) or not record_rule:
        raise LedgerError("policy.record_level_exposure_rule is required")
    if not isinstance(aggregate_rule, str) or not aggregate_rule:
        raise LedgerError("policy.aggregate_only_processing_rule is required")
    if item.get("aggregate_scans_do_not_exclude_all_records") is not True:
        raise LedgerError(
            "policy must explicitly affirm aggregate_scans_do_not_exclude_all_records"
        )
    return {
        "record_level_exposure_rule": record_rule,
        "aggregate_only_processing_rule": aggregate_rule,
        "aggregate_scans_do_not_exclude_all_records": True,
    }


def _artifact_specs(item: Mapping[str, Any], singular: str, plural: str,
                    field_name: str) -> list[Any]:
    if singular in item:
        if plural in item:
            raise LedgerError(f"{field_name} cannot contain both {singular} and {plural}")
        return [item[singular]]
    values = item.get(plural)
    if not isinstance(values, list) or not values:
        raise LedgerError(f"{field_name}.{plural} must be a non-empty list")
    return values


def build_ledger(
    root: str | Path,
    catalog: str | Path | Mapping[str, Any],
    *,
    chunk_records: int = 2_000_000,
    output_path: str | Path | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Verify a pinned evidence catalog and build its sorted exclusion union."""
    if chunk_records <= 0:
        raise LedgerError("chunk_records must be positive")
    root_path = Path(root)
    catalog_value = _load_catalog(catalog)
    if catalog_value.get("schema") != CATALOG_SCHEMA:
        raise LedgerError(f"catalog.schema must be {CATALOG_SCHEMA!r}")
    policy = _policy(catalog_value.get("policy"))
    attestation, attested = _attestation(catalog_value.get("author_attestation"))
    gaps, all_risks_resolved = _provenance_gaps(
        catalog_value.get("unresolved_provenance"))

    cache: dict[str, _Artifact] = {}
    aggregate_rows: list[dict[str, Any]] = []
    aggregate = catalog_value.get("aggregate_exposures")
    if not isinstance(aggregate, list) or not aggregate:
        raise LedgerError("aggregate_exposures must be a non-empty list")
    aggregate_ids: set[str] = set()
    for position, raw in enumerate(aggregate):
        item = _mapping(raw, f"aggregate_exposures[{position}]")
        exposure_id = _identifier(item.get("id"), f"aggregate_exposures[{position}].id")
        if exposure_id in aggregate_ids:
            raise LedgerError(f"duplicate aggregate exposure id {exposure_id!r}")
        aggregate_ids.add(exposure_id)
        if item.get("record_level_outputs_retained") is not False:
            raise LedgerError(
                f"aggregate exposure {exposure_id!r} must explicitly state that no "
                "record-level outputs were retained"
            )
        scope = item.get("scope")
        operation = item.get("operation")
        if (not isinstance(scope, str) or not scope
                or not isinstance(operation, str) or not operation):
            raise LedgerError(f"aggregate exposure {exposure_id!r} needs scope and operation")
        specs = _artifact_specs(item, "artifact", "artifacts",
                                f"aggregate_exposures[{position}]")
        artifacts = [
            _load_artifact(root_path, spec, cache,
                           f"aggregate_exposures[{position}].artifacts[{i}]")
            for i, spec in enumerate(specs)
        ]
        aggregate_rows.append({
            "id": exposure_id,
            "scope": scope,
            "operation": operation,
            "record_level_outputs_retained": False,
            "artifacts": [artifact.public() for artifact in artifacts],
        })

    evidence_raw = catalog_value.get("evidence")
    if not isinstance(evidence_raw, list) or not evidence_raw:
        raise LedgerError("evidence must be a non-empty list")
    evidence: dict[str, _Evidence] = {}
    first_caps: list[_FirstCap] = []
    indexed: dict[int, list[_IndexedPattern]] = {}
    target_members: dict[bytes, set[str]] = {}
    deferred_crosschecks: list[tuple[_Evidence, str, list[int]]] = []

    for position, raw in enumerate(evidence_raw):
        item = _mapping(raw, f"evidence[{position}]")
        evidence_id = _identifier(item.get("id"), f"evidence[{position}].id")
        if evidence_id in evidence:
            raise LedgerError(f"duplicate evidence id {evidence_id!r}")
        kind = item.get("kind")
        if kind not in {"first_cap_npz", "indexed_selection_npz",
                        "pattern_arrays", "jsonl_index_crosscheck"}:
            raise LedgerError(f"unsupported evidence kind {kind!r}")
        specs = _artifact_specs(item, "artifact", "artifacts", f"evidence[{position}]")
        artifacts = [
            _load_artifact(root_path, spec, cache, f"evidence[{position}].artifacts[{i}]")
            for i, spec in enumerate(specs)
        ]
        state = _Evidence(
            evidence_id=evidence_id,
            kind=str(kind),
            artifacts=artifacts,
            contributes=kind != "jsonl_index_crosscheck",
        )
        evidence[evidence_id] = state

        if kind == "first_cap_npz":
            if len(artifacts) != 1:
                raise LedgerError("first_cap_npz requires exactly one artifact")
            archive = _npz(artifacts[0])
            metadata_key = str(item.get("metadata_key", "meta"))
            metadata = _metadata(archive, metadata_key, artifacts[0])
            _expected_metadata(metadata, item.get("expected_metadata"),
                               f"evidence[{position}].expected_metadata")
            bands_raw = item.get("bands")
            if not isinstance(bands_raw, list) or not bands_raw:
                raise LedgerError("first_cap_npz.bands must be a non-empty list")
            bands = [_integer(value, "first_cap_npz band") for value in bands_raw]
            if len(bands) != len(set(bands)):
                raise LedgerError("first_cap_npz bands contain duplicates")
            cap = _integer(item.get("cap_per_band"), "first_cap_npz.cap_per_band", minimum=1)
            key_template = item.get("pattern_key_template", "C{band}")
            patterns_by_band: dict[int, np.ndarray] = {}
            for band in bands:
                key = _format_key(key_template, band, "first_cap_npz.pattern_key_template")
                if key not in archive:
                    raise LedgerError(f"{artifacts[0].path} lacks array {key!r}")
                patterns = _pattern_array(archive[key], f"{artifacts[0].path}:{key}")
                if len(patterns) != cap:
                    raise LedgerError(
                        f"{artifacts[0].path}:{key} has {len(patterns)} rows, expected cap {cap}"
                    )
                patterns_by_band[band] = patterns
                state.unique_patterns.update(_packed_rows(patterns))
            state.source_rows = cap * len(bands)
            state.details = {"bands": sorted(bands), "cap_per_band": cap}
            first_caps.append(_FirstCap(
                evidence_id=evidence_id,
                cap=cap,
                patterns=patterns_by_band,
                consumed={band: 0 for band in bands},
            ))

        elif kind == "indexed_selection_npz":
            if len(artifacts) != 1:
                raise LedgerError("indexed_selection_npz requires exactly one artifact")
            archive = _npz(artifacts[0])
            metadata_key = str(item.get("metadata_key", "meta"))
            metadata = _metadata(archive, metadata_key, artifacts[0])
            _expected_metadata(metadata, item.get("expected_metadata"),
                               f"evidence[{position}].expected_metadata")
            counts = _mapping(item.get("expected_counts"),
                              f"evidence[{position}].expected_counts")
            bands = sorted(_integer_key(key, "indexed_selection_npz band") for key in counts)
            index_template = item.get("index_key_template", "ridx_C{band}")
            pattern_template = item.get("pattern_key_template", "pats_C{band}")
            for band in bands:
                expected_count = _integer(counts[str(band)],
                                          f"indexed_selection_npz count C{band}")
                index_key = _format_key(index_template, band,
                                        "indexed_selection_npz.index_key_template")
                pattern_key = _format_key(pattern_template, band,
                                          "indexed_selection_npz.pattern_key_template")
                if index_key not in archive or pattern_key not in archive:
                    raise LedgerError(
                        f"{artifacts[0].path} lacks {index_key!r} or {pattern_key!r}"
                    )
                indices = np.asarray(archive[index_key])
                if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                    raise LedgerError(f"{artifacts[0].path}:{index_key} must be an integer vector")
                patterns = _pattern_array(archive[pattern_key],
                                          f"{artifacts[0].path}:{pattern_key}")
                if len(indices) != expected_count or len(patterns) != expected_count:
                    raise LedgerError(
                        f"indexed evidence C{band} count mismatch: expected {expected_count}"
                    )
                packed = _packed_rows(patterns)
                for record_index, pattern in zip(indices.tolist(), packed):
                    record_index = int(record_index)
                    if record_index < 0:
                        raise LedgerError("indexed evidence contains a negative record index")
                    if record_index in state.declared_indices:
                        raise LedgerError(
                            f"indexed evidence {evidence_id!r} contains duplicate record index "
                            f"{record_index}"
                        )
                    state.declared_indices.add(record_index)
                    state.unique_patterns.add(pattern)
                    indexed.setdefault(record_index, []).append(
                        _IndexedPattern(evidence_id, band, pattern))
            state.source_rows = len(state.declared_indices)
            state.details = {"bands": bands, "expected_counts": {
                str(band): int(counts[str(band)])
                for band in bands
            }}

        elif kind == "pattern_arrays":
            for artifact_position, (artifact, artifact_spec) in enumerate(zip(artifacts, specs)):
                patterns = _pattern_array(_npy(artifact), artifact.path)
                artifact_field = f"evidence[{position}].artifacts[{artifact_position}]"
                expected_rows = _mapping(artifact_spec, artifact_field).get("expected_rows")
                if expected_rows is not None and len(patterns) != _integer(
                        expected_rows, f"{artifact.path}.expected_rows"):
                    raise LedgerError(
                        f"{artifact.path} has {len(patterns)} rows, expected {expected_rows}"
                    )
                rows = _packed_rows(patterns)
                state.source_rows += len(rows)
                state.unique_patterns.update(rows)
                for pattern in rows:
                    target_members.setdefault(pattern, set()).add(evidence_id)
            expected_unique = item.get("expected_unique_patterns")
            if expected_unique is not None and len(state.unique_patterns) != _integer(
                    expected_unique, f"evidence[{position}].expected_unique_patterns"):
                raise LedgerError(
                    f"pattern evidence {evidence_id!r} has {len(state.unique_patterns)} unique "
                    f"patterns, expected {expected_unique}"
                )
            state.details = {"pattern_rows": state.source_rows,
                             "unique_patterns": len(state.unique_patterns)}

        else:
            field_name = item.get("record_index_field", "ridx")
            rows: list[int] = []
            for artifact in artifacts:
                rows.extend(_jsonl_record_indices(artifact.data, str(field_name), artifact))
            if len(rows) != len(set(rows)):
                raise LedgerError(f"JSONL cross-check {evidence_id!r} contains duplicate indices")
            state.declared_indices = set(rows)
            state.source_rows = len(rows)
            reference = item.get("reference_evidence_id")
            if not isinstance(reference, str) or not reference:
                raise LedgerError("jsonl_index_crosscheck.reference_evidence_id is required")
            state.details = {"reference_evidence_id": reference,
                             "record_index_field": str(field_name)}
            deferred_crosschecks.append((state, reference, rows))

    for state, reference, _ in deferred_crosschecks:
        other = evidence.get(reference)
        if other is None or other.kind != "indexed_selection_npz":
            raise LedgerError(
                f"JSONL cross-check {state.evidence_id!r} references no indexed selection "
                f"evidence named {reference!r}"
            )
        if state.declared_indices != other.declared_indices:
            missing = len(other.declared_indices - state.declared_indices)
            extra = len(state.declared_indices - other.declared_indices)
            raise LedgerError(
                f"JSONL index cross-check {state.evidence_id!r} disagrees with {reference!r}: "
                f"{missing} missing, {extra} extra"
            )

    raw_spec = _mapping(catalog_value.get("raw_source"), "raw_source")
    raw_relative, raw_path = _relative_path(root_path, raw_spec.get("path"), "raw_source.path")
    raw_expected_hash = raw_spec.get("sha256")
    if not isinstance(raw_expected_hash, str) or _HEX64.fullmatch(raw_expected_hash) is None:
        raise LedgerError("raw_source.sha256 must be a resolved lower-case SHA-256")
    record_bytes = _integer(raw_spec.get("record_bytes"), "raw_source.record_bytes", minimum=1)
    if record_bytes != RECORD_BYTES:
        raise LedgerError(f"raw_source.record_bytes must be {RECORD_BYTES}")
    expected_n_records = _integer(raw_spec.get("n_records"), "raw_source.n_records", minimum=1)
    try:
        before = raw_path.stat()
    except OSError as exc:
        raise LedgerError(f"cannot stat raw source {raw_relative}: {exc}") from exc
    if before.st_size % RECORD_BYTES:
        raise LedgerError("raw source ends with a partial record")
    n_records = before.st_size // RECORD_BYTES
    if n_records != expected_n_records:
        raise LedgerError(
            f"raw source record count changed: got {n_records}, expected {expected_n_records}"
        )
    if indexed and max(indexed) >= n_records:
        raise LedgerError("indexed evidence contains an index outside the raw source")

    target_keys = sorted(target_members)
    target_void = (np.frombuffer(b"".join(target_keys), dtype="V13")
                   if target_keys else np.empty(0, dtype="V13"))
    direct_indices = np.asarray(sorted(indexed), dtype=np.int64)
    direct_verified: set[int] = set()
    raw_hash = hashlib.sha256()
    base = 0
    try:
        with raw_path.open("rb") as fh:
            while True:
                block = fh.read(chunk_records * RECORD_BYTES)
                if not block:
                    break
                if len(block) % RECORD_BYTES:
                    raise LedgerError("raw source changed to contain a partial record")
                raw_hash.update(block)
                raw = np.frombuffer(block, dtype=np.uint8).reshape(-1, RECORD_BYTES)
                _, patterns, abnormal = decode_records(raw)
                clicks = patterns.sum(axis=1)
                packed_array = np.ascontiguousarray(np.packbits(patterns, axis=1))
                packed_void = packed_array.view("V13").reshape(-1)

                for config in first_caps:
                    state = evidence[config.evidence_id]
                    for band, expected_patterns in config.patterns.items():
                        already = config.consumed[band]
                        if already >= config.cap:
                            continue
                        local = np.flatnonzero((clicks == band) & (~abnormal))
                        take = min(config.cap - already, len(local))
                        if take == 0:
                            continue
                        actual = patterns[local[:take]]
                        expected = expected_patterns[already:already + take]
                        unequal = np.flatnonzero(np.any(actual != expected, axis=1))
                        if len(unequal):
                            offset = int(unequal[0])
                            rank = already + offset
                            record_index = base + int(local[offset])
                            raise LedgerError(
                                f"first-cap pattern mismatch in {config.evidence_id!r}, "
                                f"band {band}, rank {rank}, raw record {record_index}"
                            )
                        selected = base + local[:take]
                        state.indices.update(int(value) for value in selected)
                        state.matched_patterns.update(_packed_rows(actual))
                        config.consumed[band] += take

                if len(direct_indices):
                    start = int(np.searchsorted(direct_indices, base, side="left"))
                    stop = int(np.searchsorted(direct_indices, base + len(raw), side="left"))
                    for record_index in direct_indices[start:stop].tolist():
                        local = int(record_index - base)
                        actual_pattern = packed_void[local].tobytes()
                        for expected in indexed[int(record_index)]:
                            if abnormal[local]:
                                raise LedgerError(
                                    f"indexed evidence {expected.evidence_id!r} points to abnormal "
                                    f"record {record_index}"
                                )
                            if int(clicks[local]) != expected.band:
                                raise LedgerError(
                                    f"indexed evidence {expected.evidence_id!r} band mismatch at "
                                    f"record {record_index}"
                                )
                            if actual_pattern != expected.pattern:
                                raise LedgerError(
                                    f"indexed evidence {expected.evidence_id!r} pattern "
                                    f"mismatch at record {record_index}"
                                )
                            state = evidence[expected.evidence_id]
                            state.indices.add(int(record_index))
                            state.matched_patterns.add(actual_pattern)
                        direct_verified.add(int(record_index))

                if len(target_void):
                    local_matches = np.flatnonzero(np.isin(packed_void, target_void))
                    for local in local_matches.tolist():
                        pattern = packed_void[local].tobytes()
                        record_index = base + int(local)
                        for evidence_id in target_members[pattern]:
                            state = evidence[evidence_id]
                            state.indices.add(record_index)
                            state.matched_patterns.add(pattern)
                base += len(raw)
    except OSError as exc:
        raise LedgerError(f"cannot scan raw source {raw_relative}: {exc}") from exc

    try:
        after = raw_path.stat()
    except OSError as exc:
        raise LedgerError(f"cannot restat raw source {raw_relative}: {exc}") from exc
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise LedgerError("raw source changed while it was being scanned")
    if base != n_records:
        raise LedgerError(f"raw scan saw {base} records, expected {n_records}")
    actual_raw_hash = raw_hash.hexdigest()
    if actual_raw_hash != raw_expected_hash:
        raise LedgerError(
            f"raw source changed: {raw_relative} has SHA-256 {actual_raw_hash}, "
            f"expected {raw_expected_hash}"
        )
    if direct_verified != set(indexed):
        raise LedgerError("not every indexed evidence record was verified in the raw source")
    for config in first_caps:
        for band, count in config.consumed.items():
            if count != config.cap:
                raise LedgerError(
                    f"raw source has only {count} usable first-cap rows for band {band}; "
                    f"expected {config.cap}"
                )
    for state in evidence.values():
        if state.kind == "pattern_arrays":
            unmatched = state.unique_patterns - state.matched_patterns
            if unmatched:
                raise LedgerError(
                    f"pattern evidence {state.evidence_id!r} has {len(unmatched)} patterns "
                    "with no exact raw-source match"
                )
        if state.contributes and not state.indices:
            raise LedgerError(f"evidence {state.evidence_id!r} contributed no exclusions")

    evidence_rows: list[dict[str, Any]] = []
    contributing_ids = sorted(key for key, state in evidence.items() if state.contributes)
    for evidence_id in sorted(evidence):
        state = evidence[evidence_id]
        row = {
            "id": evidence_id,
            "kind": state.kind,
            "verified": True,
            "contributes_exclusions": state.contributes,
            "artifacts": [artifact.public() for artifact in state.artifacts],
            "source_rows": state.source_rows,
            "record_indices_count": len(state.indices if state.contributes
                                        else state.declared_indices),
            "record_indices_sha256": sha256_json(sorted(
                state.indices if state.contributes else state.declared_indices)),
            "details": state.details,
        }
        if state.unique_patterns:
            row["unique_patterns"] = len(state.unique_patterns)
            row["matched_unique_patterns"] = len(state.matched_patterns)
        evidence_rows.append(row)

    overlaps = []
    for left_position, left in enumerate(contributing_ids):
        for right in contributing_ids[left_position + 1:]:
            count = len(evidence[left].indices & evidence[right].indices)
            if count:
                overlaps.append({"evidence_ids": [left, right], "record_indices_count": count})
    union = sorted(set().union(*(evidence[key].indices for key in contributing_ids)))
    complete = bool(attested and all_risks_resolved)
    body: dict[str, Any] = {
        "schema": LEDGER_SCHEMA,
        "complete": complete,
        "catalog_payload_sha256": sha256_json(catalog_value),
        "source": {
            "path": raw_relative,
            "sha256": actual_raw_hash,
            "bytes": before.st_size,
            "record_bytes": RECORD_BYTES,
            "n_records": n_records,
        },
        "policy": policy,
        "aggregate_exposures": sorted(aggregate_rows, key=lambda row: row["id"]),
        "author_attestation": attestation,
        "unresolved_provenance": gaps,
        "verification": {
            "all_evidence_verified": True,
            "one_streaming_raw_pass": True,
            "artifact_count": len(cache) + 1,
            "contributing_evidence_count": len(contributing_ids),
        },
        "evidence": evidence_rows,
        "overlaps": overlaps,
        "record_indices": union,
        "record_indices_count": len(union),
        "exclusion_sha256": sha256_json(union),
    }
    body["ledger_payload_sha256"] = sha256_json(body)
    if require_complete and not complete:
        raise LedgerError(
            "ledger evidence verified, but author attestation or "
            "provenance-risk closure is incomplete"
        )
    if output_path is not None:
        write_canonical_json(output_path, body)
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pin = commands.add_parser("pin-catalog")
    pin.add_argument("--root", type=Path, default=REPO)
    pin.add_argument("--template", type=Path, required=True)
    pin.add_argument("--out", type=Path, required=True)
    build = commands.add_parser("build")
    build.add_argument("--root", type=Path, default=REPO)
    build.add_argument("--catalog", type=Path, required=True)
    build.add_argument("--chunk-records", type=int, default=2_000_000)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.command == "pin-catalog":
        value = pin_catalog(args.root, args.template, output_path=args.out)
        print(f"{args.out}  pinned_sha256_paths=true  complete=false")
        return 0
    ledger = build_ledger(
        args.root,
        args.catalog,
        chunk_records=args.chunk_records,
        output_path=args.out,
        require_complete=args.require_complete,
    )
    print(f"{args.out}  exclusions={ledger['record_indices_count']}  "
          f"complete={ledger['complete']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
