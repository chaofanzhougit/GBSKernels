"""Public-registration-driven, design-based Jiuzhang confirmatory selection.

Unlike the historical selector, this version uses common acquisition strata,
actual explicit record-index exclusions, proportional largest-remainder
allocation, and ranked reserves.  It makes two streaming passes over ``data.bin``:
the first fixes eligible population counts and the raw-source hash; the second
keeps only the smallest beacon-derived hash keys required by the manifest.

The command-line path requires a resolved public registration.  The library
function also accepts an explicit seed for unit tests and offline dry runs; such
manifests are clearly marked unregistered.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

import sys

sys.path.insert(0, str(HERE))
from confirmatory_contract import (  # noqa: E402
    ContractError,
    event_key,
    load_registration,
    sha256_bytes,
    sha256_json,
    validate_registration,
    write_canonical_json,
)


RECORD_BYTES = 16
DET_SLOTS = np.arange(16, 120)
IGNORED = np.array([22, 24, 25, 43])
DET_POSITIONS = np.setdiff1d(DET_SLOTS, IGNORED)
ABNORMAL_BIT = 127
DEFAULT_BANDS = (27, 28, 29, 30)
INPUT_HASH_DOMAIN = b"GBSKERNELS/CONFIRMATORY/V2/PATTERN\0"
ATTESTATION_SCOPE = "all_known_pre_registration_record_level_exposures"
ATTESTATION_STATEMENT = (
    "I attest that this catalog identifies every known Jiuzhang 1.0 raw record "
    "whose detector pattern or record-level result was deliberately retained, "
    "published, selected, inspected, or evaluated by this project before the "
    "v2 registration, including any such exposure outside this repository."
)


class SelectionError(ValueError):
    """Raised when selection cannot satisfy the registered design exactly."""


@dataclass(frozen=True)
class _Source:
    path: Path | None
    data: bytes | None
    n_records: int

    def chunks(self, chunk_records: int) -> Iterator[tuple[int, np.ndarray]]:
        if chunk_records <= 0:
            raise SelectionError("chunk_records must be positive")
        block_bytes = chunk_records * RECORD_BYTES
        offset = 0
        if self.path is not None:
            with self.path.open("rb") as fh:
                while True:
                    block = fh.read(block_bytes)
                    if not block:
                        break
                    if len(block) % RECORD_BYTES:
                        raise SelectionError("raw source ends with a partial record")
                    arr = np.frombuffer(block, dtype=np.uint8).reshape(-1, RECORD_BYTES)
                    yield offset, arr
                    offset += len(arr)
        else:
            assert self.data is not None
            raw = np.frombuffer(self.data, dtype=np.uint8).reshape(-1, RECORD_BYTES)
            for start in range(0, len(raw), chunk_records):
                arr = raw[start : start + chunk_records]
                yield offset, arr
                offset += len(arr)
        if offset != self.n_records:
            raise SelectionError(
                f"raw source changed while scanning: expected {self.n_records}, saw {offset} records"
            )


@dataclass(frozen=True)
class _Candidate:
    band: int
    record_index: int
    stratum: int
    timestamp: int
    abnormal: bool
    pattern: tuple[int, ...]
    raw: bytes
    key: str


def _coerce_source(source: str | Path | bytes | bytearray | memoryview | np.ndarray | Iterable[bytes]) -> _Source:
    if isinstance(source, (str, Path)):
        path = Path(source)
        size = path.stat().st_size
        if size % RECORD_BYTES:
            raise SelectionError(f"{path} size is not a multiple of {RECORD_BYTES}")
        return _Source(path=path, data=None, n_records=size // RECORD_BYTES)
    if isinstance(source, np.ndarray):
        if source.dtype != np.uint8:
            raise SelectionError("numpy raw source must have dtype uint8")
        data = np.ascontiguousarray(source).tobytes()
    elif isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
    else:
        try:
            data = b"".join(bytes(record) for record in source)
        except (TypeError, ValueError) as exc:
            raise SelectionError("raw source must be a path, bytes, uint8 array, or records") from exc
    if len(data) % RECORD_BYTES:
        raise SelectionError("raw source ends with a partial record")
    return _Source(path=None, data=data, n_records=len(data) // RECORD_BYTES)


def decode_records(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode raw ``(n,16)`` bytes into timestamp, detector pattern, abnormal."""
    arr = np.asarray(raw, dtype=np.uint8)
    if arr.ndim != 2 or arr.shape[1] != RECORD_BYTES:
        raise SelectionError(f"raw records must have shape (n,{RECORD_BYTES})")
    bits = np.unpackbits(arr, axis=1)
    timestamp = (arr[:, 0].astype(np.uint16) << 8) | arr[:, 1].astype(np.uint16)
    pattern = bits[:, DET_POSITIONS][:, ::-1].astype(bool)
    abnormal = bits[:, ABNORMAL_BIT].astype(bool)
    return timestamp, pattern, abnormal


def common_stratum(record_index: int | np.ndarray, n_records: int, n_strata: int):
    """Common fixed stratum ``floor(index * H / n_records)`` on ``[0,N)``."""
    if n_records <= 0 or n_strata <= 0:
        raise SelectionError("n_records and n_strata must be positive")
    idx = np.asarray(record_index)
    if np.any(idx < 0) or np.any(idx >= n_records):
        raise SelectionError("record index outside [0,n_records)")
    out = np.minimum((idx.astype(np.int64) * n_strata) // n_records, n_strata - 1)
    if np.ndim(record_index) == 0:
        return int(out)
    return out


def stratum_edges(n_records: int, n_strata: int) -> list[int]:
    """Integer boundaries shared by every click band."""
    if n_records <= 0 or n_strata <= 0:
        raise SelectionError("n_records and n_strata must be positive")
    return [(h * n_records + n_strata - 1) // n_strata for h in range(n_strata)] + [n_records]


def largest_remainder_allocation(counts: Sequence[int], total: int) -> list[int]:
    """Allocate ``total`` proportionally using exact integer largest remainder.

    Ties are broken by the lowest stratum index.  No floating-point arithmetic
    enters the allocation.
    """
    values = [int(x) for x in counts]
    if any(x < 0 for x in values):
        raise SelectionError("eligible counts must be non-negative")
    if isinstance(total, bool) or not isinstance(total, (int, np.integer)) or int(total) < 0:
        raise SelectionError("allocation total must be a non-negative integer")
    total = int(total)
    population = sum(values)
    if total > population:
        raise SelectionError(f"requested {total} records from an eligible population of {population}")
    if total == 0:
        return [0] * len(values)
    if population == 0:
        raise SelectionError("cannot allocate from an empty population")
    numerators = [total * x for x in values]
    allocation = [num // population for num in numerators]
    left = total - sum(allocation)
    order = sorted(range(len(values)), key=lambda h: (-(numerators[h] % population), h))
    for h in order[:left]:
        allocation[h] += 1
    if any(n > N for n, N in zip(allocation, values, strict=True)):
        raise SelectionError("largest-remainder allocation exceeded a stratum population")
    return allocation


def parse_count_spec(spec: str | Mapping[int | str, int]) -> dict[int, int]:
    """Parse ``27:800,28:500`` or normalize a JSON mapping."""
    if isinstance(spec, Mapping):
        out = {int(k): int(v) for k, v in spec.items()}
    else:
        out = {}
        if not spec:
            return out
        for part in str(spec).split(","):
            try:
                band, count = part.split(":", 1)
                out[int(band)] = int(count)
            except ValueError as exc:
                raise SelectionError(f"invalid count specification: {part!r}") from exc
    if any(C < 0 or n < 0 for C, n in out.items()):
        raise SelectionError("bands and counts must be non-negative")
    return dict(sorted(out.items()))


def _validated_ledger(value: Mapping[str, Any], *, require_complete: bool) -> tuple[list[int], str]:
    """Validate a canonical exploratory-exclusion ledger and its self-hashes."""
    if value.get("schema") != "gbskernels.exploratory-exclusion-ledger.v1":
        raise SelectionError("unsupported exclusion-ledger schema")
    if not isinstance(value.get("complete"), bool):
        raise SelectionError("exclusion ledger completeness must be boolean")
    rows = value.get("record_indices")
    if not isinstance(rows, list) or any(
            isinstance(index, bool) or not isinstance(index, int) for index in rows):
        raise SelectionError("exclusion ledger record_indices must be an integer list")
    indices = [int(index) for index in rows]
    if indices != sorted(set(indices)):
        raise SelectionError("exclusion ledger record_indices must be sorted and unique")
    if value.get("record_indices_count") != len(indices):
        raise SelectionError("exclusion ledger record count is inconsistent")
    if value.get("exclusion_sha256") != exclusion_sha256(indices):
        raise SelectionError("exclusion ledger record-index hash is invalid")
    payload_hash = value.get("ledger_payload_sha256")
    if not isinstance(payload_hash, str) or len(payload_hash) != 64:
        raise SelectionError("exclusion ledger payload hash is invalid")
    body = {key: item for key, item in value.items() if key != "ledger_payload_sha256"}
    if sha256_json(body) != payload_hash:
        raise SelectionError("exclusion ledger payload hash does not match its content")
    source = value.get("source")
    if (not isinstance(source, Mapping)
            or not isinstance(source.get("path"), str) or not source["path"]
            or not isinstance(source.get("sha256"), str) or len(source["sha256"]) != 64
            or isinstance(source.get("bytes"), bool) or not isinstance(source.get("bytes"), int)
            or isinstance(source.get("record_bytes"), bool)
            or not isinstance(source.get("record_bytes"), int)
            or isinstance(source.get("n_records"), bool)
            or not isinstance(source.get("n_records"), int)
            or source["record_bytes"] <= 0 or source["n_records"] <= 0
            or source["bytes"] != source["record_bytes"] * source["n_records"]):
        raise SelectionError("exclusion ledger source metadata is invalid")
    catalog_hash = value.get("catalog_payload_sha256")
    if not isinstance(catalog_hash, str) or len(catalog_hash) != 64:
        raise SelectionError("exclusion ledger lacks its catalog payload hash")
    policy = value.get("policy")
    if (not isinstance(policy, Mapping)
            or not isinstance(policy.get("record_level_exposure_rule"), str)
            or not policy["record_level_exposure_rule"]
            or not isinstance(policy.get("aggregate_only_processing_rule"), str)
            or not policy["aggregate_only_processing_rule"]
            or policy.get("aggregate_scans_do_not_exclude_all_records") is not True):
        raise SelectionError("exclusion ledger policy is incomplete")
    attestation = value.get("author_attestation")
    if (not isinstance(attestation, Mapping)
            or attestation.get("scope") != ATTESTATION_SCOPE
            or attestation.get("statement") != ATTESTATION_STATEMENT
            or not isinstance(attestation.get("attested"), bool)):
        raise SelectionError("exclusion ledger author attestation is invalid")
    attested = attestation["attested"]
    if attested:
        attestor = attestation.get("attestor")
        timestamp = attestation.get("attested_utc")
        if (not isinstance(attestor, str) or not attestor
                or "REPLACE" in attestor.upper()
                or not isinstance(timestamp, str) or not timestamp.endswith("Z")):
            raise SelectionError("completed ledger lacks a resolved author attestation")
        try:
            datetime.fromisoformat(timestamp[:-1] + "+00:00")
        except ValueError as exc:
            raise SelectionError("ledger attestation timestamp is invalid") from exc
    gaps = value.get("unresolved_provenance")
    if (not isinstance(gaps, list) or not gaps
            or any(not isinstance(gap, Mapping)
                   or not isinstance(gap.get("id"), str) or not gap["id"]
                   or not isinstance(gap.get("provenance_recovered"), bool)
                   or not isinstance(gap.get("exclusion_risk_resolved"), bool)
                   or not isinstance(gap.get("resolution"), str) or not gap["resolution"]
                   for gap in gaps)):
        raise SelectionError("exclusion ledger provenance review is invalid")
    all_risks_resolved = all(gap["exclusion_risk_resolved"] for gap in gaps)
    aggregate = value.get("aggregate_exposures")
    evidence = value.get("evidence")
    verification = value.get("verification")
    if (not isinstance(aggregate, list) or not aggregate
            or not isinstance(evidence, list) or not evidence
            or any(not isinstance(row, Mapping) or row.get("verified") is not True
                   for row in evidence)
            or not isinstance(verification, Mapping)
            or verification.get("all_evidence_verified") is not True
            or verification.get("one_streaming_raw_pass") is not True
            or isinstance(verification.get("artifact_count"), bool)
            or not isinstance(verification.get("artifact_count"), int)
            or verification["artifact_count"] < 1
            or verification.get("contributing_evidence_count") != sum(
                row.get("contributes_exclusions") is True for row in evidence)):
        raise SelectionError("exclusion ledger evidence verification is incomplete")
    derived_complete = bool(attested and all_risks_resolved)
    if value["complete"] is not derived_complete:
        raise SelectionError("exclusion ledger completeness contradicts its attestation")
    if require_complete and not derived_complete:
        raise SelectionError("exclusion ledger is not complete and author-attested")
    return indices, payload_hash


def load_exclusion_ledger(path: str | Path, *, require_complete: bool = True) \
        -> tuple[list[int], str]:
    """Load a canonical, self-hashed record-level exposure ledger."""
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SelectionError("exclusion ledger is not valid JSON") from exc
    from confirmatory_contract import canonical_bytes

    if raw != canonical_bytes(value):
        raise SelectionError("exclusion ledger is not canonical JSON")
    if not isinstance(value, Mapping):
        raise SelectionError("exclusion ledger must be a JSON object")
    return _validated_ledger(value, require_complete=require_complete)


def load_exclusion_indices(path: str | Path) -> list[int]:
    """Load indices from a ledger or legacy JSON/JSONL/text test fixture.

    Production CLIs use :func:`load_exclusion_ledger` and therefore require a
    complete author-attested record.  This compatibility loader remains for
    synthetic library tests and explicitly non-registered tools.
    """
    text = Path(path).read_text(encoding="utf-8")
    indices: list[int]
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, list):
        indices = [int(x["record_index"] if isinstance(x, Mapping) else x) for x in value]
    elif isinstance(value, Mapping):
        if value.get("schema") == "gbskernels.exploratory-exclusion-ledger.v1":
            from confirmatory_contract import canonical_bytes

            if text.encode("utf-8") != canonical_bytes(value):
                raise SelectionError("exclusion ledger is not canonical JSON")
            return _validated_ledger(value, require_complete=True)[0]
        rows = value.get("record_indices", value.get("excluded_record_indices"))
        if rows is None:
            raise SelectionError("exclusion JSON object lacks record_indices")
        indices = [int(x["record_index"] if isinstance(x, Mapping) else x) for x in rows]
    else:
        indices = []
        for line_number, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
                indices.append(int(row["record_index"] if isinstance(row, Mapping) else row))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise SelectionError(f"invalid exclusion at line {line_number}") from exc
    if len(indices) != len(set(indices)):
        raise SelectionError("exclusion list contains duplicate record indices")
    return sorted(indices)


def exclusion_sha256(indices: Iterable[int]) -> str:
    """Canonical hash of the explicit sorted exclusion record indices."""
    return sha256_json(sorted(int(i) for i in indices))


def _normalize_design(
    targets: str | Mapping[int | str, int],
    reserves: str | Mapping[int | str, int] | None,
    n_strata: int,
) -> tuple[dict[int, int], dict[int, int]]:
    primary = parse_count_spec(targets)
    reserve = parse_count_spec(reserves or {})
    if not primary:
        raise SelectionError("at least one primary target is required")
    if set(reserve) - set(primary):
        raise SelectionError("reserve counts may only name primary target bands")
    reserve = {C: reserve.get(C, 0) for C in primary}
    if n_strata <= 0:
        raise SelectionError("n_strata must be positive")
    return primary, reserve


def _exclusion_mask(indices: np.ndarray, exclusions: np.ndarray) -> np.ndarray:
    if len(exclusions) == 0:
        return np.zeros(len(indices), dtype=bool)
    return np.isin(indices, exclusions, assume_unique=True)


def _scan_counts(
    source: _Source,
    bands: Sequence[int],
    n_strata: int,
    exclusions: np.ndarray,
    chunk_records: int,
) -> tuple[dict[int, list[int]], str]:
    counts = {C: np.zeros(n_strata, dtype=np.int64) for C in bands}
    source_hash = hashlib.sha256()
    for base, raw in source.chunks(chunk_records):
        source_hash.update(raw.tobytes())
        _, patterns, abnormal = decode_records(raw)
        clicks = patterns.sum(axis=1)
        ridx = base + np.arange(len(raw), dtype=np.int64)
        strata = common_stratum(ridx, source.n_records, n_strata)
        excluded = _exclusion_mask(ridx, exclusions)
        eligible_base = (~abnormal) & (~excluded)
        for C in bands:
            selected = eligible_base & (clicks == C)
            if np.any(selected):
                counts[C] += np.bincount(strata[selected], minlength=n_strata)
    return {C: [int(x) for x in counts[C]] for C in bands}, source_hash.hexdigest()


def population_audit_from_counts(*, source_hash: str, n_records: int,
                                 exclusions: Sequence[int], bands: Sequence[int],
                                 n_strata: int,
                                 counts: Mapping[int, Sequence[int]],
                                 exclusion_ledger_sha256: str | None = None,
                                 exclusion_ledger_complete: bool | None = None) -> dict[str, Any]:
    band_list = [int(x) for x in bands]
    if len(set(band_list)) != len(band_list) or n_records <= 0 or n_strata <= 0:
        raise SelectionError("population audit dimensions are invalid")
    if (not isinstance(source_hash, str) or len(source_hash) != 64
            or any(ch not in "0123456789abcdefABCDEF" for ch in source_hash)):
        raise SelectionError("population audit source hash is invalid")
    totals = {}
    for band in band_list:
        values = [int(x) for x in counts[int(band)]]
        if len(values) != n_strata or any(x < 0 for x in values):
            raise SelectionError("population audit stratum counts are invalid")
        totals[int(band)] = sum(values)
    window_total = sum(totals.values())
    if window_total <= 0 or any(total <= 0 for total in totals.values()):
        raise SelectionError("every registered band needs a non-empty eligible population")
    body = {
        "schema": "gbskernels.selection-population-audit.v1",
        "source_raw_sha256": source_hash, "n_records": int(n_records),
        "record_bytes": RECORD_BYTES, "exclusion_sha256": exclusion_sha256(exclusions),
        "n_excluded": len(exclusions), "bands": band_list,
        "n_strata": int(n_strata),
        "eligible_by_band_stratum": {
            str(band): [int(x) for x in counts[int(band)]] for band in band_list},
        "eligible_by_band": {str(band): totals[int(band)] for band in band_list},
        "band_weights_within_window": {
            str(band): totals[int(band)] / window_total for band in band_list},
        "band_weights_exact": {
            str(band): f"{totals[int(band)]}/{window_total}" for band in band_list},
    }
    if exclusion_ledger_sha256 is not None:
        if (not isinstance(exclusion_ledger_sha256, str)
                or len(exclusion_ledger_sha256) != 64
                or any(ch not in "0123456789abcdefABCDEF"
                       for ch in exclusion_ledger_sha256)):
            raise SelectionError("population audit exclusion-ledger hash is invalid")
        body["exclusion_ledger_sha256"] = exclusion_ledger_sha256.lower()
        if exclusion_ledger_complete is None:
            exclusion_ledger_complete = True
        if not isinstance(exclusion_ledger_complete, bool):
            raise SelectionError("population audit ledger completeness must be boolean")
        body["exclusion_ledger_complete"] = exclusion_ledger_complete
        body["registration_eligible"] = exclusion_ledger_complete
    return {**body, "audit_payload_sha256": sha256_json(body)}


def _push_smallest(heap: list, capacity: int, candidate: _Candidate) -> None:
    if capacity <= 0:
        return
    key_int = int(candidate.key, 16)
    item = (-key_int, -candidate.record_index, candidate)
    if len(heap) < capacity:
        heapq.heappush(heap, item)
        return
    worst_key = -heap[0][0]
    worst_record = -heap[0][1]
    if (key_int, candidate.record_index) < (worst_key, worst_record):
        heapq.heapreplace(heap, item)


def _scan_ranked(
    source: _Source,
    allocations: Mapping[int, Mapping[str, Sequence[int]]],
    seed: int | str,
    n_strata: int,
    exclusions: np.ndarray,
    chunk_records: int,
    expected_source_hash: str,
) -> dict[tuple[int, int], list[_Candidate]]:
    bands = tuple(sorted(allocations))
    heaps: dict[tuple[int, int], list] = {(C, h): [] for C in bands for h in range(n_strata)}
    capacities = {
        (C, h): int(allocations[C]["primary"][h]) + int(allocations[C]["reserve"][h])
        for C in bands
        for h in range(n_strata)
    }
    source_hash = hashlib.sha256()
    for base, raw in source.chunks(chunk_records):
        source_hash.update(raw.tobytes())
        timestamps, patterns, abnormal = decode_records(raw)
        clicks = patterns.sum(axis=1)
        ridx = base + np.arange(len(raw), dtype=np.int64)
        strata = common_stratum(ridx, source.n_records, n_strata)
        excluded = _exclusion_mask(ridx, exclusions)
        eligible_base = (~abnormal) & (~excluded)
        for C in bands:
            local = np.flatnonzero(eligible_base & (clicks == C))
            for li in local:
                h = int(strata[li])
                if capacities[(C, h)] == 0:
                    continue
                index = int(ridx[li])
                candidate = _Candidate(
                    band=C,
                    record_index=index,
                    stratum=h,
                    timestamp=int(timestamps[li]),
                    abnormal=bool(abnormal[li]),
                    pattern=tuple(int(x) for x in patterns[li]),
                    raw=raw[li].tobytes(),
                    key=event_key(index, seed),
                )
                _push_smallest(heaps[(C, h)], capacities[(C, h)], candidate)
    if source_hash.hexdigest() != expected_source_hash:
        raise SelectionError("raw source changed between the count and ranking passes")
    ranked: dict[tuple[int, int], list[_Candidate]] = {}
    for cell, heap in heaps.items():
        ranked[cell] = sorted((item[2] for item in heap), key=lambda row: (row.key, row.record_index))
        if len(ranked[cell]) != capacities[cell]:
            raise SelectionError(f"selection shortfall in band/stratum {cell}")
    return ranked


def _probability_fields(primary: int, reserve: int, eligible: int, role: str) -> dict[str, Any]:
    role_n = primary if role == "primary" else reserve
    combined = primary + reserve
    return {
        "inclusion_probability": role_n / eligible,
        "inclusion_probability_exact": f"{role_n}/{eligible}",
        "primary_inclusion_probability": primary / eligible,
        "primary_inclusion_probability_exact": f"{primary}/{eligible}",
        "manifest_inclusion_probability": combined / eligible,
        "manifest_inclusion_probability_exact": f"{combined}/{eligible}",
    }


def _audit_row(
    candidate: _Candidate,
    *,
    role: str,
    rank: int,
    primary_quota: int,
    reserve_quota: int,
    eligible: int,
) -> dict[str, Any]:
    packed = np.packbits(np.asarray(candidate.pattern, dtype=np.uint8)).tobytes()
    pattern_sha256 = sha256_bytes(packed)
    input_hash = sha256_bytes(
        INPUT_HASH_DOMAIN + candidate.band.to_bytes(2, "big") + candidate.raw
    )
    out = {
        "role": role,
        "band": candidate.band,
        "record_index": candidate.record_index,
        "timestamp_bits": format(candidate.timestamp, "016b"),
        "timestamp_uint16": candidate.timestamp,
        "abnormal": candidate.abnormal,
        "pattern": list(candidate.pattern),
        "pattern_packed_hex": packed.hex(),
        "pattern_sha256": pattern_sha256,
        "raw_record_hex": candidate.raw.hex(),
        "key": candidate.key,
        "input_hash": input_hash,
        "source_raw_hash": sha256_bytes(candidate.raw),
        "stratum": candidate.stratum,
        "rank_in_stratum": rank,
        "eligible_in_stratum": eligible,
        "primary_quota": primary_quota,
        "reserve_quota": reserve_quota,
    }
    out.update(_probability_fields(primary_quota, reserve_quota, eligible, role))
    return out


def _plan_selection(plan: Any) -> Mapping[str, Any] | None:
    if not isinstance(plan, Mapping):
        return None
    value = plan.get("selection", plan)
    return value if isinstance(value, Mapping) else None


def _assert_registered_design(
    registration: Mapping[str, Any],
    *,
    targets: Mapping[int, int],
    reserves: Mapping[int, int],
    n_strata: int,
    n_records: int,
    source_hash: str,
    excluded_hash: str,
    exclusion_ledger_sha256: str,
    population_audit: Mapping[str, Any],
) -> None:
    selection = _plan_selection(registration.get("plan"))
    if selection is None:
        raise SelectionError("registered plan does not contain a selection object")
    required = {
        "targets": {str(C): n for C, n in targets.items()},
        "reserves": {str(C): n for C, n in reserves.items()},
        "n_strata": n_strata,
        "n_records": n_records,
        "record_bytes": RECORD_BYTES,
        "source_raw_sha256": source_hash,
        "exclusion_sha256": excluded_hash,
        "exclusion_ledger_sha256": exclusion_ledger_sha256,
        "population_audit_sha256": population_audit["audit_payload_sha256"],
    }
    for field, expected in required.items():
        if field not in selection:
            raise SelectionError(f"registered selection lacks {field}")
        actual = selection[field]
        if field in ("targets", "reserves"):
            actual = {str(C): n for C, n in parse_count_spec(actual).items()}
        elif field in ("n_strata", "n_records", "record_bytes"):
            actual = int(actual)
        elif field in ("source_raw_sha256", "exclusion_sha256",
                       "exclusion_ledger_sha256", "population_audit_sha256"):
            actual = str(actual).lower()
        if actual != expected:
            raise SelectionError(
                f"registered selection {field} mismatch: got {actual!r}, expected {expected!r}"
            )
    registered_weights = {
        str(k): float(v) for k, v in registration["plan"]["analysis"]["band_weights"].items()}
    audited_weights = {str(k): float(v) for k, v in
                       population_audit["band_weights_within_window"].items()}
    if registered_weights.keys() != audited_weights.keys() or any(
            not np.isclose(registered_weights[key], audited_weights[key], atol=1e-12, rtol=0)
            for key in registered_weights):
        raise SelectionError("registered band weights differ from the full population audit")


def select_from_raw(
    source: str | Path | bytes | bytearray | memoryview | np.ndarray | Iterable[bytes],
    *,
    targets: str | Mapping[int | str, int],
    reserves: str | Mapping[int | str, int] | None,
    n_strata: int,
    exclude_record_indices: Iterable[int],
    exclusion_ledger_sha256: str | None = None,
    registration: Mapping[str, Any] | None = None,
    seed: int | str | None = None,
    output_path: str | Path | None = None,
    chunk_records: int = 2_000_000,
) -> dict[str, Any]:
    """Select primary and reserve events from the complete raw acquisition.

    Exactly one of ``registration`` or ``seed`` must be supplied.  An explicit
    empty exclusion list is allowed, but implicit cap-based exclusions are not.
    """
    primary_targets, reserve_targets = _normalize_design(targets, reserves, n_strata)
    raw_source = _coerce_source(source)
    if raw_source.n_records == 0:
        raise SelectionError("raw source is empty")

    excluded = [int(i) for i in exclude_record_indices]
    if len(excluded) != len(set(excluded)):
        raise SelectionError("exclude_record_indices contains duplicates")
    excluded = sorted(excluded)
    if any(i < 0 or i >= raw_source.n_records for i in excluded):
        raise SelectionError("excluded record index outside [0,n_records)")
    exclusions = np.asarray(excluded, dtype=np.int64)
    excluded_hash = exclusion_sha256(excluded)

    if (registration is None) == (seed is None):
        raise SelectionError("supply exactly one of registration or seed")
    normalized_registration = None
    if registration is not None:
        try:
            normalized_registration = validate_registration(registration)
        except ContractError as exc:
            raise SelectionError(f"invalid public registration: {exc}") from exc
        # Rank with all 256 derived bits.  ``seed_uint64`` is retained only for
        # third-party APIs that cannot accept the full digest.
        selection_seed: int | str = normalized_registration["seed_derivation"]["seed_hex"]
        if exclusion_ledger_sha256 is None:
            raise SelectionError("registered selection requires a complete exclusion ledger")
    else:
        selection_seed = seed  # type: ignore[assignment]

    bands = tuple(primary_targets)
    eligible_counts, source_hash = _scan_counts(
        raw_source, bands, n_strata, exclusions, chunk_records
    )
    population_audit = population_audit_from_counts(
        source_hash=source_hash, n_records=raw_source.n_records,
        exclusions=excluded, bands=bands, n_strata=n_strata,
        counts=eligible_counts,
        exclusion_ledger_sha256=exclusion_ledger_sha256,
        exclusion_ledger_complete=True)
    allocations: dict[int, dict[str, list[int]]] = {}
    for C in bands:
        primary = largest_remainder_allocation(eligible_counts[C], primary_targets[C])
        remaining = [N - n for N, n in zip(eligible_counts[C], primary, strict=True)]
        reserve = largest_remainder_allocation(remaining, reserve_targets[C])
        if normalized_registration is not None and any(n < 2 for n in primary):
            raise SelectionError(
                f"registered fixed-stratum design has fewer than two primaries "
                f"in a C={C} cell; "
                "reduce n_strata or increase the target"
            )
        allocations[C] = {"primary": primary, "reserve": reserve}

    if normalized_registration is not None:
        _assert_registered_design(
            normalized_registration,
            targets=primary_targets,
            reserves=reserve_targets,
            n_strata=n_strata,
            n_records=raw_source.n_records,
            source_hash=source_hash,
            excluded_hash=excluded_hash,
            exclusion_ledger_sha256=str(exclusion_ledger_sha256).lower(),
            population_audit=population_audit,
        )

    ranked = _scan_ranked(
        raw_source,
        allocations,
        selection_seed,
        n_strata,
        exclusions,
        chunk_records,
        source_hash,
    )
    primary_rows: list[dict[str, Any]] = []
    reserve_rows: list[dict[str, Any]] = []
    for C in bands:
        for h in range(n_strata):
            p = allocations[C]["primary"][h]
            r = allocations[C]["reserve"][h]
            N = eligible_counts[C][h]
            rows = ranked[(C, h)]
            primary_rows.extend(
                _audit_row(
                    row,
                    role="primary",
                    rank=rank,
                    primary_quota=p,
                    reserve_quota=r,
                    eligible=N,
                )
                for rank, row in enumerate(rows[:p])
            )
            reserve_rows.extend(
                _audit_row(
                    row,
                    role="reserve",
                    rank=p + rank,
                    primary_quota=p,
                    reserve_quota=r,
                    eligible=N,
                )
                for rank, row in enumerate(rows[p : p + r])
            )

    design = {
        str(C): {
            "eligible_by_stratum": eligible_counts[C],
            "eligible_total": sum(eligible_counts[C]),
            "primary_target": primary_targets[C],
            "primary_by_stratum": allocations[C]["primary"],
            "reserve_target": reserve_targets[C],
            "reserve_by_stratum": allocations[C]["reserve"],
        }
        for C in bands
    }
    manifest: dict[str, Any] = {
        "kind": "jiuzhang1_confirmatory_selection_v2",
        "registered": normalized_registration is not None,
        "registration": normalized_registration,
        "seed": str(selection_seed),
        "source": {
            "record_bytes": RECORD_BYTES,
            "n_records": raw_source.n_records,
            "source_raw_sha256": source_hash,
        },
        "strata": {
            "count": n_strata,
            "definition": "floor(record_index * n_strata / n_records)",
            "edges": stratum_edges(raw_source.n_records, n_strata),
        },
        "exclusions": {
            "record_indices": excluded,
            "count": len(excluded),
            "sha256": excluded_hash,
            "ledger_payload_sha256": exclusion_ledger_sha256,
        },
        "population_audit": population_audit,
        "design": design,
        "hash_contract": {
            "canonical_json": "sorted keys, compact separators, ASCII, finite numbers",
            "selection_key": "SHA256 domain-separated record_index and beacon seed",
            "input_hash": "SHA256(domain || uint16 band || exact 16-byte source record)",
            "source_raw_hash": "SHA256(exact 16-byte source record)",
        },
        "primary": primary_rows,
        "reserves": reserve_rows,
    }
    manifest["manifest_payload_sha256"] = sha256_json(manifest)
    if output_path is not None:
        write_canonical_json(output_path, manifest)
    return manifest


# Stable public aliases for callers that prefer a verb matching the script name.
select_records = select_from_raw
select_confirmatory_v2 = select_from_raw


def _selection_plan(registration: Mapping[str, Any]) -> Mapping[str, Any]:
    plan = _plan_selection(registration.get("plan"))
    if plan is None:
        raise SelectionError("registration plan lacks selection configuration")
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registration", type=Path, required=True,
                    help="resolved public-registration JSON with beacon proof")
    ap.add_argument("--data", type=Path, default=REPO / "data" / "jiuzhang1" / "data.bin")
    ap.add_argument("--exclude-records", type=Path, required=True,
                    help="canonical complete exploratory-exclusion ledger")
    ap.add_argument("--out", type=Path, required=True,
                    help="canonical v2 selection manifest JSON")
    ap.add_argument("--chunk-records", type=int, default=2_000_000)
    args = ap.parse_args()

    normalized = load_registration(args.registration)
    selection = _selection_plan(normalized)
    targets = selection.get("targets")
    reserves = selection.get("reserves", {})
    n_strata = int(selection.get("n_strata", 0))
    exclusions, ledger_hash = load_exclusion_ledger(args.exclude_records)
    manifest = select_from_raw(
        args.data,
        targets=targets,
        reserves=reserves,
        n_strata=n_strata,
        exclude_record_indices=exclusions,
        exclusion_ledger_sha256=ledger_hash,
        registration=normalized,
        output_path=args.out,
        chunk_records=args.chunk_records,
    )
    print(f"source SHA256: {manifest['source']['source_raw_sha256']}")
    for C, row in manifest["design"].items():
        print(
            f"C={C}: eligible {row['eligible_total']}, "
            f"primary {row['primary_target']}, reserves {row['reserve_target']}"
        )
    print(f"manifest payload SHA256: {manifest['manifest_payload_sha256']}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
