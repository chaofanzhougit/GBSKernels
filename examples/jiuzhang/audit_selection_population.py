"""Audit the full pre-registration selection population without scoring events.

This streaming pass produces the raw-data hash, explicit-exclusion hash,
eligible band/stratum counts, and full-pool conditional band weights used to
fill the public plan template. It performs no randomness selection and no
likelihood evaluation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from confirmatory_contract import write_canonical_json  # noqa: E402
from select_confirmatory_v2 import (_coerce_source, _scan_counts,
                                    load_exclusion_ledger, parse_count_spec,
                                    population_audit_from_counts)  # noqa: E402


def audit(source: str | Path | bytes, *, bands: list[int], n_strata: int,
          exclude_record_indices: list[int], chunk_records: int = 2_000_000,
          exclusion_ledger_sha256: str | None = None,
          exclusion_ledger_complete: bool | None = None) -> dict:
    raw = _coerce_source(source)
    exclusions = sorted(int(x) for x in exclude_record_indices)
    if len(exclusions) != len(set(exclusions)):
        raise ValueError("exclusion indices contain duplicates")
    if any(index < 0 or index >= raw.n_records for index in exclusions):
        raise ValueError("exclusion index is outside the source")
    import numpy as np

    counts, source_hash = _scan_counts(
        raw, bands, n_strata, np.asarray(exclusions, dtype=np.int64), chunk_records)
    return population_audit_from_counts(
        source_hash=source_hash, n_records=raw.n_records, exclusions=exclusions,
        bands=bands, n_strata=n_strata, counts=counts,
        exclusion_ledger_sha256=exclusion_ledger_sha256,
        exclusion_ledger_complete=exclusion_ledger_complete)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path,
                    default=REPO / "data" / "jiuzhang1" / "data.bin")
    ap.add_argument("--exclude-records", type=Path, required=True)
    ap.add_argument("--bands", default="27:1,28:1,29:1,30:1")
    ap.add_argument("--n-strata", type=int, required=True)
    ap.add_argument("--chunk-records", type=int, default=2_000_000)
    ap.add_argument("--allow-incomplete-ledger", action="store_true",
                    help="write an explicitly registration-ineligible draft audit")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    bands = (list(parse_count_spec(args.bands)) if ":" in args.bands
             else [int(value) for value in args.bands.split(",") if value.strip()])
    exclusions, ledger_hash = load_exclusion_ledger(
        args.exclude_records, require_complete=not args.allow_incomplete_ledger)
    value = audit(args.data, bands=bands, n_strata=args.n_strata,
                  exclude_record_indices=exclusions,
                  exclusion_ledger_sha256=ledger_hash,
                  exclusion_ledger_complete=not args.allow_incomplete_ledger,
                  chunk_records=args.chunk_records)
    write_canonical_json(args.out, value)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
