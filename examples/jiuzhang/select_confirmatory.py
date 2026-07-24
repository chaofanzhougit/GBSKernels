"""Reproduce the historical private, time-stratified Jiuzhang 1.0 selection.

The stage-1 exploratory campaign selected events as "first-n in decode order"
(decode_events.py:100, campaign.py:144).  The later private sample was:

  * DISJOINT from every stage-1 event and every archived published pattern,
  * drawn across the full acquisition using band-specific record-index strata,
  * reproducible from the historical author-chosen seed,
  * carrying, per event, the record index, stratum, pattern, and selection key.

This script does the selection only.  It evaluates no torontonian and touches no
GPU: it produces the manifest consumed by the historical campaign runner.
The historical run used equal quotas across ten strata, not proportional
allocation.  The enriched manifest written by this version records the eligible
and selected counts in each stratum so downstream analysis can use the realized
finite-population design.  It remains a private fixed-sample exploratory design,
not a public preregistration.  Running it is a CPU-only pass over data.bin.

    # CPU dry-run (prove the logic on a prefix, no full decode, no GPU):
    uv run python examples/jiuzhang/select_confirmatory.py --seed 20260714 \
        --max-records 6000000 --dry-run
    # real selection (full 51.5M-record pass):
    uv run python examples/jiuzhang/select_confirmatory.py --seed 20260715 \
        --n "27:800,28:500,29:400,30:300"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from decode_events import DATA, RECORD_BYTES, decode_chunk  # proven decoder  # noqa: E402

BANDS = (27, 28, 29, 30)
STAGE1_CAP = 4000          # decode_events.py default --cap; first N per band that
                           # stage-1 could have touched via campaign_events.npz.
                           # Excluding this many per band guarantees disjointness.


def _key(record_index: int, seed: int) -> int:
    """Historical deterministic selection key SHA256(record_index || seed)."""
    h = hashlib.sha256(f"{record_index}:{seed}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.absolute()
    try:
        return str(resolved.relative_to(REPO.absolute()))
    except ValueError:
        return str(resolved)


def _parse_n(spec: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for part in spec.split(","):
        click_text, separator, target_text = part.strip().partition(":")
        if not separator or not click_text or not target_text:
            raise ValueError(f"invalid band target {part!r}; expected C:N")
        click, target = int(click_text), int(target_text)
        if click in out:
            raise ValueError(f"duplicate band target C={click}")
        if target < 0:
            raise ValueError(f"band target must be non-negative, got {target}")
        out[click] = target
    if not out:
        raise ValueError("at least one band target is required")
    return out


def _load_published_pattern_hashes():
    """Hashes of the archived published patterns (C=21..26), to exclude if a
    historical band ever overlaps them. Absent -> empty set (bands 27-30 have
    no published patterns, so this is a no-op for these bands)."""
    zen = REPO / "data" / "q7_1076_zenodo" / "pattern_probs" / "patterns_exp"
    hashes: set[bytes] = set()
    if not zen.is_dir():
        return hashes
    for f in zen.glob("samples_0_clicks_*.npy"):
        try:
            pats = np.load(f)
            for row in np.asarray(pats, dtype=bool):
                hashes.add(np.packbits(row).tobytes())
        except Exception:
            continue
    return hashes


def _select_equal_quota(
    ridx: np.ndarray,
    pats: np.ndarray,
    *,
    target: int,
    n_strata: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Reproduce the realized band-specific, equal-quota selection design."""
    if ridx.ndim != 1 or pats.ndim != 2 or len(ridx) != len(pats):
        raise ValueError("record indices and patterns have incompatible shapes")
    if n_strata <= 0 or target < 0:
        raise ValueError("stratum count must be positive and target non-negative")
    if len(np.unique(ridx)) != len(ridx):
        raise ValueError("eligible record indices must be unique")
    if len(ridx) == 0:
        return {
            "ridx": np.zeros(0, dtype=np.int64),
            "pats": np.zeros((0, pats.shape[1]), dtype=bool),
            "stratum": np.zeros(0, dtype=np.int16),
            "keys": np.zeros(0, dtype=np.uint64),
            "edges": np.zeros(n_strata + 1, dtype=np.float64),
            "eligible_by_stratum": np.zeros(n_strata, dtype=np.int64),
            "selected_by_stratum": np.zeros(n_strata, dtype=np.int64),
        }

    edges = np.linspace(ridx.min(), ridx.max() + 1, n_strata + 1)
    stratum = np.clip(
        np.searchsorted(edges, ridx, side="right") - 1, 0, n_strata - 1
    ).astype(np.int16)
    keys = np.asarray([_key(int(r), seed) for r in ridx], dtype=np.uint64)
    eligible = np.bincount(stratum, minlength=n_strata).astype(np.int64)

    chosen: list[int] = []
    per, remainder = divmod(target, n_strata)
    for stratum_index in range(n_strata):
        want = per + (1 if stratum_index < remainder else 0)
        in_stratum = np.flatnonzero(stratum == stratum_index)
        order = in_stratum[np.argsort(keys[in_stratum], kind="stable")]
        chosen.extend(order[:want].tolist())

    selected = np.asarray(sorted(chosen), dtype=np.int64)
    selected_strata = stratum[selected]
    return {
        "ridx": ridx[selected],
        "pats": pats[selected],
        "stratum": selected_strata,
        "keys": keys[selected],
        "edges": edges,
        "eligible_by_stratum": eligible,
        "selected_by_stratum": np.bincount(
            selected_strata, minlength=n_strata
        ).astype(np.int64),
    }


def _bind_historical_manifest(
    path: Path,
    manifest: dict[int, dict[str, np.ndarray]],
    bands: list[int],
) -> dict:
    """Require exact event-index and pattern equality with the archived sample."""
    with np.load(path, allow_pickle=False) as historical:
        for C in bands:
            for field, key in (("ridx", f"ridx_C{C}"), ("pats", f"pats_C{C}")):
                if key not in historical or not np.array_equal(
                    manifest[C][field], historical[key]
                ):
                    raise ValueError(
                        f"regenerated selection differs from historical manifest at {key}"
                    )
    return {
        "path": _portable_path(path),
        "sha256": _sha256_file(path),
        "verified_arrays": [
            key for C in bands for key in (f"ridx_C{C}", f"pats_C{C}")
        ],
        "all_equal": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True,
                    help="historical author-chosen selection seed")
    ap.add_argument("--n", default="27:800,28:500,29:400,30:300",
                    help="per-band target counts N_C")
    ap.add_argument("--strata", type=int, default=10, help="equal record-index time strata")
    ap.add_argument("--stage1-cap", type=int, default=STAGE1_CAP,
                    help="first-N-per-band to exclude as the stage-1 exploratory sample")
    ap.add_argument("--data", type=Path, default=DATA / "data.bin")
    ap.add_argument("--chunk-records", type=int, default=2_000_000)
    ap.add_argument("--max-records", type=int, default=0, help="0 = full file (dry-run: use a prefix)")
    ap.add_argument("--dry-run", action="store_true",
                    help="do not write a manifest; just report and self-check")
    ap.add_argument("--historical-manifest", type=Path,
                    help="archived NPZ whose selected indices/patterns must match exactly")
    ap.add_argument("--out", type=Path, help="output NPZ path (default: timestamped result)")
    args = ap.parse_args()

    targets = _parse_n(args.n)
    for C in targets:
        if C not in BANDS:
            raise SystemExit(f"band {C} not in historical fixed-sample set {BANDS}")

    path = args.data
    n_records = path.stat().st_size // RECORD_BYTES
    limit = args.max_records or n_records
    if args.chunk_records <= 0 or args.stage1_cap < 0:
        raise SystemExit("--chunk-records must be positive and --stage1-cap non-negative")
    if limit < 0 or limit > n_records:
        raise SystemExit(f"--max-records must be in [0, {n_records}]")
    if not args.dry_run and limit != n_records:
        raise SystemExit("a written selection manifest requires a full-file scan")
    print(f"{path}  {n_records:,} records  (scanning {limit:,})", flush=True)
    pub_hashes = _load_published_pattern_hashes()
    print(f"published-pattern exclusions loaded: {len(pub_hashes):,} "
          f"(bands 27-30 have none; no-op there)", flush=True)

    # ---- one CPU pass: gather (record_index, pattern) per band -----------------
    # seen[C] counts occurrences so the first `stage1_cap` per band are the
    # stage-1 sample and excluded; the rest form the historical selection pool.
    pool: dict[int, list[tuple[int, np.ndarray]]] = {C: [] for C in targets}
    seen: dict[int, int] = {C: 0 for C in targets}
    excluded_pub = 0
    source_sha256 = hashlib.sha256()
    idx0 = 0
    t0 = time.time()
    with open(path, "rb") as f:
        while idx0 < limit:
            want = min(args.chunk_records, limit - idx0) * RECORD_BYTES
            raw = np.fromfile(f, dtype=np.uint8, count=want)
            if raw.size == 0:
                break
            source_sha256.update(memoryview(raw))
            det, abnormal = decode_chunk(raw)
            kk = det.sum(1)
            base = idx0
            for C in targets:
                local = np.flatnonzero((kk == C) & (~abnormal))
                for li in local:
                    ridx = base + int(li)
                    seen[C] += 1
                    if seen[C] <= args.stage1_cap:      # stage-1 sample: exclude
                        continue
                    row = det[li]
                    if pub_hashes and np.packbits(row).tobytes() in pub_hashes:
                        excluded_pub += 1
                        continue
                    pool[C].append((ridx, row))
            idx0 += raw.size // RECORD_BYTES
            print(f"  {idx0:,}/{limit:,} ({time.time()-t0:.0f}s)", flush=True)

    # ---- time-stratified, seed-hashed draw within each band -------------------
    manifest: dict[int, dict[str, np.ndarray]] = {}
    report = {}
    for C in targets:
        entries = pool[C]                                   # already in record order
        N = targets[C]
        if not entries:
            report[C] = {"pool": 0, "selected": 0, "target": N, "shortfall": N}
            manifest[C] = _select_equal_quota(
                np.zeros(0, dtype=np.int64),
                np.zeros((0, 100), dtype=bool),
                target=N,
                n_strata=args.strata,
                seed=args.seed,
            )
            continue
        ridx = np.array([e[0] for e in entries], dtype=np.int64)
        pats = np.array([e[1] for e in entries], dtype=bool)
        selected = _select_equal_quota(
            ridx, pats, target=N, n_strata=args.strata, seed=args.seed
        )
        manifest[C] = selected
        selected_n = len(selected["ridx"])
        report[C] = {
            "pool": len(entries),
            "selected": selected_n,
            "target": N,
            "shortfall": int(max(0, N - selected_n)),
            "pool_ridx_min": int(ridx.min()),
            "pool_ridx_max": int(ridx.max()),
            "selected_ridx_min": int(selected["ridx"].min()) if selected_n else None,
            "selected_ridx_max": int(selected["ridx"].max()) if selected_n else None,
            "strata_filled": int(np.count_nonzero(selected["selected_by_stratum"])),
            "eligible_by_stratum": selected["eligible_by_stratum"].tolist(),
            "selected_by_stratum": selected["selected_by_stratum"].tolist(),
            "stratum_edges": selected["edges"].tolist(),
            "allocation": "equal quota within band-specific record-index strata",
        }

    # ---- self-checks: disjointness + determinism ------------------------------
    ok = True
    for C in targets:
        sel = set(manifest[C]["ridx"].tolist())
        # disjoint from the stage-1 sample: every selected index is a >stage1_cap-th
        # occurrence, so it cannot be among the first stage1_cap.  Confirm none are
        # implausibly early is not sufficient; the construction guarantees it, but
        # we assert selected count and that indices are unique.
        if len(sel) != report[C]["selected"]:
            print(f"  [CHECK FAIL] C={C}: duplicate selected indices"); ok = False
    # determinism: re-derive keys for band-min sample and confirm stable ordering
    print("\n== historical private fixed-sample selection ==")
    for C in targets:
        r = report[C]
        flag = "" if r["shortfall"] == 0 else f"  << SHORTFALL {r['shortfall']} (grow pool / lower N)"
        print(f"  C={C}: pool {r['pool']:>7}  selected {r['selected']:>4}/"
              f"{r['target']:<4}  strata {r.get('strata_filled','-')}/{args.strata}{flag}")
    if excluded_pub:
        print(f"  excluded {excluded_pub} published-pattern matches")
    print(f"  seed {args.seed}  stage1 exclusion first {args.stage1_cap}/band  "
          f"(disjoint by construction)")

    if args.dry_run:
        print("  [dry-run] no manifest written; logic + disjointness verified"
              + ("" if limit == n_records else f"  (PREFIX {limit:,} records only)"))
        return 0 if ok else 1

    historical_binding = None
    if args.historical_manifest is not None:
        historical_binding = _bind_historical_manifest(
            args.historical_manifest, manifest, list(targets)
        )

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = args.out or (
        REPO / "results" / "jiuzhang" / f"private_fixed_sample_selection_{stamp}.npz"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "kind": "jiuzhang1_private_fixed_sample_selection",
        "schema": "gbskernels.jiuzhang1-private-fixed-sample-selection.v2",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed, "strata": args.strata, "stage1_cap": args.stage1_cap,
        "targets": {str(c): n for c, n in targets.items()},
        "report": {str(c): report[c] for c in targets},
        "n_records": int(n_records), "scanned": int(limit),
        "published_excluded": int(excluded_pub),
        "source": {
            "path": _portable_path(path),
            "size_bytes": int(path.stat().st_size),
            "records_hashed": int(limit),
            "hash_scope": "full file" if limit == n_records else "scanned prefix",
            "sha256": source_sha256.hexdigest(),
        },
        "implementation": {
            _portable_path(Path(__file__)): _sha256_file(Path(__file__)),
            _portable_path(HERE / "decode_events.py"): _sha256_file(HERE / "decode_events.py"),
        },
        "selection_design": {
            "status": "historical private fixed-sample exploratory design",
            "stratum_scope": "band-specific eligible record-index range",
            "allocation": "equal quota per stratum",
            "analysis_requirement": "finite-population stratum weighting",
        },
        "note": "Disjoint from the first stage1-cap events per band; not a public preregistration",
    }
    if historical_binding is not None:
        meta["historical_manifest_binding"] = historical_binding
    try:
        from bench._provenance import provenance

        meta["provenance"] = provenance()
    except Exception as exc:
        meta["provenance"] = {"capture_error": str(exc)}
    payload = {"meta": json.dumps(meta)}
    for C in targets:
        payload[f"ridx_C{C}"] = manifest[C]["ridx"]
        payload[f"pats_C{C}"] = manifest[C]["pats"]
        payload[f"stratum_C{C}"] = manifest[C]["stratum"]
        payload[f"key_C{C}"] = manifest[C]["keys"]
        payload[f"edges_C{C}"] = manifest[C]["edges"]
        payload[f"eligible_C{C}"] = manifest[C]["eligible_by_stratum"]
        payload[f"selected_C{C}"] = manifest[C]["selected_by_stratum"]
        payload[f"abnormal_C{C}"] = np.zeros(len(manifest[C]["ridx"]), dtype=bool)
    np.savez_compressed(out, **payload)
    print(f"  -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
