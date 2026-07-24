"""Decode Jiuzhang 1.0 data.bin and extract per-band campaign event supplies.

Record format (dataset readme; also documented in the manuscript supplement):
128 bits per sample -- 16 timestamp bits, then 104 bit positions of which four
are dead/ignored and the remaining 100 are detector bits ordered detector
100 -> 1, then 8 flag bits whose last marks an abnormal sample.  Bytes are in
record order, bits MSB-first.  The ignored positions are {22, 24, 25, 43}
0-based (the readme counts them 1-based as 23/25/26/44).  The executable prefix
audit below derives the zero-rate positions over one million records and checks
that the documented 100->1 ordering reproduces ``empirical_click_rates.npy``.

The decoder is PROVEN against the committed histogram: decoding the first
3,000,000 records and dropping abnormal ones must reproduce
examples/jiuzhang/click_count_dist.npy EXACTLY (2,995,852 events, mean 43
clicks).  A convention error cannot pass this gate silently.

Output: data/jiuzhang1/campaign_events.npz with one boolean (n, 100) array per
band C in 21..32 (first-occurrence order, capped per band), the per-band total
counts in the full 51.5M-record file, and the decode provenance.

    uv run python examples/jiuzhang/decode_events.py --cap 4000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "jiuzhang1"

RECORD_BYTES = 16
DET_SLOTS = np.arange(16, 120)                      # 104 positions after the timestamp
IGNORED = np.array([22, 24, 25, 43])                # dead bits (rate exactly 0)
DET_POSITIONS = np.setdiff1d(DET_SLOTS, IGNORED)    # 100 positions, detector 100 -> 1
ABNORMAL_BIT = 127
BANDS = range(21, 33)


def decode_chunk(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(events bool (n,100) detector 1..100, abnormal bool (n,)) for a chunk of
    uint8 raw bytes whose length is a multiple of 16."""
    bits = np.unpackbits(raw.reshape(-1, RECORD_BYTES), axis=1)  # MSB-first
    det = bits[:, DET_POSITIONS][:, ::-1].astype(bool)           # -> detector 1..100
    abnormal = bits[:, ABNORMAL_BIT].astype(bool)
    return det, abnormal


def audit_decoder_prefix(raw: np.ndarray, expected_rates: np.ndarray) -> dict:
    """Derive dead slots and verify the normal-record detector ordering."""
    if raw.ndim != 1 or raw.size % RECORD_BYTES:
        raise ValueError("raw decoder audit input must contain complete records")
    bits = np.unpackbits(raw.reshape(-1, RECORD_BYTES), axis=1)
    zero_rate_slots = DET_SLOTS[np.all(bits[:, DET_SLOTS] == 0, axis=0)]
    det, abnormal = decode_chunk(raw)
    normal = det[~abnormal]
    expected = np.asarray(expected_rates, dtype=float)
    if normal.size == 0 or expected.shape != (len(DET_POSITIONS),):
        raise ValueError("decoder audit requires normal records and 100 expected rates")
    observed = normal.mean(axis=0)
    direct_rms = float(np.sqrt(np.mean((observed - expected) ** 2)))
    reversed_rms = float(np.sqrt(np.mean((observed[::-1] - expected) ** 2)))
    return {
        "records": int(len(det)),
        "normal_records": int(len(normal)),
        "zero_rate_slots": zero_rate_slots.tolist(),
        "detector_rate_rms": direct_rms,
        "reversed_order_rms": reversed_rms,
        "ordering": "detector 1..100 after reversing stored detector 100..1 bits",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap", type=int, default=4000, help="max events kept per band")
    ap.add_argument("--chunk-records", type=int, default=2_000_000)
    args = ap.parse_args()

    path = DATA / "data.bin"
    n_records = path.stat().st_size // RECORD_BYTES
    print(f"{path}  {n_records:,} records", flush=True)

    # ---- gates: derive mapping on 1M and reproduce histogram on first 3M -----
    ref = np.load(HERE / "click_count_dist.npy")
    raw = np.fromfile(path, dtype=np.uint8, count=3_000_000 * RECORD_BYTES)
    mapping_audit = audit_decoder_prefix(
        raw[: 1_000_000 * RECORD_BYTES],
        np.load(DATA / "empirical_click_rates.npy"),
    )
    if mapping_audit["zero_rate_slots"] != IGNORED.tolist():
        raise AssertionError(
            "decoder mapping gate FAILED: zero-rate positions "
            f"{mapping_audit['zero_rate_slots']} != {IGNORED.tolist()}"
        )
    if mapping_audit["detector_rate_rms"] != 0.0:
        raise AssertionError(
            "decoder mapping gate FAILED: empirical detector rates differ "
            f"(RMS {mapping_audit['detector_rate_rms']:.3e})"
        )
    print(
        "decoder mapping gate: derived dead positions "
        f"{mapping_audit['zero_rate_slots']}; detector-order RMS 0.0",
        flush=True,
    )
    det, abnormal = decode_chunk(raw)
    det = det[~abnormal]
    hist = np.bincount(det.sum(1), minlength=len(ref))[: len(ref)]
    if not np.array_equal(hist, ref):
        diff = np.flatnonzero(hist != ref)
        raise AssertionError(
            f"decoder gate FAILED: histogram mismatch at C={diff[:5].tolist()} "
            f"(got {hist[diff[:5]].tolist()}, want {ref[diff[:5]].tolist()})")
    print(f"decoder gate: first 3,000,000 records -> {len(det):,} normal events; "
          f"click histogram IDENTICAL to committed click_count_dist.npy", flush=True)

    # ---- full pass: collect per-band supplies -------------------------------
    kept = {C: [] for C in BANDS}
    totals = {C: 0 for C in BANDS}
    n_abnormal = 0
    t0 = time.time()
    with open(path, "rb") as f:
        done = 0
        while True:
            raw = np.fromfile(f, dtype=np.uint8, count=args.chunk_records * RECORD_BYTES)
            if raw.size == 0:
                break
            det, abnormal = decode_chunk(raw)
            n_abnormal += int(abnormal.sum())
            det = det[~abnormal]
            kk = det.sum(1)
            for C in BANDS:
                idx = np.flatnonzero(kk == C)
                totals[C] += len(idx)
                need = args.cap - sum(len(a) for a in kept[C])
                if need > 0 and len(idx):
                    kept[C].append(det[idx[:need]])
            done += raw.size // RECORD_BYTES
            print(f"  {done:,}/{n_records:,} records ({time.time()-t0:.0f}s)", flush=True)

    out = {}
    for C in BANDS:
        arr = np.concatenate(kept[C]) if kept[C] else np.zeros((0, 100), bool)
        out[f"C{C}"] = arr
        print(f"  C={C}: kept {len(arr):>5}  (total in file {totals[C]:,})", flush=True)
    meta = {
        "kind": "jiuzhang1_campaign_events",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_records": int(n_records), "n_abnormal": int(n_abnormal),
        "cap_per_band": args.cap,
        "band_totals_in_file": {str(C): int(totals[C]) for C in BANDS},
        "decoder_mapping_audit": mapping_audit,
        "decoder_gate": "click_count_dist.npy reproduced exactly on first 3M records",
    }
    np.savez_compressed(DATA / "campaign_events.npz", meta=json.dumps(meta), **out)
    print(f"-> {DATA / 'campaign_events.npz'}")


if __name__ == "__main__":
    main()
