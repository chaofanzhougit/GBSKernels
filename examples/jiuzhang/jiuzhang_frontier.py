"""Certified torontonian frontier on real Jiuzhang 1.0 data, at GPU scale.

Runs the single-large certified torontonian (tor_single, certified=True) over
nested subpatterns of recorded high-click Jiuzhang events to populate a
controlled condition-number-vs-size curve through 32 clicks (dimension 64).
These points are not a sample of whole events conditioned on each click count;
whole-event coverage is reported separately.

Inputs (the small payload; no 785 MB data.bin needed on the box):
    T_full.npy               50 input x 100 output transfer matrix
    squeezing parameters.txt 25 squeezer values
    events_ge40.npy          real events with >=40 clicks (for the controlled
                             sub-pattern curve to k=32)
    events_band13_32.npy     real events with 13..32 clicks (whole-event coverage)

State construction is delegated to q7_construction.build_state("squeezed"):
the published paired-source model followed by the exact real quadrature-basis
torontonian matrix O = I - Sigma_x^{-1}.

    python jiuzhang_frontier.py --kmax 32 --events 120

Outputs a JSON artifact: per-k certified relative-bound distribution (the FP64
precision wall) over many events, and the real-event certifiable coverage.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import gbskernels
import q7_construction as q7

HERE = Path(__file__).resolve().parent
# inputs: the repo-local payload dir if present (data/ is gitignored and pushed
# to the box by launch_session.sh), else next to the script (legacy box layout)
DATA = HERE.parents[1] / "data" / "jiuzhang1"
if not (DATA / "T_full.npy").exists():
    DATA = HERE
# artifacts land under results/ so a session's rsync-back collects them
OUT_DIR = HERE.parents[1] / "results" / "jiuzhang"
U = 2.0 ** -53


def build_O():
    """Published paired-source Jiuzhang 1.0 squeezed-state input.

    Do not reconstruct this state locally.  The former implementation used
    50 repeated same-sign single-mode squeezers and therefore did not match
    the 25 paired-source model.
    """
    return q7.build_state("squeezed")["O"]


def cert_tor(O, clicked_modes):
    """certified tor(O_c) -> (value, bound); tor_single for the frontier."""
    m = len(O) // 2
    idx = list(clicked_modes) + [j + m for j in clicked_modes]
    sub = np.ascontiguousarray(O[np.ix_(idx, idx)])
    k = len(clicked_modes)
    if k <= 12:                                     # batched certified GPU kernel (dim <= 24)
        v, d = gbskernels.tor(sub, precision="certified", return_diagnostics=True)
        return float(np.real(v)), float(d["abs_error_bound"]) + abs(float(np.imag(v)))
    v, d = gbskernels.tor_single(sub, groups=min(k, 14), certified=True)  # single-large (dim <= 64)
    return float(v), float(d["abs_error_bound"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kmax", type=int, default=32)
    ap.add_argument("--events", type=int, default=120)
    ap.add_argument("--band-events", type=int, default=60,
                    help="whole 13-26-click events for the coverage block (0 = skip)")
    args = ap.parse_args()

    O = build_O()
    ks = list(range(4, args.kmax + 1, 2))

    # (1) controlled sub-pattern curve to k=kmax over many real events
    big = np.load(DATA / "events_ge40.npy")[: args.events]
    per_k = {k: [] for k in ks}
    for c in big:
        S = [j for j in range(len(c)) if c[j]]
        for k in ks:
            v, b = cert_tor(O, S[:k])
            if v != 0:
                per_k[k].append(b / abs(v))
    curve = []
    for k in ks:
        a = np.array(per_k[k])
        curve.append({"clicks": k, "dim": 2 * k, "n": len(a),
                      "rel_bound_median": float(np.median(a)),
                      "rel_bound_q05": float(np.quantile(a, 0.05)),
                      "rel_bound_q95": float(np.quantile(a, 0.95)),
                      "kappa_median": float(np.median(a) / U),
                      "fp64_digits_median": float(max(0.0, -np.log10(np.median(a)))),
                      "fp64_proves_a_digit": bool(np.median(a) < 0.1)})

    # (2) whole real events in the certifiable band 13..26: coverage + kappa
    # (cap at 26 clicks: a 32-click tor_single is ~2^32 leaves = minutes/event;
    # the coverage statistic is already saturated by ~16 clicks, so ceiling-range
    # events suffice and keep per-event cost bounded).
    band_all = np.load(DATA / "events_band13_32.npy")
    band = band_all[band_all.sum(1) <= 26][: min(args.events, args.band_events)]
    band_rows = []
    for c in band:
        S = [j for j in range(len(c)) if c[j]]
        v, b = cert_tor(O, S)
        rb = (b / abs(v)) if v else float("inf")
        band_rows.append({"clicks": int(sum(c)), "rel_bound": rb,
                          "fp64_proves_a_digit": bool(rb < 0.1)})

    art = {"kind": "jiuzhang1_certified_frontier_gpu",
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "commit": os.environ.get("GBS_COMMIT"),
           "container_digest": os.environ.get("GBS_CONTAINER_DIGEST"),
           "gpu_backend": gbskernels.gpu_backend_kind() if hasattr(gbskernels, "gpu_backend_kind") else None,
           "data_source": "Jiuzhang 1.0 (quantum.ustc.edu.cn node/915)",
           "state_construction": "q7_construction.build_state('squeezed')",
           "ensemble": "nested first-k subpatterns of recorded >=40-click events",
           "reconstruction_validation_rms": 0.0146,
           "modes": len(O) // 2, "kmax": args.kmax,
           "subpattern_curve": curve,
           "band_events": len(band_rows),
           "band_fp64_proves_a_digit_frac":
               float(np.mean([r["fp64_proves_a_digit"] for r in band_rows]))
               if band_rows else None,
           "band_rows": band_rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"jiuzhang1_frontier_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    out.write_text(json.dumps(art, indent=1))
    print(f"{'k':>3} {'dim':>4} {'n':>4} {'kappa_med':>10} {'fp64 digits':>11} {'proves?':>8}")
    for r in curve:
        print(f"{r['clicks']:>3} {r['dim']:>4} {r['n']:>4} {r['kappa_median']:>10.1e} "
              f"{r['fp64_digits_median']:>11.1f} {str(r['fp64_proves_a_digit']):>8}")
    if band_rows:
        print(f"real band 13-32: {len(band_rows)} events, fp64 proves a digit in "
              f"{art['band_fp64_proves_a_digit_frac']:.1%}")
    else:
        print("real band 13-32: skipped (--band-events 0)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
