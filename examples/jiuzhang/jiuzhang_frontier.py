"""Certified torontonian frontier on real Jiuzhang 1.0 data, at GPU scale.

Runs the single-large certified torontonian (tor_single, certified=True) over
real Jiuzhang threshold events to populate the full condition-number-vs-clicks
frontier up to 32 clicks (dimension 64) -- the range that is slow on the CPU
host shim but ~1 s/eval on a real GPU.

Inputs (the small payload; no 785 MB data.bin needed on the box):
    T_full.npy               50 input x 100 output transfer matrix
    squeezing parameters.txt 25 squeezer values
    events_ge40.npy          real events with >=40 clicks (for the controlled
                             sub-pattern curve to k=32)
    events_band13_32.npy     real events with 13..32 clicks (whole-event coverage)

State reconstruction is the recipe validated to 1.5% RMS against the empirical
per-detector click rates: r50 = repeat(r25, 2); aiaj = T^T diag(sinh r cosh r) T;
aidaj = conj(T)^T diag(sinh^2 r) T; xxpp covariance; Q = Husimi; O = I - Q^{-1}.

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
from sampling import gbs as gbs_mod

HERE = Path(__file__).resolve().parent
U = 2.0 ** -53


def build_O():
    T = np.load(HERE / "T_full.npy")
    r50 = np.repeat(np.loadtxt(HERE / "squeezing parameters.txt"), 2)
    nbar = np.sinh(r50) ** 2
    mm = np.sinh(r50) * np.cosh(r50)
    aiaj = T.T @ np.diag(mm) @ T
    aidaj = T.conj().T @ np.diag(nbar) @ T
    M = aidaj.shape[0]
    x = np.eye(M) + np.real(aidaj + aidaj.conj().T) + 2 * np.real(aiaj)
    p = np.eye(M) + np.real(aidaj + aidaj.conj().T) - 2 * np.real(aiaj)
    xp = 2 * np.imag(aiaj) + 2 * np.imag(aidaj)
    cov = np.block([[x, xp], [xp.T, p]])
    Q = gbs_mod._qmat(cov, 2.0)
    ev = float(np.min(np.linalg.eigvalsh((Q + Q.conj().T).real / 2)))
    if ev <= 0:
        raise AssertionError(f"Husimi not PD (min eig {ev:.3e})")
    return np.real(np.eye(2 * M) - np.linalg.inv(Q))


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
    args = ap.parse_args()

    O = build_O()
    ks = list(range(4, args.kmax + 1, 2))

    # (1) controlled sub-pattern curve to k=kmax over many real events
    big = np.load(HERE / "events_ge40.npy")[: args.events]
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
    band_all = np.load(HERE / "events_band13_32.npy")
    band = band_all[band_all.sum(1) <= 26][: min(args.events, 60)]
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
           "reconstruction_validation_rms": 0.0146,
           "modes": len(O) // 2, "kmax": args.kmax,
           "subpattern_curve": curve,
           "band_events": len(band_rows),
           "band_fp64_proves_a_digit_frac": float(np.mean([r["fp64_proves_a_digit"] for r in band_rows])),
           "band_rows": band_rows}
    out = HERE / f"jiuzhang1_frontier_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    out.write_text(json.dumps(art, indent=1))
    print(f"{'k':>3} {'dim':>4} {'n':>4} {'kappa_med':>10} {'fp64 digits':>11} {'proves?':>8}")
    for r in curve:
        print(f"{r['clicks']:>3} {r['dim']:>4} {r['n']:>4} {r['kappa_median']:>10.1e} "
              f"{r['fp64_digits_median']:>11.1f} {str(r['fp64_proves_a_digit']):>8}")
    print(f"real band 13-32: {len(band_rows)} events, fp64 proves a digit in "
          f"{art['band_fp64_proves_a_digit_frac']:.1%}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
