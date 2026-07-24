"""Historical private fixed-sample runner (superseded by confirmatory v2).

This module is retained solely to audit the privately selected 2026-07-15
sample. It is blocked for new runs unless
``GBS_ALLOW_LEGACY_CONFIRMATORY=1`` is set. Its normalizer treatment remains a
historical sensitivity calculation, not a public confirmatory analysis, and
must not support a new scientific claim.

Distinct from the stage-1 exploratory campaign.py in every way the audit
(docs/quantum_submission_audit.md) requires:
  * events come ONLY from the frozen, disjoint, time-stratified selection manifest
    (select_confirmatory.py), never "first-n in decode order";
  * ALL N_C selected events per band are evaluated -- NO optional stopping;
  * the descriptive estimand is the prefix-mass-weighted Delta_B with predefined
    band weights;
  * arithmetic width is retained only as a legacy kernel-derived proxy, never a
    probability-level certificate or Gaussian variance.

Parallelizable: each box runs a subset of bands (and, for a big band, an event
slice) and writes a per-slice JSONL checkpoint; --aggregate merges all checkpoints
(CPU, no GPU) into Delta_B. Per-event JSONL => a dead box loses at most one eval.

    # one box, a subset (band-slice), tagged so slices don't collide on pull:
    python examples/jiuzhang/campaign_confirmatory.py --manifest M --bands 30 --slice 150:300 --tag b4
    # local aggregate over all pulled checkpoints (no GPU):
    python examples/jiuzhang/campaign_confirmatory.py --manifest M --aggregate
    # aggregate released checkpoints from an explicit directory:
    python examples/jiuzhang/campaign_confirmatory.py --manifest M --aggregate \
        --checkpoint-dir results/jiuzhang/legacy_fixed_sample
    # CPU self-test of the aggregation math:
    python examples/jiuzhang/campaign_confirmatory.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import q7_construction as q7  # noqa: E402

BANDS = (27, 28, 29, 30)
WEIGHT_COUNTS = {27: 9342, 28: 13898, 29: 19981, 30: 27671}
WEIGHT_DENOMINATOR = sum(WEIGHT_COUNTS.values())
WEIGHTS = {C: WEIGHT_COUNTS[C] / WEIGHT_DENOMINATOR for C in BANDS}
OUT_DIR = HERE.parents[1] / "results" / "jiuzhang"


def their_normalizers() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load the published grouped-click probabilities and uncertainties."""
    out = {}
    for kind, stem in (("squeezed", "squeezed"), ("squashed", "squashed")):
        pr, unc = np.load(q7.ZEN / f"click_probs/click_probs_{stem}_0.npy")
        out[kind] = (pr, unc)
    return out


# --------------------------------------------------------------------------
# Historical descriptive aggregate.  The published grouped-click uncertainty is
# an across-group standard deviation, not a pooled-estimator standard error, and
# no cross-band covariance artifact exists.  It is therefore reported separately
# as a diagonal sensitivity scale, without a confidence band or z statistic.
# --------------------------------------------------------------------------
def aggregate(band_rows: dict[int, dict]) -> dict:
    w = WEIGHTS
    B = [C for C in BANDS if C in band_rows]
    dB = sum(w[C] * band_rows[C]["delta_H"] for C in B)
    stat = float(np.sqrt(sum((w[C] * (band_rows[C]["event_se"] or 0.0)) ** 2 for C in B)))
    norm = float(np.sqrt(sum((w[C] * band_rows[C]["normalizer_scale"]) ** 2 for C in B)))
    arith = float(sum(w[C] * band_rows[C]["arithmetic_proxy"] for C in B))
    return {
        "Delta_B": dB,
        "arithmetic_proxy_displacement": arith,
        "arithmetic_proxy_range": [dB - arith, dB + arith],
        "event_se": stat,
        "normalizer_diagonal_sensitivity_scale": norm,
        "normalizer_assumption": (
            "quadrature of published per-band across-group standard deviations; "
            "cross-band covariance unavailable; not an estimator standard error"
        ),
        "weights": {str(C): w[C] for C in B}, "bands_present": B,
        "weight_counts": {str(C): WEIGHT_COUNTS[C] for C in B},
        "weight_denominator": WEIGHT_DENOMINATOR,
        "weight_scope": (
            "exact normalized composition of C=27..30 within the first-three-million "
            "normal-record prefix"
        ),
        "estimand": (
            "prefix-composition-weighted combination of full-acquisition, post-first-4000-"
            "per-band eligible-population means"
        ),
        "inference_status": "historical exploratory point estimate with separated sensitivity scales",
        "reconstruction_layer": "not computed; result conditional on frozen point models",
        "scope": "conditional cross-entropy comparison of two frozen point models; "
                 "not absolute fit, not classical simulability, not quantum-feature absence",
    }


def stratified_delta_H_band(
    ln_ratio_mid: np.ndarray,
    ln_ratio_halfwidth: np.ndarray,
    strata: np.ndarray,
    eligible_by_stratum: np.ndarray,
    C: int,
    norms: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Finite-population weighted estimate for the realized equal-quota design."""
    mids = np.asarray(ln_ratio_mid, dtype=float)
    halfs = np.asarray(ln_ratio_halfwidth, dtype=float)
    strata = np.asarray(strata, dtype=int)
    eligible = np.asarray(eligible_by_stratum, dtype=np.int64)
    if not (len(mids) == len(halfs) == len(strata)):
        raise ValueError("event values, arithmetic proxies, and strata must align")
    if len(eligible) == 0 or np.any(eligible < 0) or int(eligible.sum()) <= 0:
        raise ValueError("eligible stratum populations are invalid")
    if np.any(strata < 0) or np.any(strata >= len(eligible)):
        raise ValueError("event stratum labels fall outside the eligible population table")
    if not np.all(np.isfinite(mids)) or not np.all(np.isfinite(halfs)):
        raise ValueError("event values and arithmetic proxies must be finite")
    if np.any(halfs < 0):
        raise ValueError("arithmetic proxy halfwidths must be non-negative")

    population_weights = eligible / float(eligible.sum())
    sample_counts = np.bincount(strata, minlength=len(eligible)).astype(np.int64)
    if int(sample_counts.sum()) != len(mids):
        raise AssertionError("stratum accounting dropped evaluated events")
    band_mean = 0.0
    arithmetic_proxy = 0.0
    event_variance = 0.0
    for stratum_index, population_n in enumerate(eligible):
        weight = float(population_weights[stratum_index])
        if weight == 0.0:
            continue
        values = mids[strata == stratum_index]
        widths = halfs[strata == stratum_index]
        if len(values) < 2:
            raise ValueError(
                f"stratum {stratum_index} requires at least two evaluated events"
            )
        if len(values) > population_n:
            raise ValueError("selected stratum count exceeds its eligible population")
        band_mean += weight * float(np.mean(values))
        arithmetic_proxy += weight * float(np.mean(widths))
        finite_population = 1.0 - len(values) / float(population_n)
        event_variance += (
            weight * weight * finite_population * float(np.var(values, ddof=1)) / len(values)
        )

    (prz, uncz), (pra, unca) = norms["squeezed"], norms["squashed"]
    shift = float(np.log(prz[C]) - np.log(pra[C]))
    point = band_mean + shift
    return {
        "C": C,
        "n_events": int(len(mids)),
        "eligible_population": int(eligible.sum()),
        "delta_H": point,
        "pattern_log_ratio": band_mean,
        "arithmetic_proxy": arithmetic_proxy,
        "event_se": float(np.sqrt(event_variance)),
        "normalizer_scale": float(np.hypot(uncz[C] / prz[C], unca[C] / pra[C])),
        "normalizer_shift": shift,
        "population_by_stratum": eligible.tolist(),
        "sample_by_stratum": sample_counts.tolist(),
        "population_weights": population_weights.tolist(),
        "estimator": "finite-population weighted mean for realized band-specific equal-quota strata",
        "squashed_wins": bool(point > 0),
    }


# --------------------------------------------------------------------------
def _slice_log(C: int, tag: str) -> Path:
    return OUT_DIR / f"confirmatory_C{C}{('_' + tag) if tag else ''}.jsonl"


def eval_event(states, S, k) -> dict:
    """Evaluate one historical event with the DD kernel enclosure.

    The bound applies to the torontonian of the supplied binary64 matrix. The
    state construction, determinant normalization, and transcendental steps are
    outside that kernel-level certificate.
    """
    import gbskernels

    probabilities, bounds = {}, {}
    seconds = 0.0
    for kind in q7.KINDS:
        state = states[kind]
        indices = list(S) + [j + 100 for j in S]
        submatrix = np.ascontiguousarray(state["O"][np.ix_(indices, indices)])
        started = time.time()
        value, diagnostic = gbskernels.tor_single(
            submatrix, groups=min(k, 14), dd=True
        )
        seconds += time.time() - started
        error = diagnostic["abs_error_bound"]
        if not (value > 0 and value - error > 0):
            return {"refused": True, "reason": "nonpositive_kernel_enclosure", "sec": seconds}
        scale = np.exp(-state["log_sqrt_detQ"])
        probabilities[kind] = float(value * scale)
        bounds[kind] = float(error * scale)

    slack = 2 * q7.LSDQ_SLACK
    squashed_lo = probabilities["squashed"] * (1 - slack) - bounds["squashed"]
    squashed_hi = probabilities["squashed"] * (1 + slack) + bounds["squashed"]
    squeezed_lo = probabilities["squeezed"] * (1 - slack) - bounds["squeezed"]
    squeezed_hi = probabilities["squeezed"] * (1 + slack) + bounds["squeezed"]
    endpoints = (squashed_lo, squashed_hi, squeezed_lo, squeezed_hi)
    if not all(np.isfinite(endpoint) and endpoint > 0 for endpoint in endpoints):
        return {"refused": True, "reason": "nonpositive_log_endpoint", "sec": seconds}
    lower = np.log(squashed_lo) - np.log(squeezed_hi)
    upper = np.log(squashed_hi) - np.log(squeezed_lo)
    return {
        "refused": False,
        "x_mid": float((lower + upper) / 2),
        "x_half": float((upper - lower) / 2),
        "sec": seconds,
    }


def run_slice(states, norms, C: int, ridx, pats, lo: int, hi: int, tag: str) -> None:
    """Evaluate events [lo:hi] of band C into a per-slice JSONL (resumable)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _slice_log(C, tag)
    done = set()
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            if r.get("event") is not None:
                done.add(r["event"])
    hi = min(hi, len(pats))
    todo = [i for i in range(lo, hi) if i not in done]
    print(f"  C={C} slice[{lo}:{hi}] tag={tag or '-'}: {len(done)} done, {len(todo)} to do", flush=True)
    t0 = time.time()
    with open(log_path, "a") as fh:
        for i in todo:
            S = np.flatnonzero(pats[i])
            r = eval_event(states, S, C)
            r["event"] = int(i)
            r["ridx"] = int(ridx[i])
            fh.write(json.dumps(r) + "\n")
            fh.flush()
    print(f"  C={C} slice[{lo}:{hi}] done ({time.time()-t0:.0f}s)", flush=True)


def load_band_merged(
    C: int,
    norms,
    selection,
    checkpoint_dir: Path | None = None,
) -> dict | None:
    """Merge every confirmatory_C{C}*.jsonl checkpoint by event index -> band row."""
    checkpoint_dir = checkpoint_dir or OUT_DIR
    by_i: dict[int, dict] = {}
    for f in sorted(checkpoint_dir.glob(f"confirmatory_C{C}.jsonl")) + \
             sorted(checkpoint_dir.glob(f"confirmatory_C{C}_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("event") is not None:
                event = int(r["event"])
                if event in by_i and by_i[event] != r:
                    raise ValueError(
                        f"conflicting checkpoint rows for band {C}, event {event}"
                    )
                by_i[event] = r
    if not by_i:
        return None
    ordered = sorted(by_i.items())
    ok = [(event, row) for event, row in ordered if not row.get("refused")]
    refused = sum(1 for r in by_i.values() if r.get("refused"))
    if refused:
        raise ValueError("the historical stratified estimate is undefined with refusals")
    mids = np.array([r["x_mid"] for _, r in ok])
    halfs = np.array([r["x_half"] for _, r in ok])
    if f"stratum_C{C}" not in selection or f"eligible_C{C}" not in selection:
        raise ValueError(
            "selection manifest lacks stratum populations; regenerate it with "
            "select_confirmatory.py before aggregating"
        )
    event_indices = np.asarray([event for event, _ in ok], dtype=np.int64)
    selected_strata = np.asarray(selection[f"stratum_C{C}"], dtype=np.int64)
    selected_ridx = np.asarray(selection[f"ridx_C{C}"], dtype=np.int64)
    if np.any(event_indices < 0) or np.any(event_indices >= len(selected_strata)):
        raise ValueError("checkpoint event indices exceed the selection manifest")
    for event, checkpoint in ok:
        if int(checkpoint.get("ridx", -1)) != int(selected_ridx[event]):
            raise ValueError(
                f"checkpoint record index differs from selection for band {C}, event {event}"
            )
    row = stratified_delta_H_band(
        mids,
        halfs,
        selected_strata[event_indices],
        np.asarray(selection[f"eligible_C{C}"], dtype=np.int64),
        C,
        norms,
    )
    row.update({
        "n_evaluated": len(by_i),
        "n_refused": int(refused),
        "selection_complete": bool(
            np.array_equal(event_indices, np.arange(len(selected_ridx), dtype=np.int64))
        ),
    })
    return row


def _selftest() -> int:
    rows = {
        27: {"delta_H": 0.018, "event_se": 0.008, "normalizer_scale": 0.004, "arithmetic_proxy": 6e-6},
        28: {"delta_H": 0.035, "event_se": 0.011, "normalizer_scale": 0.004, "arithmetic_proxy": 9e-6},
        29: {"delta_H": 0.035, "event_se": 0.013, "normalizer_scale": 0.003, "arithmetic_proxy": 7e-5},
        30: {"delta_H": 0.032, "event_se": 0.018, "normalizer_scale": 0.003, "arithmetic_proxy": 2.5e-4},
    }
    agg = aggregate(rows)
    w = WEIGHTS
    assert abs(agg["Delta_B"] - sum(w[C] * rows[C]["delta_H"] for C in BANDS)) < 1e-12
    assert agg["arithmetic_proxy_displacement"] < agg["event_se"]
    assert agg["normalizer_diagonal_sensitivity_scale"] > 0
    print("[selftest] Delta_B=%.4f event SE=%.4f norm(diagonal sensitivity)=%.4f "
          "arith(proxy)=%.1e OK"
          % (agg["Delta_B"], agg["event_se"],
             agg["normalizer_diagonal_sensitivity_scale"],
             agg["arithmetic_proxy_displacement"]))
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.absolute()
    try:
        return str(resolved.relative_to(HERE.parents[1].absolute()))
    except ValueError:
        return str(resolved)


def do_aggregate(
    manifest_path: Path,
    manifest_meta: dict,
    selection,
    output_path: Path | None = None,
    checkpoint_dir: Path | None = None,
) -> int:
    checkpoint_dir = checkpoint_dir or OUT_DIR
    norms = their_normalizers()
    band_rows = {}
    targets = {int(k): v for k, v in (manifest_meta.get("targets") or {}).items()}
    for C in BANDS:
        row = load_band_merged(C, norms, selection, checkpoint_dir)
        if row is None:
            print(f"  C={C}: no checkpoints yet", flush=True)
            continue
        band_rows[C] = row
        tgt = targets.get(C, "?")
        print(f"  C={C}: {row['n_events']}/{tgt} evaluated ({row['n_refused']} refused)  "
              f"dH={row['delta_H']:+.4f}  event {row['event_se'] or float('nan'):.4f}  "
              f"norm-scale {row['normalizer_scale']:.4f}  "
              f"arith-proxy {row['arithmetic_proxy']:.1e}", flush=True)
    complete = all(
        C in band_rows
        and band_rows[C]["selection_complete"]
        and band_rows[C]["n_events"] == targets.get(C)
        for C in BANDS
    ) if targets else False
    agg = aggregate(band_rows)
    agg["complete"] = bool(complete)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = output_path or OUT_DIR / f"private_fixed_sample_result_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = sorted(
        {
            path
            for C in BANDS
            for path in checkpoint_dir.glob(f"confirmatory_C{C}*.jsonl")
        }
    )
    normalizer_paths = [
        q7.ZEN / "click_probs" / "click_probs_squeezed_0.npy",
        q7.ZEN / "click_probs" / "click_probs_squashed_0.npy",
    ]
    histogram_path = HERE / "click_count_dist.npy"
    try:
        from bench._provenance import provenance

        aggregate_provenance = provenance()
    except Exception as exc:
        aggregate_provenance = {"capture_error": str(exc)}
    out.write_text(json.dumps({
        "kind": "jiuzhang1_private_fixed_sample_legacy_result",
        "schema": "gbskernels.jiuzhang1-private-fixed-sample.v2",
        "inference_status": "historical exploratory private fixed sample; not held-out or confirmatory",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": aggregate_provenance,
        "inputs": {
            "selection_manifest": {
                "path": _portable_path(manifest_path), "sha256": _sha256(manifest_path)
            },
            "checkpoints": [
                {
                    "path": _portable_path(path),
                    "sha256": _sha256(path),
                }
                for path in checkpoint_paths
            ],
            "normalizers": {
                "files": [
                    {"path": _portable_path(path), "sha256": _sha256(path)}
                    for path in normalizer_paths
                ],
                "uncertainty_convention": (
                    "across-group standard deviation of 100 group means; "
                    "not divided by sqrt(100)"
                ),
                "cross_band_covariance": "unavailable",
            },
            "band_weight_histogram": {
                "path": _portable_path(histogram_path),
                "sha256": _sha256(histogram_path),
                "prefix_records": 3_000_000,
                "normal_records": 2_995_852,
                "selected_band_counts": {
                    str(C): WEIGHT_COUNTS[C] for C in BANDS
                },
                "normalization": "within C=27..30 only",
            },
        },
        "aggregation_implementation": {
            _portable_path(Path(__file__)): _sha256(Path(__file__)),
        },
        "historical_event_implementation": {
            "status": "unresolved",
            "reason": "checkpoint rows did not record GPU, container, extension, or source identity",
        },
        "unresolved_provenance": [
            "historical per-event GPU/container/extension identity was not recorded",
            "historical event rows do not retain separate torontonian values and bounds",
        ],
        "complete": bool(complete), "manifest_meta": manifest_meta,
        "bands": band_rows, "aggregate": agg,
    }, indent=1))
    tag = "" if complete else "  [PARTIAL -- not all N_C evaluated]"
    print(f"\n== HISTORICAL PRIVATE FIXED-SAMPLE {'RESULT' if complete else 'PARTIAL'} =={tag}")
    print(f"  Delta_B = {agg['Delta_B']:+.5f}   event SE={agg['event_se']:.5f}  "
          f"normalizer diagonal sensitivity="
          f"{agg['normalizer_diagonal_sensitivity_scale']:.5f}")
    print(f"  arithmetic proxy displacement: +/-{agg['arithmetic_proxy_displacement']:.1e}")
    print(f"  -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="historical selection NPZ")
    ap.add_argument("--bands", help="comma list of bands to run this box, e.g. 27,28 (default: all)")
    ap.add_argument("--slice", help="lo:hi event range (only with a single --bands value)")
    ap.add_argument("--tag", default="", help="output-filename suffix so parallel slices don't collide")
    ap.add_argument("--aggregate", action="store_true", help="merge all checkpoints -> Delta_B (CPU, no GPU)")
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="checkpoint JSONL directory for --aggregate (default: results/jiuzhang)",
    )
    ap.add_argument("--out", type=Path, help="aggregate JSON path")
    ap.add_argument("--selftest", action="store_true", help="CPU aggregation self-test")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if os.environ.get("GBS_ALLOW_LEGACY_CONFIRMATORY") != "1":
        raise SystemExit(
            "legacy private fixed-sample runner disabled; use the public v2 workflow "
            "(docs/confirmatory_v2.md). Set GBS_ALLOW_LEGACY_CONFIRMATORY=1 "
            "only for historical audit reproduction."
        )
    if not args.manifest:
        ap.error("--manifest required (or --selftest)")
    manifest_path = Path(args.manifest)
    z = np.load(manifest_path, allow_pickle=False)
    meta = json.loads(str(z["meta"])) if "meta" in z else {}

    if args.aggregate:
        return do_aggregate(
            manifest_path,
            meta,
            z,
            args.out,
            checkpoint_dir=args.checkpoint_dir,
        )

    bands = [int(b) for b in args.bands.split(",")] if args.bands else list(BANDS)
    lo, hi = 0, 10**9
    if args.slice:
        assert len(bands) == 1, "--slice requires exactly one --bands value"
        lo, hi = (int(x) for x in args.slice.split(":"))
    print(f"manifest seed={meta.get('seed')} | this box runs bands {bands}"
          + (f" slice[{lo}:{hi}]" if args.slice else "") + f" tag={args.tag or '-'}", flush=True)
    states = {k: q7.build_state(k) for k in q7.KINDS}
    norms = their_normalizers()
    for C in bands:
        pats, ridx = z[f"pats_C{C}"], z[f"ridx_C{C}"]
        run_slice(states, norms, C, ridx, pats, lo, hi, args.tag)
    print("this box's slice(s) done; aggregate centrally after pulling all checkpoints.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
