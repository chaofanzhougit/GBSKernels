"""Stage-1 certified campaign: conditioned Delta H(C) at and beyond the ceiling.

For each band C in the plan: per-event conditioned log-ratios built from DD
torontonian values and kernel-level bounds on frozen binary64 matrices
x_k = ln Pr_SQUA(s_k) - ln Pr_SQUE(s_k) + ln[Pr_SQUE(C)/Pr_SQUA(C)]
(frozen Q7-1076 point-model states via q7_construction; their published grouped-click
normalizers), evaluated event by event. Fixed caps are the default.
``--adaptive`` enables an explicitly exploratory repeated-look boundary:

  stop when n >= n_min and
  |Delta H(C)| - err_arith >= 3 * sqrt(SE_stat^2 + err_norm^2)
  or at the event cap or wall-clock budget. Crossing this boundary is not a
  fixed-sample 3-sigma result. Arithmetic enclosure width is subtracted from
  the observed effect before the boundary score is formed; refused events are
  counted, never imputed.

Crash safety: every evaluation appends one JSON line to
results/jiuzhang/campaign_C<band>.jsonl BEFORE statistics update, so a dead
box loses at most one evaluation; re-running skips already-recorded events
(idempotent resume from the same file).

    python examples/jiuzhang/campaign.py \
        --plan "21:60:900,22:60:900,23:60:1200,24:60:1800,25:60:2700,26:60:4200,27:400:15000,28:500:28000,29:150:17000,30:100:23000"
    python examples/jiuzhang/campaign.py --dry     # shim plumbing run

Bands 21-26 (fixed small n, no early stop) reproduce their published verdict
on OUR decoded events; bands 27-30 are new territory. The final artifact also
reports a descriptive pooled aggregate, marked invalid for confirmatory
inference whenever any contributing band stopped adaptively.

Artifact: results/jiuzhang/campaign_stage1_<UTC>.json (append-only).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import q7_construction as q7
from q7_conditioned import delta_H_band, their_normalizers

OUT_DIR = HERE.parents[1] / "results" / "jiuzhang"
N_MIN_STOP = 24         # no verdict from fewer events than this
CEILING = 26            # published exact-validation ceiling


def boundary_score(row: dict) -> float | None:
    """Nominal repeated-look score after worst-case arithmetic displacement.

    This is a campaign-control diagnostic, not an anytime-valid p-value.
    """
    if row["err_stat"] is None:
        return None
    denom = float(np.hypot(row["err_stat"], row["err_norm"]))
    effect = max(0.0, abs(row["delta_H"]) - row["err_arith"])
    if denom == 0.0:
        return float("inf") if effect > 0.0 else 0.0
    return float(effect / denom)


def first_boundary_crossing(mids: np.ndarray, halfs: np.ndarray, C: int,
                            norms, check_every: int = 8) -> int | None:
    """Replay the adaptive rule and return its first terminal prefix."""
    for n in range(N_MIN_STOP, len(mids) + 1):
        if n % check_every:
            continue
        row = delta_H_band(mids[:n], C, norms, halfs[:n])
        score = boundary_score(row)
        if score is not None and score >= 3.0:
            return n
    return None


def load_band_events(C: int) -> np.ndarray:
    z = np.load(q7.DATA / "campaign_events.npz")
    return z[f"C{C}"]


def eval_event(states, S, k) -> dict | None:
    """Conditioned interval diagnostic for one event, or a refusal.

    The DD kernel bounds the torontonian of the supplied binary64 matrix. The
    FP64 state construction, determinant normalization, and transcendental
    propagation are not yet an end-to-end probability certificate.
    """
    import gbskernels

    p, b = {}, {}
    sec = 0.0
    for kind in q7.KINDS:
        st = states[kind]
        idx = list(S) + [j + 100 for j in S]
        sub = np.ascontiguousarray(st["O"][np.ix_(idx, idx)])
        t0 = time.time()
        v, d = gbskernels.tor_single(sub, groups=min(k, 14), dd=True)
        sec += time.time() - t0
        E = d["abs_error_bound"]
        if not (v > 0 and v - E > 0):
            return {"refused": True, "sec": sec}
        p[kind] = float(v * np.exp(-st["log_sqrt_detQ"]))
        b[kind] = float(E * np.exp(-st["log_sqrt_detQ"]))
    slack = 2 * q7.LSDQ_SLACK
    lo = np.log(p["squashed"] * (1 - slack) - b["squashed"]) \
        - np.log(p["squeezed"] * (1 + slack) + b["squeezed"])
    hi = np.log(p["squashed"] * (1 + slack) + b["squashed"]) \
        - np.log(p["squeezed"] * (1 - slack) - b["squeezed"])
    return {"refused": False, "x_mid": float((lo + hi) / 2),
            "x_half": float((hi - lo) / 2), "sec": sec}


def run_band(states, norms, C: int, cap: int, budget_s: float,
             adaptive: bool, check_every: int = 8) -> dict:
    events = load_band_events(C)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / f"campaign_C{C}.jsonl"
    done: dict[int, dict] = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            if r.get("event") is not None:
                done[r["event"]] = r
    ordered = [r for _, r in sorted(done.items())]
    mids = [r["x_mid"] for r in ordered if not r.get("refused")]
    halfs = [r["x_half"] for r in ordered if not r.get("refused")]
    refused = sum(1 for r in done.values() if r.get("refused"))
    if done:
        print(f"  C={C}: resuming with {len(done)} recorded evaluations", flush=True)

    t0 = time.time()
    terminal_n = first_boundary_crossing(
        np.asarray(mids), np.asarray(halfs), C, norms, check_every
    ) if adaptive else None
    stopped = "exploratory-boundary" if terminal_n is not None else "cap"
    if terminal_n is not None:
        mids, halfs = mids[:terminal_n], halfs[:terminal_n]
    with open(log_path, "a") as fh:
        for i in range(min(cap, len(events))):
            if terminal_n is not None:
                break
            if i in done:
                continue
            if time.time() - t0 > budget_s:
                stopped = "budget"
                break
            S = np.flatnonzero(events[i])
            r = eval_event(states, S, C)
            r["event"] = int(i)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if r.get("refused"):
                refused += 1
                continue
            # mids stay UNSHIFTED everywhere (matching the JSONL and the
            # resume load); delta_H_band applies the normalizer shift itself
            mids.append(r["x_mid"])
            halfs.append(r["x_half"])
            n = len(mids)
            if adaptive and n >= N_MIN_STOP and n % check_every == 0:
                row = delta_H_band(np.array(mids), C, norms, np.array(halfs))
                score = boundary_score(row)
                if score is not None and score >= 3.0:
                    stopped = "exploratory-boundary"
                    break

    row = delta_H_band(np.array(mids), C, norms,
                       np.array(halfs) if halfs else None)
    row.update({"stopped_by": stopped, "n_refused": int(refused),
                "wall_seconds": time.time() - t0,
                "beyond_ceiling": bool(C > CEILING)})
    row["boundary_score"] = boundary_score(row)
    row["meets_exploratory_boundary"] = bool(
        row["boundary_score"] is not None and row["boundary_score"] >= 3.0
    )
    z_txt = f"{row['boundary_score']:.2f}" if row["boundary_score"] is not None else "--"
    print(f"  C={C}: n={row['n_events']} dH={row['delta_H']:+.4f} "
          f"stat {row['err_stat'] if row['err_stat'] else float('nan'):.4f} "
          f"norm {row['err_norm']:.4f} "
          f"arith {row['err_arith']:.1e}  boundary-score={z_txt} "
          f"[{stopped}] ({row['wall_seconds']:.0f}s)", flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=str,
                    default="21:60:900,22:60:900,23:60:1200,24:60:1800,"
                            "25:60:2700,26:60:4200,27:400:15000,28:500:28000,"
                            "29:150:17000,30:100:23000",
                    help="comma list C:cap:budget_seconds")
    ap.add_argument("--check-every", type=int, default=8)
    ap.add_argument("--adaptive", action="store_true",
                    help="enable exploratory repeated-look stopping above C=26")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    if args.dry:
        args.plan = "21:4:600"

    plan = []
    for kv in args.plan.split(","):
        C, cap, bud = kv.split(":")
        plan.append((int(C), int(cap), float(bud)))

    states = {k: q7.build_state(k) for k in q7.KINDS}
    norms = their_normalizers()
    import gbskernels
    backend = gbskernels.gpu_backend_kind() if hasattr(gbskernels, "gpu_backend_kind") else None
    print(f"stage-1 campaign on backend={backend}; plan {args.plan}", flush=True)

    rows = []
    for C, cap, bud in plan:
        adaptive = args.adaptive and C > CEILING
        rows.append(run_band(states, norms, C, cap, bud, adaptive,
                             args.check_every))

    # pooled beyond-ceiling aggregate (all events, band shifts already folded
    # into each x via delta_H_band's per-band handling -- recompute from the
    # JSONL so the pool survives resumes)
    pooled_x, pooled_half = [], []
    row_by_C = {r["C"]: r for r in rows}
    for C, cap, bud in plan:
        if C <= CEILING:
            continue
        (prz, _), (pra, _) = norms["squeezed"], norms["squashed"]
        shift = float(np.log(prz[C]) - np.log(pra[C]))
        p = OUT_DIR / f"campaign_C{C}.jsonl"
        if not p.exists():
            continue
        accepted = 0
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if not r.get("refused") and r.get("x_mid") is not None:
                if accepted >= row_by_C[C]["n_events"]:
                    break
                pooled_x.append(r["x_mid"] + shift)
                pooled_half.append(r["x_half"])
                accepted += 1
    pooled = None
    if len(pooled_x) >= 2:
        px = np.array(pooled_x)
        se = float(px.std(ddof=1) / np.sqrt(len(px)))
        norm_worst = max(
            float(np.hypot(norms["squeezed"][1][C] / norms["squeezed"][0][C],
                           norms["squashed"][1][C] / norms["squashed"][0][C]))
            for C, _, _ in plan if C > CEILING)
        tot = float(np.hypot(se, norm_worst))
        arith = float(np.mean(pooled_half))
        score = max(0.0, abs(float(px.mean())) - arith) / tot
        adaptively_selected = any(
            row_by_C[C]["stopped_by"] == "exploratory-boundary"
            for C, _, _ in plan if C > CEILING
        )
        pooled = {"n_events": len(px), "delta_H": float(px.mean()),
                  "err_stat": se, "err_norm_worst_band": norm_worst,
                  "err_arith": arith, "boundary_score": float(score),
                  "adaptively_selected": adaptively_selected,
                  "confirmatory_inference_valid": not adaptively_selected}
        print(f"  POOLED beyond-ceiling: n={pooled['n_events']} "
              f"dH={pooled['delta_H']:+.4f} boundary-score={pooled['boundary_score']:.2f} "
              f"({'descriptive: adaptive selection' if adaptively_selected else 'fixed caps'})",
              flush=True)

    from bench._provenance import provenance

    art = {
        "kind": "jiuzhang1_campaign_stage1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": provenance(),
        "gpu_backend": backend,
        "dry": bool(args.dry),
        "analysis_mode": "exploratory-adaptive" if args.adaptive else "fixed-caps",
        "plan": args.plan,
        "normalizers": "their published grouped Pr(C) with uncertainties "
                       "(zenodo 7141021; conservative across-group-std convention)",
        "bands": rows,
        "pooled_beyond_ceiling": pooled,
    }
    out = OUT_DIR / f"campaign_stage1_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    out.write_text(json.dumps(art, indent=1, default=float))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
