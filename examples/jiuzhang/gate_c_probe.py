"""Gate C: on-device cost probe + beyond-ceiling sigma sample (campaign sizing).

On the exact, parity-validated Q7-1076 states (q7_construction):

  timing  seconds/eval of tor_single certified-fp64 and certified-DD at
          k = 25..32, one whole real event per k -- the campaign cost curve,
          and the headline comparison point (their published operating point:
          ~3.5 h per single 25-click pattern on a 64-core node, double-double,
          no error bound).
  sigma   per-event click-conditioned log-ratio (squashed minus squeezed, the
          Delta H integrand) on whole real events BEYOND the published
          26-click ceiling: k = 27/28 (up to 16 events each) and k = 29/30
          (up to 6), DD-certified under both hypotheses, conditioned with
          their published Pr(C) normalizers -> per-band sigma and the events
          needed for a 3-sigma proven-sign verdict.  No published work can
          supply this number; it sizes the campaign.

Budget guard: --budget-seconds (default 2400).  Every block times its first
evaluation and skips whatever the projection says will not fit; anything
dropped is COUNTED in the artifact, never silently truncated.

    python examples/jiuzhang/gate_c_probe.py                 # on the GPU box
    python examples/jiuzhang/gate_c_probe.py --dry           # shim plumbing run

Artifact: results/jiuzhang/gate_c_probe_<UTC>.json (append-only).
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

OUT_DIR = HERE.parents[1] / "results" / "jiuzhang"


def _normalizers():
    pr = {}
    for kind, stem in (("squeezed", "squeezed"), ("squashed", "squashed")):
        arr, unc = np.load(q7.ZEN / f"click_probs/click_probs_{stem}_0.npy")
        pr[kind] = (arr, unc)
    return pr


def _sub(O, S):
    idx = list(S) + [j + 100 for j in S]
    return np.ascontiguousarray(O[np.ix_(idx, idx)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget-seconds", type=float, default=2400.0)
    ap.add_argument("--timing-ks", type=int, nargs="*", default=list(range(25, 33)))
    ap.add_argument("--sigma-plan", type=str, default="27:16,28:16,29:6,30:6",
                    help="comma list k:events for the beyond-ceiling sigma sample")
    ap.add_argument("--dry", action="store_true",
                    help="shim plumbing run: tiny k, tiny counts")
    args = ap.parse_args()
    if args.dry:
        # the band file's decoded events start at 21 clicks; keep the shim run
        # small and let the budget guard demonstrably cut the second band
        args.timing_ks = [21, 22]
        args.sigma_plan = "21:2,22:2"
        args.budget_seconds = min(args.budget_seconds, 300.0)

    import gbskernels

    t_start = time.time()
    remaining = lambda: args.budget_seconds - (time.time() - t_start)

    states = {k: q7.build_state(k) for k in q7.KINDS}
    Ox = {k: states[k]["O"] for k in q7.KINDS}
    lsd = {k: states[k]["log_sqrt_detQ"] for k in q7.KINDS}
    norms = _normalizers()

    band = np.load(q7.DATA / "events_band13_32.npy")
    kk = band.sum(1)

    def events_at(k, n):
        rows = np.where(kk == k)[0][:n]
        return [np.flatnonzero(band[ri]) for ri in rows]

    backend = gbskernels.gpu_backend_kind() if hasattr(gbskernels, "gpu_backend_kind") else None
    print(f"gate C probe on backend={backend}; budget {args.budget_seconds:.0f}s", flush=True)

    # ---- timing block -------------------------------------------------------
    timing = []
    skipped_timing = []
    last = {"fp64": None, "dd": None}  # (k, seconds)
    for k in args.timing_ks:
        evs = events_at(k, 1)
        if not evs:
            skipped_timing.append({"k": k, "reason": "no event at this k"})
            print(f"  timing k={k}: skipped (no event at this k)", flush=True)
            continue
        S = evs[0]
        row = {"k": int(k), "dim": 2 * int(k)}
        for tier, kw in (("fp64", {"certified": True}), ("dd", {"dd": True})):
            proj = last[tier][1] * 2.0 ** (k - last[tier][0]) if last[tier] else 0.0
            if proj > remaining():
                skipped_timing.append({"k": k, "tier": tier, "reason":
                                       f"projected {proj:.0f}s > remaining {remaining():.0f}s"})
                print(f"  timing k={k} {tier}: skipped (projected {proj:.0f}s > "
                      f"remaining {remaining():.0f}s)", flush=True)
                continue
            t0 = time.time()
            v, d = gbskernels.tor_single(_sub(Ox["squeezed"], S),
                                         groups=min(k, 14), **kw)
            dt = time.time() - t0
            row[f"seconds_{tier}"] = dt
            row[f"rel_bound_{tier}"] = float(d["rel_error_bound"])
            last[tier] = (k, dt)
            print(f"  timing k={k} {tier}: {dt:.2f}s  rel bound {d['rel_error_bound']:.1e}",
                  flush=True)
        timing.append(row)

    # ---- beyond-ceiling sigma block ----------------------------------------
    sigma_rows = []
    plan = [(int(a), int(b)) for a, b in
            (kv.split(":") for kv in args.sigma_plan.split(","))]
    for k, n_want in plan:
        evs = events_at(k, n_want)
        (prz, uncz), (pra, unca) = norms["squeezed"], norms["squashed"]
        shift = float(np.log(prz[k]) - np.log(pra[k]))
        per_dd = last["dd"][1] * 2.0 ** (k - last["dd"][0]) if last["dd"] else 1.0
        mids, halfs, secs, refused = [], [], 0.0, 0
        done = 0
        for S in evs:
            if 2.2 * per_dd > remaining():   # both hypotheses + slack
                break
            p, b = {}, {}
            ok = True
            for kind in q7.KINDS:
                t0 = time.time()
                v, d = gbskernels.tor_single(_sub(Ox[kind], S), groups=min(k, 14), dd=True)
                secs += time.time() - t0
                E = d["abs_error_bound"]
                if not (v > 0 and v - E > 0):
                    ok = False
                    break
                p[kind] = v * np.exp(-lsd[kind])
                b[kind] = E * np.exp(-lsd[kind])
            if not ok:
                refused += 1
                continue
            slack = 2 * q7.LSDQ_SLACK
            lo = np.log(p["squashed"] * (1 - slack) - b["squashed"]) \
                - np.log(p["squeezed"] * (1 + slack) + b["squeezed"])
            hi = np.log(p["squashed"] * (1 + slack) + b["squashed"]) \
                - np.log(p["squeezed"] * (1 - slack) - b["squeezed"])
            mids.append((lo + hi) / 2 + shift)
            halfs.append((hi - lo) / 2)
            done += 1
            per_dd = secs / max(1, 2 * done)
        row = {"k": int(k), "n_wanted": int(n_want), "n_available": len(evs),
               "n_done": done, "n_refused": refused, "seconds": secs,
               "err_norm": float(np.hypot(uncz[k] / prz[k], unca[k] / pra[k])),
               "normalizer_shift": shift}
        if done >= 2:
            m = np.array(mids)
            row.update({
                "delta_H_estimate": float(m.mean()),
                "sigma_per_event": float(m.std(ddof=1)),
                "err_arith": float(np.mean(halfs)),
                "N_3sigma_at_0.027": int(np.ceil((3 * m.std(ddof=1) / 0.027) ** 2)),
            })
            print(f"  sigma k={k}: n={done} dH~{m.mean():+.3f} sigma={m.std(ddof=1):.3f} "
                  f"arith {np.mean(halfs):.1e} N(3sig@0.027)={row['N_3sigma_at_0.027']} "
                  f"({secs:.0f}s)", flush=True)
        elif not evs:
            print(f"  sigma k={k}: no events at this k in the band file", flush=True)
        else:
            print(f"  sigma k={k}: only {done} events fit the budget "
                  f"({refused} refused); recorded without statistics", flush=True)
        sigma_rows.append(row)

    from bench._provenance import provenance

    art = {
        "kind": "jiuzhang1_gate_c_probe",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": provenance(),
        "gpu_backend": backend,
        "dry": bool(args.dry),
        "budget_seconds": args.budget_seconds,
        "wall_seconds": time.time() - t_start,
        "their_operating_point": "~3.5 h per single 25-click pattern, 64-core "
                                 "dual AMD Rome, DoubleFloats.jl DD, no bound "
                                 "(Quantum 7, 1076, Sec. 3.2)",
        "timing": timing,
        "timing_skipped": skipped_timing,
        "sigma": sigma_rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"gate_c_probe_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    out.write_text(json.dumps(art, indent=1, default=float))
    print(f"-> {out}  (wall {art['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
