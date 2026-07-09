"""On-device validation of the certified double-double single-large torontonian.

Confirms, on real GPU hardware: (1) the DD kernel compiles under nvcc and runs;
(2) it encloses the mpmath reference where affordable; (3) it is strictly tighter
than the fp64 certificate; (4) on the real Jiuzhang state it stays certifiable
(rel bound < 1) across and past the click range where the fp64 certificate is
already meaningless. Fast on a GPU (the shim could not reach the high-k points).
"""
import argparse, json, os, time
import numpy as np
from sampling import gbs as gbs_mod
import gbskernels
from highprec_ref import torontonian_mp

def build_O():
    T = np.load("data/jiuzhang1/T_full.npy")
    r = np.repeat(np.loadtxt("data/jiuzhang1/squeezing parameters.txt"), 2)
    nb = np.sinh(r) ** 2; mm = np.sinh(r) * np.cosh(r)
    aiaj = T.T @ np.diag(mm) @ T; aidaj = T.conj().T @ np.diag(nb) @ T; M = 100
    x = np.eye(M) + np.real(aidaj + aidaj.conj().T) + 2 * np.real(aiaj)
    p = np.eye(M) + np.real(aidaj + aidaj.conj().T) - 2 * np.real(aiaj)
    xp = 2 * np.imag(aiaj) + 2 * np.imag(aidaj)
    cov = np.block([[x, xp], [xp.T, p]]); Q = gbs_mod._qmat(cov)
    return np.real(np.eye(200) - np.linalg.inv(Q))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=40)
    ap.add_argument("--kmax", type=int, default=28)
    args = ap.parse_args()
    print("backend:", gbskernels.gpu_backend_kind(), flush=True)
    # (1) on-device enclosure gate on a small physical instance
    from bench._inputs import physical_torontonian
    O5 = np.real(physical_torontonian(6, 3))
    vd, dd = gbskernels.tor_single(O5, groups=3, dd=True)
    ex = complex(torontonian_mp(O5, dps=60)).real
    vf, df = gbskernels.tor_single(O5, groups=3, certified=True)
    assert abs(vd - ex) <= dd["abs_error_bound"], "DD enclosure FAILED"
    assert dd["rel_error_bound"] < df["rel_error_bound"], "DD not tighter"
    print(f"gate: DD encloses mpmath (|err|={abs(vd-ex):.1e}<=bound {dd['abs_error_bound']:.1e}); "
          f"DD rel {dd['rel_error_bound']:.1e} < fp64 {df['rel_error_bound']:.1e}  OK", flush=True)

    # (2) DD frontier on the real Jiuzhang state, now reaching high k on-device
    O = build_O()
    events = np.load("data/jiuzhang1/events_ge40.npy")[:args.events]
    ks = [k for k in [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28] if k <= args.kmax]
    fp = {k: [] for k in ks}; ddb = {k: [] for k in ks}; encl_fail = 0
    for c in events:
        S = [j for j in range(100) if c[j]]
        for k in ks:
            Sk = S[:k]; idx = Sk + [j + 100 for j in Sk]
            sub = np.ascontiguousarray(O[np.ix_(idx, idx)])
            vf, df = gbskernels.tor_single(sub, groups=min(k, 13), certified=True)
            vd, dd = gbskernels.tor_single(sub, groups=min(k, 13), dd=True)
            if k <= 12:
                exk = complex(torontonian_mp(sub, dps=60)).real
                if abs(vd - exk) > dd["abs_error_bound"]:
                    encl_fail += 1
            if vf != 0: fp[k].append(df["rel_error_bound"])
            if vd != 0: ddb[k].append(dd["rel_error_bound"])
    rows = [{"clicks": k, "dim": 2 * k, "n": len(ddb[k]),
             "fp64_rel_bound_median": float(np.median(fp[k])),
             "dd_rel_bound_median": float(np.median(ddb[k])),
             "dd_certifies": bool(np.median(ddb[k]) < 0.1)} for k in ks]
    dd_frontier = max((r["clicks"] for r in rows if r["dd_certifies"]), default=0)
    art = {"kind": "jiuzhang1_dd_frontier_gpu",
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "commit": os.environ.get("GBS_COMMIT"),
           "container_digest": os.environ.get("GBS_CONTAINER_DIGEST"),
           "gpu_backend": gbskernels.gpu_backend_kind(),
           "n_events": len(events), "enclosure_failures_le12": encl_fail,
           "dd_tight_frontier_clicks": dd_frontier, "rows": rows}
    out = f"data/jiuzhang1/dd_frontier_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    json.dump(art, open(out, "w"), indent=1)
    print(f"\n{'k':>3} {'fp64 rel':>10} {'DD rel':>10} {'DD certifies?':>14}")
    for r in rows:
        print(f"{r['clicks']:>3} {r['fp64_rel_bound_median']:>10.1e} "
              f"{r['dd_rel_bound_median']:>10.1e} {str(r['dd_certifies']):>14}")
    print(f"enclosure failures (k<=12): {encl_fail} (must be 0)")
    print(f"DD tightly-certifiable frontier: {dd_frontier} clicks (dim {2*dd_frontier})")
    print(f"-> {out}")

if __name__ == "__main__":
    main()
