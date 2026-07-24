"""Q7-1076 parity gates + certified cross-validation against published values.

Campaign precondition 1 (Gate A report, next action 1): before quoting any
their-statistic number, prove construction and pipeline parity with
Martinez-Cifuentes et al., Quantum 7, 1076, against their published data
(zenodo DOI 10.5281/zenodo.7141021 -> record 7194775, in data/q7_1076_zenodo):

  G1  input parity      their sq_par.txt / t_matrix.mtx are bit-identical to
                        the USTC inputs our loader uses (T transposed).
  G2  construction      sigma_OUT from Eqs. (3)-(13) reproduces their
                        published out_cov_{squeezed,squashed}_0.npy to
                        <= 1e-12; the swapped sign alternation must differ by
                        > 1e-2 (the gate discriminates).
  G3  moment route      the Gate A pilot's cross-<aa> moment construction
                        (build_state(tmss_pairs=True)) equals the verbatim
                        xxpp route for both hypotheses (so campaign code may
                        use the moment machinery).
  G4  observables       exact (no-MC) Cbar and sigma(C) match their Table 2
                        within 4x their stated MC uncertainty, both
                        hypotheses; marginal RMS vs empirical rates recorded.
  G5  cross-validation  DD-certified probabilities of THEIR patterns lie
                        within our proven enclosure of THEIR published
                        values (plus log-sqrt-det slack and fp64 storage
                        slack); mpmath enclosure spot-gate at k=10.
  G6  legacy cast       the pre-parity scripts' O = Re(I - Q^{-1}) is a
                        DIFFERENT matrix (max|imag O| ~ 0.3 here); record how
                        far its values sit from the published ones so the bug
                        this work uncovered stays measured and cannot creep
                        back.  The exact real input is
                        q7_construction.threshold_O_xxpp.

    uv run python examples/jiuzhang/q7_parity.py --procs 8
    uv run python examples/jiuzhang/q7_parity.py --quick          # G1-G4 only

Artifact: q7_parity_<UTC>.json next to this script (append-only convention).
Exit status: 0 all gates green, 2 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import q7_construction as q7

TABLE2 = {  # Q7-1076 Table 2 (MC estimates): value, MC sigma
    "squeezed": {"Cbar": (41.042, 0.007), "sigmaC": (6.509, 0.022)},
    "squashed": {"Cbar": (41.162, 0.007), "sigmaC": (6.343, 0.022)},
}
FP_STORE_SLACK = 1e-13  # their values are DD results stored as float64

# Their probs_sqz_0_clicks_26.npy is PERMUTED relative to the patterns file
# (regenerated 2022-10-13; cross-hypothesis corr +0.10 vs +0.997 elsewhere --
# see q7_conditioned.band_pairing_diagnostics).  Per-index comparison is
# meaningless there; validate by NEAREST-VALUE set membership instead, with
# the second-nearest gap recorded so an accidental collision would be visible.
SET_MATCH = {("squeezed", 26)}


# ------------------------------------------------------------------- G1-G4
def gate_inputs() -> dict:
    r_theirs = np.loadtxt(q7.ZEN / "sq_parameters/sq_par.txt")
    r_ours = q7.load_r25()
    from scipy.io import mmread

    T_theirs = np.asarray(mmread(str(q7.ZEN / "transfer_matrices/t_matrix.mtx")))
    T_ours = q7.load_T_out_by_in()
    dr = float(np.abs(r_theirs - r_ours).max())
    dT = float(np.abs(T_theirs - T_ours).max())
    ok = dr == 0.0 and dT == 0.0
    return {"max_dr": dr, "max_dT": dT, "pass": ok}


def gate_construction() -> dict:
    out = {}
    ok = True
    for kind in q7.KINDS:
        ours = q7.build_cov(kind)
        theirs = np.load(q7.ZEN / f"covariance_matrices/out_cov_{kind}_0.npy")
        d = float(np.abs(ours - theirs).max())
        out[kind] = d
        ok &= d <= 1e-12
    swapped = q7.build_cov("squeezed", sign_order="swapped")
    theirs = np.load(q7.ZEN / "covariance_matrices/out_cov_squeezed_0.npy")
    d_sw = float(np.abs(swapped - theirs).max())
    out["swapped_sign_diff"] = d_sw
    ok &= d_sw > 1e-2
    out["pass"] = bool(ok)
    return out


def gate_moment_route() -> dict:
    from gate_a_pilot import build_state as pilot_build

    T_km = np.load(q7.DATA / "T_full.npy")  # pilot orientation: (K x M)
    r50 = np.repeat(q7.load_r25(), 2)
    out = {}
    ok = True
    for kind, (nb, mm) in {
        "squeezed": (np.sinh(r50) ** 2, np.sinh(r50) * np.cosh(r50)),
        "squashed": (np.sinh(r50) ** 2, np.sinh(r50) ** 2),
    }.items():
        cov_moment = pilot_build(T_km, nb, mm, tmss_pairs=True)[0]
        cov_verbatim = q7.build_cov(kind)
        d = float(np.abs(cov_moment - cov_verbatim).max())
        out[kind] = d
        ok &= d <= 1e-12
    out["pass"] = bool(ok)
    return out


def gate_observables(states: dict) -> dict:
    emp = np.load(q7.DATA / "empirical_click_rates.npy")
    out = {}
    ok = True
    for kind, st in states.items():
        p, cbar, sc = q7.click_statistics(st["Q"])
        (cb_t, cb_s), (sc_t, sc_s) = TABLE2[kind]["Cbar"], TABLE2[kind]["sigmaC"]
        pass_k = abs(cbar - cb_t) <= 4 * cb_s and abs(sc - sc_t) <= 4 * sc_s
        out[kind] = {
            "Cbar": cbar,
            "sigmaC": sc,
            "table2_Cbar": cb_t,
            "table2_sigmaC": sc_t,
            "marginal_rms_vs_empirical": float(np.sqrt(np.mean((p - emp) ** 2))),
            "mean_photons_out": st["mean_photons_out"],
            "pass": bool(pass_k),
        }
        ok &= pass_k
    out["pass"] = bool(ok)
    return out


# ---------------------------------------------------------------- G5 workers
_W: dict = {}


def _init(payload):
    global _W
    _W = payload


def _one(task):
    import gbskernels

    which, kind, k, i = task
    O = _W[f"O_{which}_{kind}"]
    lsd = _W[f"lsd_{kind}"]
    pats = _W[f"pats_{k}"]
    S = np.flatnonzero(pats[i])
    idx = list(S) + [j + 100 for j in S]
    sub = np.ascontiguousarray(O[np.ix_(idx, idx)])
    t0 = time.time()
    v, d = gbskernels.tor_single(sub, groups=min(k, 13), dd=True)
    return {
        "which": which, "kind": kind, "k": k, "event": int(i),
        "value": float(v), "abs_bound": float(d["abs_error_bound"]),
        "rel_bound": float(d["rel_error_bound"]),
        "prob": float(v * np.exp(-lsd)),
        "prob_bound": float(d["abs_error_bound"] * np.exp(-lsd)),
        "seconds": time.time() - t0,
    }


def gate_crossval(states: dict, bands: dict[int, int], legacy_events: int,
                  procs: int) -> tuple[dict, dict]:
    from highprec_ref import torontonian_mp

    payload: dict = {}
    for kind, st in states.items():
        payload[f"O_exact_{kind}"] = st["O"]
        payload[f"O_legacy_{kind}"] = st["O_legacy_recast"]
        payload[f"lsd_{kind}"] = st["log_sqrt_detQ"]
    their = {}
    tasks = []
    for k, n in bands.items():
        pats = np.load(q7.ZEN / f"pattern_probs/patterns_exp/samples_0_clicks_{k}.npy")
        payload[f"pats_{k}"] = pats
        their[("squeezed", k)] = np.load(q7.ZEN / f"pattern_probs/probs_sqz_0_clicks_{k}.npy")
        their[("squashed", k)] = np.load(q7.ZEN / f"pattern_probs/probs_sqs_0_clicks_{k}.npy")
        for kind in q7.KINDS:
            tasks += [("exact", kind, k, i) for i in range(min(n, len(pats)))]
    kmin = min(bands)
    for kind in q7.KINDS:  # G6: legacy cast, cheapest band only
        tasks += [("legacy", kind, kmin, i) for i in range(legacy_events)]

    # mpmath enclosure spot-gate at k=10 (exact matrix, both hypotheses)
    pats0 = payload[f"pats_{kmin}"]
    S10 = list(np.flatnonzero(pats0[0])[:10])
    idx10 = S10 + [j + 100 for j in S10]
    import gbskernels

    spot_ok = True
    for kind in q7.KINDS:
        sub = np.ascontiguousarray(states[kind]["O"][np.ix_(idx10, idx10)])
        v, d = gbskernels.tor_single(sub, groups=10, dd=True)
        ex = complex(torontonian_mp(sub, dps=50)).real
        spot_ok &= abs(v - ex) <= d["abs_error_bound"]
    if not spot_ok:
        raise AssertionError("mpmath enclosure spot-gate FAILED")
    print(f"mpmath enclosure spot-gate (k=10, both hypotheses): OK", flush=True)

    heavy = sum(2 ** t[2] for t in tasks)
    print(f"{len(tasks)} certified evaluations queued "
          f"(~{heavy / 2**21 * 100 / procs:.0f}+ s wall estimated on {procs} procs)",
          flush=True)
    t0 = time.time()
    results = []
    if procs == 1:
        # serial in-process: the GPU path must NOT be exercised from a forked
        # worker after the parent initialized CUDA (the spot-gate above did)
        _init(payload)
        pool, stream = None, map(_one, tasks)
    else:
        pool = Pool(procs, initializer=_init, initargs=(payload,))
        stream = pool.imap_unordered(_one, tasks, chunksize=1)
    try:
        for r in stream:
            tag = "sqz" if r["kind"] == "squeezed" else "sqs"
            arr = their[(r["kind"], r["k"])]
            if (r["kind"], r["k"]) in SET_MATCH:
                order = np.argsort(np.abs(arr - r["prob"]))
                th = float(arr[order[0]])
                r["match"] = "nearest-set"
                r["second_nearest_rel_gap"] = float(
                    abs(arr[order[1]] - r["prob"]) / max(r["prob"], 1e-300))
            else:
                th = float(arr[r["event"]])
                r["match"] = "index"
            r["their_value"] = th
            if r["which"] == "exact":
                tol = r["prob_bound"] + th * (2 * q7.LSDQ_SLACK + FP_STORE_SLACK)
                r["pass"] = bool(abs(r["prob"] - th) <= tol)
                print(f"  exact  {tag} k={r['k']} ev{r['event']}: "
                      f"ours {r['prob']:.9e}  theirs {th:.9e}  "
                      f"|diff|/theirs {abs(r['prob'] - th) / th:.1e}  "
                      f"relb {r['rel_bound']:.1e}  "
                      f"{'ENCLOSED' if r['pass'] else '*** OUTSIDE ***'} "
                      f"({r['seconds']:.0f}s)", flush=True)
            else:
                r["log_ratio_vs_their"] = float(np.log(r["prob"] / th))
                print(f"  legacy {tag} k={r['k']} ev{r['event']}: "
                      f"value {r['prob']:.3e} vs theirs {th:.3e}  "
                      f"(x{r['prob'] / th:.3f}, wrong matrix -- documented)",
                      flush=True)
            results.append(r)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    exact = [r for r in results if r["which"] == "exact"]
    legacy = [r for r in results if r["which"] == "legacy"]
    n_pass = sum(r["pass"] for r in exact)
    g5 = {
        "n": len(exact), "n_enclosed": n_pass,
        "worst_rel_diff": max(abs(r["prob"] - r["their_value"]) / r["their_value"]
                              for r in exact),
        "median_rel_bound": float(np.median([r["rel_bound"] for r in exact])),
        "mpmath_spot_gate": "ok",
        "wall_seconds": time.time() - t0,
        "pass": bool(n_pass == len(exact)),
        "rows": exact,
    }
    ratios = [r["prob"] / r["their_value"] for r in legacy]
    g6 = {
        "n": len(legacy),
        "value_ratio_range": [float(min(ratios)), float(max(ratios))] if ratios else None,
        "max_imag_O_legacy": {k: states[k]["max_imag_O_legacy"] for k in states},
        # the legacy cast must remain measurably wrong; if this ever "passes"
        # the state build changed and the gate needs rethinking.  With
        # --legacy-events 0 the check is skipped and does not gate.
        "pass": bool(all(abs(np.log(x)) > 1.0 for x in ratios)) if ratios else True,
        "skipped": not ratios,
        "rows": legacy,
    }
    return g5, g6


# ----------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--bands", type=int, nargs="*", default=[21, 22, 23])
    ap.add_argument("--per-band", type=int, default=4)
    ap.add_argument("--legacy-events", type=int, default=2,
                    help="G6 legacy-cast evaluations at the cheapest band, per hypothesis")
    ap.add_argument("--quick", action="store_true", help="G1-G4 only (seconds)")
    args = ap.parse_args()

    gates: dict = {}
    gates["G1_inputs"] = gate_inputs()
    print(f"G1 inputs:        max|dr|={gates['G1_inputs']['max_dr']:.1e} "
          f"max|dT|={gates['G1_inputs']['max_dT']:.1e}  "
          f"{'PASS' if gates['G1_inputs']['pass'] else 'FAIL'}")
    gates["G2_construction"] = gate_construction()
    g2 = gates["G2_construction"]
    print(f"G2 construction:  sqz {g2['squeezed']:.2e}  sqs {g2['squashed']:.2e}  "
          f"swapped {g2['swapped_sign_diff']:.2e}  {'PASS' if g2['pass'] else 'FAIL'}")
    gates["G3_moment_route"] = gate_moment_route()
    g3 = gates["G3_moment_route"]
    print(f"G3 moment route:  sqz {g3['squeezed']:.2e}  sqs {g3['squashed']:.2e}  "
          f"{'PASS' if g3['pass'] else 'FAIL'}")

    states = {kind: q7.build_state(kind) for kind in q7.KINDS}
    gates["G4_observables"] = gate_observables(states)
    for kind in q7.KINDS:
        g = gates["G4_observables"][kind]
        print(f"G4 {kind:9s}:    Cbar {g['Cbar']:.3f} (their {g['table2_Cbar']}) "
              f"sigmaC {g['sigmaC']:.3f} (their {g['table2_sigmaC']}) "
              f"marg RMS {g['marginal_rms_vs_empirical'] * 100:.2f}%  "
              f"{'PASS' if g['pass'] else 'FAIL'}")

    if not args.quick:
        bands = {k: args.per_band for k in args.bands}
        g5, g6 = gate_crossval(states, bands, args.legacy_events, args.procs)
        gates["G5_crossval"], gates["G6_legacy_cast"] = g5, g6
        print(f"G5 crossval:      {g5['n_enclosed']}/{g5['n']} published values inside "
              f"our certified enclosures (worst rel diff {g5['worst_rel_diff']:.1e})  "
              f"{'PASS' if g5['pass'] else 'FAIL'}")
        print(f"G6 legacy cast:   value ratio range {g6['value_ratio_range']}  "
              f"(documented-wrong: {'PASS' if g6['pass'] else 'FAIL'})")

    ok = all(g.get("pass") for g in gates.values())
    from bench._provenance import provenance

    art = {
        "kind": "q7_parity_crossval",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "zenodo_record": "10.5281/zenodo.7141021 (resolves to 7194775)",
        "provenance": provenance(),
        "quick": bool(args.quick),
        "gates": gates,
        "all_pass": bool(ok),
    }
    out = HERE / f"q7_parity_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    out.write_text(json.dumps(art, indent=1, default=float))
    print(f"\n{'ALL GATES PASS' if ok else '*** GATE FAILURE ***'}  -> {out}")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
