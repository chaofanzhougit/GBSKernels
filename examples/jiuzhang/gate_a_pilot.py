"""Gate A pilot: certified model-comparison power study on real Jiuzhang 1.0 events.

Question this answers before any kernel work or GPU spend (goals: Gate A):
how large is the per-event certified log-likelihood ratio between the targeted
squeezed model and classical alternatives (squashed / thermal), how does it
scale with click number k, and how many events per click band are needed for a
proven-sign verdict at and beyond the published 26-click ceiling?

Models (all through the same validated transfer-matrix pipeline; input moments
per mode with self-<aa> structure, mirroring the 1.5%-RMS-validated loader):
  ideal        nb = sinh^2 r,        mm = sinh r cosh r   (targeted SMSV)
  squashed_pm  nb = sinh^2 r,        mm = nb              (classical, V_-=1,
                                                           photon-matched)
  squashed_raw nb = mm = (e^{2r}-1)/4                     (classical, V_+ kept)
  thermal      nb = sinh^2 r,        mm = 0               (same input energy)
A TMSS-structured variant (cross <a_{2k} a_{2k+1}> = m, zero self terms) is
included in the marginal diagnostic only.

Per event, per model: DD-certified single-large torontonian (tor_single dd=True)
-> interval for log P; interval-propagated per-event log-ratio (ideal - alt);
per-band mean/SD -> required N for a 3-sigma proven-sign verdict. fp64-certified
is also recorded where affordable to show the wall applies to the statistic.

Ensembles: (a) truncated sub-patterns of >=40-click events at k = 13..21 (the
k-trend; whole low-k events are ~1e-5 of the data); (b) whole real events at
k = 21..27 from the 13-32 band file (their-window top end + first beyond-ceiling
points). k >= 28 whole events are the GPU top-up, marked pending.

    uv run python examples/jiuzhang/gate_a_pilot.py --procs 8

Pilot-grade caveat: log sqrt(det Q) per model uses fp64 slogdet with a 1e-9
slack charge added to every per-event bar (campaign version: mpmath). The
squashed definition is the standard classical-envelope construction; it is
cross-checked against the published construction before any campaign claim.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "jiuzhang1"
LSDQ_SLACK = 1e-9          # fp64 slogdet slack, charged to every per-event bar


# ---------------------------------------------------------------- state build
def input_moments(r50: np.ndarray, model: str):
    nb = np.sinh(r50) ** 2
    if model == "ideal":
        return nb, np.sinh(r50) * np.cosh(r50)
    if model == "squashed_pm":
        return nb, nb.copy()
    if model == "squashed_raw":
        nu = (np.exp(2 * r50) - 1) / 4
        return nu, nu.copy()
    if model == "thermal":
        return nb, np.zeros_like(nb)
    raise ValueError(model)


def build_state(T: np.ndarray, nb: np.ndarray, mm: np.ndarray,
                tmss_pairs: bool = False):
    """Output (cov, Q, O, log_sqrt_detQ, mean_photons) for input moments."""
    from sampling import gbs as gbs_mod
    M = T.shape[1]
    if tmss_pairs:                      # cross <a_{2k} a_{2k+1}> structure
        P = np.zeros((len(nb), len(nb)))
        for k in range(0, len(nb), 2):
            P[k, k + 1] = P[k + 1, k] = mm[k]
        aiaj = T.T @ P @ T
    else:                               # self <a_i a_i> structure (validated)
        aiaj = T.T @ np.diag(mm) @ T
    aidaj = T.conj().T @ np.diag(nb) @ T
    x = np.eye(M) + np.real(aidaj + aidaj.conj().T) + 2 * np.real(aiaj)
    p = np.eye(M) + np.real(aidaj + aidaj.conj().T) - 2 * np.real(aiaj)
    xp = 2 * np.imag(aiaj) + 2 * np.imag(aidaj)
    cov = np.block([[x, xp], [xp.T, p]])
    Q = gbs_mod._qmat(cov)
    Qh = (Q + Q.conj().T).real / 2
    ev = float(np.min(np.linalg.eigvalsh(Qh)))
    if ev <= 0:
        raise AssertionError(f"Husimi not PD (min eig {ev:.3e})")
    sign, logdet = np.linalg.slogdet(Q)
    assert sign.real > 0
    # Exact real torontonian matrix (quadrature basis).  The 2026-07-09 pilot
    # run used Re(I - Q^{-1}) here -- the WRONG matrix for this complex-T
    # state (values 20-190x too small, log-ratios unreliable; see
    # sampling.gbs.threshold_O_xxpp and q7_parity.py G5/G6).  The
    # gate_a_pilot_20260709T234321Z.json artifact is superseded: its power
    # arithmetic must be re-derived from a rerun before campaign sizing.
    O = gbs_mod.threshold_O_xxpp(cov, hbar=2.0)[0]
    return cov, Q, O, 0.5 * float(logdet.real), float(np.trace(aidaj).real)


def click_marginals(Q: np.ndarray) -> np.ndarray:
    M = len(Q) // 2
    out = np.empty(M)
    for d in range(M):
        idx = [d, d + M]
        q = Q[np.ix_(idx, idx)]
        out[d] = 1.0 - 1.0 / np.sqrt(abs(np.linalg.det(q).real))
    return out


# ------------------------------------------------------------------- workers
_O_BY_MODEL: dict[str, np.ndarray] = {}


def _init(models_O):
    global _O_BY_MODEL
    _O_BY_MODEL = models_O


def _one(task):
    """(tag, k, event_idx, model, clicked_modes) -> certified DD log-tor pieces."""
    import gbskernels
    tag, k, ei, model, S = task
    O = _O_BY_MODEL[model]
    idx = list(S) + [j + 100 for j in S]
    sub = np.ascontiguousarray(O[np.ix_(idx, idx)])
    t0 = time.time()
    v, d = gbskernels.tor_single(sub, groups=min(k, 13), dd=True)
    dt = time.time() - t0
    E = d["abs_error_bound"]
    if not (v > 0 and v - E > 0):
        return {"tag": tag, "k": k, "event": ei, "model": model, "refused": True,
                "seconds": dt}
    return {"tag": tag, "k": k, "event": ei, "model": model, "refused": False,
            "log_lo": float(np.log(v - E)), "log_hi": float(np.log(v + E)),
            "rel_bound": float(d["rel_error_bound"]), "seconds": dt}


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--trend-events", type=int, default=12)
    ap.add_argument("--trend-ks", type=int, nargs="*", default=[13, 15, 17, 19, 21])
    ap.add_argument("--whole-kmax", type=int, default=27)
    ap.add_argument("--whole-k26", type=int, default=6, help="k=26 events to run")
    ap.add_argument("--whole-k27", type=int, default=2, help="k=27 events to run")
    ap.add_argument("--enclosure-k", type=int, default=10)
    args = ap.parse_args()

    T = np.load(DATA / "T_full.npy")
    r50 = np.repeat(np.loadtxt(DATA / "squeezing parameters.txt"), 2)
    emp = np.load(DATA / "empirical_click_rates.npy")

    models = ["ideal", "squashed_pm", "squashed_raw", "thermal"]
    built, lsdq, nbar_out = {}, {}, {}
    for m in models:
        nb, mm = input_moments(r50, m)
        if m != "ideal":                 # mockups must be classical (m <= nbar)
            assert np.all(mm <= nb + 1e-12), f"{m} not classical-constrained"
        else:                            # the targeted model must NOT be
            assert np.all(mm > nb), "ideal model unexpectedly classical"
        cov, Q, O, ls, nout = build_state(T, nb, mm)
        built[m], lsdq[m], nbar_out[m] = (Q, O), ls, nout

    # ---- marginal diagnostic (incl. TMSS-structured variant, cheap) --------
    marg = {}
    for m in models:
        marg[m] = click_marginals(built[m][0])
    nb, mm = input_moments(r50, "ideal")
    _, Qt, _, _, _ = build_state(T, nb, mm, tmss_pairs=True)[0:5]
    marg["ideal_tmss_pairs"] = click_marginals(Qt)
    marg_rms = {m: float(np.sqrt(np.mean((marg[m] - emp) ** 2))) for m in marg}

    # ---- enclosure spot-gate vs mpmath at small k ---------------------------
    import gbskernels
    from highprec_ref import torontonian_mp
    ev40 = np.load(DATA / "events_ge40.npy")
    S0 = [j for j in range(100) if ev40[0][j]][: args.enclosure_k]
    for m in models:
        O = built[m][1]
        idx = S0 + [j + 100 for j in S0]
        sub = np.ascontiguousarray(O[np.ix_(idx, idx)])
        v, d = gbskernels.tor_single(sub, groups=min(len(S0), 13), dd=True)
        ex = complex(torontonian_mp(sub, dps=50)).real
        assert abs(v - ex) <= d["abs_error_bound"], f"enclosure FAILED for {m}"
    print(f"enclosure gate: DD interval contains 50-dps mpmath for all "
          f"{len(models)} models at k={args.enclosure_k}  OK", flush=True)

    # ---- task lists ---------------------------------------------------------
    tasks = []
    for ei in range(min(args.trend_events, len(ev40))):
        S = [j for j in range(100) if ev40[ei][j]]
        for k in args.trend_ks:
            for m in models:
                tasks.append(("trend", k, ei, m, tuple(S[:k])))

    band = np.load(DATA / "events_band13_32.npy")
    kk = band.sum(1)
    whole = []
    for k in range(21, args.whole_kmax + 1):
        rows = np.where(kk == k)[0]
        if k == 26:
            rows = rows[: args.whole_k26]
        elif k == 27:
            rows = rows[: args.whole_k27]
        whole.extend((k, int(ri)) for ri in rows)
    for k, ri in whole:
        S = [j for j in range(100) if band[ri][j]]
        for m in models:
            tasks.append(("whole", k, ri, m, tuple(S)))

    cost = {13: .1, 15: .2, 17: .8, 19: 3.2, 21: 16, 22: 32, 23: 64,
            24: 128, 25: 256, 26: 512, 27: 1024}
    tasks.sort(key=lambda t: -cost.get(t[1], 2048))   # longest first
    proj = sum(cost.get(t[1], 2048) for t in tasks) / args.procs
    print(f"{len(tasks)} evaluations queued; projected ~{proj/60:.0f} min "
          f"wall on {args.procs} procs", flush=True)

    O_by_model = {m: built[m][1] for m in models}
    t0 = time.time()
    results = []
    with Pool(args.procs, initializer=_init, initargs=(O_by_model,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, tasks, chunksize=1)):
            results.append(r)
            if (i + 1) % 40 == 0:
                print(f"  {i+1}/{len(tasks)} done ({time.time()-t0:.0f}s)",
                      flush=True)

    # ---- assemble per-band certified log-ratios -----------------------------
    by = {}
    for r in results:
        by[(r["tag"], r["k"], r["event"], r["model"])] = r

    def band_stats(tag, k, events):
        rows = []
        for ei in events:
            gi = by.get((tag, k, ei, "ideal"))
            if gi is None or gi["refused"]:
                continue
            row = {"event": ei}
            for alt in models[1:]:
                ga = by.get((tag, k, ei, alt))
                if ga is None or ga["refused"]:
                    row[alt] = None
                    continue
                lo = gi["log_lo"] - ga["log_hi"] - (lsdq["ideal"] - lsdq[alt]) \
                    - 2 * LSDQ_SLACK
                hi = gi["log_hi"] - ga["log_lo"] - (lsdq["ideal"] - lsdq[alt]) \
                    + 2 * LSDQ_SLACK
                row[alt] = (lo, hi)
            rows.append(row)
        out = {}
        for alt in models[1:]:
            iv = [r[alt] for r in rows if r.get(alt)]
            if len(iv) < 2:
                out[alt] = {"n": len(iv)}
                continue
            mid = np.array([(a + b) / 2 for a, b in iv])
            bar = float(np.mean([(b - a) / 2 for a, b in iv]))
            mu, sd = float(np.mean(mid)), float(np.std(mid, ddof=1))
            out[alt] = {"n": len(iv), "mu_per_event": mu, "sd_per_event": sd,
                        "arith_bar_per_event": bar,
                        "N_for_3sigma": int(np.ceil((3 * sd / mu) ** 2))
                        if mu != 0 else None,
                        "ideal_wins_sign": bool(mu > 0)}
        return out

    trend_events = list(range(min(args.trend_events, len(ev40))))
    bands = {}
    for k in args.trend_ks:
        bands[f"trend_k{k}"] = band_stats("trend", k, trend_events)
    for k in sorted({k for k, _ in whole}):
        evs = [ri for kk_, ri in whole if kk_ == k]
        bands[f"whole_k{k}"] = band_stats("whole", k, evs)

    art = {"kind": "jiuzhang1_gate_a_power_pilot",
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "backend": gbskernels.gpu_backend_kind()
           if hasattr(gbskernels, "gpu_backend_kind") else None,
           "models": models,
           "marginal_rms_vs_empirical": marg_rms,
           "mean_photons_out": nbar_out,
           "log_sqrt_detQ_fp64": lsdq, "lsdq_slack": LSDQ_SLACK,
           "note": "log-ratio = log P_ideal - log P_alt; positive = targeted "
                   "squeezed model wins. DD-certified intervals per event.",
           "bands": bands,
           "wall_seconds": time.time() - t0}
    out = HERE / f"gate_a_pilot_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    out.write_text(json.dumps(art, indent=1))

    print(f"\nmarginal RMS vs empirical: " +
          ", ".join(f"{m}={v:.4f}" for m, v in marg_rms.items()))
    print(f"mean photons out: " +
          ", ".join(f"{m}={v:.2f}" for m, v in nbar_out.items()))
    hdr = f"{'band':>10} {'alt':>13} {'n':>3} {'mu/event':>10} {'sd':>8} " \
          f"{'arith bar':>10} {'N(3sig)':>8} {'ideal wins?':>11}"
    print("\n" + hdr)
    for b, alts in bands.items():
        for alt, s in alts.items():
            if "mu_per_event" not in s:
                print(f"{b:>10} {alt:>13} {s['n']:>3}  (insufficient)")
                continue
            print(f"{b:>10} {alt:>13} {s['n']:>3} {s['mu_per_event']:>10.4f} "
                  f"{s['sd_per_event']:>8.4f} {s['arith_bar_per_event']:>10.1e} "
                  f"{str(s['N_for_3sigma']):>8} {str(s['ideal_wins_sign']):>11}")
    print(f"\n-> {out}  ({art['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
