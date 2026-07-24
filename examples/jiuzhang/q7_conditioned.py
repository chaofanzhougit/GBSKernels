"""Click-conditioned Delta H(C) with three separated error budgets.

Campaign precondition 2 (Gate A report, next action 2).  Implements the
Bayesian-test statistic of Q7-1076 Eqs. (18)-(21),

    Delta H(C) = H_SQUE(C) - H_SQUA(C)
               = mean_k[ ln Pr_SQUA(s_k) - ln Pr_SQUE(s_k) ]      (per-event)
                 + ln[ Pr_SQUE(C) / Pr_SQUA(C) ],                 (normalizer)

with Pr(s|C) = Pr(s)/Pr(C); Delta H(C) > 0 means the squashed hypothesis
assigns the events higher conditional likelihood (their Jiuzhang 1.0 verdict:
+0.027 per event, roughly flat over C = 21..26).

The error report keeps three budgets SEPARATE, never merged into one number:
  arith  proven bound on the per-event part (certified DD enclosures through
         interval arithmetic) -- zero for point-value inputs;
  stat   standard error of the per-event mean (the physics noise);
  norm   MC uncertainty of the grouped-click normalizers Pr_HYP(C) -- the
         one term that is NOT certifiable (positive-P Monte Carlo).

Modes (all write one artifact):

  --validate     Delta H(C), C = 21..26, from THEIR published per-pattern
                 probabilities and THEIR normalizers (zenodo 7141021).  Pure
                 array arithmetic; must reproduce their Fig. 4.  This gates
                 the statistic implementation end to end.
  --certified    the same statistic from OUR committed certified enclosures
                 (rows of the q7_parity artifact): per-event intervals ->
                 proven arithmetic budget.  Few events per band (spot check);
                 the campaign scales this on GPU.
  --recompute N  recompute the normalizers independently with
                 thewalrus.grouped_click_probabilities (>= 0.20; the paper's
                 own method, upstreamed by its authors), from OUR
                 parity-validated hypothesis inputs, and gate against their
                 published arrays within combined MC uncertainty.

    uv run python examples/jiuzhang/q7_conditioned.py --validate --certified
    uv run python examples/jiuzhang/q7_conditioned.py --recompute 1000000

Artifact: q7_conditioned_<UTC>.json next to this script (append-only).
Exit 0 iff every requested gate passes.
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

BANDS_PUBLISHED = (21, 22, 23, 24, 25, 26)


# ------------------------------------------------------------------ loaders
def their_normalizers() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Their published grouped-click Pr(C) (positive-P MC, N=1e8, 100 groups)
    with per-C uncertainties; support C = 0..100 (covers beyond-ceiling)."""
    out = {}
    for kind, stem in (("squeezed", "squeezed"), ("squashed", "squashed")):
        pr, unc = np.load(q7.ZEN / f"click_probs/click_probs_{stem}_0.npy")
        out[kind] = (pr, unc)
    return out


def their_band(C: int) -> tuple[np.ndarray, np.ndarray]:
    """Their published per-pattern probabilities for all L events at C clicks."""
    pz = np.load(q7.ZEN / f"pattern_probs/probs_sqz_0_clicks_{C}.npy")
    pa = np.load(q7.ZEN / f"pattern_probs/probs_sqs_0_clicks_{C}.npy")
    return pz, pa


# ------------------------------------------------------------ the statistic
def delta_H_band(
    ln_ratio_mid: np.ndarray,
    C: int,
    norms: dict[str, tuple[np.ndarray, np.ndarray]],
    ln_ratio_halfwidth: np.ndarray | None = None,
) -> dict:
    """Delta H(C) for one band from per-event ln[Pr_SQUA(s)/Pr_SQUE(s)].

    ``ln_ratio_mid`` are per-event midpoints; ``ln_ratio_halfwidth`` (optional)
    are per-event numerical interval half-widths, giving an arithmetic budget
    on the mean (bounds add linearly). Whether this is a rigorous end-to-end
    enclosure depends on the upstream state and transcendental calculations. The normalizer
    shift ln[Pr_SQUE(C)/Pr_SQUA(C)] carries the MC budget (first order:
    relative Pr(C) uncertainties added in quadrature).
    """
    (prz, uncz), (pra, unca) = norms["squeezed"], norms["squashed"]
    n = len(ln_ratio_mid)
    shift = float(np.log(prz[C]) - np.log(pra[C]))
    dH = float(np.mean(ln_ratio_mid)) + shift
    sigma = float(np.std(ln_ratio_mid, ddof=1)) if n > 1 else None
    return {
        "C": C,
        "n_events": int(n),
        "delta_H": dH,
        "err_arith": float(np.mean(ln_ratio_halfwidth))
        if ln_ratio_halfwidth is not None else 0.0,
        "err_stat": float(sigma / np.sqrt(n)) if sigma is not None else None,
        "err_norm": float(np.hypot(uncz[C] / prz[C], unca[C] / pra[C])),
        "normalizer_shift": shift,
        "squashed_wins": bool(dH > 0),
        # campaign power arithmetic (Gate A re-derivation on trustworthy data):
        # events per band for a 3-sigma proven-sign verdict at this band's own
        # effect size -- meaningful only where the source pairing is valid
        "sigma_per_event": sigma,
        "N_3sigma_at_band_effect": int(np.ceil((3 * sigma / dH) ** 2))
        if sigma is not None and dH != 0 else None,
    }


# ------------------------------------------------------------------ mode A
def band_pairing_diagnostics() -> dict:
    """Source-data quality per band (their Zenodo record has three quirks):
    (a) C=26 probs_sqz is PERMUTED relative to probs_sqs and to the patterns
        file (cross-hypothesis corr +0.10 vs +0.997 elsewhere; the sqz file
        was regenerated 2022-10-13); value SETS remain consistent (sorted-pair
        sigma matches the other bands).
    (b) the C=23 and C=25 patterns files carry ONE extra row inserted past the
        head (head indices verified aligned by the certified G5 cross-check).
    (c) sprobs_sqz_0_clicks_26 has 56000 entries vs L=53554.
    Per-event pairing pz<->pa is therefore valid for C=21..25 and broken at
    C=26, where only the band MEAN (pairing-invariant) may be used."""
    out = {}
    for C in BANDS_PUBLISHED:
        pz, pa = their_band(C)
        n_pat = len(np.load(q7.ZEN / f"pattern_probs/patterns_exp/samples_0_clicks_{C}.npy"))
        r = float(np.corrcoef(np.log(pz), np.log(pa))[0, 1])
        out[C] = {"n_probs": len(pz), "n_patterns": n_pat,
                  "cross_hypothesis_corr": r, "pairing_valid": bool(r > 0.9)}
    return out


def mode_validate(norms, diagnostics: dict) -> dict:
    """Their arrays -> their statistic; must reproduce Fig. 4 (Delta H ~ +0.027,
    positive and roughly flat over C = 21..26).  Bands whose cross-hypothesis
    pairing is broken in the source data (C=26) get the pairing-free
    statistical budget sqrt((var_z + var_a)/n) instead of the paired SE."""
    rows = []
    for C in BANDS_PUBLISHED:
        pz, pa = their_band(C)
        row = delta_H_band(np.log(pa) - np.log(pz), C, norms)
        if not diagnostics[C]["pairing_valid"]:
            lz, la = np.log(pz), np.log(pa)
            row["err_stat"] = float(np.sqrt((lz.var(ddof=1) + la.var(ddof=1)) / len(lz)))
            row["pairing_broken_in_source"] = True
        rows.append(row)
    vals = np.array([r["delta_H"] for r in rows])
    # gate: all bands positive; each within 3x its own (stat+norm) budget of
    # their stated +0.027-flat verdict; band-to-band spread small
    budgets = np.array([3 * np.hypot(r["err_stat"], r["err_norm"]) for r in rows])
    ok = bool(np.all(vals > 0) and np.all(np.abs(vals - 0.027) <= np.maximum(budgets, 0.02)))
    return {"rows": rows, "mean_delta_H": float(vals.mean()), "pass": ok}


# ------------------------------------------------------------------ mode B
def mode_certified(norms, parity_artifact: Path) -> dict:
    """Certified per-event enclosures (parity artifact rows) -> Delta H(C)
    with a PROVEN arithmetic budget.  Also checks each certified band value
    against the mode-A full-band value within the spot-check's statistical
    error (n is small; this validates plumbing, not physics power)."""
    art = json.loads(parity_artifact.read_text())
    rows_in = [r for r in art["gates"]["G5_crossval"]["rows"] if r["which"] == "exact"]
    by_band: dict[int, dict[str, dict[int, dict]]] = {}
    for r in rows_in:
        by_band.setdefault(r["k"], {}).setdefault(r["kind"], {})[r["event"]] = r

    out_rows = []
    ok = True
    for C, kinds in sorted(by_band.items()):
        events = sorted(set(kinds["squeezed"]) & set(kinds["squashed"]))
        mids, halfs = [], []
        for e in events:
            rz, ra = kinds["squeezed"][e], kinds["squashed"][e]
            # enclosure intervals [prob - bound, prob + bound]; the lsd slack
            # is already inside prob_bound's construction to first order --
            # charge it again, outward, for safety
            slack = 2 * q7.LSDQ_SLACK
            pz_lo = rz["prob"] * (1 - slack) - rz["prob_bound"]
            pz_hi = rz["prob"] * (1 + slack) + rz["prob_bound"]
            pa_lo = ra["prob"] * (1 - slack) - ra["prob_bound"]
            pa_hi = ra["prob"] * (1 + slack) + ra["prob_bound"]
            assert pz_lo > 0 and pa_lo > 0, "enclosure crosses zero; refuse"
            lo = np.log(pa_lo) - np.log(pz_hi)
            hi = np.log(pa_hi) - np.log(pz_lo)
            mids.append((lo + hi) / 2)
            halfs.append((hi - lo) / 2)
        band = delta_H_band(np.array(mids), C, norms, np.array(halfs))
        # cross-check vs the full published band (mode A on all L events)
        pz_all, pa_all = their_band(C)
        full = delta_H_band(np.log(pa_all) - np.log(pz_all), C, norms)
        band["full_band_delta_H"] = full["delta_H"]
        band["spotcheck_sigma"] = float(np.std(mids, ddof=1) / np.sqrt(len(mids)))
        consistent = abs(band["delta_H"] - full["delta_H"]) <= 3 * band["spotcheck_sigma"]
        band["consistent_with_full_band"] = bool(consistent)
        ok &= consistent
        out_rows.append(band)
    arith = max(r["err_arith"] for r in out_rows)
    return {"rows": out_rows, "max_err_arith": arith, "pass": bool(ok)}


# ------------------------------------------------------------------ mode C
def q7_walrus_inputs(kind: str):
    """(phn, chn, t_prime) for thewalrus.grouped_click_probabilities, from the
    parity-validated construction: per-source-mode photon numbers and
    coherences in the alternating {-r_k, +r_k} basis, and T' = T B with B the
    amplitude-space pair beamsplitter (the paper's Appendix-B prescription)."""
    r25 = q7.load_r25()
    K = 2 * len(r25)
    phn = np.repeat(np.sinh(r25) ** 2, 2)
    if kind == "squeezed":
        m = np.sinh(r25) * np.cosh(r25)
    elif kind == "squashed":
        m = np.sinh(r25) ** 2
    else:
        raise ValueError(kind)
    chn = np.empty(K)
    chn[0::2], chn[1::2] = +m, -m  # param -r_k -> <a^2> = +m; +r_k -> -m
    B_amp = q7.pair_beamsplitter(K)[:K, :K]  # amplitude-space block (real)
    t_prime = q7.load_T_out_by_in() @ B_amp
    return phn, chn, np.ascontiguousarray(t_prime)


def mode_recompute(norms, n_samples: int, n_groups: int = 100, seed: int = 1990) -> dict:
    """Independent normalizers via thewalrus (the paper's own estimator),
    gated against their published arrays within combined MC uncertainty.

    Error convention: the library returns the ACROSS-GROUP STD of the
    ``n_groups`` group means, NOT divided by sqrt(n_groups) -- i.e. a
    conservative (~10x for G=100) stand-in for the SE of the pooled
    estimate.  Their published uncertainty arrays follow the same
    convention, so the z-gate below compares like with like, and the
    err_norm budget charged in :func:`delta_H_band` (built from their
    published arrays) inherits the same conservatism: the true normalizer
    budget is ~sqrt(G) smaller than reported.  We keep the conservative
    reading everywhere -- never overclaim."""
    from thewalrus.grouped_click_probabilities import grouped_click_probabilities

    out = {"n_samples": int(n_samples), "n_groups": int(n_groups), "seed": int(seed)}
    ok = True
    for kind in q7.KINDS:
        phn, chn, tp = q7_walrus_inputs(kind)
        t0 = time.time()
        pr, unc = grouped_click_probabilities(phn, chn + 0j * chn, tp.astype(np.complex128),
                                              n_samples, n_groups, seed)
        dt = time.time() - t0
        prt, unct = norms[kind]
        # gate on the region that matters and where MC noise is under control
        sel = np.arange(15, 61)
        comb = np.sqrt(unc[sel] ** 2 + unct[sel] ** 2)
        z = np.abs(pr[sel] - prt[sel]) / comb
        worst = float(z.max())
        pass_k = bool(worst <= 5.0)  # 46 comparisons; 5 combined sigmas
        ok &= pass_k
        out[kind] = {
            "seconds": dt,
            "worst_z_C15_60": worst,
            "median_z_C15_60": float(np.median(z)),
            "rel_unc_at_C27": float(unc[27] / pr[27]),
            "error_convention": "across-group std of group means "
                                "(conservative; /sqrt(n_groups) for the SE)",
            "pass": pass_k,
            "prc": pr.tolist(),
            "prc_unc": unc.tolist(),
        }
        print(f"  recompute {kind}: N={n_samples:.0e} in {dt:.0f}s; vs published "
              f"C=15..60 worst z={worst:.2f}, median z={out[kind]['median_z_C15_60']:.2f}  "
              f"{'PASS' if pass_k else 'FAIL'}", flush=True)
    out["pass"] = bool(ok)
    return out


# ----------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--certified", action="store_true")
    ap.add_argument("--recompute", type=int, default=0, metavar="N",
                    help="positive-P samples for the independent normalizer run")
    ap.add_argument("--seed", type=int, default=1990,
                    help="recompute RNG seed. 1990 (thewalrus default, very "
                         "likely theirs) REPLAYS the head of their sample "
                         "stream -- a same-stream input-mapping validation, "
                         "not an independent MC check; use any other seed "
                         "for independence")
    ap.add_argument("--parity-artifact", type=Path,
                    default=HERE / "q7_parity_20260710T180631Z.json")
    args = ap.parse_args()
    if not (args.validate or args.certified or args.recompute):
        args.validate = args.certified = True

    norms = their_normalizers()
    results: dict = {}
    diagnostics = band_pairing_diagnostics()
    results["band_diagnostics"] = {"per_band": diagnostics,
                                   "pass": True}  # informational, never gates

    if args.validate:
        results["validate"] = mode_validate(norms, diagnostics)
        print("Delta H(C) from THEIR arrays (must reproduce Fig. 4):")
        for r in results["validate"]["rows"]:
            broken = "  [source pairing broken -> pairing-free budget]" \
                if r.get("pairing_broken_in_source") else ""
            print(f"  C={r['C']}: dH={r['delta_H']:+.4f}  stat {r['err_stat']:.4f}  "
                  f"norm {r['err_norm']:.4f}  (n={r['n_events']}, "
                  f"squashed_wins={r['squashed_wins']}){broken}")
        print(f"  mean over bands {results['validate']['mean_delta_H']:+.4f}  "
              f"{'PASS' if results['validate']['pass'] else 'FAIL'}")

    if args.certified:
        results["certified"] = mode_certified(norms, args.parity_artifact)
        print("certified spot-check (parity-artifact enclosures):")
        for r in results["certified"]["rows"]:
            print(f"  C={r['C']}: dH={r['delta_H']:+.4f} +/- {r['spotcheck_sigma']:.3f} (stat, n={r['n_events']})"
                  f"  arith {r['err_arith']:.1e}  norm {r['err_norm']:.4f}  "
                  f"full-band {r['full_band_delta_H']:+.4f}  "
                  f"{'consistent' if r['consistent_with_full_band'] else '*** INCONSISTENT ***'}")
        print(f"  max certified arithmetic budget {results['certified']['max_err_arith']:.1e}  "
              f"{'PASS' if results['certified']['pass'] else 'FAIL'}")

    if args.recompute:
        tag = ("SAME-STREAM mapping validation (seed 1990 = thewalrus default)"
               if args.seed == 1990 else f"independent (seed {args.seed})")
        print(f"recomputed normalizers, {tag}:")
        results["recompute"] = mode_recompute(norms, args.recompute, seed=args.seed)
        results["recompute"]["same_stream_as_published"] = bool(args.seed == 1990)

    ok = all(v.get("pass") for v in results.values())
    from bench._provenance import provenance

    art = {
        "kind": "q7_conditioned_deltaH",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": provenance(),
        "normalizer_source": "published (zenodo 7141021, click_probs_*_0.npy)",
        "results": results,
        "all_pass": bool(ok),
    }
    out = HERE / f"q7_conditioned_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    out.write_text(json.dumps(art, indent=1, default=float))
    print(f"\n{'ALL REQUESTED GATES PASS' if ok else '*** GATE FAILURE ***'}  -> {out}")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
