"""Certified-bound tightness distributions -- the numerics figure.

For a CS/numerics reviewer, "the enclosure is always safe" is table stakes; the
question is *how tight*. This measures, across ensembles rather than single
worst cases, the distribution of three quantities for each certified function:

  rel_bound  = bound / |value|           -- what the certificate PROMISES (≈ κ·u)
  rel_err    = |value − exact| / |value|  -- what actually happened (vs mpmath)
  tightness  = bound / max(|value−exact|, u·|value|)
                                          -- the bound's slack over the true
                                             error, floored at the fp64 ulp
                                             (you cannot resolve error below u·|v|)

The hard invariant (`rel_err ≤ rel_bound` on every sample -- the enclosure) is
checked here too and must never fail. The story the distribution tells: on the
physical ensemble the bound is tight (small, bounded tightness) and on the
adversarial cancellation families it widens *correctly* -- because those inputs
are genuinely ill-conditioned, and the certificate reports that honestly rather
than hiding it. That is the difference between a rigorous certificate and a
heuristic κ.

    uv run python -m bench.tightness --samples 150 [--plot]

Writes results/tightness/tightness_<utc>.json (per function×regime distribution
stats + enclosure check) and, with --plot, results/tightness/tightness.png.
CPU-only; references are mpmath. Sizes are kept modest so the reference is
affordable -- the point is the distribution shape, not large N.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

import cpu_ref
import highprec_ref as href
from bench import _inputs as inp
from bench._provenance import provenance

RESULTS = Path(__file__).resolve().parent.parent / "results" / "tightness"
U = 2.0 ** -53

# func -> (mpmath reference, {regime: sampler(seed) -> matrix})
_CASES: dict[str, tuple[Callable, dict[str, Callable]]] = {
    "perm": (href.permanent_mp, {
        "physical": lambda s: inp.physical_permanent(10, s),
        "adversarial": lambda s: inp.pm1_matrix(10, s),            # ±1: heavy cancellation
    }),
    "haf": (href.hafnian_mp, {
        "physical": lambda s: inp.physical_hafnian(10, s),
        "adversarial": lambda s: inp.cancellation_hafnian(10.0 ** -(3 + s % 9), s),
    }),
    "lhaf": (href.loop_hafnian_mp, {
        "physical": lambda s: inp.physical_loop_hafnian(10, s),
        "adversarial": lambda s: inp.cancellation_loop_hafnian(10.0 ** -(3 + s % 9), s),
    }),
    "tor": (href.torontonian_mp, {
        "physical": lambda s: np.real(inp.physical_torontonian(4, s)),
        "adversarial": lambda s: inp.cancellation_torontonian(10.0 ** -(1 + s % 6)),
    }),
}


def _quantiles(x: np.ndarray) -> dict[str, float]:
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"q05": float(q[0]), "q25": float(q[1]), "median": float(q[2]),
            "q75": float(q[3]), "q95": float(q[4]), "max": float(x.max()),
            "min": float(x.min())}


def _collect(func: str, regime: str, gen: Callable, ref: Callable,
             samples: int) -> dict[str, Any]:
    rel_bound, rel_err, tight = [], [], []
    enclosure_fails = 0
    for s in range(samples):
        A = gen(s)
        value, bound = cpu_ref.certified(func, A)
        av = abs(value)
        if av == 0.0 or not np.isfinite(bound):
            continue
        exact = complex(ref(A, dps=50))
        err = abs(complex(value) - exact)
        rb, re = bound / av, err / av
        if re > rb * (1 + 1e-9):           # the enclosure invariant -- must hold
            enclosure_fails += 1
        rel_bound.append(rb)
        rel_err.append(re)
        tight.append(bound / max(err, U * av))
    return {"n": len(tight), "enclosure_fails": enclosure_fails,
            "rel_bound": _quantiles(np.array(rel_bound)),
            "rel_err": _quantiles(np.array(rel_err)),
            "tightness": _quantiles(np.array(tight)),
            "_raw_tightness": [float(x) for x in tight],
            "_raw_rel_bound": [float(x) for x in rel_bound]}


def run(samples: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    print(f"{'func/regime':>18} {'N':>4} {'encl':>5} "
          f"{'rel_bound med':>14} {'rel_err med':>12} {'tightness med (q95)':>22}")
    for func, (ref, regimes) in _CASES.items():
        out[func] = {}
        for regime, gen in regimes.items():
            d = _collect(func, regime, gen, ref, samples)
            out[func][regime] = d
            t = d["tightness"]
            print(f"{func + '/' + regime:>18} {d['n']:>4} {d['enclosure_fails']:>5} "
                  f"{d['rel_bound']['median']:>14.2e} {d['rel_err']['median']:>12.2e} "
                  f"{t['median']:>10.1f} ({t['q95']:.1f})")
    total_fails = sum(out[f][r]["enclosure_fails"]
                      for f in out for r in out[f])
    print(f"\nenclosure violations across all ensembles: {total_fails} "
          f"(MUST be 0 -- the certificate is only meaningful if it never under-claims)")
    return {"kind": "certified_tightness", **provenance(),
            "params": {"samples": samples, "sizes": "perm/haf/lhaf n=10, tor 2n=8"},
            "enclosure_violations_total": total_fails, "by_function": out}


def plot(art: dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    funcs = list(art["by_function"])
    fig, ax = plt.subplots(figsize=(9, 5))
    positions, data, colors, labels = [], [], [], []
    pos = 0
    for func in funcs:
        for regime, col in (("physical", "#2a7"), ("adversarial", "#d63")):
            raw = art["by_function"][func][regime].get("_raw_tightness", [])
            if raw:
                positions.append(pos)
                data.append(np.log10(np.maximum(raw, 1.0)))
                colors.append(col)
                labels.append(f"{func}\n{regime}")
                pos += 1
        pos += 0.6
    parts = ax.violinplot(data, positions=positions, showmedians=True, widths=0.8)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"$\log_{10}$ tightness  =  bound / max(actual error, $u\,|v|$)")
    ax.set_title("Certified-bound tightness distribution (rigorous, never < 1)\n"
                 "physical ensembles tight; adversarial families widen honestly")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"-> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    art = run(args.samples)
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = RESULTS / f"tightness_{stamp}.json"
    out.write_text(json.dumps(art, indent=1))
    print(f"-> {out}")
    if args.plot:
        plot(art, RESULTS / "tightness.png")


if __name__ == "__main__":
    main()
