"""Batch-size sweep + accuracy-normalized GPU/CPU/Walrus crossover (docs/DESIGN.md §9).

Sweeps the public-path end-to-end throughput over a range of batch sizes and records, per
(function, matrix size), the GPU per-eval rate at each batch against two ~flat baselines on
the **same shared workload** (bench._inputs.bench_batch): our CPU reference, and the
**same-instance The Walrus** (measured once -- Walrus is one-at-a-time, so its per-eval rate
is batch-independent). At small batch the GPU's fixed H2D/launch/D2H overhead loses; as the
batch grows the GPU's batched throughput overtakes. The **crossover batch** -- the smallest
batch where the GPU median overtakes a baseline -- is the headline of the batched-throughput
thesis, reported separately vs the CPU and vs The Walrus.

To find the *actual* crossover (not just "the smallest batch we tried"), the sweep must include
**low** batches -- the default starts at 1 -- so the GPU genuinely loses at the small end.

Evidence discipline (docs/DESIGN.md §9): the artifact retains **median + IQR + the raw repetitions**
for GPU/CPU at every batch and for the Walrus baseline (not just medians); each series is tagged
with the **achieved FP64 error** (vs mpmath) and its **precision tier**; and an **official** run
(``strict=True``) is **fatal** if e2e reports a GPU/CPU public-path checksum disagreement at any
batch, or if the same-instance Walrus baseline is unavailable -- so no crossover is published
past a backend mismatch or without its baseline.

    uv run python -m bench.crossover --batches 1,4,16,64,256,1024,4096,16384 --repeats 7
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import gbskernels
from bench import _inputs, _provenance, throughput_end_to_end as e2e, walrus_baseline

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "results" / "throughput"
DEFAULT_BATCHES = [1, 4, 16, 64, 256, 1024, 4096, 16384]

_MP = {"perm": "permanent_mp", "haf": "hafnian_mp",
       "lhaf": "loop_hafnian_mp", "tor": "torontonian_mp"}


def _achieved_error(func: str, dim: int, regime: str, n: int, dps: int) -> float | None:
    """Max relative error of the public FP64 GPU path vs the mpmath ground truth on ``n``
    elements of the shared workload -- the accuracy the throughput is bought at. The SAME
    matrices the sweep timed (seed convention ``1000+dim``). ``n<=0`` or ``dps<=0`` disables."""
    if n <= 0 or dps <= 0:
        return None
    import highprec_ref
    stack = _inputs.bench_batch(func, dim, n, regime, 1000 + dim)
    fp = np.asarray(getattr(gbskernels, f"{func}_batched")(stack, backend="gpu", precision="fp64"))
    mp_fn = getattr(highprec_ref, _MP[func])
    rel = 0.0
    for k in range(n):
        ref = complex(mp_fn(stack[k], dps=dps))
        rel = max(rel, abs(complex(fp[k]) - ref) / max(abs(ref), 1e-300))
    return rel


def _pull(summary, raw, func, dim, backend):
    """(median, iqr, raw evals/sec list) for (func, dim, backend) from an e2e artifact --
    so the crossover keeps the dispersion + raw repetitions, not just the median."""
    row = next((r for r in summary if r["func"] == func and r["matrix_dim"] == dim), {})
    sub = row.get(backend, {})
    rawlist = [r["evals_per_sec"] for r in raw
               if r["backend"] == backend and r["func"] == func and r["matrix_dim"] == dim]
    return sub.get("evals_per_sec_median"), sub.get("evals_per_sec_iqr"), rawlist


def run(batches: list[int], repeats: int = 7, seed: int = 0, warmup: int = 2,
        regime: str = "physical", error_n: int = 2, error_dps: int = 30,
        strict: bool = True, out_dir: Path | None = None) -> tuple[dict[str, Any], Path]:
    base = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    tmp = base / "_sweep_tmp"

    # GPU + CPU per-eval median/IQR/raw at each batch (same shared workload as the Walrus
    # baseline). An e2e GPU/CPU checksum disagreement is FATAL for an official (strict) run.
    curves: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for b in sorted(set(batches)):
        art, _ = e2e.run(batch=b, repeats=repeats, seed=seed, warmup=warmup,
                         regime=regime, out_dir=tmp)
        if strict and not art["all_backends_agree"]:
            bad = [f"{r['func']}/d{r['matrix_dim']}" for r in art["summary"] if not r["backends_agree"]]
            raise RuntimeError(
                f"crossover ABORT: e2e GPU/CPU public-path checksum disagreement at batch {b} "
                f"(regime {regime}) on {bad} -- no official crossover from a backend mismatch.")
        for row in art["summary"]:
            func, dim = row["func"], row["matrix_dim"]
            gm, gi, gr = _pull(art["summary"], art["raw"], func, dim, "gpu")
            cm, ci, cr = _pull(art["summary"], art["raw"], func, dim, "cpu")
            curves.setdefault((func, dim), []).append({
                "batch": b, "gpu_median": gm, "gpu_iqr": gi, "gpu_raw": gr,
                "cpu_median": cm, "cpu_iqr": ci, "cpu_raw": cr})

    # The Walrus per-eval rate is batch-independent (one-at-a-time) -> measure it ONCE on the
    # same workload, keeping median + IQR + raw. Unavailable Walrus is FATAL for a strict run.
    walrus: dict[tuple[str, int], dict[str, Any]] = {}
    walrus_meta: dict[str, Any]
    try:
        wb_batch = max(min(max(batches), 128), 8)
        wart, _ = walrus_baseline.run(batch=wb_batch, repeats=repeats, seed=seed,
                                      warmup=warmup, regime=regime, out_dir=tmp)
        for r in wart["rows"]:
            rraw = [x["evals_per_sec"] for x in wart["raw"]
                    if x["func"] == r["func"] and x["matrix_dim"] == r["matrix_dim"]]
            walrus[(r["func"], r["matrix_dim"])] = {
                "median": r["evals_per_sec_median"], "iqr": r["evals_per_sec_iqr"], "raw": rraw}
        walrus_meta = {"library_version": wart["library_version"], "batch": wb_batch}
    except Exception as e:  # pragma: no cover - depends on the box
        walrus_meta = {"skipped": str(e)}
    if strict and "skipped" in walrus_meta:
        raise RuntimeError(
            f"crossover ABORT: the same-instance The Walrus baseline is unavailable "
            f"({walrus_meta['skipped']}) -- an official crossover requires it.")

    series = []
    for (func, dim), pts in sorted(curves.items()):
        pts.sort(key=lambda p: p["batch"])
        wb = walrus.get((func, dim), {})
        wmed = wb.get("median")

        def _gt(p, beat):
            return p["gpu_median"] is not None and beat is not None and p["gpu_median"] > beat

        x_cpu = next((p["batch"] for p in pts if _gt(p, p["cpu_median"])), None)
        x_wal = next((p["batch"] for p in pts if _gt(p, wmed)), None)
        x_best = next((p["batch"] for p in pts
                       if p["gpu_median"] is not None
                       and any(v is not None for v in (p["cpu_median"], wmed))
                       and p["gpu_median"] > max(v for v in (p["cpu_median"], wmed) if v is not None)),
                      None)
        series.append({
            "func": func, "matrix_dim": dim, "precision_tier": "fp64",
            "achieved_rel_err_fp64": _achieved_error(func, dim, regime, error_n, error_dps),
            "walrus_median": wmed, "walrus_iqr": wb.get("iqr"), "walrus_raw": wb.get("raw"),
            "crossover_batch_vs_cpu": x_cpu,
            "crossover_batch_vs_walrus": x_wal,
            "crossover_batch_vs_best": x_best,
            "points": pts,
        })

    artifact = {
        "kind": "crossover_batch_sweep",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **_provenance.provenance(),
        "gpu_backend": gbskernels.gpu_backend_kind(),
        "metric": "per-eval evals/sec (public path) vs batch; GPU rises, CPU & same-instance "
                  "Walrus are ~flat baselines on the same workload; crossover = smallest batch "
                  "where GPU median > baseline; median+IQR+raw retained; achieved FP64 error per series",
        "params": {"batches": sorted(set(batches)), "repeats": repeats, "warmup": warmup,
                   "regime": regime, "seed": seed, "strict": strict,
                   "error_n": error_n, "error_dps": error_dps},
        "walrus": walrus_meta,
        "series": series,
    }
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = base / f"crossover_{stamp}.json"
    path.write_text(json.dumps(artifact, indent=2))
    return artifact, path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batches", default=",".join(map(str, DEFAULT_BATCHES)),
                   help="comma-separated batch sizes to sweep (start low to find the real crossover)")
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--regime", default="physical", choices=list(_inputs.BENCH_REGIMES))
    p.add_argument("--error-n", type=int, default=2, help="elements checked vs mpmath (0=off)")
    p.add_argument("--error-dps", type=int, default=30, help="mpmath precision for the error check")
    p.add_argument("--no-strict", dest="strict", action="store_false",
                   help="do NOT fail on e2e disagreement / missing Walrus (non-official run)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    batches = [int(x) for x in args.batches.split(",")]
    artifact, path = run(batches, args.repeats, seed=args.seed, warmup=args.warmup, regime=args.regime,
                         error_n=args.error_n, error_dps=args.error_dps, strict=args.strict, out_dir=args.out)
    w = artifact["walrus"]
    print(f"# batch-size sweep / crossover ({artifact['gpu_backend']}) -> {path}")
    print(f"#   commit {artifact['commit']}; container {artifact['container_digest']}; "
          f"regime={args.regime}; strict={args.strict}; batches={batches}")
    print(f"#   walrus baseline: {w.get('library_version', 'SKIPPED: ' + str(w.get('skipped')))}")
    print(f"  {'func':>5} {'dim':>4}  {'fp64 rel.err':>12}  {'x vs cpu':>9} {'x vs walrus':>11}")
    for s in artifact["series"]:
        err = s["achieved_rel_err_fp64"]
        print(f"  {s['func']:>5} {s['matrix_dim']:>4}  {('%.2e' % err) if err is not None else 'n/a':>12}  "
              f"{str(s['crossover_batch_vs_cpu']):>9} {str(s['crossover_batch_vs_walrus']):>11}")


if __name__ == "__main__":
    main()
