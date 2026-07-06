"""Render the batch-size crossover curves from a sweep artifact (docs/DESIGN.md §9 figures).

Reads a ``crossover_*.json`` (from ``bench.crossover``) and draws, per (function,
matrix size), GPU vs CPU per-eval rate against batch size on log-log axes, marking the
crossover batch. These are the paper figures for the batched-throughput thesis.

matplotlib is imported lazily and is **optional** -- without it (or with ``--data-only``)
the script emits the plot data as CSV so the curves are reproducible regardless. The
package keeps numpy/mpmath as its only hard deps; matplotlib is a dev/plotting extra.

    uv run python -m bench.plot_crossover results/throughput/crossover_*.json -o fig/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_csv(artifact: dict[str, Any], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["func", "matrix_dim", "batch", "gpu_median", "gpu_iqr", "cpu_median", "cpu_iqr",
                    "walrus_median", "crossover_batch_vs_cpu", "crossover_batch_vs_walrus",
                    "achieved_rel_err_fp64", "precision_tier"])
        for s in artifact["series"]:
            for p in s["points"]:
                w.writerow([s["func"], s["matrix_dim"], p["batch"], p.get("gpu_median"), p.get("gpu_iqr"),
                            p.get("cpu_median"), p.get("cpu_iqr"), s.get("walrus_median"),
                            s.get("crossover_batch_vs_cpu"), s.get("crossover_batch_vs_walrus"),
                            s.get("achieved_rel_err_fp64"), s.get("precision_tier")])
    return out


def plot(artifact: dict[str, Any], out_dir: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("# matplotlib not available -- emitting CSV only "
              "(pip install matplotlib, or use the CSV).", file=sys.stderr)
        return [write_csv(artifact, out_dir / "crossover.csv")]

    out_dir.mkdir(parents=True, exist_ok=True)
    funcs = sorted({s["func"] for s in artifact["series"]})
    written = []
    for func in funcs:
        fig, ax = plt.subplots(figsize=(6, 4))
        for s in (s for s in artifact["series"] if s["func"] == func):
            bs = [p["batch"] for p in s["points"]]
            ax.plot(bs, [p["gpu_median"] for p in s["points"]], "-o", label=f"GPU d={s['matrix_dim']}")
            ax.plot(bs, [p["cpu_median"] for p in s["points"]], "--s", label=f"CPU d={s['matrix_dim']}")
            # the same-instance Walrus baseline is batch-independent -> a horizontal line
            wv = s.get("walrus_median")
            if wv is not None:
                ax.axhline(wv, ls="-.", alpha=0.4, label=f"Walrus d={s['matrix_dim']}")
            if s.get("crossover_batch_vs_cpu"):
                ax.axvline(s["crossover_batch_vs_cpu"], color="grey", ls=":", alpha=0.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("batch size"); ax.set_ylabel("evals / sec (per eval)")
        ax.set_title(f"{func} -- GPU vs CPU throughput crossover\n"
                     f"commit {artifact.get('commit')} | {artifact.get('gpu_backend')}")
        ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.3)
        p = out_dir / f"crossover_{func}.png"
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
        written.append(p)
    written.append(write_csv(artifact, out_dir / "crossover.csv"))  # always emit the data too
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact", type=Path, help="a crossover_*.json from bench.crossover")
    p.add_argument("-o", "--out", type=Path, default=Path("fig"))
    p.add_argument("--data-only", action="store_true", help="emit CSV only, skip plotting")
    args = p.parse_args()
    art = _load(args.artifact)
    outs = [write_csv(art, args.out / "crossover.csv")] if args.data_only else plot(art, args.out)
    for o in outs:
        print(f"wrote {o}")


if __name__ == "__main__":
    main()
