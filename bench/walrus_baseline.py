"""Same-instance The Walrus baseline (docs/DESIGN.md §9 / the reproducibility contract).

Times Xanadu's The Walrus (the canonical CPU reference) for the four functions on the
**same GPU instance** that ran our kernels, with the same hygiene -- warm-up, median +
IQR over repeats, randomized cell order, the shared provenance block -- so the
GPU-vs-The-Walrus comparison is apples-to-apples on one machine (not our GPU on a
rented box vs The Walrus on a laptop). The crossover analysis (`bench.crossover`) pairs
this artifact with the kernel throughput artifact from the same session.

The Walrus is one-evaluation-at-a-time, so each cell loops a small batch and reports
per-eval evals/sec, directly comparable to our batched per-eval rate. Runs wherever
``thewalrus`` is importable (the GPU session installs it); skips with a clear error
otherwise.

    uv run python -m bench.walrus_baseline --batch 64 --repeats 7
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from bench import _inputs, _provenance

# The Walrus's high-level paths use np.find_common_type (removed in NumPy 2.0); the
# low-level perm/hafnian/tor used here do not, but apply the NumPy-2.0 compat shim
# defensively so the import + any internal use is safe.
if not hasattr(np, "find_common_type"):
    np.find_common_type = lambda a, s: np.result_type(*a, *s)

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "results" / "throughput"

# func -> (The Walrus call, matrix sizes). The matrices come from the SAME shared generator
# (bench._inputs.bench_batch) and the SAME seed convention as bench.throughput_end_to_end, so
# The Walrus is timed on the *identical inputs* as our GPU/CPU path -- a same-input baseline
# (docs/DESIGN.md §9). For the torontonian the matrix dim is 2*modes (so [8,12,16] = 4..8 modes).
_SPECS: dict[str, Any] = {
    "perm": (lambda tw, M: tw.perm(M), [8, 12, 16]),
    "haf": (lambda tw, M: tw.hafnian(M), [8, 12, 16]),
    "lhaf": (lambda tw, M: tw.hafnian(M, loop=True), [8, 12]),
    "tor": (lambda tw, M: tw.tor(np.real(M)), [8, 12, 16]),
}


def _quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    x = q * (len(s) - 1)
    lo = int(x); frac = x - lo
    return s[lo] * (1 - frac) + s[lo + 1] * frac if lo + 1 < len(s) else s[lo]


def run(batch: int, repeats: int, seed: int = 0, warmup: int = 2,
        regime: str = "physical", out_dir: Path | None = None) -> tuple[dict[str, Any], Path]:
    try:
        import thewalrus as tw
    except Exception as e:  # pragma: no cover - depends on the box
        raise RuntimeError(
            "thewalrus is not importable; install it on the instance to record the "
            "same-instance baseline (the GPU session does this)."
        ) from e

    cells = [(func, d, rep) for func, (_, sizes) in _SPECS.items()
             for d in sizes for rep in range(repeats)]
    random.Random(seed).shuffle(cells)

    # the SAME shared workload (and seed convention 1000+d) as throughput_end_to_end, so the
    # Walrus baseline is timed on the identical matrices our GPU/CPU path is.
    mats = {(func, d): list(_inputs.bench_batch(func, d, batch, regime, 1000 + d))
            for func, (_, sizes) in _SPECS.items() for d in sizes}

    call = {func: _SPECS[func][0] for func in _SPECS}
    # warm-up (untimed) per distinct cell -- steady state before timing
    for func, d in {(f, dd) for f, dd, _ in cells}:
        for _ in range(warmup):
            for M in mats[(func, d)][: min(batch, 8)]:
                call[func](tw, M)

    raw: list[dict[str, Any]] = []
    for func, d, rep in cells:
        batch_mats = mats[(func, d)]
        t0 = time.perf_counter()
        s = 0.0
        for M in batch_mats:
            s += abs(complex(call[func](tw, M)))
        elapsed = time.perf_counter() - t0
        raw.append({"func": func, "matrix_dim": d, "repeat": rep, "batch": batch,
                    "seconds": elapsed, "evals_per_sec": batch / elapsed if elapsed else float("inf"),
                    "checksum": s})

    rows = []
    for func, (_, sizes) in _SPECS.items():
        for d in sizes:
            eps = [r["evals_per_sec"] for r in raw if r["func"] == func and r["matrix_dim"] == d]
            rows.append({"func": func, "matrix_dim": d,
                         "evals_per_sec_median": _quantile(eps, 0.5),
                         "evals_per_sec_iqr": _quantile(eps, 0.75) - _quantile(eps, 0.25)})

    artifact = {
        "kind": "walrus_baseline",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **_provenance.provenance(),
        "library": "thewalrus", "library_version": version("thewalrus"),
        "metric": "per-eval evals/sec (The Walrus, one-at-a-time, looped over a batch); "
                  "median + IQR; randomized cell order; warm-up discarded",
        "params": {"batch": batch, "repeats": repeats, "warmup": warmup, "precision": "fp64",
                   "regime": regime, "seed": seed},
        "env": {"platform": platform.platform(), "numpy": version("numpy")},
        "rows": rows, "raw": raw,
    }
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"walrus_baseline_{stamp}.json"
    path.write_text(json.dumps(artifact, indent=2))
    return artifact, path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--regime", default="physical", choices=list(_inputs.BENCH_REGIMES),
                   help="shared input family (same generator as throughput_end_to_end)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    artifact, path = run(args.batch, args.repeats, warmup=args.warmup,
                         regime=args.regime, out_dir=args.out)
    print(f"# The Walrus baseline (same instance) -> {path}")
    print(f"#   thewalrus {artifact['library_version']}; commit {artifact['commit']}; "
          f"container {artifact['container_digest']}")
    print(f"  {'func':>5} {'dim':>4}  {'evals/sec median':>17}  {'IQR':>11}")
    for r in artifact["rows"]:
        print(f"  {r['func']:>5} {r['matrix_dim']:>4}  {r['evals_per_sec_median']:>17.3e}  "
              f"{r['evals_per_sec_iqr']:>11.2e}")


if __name__ == "__main__":
    main()
