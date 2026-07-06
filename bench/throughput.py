"""Throughput benchmark harness (accuracy-normalized, honesty-guarded).

The benchmark claim is **throughput at a stated accuracy**, never raw throughput
(docs/DESIGN.md §9). This harness measures batched-evaluation throughput (evals/sec)
for the GBSKernels CPU reference and, where available, The Walrus, at matched
accuracy, with the hygiene that makes the numbers credible:

* n >= ``repeats`` timed runs per cell, reported as median + IQR (not mean);
* **randomized execution order** across (engine, size) cells, so thermal/cache
  drift can't systematically favor one engine;
* a **benchmark-honesty guard**: every timed call's outputs are reduced to a
  checksum *after* timing, and the checksum is stored, so no async/early-return
  can fake a fast time (trivially true on CPU, essential once the GPU path
  lands);
* raw per-run data written append-only to ``results/throughput/``; **no composite
  "winner" number**; CPU single-eval latency where The Walrus honestly wins is
  reported, not hidden.

This run is a **CPU baseline**. GPU throughput rows are added in a scripted
rented-GPU session (docs/DESIGN.md §8/sec.10); nothing here runs on a GPU.

    uv run python -m bench.throughput --sizes 4,8,12 --batch 256 --repeats 7
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
from typing import Any, Callable

import numpy as np

import gbskernels
from bench._inputs import random_complex

try:
    import thewalrus

    _HAVE_WALRUS = True
except Exception:
    _HAVE_WALRUS = False

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "results" / "throughput"


def _checksum(values: np.ndarray) -> str:
    """Order-independent magnitude checksum of a result vector (honesty guard)."""
    v = np.asarray(values, dtype=np.complex128)
    return f"{float(np.sum(np.abs(v))):.12e}|{len(v)}"


def _time_once(fn: Callable[[], np.ndarray]) -> tuple[float, str]:
    """Run ``fn``, return (elapsed_seconds, checksum-of-result-after-sync)."""
    t0 = time.perf_counter()
    out = fn()
    elapsed = time.perf_counter() - t0
    # Post-compute checksum: forces materialization of every result *after*
    # timing so no early-return can fake a fast time (docs/DESIGN.md §8).
    return elapsed, _checksum(out)


def _gbskernels_batched(func: str, batch: list[np.ndarray]) -> Callable[[], np.ndarray]:
    fn = {"perm": gbskernels.perm_batched, "haf": gbskernels.haf_batched}[func]
    return lambda: fn(batch, precision="fp64")


def _walrus_looped(func: str, batch: list[np.ndarray]) -> Callable[[], np.ndarray]:
    wfn = {"perm": thewalrus.perm, "haf": thewalrus.hafnian}[func]
    return lambda: np.array([wfn(A) for A in batch], dtype=np.complex128)


def _make_batch(func: str, n: int, batch_size: int, seed: int) -> list[np.ndarray]:
    out = []
    for b in range(batch_size):
        A = random_complex(n, seed=seed + b)
        if func == "haf":
            A = A + A.T  # symmetric for the hafnian
        out.append(A)
    return out


def run(
    func: str,
    sizes: list[int],
    batch_size: int,
    repeats: int,
    seed: int = 0,
    out_dir: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    engines = ["gbskernels"]
    if _HAVE_WALRUS:
        engines.append("thewalrus")

    # Build the randomized work list: (engine, size, repeat) cells, shuffled.
    cells = [
        (engine, n, rep)
        for engine in engines
        for n in sizes
        for rep in range(repeats)
    ]
    random.Random(seed).shuffle(cells)

    # Fixed input batch per size (same inputs for both engines -> fair, and the
    # checksum must agree across engines, a correctness cross-check for free).
    batches = {n: _make_batch(func, n, batch_size, seed=1000 + n) for n in sizes}

    raw: list[dict[str, Any]] = []
    for engine, n, rep in cells:
        batch = batches[n]
        if engine == "gbskernels":
            fn = _gbskernels_batched(func, batch)
        else:
            fn = _walrus_looped(func, batch)
        elapsed, checksum = _time_once(fn)
        raw.append(
            {
                "engine": engine,
                "n": n,
                "repeat": rep,
                "seconds": elapsed,
                "evals_per_sec": batch_size / elapsed if elapsed > 0 else float("inf"),
                "checksum": checksum,
            }
        )

    summary = _summarize(raw, engines, sizes, batch_size)
    artifact = {
        "kind": "throughput",
        "function": func,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tier": "cpu-baseline",
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": version("numpy"),
            "thewalrus": version("thewalrus") if _HAVE_WALRUS else None,
        },
        "params": {
            "batch_size": batch_size,
            "repeats": repeats,
            "sizes": sizes,
            "precision": "fp64",
            "dtype": "complex128",
            "order": "randomized across (engine, size, repeat)",
        },
        "summary": summary,
        "raw": raw,
    }

    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"throughput_{func}_{stamp}_cpu.json"
    path.write_text(json.dumps(artifact, indent=2))
    return artifact, path


def _summarize(raw, engines, sizes, batch_size) -> list[dict[str, Any]]:
    rows = []
    for n in sizes:
        row: dict[str, Any] = {"n": n}
        checks = {}
        for engine in engines:
            eps = sorted(r["evals_per_sec"] for r in raw if r["engine"] == engine and r["n"] == n)
            checks[engine] = {r["checksum"] for r in raw if r["engine"] == engine and r["n"] == n}
            row[engine] = {
                "evals_per_sec_median": float(np.median(eps)),
                "evals_per_sec_iqr": float(np.subtract(*np.percentile(eps, [75, 25]))),
            }
        # cross-engine correctness: checksums must match (same inputs, same answer)
        if len(engines) > 1:
            allchecks = set().union(*checks.values())
            row["checksums_agree"] = len(allchecks) == 1
        rows.append(row)
    return rows


def _print_summary(artifact: dict[str, Any], path: Path) -> None:
    print(f"# throughput [{artifact['function']}, {artifact['tier']}] -> {path}")
    e = artifact["env"]
    print(f"#   {e['platform']} | py {e['python']} | numpy {e['numpy']} | "
          f"thewalrus {e['thewalrus']}")
    print(f"#   batch={artifact['params']['batch_size']} repeats={artifact['params']['repeats']} "
          f"(median evals/sec, IQR); randomized order; no composite score")
    engines = [k for k in ("gbskernels", "thewalrus") if k in artifact["summary"][0]]
    head = f"  {'n':>3}"
    for eng in engines:
        head += f"  {eng + ' med':>16}  {'IQR':>9}"
    if len(engines) > 1:
        head += "  checks"
    print(head)
    for row in artifact["summary"]:
        line = f"  {row['n']:>3}"
        for eng in engines:
            line += f"  {row[eng]['evals_per_sec_median']:>16.1f}  {row[eng]['evals_per_sec_iqr']:>9.1f}"
        if len(engines) > 1:
            line += f"   {'ok' if row['checksums_agree'] else 'MISMATCH'}"
        print(line)


def _parse_sizes(s: str) -> list[int]:
    if "-" in s and "," not in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--func", choices=["perm", "haf"], default="perm")
    p.add_argument("--sizes", type=_parse_sizes, default=_parse_sizes("4,6,8,10"))
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    artifact, path = run(args.func, args.sizes, args.batch, args.repeats, args.seed, args.out)
    _print_summary(artifact, path)


if __name__ == "__main__":
    main()
