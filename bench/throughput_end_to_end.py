"""End-to-end (public-API) throughput: the GPU binding path vs the CPU backend.

This measures the **public** call path -- `gbskernels.X_batched(stack,
backend=...)` -- which for the GPU backend includes the numpy -> host_api ->
H2D -> launch -> sync -> D2H -> numpy round trip on *every* call (the binding has
no device-resident workspace; see the README "Known limitations"). It is
deliberately separate from the kernel-only timing in `core/bench_kernels.cu`:
together they bound how much of the cost is data movement vs compute.

Benchmark hygiene (docs/DESIGN.md §9): n>=repeats timed runs per cell reported as
median + IQR; **randomized execution order** across (backend, func, size) cells;
a post-call **checksum honesty guard** that also cross-checks the GPU and CPU
backends agree; raw data appended to `results/throughput/`; no composite score.

Runs wherever the GPU extension is importable -- on a CUDA box it times the real
device, on a host-shim build it times the CPU emulation (the timing is then only
indicative, but the harness and the GPU/CPU agreement are validated). The
accuracy-normalized vs-The-Walrus comparison lives in `bench/throughput.py`.

    uv run python -m bench.throughput_end_to_end --batch 1024 --repeats 7
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

import gbskernels
from bench import _inputs, _provenance

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "results" / "throughput"

# func -> (batched fn, matrix sizes within the GPU kernels' caps). The matrices come from
# the SINGLE shared workload generator (bench._inputs.bench_batch) so the GPU, CPU, and the
# same-instance The Walrus baseline are all timed on the *same input* per (func, dim, regime)
# -- an apples-to-apples, accuracy-normalizable comparison (docs/DESIGN.md §9). For the
# torontonian the matrix dim is 2*modes.
_SPECS = {
    "perm": (gbskernels.perm_batched, [8, 12, 16]),
    "haf": (gbskernels.haf_batched, [8, 12, 16]),
    "lhaf": (gbskernels.lhaf_batched, [8, 12]),
    "tor": (gbskernels.tor_batched, [8, 12, 16]),
}


def _checksum(v: np.ndarray) -> float:
    """Order-independent magnitude checksum (the post-sync honesty guard)."""
    return float(np.sum(np.abs(np.asarray(v, dtype=np.complex128))))


def _quantile(sorted_vals: list[float], q: float) -> float:
    x = q * (len(sorted_vals) - 1)
    lo = int(x)
    frac = x - lo
    return (sorted_vals[lo] * (1 - frac) + sorted_vals[lo + 1] * frac
            if lo + 1 < len(sorted_vals) else sorted_vals[lo])




def run(batch: int, repeats: int, seed: int = 0, warmup: int = 2,
        regime: str = "physical", out_dir: Path | None = None) -> tuple[dict[str, Any], Path]:
    if not gbskernels.gpu_available():
        raise RuntimeError(
            "GPU extension not importable; build bindings/ (or the host-shim) first."
        )
    backends = ["gpu", "cpu"]
    cells = [(eng, func, d, rep)
             for eng in backends for func, (_, sizes) in _SPECS.items()
             for d in sizes for rep in range(repeats)]
    random.Random(seed).shuffle(cells)

    # one shared workload per (func, dim) from bench._inputs (regime-aware): GPU, CPU, and
    # the Walrus baseline all consume the SAME matrices -> same-input comparison.
    stacks = {(func, d): _inputs.bench_batch(func, d, batch, regime, 1000 + d)
              for func, (_, sizes) in _SPECS.items() for d in sizes}

    # The CPU backend loops the Python reference, so it is timed on a capped
    # sub-batch; evals/sec is per-eval and stays comparable to the GPU full batch.
    cpu_cap = 128

    # Warm-up policy (docs/DESIGN.md §9): each (backend, func, size) cell is run `warmup` times
    # UNTIMED before any timed run, to reach steady-state GPU clocks and prime caches /
    # the kernel JIT. The cold first run is never a headline number -- this is what
    # tames the large cold-start IQRs. Warm-up runs are discarded.
    for eng, func, d in {(e, f, dd) for e, f, dd, _ in cells}:
        fn, n = _SPECS[func][0], (batch if eng == "gpu" else min(batch, cpu_cap))
        for _ in range(warmup):
            fn(stacks[(func, d)][:n], backend=eng)

    raw: list[dict[str, Any]] = []
    for eng, func, d, rep in cells:
        fn = _SPECS[func][0]
        n = batch if eng == "gpu" else min(batch, cpu_cap)
        A = stacks[(func, d)][:n]
        t0 = time.perf_counter()
        out = fn(A, backend=eng)
        elapsed = time.perf_counter() - t0  # GPU path syncs inside host_api
        # checksum over the first cpu_cap results (the same matrices both backends
        # compute) so GPU and CPU are comparable despite different timed batch sizes.
        raw.append({"backend": eng, "func": func, "matrix_dim": d, "repeat": rep,
                    "batch_timed": n, "seconds": elapsed,
                    "evals_per_sec": n / elapsed if elapsed else float("inf"),
                    "checksum": _checksum(out[:min(batch, cpu_cap)])})

    summary = _summarize(raw)
    # The public-path honesty gate: on a real device the GPU and CPU backends must agree to
    # FP64 tolerance on every well-conditioned (physical) cell. A single disagreement means a
    # kernel/binding bug, and NO timing from this run is publishable -- the caller (main / the
    # GPU session) turns this into a non-zero exit. (Adversarial inputs may legitimately
    # diverge; gate on the physical regime.)
    all_agree = all(r["backends_agree"] for r in summary)
    artifact = {
        "kind": "throughput_end_to_end",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **_provenance.provenance(),  # commit + container_digest + hostname
        "path": "public gbskernels.*_batched(backend=...) incl. H2D/launch/sync/D2H",
        # build-flag provenance, not host OS: a no-GPU runner is "host-shim".
        "gpu_backend": gbskernels.gpu_backend_kind(),
        "env": {"platform": platform.platform(), "numpy": version("numpy")},
        "params": {"batch": batch, "repeats": repeats, "warmup": warmup, "precision": "fp64",
                   "regime": regime, "seed": seed,
                   "metric": "evals/sec median + IQR; randomized cell order; "
                             "warm-up discarded; checksum guard"},
        "all_backends_agree": all_agree,
        "summary": summary,
        "raw": raw,
    }
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"throughput_e2e_{stamp}.json"
    path.write_text(json.dumps(artifact, indent=2))
    return artifact, path


def _summarize(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    keys = sorted({(r["func"], r["matrix_dim"]) for r in raw})
    for func, d in keys:
        row: dict[str, Any] = {"func": func, "matrix_dim": d}
        checks = {}
        for eng in ("gpu", "cpu"):
            eps = sorted(r["evals_per_sec"] for r in raw
                         if r["backend"] == eng and r["func"] == func and r["matrix_dim"] == d)
            cs = [r["checksum"] for r in raw
                  if r["backend"] == eng and r["func"] == func and r["matrix_dim"] == d]
            checks[eng] = cs[0] if cs else None
            if eps:
                row[eng] = {"evals_per_sec_median": _quantile(eps, 0.5),
                            "evals_per_sec_iqr": _quantile(eps, 0.75) - _quantile(eps, 0.25)}
        # GPU vs CPU agree to FP64 tolerance -- NOT bit-exact: the two backends use
        # different determinant/linear-algebra routines, so mildly-conditioned
        # inputs legitimately differ at ~1e-11 (a strict string check would flag it).
        g, c = checks.get("gpu"), checks.get("cpu")
        row["backends_agree"] = (g is not None and c is not None
                                 and abs(g - c) <= 1e-7 * max(abs(c), 1e-300))
        row["checksum_gpu"], row["checksum_cpu"] = g, c
        rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--warmup", type=int, default=2, help="untimed warm-up runs per cell")
    p.add_argument("--regime", default="physical", choices=list(_inputs.BENCH_REGIMES),
                   help="shared input family for every (func,dim) cell")
    args = p.parse_args()
    artifact, path = run(args.batch, args.repeats, args.seed, args.warmup,
                         regime=args.regime, out_dir=args.out)
    print(f"# end-to-end throughput ({artifact['gpu_backend']}) -> {path}")
    print(f"#   public binding path; commit {artifact['commit']}; digest "
          f"{artifact['container_digest']}; batch={args.batch} repeats={args.repeats} "
          f"warmup={args.warmup} regime={args.regime}")
    print(f"  {'func':>5} {'dim':>4}  {'gpu med':>12} {'cpu med':>12}  agree")
    for r in artifact["summary"]:
        gm = r.get("gpu", {}).get("evals_per_sec_median", float("nan"))
        cm = r.get("cpu", {}).get("evals_per_sec_median", float("nan"))
        print(f"  {r['func']:>5} {r['matrix_dim']:>4}  {gm:>12.3e} {cm:>12.3e}  "
              f"{'ok' if r['backends_agree'] else 'MISMATCH'}")
    # P0.8: a public-path checksum disagreement fails the run (non-zero exit), so the GPU
    # session aborts instead of publishing a number from a backend mismatch.
    if not artifact["all_backends_agree"]:
        print("\n[FAIL] GPU and CPU disagree on a public-path cell -- no timing from this run "
              "is publishable. Investigate the kernel; rerun on the physical regime.")
        sys.exit(2)


if __name__ == "__main__":
    main()
