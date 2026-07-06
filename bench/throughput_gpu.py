"""GPU throughput driver — runs the compiled CUDA timing harness with provenance.

Thin wrapper (docs/DESIGN.md §9/sec.10) around ``core/build/bench_kernels`` (built by
CMake under nvcc in a rented-GPU session). It runs the harness, captures the
per-(func, size) evals/sec + honesty checksums it emits, attaches GPU/driver/clock
provenance from ``nvidia-smi``, and writes an append-only artifact to
``results/throughput/``. No composite "winner" number; raw data retained.

This module does **not** compile or run anything on a GPU by itself — it requires
that ``core/build/bench_kernels`` already exists (the session's manifest builds
it). Invoked as ``python -m bench.throughput_gpu`` from ``scripts/session.py``.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench import _provenance

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BIN = REPO / "core" / "build" / "bench_kernels"
DEFAULT_OUT = REPO / "results" / "throughput"


def _nvidia_smi() -> dict[str, Any]:
    """GPU name, driver, and clocks for provenance (empty dict if unavailable)."""
    q = ("name,driver_version,memory.total,memory.used,temperature.gpu,"
         "clocks.sm,clocks.mem,clocks.max.sm,power.draw")
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        fields = [f.strip() for f in out.split(",")]
        keys = ["name", "driver_version", "memory_total", "memory_used", "temperature_c",
                "clock_sm", "clock_mem", "clock_sm_max", "power_draw"]
        return dict(zip(keys, fields))
    except Exception:
        return {}


def run(bench_bin: Path, batch: int, repeats: int, out_dir: Path) -> tuple[dict[str, Any], Path]:
    if not bench_bin.exists():
        raise FileNotFoundError(
            f"{bench_bin} not found — build it first (cmake --build core/build) in the "
            "rented-GPU session; this driver only runs the compiled harness."
        )
    proc = subprocess.run(
        [str(bench_bin), str(batch), str(repeats)],
        capture_output=True, text=True, check=True,
    )

    def _parse(line: str) -> dict[str, Any]:
        # C printf emits bare nan/inf; rewrite to the JSON tokens json accepts so
        # an anomalous row is recorded (as NaN/inf) rather than crashing the run.
        import re
        line = re.sub(r":\s*-inf\b", ": -Infinity", line)
        line = re.sub(r":\s*inf\b", ": Infinity", line)
        line = re.sub(r":\s*nan\b", ": NaN", line)
        return json.loads(line)

    rows = [_parse(line) for line in proc.stdout.splitlines() if line.strip().startswith("{")]

    artifact = {
        "kind": "throughput",
        "tier": "gpu",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **_provenance.provenance(),  # commit + container_digest + hostname
        "gpu": _nvidia_smi(),  # name/driver/memory + temperature & live clocks at run time
        "host": {"platform": platform.platform()},
        "params": {"batch": batch, "repeats": repeats, "precision": "fp64+dd",
                   "metric": "evals_per_sec median + IQR over repeats; bench_kernels.cu "
                             "warms each cell before timing; post-sync checksum/row"},
        "methodology_caveats": (
            "Toward docs/DESIGN.md §9 bar: the container digest + commit are now recorded and "
            "the kernel timing is warmed up. Still single-instance per run (sweep the 4090 "
            "AND a datacenter GPU across runs); the on-the-same-instance The Walrus "
            "comparison runs in the GPU session (bench.walrus_baseline)."
        ),
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    gpu_tag = (artifact["gpu"].get("name", "gpu") or "gpu").replace(" ", "_")
    path = out_dir / f"throughput_{gpu_tag}_{stamp}.json"
    path.write_text(json.dumps(artifact, indent=2))
    return artifact, path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    # accepted for manifest-compatibility with the CPU harness; the compiled
    # binary fixes its own size grid, so these are informational here.
    p.add_argument("--func", default=None)
    p.add_argument("--sizes", default=None)
    args = p.parse_args()

    artifact, path = run(args.bin, args.batch, args.repeats, args.out)
    print(f"# gpu throughput -> {path}")
    g = artifact["gpu"]
    print(f"#   GPU: {g.get('name','?')} | driver {g.get('driver_version','?')} | "
          f"{g.get('temperature_c','?')}C | clk {g.get('clock_sm','?')}")
    print(f"#   commit {artifact.get('commit')} | container {artifact.get('container_digest')} | "
          f"batch={args.batch} repeats={args.repeats}")
    print(f"  {'func':>9}  {'matrix_dim':>10}  {'evals/sec median':>17}  {'IQR':>11}")
    for r in artifact["rows"]:
        med = r.get("evals_per_sec_median", r.get("evals_per_sec_best", float("nan")))
        iqr = r.get("evals_per_sec_iqr", float("nan"))
        # the cooperative permanent rows carry a "groups" width; surface it so the
        # perm-vs-perm_coop crossover is legible (raw rows keep the field regardless).
        label = r["func"] + (f"/g{r['groups']}" if "groups" in r else "")
        print(f"  {label:>9}  {r['matrix_dim']:>10}  {med:>17.3e}  {iqr:>11.2e}")


if __name__ == "__main__":
    main()
