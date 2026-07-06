"""Accuracy characterization for the permanent (FP64 vs mpmath reference).

The CPU-runnable, throughput-free half of docs/DESIGN.md §9: relative error vs matrix
size and vs the Glynn condition number ``kappa``, measured against the
independent arbitrary-precision reference. This is the "measured boundary" of
docs/DESIGN.md §6 -- the thing that tells a user exactly when FP64 suffices and when
the (GPU) double-double tier is required.

Writes a timestamped, self-describing JSON artifact to ``results/accuracy/``
(append-only). **No timing is recorded here** -- throughput numbers come only
from scripted rented-GPU sessions (docs/DESIGN.md §8 CI policy).

    uv run python -m bench.accuracy_permanent --sizes 2-12 --seeds 8 --dps 60
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mpmath
import numpy as np

from cpu_ref import cancellation_ratio, permanent_glynn
from cpu_ref.permanent import permanent_glynn_dd
from highprec_ref import permanent_mp

from ._inputs import make_cancellation_matrix, random_complex

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "results" / "accuracy"
CANCELLATION_DELTAS = [1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).parent,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _rel_err_mp(approx: complex, exact_mp: Any, dps: int) -> float:
    """Relative error of an FP64 result vs the mpmath reference, computed in
    mpmath so the reference's full precision is used (not truncated to FP64)."""
    with mpmath.workdps(dps):
        a = mpmath.mpc(approx.real, approx.imag)
        return float(abs(a - exact_mp) / abs(exact_mp))


def run(
    sizes: list[int],
    seeds_per_size: int,
    dps: int = 60,
    out_dir: Path | None = None,
    tag: str = "",
) -> tuple[dict[str, Any], Path]:
    """Run the accuracy sweeps and write one append-only artifact. Returns it."""
    size_sweep = []
    for n in sizes:
        rel_errs, kappas = [], []
        for seed in range(seeds_per_size):
            A = random_complex(n, seed=1000 * n + seed)
            exact_mp = permanent_mp(A, dps=dps)
            approx = permanent_glynn(A)
            rel_errs.append(_rel_err_mp(approx, exact_mp, dps))
            kappas.append(cancellation_ratio(A, perm_value=complex(exact_mp)))
        size_sweep.append(
            {
                "n": n,
                "rel_err_median": float(np.median(rel_errs)),
                "rel_err_max": float(np.max(rel_errs)),
                "kappa_median": float(np.median(kappas)),
                "rel_errs": [float(x) for x in rel_errs],
                "kappas": [float(x) for x in kappas],
            }
        )

    # Conditioning axis: drive kappa with the cancellation knob at fixed n.
    cancellation_sweep = []
    cancel_dps = max(dps, 80)
    for delta in CANCELLATION_DELTAS:
        A = make_cancellation_matrix(6, delta=delta, seed=1)
        exact_mp = permanent_mp(A, dps=cancel_dps)
        cancellation_sweep.append(
            {
                "n": 6,
                "delta": delta,
                "kappa": cancellation_ratio(A, perm_value=complex(exact_mp)),
                "rel_err_fp64": _rel_err_mp(permanent_glynn(A), exact_mp, cancel_dps),
                "rel_err_dd": _rel_err_mp(permanent_glynn_dd(A), exact_mp, cancel_dps),
            }
        )

    artifact = {
        "kind": "accuracy_permanent",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "mpmath": version("mpmath"),
            "git_commit": _git_commit(),
        },
        "params": {
            "sizes": sizes,
            "seeds_per_size": seeds_per_size,
            "dps": dps,
            "dtype": "complex128",
            "input_family": "random_complex (size sweep) / cancellation block (kappa sweep)",
        },
        "size_sweep": size_sweep,
        "cancellation_sweep": cancellation_sweep,
    }

    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{tag}" if tag else ""
    path = out_dir / f"accuracy_permanent_{stamp}{suffix}.json"
    path.write_text(json.dumps(artifact, indent=2))
    return artifact, path


def _print_summary(artifact: dict[str, Any], path: Path) -> None:
    print(f"# accuracy_permanent  ->  {path}")
    print(f"#   env: py {artifact['env']['python']}, numpy {artifact['env']['numpy']}, "
          f"mpmath {artifact['env']['mpmath']}, commit {artifact['env']['git_commit']}")
    print("\n  size sweep (FP64 vs mpmath, random complex):")
    print(f"  {'n':>3}  {'rel_err median':>15}  {'rel_err max':>13}  {'kappa median':>13}")
    for row in artifact["size_sweep"]:
        print(f"  {row['n']:>3}  {row['rel_err_median']:>15.2e}  "
              f"{row['rel_err_max']:>13.2e}  {row['kappa_median']:>13.2e}")
    print("\n  conditioning sweep (the measured FP64<->DD boundary, n=6):")
    print(f"  {'delta':>8}  {'kappa':>11}  {'rel_err FP64':>13}  {'rel_err DD':>13}")
    for row in artifact["cancellation_sweep"]:
        print(f"  {row['delta']:>8.0e}  {row['kappa']:>11.2e}  "
              f"{row['rel_err_fp64']:>13.2e}  {row['rel_err_dd']:>13.2e}")


def _parse_sizes(s: str) -> list[int]:
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=_parse_sizes, default=_parse_sizes("2-12"))
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--dps", type=int, default=60)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()
    artifact, path = run(args.sizes, args.seeds, args.dps, args.out, args.tag)
    _print_summary(artifact, path)


if __name__ == "__main__":
    main()
