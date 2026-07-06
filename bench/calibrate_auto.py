"""Calibrate precision="auto": is kappa a trustworthy FP64-risk indicator? (docs/DESIGN.md §6)

``precision="auto"`` trusts the FP64 result when the cancellation indicator
``kappa = sum|terms| / |result|`` (``cpu_ref.summation_condition_number``) is below
``_AUTO_KAPPA_MAX`` (1e8), and reruns in the high-precision tier otherwise. ``kappa`` is a
**heuristic**, NOT a rigorous error certificate: the model is ``rel_err_fp64 ~ kappa * eps``,
but ``kappa`` is itself formed from the (FP64-inexact) computed ``|result|``, so under extreme
cancellation it can mis-estimate. So we MEASURE the relationship on physical / loss /
adversarial ensembles and calibrate the threshold rather than assume it:

* per ``(func, regime, size)``: ``kappa``, the actual FP64 relative error vs the mpmath
  ground truth, and whether ``auto`` would trust FP64 (``kappa < threshold``);
* a calibration summary: the **worst FP64 error among the TRUSTED cases** (the accuracy
  ``auto`` actually delivers), the **false-trust count** (``kappa < threshold`` yet the error
  is large), and the kappa-vs-error correlation.

The headline a truthful claim can make: on these ensembles, ``kappa < 1e8`` keeps the FP64
error below ``max_rel_err_when_trusted`` with ``false_trust_count`` exceptions -- a calibrated
heuristic, stated with its measured failure rate, not a guarantee.

    uv run python -m bench.calibrate_auto --dps 60
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mpmath
import numpy as np

import cpu_ref
import gbskernels
import highprec_ref
from bench import _inputs, _provenance

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "results" / "accuracy"

_MP = {"perm": highprec_ref.permanent_mp, "haf": highprec_ref.hafnian_mp,
       "lhaf": highprec_ref.loop_hafnian_mp, "tor": highprec_ref.torontonian_mp}
_FN = {"perm": gbskernels.perm, "haf": gbskernels.haf, "lhaf": gbskernels.lhaf, "tor": gbskernels.tor}
_EPS = float(np.finfo(np.float64).eps)
_SIZES = {"perm": [6, 8], "haf": [6, 8], "lhaf": [6, 8], "tor": [6, 8]}  # tor dim = 2*modes


def _pearson(pts: list[tuple[float, float]]) -> float | None:
    n = len(pts)
    if n < 2:
        return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in pts)
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs)); dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return float(num / (dx * dy)) if dx > 0 and dy > 0 else None


def _eval(func: str, A, backend: str, threshold: float):
    """(FP64 value, kappa, trusted) for one input on ``backend``. On the GPU backend the kappa
    comes from the on-device ``*_kappa`` kernel and the FP64 value from the GPU FP64 path -- so
    the calibration measures the GPU AUTO decision (not a CPU proxy); ``trusted`` == auto would
    keep FP64 (tier 'fp64'). On CPU it uses cpu_ref + summation_condition_number."""
    if backend == "gpu":
        _val, diag = _FN[func](A, precision="auto", backend="gpu", return_diagnostics=True)
        kappa = float(diag["cancellation"])
        fp = complex(_FN[func](A, precision="fp64", backend="gpu"))
        return fp, kappa, (diag["tier"] == "fp64")
    fp = complex(_FN[func](A, precision="fp64", backend="cpu"))
    kappa = float(cpu_ref.summation_condition_number(func, A, fp))
    return fp, kappa, (kappa < threshold)


def run(dps: int = 60, seeds: int = 5, threshold: float | None = None,
        backend: str | None = None, out_dir: Path | None = None) -> tuple[dict[str, Any], Path]:
    threshold = gbskernels._AUTO_KAPPA_MAX if threshold is None else threshold
    # Calibrate the REAL path the claim is used for: the GPU auto path when the extension is
    # importable (its kappa comes from the device *_kappa kernels), else CPU.
    if backend is None:
        backend = "gpu" if gbskernels.gpu_available() else "cpu"
    rows: list[dict[str, Any]] = []
    for func in ("perm", "haf", "lhaf", "tor"):
        for regime in _inputs.BENCH_REGIMES:
            for dim in _SIZES[func]:
                for s in range(seeds):
                    A = _inputs.bench_batch(func, dim, 1, regime, 1000 + dim + s)[0]
                    fp, kappa, trusted = _eval(func, A, backend, threshold)
                    with mpmath.workdps(dps):
                        exact = _MP[func](A, dps=dps)
                        rel = float(abs(mpmath.mpc(fp) - exact) / max(abs(exact), mpmath.mpf("1e-300")))
                    rows.append({"func": func, "regime": regime, "dim": dim, "seed": s,
                                 "kappa": kappa, "rel_err_fp64": rel, "trusted": trusted})

    trusted = [r for r in rows if r["trusted"]]
    per_regime = {}
    for regime in _inputs.BENCH_REGIMES:
        rr = [r for r in rows if r["regime"] == regime]
        tr = [r for r in rr if r["trusted"]]
        per_regime[regime] = {
            "n": len(rr), "n_trusted": len(tr),
            "max_kappa": max(r["kappa"] for r in rr),
            "max_rel_err_when_trusted": max((r["rel_err_fp64"] for r in tr), default=0.0),
        }
    summary = {
        "threshold": threshold, "eps": _EPS,
        "n_total": len(rows), "n_trusted": len(trusted),
        "max_rel_err_when_trusted": max((r["rel_err_fp64"] for r in trusted), default=0.0),
        "false_trust_count": sum(1 for r in trusted if r["rel_err_fp64"] > 1e-6),
        "log_kappa_vs_log_relerr_corr": _pearson(
            [(math.log10(max(r["kappa"], 1.0)), math.log10(max(r["rel_err_fp64"], 1e-300))) for r in rows]),
        "per_regime": per_regime,
    }
    artifact = {
        "kind": "auto_calibration",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **_provenance.provenance(),
        "indicator": "kappa = sum|terms| / |result| (a-posteriori; a HEURISTIC indicator, "
                     "NOT a rigorous error certificate)",
        "model": "rel_err_fp64 ~ kappa * eps",
        "backend": backend,                              # the path calibrated (gpu kappa kernels or cpu)
        "gpu_backend": gbskernels.gpu_backend_kind(),    # real device vs host-shim (honest provenance)
        "params": {"dps": dps, "seeds": seeds, "backend": backend},
        "env": {"platform": platform.platform()},
        "summary": summary,
        "rows": rows,
    }
    out = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"auto_calibration_{stamp}.json"
    path.write_text(json.dumps(artifact, indent=2))
    return artifact, path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dps", type=int, default=60)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--backend", choices=["gpu", "cpu"], default=None,
                   help="path to calibrate (default: gpu if the extension is importable)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    art, path = run(dps=args.dps, seeds=args.seeds, backend=args.backend, out_dir=args.out)
    s = art["summary"]
    print(f"# precision='auto' calibration ({art['backend']} path / {art['gpu_backend']}) -> {path}")
    print(f"#   commit {art['commit']}; trust when kappa < {s['threshold']:.0e}; eps = {s['eps']:.2e}")
    print(f"  trusted {s['n_trusted']}/{s['n_total']}; worst FP64 rel.err when trusted = "
          f"{s['max_rel_err_when_trusted']:.2e}; false-trust (>1e-6) = {s['false_trust_count']}")
    print(f"  log(kappa) vs log(rel.err) correlation = {s['log_kappa_vs_log_relerr_corr']}")
    for regime, r in s["per_regime"].items():
        print(f"  {regime:>11}: max kappa {r['max_kappa']:.1e}; "
              f"worst rel.err when trusted {r['max_rel_err_when_trusted']:.1e}")


if __name__ == "__main__":
    main()
