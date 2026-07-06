"""Accuracy characterization for ALL FOUR functions (the measured boundary, §6).

For each of permanent / hafnian / loop hafnian / torontonian, sweep a tunable
cancellation family and record, against the independent ``mpmath`` reference:

* ``rel_err_fp64`` -- the native double-precision relative error (the boundary:
  it grows as the result cancels);
* ``rel_err_dd``   -- the double-double relative error, when the compiled GPU
  extension is available (the *real* DD kernels, host-emulated through the
  host-shim build or run on a GPU). DD arithmetic is **internal**: the kernels
  collapse the double-double result back to ``complex128`` on output, so this
  measures DD's ability to recover a correct FP64 answer after cancellation, not
  31-digit output.

Writes a self-describing JSON artifact to ``results/accuracy/`` (append-only).
This generalizes ``accuracy_permanent.py`` (which keeps the detailed
condition-number sweep for the permanent) to the whole library.

    uv run python -m bench.accuracy --dps 60
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import mpmath
import numpy as np

import gbskernels
import highprec_ref
from bench import _inputs, _provenance

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "results" / "accuracy"

# mpmath ground-truth reference per function.
_REF = {
    "perm": highprec_ref.permanent_mp, "haf": highprec_ref.hafnian_mp,
    "lhaf": highprec_ref.loop_hafnian_mp, "tor": highprec_ref.torontonian_mp,
}

# ADVERSARIAL: a tunable cancellation family (knob -> matrix) + the knob grid.
_ADVERSARIAL: dict[str, tuple[Callable[[float], np.ndarray], list[float]]] = {
    "perm": (lambda d: _inputs.make_cancellation_matrix(6, d, seed=1),
             [1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]),
    "haf": (lambda d: _inputs.cancellation_hafnian(d, seed=1), [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]),
    "lhaf": (lambda d: _inputs.cancellation_loop_hafnian(d, seed=1), [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]),
    "tor": (lambda d: _inputs.cancellation_torontonian(d), [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]),
}

# PHYSICAL: realistic well-conditioned inputs (size -> matrix) + the size grid
# (capped so the DD kernels and the mpmath reference stay feasible).
_PHYSICAL: dict[str, tuple[Callable[[int, int], np.ndarray], list[int]]] = {
    "perm": (_inputs.physical_permanent, [4, 8, 12]),
    "haf": (_inputs.physical_hafnian, [4, 8, 12]),
    "lhaf": (_inputs.physical_loop_hafnian, [4, 8, 10]),
    "tor": (_inputs.physical_torontonian, [2, 4, 6]),  # n = modes (matrix 2n)
}

# LOSS / mixed-state regime: matrices from a lossy (mixed) Gaussian state -- the third
# realistic regime (the permanent is a unitary boson-sampling amplitude, with no
# Gaussian loss analog, so it has no loss section).
_LOSS: dict[str, tuple[Callable[[int, int], np.ndarray], list[int]]] = {
    "haf": (_inputs.loss_hafnian, [4, 8]),
    "lhaf": (_inputs.loss_loop_hafnian, [4, 8]),
    "tor": (_inputs.loss_torontonian, [2, 4]),  # n = modes (matrix 2n)
}


def _rel_err_mp(approx: complex, exact_mp: Any, dps: int) -> float:
    with mpmath.workdps(dps):
        a = mpmath.mpc(complex(approx).real, complex(approx).imag)
        return float(abs(a - exact_mp) / abs(exact_mp))


def _errors(func: str, A: np.ndarray, dps: int, have_dd: bool) -> dict[str, float]:
    """FP64 and (when the GPU backend is present) DD relative error vs mpmath.

    FP64 is evaluated on the SAME backend as DD when the extension is present, so
    the artifact records the FP64<->DD crossover on the device (not a CPU FP64 vs
    GPU DD comparison). With no extension, FP64 falls back to the CPU reference."""
    backend = "gpu" if have_dd else "cpu"
    fp = getattr(gbskernels, func)(A, backend=backend)
    with mpmath.workdps(dps):
        exact = _REF[func](A, dps=dps)
    out = {"rel_err_fp64": _rel_err_mp(fp, exact, dps)}
    if have_dd:
        dd = getattr(gbskernels, func)(A, precision="dd", backend="gpu")
        out["rel_err_dd"] = _rel_err_mp(dd, exact, dps)
    return out


def run(dps: int = 60, out_dir: Path | None = None) -> tuple[dict[str, Any], Path]:
    have_dd = gbskernels.gpu_available()
    dps = max(dps, 80)
    sweeps: dict[str, Any] = {}
    for func in _REF:
        family, deltas = _ADVERSARIAL[func]
        adversarial = [dict(delta=d, **_errors(func, family(d), dps, have_dd)) for d in deltas]
        pfamily, sizes = _PHYSICAL[func]
        physical = [dict(n=n, **_errors(func, pfamily(n, seed=1), dps, have_dd)) for n in sizes]
        sweeps[func] = {"adversarial": adversarial, "physical": physical}
        if func in _LOSS:  # third regime: lossy / mixed-state inputs (Gaussian only)
            lfamily, lsizes = _LOSS[func]
            sweeps[func]["loss"] = [dict(n=n, **_errors(func, lfamily(n, seed=1), dps, have_dd))
                                    for n in lsizes]

    # Whether the DD numbers were produced on a real GPU or host-emulated (the
    # same kernel source either way; the distinction matters for the "GPU-vs-mpmath"
    # claim, so it is recorded honestly). The signal is the extension's own
    # compile-time build flag -- NOT the host OS -- so a no-GPU CI runner is
    # correctly "host-shim", never mislabelled "gpu".
    dd_device = gbskernels.gpu_backend_kind()
    # FP64 was measured on the same backend as DD when the extension is present
    # (so the FP64<->DD crossover is established on-device), else the CPU reference.
    fp64_device = dd_device if have_dd else "cpu"

    artifact = {
        "kind": "accuracy_all_functions",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **_provenance.provenance(),  # commit + container_digest + hostname (every artifact)
        "dd_measured": have_dd,
        "fp64_backend": fp64_device,  # where the FP64 numbers came from
        "dd_backend": dd_device,  # "gpu" (real device) | "host-shim" (CPU emulation)
        "note": ("Work-precision study for all four functions across THREE input regimes vs an "
                 "independent mpmath reference: PHYSICAL (pure, well-conditioned), ADVERSARIAL "
                 "(tunable cancellation), and LOSS (lossy/mixed-state matrices; Gaussian "
                 "functions only). DD is an INTERNAL precision tier: kernels collapse the "
                 "double-double result to complex128 on output, recovering a correct FP64 "
                 "answer under cancellation (not exposing 31-digit results)."),
        "env": {"python": platform.python_version(), "numpy": version("numpy"),
                "mpmath": version("mpmath"), "platform": platform.platform()},
        "params": {"dps": dps, "dtype": "complex128",
                   "sections": "physical (size sweep) + adversarial (cancellation sweep) "
                               "+ loss (mixed-state, haf/lhaf/tor)"},
        "sweeps": sweeps,
    }
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"accuracy_all_{stamp}.json"
    path.write_text(json.dumps(artifact, indent=2))
    return artifact, path


def _print(artifact: dict[str, Any], path: Path) -> None:
    dd = artifact["dd_measured"]
    print(f"# accuracy: all four functions, physical + adversarial -> {path}")
    print(f"#   DD measured: {dd} (backend: {artifact.get('dd_backend')})  | the work-precision study, §6")
    for func, sec in artifact["sweeps"].items():
        print(f"\n  {func} -- physical (well-conditioned):  {'n':>3}  {'rel_err_fp64':>13}" +
              ("  rel_err_dd" if dd else ""))
        for r in sec["physical"]:
            line = f"        {r['n']:>3}  {r['rel_err_fp64']:>13.2e}"
            if "rel_err_dd" in r:
                line += f"  {r['rel_err_dd']:>10.2e}"
            print(line)
        print(f"  {func} -- adversarial (cancellation):  {'delta':>8}  {'rel_err_fp64':>13}" +
              ("  rel_err_dd" if dd else ""))
        for r in sec["adversarial"]:
            line = f"        {r['delta']:>8.0e}  {r['rel_err_fp64']:>13.2e}"
            if "rel_err_dd" in r:
                line += f"  {r['rel_err_dd']:>10.2e}"
            print(line)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dps", type=int, default=60)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    artifact, path = run(args.dps, args.out)
    _print(artifact, path)


if __name__ == "__main__":
    main()
