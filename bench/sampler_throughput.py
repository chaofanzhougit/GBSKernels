"""End-to-end GBS sampler throughput -- **samples/sec**, not kernel evals/sec.

The conditional sampler (`sampling.sampler.sample`) draws photon-number samples by
the reduced-covariance chain rule, routing each mode's conditional-probability
hafnian batch through one `gbskernels.Workspace` (ragged bucketing + device-buffer
residency across the whole chain). This measures the headline product metric --
samples/sec for a real GBS workload -- on the CPU and GPU backends, with the same
honest provenance as the kernel benchmarks (commit, GPU/host-shim backend).

Runs wherever the GPU extension is importable: a real device times the GPU chain, a
host-shim build times the CPU emulation (then the GPU number is only indicative, but
the path and the GPU==CPU sample agreement are validated -- see tests/).

    uv run python -m bench.sampler_throughput --modes 6 --num-samples 2000 --cutoff 6
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

import gbskernels
from bench import _provenance
from sampling import sampler

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "results" / "sampling"


def _random_cov(modes: int, seed: int, hbar: float = 2.0) -> np.ndarray:
    """xxpp covariance of a zero-displacement pure GBS state (squeeze + interferometer),
    built without third-party deps so the benchmark is self-contained on the box."""
    g = np.random.default_rng(seed)
    r = g.uniform(0.3, 0.6, modes)
    z = (g.standard_normal((modes, modes)) + 1j * g.standard_normal((modes, modes))) / np.sqrt(2.0)
    U, rr = np.linalg.qr(z); ph = np.diagonal(rr).copy(); ph /= np.abs(ph); U = U * ph
    sq = np.block([[np.diag(np.exp(-r)), np.zeros((modes, modes))],
                   [np.zeros((modes, modes)), np.diag(np.exp(r))]])        # xxpp squeezing
    interf = np.block([[U.real, -U.imag], [U.imag, U.real]])               # xxpp interferometer
    S = interf @ sq
    return (hbar / 2.0) * S @ S.T


def _quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    x = q * (len(s) - 1)
    lo = int(x); frac = x - lo
    return s[lo] * (1 - frac) + s[lo + 1] * frac if lo + 1 < len(s) else s[lo]


def run(modes: int, num_samples: int, cutoff: int, repeats: int = 7, seed: int = 0,
        out_dir: Path | None = None, repeated_sieve: bool = False) -> tuple[dict[str, Any], Path]:
    cov = _random_cov(modes, seed)
    backends = ["cpu"] + (["gpu"] if gbskernels.gpu_available() else [])
    rows = []

    def _time(backend: str, resident: bool = False, sieve: bool | None = None) -> dict[str, Any]:
        # warm-up (untimed) then RAW repetitions -> median + IQR (NOT best-of-N, which hides the
        # spread and biases the headline; the median+IQR matches the kernel/Walrus benches).
        # The warm-up is also a COST PROBE: if one timed draw-set already exceeds
        # SLOW_CELL_S, drop to a single measured repetition rather than `repeats`
        # of them -- the deep-cutoff CPU non-sieve baseline is ~minutes/rep and
        # timing it 7x (twice: default + explicit nosieve) made the sweep
        # intractable (it hung a cell ~2h). A slow baseline still gets ONE honest
        # number; it just does not get the full IQR treatment it cannot afford.
        SLOW_CELL_S = 20.0
        import time as _t
        sampler.sample(cov, max(num_samples // 10, 1), cutoff=cutoff, backend=backend,
                       resident=resident, seed=seed, repeated_sieve=sieve)
        _p0 = _t.perf_counter()
        first = sampler.samples_per_second(cov, num_samples, cutoff=cutoff, backend=backend,
                                           resident=resident, seed=seed, repeated_sieve=sieve)
        n_rep = repeats if (_t.perf_counter() - _p0) < SLOW_CELL_S else 1
        reps = [first] + [sampler.samples_per_second(cov, num_samples, cutoff=cutoff, backend=backend,
                                           resident=resident, seed=seed + rep,
                                           repeated_sieve=sieve) for rep in range(1, n_rep)]
        sps = sorted(r["samples_per_sec"] for r in reps)
        secs = [r["seconds"] for r in reps]
        return {"backend": reps[0]["backend"], "num_samples": num_samples, "modes": modes,
                "cutoff": cutoff, "repeats_requested": repeats, "repeats_timed": len(reps),
                "repeated_sieve_effective": reps[0].get("repeated_sieve_effective"),
                "samples_per_sec_median": _quantile(sps, 0.5),
                "samples_per_sec_iqr": _quantile(sps, 0.75) - _quantile(sps, 0.25),
                "seconds_median": _quantile(secs, 0.5),
                "mean_photons": reps[0]["mean_photons"], "raw_samples_per_sec": sps}

    for b in backends:
        rows.append(_time(b))                # sieve=None: the DEFAULT path, as shipped
    if repeated_sieve:                       # R4 A/B rows: explicitly pinned paths
        for b in backends:
            rows.append(_time(b, sieve=False))
            rows.append(_time(b, sieve=True))
    # v3 fully on-device (resident) chain -- the before/after vs the hybrid 'gpu' backend -- when
    # the extension exposes it and the config is within the hafnian cap (worst-case 2*modes*cutoff).
    _ext = gbskernels._load_gpu_ext()
    _cap = getattr(gbskernels, "_GPU_MAX_DIM", {}).get("haf", 20)
    if (gbskernels.gpu_available() and _ext is not None and hasattr(_ext, "sample_resident")
            and 2 * modes * cutoff <= _cap):
        rows.append(_time("gpu", resident=True))

    artifact = {
        "kind": "sampler_throughput",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **_provenance.provenance(),  # commit + container_digest + hostname
        "gpu_backend": gbskernels.gpu_backend_kind(),
        "metric": "end-to-end samples/sec (chain-rule conditional GBS sampler); "
                  "median + IQR over raw repeats; warm-up discarded; the 'gpu-resident' row is the "
                  "v3 FULLY on-device chain (before/after vs the hybrid 'gpu' row, where in-cap)",
        "params": {"modes": modes, "num_samples": num_samples, "cutoff": cutoff,
                   "repeats": repeats, "seed": seed, "warmup": "1 untimed draw/backend"},
        "env": {"platform": platform.platform(), "numpy": version("numpy")},
        "rows": rows,
    }
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"sampler_throughput_{stamp}.json"
    path.write_text(json.dumps(artifact, indent=2))
    return artifact, path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--cutoff", type=int, default=6)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--repeated-sieve", action="store_true",
                   help="also time the sieve-routed chain (cpu+sieve / gpu+sieve rows)")
    p.add_argument("--sweep", action="store_true",
                   help="characterization grid: modes x cutoff cells (each its own artifact) "
                        "instead of the single default cell -- the resident/hybrid/sieve "
                        "story needs the surface, not one point")
    args = p.parse_args()
    if args.sweep:
        # the characterization surface: shallow->deep collisions, small->mid modes.
        # Each cell is an independent append-only artifact (same hygiene).
        for m_, c_ in [(4, 3), (6, 4), (6, 6), (8, 5), (10, 4), (5, 8)]:
            n_ = max(200, args.num_samples // 4)
            print(f"--- sweep cell: modes={m_} cutoff={c_} samples={n_} ---")
            run(m_, n_, c_, max(3, args.repeats // 2), out_dir=args.out,
                repeated_sieve=args.repeated_sieve)
        return
    artifact, path = run(args.modes, args.num_samples, args.cutoff, args.repeats,
                         out_dir=args.out, repeated_sieve=args.repeated_sieve)
    print(f"# GBS sampler throughput ({artifact['gpu_backend']}) -> {path}")
    print(f"#   {args.modes} modes, cutoff {args.cutoff}, {args.num_samples} samples, "
          f"{args.repeats} repeats; commit {artifact['commit']}")
    for r in artifact["rows"]:
        print(f"  {r['backend']:>4}  {r['samples_per_sec_median']:>10.1f} samples/sec median  "
              f"(IQR {r['samples_per_sec_iqr']:.1f}, {r['seconds_median']:.2f}s, "
              f"mean photons {r['mean_photons']:.2f})")


if __name__ == "__main__":
    main()
