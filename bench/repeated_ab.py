"""R4 device A/B: the repeated-row sieve kernel vs the expanded hafnian kernel.

Times, on the SAME device through the public path (both pay H2D/launch/D2H),
the two ways of evaluating the identical workload — a batch of loop hafnians of
``A`` expanded by collision patterns ``(q, ..., q)``:

* ``gbskernels.lhaf_repeated(A, 0, reps, backend="gpu")`` — the sieve
  (``core/repeated.cu``), cost ``prod(q+1)`` per element;
* ``gbskernels.haf_batched(expanded, backend="gpu")`` — the shipped power-trace
  kernel on the ``N = M*q`` expansion, cost ``2^(N/2)`` — only where ``N`` is
  within its cap (the rows above the cap are the capability gap the sieve
  closes, reported as such, not as a timing).

Hygiene per docs/benchmark_protocol.md: warm-up before timing, raw repeats with
median + IQR, a checksum honesty guard (sieve vs expanded values must agree),
append-only artifact with full provenance. The MEASURED crossover here decides
whether the sampler's sieve path defaults on for the GPU chain
(the finite-difference sieve, measured against the expanded-hafnian kernel).

    python -m bench.repeated_ab --batch 2048 --qs 2,3,4,5,6 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import gbskernels
from bench._provenance import provenance

RESULTS = Path(__file__).resolve().parent.parent / "results" / "throughput"


def _median_iqr(ts: list[float]) -> dict[str, float]:
    a = np.sort(np.asarray(ts))
    return {"median_s": float(np.median(a)),
            "iqr_s": float(np.percentile(a, 75) - np.percentile(a, 25)),
            "raw_s": [float(x) for x in a]}


def run(modes: int, qs: list[int], batch: int, repeats: int, warmup: int,
        seed: int) -> dict:
    M = 2 * modes
    g = np.random.default_rng(seed)
    A = g.standard_normal((M, M)) + 1j * g.standard_normal((M, M))
    A = 0.3 * (A + A.T)                      # modest norm, physical-ish
    gam = np.zeros(M, dtype=np.complex128)

    rows = []
    for q in qs:
        reps = np.full((batch, M), q, dtype=np.int32)
        N = M * q
        row: dict = {"q": q, "M": M, "N_expanded": N, "batch": batch,
                     "sieve_terms": int((q + 1) ** M)}

        # --- sieve kernel ---
        for _ in range(warmup):
            gbskernels.lhaf_repeated(A, gam, reps[: max(1, batch // 8)], backend="gpu")
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            v_sieve = gbskernels.lhaf_repeated(A, gam, reps, backend="gpu")
            ts.append(time.perf_counter() - t0)
        chk = complex(np.sum(v_sieve))       # post-timing checksum (honesty guard)
        row["sieve"] = _median_iqr(ts) | {"evals_per_s": batch / float(np.median(ts)),
                                          "checksum": [chk.real, chk.imag]}

        # --- expanded power-trace kernel (only within its cap) ---
        cap = 20                              # HAF_MAX_N
        if N <= cap:
            idx = [i for i in range(M) for _ in range(q)]
            E = np.ascontiguousarray(A[np.ix_(idx, idx)])
            stack = np.broadcast_to(E, (batch, N, N)).copy()
            for _ in range(warmup):
                gbskernels.haf_batched(stack[: max(1, batch // 8)], backend="gpu")
            ts = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                v_exp = gbskernels.haf_batched(stack, backend="gpu")
                ts.append(time.perf_counter() - t0)
            chk_e = complex(np.sum(v_exp))
            row["expanded"] = _median_iqr(ts) | {
                "evals_per_s": batch / float(np.median(ts)),
                "checksum": [chk_e.real, chk_e.imag]}
            rel = abs(chk - chk_e) / max(abs(chk_e), 1e-300)
            row["values_agree_rel"] = float(rel)
            if not rel <= 1e-8:
                raise SystemExit(f"HONESTY GATE: sieve vs expanded disagree at q={q}: {rel:.2e}")
            row["speedup_sieve_over_expanded"] = (
                row["expanded"]["median_s"] / row["sieve"]["median_s"])
        else:
            row["expanded"] = None            # over the expanded kernel's cap:
            row["note"] = f"N={N} > HAF cap {cap}: the sieve evaluates what the expansion cannot"
        rows.append(row)
        print(f"q={q}: sieve {row['sieve']['evals_per_s']:.3g} ev/s"
              + (f", expanded {row['expanded']['evals_per_s']:.3g} ev/s, "
                 f"speedup {row['speedup_sieve_over_expanded']:.2f}x"
                 if row["expanded"] else f"  ({row['note']})"))
    return {"bench": "repeated_ab", "params": {"modes": modes, "batch": batch,
            "repeats": repeats, "warmup": warmup, "seed": seed},
            "gpu_backend_kind": gbskernels.gpu_backend_kind(),
            "provenance": provenance(), "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modes", type=int, default=3)
    p.add_argument("--qs", type=str, default="2,3,4,5,6")
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    art = run(args.modes, [int(x) for x in args.qs.split(",")], args.batch,
              args.repeats, args.warmup, args.seed)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"repeated_ab_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    out.write_text(json.dumps(art, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
