# results/ — versioned validation and performance evidence

This directory is append-only: never rewrite an existing artifact. Current
harnesses emit timestamped, self-describing JSON, JSONL, NPZ, text, CSV, or
image artifacts as appropriate. Historical files are retained for audit even
when they predate the current schema or are explicitly marked superseded.

## Provenance

Every new official benchmark or scientific artifact records the applicable
information needed to reproduce it
([`docs/benchmark_protocol.md`](../docs/benchmark_protocol.md)):

- the **commit** it was produced at (`GBS_COMMIT`);
- the **machine environment** — GPU model, driver, CUDA version, CPU, and
  BLAS-thread caps (`bench/_provenance.py`);
- the **input seed** or deterministic selection binding, where applicable;
- the **container digest** of the image the run used, which is mandatory for an
  official (publishable) run — `scripts/launch_session.sh` aborts without one.

Simple gate sentinels, profiler diagnostics, images, and retained legacy files
do not necessarily carry every field themselves; their surrounding session or
README supplies the provenance boundary. Timing artifacts come only from
scripted GPU sessions on dedicated hardware, never from a shared CI runner
([`docs/DESIGN.md`](../docs/DESIGN.md) §8).

## Layout

```
results/
├── accuracy/     relative error vs (size, conditioning, precision tier), per regime
├── throughput/   evaluations/sec vs (batch, matrix size); end-to-end, baseline, crossover
├── sampling/     end-to-end samples/sec (median + interquartile range)
├── validation/   certified-validation artifacts (log-likelihood ratios, error bounds)
├── gpu_gates/    on-device gate summaries and per-gate PASS sentinels
├── confirmatory_v2/  generated content-addressed v2 run directories (local; not shipped)
├── tightness/    certified-bound tightness distributions
└── perf/         static resource usage and profiler captures (see perf/README.md)
```

Files named `*_PROFILER_FAILED.csv` under `perf/` are Nsight Compute captures
that ran without counter permission and carry no hardware-counter data; see
[`perf/README.md`](perf/README.md).
