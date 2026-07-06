# results/ — raw accuracy and throughput data (append-only)

This directory holds the raw benchmark and gate artifacts. It is append-only:
never edit or delete a file, only add new ones. Each run writes a timestamped,
self-describing JSON artifact.

## Provenance

Every artifact records the information needed to reproduce it
([`docs/benchmark_protocol.md`](../docs/benchmark_protocol.md)):

- the **commit** it was produced at (`GBS_COMMIT`);
- the **machine environment** — GPU model, driver, CUDA version, CPU, and
  BLAS-thread caps (`bench/_provenance.py`);
- the **input seed**;
- the **container digest** of the image the run used, which is mandatory for an
  official (publishable) run — `scripts/launch_session.sh` aborts without one.

Timing artifacts come only from scripted GPU sessions on dedicated hardware,
never from a shared CI runner ([`docs/DESIGN.md`](../docs/DESIGN.md) §8).

## Layout

```
results/
├── accuracy/     relative error vs (size, conditioning, precision tier), per regime
├── throughput/   evaluations/sec vs (batch, matrix size); end-to-end, baseline, crossover
├── sampling/     end-to-end samples/sec (median + interquartile range)
├── validation/   certified-validation artifacts (log-likelihood ratios, error bounds)
├── tightness/    certified-bound tightness distributions
└── perf/         static resource usage and profiler captures (see perf/README.md)
```

Files named `*_PROFILER_FAILED.csv` under `perf/` are Nsight Compute captures
that ran without counter permission and carry no hardware-counter data; see
[`perf/README.md`](perf/README.md).
