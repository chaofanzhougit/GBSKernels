# results/ — versioned validation and performance evidence

This directory is append-only: never rewrite an existing artifact. Current
harnesses emit timestamped, self-describing JSON, JSONL, NPZ, text, CSV, or
image artifacts as appropriate. Historical files are retained for audit even
when they predate the current schema or are explicitly marked superseded.

## Provenance

Every new official benchmark or scientific artifact records the applicable
information needed to reproduce it
([`docs/benchmark_protocol.md`](../docs/benchmark_protocol.md)):

- the **commit** it was produced at (`GBS_COMMIT`) plus the local tracked-dirty
  state when Git metadata is available;
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

## Jiuzhang historical fixed-sample audit

`jiuzhang/legacy_fixed_sample/` contains the corrected v0.2.1 audit package:
the hash-bound deterministic selection, the finite-population stratified
summary, and the five historical per-band checkpoint JSONL files used as its
row-level inputs. The selected sample was exposed during exploratory
development. It is not held out, preregistered, or confirmatory.

The checkpoint rows predate the common provenance block and do not contain
recoverable original GPU, container, driver, or commit metadata. Their exact
file hashes are retained; the regenerated selection and summary record the
current implementation and Git state. Missing historical environment fields
are disclosed rather than reconstructed.

## v0.2.0 device validation

The release commit `b2b42c3` completed a fresh RTX 4090 (`sm_89`) session on
2026-07-23 with driver 580.105.08, CUDA 13.0, and Python 3.12. The image was
`vastai/base-image:cuda-13.0.3-auto` at digest
`sha256:8f20625442b1bdbed1ed7ea39005db2120212c3ca7439a809bad798b847e923d`.

- All 24 on-device gates and the nanobind GPU smoke passed
  ([gate summary](gpu_gates/gates_4090_v020_20260723.json)).
- The [kernel benchmark](throughput/throughput_NVIDIA_GeForce_RTX_4090_20260723T012923Z.json)
  contains 73 finite rows; all 11 cells in the
  [physical public-path benchmark](throughput/throughput_e2e_20260723T013248Z.json)
  have matching GPU and CPU checksums.
- The session also recorded [accuracy](accuracy/accuracy_all_20260723T012926Z.json),
  [sampler throughput](sampling/sampler_throughput_20260723T014154Z.json), a strict
  [same-instance crossover](throughput/crossover_20260723T040334Z.json), and a
  [zero-violation certified-bound study](tightness/tightness_20260723T040406Z.json).
- Nsight Compute lacked performance-counter permission; the retained
  `_PROFILER_FAILED.csv` is diagnostic only, while the
  [ptxas capture](perf/ptxas_sm89_20260723T012006Z.txt) remains valid static
  compiler evidence.

The loss-regime public-path checks also agreed. Two adversarial torontonian
cells diverged between FP64 GPU and CPU, as expected for the deliberately
ill-conditioned non-gating regime; the physical honesty gate remained green.

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
