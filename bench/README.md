# bench/ — accuracy and throughput harnesses

The measured quantity is **throughput at a stated accuracy**, never raw
throughput ([`docs/DESIGN.md`](../docs/DESIGN.md) §9; the protocol is
[`docs/benchmark_protocol.md`](../docs/benchmark_protocol.md)).

| Module | What it measures |
|---|---|
| `_inputs.py` | The single shared workload generator `bench_batch(func, dim, batch, regime, seed)` over the `physical` / `loss` / `adversarial` regimes, so CPU, GPU, and The Walrus are timed on identical inputs per `(func, dim, regime)`. |
| `accuracy.py`, `accuracy_permanent.py` | Double-precision (and double-double) relative error versus size and conditioning against the arbitrary-precision reference — the measured accuracy boundary for all four functions. |
| `calibrate_auto.py` | Calibrates the `auto` tier's cancellation indicator against measured FP64 error on physical, loss, and adversarial ensembles; it reports false-trusts rather than treating the heuristic as a certificate. |
| `throughput.py`, `throughput_gpu.py` | Batched evaluations/sec: `throughput.py` is the CPU baseline (accuracy-normalized, versus The Walrus); `throughput_gpu.py` (with `core/bench_kernels.cu`) is the kernel-only GPU measurement. |
| `throughput_end_to_end.py` | Public-API GPU throughput including host↔device transfers, versus the CPU backend. The GPU-vs-CPU checksum is a gate: a physical-regime disagreement fails the run. |
| `walrus_baseline.py` | A same-instance The Walrus per-evaluation baseline on the same inputs. |
| `torontonian_baselines.py` | The prospective publication comparison: one frozen real xxpp corpus is evaluated by the public GBSKernels double-double GPU call and the recursive The Walrus and Piquasso CPU calls at 4, 8, 12, 16, and 20 modes. |
| `crossover.py`, `plot_crossover.py` | A batch-size sweep producing the GPU-vs-CPU-vs-Walrus crossover, each series tagged with the accuracy it achieved. |
| `sampler_throughput.py` | End-to-end GBS samples/sec for the conditional sampler. |
| `tightness.py` | The distribution of certified-bound tightness across ensembles, physical versus adversarial. |
| `kernel_footprint.py` | Static per-thread kernel-buffer accounting used to separate source-level footprint from device-only profiler evidence. |
| `repeated_ab.py` | Same-device A/B measurement of the repeated-row sieve against expanded hafnians, with checksum and provenance gates. |

Release validation also runs
`examples/jiuzhang/dd_adversarial_enclosure.py` against an independent mpmath
reference. The gate fails on any enclosure violation, empty/all-refusal run,
excessive refusal fraction, or missing required physical-family coverage; its
artifact is hash-bound into the release-validation manifest.

The cross-engine timing harnesses share the same discipline: one input generator
(so timings are same-input), randomized execution order, raw repeats reported as
median and interquartile range (never best-of-N), append-only output to
`results/`, no composite "winner" number, and a post-computation checksum guard
so an asynchronous early return cannot fake a fast time. Static evidence such as
`kernel_footprint.py` reports source facts rather than timing repeats.

`torontonian_baselines.py` is narrower than `walrus_baseline.py` and records more
of the comparison contract. It embeds every canonical little-endian binary64
matrix in the JSON artifact, with per-matrix and whole-workload hashes, and
records package versions, implementation entry points, execution devices,
precision tiers, raw randomized repeats, and median/IQR summaries. GBSKernels
and The Walrus consume the frozen xxpp matrices; Piquasso requires a deterministic
xxpp-to-xpxp row/column permutation, whose output is hashed and whose preparation
is outside the timed region. Pairwise numerical agreement is reported separately
and is not treated as a high-precision oracle or used to discard timing rows.

The registered publication profile is a heterogeneous comparison: certified
double-double GBSKernels runs on GPU, while The Walrus and Piquasso run in CPU
fp64. It therefore measures the stated end-to-end implementations on the same
mathematical inputs; it does not isolate hardware, algorithm, and precision into
a single causal speedup. No result from this new profile is claimed until the
archive-bound GPU session has run and its final manifest has been validated.

```bash
python -m bench.accuracy_permanent --sizes 2-12 --seeds 8   # the accuracy boundary
python -m bench.throughput --func perm --sizes 4,6,8,10      # CPU throughput baseline
python -m bench.torontonian_baselines --modes 4,8,12,16,20 \
  --include-gbskernels-dd --out /new/path/torontonian-baselines.json
```

GPU throughput runs only in a scripted GPU session (`scripts/`); nothing in this
directory requires a GPU by default. The publication driver adds
`--require-provenance` and fixes the corpus size, seed, warmups, and repeats; see
[`scripts/README.md`](../scripts/README.md#publication-evidence-workflow-prospective).
