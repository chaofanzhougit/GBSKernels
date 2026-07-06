# Contributing to GBSKernels

Thanks for your interest! Bug reports, accuracy anomalies, and kernel
improvements are all welcome. This page tells you how to get a working setup,
what the verification bar is, and how to send changes that can be merged.

## Getting set up

```bash
git clone https://github.com/chaofanzhougit/GBSKernels
cd GBSKernels
uv sync                      # Python 3.12 env + deps + editable install
uv run pytest -m "not slow"  # the fast verification tier (~5–10 min)
uv run pytest                # everything (adds the statistical/oracle tier)
```

No GPU is needed for development: every CUDA kernel also compiles and runs on
the CPU through the host shim (`core/preflight/`, exercised by the test suite),
and the GPU extension has a host-shim build (`bindings/README.md`) so the whole
Python→kernel path is testable on any machine.

## The ground rules (what review checks)

The project's design document is [`docs/DESIGN.md`](docs/DESIGN.md) — code
comments cite it as "docs/DESIGN.md §N". The two rules that shape every change:

1. **Verification is the product** (§8). Every kernel change must keep the
   five-layer contract green, and anything new needs tests at the layers it
   touches. In particular, Layer 1 (independent combinatorial ground truth —
   matching counts, closed forms) must pass *before* differential agreement
   with The Walrus is even considered; agreeing with the incumbent is not the
   bar, agreeing with mathematics is.
2. **Measure, don't assume** (§6, §9). Precision claims come from the measured
   FP64↔DD boundary against the mpmath reference; performance claims come from
   on-device A/B runs with the protocol in `docs/benchmark_protocol.md`
   (median+IQR, matched sizes, checksum honesty guard). A negative result
   (a "faster" kernel that measures slower) is a result — we record it and
   don't ship the variant. Optimizations are welcome, but they land
   *not-dispatched* until a real-device measurement justifies routing the
   public API to them.

Operational conventions:

- **CPU-first**: nothing runs on a (rented) GPU until its CPU dry-run is green.
- **`results/` is append-only** — never edit or delete an artifact; new runs
  add new timestamped files.
- **No timings from CI** — shared runners never produce a published number;
  timing artifacts come from the scripted GPU sessions (`scripts/`).
- Match the surrounding code's style; there is no formatter config — keep
  diffs minimal and readable.

## Sending a change

1. Fork, branch, make the change with its tests.
2. `uv run pytest` locally (both tiers pass on a laptop; the slow tier takes
   longer but needs nothing special).
3. If you touched CUDA code, also run the host pre-flight
   (`core/preflight/run_preflight.sh`, or just the test suite, which invokes
   it) — it compiles the kernels as plain C++ and runs the differential gates
   on CPU.
4. Open a PR describing *what* changed and *how it is verified* (which layers,
   which gates, which measurements). CI runs the fast and slow CPU tiers plus a
   wheel-install smoke test.

For kernel/performance work, please include the measurement (or say plainly
that it is not yet measured on a device): an A/B against the per-thread
baseline at matched sizes, median + IQR, randomized order, with a checksum
guard, holding one variable at a time.

## Reporting issues

Open a GitHub issue with:

- the input (or a generator seed) and the call — function, `precision`,
  `backend`;
- what you got vs. what you expected — for accuracy issues, ideally against
  `precision="ref"` (the built-in mpmath ground truth), which makes a
  suspected-wrong-value report self-contained;
- for GPU issues: GPU model, driver, CUDA toolkit version, and whether the
  differential gates (`core/build/check_*`) pass on your box.

Suspected accuracy problems are the most valuable reports this project can
receive — the measured precision boundary is its central claim, so
counterexamples get priority.

## Scope

GBSKernels is deliberately a *kernel library*: the four matrix functions,
their precision tiers, batching, and the thin sampling layer used to validate
them (`docs/DESIGN.md` §4). Circuit simulation, application suites (graph
kernels, vibronic spectra), autodiff, and multi-GPU are out of scope — but
interoperability requests (e.g., consuming the DLPack-resident outputs from
your framework) are very much in scope.
