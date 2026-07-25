# GBSKernels

A GPU-native, batched library of the #P-hard matrix functions behind photonic
quantum sampling — the **permanent, hafnian, loop hafnian, and torontonian** —
with an explicit floating-point accuracy model that ranges from native double
precision to rigorous a-posteriori error bounds.

Gaussian boson sampling and its displaced- and threshold-detector variants, and
standard boson sampling, reduce computationally to evaluating *large batches of
independent, medium-sized* instances of these four functions. Each is an
alternating signed sum whose value can be many orders of magnitude smaller than
its terms, so it is cancellation-prone in floating point: both throughput and
*known* accuracy matter. GBSKernels targets exactly this workload — batched-first
kernels on CPU and CUDA, every function validated against independent
combinatorial ground truth, and a precision model in which the accuracy of every
result is either measured or provably bounded.

GBSKernels is designed to **complement**
[The Walrus](https://github.com/XanaduAI/thewalrus), the canonical CPU/C++
reference implementation, rather than to replace it: a GPU-native, batched
companion with an explicit accuracy characterization. The Walrus is used here as
one of several independent test oracles.

> **Status.** The CPU library is complete and `pip`-installable (pure Python,
> `numpy` + `mpmath` only). The v0.2.1 GitHub release includes a GPU validation
> manifest covering all 24 device gates, the Python binding smoke, adversarial
> and physical enclosure families, and the Jiuzhang Gate C probe. The session is
> release-commit-attributed through the environment-supplied `GBS_COMMIT`. The
> rsynced host had no Git metadata, so `tracked_dirty` is unknown; the evidence
> hashes and container digest do not hash-bind the uploaded source tree or loaded
> CUDA extension. Historical v0.2.0 RTX 4090 and A100 evidence is retained
> separately; see the [device evidence](results/README.md#v020-device-validation).
> This is pre-1.0 software and the public API may change.

Version 0.2.1 is a correctness patch that adds subnormal-aware certified bounds
and fail-closed certificate handling, makes the single-large real torontonian
input contract explicit, removes inversion roundoff skew from the xxpp
construction, records whether artifacts came from modified tracked sources,
and packages the corrected historical Jiuzhang fixed-sample audit. That audit
is exploratory, not a retrospective confirmatory result. See the
[changelog](CHANGELOG.md) for the release scope.

## Installation

The CPU package needs only `numpy` + `mpmath`. PyPI publication of v0.2.1 is
pending; until it is live, install the exact wheel attached to the
[GitHub release](https://github.com/chaofanzhougit/GBSKernels/releases/tag/v0.2.1).
Once publication completes, the equivalent pinned PyPI requirement is
`gbskernels==0.2.1`.
The CUDA extension is a separate, optional source build (see
[`bindings/README.md`](bindings/README.md)):

```bash
python -m pip install https://github.com/chaofanzhougit/GBSKernels/releases/download/v0.2.1/gbskernels-0.2.1-py3-none-any.whl
```

## Quick start

```python
import numpy as np
import gbskernels

A = np.array([[1., 2.], [3., 4.]])
gbskernels.perm(A)                              # (10+0j)

# The API is batched-first: one call evaluates a stack of matrices.
rng = np.random.default_rng(0)
stack = rng.standard_normal((4096, 8, 8))
stack = stack + stack.transpose(0, 2, 1)        # symmetric, as the hafnian needs

gbskernels.haf_batched(stack)                   # double precision (default)
gbskernels.haf_batched(stack, precision="auto") # double precision + cancellation guard
gbskernels.haf_batched(stack, backend="gpu")    # CUDA, if the extension is built
gbskernels.haf(stack[0], precision="ref")       # arbitrary-precision reference
```

A short guided tour — closed-form checks, the `auto` tier catching real
double-precision cancellation, a GBS distribution, and a sampling run — runs
with wheel dependencies only:

```bash
python examples/gbs_demo.py
```

## Precision model

Precision is always an explicit tier, so the accuracy of a result is never left
implicit:

| Tier | Description |
|---|---|
| `"fp64"` (default) | Native double precision — the throughput path, with a measured accuracy boundary. |
| `"dd"` | Double-double arithmetic carried internally through the cancelling sum (a GPU tier); recovers a correct double-precision result where plain double precision cancels. |
| `"ref"` | Arbitrary-precision reference (`mpmath`); the accuracy ground truth. |
| `"auto"` | Double precision plus a per-evaluation cancellation indicator; evaluations flagged as risky are recomputed in a higher tier (`mpmath` on CPU, double-double on GPU). The indicator is a calibrated heuristic, not a certificate. |
| `"certified"` | The double-precision value together with a **rigorous a-posteriori error bound**: `\|value − exact\| ≤ abs_error_bound`, including explicit absolute terms for subnormal underflow (on the GPU, bound arithmetic uses per-instruction directed rounding). Non-finite inputs and invalid/non-finite certificates fail closed. Available for all four functions on CPU and GPU. With `rtol=`, a failed or refused certified-fp64 result escalates to the arbitrary-precision reference; GPU permanent and hafnian first try a certified double-double bound. The reference result is flagged as non-certified. `tor_single(..., dd=True)` provides a separate certified-double-double large-torontonian path. |

## Features

- **All four functions, batched, on CPU and CUDA.** Field-standard algorithms
  (Glynn/BB–FG for the permanent, the power-trace form for the hafnian and loop
  hafnian, subset-determinant inclusion–exclusion for the torontonian), each with
  an independent naive reference used for verification.
- **A measured accuracy boundary.** The double-precision–vs–double-double
  crossover is characterized for every function against an arbitrary-precision
  reference, on physical and adversarial inputs.
- **Certified evaluation.** Every value can be returned with a rigorous error
  bound whose enclosure of the true value is a hard test invariant — verified on
  closed forms, random ensembles, and adversarial cancellation families.
- **A device-resident workspace.** `gbskernels.Workspace` evaluates ragged
  batches with a persistent stream, pinned staging, and a zero-copy DLPack output
  that CuPy / PyTorch / JAX can consume without a device-to-host copy
  ([`docs/device_resident_contract.md`](docs/device_resident_contract.md)).
- **A conditional GBS sampler** (`sampling/`) that draws photon-number samples by
  the chain rule, evaluating each mode's batch of hafnians on the GPU, validated
  distributionally (total-variation distance below 0.03 against both the exact
  distribution and The Walrus, and a chi-square goodness-of-fit p > 10⁻³ against
  The Walrus).
- **Structure-aware kernels** for the sampling workload: a repeated-row
  finite-difference sieve for the loop hafnian, and a recursive prefix-Cholesky
  torontonian — including a single-large mode that splits one evaluation across
  the GPU to dimension 64 (32 modes).
- **A fail-closed confirmatory workflow** for the public Jiuzhang example,
  covering exposure audit, frozen design, future-beacon selection,
  content-addressed evaluation, refusal recovery, simultaneous coherence-grid
  inference, predictive checks, and release verification
  ([protocol](docs/confirmatory_v2.md),
  [tools](examples/jiuzhang/README.md)). The repository ships templates and
  validation contracts, not a fabricated registration or completed v2 outcome.

## Verification

Verification is treated as the deliverable ([`docs/DESIGN.md`](docs/DESIGN.md), §8).
Five layers are exercised by the test suite on every commit:

1. **Independent combinatorial ground truth** — perfect-matching counts and
   closed forms (`perm(J_n) = n!`, `haf(J_{2n}) = (2n−1)!!`, …), sharing no code
   with any existing library.
2. **Differential testing against The Walrus** across sizes, seeds, and
   real/complex inputs.
3. **Property-based invariants** (Hypothesis): scaling, permutation, transpose,
   the loop-hafnian → hafnian reduction, and others.
4. **Statistical / end-to-end validation** — the kernels compute GBS
   distributions that sum to one and match The Walrus per pattern in all three
   detector regimes.
5. **Numerical-accuracy characterization** against the arbitrary-precision
   reference, including adversarial cancellation families that defeat double
   precision and must survive double-double.

The Jiuzhang v2 tooling adds workflow-contract tests for canonical hashing,
registration timing, exclusion-ledger completeness, selection, immutable run
reduction, refusal recovery, inference, reconstruction, and release assembly.

To run the public v0.2.1 GPU validation from a fresh clone, first stage the
small hash-bound inputs (the raw USTC and Zenodo archives remain
external; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for data
attribution and licensing scope):

```bash
python scripts/prepare_validation_data.py
```

The development discipline is CPU-first: host-shim-compatible CUDA kernels are
compiled and run through the CPU pre-flight (part of the test suite), then the
full source set is checked on-device against independent CPU references before
any throughput result is considered publishable. Device-only variants such as
the warp-specialized permanent are covered by the on-device gates; profiler
diagnostics collected before those gates remain provisional until they pass.

## Performance

The library is batched-first: throughput comes from evaluating thousands of
independent instances per launch, and the benchmark harness reports where the
CPU wins (small single evaluations) as plainly as where the GPU wins (large
batches), always alongside the accuracy each point was measured at. The
structure-aware kernels give substantial speedups in their regimes — the
recursive torontonian over the LU form, and the finite-difference sieve over the
expanded power-trace at repeated-row patterns. Raw benchmark data is written,
append-only, to `results/`; the protocol is described in
[`docs/benchmark_protocol.md`](docs/benchmark_protocol.md).

```bash
python -m bench.accuracy                          # the accuracy boundary, all four functions
python -m bench.throughput --func perm --sizes 4,6,8,10
bash scripts/gpu_session.sh                       # full on-device session (on a CUDA host)
```

## Scope and limitations

These are correctness-first kernels with a deliberately narrow scope:

- **One evaluation per GPU thread** is the default mapping. Warp/block-cooperative
  variants exist and are dispatched only where they measurably win (the
  cooperative permanent); for the hafnian, loop hafnian, and torontonian the
  per-thread linear algebra is memory-bound, and the cooperative strategy does
  not help them.
- **The four functions do not share a single subset-enumeration engine.** Only
  the permanent uses the Gray-code delta walk; the others enumerate masks
  independently, because their per-subset work is memory-bound.
- **Fixed size limits** (per-thread buffers): permanent ≤ 28, hafnian ≤ 20 (DD
  ≤ 16), loop hafnian ≤ 20 (DD ≤ 14), torontonian 2n ≤ 24. Larger inputs raise on
  `backend="gpu"`; use the CPU backend. The single-large recursive torontonian
  extends to dimension 64.
- **The double-double tier is internal**: kernels carry double-double through the
  cancelling summation and collapse to `complex128` on output — they recover a
  correct double-precision answer under cancellation rather than exposing
  extended-precision results. The double-double torontonian is real-input only.
- **The sampler is pure-state and hybrid host-orchestrated** (the host drives the
  chain rule; the GPU evaluates each step's batch of hafnians). A fully
  on-device variant is available but limited to double precision within a
  submatrix-size cap.
- **Single GPU, CUDA only** — no multi-GPU, ROCm, or automatic differentiation.

## Repository layout

```
gbskernels/     batched-first Python API (the public surface)
cpu_ref/        independent CPU reference implementations (double + double-double)
highprec_ref/   arbitrary-precision reference (mpmath)
core/           CUDA C++: subset-enumeration utilities, the kernels, on-device gates
bindings/       nanobind extension exposing the GPU backend to Python
sampling/       boson-sampling orchestration and the conditional GBS sampler
examples/       runnable CPU demo plus Jiuzhang reproduction and v2 workflow tools
tests/          the five-layer verification suite
bench/          accuracy and throughput harnesses
scripts/ envs/  scripted GPU-session runner and container definitions
docs/           design, benchmark/device contracts, and v2 protocol/templates
results/        append-only validation, gate, benchmark, and profiler artifacts
```

## Development

```bash
uv sync                      # Python 3.12 environment, dependencies, editable install
uv run pytest                # the CPU verification suite
uv run pytest -m layer1      # the independent combinatorial ground truth
uv run pytest -m "not slow"  # the fast tier
```

Building the CUDA extension requires `nvcc` and is a separate, optional step
([`bindings/README.md`](bindings/README.md)); the pure-Python wheel never
requires it. The full design rationale — algorithms, precision strategy, and the
verification contract — is in [`docs/DESIGN.md`](docs/DESIGN.md).

## Relationship to The Walrus

GBSKernels would not exist without [The Walrus](https://github.com/XanaduAI/thewalrus),
which remains the canonical reference implementation and this library's primary
differential oracle. The scope here is deliberately narrower: the four batched
kernels and their precision characterization.

## License

[Apache-2.0](LICENSE).
