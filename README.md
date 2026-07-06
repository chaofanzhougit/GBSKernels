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
> `numpy` + `mpmath` only). The CUDA kernels are validated on NVIDIA hardware
> (RTX 4090, A100). This is pre-1.0 software and the public API may change.

## Installation

```bash
pip install gbskernels          # once published — pulls only numpy + mpmath
```

Or from source (CPU backend; the CUDA extension is a separate, optional build —
see [`bindings/README.md`](bindings/README.md)):

```bash
uv build                        # -> dist/gbskernels-*.whl
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
| `"certified"` | The double-precision value together with a **rigorous a-posteriori error bound**: `\|value − exact\| ≤ abs_error_bound`, a running error bound in the standard model of floating-point arithmetic (on the GPU, the bound arithmetic uses per-instruction directed rounding). With `rtol=`, a bound-driven ladder escalates certified-fp64 → certified-double-double → arbitrary precision, each step justified by the bound rather than a heuristic. Available for all four functions on CPU and GPU. |

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
  distributionally against The Walrus (total-variation distance and a chi-square
  test).
- **Structure-aware kernels** for the sampling workload: a repeated-row
  finite-difference sieve for the loop hafnian, and a recursive prefix-Cholesky
  torontonian — including a single-large mode that splits one evaluation across
  the GPU to dimension 64 (32 modes).

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

The development discipline is CPU-first: each CUDA kernel is first compiled and
run on the CPU through a host pre-flight (part of the test suite), then checked
on-device against the independent CPU reference before any performance number is
recorded.

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
examples/       runnable end-to-end demo (CPU only, wheel dependencies only)
tests/          the five-layer verification suite
bench/          accuracy and throughput harnesses
scripts/ envs/  scripted GPU-session runner and container definitions
docs/           design document, benchmark protocol, device contracts
results/        raw accuracy and throughput artifacts (append-only)
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
