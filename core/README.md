# core/ — CUDA C++ kernels

The four functions in native double, double-double, and directed-rounding
certified variants, plus the subset utilities, host-facing wrappers, sampler
kernels, and on-device differential gates.

| File | Purpose |
|---|---|
| `subset_engine.cuh` | Subset-enumeration utilities, the Gray-code walk (used by the permanent), and the launch macro. |
| `dd.cuh` | Double-double arithmetic (error-free transforms, division, square root, complex double-double). |
| `certified_rounding.cuh` | Shared upward/downward-directed scalar operations used to construct rigorous error bounds. |
| `permanent.cu`, `permanent_dd.cu` | Batched Glynn/BB–FG permanent, double and double-double. |
| `permanent_coop.cu`, `permanent_coop.cuh`, `permanent_warp.cu` | Cooperative and warp-specialized permanent implementations and dispatch support. |
| `hafnian.cu`, `hafnian_dd.cu` | Batched power-trace hafnian (even N), double and double-double. |
| `loop_hafnian.cu`, `loop_hafnian_dd.cu` | Batched power-trace loop hafnian, double and double-double. |
| `torontonian.cu`, `torontonian_dd.cu` | Batched torontonian (subset determinants; double-double is real-input). |
| `repeated.cu` | Repeated-row loop hafnian (finite-difference sieve), plain and certified. |
| `tor_recursive.cu` | Recursive prefix-Cholesky torontonian, including FP64- and double-double-certified single-large enclosures and collapse accounting. |
| `certified.cu`, `certified_dd.cu` | Certified kernels: value plus a rigorous error bound in directed rounding. |
| `sampler_draw.cu`, `sampler_gather.cu`, `sampler_session.cu` | Device-side conditional draws, ragged gathers, and the resident sampling chain. |
| `host_api.cu` | Host-pointer wrappers (host→device, launch, synchronize, device→host) with checked CUDA returns. |
| `check_*.cu` | GPU-versus-CPU-reference differential gates, including the precision gates. |
| `bench_kernels.cu` | Kernel-only throughput timing harness. |

**Validation is two-stage and CPU-first.** Host-shim-compatible kernels compile
and run on the CPU (`core/preflight/`, exercised by the test suite via
`run_preflight.sh`), then the full source set passes on-device differential gates
against independent CPU references. Device-only variants, including
`permanent_warp.cu`, are validated in the CUDA session rather than the host shim.

**Certified arithmetic is relative-plus-absolute.** Every recurrence includes
an explicit binary64 subnormal floor, and mixed-scale products propagate that
absolute error through later factors. Double-double division and square root
are bounded from outward FMA residuals of the values actually returned rather
than assumed forward constants. Unsafe division ranges and any non-finite or
negative certificate refuse instead of producing a finite claim; see
[`docs/dd_certificate_proof.md`](../docs/dd_certificate_proof.md).

**Execution model.** The default kernels evaluate one instance per thread and
batch across the grid. Only the permanent uses the Gray-code delta walk; the
hafnian, loop hafnian, and torontonian enumerate the `2^k` subsets independently,
because their per-subset linear algebra is memory-bound (several n×n buffers per
thread) rather than enumeration-bound. Warp/block-cooperative variants exist for
all four and are dispatched only where they measurably win — the cooperative
permanent; for the other three the cooperative strategy does not help. A
size-specialized hafnian (buffers sized to a smaller cap) is dispatched for small
inputs.

**Size limits** (fixed per-thread buffers): permanent ≤ 28, hafnian ≤ 20 (double-
double ≤ 16), loop hafnian ≤ 20 (double-double ≤ 14), torontonian 2n ≤ 24; larger
inputs are rejected at the Python boundary. The single-large recursive torontonian
extends to dimension 64 and requires a finite matrix that is exactly symmetric
after conversion to binary64, because its Cholesky walk consumes one triangle.

### Build and run

```bash
cmake -S core -B core/build -DCMAKE_CUDA_ARCHITECTURES=89   # e.g. 4090 / Ada
cmake --build core/build -j
./core/build/check_permanent    # and the other gates; each must print PASS
```

`scripts/gpu_session.sh` runs the whole sequence (pre-flight, build, gates,
bindings, throughput) on a CUDA host.
