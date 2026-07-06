# core/ — CUDA C++ kernels

The four functions in two precision tiers (double and double-double), the
subset-enumeration utilities they share, the host-facing wrappers the Python
bindings call, and the on-device differential gates.

| File | Purpose |
|---|---|
| `subset_engine.cuh` | Subset-enumeration utilities, the Gray-code walk (used by the permanent), and the launch macro. |
| `dd.cuh` | Double-double arithmetic (error-free transforms, division, square root, complex double-double). |
| `permanent.cu`, `permanent_dd.cu` | Batched Glynn/BB–FG permanent, double and double-double. |
| `hafnian.cu`, `hafnian_dd.cu` | Batched power-trace hafnian (even N), double and double-double. |
| `loop_hafnian.cu`, `loop_hafnian_dd.cu` | Batched power-trace loop hafnian, double and double-double. |
| `torontonian.cu`, `torontonian_dd.cu` | Batched torontonian (subset determinants; double-double is real-input). |
| `repeated.cu` | Repeated-row loop hafnian (finite-difference sieve), plain and certified. |
| `tor_recursive.cu` | Recursive prefix-Cholesky torontonian, batched and single-large. |
| `certified.cu`, `certified_dd.cu` | Certified kernels: value plus a rigorous error bound in directed rounding. |
| `host_api.cu` | Host-pointer wrappers (host→device, launch, synchronize, device→host) with checked CUDA returns. |
| `check_*.cu` | GPU-versus-CPU-reference differential gates, including the precision gates. |
| `bench_kernels.cu` | Kernel-only throughput timing harness. |

**Validation is two-stage and CPU-first.** Every kernel first compiles and runs
on the CPU through a host shim (`core/preflight/`, exercised by the test suite via
`run_preflight.sh`), then passes an on-device differential gate against the
independent CPU reference.

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
extends to dimension 64.

### Build and run

```bash
cmake -S core -B core/build -DCMAKE_CUDA_ARCHITECTURES=89   # e.g. 4090 / Ada
cmake --build core/build -j
./core/build/check_permanent    # and the other gates; each must print PASS
```

`scripts/gpu_session.sh` runs the whole sequence (pre-flight, build, gates,
bindings, throughput) on a CUDA host.
