// permanent_coop.cu -- warp/block-COOPERATIVE batched permanent (Glynn, FP64).
//
// The first cooperative kernel (docs/DESIGN.md §5, "one evaluation per warp/block").
// permanent.cu maps one matrix per thread, so a single thread walks all 2^(n-1)
// Glynn terms serially -- at large n that serial exponential is exactly where its
// throughput collapses. Here a GROUP of `groups` threads cooperates on one matrix:
// each walks a contiguous Gray-code sub-range of the term space into a partial, and
// a second pass sums the partials. The exponential subset sum is thus parallelized
// ~`groups`x.
//
// MAP/REDUCE over a global partials buffer -- deliberately NOT a shared-memory or
// __shfl warp reduction -- so the whole thing is validated on the CPU host shim
// before any paid GPU session. The shim runs threads serially, which faithfully
// emulates two independent-thread kernels exchanging data through global memory,
// but cannot emulate lockstep warp shuffles (a thread would read lanes that have
// not run yet). The fused shuffle/shared-mem variant is a GPU-session micro-opt on
// top of this, validated there by its own differential gate.
//
// A warp/block-cooperative permanent (validated against the per-thread kernel
// in check_permanent_coop.cu). Dispatched for large n, where it wins; small n
// uses the per-thread permanent.cu.

#include <cuComplex.h>
#include <cstdint>

#include "subset_engine.cuh"
#include "permanent_coop.cuh"   // PERM_COOP_MAX_N, cscale_c, glynn_range_partial

namespace gbs {

// MAP: thread t -> matrix b = t/groups, group g = t%groups. Sum this group's
// contiguous Gray-code sub-range [start,end) of the 2^(n-1) term space into the
// (unscaled) signed partial. The per-lane math is shared with the fused warp
// kernel via glynn_range_partial (permanent_coop.cuh).
__global__ void perm_glynn_coop_map_kernel(const cuDoubleComplex* __restrict__ mats,
                                           int n, int batch, int groups,
                                           cuDoubleComplex* __restrict__ partials) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= batch * groups) return;
  const int b = t / groups, g = t % groups;
  const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

  // n==0: perm of a 0x0 matrix = 1, carried by group 0 (reduce scale is 1).
  if (n == 0) { partials[t] = (g == 0) ? make_cuDoubleComplex(1.0, 0.0) : zero; return; }
  if (n > PERM_COOP_MAX_N) { partials[t] = make_cuDoubleComplex(NAN, NAN); return; }

  const cuDoubleComplex* A = mats + (size_t)b * n * n;
  const uint64_t terms = 1ull << (n - 1);
  const uint64_t R = (terms + groups - 1) / groups;   // ceil: contiguous range size
  uint64_t start = (uint64_t)g * R, end = start + R;
  if (end > terms) end = terms;
  partials[t] = glynn_range_partial(A, n, start, end);  // 0 for an empty (remainder) range
}

// REDUCE: thread b sums the `groups` partials for matrix b (fixed order ->
// deterministic) and applies the Glynn 1/2^(n-1) scale.
__global__ void perm_glynn_coop_reduce_kernel(const cuDoubleComplex* __restrict__ partials,
                                              int batch, int groups, double scale,
                                              cuDoubleComplex* __restrict__ out) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  cuDoubleComplex s = make_cuDoubleComplex(0.0, 0.0);
  for (int g = 0; g < groups; ++g) s = cuCadd(s, partials[(size_t)b * groups + g]);
  out[b] = cscale_c(s, scale);
}

// Host wrapper: allocate the partials buffer, launch map then reduce on `stream`
// (the reduce reads the map's output in stream order), sync, free. `groups` is the
// cooperation width (clamped to [1, 2^(n-1)]). Returns 0 or a nonzero cudaError_t.
extern "C" int gbs_perm_glynn_coop_batched(const cuDoubleComplex* d_mats, int n,
                                           int batch, int groups,
                                           cuDoubleComplex* d_out, cudaStream_t stream) {
  if (batch <= 0) return 0;
  if (groups < 1) groups = 1;
  if (n >= 1) {
    const uint64_t terms = 1ull << (n - 1);
    if ((uint64_t)groups > terms) groups = (int)terms;   // no more groups than terms
  } else {
    groups = 1;
  }
  const double scale = (n >= 1) ? 1.0 / (double)(1ull << (n - 1)) : 1.0;

  cuDoubleComplex* d_partials = nullptr;
  cudaError_t err = cudaMalloc(&d_partials, (size_t)batch * groups * sizeof(cuDoubleComplex));
  if (err != cudaSuccess) return (int)err;

  const int block = 128;
  const int map_grid = (batch * groups + block - 1) / block;
  GBS_LAUNCH_1D(perm_glynn_coop_map_kernel, map_grid, block, stream,
                d_mats, n, batch, groups, d_partials);
  err = cudaGetLastError();   // map launch errors
  if (err == cudaSuccess) {
    const int red_grid = (batch + block - 1) / block;
    GBS_LAUNCH_1D(perm_glynn_coop_reduce_kernel, red_grid, block, stream,
                  d_partials, batch, groups, scale, d_out);
    err = cudaGetLastError();   // reduce launch errors
  }
  if (err == cudaSuccess) err = cudaDeviceSynchronize();   // execution errors (+ finish before free)
  cudaFree(d_partials);
  return (int)err;
}

}  // namespace gbs
