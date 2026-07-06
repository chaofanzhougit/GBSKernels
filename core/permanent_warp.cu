// permanent_warp.cu -- FUSED warp-cooperative permanent (one warp per matrix).
//
// The fused variant of permanent_coop.cu: one warp (32 lanes) cooperates on a
// single permanent. Each lane sums a contiguous Gray-code range into a partial
// (the host-validated glynn_range_partial), then __shfl_down_sync tree-reduces the
// 32 lane partials IN-WARP -- no global partials buffer and no second kernel, so it
// avoids the map/reduce round trip the cooperative kernel pays.
//
// GPU-ONLY. The CPU host shim runs threads serially and cannot emulate lockstep
// warp shuffles (a lane would read peers that have not run), so this kernel is NOT
// in run_preflight.sh; it is built by nvcc and validated on-device by
// check_permanent_warp.cu in a rented-GPU session (scripts/gpu_session.sh). Its
// per-lane math is the same host-validated glynn_range_partial as permanent_coop.cu,
// so only the __shfl reduction is device-only. Whether it beats the map/reduce
// cooperative kernel is measured on-device.

#include <cuComplex.h>
#include <cstdint>

#include "subset_engine.cuh"
#include "permanent_coop.cuh"   // glynn_range_partial, cscale_c, PERM_COOP_MAX_N

namespace gbs {

constexpr unsigned PERM_WARP_LANES = 32;

__global__ void perm_glynn_warp_kernel(const cuDoubleComplex* __restrict__ mats, int n,
                                       int batch, cuDoubleComplex* __restrict__ out) {
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const int warp = gtid / (int)PERM_WARP_LANES;
  const int lane = gtid & (int)(PERM_WARP_LANES - 1);
  if (warp >= batch) return;

  if (n == 0)              { if (lane == 0) out[warp] = make_cuDoubleComplex(1.0, 0.0); return; }
  if (n > PERM_COOP_MAX_N) { if (lane == 0) out[warp] = make_cuDoubleComplex(NAN, NAN); return; }

  const cuDoubleComplex* A = mats + (size_t)warp * n * n;
  const uint64_t terms = 1ull << (n - 1);
  const uint64_t R = (terms + PERM_WARP_LANES - 1) / PERM_WARP_LANES;   // ceil
  uint64_t start = (uint64_t)lane * R, end = start + R;
  if (end > terms) end = terms;
  cuDoubleComplex acc = glynn_range_partial(A, n, start, end);   // this lane's partial (0 if empty)

  // in-warp tree reduction (sum) of the 32 lane partials; real+imag shuffled apart.
  for (int off = (int)PERM_WARP_LANES / 2; off > 0; off >>= 1) {
    double re = __shfl_down_sync(0xffffffffu, cuCreal(acc), off);
    double im = __shfl_down_sync(0xffffffffu, cuCimag(acc), off);
    acc = cuCadd(acc, make_cuDoubleComplex(re, im));
  }
  if (lane == 0) out[warp] = cscale_c(acc, 1.0 / (double)terms);
}

extern "C" int gbs_perm_glynn_warp_batched(const cuDoubleComplex* d_mats, int n, int batch,
                                           cuDoubleComplex* d_out, cudaStream_t stream) {
  if (batch <= 0) return 0;
  const int warps_per_block = 4;                          // 128 threads / block
  const int block = warps_per_block * (int)PERM_WARP_LANES;
  const int grid = (batch + warps_per_block - 1) / warps_per_block;
  GBS_LAUNCH_1D(perm_glynn_warp_kernel, grid, block, stream, d_mats, n, batch, d_out);
  return (int)cudaGetLastError();   // launch errors (caller syncs for execution errors)
}

}  // namespace gbs
