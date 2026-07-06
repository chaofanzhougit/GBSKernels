// permanent_coop.cuh -- shared per-lane Glynn partial for the cooperative permanent.
//
// The contiguous-range Glynn partial sum is used by BOTH the map/reduce cooperative
// kernel (permanent_coop.cu, host-shim validated) and the fused warp-shuffle kernel
// (permanent_warp.cu, gated on-device). Sharing it as a __device__ inline means the
// fused kernel's per-lane math is already validated on CPU -- only its __shfl
// reduction is device-only.

#pragma once

#include <cuComplex.h>
#include <cstdint>

#include "subset_engine.cuh"

namespace gbs {

constexpr int PERM_COOP_MAX_N = 28;  // matches permanent.cu's per-thread buffer

__device__ inline cuDoubleComplex cscale_c(cuDoubleComplex a, double s) {
  return make_cuDoubleComplex(cuCreal(a) * s, cuCimag(a) * s);
}

// Signed Glynn partial over the contiguous Gray-code term range [start, end) of the
// 2^(n-1) term space: build the row-sum directly at gray(start), then delta-walk to
// end (one O(n) rank-update per Gray step). The 1/terms scaling is the caller's.
// Returns 0 for an empty range (start >= end).
__device__ inline cuDoubleComplex glynn_range_partial(const cuDoubleComplex* __restrict__ A,
                                                      int n, uint64_t start, uint64_t end) {
  if (start >= end) return make_cuDoubleComplex(0.0, 0.0);
  const uint64_t gstart = start ^ (start >> 1);   // gray_code(start)
  cuDoubleComplex rowsum[PERM_COOP_MAX_N];
  for (int r = 0; r < n; ++r) {
    cuDoubleComplex s = A[r * n + 0];   // column 0: delta_0 = +1
    for (int c = 1; c < n; ++c) {
      const double d = ((gstart >> (c - 1)) & 1ull) ? -1.0 : 1.0;
      s = cuCadd(s, cscale_c(A[r * n + c], d));
    }
    rowsum[r] = s;
  }
  // term sign at index i is (-1)^i == (-1)^popcount(gray(i)); seed from `start`.
  int sign = (__popcll((long long)gstart) & 1) ? -1 : 1;
  cuDoubleComplex p = rowsum[0];
  for (int r = 1; r < n; ++r) p = cuCmul(p, rowsum[r]);
  cuDoubleComplex acc = (sign > 0) ? p : make_cuDoubleComplex(-cuCreal(p), -cuCimag(p));

  for (uint64_t i = start + 1; i < end; ++i) {
    const int k = flipped_index(i);
    const int col = k + 1;
    const double step = gray_bit_of(i, k) ? -2.0 : +2.0;
    for (int r = 0; r < n; ++r) rowsum[r] = cuCadd(rowsum[r], cscale_c(A[r * n + col], step));
    sign = -sign;
    cuDoubleComplex q = rowsum[0];
    for (int r = 1; r < n; ++r) q = cuCmul(q, rowsum[r]);
    acc = (sign > 0) ? cuCadd(acc, q) : cuCsub(acc, q);
  }
  return acc;
}

}  // namespace gbs
