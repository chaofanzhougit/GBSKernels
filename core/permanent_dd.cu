// permanent_dd.cu -- batched permanent via Glynn/BB-FG in double-double (DD).
//
// STATUS: validated on-device (RTX 4090) AND via the CPU host pre-flight; gate via
// check_permanent_dd in a rented-GPU session. The DD precision tier (anchor
// sec.6, "the project's most novel contribution"): GPUs have no native quad,
// so each value is an unevaluated sum of two FP64 (dd.cuh error-free
// transforms). This restores accuracy where the FP64 Glynn alternating sum
// cancels. Faithful transcription of the *validated* CPU reference
// (cpu_ref/permanent.py :: permanent_glynn_dd, tested vs mpmath on the tunable
// cancellation family). Same Gray-code walk as permanent.cu, but every
// add/sub/mul runs in DD.
//
// Output is the DD result collapsed to cuDoubleComplex (hi+lo -> double): the
// internal arithmetic avoids the cancellation, then returns a fully-accurate
// FP64 answer. One evaluation per thread; batch across the grid (docs/DESIGN.md §5).

#include <cuComplex.h>
#include <cstdint>

#include "dd.cuh"
#include "subset_engine.cuh"

namespace gbs {

constexpr int PERM_DD_MAX_N = 28;

__host__ __device__ inline ddcomplex ddc_from_cu(cuDoubleComplex z) {
  return ddcomplex{dd{cuCreal(z), 0.0}, dd{cuCimag(z), 0.0}};
}
__host__ __device__ inline cuDoubleComplex ddc_to_cu(ddcomplex a) {
  return make_cuDoubleComplex(a.re.hi + a.re.lo, a.im.hi + a.im.lo);
}

__global__ void perm_glynn_dd_kernel(const cuDoubleComplex* __restrict__ mats,
                                     int n, int batch,
                                     cuDoubleComplex* __restrict__ out) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  const cuDoubleComplex* A = mats + (size_t)b * n * n;

  if (n == 0) { out[b] = make_cuDoubleComplex(1.0, 0.0); return; }
  if (n > PERM_DD_MAX_N) { out[b] = make_cuDoubleComplex(NAN, NAN); return; }

  ddcomplex rowsum[PERM_DD_MAX_N];
  // delta = all +1 -> rowsum[r] = sum_c A[r, c] (in DD)
  for (int r = 0; r < n; ++r) {
    ddcomplex s = ddc_from_cu(make_cuDoubleComplex(0.0, 0.0));
    for (int c = 0; c < n; ++c) s = ddc_add(s, ddc_from_cu(A[r * n + c]));
    rowsum[r] = s;
  }

  ddcomplex prod = rowsum[0];
  for (int r = 1; r < n; ++r) prod = ddc_mul(prod, rowsum[r]);
  ddcomplex total = prod;

  int sign = 1;
  const uint64_t terms = 1ull << (n - 1);
  for (uint64_t i = 1; i < terms; ++i) {
    int k = flipped_index(i);
    int col = k + 1;
    double step = gray_bit_of(i, k) ? -2.0 : +2.0;
    for (int r = 0; r < n; ++r)
      rowsum[r] = ddc_add(rowsum[r], ddc_mul_d(ddc_from_cu(A[r * n + col]), step));
    sign = -sign;

    ddcomplex p = rowsum[0];
    for (int r = 1; r < n; ++r) p = ddc_mul(p, rowsum[r]);
    total = sign > 0 ? ddc_add(total, p) : ddc_sub(total, p);
  }

  total = ddc_mul_d(total, 1.0 / (double)terms);
  out[b] = ddc_to_cu(total);
}

extern "C" void gbs_perm_glynn_dd_batched(const cuDoubleComplex* d_mats, int n,
                                          int batch, cuDoubleComplex* d_out,
                                          cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 128;
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(perm_glynn_dd_kernel, grid, block, stream, d_mats, n, batch, d_out);
}

} // namespace gbs
