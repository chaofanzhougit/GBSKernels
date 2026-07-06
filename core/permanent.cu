// permanent.cu -- batched permanent via Glynn/BB-FG on the GPU (FP64 tier).
//
// STATUS: validated on-device (RTX 4090, CUDA 12.4, sm_89) AND via the CPU host pre-flight.
//   This is a faithful transcription of the *verified* CPU reference
//   (cpu_ref/permanent.py :: permanent_glynn). The differential check
//   check_permanent.cu must pass (GPU == CPU reference within FP64 tolerance,
//   docs/DESIGN.md §8 Layer 5) in a rented-GPU session before this is trusted.
//
// Mapping (docs/DESIGN.md §5, batched regime -- the priority): one matrix evaluation
// per thread; the batch is spread across the grid. Per-thread cost O(n 2^n).
// This per-thread mapping suits small/medium n; warp- and block-cooperative
// variants for larger n are a later optimization (docs/DESIGN.md §5).

#include <cuComplex.h>
#include <cstdint>

#include "subset_engine.cuh"

namespace gbs {

// Largest n a single thread keeps in its local row-sum buffer. Batched
// permanents target modest n; revisit during GPU tuning.
constexpr int PERM_MAX_N = 28;

__device__ inline cuDoubleComplex cscale(cuDoubleComplex a, double s) {
  return make_cuDoubleComplex(cuCreal(a) * s, cuCimag(a) * s);
}

// out[b] = perm(mats + b*n*n), row-major n x n, complex128. One thread per b.
__global__ void perm_glynn_fp64_kernel(const cuDoubleComplex* __restrict__ mats,
                                       int n, int batch,
                                       cuDoubleComplex* __restrict__ out) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;

  const cuDoubleComplex* A = mats + (size_t)b * n * n;

  if (n == 0) {
    out[b] = make_cuDoubleComplex(1.0, 0.0);
    return;
  }
  if (n > PERM_MAX_N) {  // would overflow the per-thread local buffer; signal NaN
    out[b] = make_cuDoubleComplex(NAN, NAN);
    return;
  }

  cuDoubleComplex rowsum[PERM_MAX_N];
  // delta = all +1  ->  rowsum[r] = sum_c A[r, c]
  for (int r = 0; r < n; ++r) {
    cuDoubleComplex s = make_cuDoubleComplex(0.0, 0.0);
    for (int c = 0; c < n; ++c) s = cuCadd(s, A[r * n + c]);
    rowsum[r] = s;
  }

  // running product of rowsum (delta = all +1)
  cuDoubleComplex prod = rowsum[0];
  for (int r = 1; r < n; ++r) prod = cuCmul(prod, rowsum[r]);
  cuDoubleComplex total = prod;

  int sign = 1;
  const uint64_t terms = 1ull << (n - 1);  // 2^(n-1) sign vectors (delta_0 fixed)
  for (uint64_t i = 1; i < terms; ++i) {
    int k = flipped_index(i);  // varying-bit index in [0, n-2]
    int col = k + 1;           // column 0 is the fixed +1
    double step = gray_bit_of(i, k) ? -2.0 : +2.0;  // delta_col: +1->-1 or -1->+1
    for (int r = 0; r < n; ++r)
      rowsum[r] = cuCadd(rowsum[r], cscale(A[r * n + col], step));
    sign = -sign;

    cuDoubleComplex p = rowsum[0];
    for (int r = 1; r < n; ++r) p = cuCmul(p, rowsum[r]);
    total = sign > 0 ? cuCadd(total, p) : cuCsub(total, p);
  }

  out[b] = cscale(total, 1.0 / (double)terms);
}

// Host launch wrapper. Pointers are device pointers.
extern "C" void gbs_perm_glynn_fp64_batched(const cuDoubleComplex* d_mats, int n,
                                            int batch, cuDoubleComplex* d_out,
                                            cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 128;
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(perm_glynn_fp64_kernel, grid, block, stream, d_mats, n, batch, d_out);
}

// FP64 result PLUS the cancellation indicator, in one pass (precision="auto"): the
// same Glynn walk, but it also accumulates sum|term| (= sum over Gray codes of the
// running product's modulus, scaled by 1/terms to match cpu_ref.glynn_abs_term_sum).
// The host forms kappa = absnorm[b] / |out[b]| and reruns the risky elements in DD --
// so the indicator is computed on the device alongside the fast FP64 result, not by
// a second full pass on the host.
__global__ void perm_glynn_fp64_kappa_kernel(const cuDoubleComplex* __restrict__ mats,
                                             int n, int batch,
                                             cuDoubleComplex* __restrict__ out,
                                             double* __restrict__ absnorm) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  const cuDoubleComplex* A = mats + (size_t)b * n * n;
  if (n == 0) { out[b] = make_cuDoubleComplex(1.0, 0.0); absnorm[b] = 1.0; return; }
  if (n > PERM_MAX_N) { out[b] = make_cuDoubleComplex(NAN, NAN); absnorm[b] = NAN; return; }

  cuDoubleComplex rowsum[PERM_MAX_N];
  for (int r = 0; r < n; ++r) {
    cuDoubleComplex s = make_cuDoubleComplex(0.0, 0.0);
    for (int c = 0; c < n; ++c) s = cuCadd(s, A[r * n + c]);
    rowsum[r] = s;
  }
  cuDoubleComplex prod = rowsum[0];
  for (int r = 1; r < n; ++r) prod = cuCmul(prod, rowsum[r]);
  cuDoubleComplex total = prod;
  double abs_acc = cuCabs(prod);   // sum |term|

  int sign = 1;
  const uint64_t terms = 1ull << (n - 1);
  for (uint64_t i = 1; i < terms; ++i) {
    int k = flipped_index(i);
    int col = k + 1;
    double step = gray_bit_of(i, k) ? -2.0 : +2.0;
    for (int r = 0; r < n; ++r) rowsum[r] = cuCadd(rowsum[r], cscale(A[r * n + col], step));
    sign = -sign;
    cuDoubleComplex p = rowsum[0];
    for (int r = 1; r < n; ++r) p = cuCmul(p, rowsum[r]);
    total = sign > 0 ? cuCadd(total, p) : cuCsub(total, p);
    abs_acc += cuCabs(p);
  }
  out[b] = cscale(total, 1.0 / (double)terms);
  absnorm[b] = abs_acc / (double)terms;   // same 1/terms scaling as the result
}

extern "C" void gbs_perm_glynn_fp64_kappa_batched(const cuDoubleComplex* d_mats, int n,
                                                  int batch, cuDoubleComplex* d_out,
                                                  double* d_absnorm, cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 128;
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(perm_glynn_fp64_kappa_kernel, grid, block, stream, d_mats, n, batch, d_out, d_absnorm);
}

}  // namespace gbs
