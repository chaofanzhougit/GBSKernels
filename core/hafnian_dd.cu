// hafnian_dd.cu -- batched hafnian via the power-trace algorithm in double-double.
//
// STATUS: validated on-device (RTX 4090) AND via the CPU host pre-flight; gate check_hafnian_dd.
//   a rented-GPU session. The DD precision tier (docs/DESIGN.md §6) for the hardest
//   kernel: every entry, matrix multiply, trace, and the exp/Newton recurrence
//   run in double-double (dd.cuh), restoring accuracy where the FP64 power-trace
//   subset sum cancels. Faithful transcription of the *validated* FP64 algorithm
//   (hafnian.cu / cpu_ref hafnian_powertrace); the only structural difference is
//   that the recurrence's divide-by-j uses dd_div_d (1/j is not exact, so a plain
//   scale would cap precision at FP64).
//
// Output is the DD result collapsed to cuDoubleComplex. One hafnian per thread;
// the per-thread double-double matrix buffers are large, so the block is small
// and the size cap modest -- DD is the precision path, not the throughput path.

#include <cuComplex.h>
#include <cstdint>

#include "dd.cuh"
#include "subset_engine.cuh"

namespace gbs {

constexpr int HAF_DD_MAX_N = 16;  // matrix side N (n = N/2 <= 8); DD buffers are 4x FP64

__host__ __device__ inline ddcomplex hdd_from_cu(cuDoubleComplex z) {
  return ddcomplex{dd{cuCreal(z), 0.0}, dd{cuCimag(z), 0.0}};
}
__host__ __device__ inline cuDoubleComplex hdd_to_cu(ddcomplex a) {
  return make_cuDoubleComplex(a.re.hi + a.re.lo, a.im.hi + a.im.lo);
}

// [lambda^n] exp(sum_k tr(C^k)/(2k) lambda^k) in double-double.
__device__ ddcomplex exp_newton_coeff_dd(const ddcomplex* C, int size, int n) {
  ddcomplex zero = hdd_from_cu(make_cuDoubleComplex(0.0, 0.0));
  ddcomplex one = hdd_from_cu(make_cuDoubleComplex(1.0, 0.0));
  if (size == 0) return (n == 0) ? one : zero;

  ddcomplex p[HAF_DD_MAX_N + 1];
  ddcomplex P[HAF_DD_MAX_N * HAF_DD_MAX_N];
  for (int i = 0; i < size * size; ++i) P[i] = C[i];

  for (int k = 1; k <= n; ++k) {
    ddcomplex tr = zero;
    for (int i = 0; i < size; ++i) tr = ddc_add(tr, P[i * size + i]);
    p[k] = tr;
    if (k < n) {  // P <- P * C
      ddcomplex T[HAF_DD_MAX_N * HAF_DD_MAX_N];
      for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j) {
          ddcomplex s = zero;
          for (int t = 0; t < size; ++t)
            s = ddc_add(s, ddc_mul(P[i * size + t], C[t * size + j]));
          T[i * size + j] = s;
        }
      for (int i = 0; i < size * size; ++i) P[i] = T[i];
    }
  }

  ddcomplex e[HAF_DD_MAX_N + 1];
  e[0] = one;
  for (int j = 1; j <= n; ++j) {
    ddcomplex acc = zero;
    for (int k = 1; k <= j; ++k)
      acc = ddc_add(acc, ddc_mul(ddc_mul_d(p[k], 0.5), e[j - k]));
    e[j] = ddc_div_d(acc, (double)j);  // 1/j not exact -> proper DD division
  }
  return e[n];
}

__global__ void haf_powertrace_dd_kernel(const cuDoubleComplex* __restrict__ mats,
                                         int N, int batch,
                                         cuDoubleComplex* __restrict__ out) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  const cuDoubleComplex* A = mats + (size_t)b * N * N;

  if (N == 0) { out[b] = make_cuDoubleComplex(1.0, 0.0); return; }
  if (N & 1)  { out[b] = make_cuDoubleComplex(0.0, 0.0); return; }  // haf of odd size = 0
  if (N > HAF_DD_MAX_N) { out[b] = make_cuDoubleComplex(NAN, NAN); return; }
  int n = N / 2;

  ddcomplex total = hdd_from_cu(make_cuDoubleComplex(0.0, 0.0));
  for (uint64_t mask = 0; mask < (1ull << n); ++mask) {
    int m = __popcll((long long)mask);
    int size = 2 * m;

    ddcomplex BX[HAF_DD_MAX_N * HAF_DD_MAX_N];
    int pidx[HAF_DD_MAX_N / 2];
    int pc = 0;
    for (int i = 0; i < n; ++i) if ((mask >> i) & 1ull) pidx[pc++] = i;

    for (int r = 0; r < size; ++r) {
      int pr = pidx[r >> 1], rr = 2 * pr + (r & 1);
      for (int c = 0; c < size; ++c) {
        int pc2 = pidx[c >> 1];
        int cc = 2 * pc2 + ((c & 1) ^ 1);  // X swap within the pair
        BX[r * size + c] = hdd_from_cu(A[rr * N + cc]);
      }
    }

    ddcomplex coeff = exp_newton_coeff_dd(BX, size, n);
    if ((n - m) & 1) total = ddc_sub(total, coeff);
    else             total = ddc_add(total, coeff);
  }
  out[b] = hdd_to_cu(total);
}

extern "C" void gbs_haf_powertrace_dd_batched(const cuDoubleComplex* d_mats, int N,
                                              int batch, cuDoubleComplex* d_out,
                                              cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 32;  // DD power-trace has a large per-thread footprint
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(haf_powertrace_dd_kernel, grid, block, stream, d_mats, N, batch, d_out);
}

}  // namespace gbs
