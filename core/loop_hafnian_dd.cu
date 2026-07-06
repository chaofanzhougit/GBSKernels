// loop_hafnian_dd.cu -- batched loop hafnian (power-trace) in double-double.
//
// STATUS: validated on-device (RTX 4090) AND via the CPU host pre-flight; gate check_loop_hafnian_dd.
//   The DD precision tier (docs/DESIGN.md §6) for the loop hafnian: the hafnian DD
//   kernel plus the diagonal (loop) term v_k = (1/2) d^T X C^{k-1} d, all in
//   double-double (dd.cuh). Faithful transcription of the validated FP64
//   loop_hafnian.cu; the recurrence's divide-by-j uses dd_div_d. Even-N path
//   (odd N is the CPU naive reference). Output collapsed to cuDoubleComplex.

#include <cuComplex.h>
#include <cstdint>

#include "dd.cuh"
#include "subset_engine.cuh"

namespace gbs {

constexpr int LHAF_DD_MAX_N = 14;  // matrix side N (n = N/2 <= 7); DD buffers are heavy

__host__ __device__ inline ddcomplex ldd_from_cu(cuDoubleComplex z) {
  return ddcomplex{dd{cuCreal(z), 0.0}, dd{cuCimag(z), 0.0}};
}
__host__ __device__ inline cuDoubleComplex ldd_to_cu(ddcomplex a) {
  return make_cuDoubleComplex(a.re.hi + a.re.lo, a.im.hi + a.im.lo);
}

// [lambda^n] exp(sum_k g_k lambda^k) in DD, given kg[k] = k*g_k.
__device__ ddcomplex exp_coeff_from_kg_dd(const ddcomplex* kg, int n) {
  ddcomplex zero = ldd_from_cu(make_cuDoubleComplex(0.0, 0.0));
  ddcomplex e[LHAF_DD_MAX_N + 1];
  e[0] = ldd_from_cu(make_cuDoubleComplex(1.0, 0.0));
  for (int j = 1; j <= n; ++j) {
    ddcomplex acc = zero;
    for (int k = 1; k <= j; ++k) acc = ddc_add(acc, ddc_mul(kg[k], e[j - k]));
    e[j] = ddc_div_d(acc, (double)j);
  }
  return e[n];
}

__global__ void loop_haf_dd_kernel(const cuDoubleComplex* __restrict__ mats, int N,
                                   int batch, cuDoubleComplex* __restrict__ out) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  const cuDoubleComplex* A = mats + (size_t)b * N * N;
  if (N == 0) { out[b] = make_cuDoubleComplex(1.0, 0.0); return; }
  if (N & 1) { out[b] = make_cuDoubleComplex(NAN, NAN); return; }  // even-N path only
  if (N > LHAF_DD_MAX_N) { out[b] = make_cuDoubleComplex(NAN, NAN); return; }
  int n = N / 2;

  ddcomplex zero = ldd_from_cu(make_cuDoubleComplex(0.0, 0.0));
  ddcomplex total = zero;
  for (uint64_t mask = 0; mask < (1ull << n); ++mask) {
    int m = __popcll((long long)mask);
    if (m == 0) continue;
    int size = 2 * m;
    int pidx[LHAF_DD_MAX_N / 2];
    int pc = 0;
    for (int i = 0; i < n; ++i) if ((mask >> i) & 1ull) pidx[pc++] = i;

    ddcomplex C[LHAF_DD_MAX_N * LHAF_DD_MAX_N];
    ddcomplex d[LHAF_DD_MAX_N];
    for (int r = 0; r < size; ++r) {
      int rr = 2 * pidx[r >> 1] + (r & 1);
      for (int c = 0; c < size; ++c) {
        int cc = 2 * pidx[c >> 1] + ((c & 1) ^ 1);  // X swap within the pair
        C[r * size + c] = ldd_from_cu(A[rr * N + cc]);
      }
      d[r] = ldd_from_cu(A[rr * N + rr]);  // diagonal loop weight
    }

    ddcomplex kg[LHAF_DD_MAX_N + 1];
    ddcomplex P[LHAF_DD_MAX_N * LHAF_DD_MAX_N];
    ddcomplex Q[LHAF_DD_MAX_N * LHAF_DD_MAX_N];
    for (int i = 0; i < size * size; ++i) { P[i] = C[i]; Q[i] = zero; }
    for (int i = 0; i < size; ++i) Q[i * size + i] = ldd_from_cu(make_cuDoubleComplex(1.0, 0.0));

    for (int k = 1; k <= n; ++k) {
      ddcomplex pk = zero;
      for (int i = 0; i < size; ++i) pk = ddc_add(pk, P[i * size + i]);  // tr(C^k)

      // v_k = (1/2) d^T X (C^{k-1} d).  w = Q d (Q = C^{k-1}); then X swaps rows.
      ddcomplex w[LHAF_DD_MAX_N];
      for (int a = 0; a < size; ++a) {
        ddcomplex s = zero;
        for (int c = 0; c < size; ++c) s = ddc_add(s, ddc_mul(Q[a * size + c], d[c]));
        w[a] = s;
      }
      ddcomplex vk = zero;
      for (int a = 0; a < size; ++a) vk = ddc_add(vk, ddc_mul(d[a], w[a ^ 1]));
      vk = ddc_mul_d(vk, 0.5);

      // kg[k] = p_k/2 + k*v_k
      kg[k] = ddc_add(ddc_mul_d(pk, 0.5), ddc_mul_d(vk, (double)k));

      if (k < n) {  // Q <- C^k, P <- C^{k+1}
        for (int i = 0; i < size * size; ++i) Q[i] = P[i];
        ddcomplex T[LHAF_DD_MAX_N * LHAF_DD_MAX_N];
        for (int i = 0; i < size; ++i)
          for (int j = 0; j < size; ++j) {
            ddcomplex s = zero;
            for (int t = 0; t < size; ++t) s = ddc_add(s, ddc_mul(P[i * size + t], C[t * size + j]));
            T[i * size + j] = s;
          }
        for (int i = 0; i < size * size; ++i) P[i] = T[i];
      }
    }

    ddcomplex coeff = exp_coeff_from_kg_dd(kg, n);
    if ((n - m) & 1) total = ddc_sub(total, coeff);
    else             total = ddc_add(total, coeff);
  }
  out[b] = ldd_to_cu(total);
}

extern "C" void gbs_loop_haf_dd_batched(const cuDoubleComplex* d_mats, int N,
                                        int batch, cuDoubleComplex* d_out,
                                        cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 32;
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(loop_haf_dd_kernel, grid, block, stream, d_mats, N, batch, d_out);
}

}  // namespace gbs
