// torontonian_dd.cu -- batched torontonian in double-double (real-O domain).
//
// STATUS: validated on-device (RTX 4090) AND via the CPU host pre-flight; gate check_torontonian_dd.
//   The DD precision tier (docs/DESIGN.md §6) for the torontonian. Unlike the
//   power-trace kernels, the per-subset work is a determinant of I - O_S and a
//   1/sqrt; the torontonian's physical inputs are real O, so this runs in real
//   double-double (dd.cuh: dd_det via Gauss elimination, dd_sqrt, dd_div).
//   Faithful transcription of the validated FP64 torontonian.cu. Output is real
//   (the imaginary part is zero on the physical domain), collapsed to FP64.

#include <cuComplex.h>
#include <cstdint>

#include "dd.cuh"
#include "subset_engine.cuh"

namespace gbs {

constexpr int TOR_DD_MAX_DIM = 24;  // max 2|S|

// det of an mm x mm real-DD matrix via Gaussian elimination w/ partial pivoting.
// M is overwritten.
__device__ dd dd_det(dd* M, int mm) {
  dd det = dd{1.0, 0.0};
  for (int c = 0; c < mm; ++c) {
    int piv = c;
    double best = fabs(M[c * mm + c].hi);
    for (int r = c + 1; r < mm; ++r) {
      double v = fabs(M[r * mm + c].hi);
      if (v > best) { best = v; piv = r; }
    }
    if (best == 0.0) return dd{0.0, 0.0};
    if (piv != c) {
      for (int j = 0; j < mm; ++j) { dd t = M[c * mm + j]; M[c * mm + j] = M[piv * mm + j]; M[piv * mm + j] = t; }
      det = dd_neg(det);
    }
    dd pivot = M[c * mm + c];
    det = dd_mul(det, pivot);
    for (int r = c + 1; r < mm; ++r) {
      dd f = dd_div(M[r * mm + c], pivot);
      for (int j = c; j < mm; ++j)
        M[r * mm + j] = dd_add(M[r * mm + j], dd_neg(dd_mul(f, M[c * mm + j])));
    }
  }
  return det;
}

__global__ void tor_dd_kernel(const cuDoubleComplex* __restrict__ mats, int n,
                              int batch, cuDoubleComplex* __restrict__ out) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  const cuDoubleComplex* O = mats + (size_t)b * (2 * n) * (2 * n);
  int N = 2 * n;
  if (n == 0) { out[b] = make_cuDoubleComplex(1.0, 0.0); return; }
  if (2 * n > TOR_DD_MAX_DIM) { out[b] = make_cuDoubleComplex(NAN, NAN); return; }

  dd total = dd{0.0, 0.0};
  for (uint64_t mask = 0; mask < (1ull << n); ++mask) {
    int m = __popcll((long long)mask);
    dd term;
    if (m == 0) {
      term = dd{1.0, 0.0};  // 1/sqrt(det I) = 1
    } else {
      int idx[TOR_DD_MAX_DIM];
      int cnt = 0;
      for (int i = 0; i < n; ++i) if ((mask >> i) & 1ull) idx[cnt++] = i;      // x-indices
      for (int i = 0; i < n; ++i) if ((mask >> i) & 1ull) idx[cnt++] = i + n;  // p-indices
      int mm = 2 * m;
      dd sub[TOR_DD_MAX_DIM * TOR_DD_MAX_DIM];
      for (int r = 0; r < mm; ++r)
        for (int c = 0; c < mm; ++c) {
          double v = cuCreal(O[idx[r] * N + idx[c]]);  // real-O domain
          if (r == c) {
            double e;
            double hi = two_sum(1.0, -v, e);  // 1 - O[r,r] captured exactly in DD
            sub[r * mm + c] = dd{hi, e};
          } else {
            sub[r * mm + c] = dd{-v, 0.0};
          }
        }
      dd det = dd_det(sub, mm);
      term = dd_div(dd{1.0, 0.0}, dd_sqrt(det));  // 1/sqrt(det)
    }
    if ((n - m) & 1) total = dd_add(total, dd_neg(term));
    else             total = dd_add(total, term);
  }
  out[b] = make_cuDoubleComplex(total.hi + total.lo, 0.0);
}

extern "C" void gbs_tor_dd_batched(const cuDoubleComplex* d_mats, int n, int batch,
                                   cuDoubleComplex* d_out, cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 64;
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(tor_dd_kernel, grid, block, stream, d_mats, n, batch, d_out);
}

}  // namespace gbs
