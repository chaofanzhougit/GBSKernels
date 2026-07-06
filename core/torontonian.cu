// torontonian.cu -- batched torontonian via subset determinants (FP64 tier).
//
// STATUS: validated on-device (RTX 4090, CUDA 12.4, sm_89) AND via the CPU host pre-flight.
//   Faithful transcription of the *verified* CPU reference
//   (cpu_ref/torontonian.py). A check_torontonian.cu differential gate must
//   PASS in a rented-GPU session before this is trusted (docs/DESIGN.md §8, Layer 5).
//
//   tor(O) = sum_{S subset [n]} (-1)^(n-|S|) / sqrt(det(I - O_S))
//
// O is 2n x 2n in xxpp ordering (mode i owns indices i and i+n). One torontonian
// per thread; the batch spreads across the grid. Per-subset kernel is an LU
// determinant of (I - O_S), O(|S|^3); incremental rank-1 Cholesky updates along
// a Gray-code walk are the planned optimization (docs/DESIGN.md §5). The physical
// domain is real O; this FP64 draft computes in complex to match the reference.

#include <cuComplex.h>
#include <cstdint>

#include "subset_engine.cuh"

namespace gbs {

constexpr int TOR_MAX_DIM = 24;  // max 2|S| a thread factorizes; modest by design

__device__ inline cuDoubleComplex cdiv(cuDoubleComplex a, cuDoubleComplex b) { return cuCdiv(a, b); }

// det(M) for an mm x mm complex matrix via Gaussian elimination with partial
// pivoting. M is overwritten. Returns the determinant.
__device__ cuDoubleComplex det_lu(cuDoubleComplex* M, int mm) {
  cuDoubleComplex det = make_cuDoubleComplex(1.0, 0.0);
  for (int c = 0; c < mm; ++c) {
    // pivot: largest |M[r,c]| for r >= c
    int piv = c;
    double best = cuCabs(M[c * mm + c]);
    for (int r = c + 1; r < mm; ++r) {
      double v = cuCabs(M[r * mm + c]);
      if (v > best) { best = v; piv = r; }
    }
    if (best == 0.0) return make_cuDoubleComplex(0.0, 0.0);
    if (piv != c) {  // swap rows -> flips det sign
      for (int j = 0; j < mm; ++j) {
        cuDoubleComplex t = M[c * mm + j];
        M[c * mm + j] = M[piv * mm + j];
        M[piv * mm + j] = t;
      }
      det = make_cuDoubleComplex(-cuCreal(det), -cuCimag(det));
    }
    cuDoubleComplex pivot = M[c * mm + c];
    det = cuCmul(det, pivot);
    for (int r = c + 1; r < mm; ++r) {
      cuDoubleComplex f = cdiv(M[r * mm + c], pivot);
      for (int j = c; j < mm; ++j)
        M[r * mm + j] = cuCsub(M[r * mm + j], cuCmul(f, M[c * mm + j]));
    }
  }
  return det;
}

// Principal complex square root (matches the CPU np.sqrt convention).
__device__ cuDoubleComplex csqrt_principal(cuDoubleComplex z) {
  double r = cuCabs(z);
  double x = cuCreal(z), y = cuCimag(z);
  double re = sqrt(0.5 * (r + x));
  double im = sqrt(0.5 * (r - x));
  if (y < 0.0) im = -im;
  return make_cuDoubleComplex(re, im);
}

// Signed contribution of one subset `mask` to tor(O): (-1)^(n-|S|)/sqrt(det(I-O_S)).
// N = 2n (matrix dim). Shared by the per-thread kernel and the cooperative map.
__device__ cuDoubleComplex tor_subset_term(const cuDoubleComplex* __restrict__ O,
                                           int n, int N, uint64_t mask) {
  int m = __popcll((long long)mask);
  cuDoubleComplex term;
  if (m == 0) {
    term = make_cuDoubleComplex(1.0, 0.0);  // 1/sqrt(det I) = 1
  } else {
    int idx[TOR_MAX_DIM];
    int cnt = 0;
    for (int i = 0; i < n; ++i) if ((mask >> i) & 1ull) idx[cnt++] = i;       // x-indices
    for (int i = 0; i < n; ++i) if ((mask >> i) & 1ull) idx[cnt++] = i + n;   // p-indices
    int mm = 2 * m;
    cuDoubleComplex sub[TOR_MAX_DIM * TOR_MAX_DIM];
    for (int r = 0; r < mm; ++r)
      for (int c = 0; c < mm; ++c) {
        cuDoubleComplex v = O[idx[r] * N + idx[c]];
        if (r == c) sub[r * mm + c] = make_cuDoubleComplex(1.0 - cuCreal(v), -cuCimag(v));
        else        sub[r * mm + c] = make_cuDoubleComplex(-cuCreal(v), -cuCimag(v));   // I - O_S
      }
    cuDoubleComplex det = det_lu(sub, mm);
    term = cdiv(make_cuDoubleComplex(1.0, 0.0), csqrt_principal(det));
  }
  return ((n - m) & 1) ? make_cuDoubleComplex(-cuCreal(term), -cuCimag(term)) : term;
}

__global__ void tor_fp64_kernel(const cuDoubleComplex* __restrict__ mats, int n,
                                int batch, cuDoubleComplex* __restrict__ out) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  const cuDoubleComplex* O = mats + (size_t)b * (2 * n) * (2 * n);
  int N = 2 * n;

  if (n == 0) { out[b] = make_cuDoubleComplex(1.0, 0.0); return; }
  if (2 * n > TOR_MAX_DIM) { out[b] = make_cuDoubleComplex(NAN, NAN); return; }  // submatrix cap

  cuDoubleComplex total = make_cuDoubleComplex(0.0, 0.0);
  for (uint64_t mask = 0; mask < (1ull << n); ++mask)
    total = cuCadd(total, tor_subset_term(O, n, N, mask));
  out[b] = total;
}

extern "C" void gbs_tor_fp64_batched(const cuDoubleComplex* d_mats, int n, int batch,
                                     cuDoubleComplex* d_out, cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 64;
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(tor_fp64_kernel, grid, block, stream, d_mats, n, batch, d_out);
}

// FP64 result + cancellation indicator in one pass (precision="auto"): same subset
// sum, also accumulating sum|term| = sum_S |tor_subset_term| (= sum |1/sqrt(det(I-O_S))|).
// The host forms kappa = absnorm[b] / |out[b]| and reruns the risky elements in DD.
__global__ void tor_fp64_kappa_kernel(const cuDoubleComplex* __restrict__ mats, int n,
                                      int batch, cuDoubleComplex* __restrict__ out,
                                      double* __restrict__ absnorm) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  const cuDoubleComplex* O = mats + (size_t)b * (2 * n) * (2 * n);
  int N = 2 * n;
  if (n == 0) { out[b] = make_cuDoubleComplex(1.0, 0.0); absnorm[b] = 1.0; return; }
  if (2 * n > TOR_MAX_DIM) { out[b] = make_cuDoubleComplex(NAN, NAN); absnorm[b] = NAN; return; }
  cuDoubleComplex total = make_cuDoubleComplex(0.0, 0.0);
  double abs_acc = 0.0;
  for (uint64_t mask = 0; mask < (1ull << n); ++mask) {
    cuDoubleComplex t = tor_subset_term(O, n, N, mask);
    total = cuCadd(total, t);
    abs_acc += cuCabs(t);
  }
  out[b] = total;
  absnorm[b] = abs_acc;
}

extern "C" void gbs_tor_fp64_kappa_batched(const cuDoubleComplex* d_mats, int n, int batch,
                                           cuDoubleComplex* d_out, double* d_absnorm,
                                           cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 64;
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(tor_fp64_kappa_kernel, grid, block, stream, d_mats, n, batch, d_out, d_absnorm);
}

// --- real-Cholesky torontonian (physical domain: O real, I - O_S SPD) -----------
// In the physical domain O is real and I - O_S = (Q^{-1})_S is symmetric positive
// definite (O = I - Q^{-1}, Q the real-symmetric-PD Husimi covariance), so
// det(I - O_S) is real > 0 and sqrt(det) = prod_i L_ii of its real Cholesky factor.
// That is REAL arithmetic (~1/4 the flops of the complex LU) over a REAL `sub`
// buffer (half the footprint), physical SPD domain. The VALUE is
// identical to the complex-LU kernel on real O (check_torontonian_real_chol.cu);
// off the physical domain (I - O_S not SPD) it reports NAN so the host falls back.

constexpr int TOR_REAL_MAX_DIM = TOR_MAX_DIM;  // same submatrix cap, half the bytes/term

// Real Cholesky of an mm x mm symmetric matrix A (lower triangle read + overwritten
// with the factor L). Returns prod_i L_ii = sqrt(det A); 0.0 if A is not positive
// definite (also catches a NaN pivot), so the caller can signal a domain miss rather
// than return a wrong real value.
__device__ inline double tor_chol_sqrtdet(double* A, int mm) {
  double prod = 1.0;
  for (int j = 0; j < mm; ++j) {
    double sum = A[j * mm + j];
    for (int k = 0; k < j; ++k) { double v = A[j * mm + k]; sum -= v * v; }
    if (!(sum > 0.0)) return 0.0;                 // not PD (the !> also rejects NaN)
    double Ljj = sqrt(sum);
    A[j * mm + j] = Ljj;
    prod *= Ljj;
    double inv = 1.0 / Ljj;
    for (int i = j + 1; i < mm; ++i) {
      double s = A[i * mm + j];
      for (int k = 0; k < j; ++k) s -= A[i * mm + k] * A[j * mm + k];
      A[i * mm + j] = s * inv;
    }
  }
  return prod;
}

// Real signed subset contribution (-1)^(n-|S|)/sqrt(det(I - O_S)) via real Cholesky.
// Returns NAN if I - O_S is not SPD (out of the validated physical domain).
__device__ inline double tor_subset_term_real(const double* __restrict__ O,
                                              int n, int N, uint64_t mask) {
  int m = __popcll((long long)mask);
  if (m == 0) return (n & 1) ? -1.0 : 1.0;        // 1/sqrt(det I)=1, sign (-1)^(n-0)
  int idx[TOR_REAL_MAX_DIM];
  int cnt = 0;
  for (int i = 0; i < n; ++i) if ((mask >> i) & 1ull) idx[cnt++] = i;       // x-indices
  for (int i = 0; i < n; ++i) if ((mask >> i) & 1ull) idx[cnt++] = i + n;   // p-indices
  int mm = 2 * m;
  double sub[TOR_REAL_MAX_DIM * TOR_REAL_MAX_DIM];
  for (int r = 0; r < mm; ++r)
    for (int c = 0; c < mm; ++c) {
      double v = O[idx[r] * N + idx[c]];
      sub[r * mm + c] = (r == c) ? (1.0 - v) : (-v);     // I - O_S
    }
  double sd = tor_chol_sqrtdet(sub, mm);
  if (sd == 0.0) return NAN;                       // non-SPD -> domain miss (host falls back)
  double term = 1.0 / sd;
  return ((n - m) & 1) ? -term : term;
}

__global__ void tor_real_chol_fp64_kernel(const double* __restrict__ mats, int n,
                                          int batch, double* __restrict__ out) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  if (n == 0) { out[b] = 1.0; return; }
  if (2 * n > TOR_REAL_MAX_DIM) { out[b] = NAN; return; }                   // submatrix cap
  const double* O = mats + (size_t)b * (2 * n) * (2 * n);
  int N = 2 * n;
  double total = 0.0;
  for (uint64_t mask = 0; mask < (1ull << n); ++mask)
    total += tor_subset_term_real(O, n, N, mask);
  out[b] = total;
}

extern "C" void gbs_tor_real_chol_fp64_batched(const double* d_mats, int n, int batch,
                                               double* d_out, cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 64;
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(tor_real_chol_fp64_kernel, grid, block, stream, d_mats, n, batch, d_out);
}

// --- cooperative (map/reduce) torontonian (perf; see permanent_coop.cu) ----------
// `groups` threads split the 2^n subset sum (each term an O(|S|^3) determinant),
// then a second pass sums the partials. `n` = modes (matrices are 2n x 2n).

__global__ void tor_coop_map_kernel(const cuDoubleComplex* __restrict__ mats, int n,
                                    int batch, int groups,
                                    cuDoubleComplex* __restrict__ partials) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= batch * groups) return;
  int b = t / groups, g = t % groups;
  const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
  if (n == 0)              { partials[t] = (g == 0) ? make_cuDoubleComplex(1.0, 0.0) : zero; return; }
  if (2 * n > TOR_MAX_DIM) { partials[t] = (g == 0) ? make_cuDoubleComplex(NAN, NAN) : zero; return; }

  const cuDoubleComplex* O = mats + (size_t)b * (2 * n) * (2 * n);
  int N = 2 * n;
  uint64_t terms = 1ull << n;
  uint64_t R = (terms + groups - 1) / groups;
  uint64_t start = (uint64_t)g * R, end = start + R;
  if (end > terms) end = terms;
  cuDoubleComplex partial = zero;
  for (uint64_t mask = start; mask < end; ++mask)
    partial = cuCadd(partial, tor_subset_term(O, n, N, mask));
  partials[t] = partial;
}

__global__ void tor_coop_reduce_kernel(const cuDoubleComplex* __restrict__ partials,
                                       int batch, int groups,
                                       cuDoubleComplex* __restrict__ out) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  cuDoubleComplex s = make_cuDoubleComplex(0.0, 0.0);
  for (int g = 0; g < groups; ++g) s = cuCadd(s, partials[(size_t)b * groups + g]);
  out[b] = s;
}

extern "C" int gbs_tor_coop_batched(const cuDoubleComplex* d_mats, int n, int batch,
                                    int groups, cuDoubleComplex* d_out, cudaStream_t stream) {
  if (batch <= 0) return 0;
  if (groups < 1) groups = 1;
  if (n >= 1 && 2 * n <= TOR_MAX_DIM) {
    uint64_t terms = 1ull << n;
    if ((uint64_t)groups > terms) groups = (int)terms;
  } else {
    groups = 1;
  }
  cuDoubleComplex* d_partials = nullptr;
  cudaError_t err = cudaMalloc(&d_partials, (size_t)batch * groups * sizeof(cuDoubleComplex));
  if (err != cudaSuccess) return (int)err;
  const int block = 64;
  int map_grid = (batch * groups + block - 1) / block;
  GBS_LAUNCH_1D(tor_coop_map_kernel, map_grid, block, stream, d_mats, n, batch, groups, d_partials);
  err = cudaGetLastError();
  if (err == cudaSuccess) {
    int red_grid = (batch + block - 1) / block;
    GBS_LAUNCH_1D(tor_coop_reduce_kernel, red_grid, block, stream, d_partials, batch, groups, d_out);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  cudaFree(d_partials);
  return (int)err;
}

}  // namespace gbs
