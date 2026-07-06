// hafnian.cu -- batched hafnian via the power-trace algorithm (FP64 tier).
//
// STATUS: validated on-device (RTX 4090, CUDA 12.4, sm_89) AND via the CPU host pre-flight.
// Faithful transcription of the *verified* CPU reference
// (cpu_ref/hafnian.py :: hafnian_powertrace), the Bjorklund-Cygan-Pilipczuk
// algorithm The Walrus uses. A check_hafnian.cu differential gate (GPU vs CPU
// reference) must PASS in a rented-GPU session before this is trusted
// (docs/DESIGN.md §8, Layer 5).
//
// Mapping: one hafnian per thread; the batch spreads across the grid (anchor
// sec.5, batched regime). Per-thread it walks the 2^n subsets of the n
// index-pairs, builds B_S X, accumulates power traces p_k = tr((B_S X)^k), and
// recovers the lambda^n coefficient of exp(sum_k p_k/(2k) lambda^k) via the
// exp/Newton recurrence. Per-thread cost O(n^3 2^n). Register pressure caps N;
// warp/block-cooperative power-trace variants for larger N are the planned GPU
// optimization (docs/DESIGN.md §5) -- this per-thread form is the correctness-first
// draft, matching permanent.cu.

#include <cuComplex.h>
#include <cstdint>

#include "subset_engine.cuh"

namespace gbs {

// Max matrix dimension N = 2n a single thread holds (B_S X is N x N). Modest by
// design; batched hafnians target small/medium N. Revisit in GPU tuning.
constexpr int HAF_MAX_N = 20;

__device__ inline cuDoubleComplex cadd(cuDoubleComplex a, cuDoubleComplex b) { return cuCadd(a, b); }
__device__ inline cuDoubleComplex cmul(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }

// e_n = [lambda^n] exp(sum_k p_k/2 lambda^k), p_k = tr(C^k), via
// e_0=1, e_j = (1/j) sum_{k=1}^j (p_k/2) e_{j-k}. C is size x size (= B_S X).
//
// Templated on the compile-time buffer cap MAXN: the per-thread local arrays (the
// dominant register-spill / local-memory source -- see bench/kernel_footprint.py) are
// sized to MAXN, so a SIZE-SPECIALIZED instantiation (small MAXN) has a much smaller
// per-thread footprint and higher occupancy for small matrices. The default MAXN =
// HAF_MAX_N reproduces the original kernel byte-for-byte.
template <int MAXN>
__device__ cuDoubleComplex exp_newton_coeff_t(const cuDoubleComplex* C, int size, int n) {
  if (size == 0) return make_cuDoubleComplex(n == 0 ? 1.0 : 0.0, 0.0);

  cuDoubleComplex p[MAXN + 1]; // power traces p[1..n]
  cuDoubleComplex P[MAXN * MAXN]; // running C^k
  for (int i = 0; i < size * size; ++i) P[i] = C[i];

  for (int k = 1; k <= n; ++k) {
    cuDoubleComplex tr = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < size; ++i) tr = cadd(tr, P[i * size + i]);
    p[k] = tr;
    if (k < n) { // P <- P * C
      cuDoubleComplex T[MAXN * MAXN];
      for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j) {
          cuDoubleComplex s = make_cuDoubleComplex(0.0, 0.0);
          for (int t = 0; t < size; ++t) s = cadd(s, cmul(P[i * size + t], C[t * size + j]));
          T[i * size + j] = s;
        }
      for (int i = 0; i < size * size; ++i) P[i] = T[i];
    }
  }

  cuDoubleComplex e[MAXN + 1];
  e[0] = make_cuDoubleComplex(1.0, 0.0);
  for (int j = 1; j <= n; ++j) {
    cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
    for (int k = 1; k <= j; ++k) {
      cuDoubleComplex half_pk = make_cuDoubleComplex(0.5 * cuCreal(p[k]), 0.5 * cuCimag(p[k]));
      acc = cadd(acc, cmul(half_pk, e[j - k]));
    }
    e[j] = make_cuDoubleComplex(cuCreal(acc) / j, cuCimag(acc) / j);
  }
  return e[n];
}

// Non-templated alias at the full cap, for the cooperative + kappa kernels (unchanged).
__device__ cuDoubleComplex exp_newton_coeff(const cuDoubleComplex* C, int size, int n) {
  return exp_newton_coeff_t<HAF_MAX_N>(C, size, n);
}

// Signed contribution of one subset `mask` to haf(A): (-1)^(n-|S|) [lambda^n]
// exp(...) of B_S X. Templated on the buffer cap MAXN (see exp_newton_coeff_t).
template <int MAXN>
__device__ cuDoubleComplex haf_subset_term_t(const cuDoubleComplex* __restrict__ A,
                                             int N, int n, uint64_t mask) {
  int m = __popcll((long long)mask);
  int size = 2 * m;
  cuDoubleComplex BX[MAXN * MAXN];
  int pidx[MAXN / 2];
  int pc = 0;
  for (int i = 0; i < n; ++i) if ((mask >> i) & 1ull) pidx[pc++] = i;
  for (int r = 0; r < size; ++r) {
    int pr = pidx[r >> 1], rr = 2 * pr + (r & 1);
    for (int c = 0; c < size; ++c) {
      int pc2 = pidx[c >> 1];
      int cc = 2 * pc2 + ((c & 1) ^ 1); // swap within the pair (the X action)
      BX[r * size + c] = A[rr * N + cc];
    }
  }
  cuDoubleComplex coeff = exp_newton_coeff_t<MAXN>(BX, size, n);
  return ((n - m) & 1) ? make_cuDoubleComplex(-cuCreal(coeff), -cuCimag(coeff)) : coeff;
}

// Full-cap alias for the cooperative + kappa kernels (unchanged behavior).
__device__ cuDoubleComplex haf_subset_term(const cuDoubleComplex* __restrict__ A,
                                           int N, int n, uint64_t mask) {
  return haf_subset_term_t<HAF_MAX_N>(A, N, n, mask);
}

// out[b] = haf(mats + b*N*N), symmetric N x N (N = 2n), diagonal ignored. Templated
// on the buffer cap MAXN so a size-specialized (small-MAXN) instantiation cuts the
// per-thread footprint for small matrices (the perf-research candidate; the win is
// MEASURED on a device against the full-cap baseline -- see bench_kernels.cu).
template <int MAXN>
__global__ void haf_powertrace_fp64_kernel_t(const cuDoubleComplex* __restrict__ mats,
                                             int N, int batch,
                                             cuDoubleComplex* __restrict__ out) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  const cuDoubleComplex* A = mats + (size_t)b * N * N;

  if (N == 0) { out[b] = make_cuDoubleComplex(1.0, 0.0); return; }
  if (N & 1) { out[b] = make_cuDoubleComplex(0.0, 0.0); return; } // haf of odd size = 0
  if (N > MAXN) { out[b] = make_cuDoubleComplex(NAN, NAN); return; } // buffer cap
  int n = N / 2;

  cuDoubleComplex total = make_cuDoubleComplex(0.0, 0.0);
  for (uint64_t mask = 0; mask < (1ull << n); ++mask) // (-1)^(n-|S|) [lambda^n] exp(...)
    total = cuCadd(total, haf_subset_term_t<MAXN>(A, N, n, mask));
  out[b] = total;
}

// Size-specialized cap for small matrices (the common GBS regime). Footprint at MAXN
// = HAF_SMALL_N is ~3x smaller than at HAF_MAX_N (bench/kernel_footprint.py), so the
// small kernel should reach much higher occupancy -- to be confirmed on a device.
constexpr int HAF_SMALL_N = 12;

// Production path: the full-cap kernel (the per-thread baseline, unchanged). The small
// kernel below is exposed separately so the bench can A/B it at the same N before any
// dispatch is wired in (measure first; do not optimize by analogy).
extern "C" void gbs_haf_powertrace_fp64_batched(const cuDoubleComplex* d_mats, int N,
                                                int batch, cuDoubleComplex* d_out,
                                                cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 64; // larger local-memory footprint than permanent -> smaller block
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(haf_powertrace_fp64_kernel_t<HAF_MAX_N>, grid, block, stream, d_mats, N, batch, d_out);
}

// Size-specialized hafnian (small buffer cap). Valid for N <= HAF_SMALL_N; returns NaN
// above (the caller picks this only for small N). Same result as the full kernel.
extern "C" void gbs_haf_powertrace_fp64_small_batched(const cuDoubleComplex* d_mats, int N,
                                                      int batch, cuDoubleComplex* d_out,
                                                      cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 64; // SAME block as the full kernel -> a clean A/B isolating the cap
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(haf_powertrace_fp64_kernel_t<HAF_SMALL_N>, grid, block, stream, d_mats, N, batch, d_out);
}

// --- variable-N hafnian (v3 on-device sampler) -----------------------------
// Each matrix has its OWN even size N = d_n[b], stored in a maxn x maxn slot (top-left N x N --
// the on-device gather's ragged, cap-padded layout). Copies the top-left N x N into a contiguous
// local buffer and runs the SAME power-trace as the full kernel above, so the validated hafnian
// logic (haf_subset_term_t) is reused UNCHANGED -- only the matrix load differs. N must be even
// and <= MAXN; the gather/caller keep submatrices within the cap. (The contiguous copy adds one
// MAXN^2 buffer vs the baseline; a stride-aware variant is a later perf option.)
template <int MAXN>
__global__ void haf_powertrace_fp64_varn_kernel(const cuDoubleComplex* __restrict__ mats,
                                                int maxn, int batch, const int* __restrict__ d_n,
                                                cuDoubleComplex* __restrict__ out) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  int N = d_n[b];
  if (N == 0) { out[b] = make_cuDoubleComplex(1.0, 0.0); return; } // haf of 0x0 = 1 (vacuum)
  if (N & 1) { out[b] = make_cuDoubleComplex(0.0, 0.0); return; } // haf of odd size = 0
  if (N > MAXN) { out[b] = make_cuDoubleComplex(NAN, NAN); return; } // buffer cap
  const cuDoubleComplex* src = mats + (size_t)b * maxn * maxn;
  cuDoubleComplex A[MAXN * MAXN];
  for (int r = 0; r < N; ++r)
    for (int c = 0; c < N; ++c) A[r * N + c] = src[(size_t)r * maxn + c];
  int n = N / 2;
  cuDoubleComplex total = make_cuDoubleComplex(0.0, 0.0);
  for (uint64_t mask = 0; mask < (1ull << n); ++mask)
    total = cuCadd(total, haf_subset_term_t<MAXN>(A, N, n, mask));
  out[b] = total;
}

extern "C" void gbs_haf_powertrace_fp64_varn_batched(const cuDoubleComplex* d_mats, int maxn,
                                                     int batch, const int* d_n,
                                                     cuDoubleComplex* d_out, cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 64;
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(haf_powertrace_fp64_varn_kernel<HAF_MAX_N>, grid, block, stream,
                d_mats, maxn, batch, d_n, d_out);
}

// FP64 result + cancellation indicator in one pass (precision="auto"): the same
// subset sum, also accumulating sum|term| = sum_S |haf_subset_term| (the magnitude
// is sign-independent). No 1/terms scaling (unlike the permanent's Glynn form). The
// host forms kappa = absnorm[b] / |out[b]| and reruns the risky elements in DD.
__global__ void haf_powertrace_fp64_kappa_kernel(const cuDoubleComplex* __restrict__ mats,
                                                 int N, int batch,
                                                 cuDoubleComplex* __restrict__ out,
                                                 double* __restrict__ absnorm) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  const cuDoubleComplex* A = mats + (size_t)b * N * N;
  if (N == 0) { out[b] = make_cuDoubleComplex(1.0, 0.0); absnorm[b] = 1.0; return; }
  if (N & 1) { out[b] = make_cuDoubleComplex(0.0, 0.0); absnorm[b] = 0.0; return; }
  if (N > HAF_MAX_N) { out[b] = make_cuDoubleComplex(NAN, NAN); absnorm[b] = NAN; return; }
  int n = N / 2;
  cuDoubleComplex total = make_cuDoubleComplex(0.0, 0.0);
  double abs_acc = 0.0;
  for (uint64_t mask = 0; mask < (1ull << n); ++mask) {
    cuDoubleComplex t = haf_subset_term(A, N, n, mask);
    total = cuCadd(total, t);
    abs_acc += cuCabs(t);
  }
  out[b] = total;
  absnorm[b] = abs_acc;
}

extern "C" void gbs_haf_powertrace_fp64_kappa_batched(const cuDoubleComplex* d_mats, int N,
                                                      int batch, cuDoubleComplex* d_out,
                                                      double* d_absnorm, cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 64;
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(haf_powertrace_fp64_kappa_kernel, grid, block, stream, d_mats, N, batch, d_out, d_absnorm);
}

// --- cooperative (map/reduce) hafnian (perf; see permanent_coop.cu rationale) ---
// `groups` threads cooperate on one hafnian: each sums a contiguous sub-range of
// the 2^(N/2) subset masks into a partial, then a second pass sums the partials.
// The 2^(N/2) subset sum (each term an O(n^3) power trace) is parallelized
// ~groups x. Map/reduce over global memory -> validated on the CPU host shim.

// MAP: thread t -> matrix b = t/groups, group g = t%groups; sum masks [start,end).
__global__ void haf_coop_map_kernel(const cuDoubleComplex* __restrict__ mats, int N,
                                    int batch, int groups,
                                    cuDoubleComplex* __restrict__ partials) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= batch * groups) return;
  int b = t / groups, g = t % groups;
  const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
  // kernel-level guards fix the whole answer regardless of the subset sum.
  if (N == 0) { partials[t] = (g == 0) ? make_cuDoubleComplex(1.0, 0.0) : zero; return; }
  if (N & 1) { partials[t] = zero; return; } // haf odd = 0
  if (N > HAF_MAX_N) { partials[t] = (g == 0) ? make_cuDoubleComplex(NAN, NAN) : zero; return; }

  const cuDoubleComplex* A = mats + (size_t)b * N * N;
  int n = N / 2;
  uint64_t terms = 1ull << n;
  uint64_t R = (terms + groups - 1) / groups;
  uint64_t start = (uint64_t)g * R, end = start + R;
  if (end > terms) end = terms;
  cuDoubleComplex partial = zero;
  for (uint64_t mask = start; mask < end; ++mask)
    partial = cuCadd(partial, haf_subset_term(A, N, n, mask));
  partials[t] = partial;
}

// REDUCE: thread b sums its `groups` partials (fixed order -> deterministic).
__global__ void haf_coop_reduce_kernel(const cuDoubleComplex* __restrict__ partials,
                                       int batch, int groups,
                                       cuDoubleComplex* __restrict__ out) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  cuDoubleComplex s = make_cuDoubleComplex(0.0, 0.0);
  for (int g = 0; g < groups; ++g) s = cuCadd(s, partials[(size_t)b * groups + g]);
  out[b] = s;
}

extern "C" int gbs_haf_coop_batched(const cuDoubleComplex* d_mats, int N, int batch,
                                    int groups, cuDoubleComplex* d_out, cudaStream_t stream) {
  if (batch <= 0) return 0;
  if (groups < 1) groups = 1;
  if (N >= 2 && !(N & 1) && N <= HAF_MAX_N) {
    uint64_t terms = 1ull << (N / 2);
    if ((uint64_t)groups > terms) groups = (int)terms; // no more groups than subsets
  } else {
    groups = 1; // edge cases: a single group carries the fixed answer
  }
  cuDoubleComplex* d_partials = nullptr;
  cudaError_t err = cudaMalloc(&d_partials, (size_t)batch * groups * sizeof(cuDoubleComplex));
  if (err != cudaSuccess) return (int)err;
  const int block = 64;
  int map_grid = (batch * groups + block - 1) / block;
  GBS_LAUNCH_1D(haf_coop_map_kernel, map_grid, block, stream, d_mats, N, batch, groups, d_partials);
  err = cudaGetLastError();
  if (err == cudaSuccess) {
    int red_grid = (batch + block - 1) / block;
    GBS_LAUNCH_1D(haf_coop_reduce_kernel, red_grid, block, stream, d_partials, batch, groups, d_out);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  cudaFree(d_partials);
  return (int)err;
}

} // namespace gbs
