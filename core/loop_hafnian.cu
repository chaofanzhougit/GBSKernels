// loop_hafnian.cu -- batched loop hafnian via the power-trace formula (FP64).
//
// STATUS: validated on-device (RTX 4090, CUDA 12.4, sm_89) AND via the CPU host pre-flight.
//   Faithful transcription of the *verified* CPU reference
//   (cpu_ref/loop_hafnian.py :: loop_hafnian_powertrace, even N). It generalizes
//   hafnian.cu: the same subset engine and exp/Newton recurrence, plus the loop
//   (diagonal) term  v_k = (1/2) d^T X C^{k-1} d  in the generating-function
//   exponent (d = diag of the selected submatrix). This term was derived and
//   pinned numerically against the naive enumeration AND The Walrus loop=True
//   (see tests). A check_loop_hafnian.cu gate must PASS in a rented-GPU session
//   before this is trusted (docs/DESIGN.md §8, Layer 5).
//
// One loop hafnian per thread; batch across the grid (docs/DESIGN.md §5). Per-thread
// O(n^3 2^n), register-pressure-capped N; cooperative variants are the GPU
// optimization. Reduces to hafnian.cu when the diagonal is zero (v_k = 0).

#include <cuComplex.h>
#include <cstdint>

#include "subset_engine.cuh"

namespace gbs {

constexpr int LHAF_MAX_N = 20;

__device__ inline cuDoubleComplex ladd(cuDoubleComplex a, cuDoubleComplex b) { return cuCadd(a, b); }
__device__ inline cuDoubleComplex lmul(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }

// [lambda^n] exp(sum_k g_k lambda^k) given kg[k] = k*g_k, via
// e_0=1, e_j = (1/j) sum_{k=1}^j kg[k] e_{j-k}.
__device__ cuDoubleComplex exp_coeff_from_kg(const cuDoubleComplex* kg, int n) {
  cuDoubleComplex e[LHAF_MAX_N + 1];
  e[0] = make_cuDoubleComplex(1.0, 0.0);
  for (int j = 1; j <= n; ++j) {
    cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
    for (int k = 1; k <= j; ++k) acc = ladd(acc, lmul(kg[k], e[j - k]));
    e[j] = make_cuDoubleComplex(cuCreal(acc) / j, cuCimag(acc) / j);
  }
  return e[n];
}

// Signed contribution of one subset `mask` to lhaf(A): (-1)^(n-|S|) [lambda^n]
// exp(sum_k g_k lambda^k) with the loop term v_k. Shared by the per-thread kernel
// and the cooperative map. Empty subset -> 0.
__device__ cuDoubleComplex lhaf_subset_term(const cuDoubleComplex* __restrict__ A,
                                            int N, int n, uint64_t mask) {
  int m = __popcll((long long)mask);
  if (m == 0) return make_cuDoubleComplex(0.0, 0.0);  // empty -> [lambda^n] exp(0) = 0
  int size = 2 * m;
  int pidx[LHAF_MAX_N / 2];
  int pc = 0;
  for (int i = 0; i < n; ++i) if ((mask >> i) & 1ull) pidx[pc++] = i;

  // C = B_S X (swap columns within each pair); d = diag(B_S)
  cuDoubleComplex C[LHAF_MAX_N * LHAF_MAX_N];
  cuDoubleComplex d[LHAF_MAX_N];
  for (int r = 0; r < size; ++r) {
    int rr = 2 * pidx[r >> 1] + (r & 1);
    for (int c = 0; c < size; ++c) {
      int cc = 2 * pidx[c >> 1] + ((c & 1) ^ 1);  // X swap within the pair
      C[r * size + c] = A[rr * N + cc];
    }
    d[r] = A[rr * N + rr];  // diagonal loop weight
  }

  // power traces p_k = tr(C^k) and loop terms v_k = (1/2) d^T X C^{k-1} d
  cuDoubleComplex kg[LHAF_MAX_N + 1];
  cuDoubleComplex P[LHAF_MAX_N * LHAF_MAX_N];   // running C^k
  cuDoubleComplex Q[LHAF_MAX_N * LHAF_MAX_N];   // running C^{k-1}
  for (int i = 0; i < size * size; ++i) { P[i] = C[i]; Q[i] = make_cuDoubleComplex(0.0, 0.0); }
  for (int i = 0; i < size; ++i) Q[i * size + i] = make_cuDoubleComplex(1.0, 0.0);  // C^0 = I

  for (int k = 1; k <= n; ++k) {
    cuDoubleComplex pk = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < size; ++i) pk = ladd(pk, P[i * size + i]);  // tr(C^k)

    // v_k = (1/2) d^T X Q d: w = Q d, then swap within each pair.
    cuDoubleComplex w[LHAF_MAX_N];
    for (int a = 0; a < size; ++a) {
      cuDoubleComplex s = make_cuDoubleComplex(0.0, 0.0);
      for (int c = 0; c < size; ++c) s = ladd(s, lmul(Q[a * size + c], d[c]));
      w[a] = s;
    }
    cuDoubleComplex vk = make_cuDoubleComplex(0.0, 0.0);
    for (int a = 0; a < size; ++a) {
      int xa = a ^ 1;  // X swaps rows within each consecutive pair
      vk = ladd(vk, lmul(d[a], w[xa]));
    }
    vk = make_cuDoubleComplex(0.5 * cuCreal(vk), 0.5 * cuCimag(vk));

    // kg[k] = k*g_k = p_k/2 + k*v_k
    cuDoubleComplex half_pk = make_cuDoubleComplex(0.5 * cuCreal(pk), 0.5 * cuCimag(pk));
    kg[k] = ladd(half_pk, make_cuDoubleComplex(k * cuCreal(vk), k * cuCimag(vk)));

    if (k < n) {  // advance Q <- C^k (current P), P <- C^{k+1}
      for (int i = 0; i < size * size; ++i) Q[i] = P[i];
      cuDoubleComplex T[LHAF_MAX_N * LHAF_MAX_N];
      for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j) {
          cuDoubleComplex s = make_cuDoubleComplex(0.0, 0.0);
          for (int t = 0; t < size; ++t) s = ladd(s, lmul(P[i * size + t], C[t * size + j]));
          T[i * size + j] = s;
        }
      for (int i = 0; i < size * size; ++i) P[i] = T[i];
    }
  }

  cuDoubleComplex coeff = exp_coeff_from_kg(kg, n);
  return ((n - m) & 1) ? make_cuDoubleComplex(-cuCreal(coeff), -cuCimag(coeff)) : coeff;
}

// out[b] = lhaf(mats + b*N*N), symmetric N x N (N = 2n), diagonal = loop weights.
__global__ void loop_haf_fp64_kernel(const cuDoubleComplex* __restrict__ mats, int N,
                                     int batch, cuDoubleComplex* __restrict__ out) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  const cuDoubleComplex* A = mats + (size_t)b * N * N;
  if (N == 0) { out[b] = make_cuDoubleComplex(1.0, 0.0); return; }
  // This power-trace kernel is the even-N path. The loop hafnian IS defined for
  // odd N (a leftover vertex becomes a self-loop), but that case is the CPU naive
  // reference; reject it here with NaN rather than silently computing N/2 wrong.
  if (N & 1) { out[b] = make_cuDoubleComplex(NAN, NAN); return; }
  if (N > LHAF_MAX_N) { out[b] = make_cuDoubleComplex(NAN, NAN); return; }
  int n = N / 2;

  cuDoubleComplex total = make_cuDoubleComplex(0.0, 0.0);
  for (uint64_t mask = 0; mask < (1ull << n); ++mask)
    total = cuCadd(total, lhaf_subset_term(A, N, n, mask));
  out[b] = total;
}

extern "C" void gbs_loop_haf_fp64_batched(const cuDoubleComplex* d_mats, int N,
                                          int batch, cuDoubleComplex* d_out,
                                          cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 64;
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(loop_haf_fp64_kernel, grid, block, stream, d_mats, N, batch, d_out);
}

// FP64 result + cancellation indicator in one pass (precision="auto"): same subset
// sum, also accumulating sum|term| = sum_S |lhaf_subset_term|. The host forms
// kappa = absnorm[b] / |out[b]| and reruns the risky elements in DD.
__global__ void loop_haf_fp64_kappa_kernel(const cuDoubleComplex* __restrict__ mats, int N,
                                           int batch, cuDoubleComplex* __restrict__ out,
                                           double* __restrict__ absnorm) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  const cuDoubleComplex* A = mats + (size_t)b * N * N;
  if (N == 0) { out[b] = make_cuDoubleComplex(1.0, 0.0); absnorm[b] = 1.0; return; }
  if (N & 1) { out[b] = make_cuDoubleComplex(NAN, NAN); absnorm[b] = NAN; return; }
  if (N > LHAF_MAX_N) { out[b] = make_cuDoubleComplex(NAN, NAN); absnorm[b] = NAN; return; }
  int n = N / 2;
  cuDoubleComplex total = make_cuDoubleComplex(0.0, 0.0);
  double abs_acc = 0.0;
  for (uint64_t mask = 0; mask < (1ull << n); ++mask) {
    cuDoubleComplex t = lhaf_subset_term(A, N, n, mask);
    total = cuCadd(total, t);
    abs_acc += cuCabs(t);
  }
  out[b] = total;
  absnorm[b] = abs_acc;
}

extern "C" void gbs_loop_haf_fp64_kappa_batched(const cuDoubleComplex* d_mats, int N,
                                                int batch, cuDoubleComplex* d_out,
                                                double* d_absnorm, cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 64;
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(loop_haf_fp64_kappa_kernel, grid, block, stream, d_mats, N, batch, d_out, d_absnorm);
}

// --- cooperative (map/reduce) loop hafnian (perf; see permanent_coop.cu) ---------
// `groups` threads split the 2^(N/2) subset sum (each term an O(n^3) power trace
// + loop term), then a second pass sums the partials.

__global__ void loop_haf_coop_map_kernel(const cuDoubleComplex* __restrict__ mats, int N,
                                         int batch, int groups,
                                         cuDoubleComplex* __restrict__ partials) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= batch * groups) return;
  int b = t / groups, g = t % groups;
  const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
  if (N == 0)         { partials[t] = (g == 0) ? make_cuDoubleComplex(1.0, 0.0) : zero; return; }
  if (N & 1)          { partials[t] = (g == 0) ? make_cuDoubleComplex(NAN, NAN) : zero; return; }
  if (N > LHAF_MAX_N) { partials[t] = (g == 0) ? make_cuDoubleComplex(NAN, NAN) : zero; return; }

  const cuDoubleComplex* A = mats + (size_t)b * N * N;
  int n = N / 2;
  uint64_t terms = 1ull << n;
  uint64_t R = (terms + groups - 1) / groups;
  uint64_t start = (uint64_t)g * R, end = start + R;
  if (end > terms) end = terms;
  cuDoubleComplex partial = zero;
  for (uint64_t mask = start; mask < end; ++mask)
    partial = cuCadd(partial, lhaf_subset_term(A, N, n, mask));
  partials[t] = partial;
}

__global__ void loop_haf_coop_reduce_kernel(const cuDoubleComplex* __restrict__ partials,
                                            int batch, int groups,
                                            cuDoubleComplex* __restrict__ out) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  cuDoubleComplex s = make_cuDoubleComplex(0.0, 0.0);
  for (int g = 0; g < groups; ++g) s = cuCadd(s, partials[(size_t)b * groups + g]);
  out[b] = s;
}

extern "C" int gbs_loop_haf_coop_batched(const cuDoubleComplex* d_mats, int N, int batch,
                                         int groups, cuDoubleComplex* d_out, cudaStream_t stream) {
  if (batch <= 0) return 0;
  if (groups < 1) groups = 1;
  if (N >= 2 && !(N & 1) && N <= LHAF_MAX_N) {
    uint64_t terms = 1ull << (N / 2);
    if ((uint64_t)groups > terms) groups = (int)terms;
  } else {
    groups = 1;
  }
  cuDoubleComplex* d_partials = nullptr;
  cudaError_t err = cudaMalloc(&d_partials, (size_t)batch * groups * sizeof(cuDoubleComplex));
  if (err != cudaSuccess) return (int)err;
  const int block = 64;
  int map_grid = (batch * groups + block - 1) / block;
  GBS_LAUNCH_1D(loop_haf_coop_map_kernel, map_grid, block, stream, d_mats, N, batch, groups, d_partials);
  err = cudaGetLastError();
  if (err == cudaSuccess) {
    int red_grid = (batch + block - 1) / block;
    GBS_LAUNCH_1D(loop_haf_coop_reduce_kernel, red_grid, block, stream, d_partials, batch, groups, d_out);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  cudaFree(d_partials);
  return (int)err;
}

}  // namespace gbs
