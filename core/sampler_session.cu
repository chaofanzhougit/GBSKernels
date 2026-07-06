// core/sampler_session.cu -- on-device sampler: the resident chain session.
// the RESIDENT chain. The reduced A-matrices {A_k}, the photon-count state, and the cuRAND
// states all live on the device; per mode k we launch gather -> variable-N hafnian -> real
// extract -> draw, threading the state in place with NO per-mode host round trip. Only the
// final state is copied D2H (by the caller). The validated gather / varn-hafnian / draw kernels
// are reused UNCHANGED; this file only adds three tiny copy/extract helpers + the orchestration.

#include <cuComplex.h>
#include <curand_kernel.h>
#include <cstdlib>
#include "subset_engine.cuh"   // <cuda_runtime.h> + GBS_LAUNCH_1D

// kernels from the sibling .cu files (linked together):
extern "C" void gbs_sampler_init_states(unsigned long long, int, curandState_t*, cudaStream_t);
extern "C" void gbs_sampler_gather(const cuDoubleComplex*, int, const int*, int, int, int,
                                   cuDoubleComplex*, int*, cudaStream_t);
extern "C" void gbs_haf_powertrace_fp64_varn_batched(const cuDoubleComplex*, int, int, const int*,
                                                     cuDoubleComplex*, cudaStream_t);
extern "C" void gbs_sampler_draw(const double*, const double*, int, int, curandState_t*, int*, cudaStream_t);

// --- tiny helpers (no D2H) --------------------------------------------------
// compact the first Kc columns of the (num_draws x M) state into a (num_draws x Kc) buffer
// (the gather reads a compact prefix); Kc = k-1.
__global__ void sampler_compact_prefix_kernel(const int* __restrict__ state, int M, int num_draws,
                                              int Kc, int* __restrict__ prefix_k) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= num_draws * Kc) return;
  int d = t / Kc, i = t % Kc;
  prefix_k[(size_t)d * Kc + i] = state[(size_t)d * M + i];
}

// scatter the per-draw mode-k photon count into column `col` of the (num_draws x M) state.
__global__ void sampler_scatter_col_kernel(const int* __restrict__ out_mode, int num_draws,
                                           int* __restrict__ state, int M, int col) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= num_draws) return;
  state[(size_t)d * M + col] = out_mode[d];
}

// real part of each (complex) conditional hafnian -> the draw kernel's weights.
__global__ void sampler_extract_real_kernel(const cuDoubleComplex* __restrict__ c, int n,
                                            double* __restrict__ r) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  r[t] = cuCreal(c[t]);
}

static void compact_prefix(const int* state, int M, int num_draws, int Kc, int* prefix_k, cudaStream_t s) {
  int total = num_draws * Kc; if (total <= 0) return;
  int block = 128, grid = (total + block - 1) / block;
  GBS_LAUNCH_1D(sampler_compact_prefix_kernel, grid, block, s, state, M, num_draws, Kc, prefix_k);
}
static void scatter_col(const int* out_mode, int num_draws, int* state, int M, int col, cudaStream_t s) {
  int block = 128, grid = (num_draws + block - 1) / block;
  GBS_LAUNCH_1D(sampler_scatter_col_kernel, grid, block, s, out_mode, num_draws, state, M, col);
}
static void extract_real(const cuDoubleComplex* c, int n, double* r, cudaStream_t s) {
  int block = 128, grid = (n + block - 1) / block;
  GBS_LAUNCH_1D(sampler_extract_real_kernel, grid, block, s, c, n, r);
}

// --- the resident chain -----------------------------------------------------
// d_Ak: the M reduced A-matrices concatenated (A_k is 2k x 2k row-major; offset of A_k is
// ak_off[k-1]). d_state: (num_draws x M) int -- the output, also the running prefix. The cuRAND
// states must already be initialised (gbs_sampler_init_states). All scratch is caller-allocated:
//   prefix_k: num_draws*(M-1) ;  sub: num_draws*(cutoff+1)*maxn*maxn ;  n,out_mode,hafr sized
//   num_draws*(cutoff+1) (n/hafr) and num_draws (out_mode) ; hafc: num_draws*(cutoff+1).
extern "C" void gbs_sampler_run(const cuDoubleComplex* d_Ak, const int* h_ak_off, int M,
                                int num_draws, int cutoff, int maxn, const double* d_invfac,
                                curandState_t* d_states, int* d_state, int* d_prefix_k,
                                cuDoubleComplex* d_sub, int* d_n, cuDoubleComplex* d_hafc,
                                double* d_hafr, int* d_out_mode, cudaStream_t stream) {
  // h_ak_off is a HOST pointer: it is only ever dereferenced here, on the host,
  // to offset d_Ak per mode step. (It was a device pointer once -- the host shim
  // cannot catch that class of bug, and the first real-device run did: SIGSEGV.)
  int batch = num_draws * (cutoff + 1);
  for (int k = 1; k <= M; ++k) {
    int Kc = k - 1;
    compact_prefix(d_state, M, num_draws, Kc, d_prefix_k, stream);     // prefix = state[:, :k-1]
    gbs_sampler_gather(d_Ak + h_ak_off[k - 1], k, d_prefix_k, num_draws, cutoff, maxn,
                       d_sub, d_n, stream);                            // submatrices A_k[idx,idx]
    gbs_haf_powertrace_fp64_varn_batched(d_sub, maxn, batch, d_n, d_hafc, stream);  // conditional hafs
    extract_real(d_hafc, batch, d_hafr, stream);                      // weights = Re(haf)
    gbs_sampler_draw(d_hafr, d_invfac, cutoff, num_draws, d_states, d_out_mode, stream);  // inverse-CDF + cuRAND
    scatter_col(d_out_mode, num_draws, d_state, M, Kc, stream);       // state[:, k-1] = photons
  }
}

// --- all-in-one resident sample (the binding-facing entry) ------------------
// Host inputs -> device -> run the resident chain -> host samples; manages all device scratch.
// h_Ak: the M reduced A-matrices concatenated (ak_total cuDoubleComplex; A_k starts at h_off[k-1]).
// h_out (caller-allocated): num_draws*M int, row-major. Submatrices must fit the hafnian cap (the
// caller checks 2*M*cutoff <= cap; maxn is the gather stride, typically that cap). Returns 0 on
// success, nonzero (the cudaError_t) on a device error.
extern "C" int gbs_sampler_sample(const cuDoubleComplex* h_Ak, int ak_total, const int* h_off,
                                  int M, int num_draws, int cutoff, int maxn,
                                  const double* h_invfac, unsigned long long seed, int* h_out) {
  int batch = num_draws * (cutoff + 1);
  cuDoubleComplex *d_Ak, *d_sub, *d_hafc; int *d_state, *d_prefix_k, *d_n, *d_out_mode;
  double *d_invfac, *d_hafr; curandState_t* d_states;
  if (cudaMalloc(&d_Ak, (size_t)ak_total * sizeof(cuDoubleComplex)) != cudaSuccess) return 1;
  cudaMalloc(&d_invfac, (size_t)(cutoff + 1) * sizeof(double));
  cudaMalloc(&d_states, (size_t)num_draws * sizeof(curandState_t));
  cudaMalloc(&d_state, (size_t)num_draws * M * sizeof(int));
  cudaMalloc(&d_prefix_k, (size_t)num_draws * (M > 1 ? M - 1 : 1) * sizeof(int));
  cudaMalloc(&d_sub, (size_t)batch * maxn * maxn * sizeof(cuDoubleComplex));
  cudaMalloc(&d_n, (size_t)batch * sizeof(int));
  cudaMalloc(&d_hafc, (size_t)batch * sizeof(cuDoubleComplex));
  cudaMalloc(&d_hafr, (size_t)batch * sizeof(double));
  cudaMalloc(&d_out_mode, (size_t)num_draws * sizeof(int));

  cudaMemcpy(d_Ak, h_Ak, (size_t)ak_total * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  cudaMemcpy(d_invfac, h_invfac, (size_t)(cutoff + 1) * sizeof(double), cudaMemcpyHostToDevice);
  int* zbuf = (int*)calloc((size_t)num_draws * M, sizeof(int));   // zero the state (no cudaMemset shim)
  cudaMemcpy(d_state, zbuf, (size_t)num_draws * M * sizeof(int), cudaMemcpyHostToDevice);
  free(zbuf);

  gbs_sampler_init_states(seed, num_draws, d_states, (cudaStream_t)0);
  gbs_sampler_run(d_Ak, h_off, M, num_draws, cutoff, maxn, d_invfac, d_states, d_state,
                  d_prefix_k, d_sub, d_n, d_hafc, d_hafr, d_out_mode, (cudaStream_t)0);
  cudaError_t err = cudaDeviceSynchronize();
  cudaMemcpy(h_out, d_state, (size_t)num_draws * M * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_Ak); cudaFree(d_invfac); cudaFree(d_states); cudaFree(d_state);
  cudaFree(d_prefix_k); cudaFree(d_sub); cudaFree(d_n); cudaFree(d_hafc); cudaFree(d_hafr);
  cudaFree(d_out_mode);
  return (err == cudaSuccess) ? 0 : (int)err;
}
