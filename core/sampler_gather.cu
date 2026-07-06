// core/sampler_gather.cu -- on-device sampler: the submatrix-gather kernel.
// build each draw's (cutoff+1) growing conditional submatrices A_k[idx, idx] ON THE DEVICE
// from the resident reduced A-matrix A_k and the resident prefix state -- replacing the host
// np.ix_ gather. One thread per (draw, candidate j).
//
// For draw d (prefix = photon counts of modes 0..k-2) and candidate j, the index is exactly
// the host sampler's:  idx = [a-block: mode i repeated prefix[i]] + [mode k-1 repeated j]
//                          + [adagger-block: mode i+k repeated prefix[i]] + [mode 2k-1 repeated j]
// so the submatrix has size n = 2*sum(prefix) + 2*j. Submatrices are RAGGED, so each is written
// into a maxn*maxn slot (top-left n*n filled) and its actual size n recorded in out_n -- the
// variable-n hafnian + the resident chain are increment C.

#include <cuComplex.h>
#include "subset_engine.cuh"   // <cuda_runtime.h> + GBS_LAUNCH_1D

#ifndef SAMPLER_GATHER_MAXN
#define SAMPLER_GATHER_MAXN 32   // index scratch cap (>= the hafnian kernel's cap)
#endif

__global__ void sampler_gather_kernel(const cuDoubleComplex* __restrict__ Ak, int k,
                                      const int* __restrict__ prefix, int num_draws, int cutoff,
                                      int maxn, cuDoubleComplex* __restrict__ out_sub,
                                      int* __restrict__ out_n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_draws * (cutoff + 1);
  if (t >= total) return;
  int d = t / (cutoff + 1);
  int j = t % (cutoff + 1);
  const int* pre = prefix + (size_t)d * (k - 1);
  const int twok = 2 * k;

  int idx[SAMPLER_GATHER_MAXN];
  int n = 0;
  for (int i = 0; i < k - 1; ++i)                       // a-block: mode i x prefix[i]
    for (int r = 0; r < pre[i]; ++r) if (n < maxn) idx[n++] = i;
  for (int r = 0; r < j; ++r) if (n < maxn) idx[n++] = k - 1;             // mode k-1 x j
  for (int i = 0; i < k - 1; ++i)                       // a-dagger block: mode i+k x prefix[i]
    for (int r = 0; r < pre[i]; ++r) if (n < maxn) idx[n++] = i + k;
  for (int r = 0; r < j; ++r) if (n < maxn) idx[n++] = twok - 1;          // mode 2k-1 x j
  out_n[t] = n;

  cuDoubleComplex* sub = out_sub + (size_t)t * maxn * maxn;
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < n; ++c)
      sub[(size_t)r * maxn + c] = Ak[(size_t)idx[r] * twok + idx[c]];
}

// Gather the (cutoff+1) conditional submatrices for every draw. Ak: 2k x 2k (row-major).
// prefix: num_draws x (k-1) photon counts. out_sub: num_draws*(cutoff+1) slots of maxn*maxn.
extern "C" void gbs_sampler_gather(const cuDoubleComplex* d_Ak, int k, const int* d_prefix,
                                   int num_draws, int cutoff, int maxn,
                                   cuDoubleComplex* d_out_sub, int* d_out_n, cudaStream_t stream) {
  int total = num_draws * (cutoff + 1);
  int block = 128, grid = (total + block - 1) / block;
  GBS_LAUNCH_1D(sampler_gather_kernel, grid, block, stream,
                d_Ak, k, d_prefix, num_draws, cutoff, maxn, d_out_sub, d_out_n);
}
