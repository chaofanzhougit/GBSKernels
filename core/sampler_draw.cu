// core/sampler_draw.cu -- on-device sampler: the conditional-draw kernel.
// the conditional DRAW step on the device. One thread per draw turns that draw's (cutoff+1)
// conditional hafnian weights into a photon count via the SAME clip(haf,0)/j! -> normalise ->
// cumsum inverse-CDF the host sampler uses, but drawing u from a per-draw cuRAND stream.
//
// cuRAND != numpy RNG, so the device sampler is validated DISTRIBUTIONALLY (chi-square /
// TV), never bit-for-bit vs the CPU sampler. The (cutoff+1) hafnians are computed upstream by
// the existing even-N hafnian kernel; this kernel does only the normalise + draw (increment B
// adds the on-device gather, increment C keeps the whole chain resident).

#include <curand_kernel.h>
#include "subset_engine.cuh"   // pulls <cuda_runtime.h> + GBS_LAUNCH_1D

// Seed one cuRAND stream per draw (seed + the draw's index as the subsequence).
__global__ void sampler_init_states_kernel(unsigned long long seed, int num_draws,
                                           curandState_t* __restrict__ states) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= num_draws) return;
  curand_init(seed, (unsigned long long)d, 0ULL, &states[d]);
}

// Draw mode-k photon counts. haf: row-major (num_draws x (cutoff+1)) REAL hafnians;
// invfac[j] = 1/j!. out[d] in {0..cutoff}.
__global__ void sampler_draw_kernel(const double* __restrict__ haf,
                                    const double* __restrict__ invfac,
                                    int cutoff, int num_draws,
                                    curandState_t* __restrict__ states,
                                    int* __restrict__ out) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= num_draws) return;
  const double* w = haf + (size_t)d * (cutoff + 1);
  double sum = 0.0;
  for (int j = 0; j <= cutoff; ++j) {
    double wj = w[j] > 0.0 ? w[j] : 0.0;          // clip(haf, 0)
    sum += wj * invfac[j];
  }
  if (sum <= 0.0) { out[d] = 0; return; }         // degenerate conditional -> vacuum
  double u = curand_uniform_double(&states[d]);
  double c = 0.0;
  int photon = 0;
  for (int j = 0; j <= cutoff; ++j) {             // photon = #{ cumsum < u } (inverse CDF)
    double wj = w[j] > 0.0 ? w[j] : 0.0;
    c += (wj * invfac[j]) / sum;
    if (c < u) ++photon;
  }
  out[d] = photon;
}

extern "C" void gbs_sampler_init_states(unsigned long long seed, int num_draws,
                                        curandState_t* d_states, cudaStream_t stream) {
  int block = 128, grid = (num_draws + block - 1) / block;
  GBS_LAUNCH_1D(sampler_init_states_kernel, grid, block, stream, seed, num_draws, d_states);
}

extern "C" void gbs_sampler_draw(const double* d_haf, const double* d_invfac,
                                 int cutoff, int num_draws, curandState_t* d_states,
                                 int* d_out, cudaStream_t stream) {
  int block = 128, grid = (num_draws + block - 1) / block;
  GBS_LAUNCH_1D(sampler_draw_kernel, grid, block, stream,
                d_haf, d_invfac, cutoff, num_draws, d_states, d_out);
}
