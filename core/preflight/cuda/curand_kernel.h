// core/preflight/cuda/curand_kernel.h  --  CPU shim of CUDA's <curand_kernel.h>.
//
// PRE-FLIGHT ONLY (see cuComplex.h / cuda_runtime.h). Provides just enough of the
// cuRAND device API for the v3 on-device sampler's draw kernel to compile and run on
// the host shim. On a real nvcc build the toolkit's <curand_kernel.h> is used instead
// (this file is only on the host-shim include path), so the SAME kernel source builds
// both ways. cuRAND != numpy RNG, so the device sampler is validated DISTRIBUTIONALLY,
// never bit-for-bit vs the CPU sampler -- this shim just needs a decent per-state stream.
//
// Implements: curandState_t / curandState, curand_init(seed, subseq, offset, &st),
// curand_uniform_double(&st) -> (0, 1].  Stream: SplitMix64 seed -> xorshift64* draws.

#pragma once

#include "cuda_runtime.h"  // for the __device__ no-op macro on the host shim

#include <cstdint>

struct curandStateXORWOW_t {
  unsigned long long s;
};
using curandState_t = curandStateXORWOW_t;
using curandState = curandStateXORWOW_t;

// Seed one draw's stream. SplitMix64 mixes (seed, subsequence, offset) so distinct draws
// (distinct subsequence) get well-separated, non-degenerate streams.
__device__ inline void curand_init(unsigned long long seed, unsigned long long subsequence,
                                   unsigned long long offset, curandState_t* st) {
  unsigned long long z = seed + 0x9E3779B97F4A7C15ULL * (subsequence + 1ULL) + offset;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z = z ^ (z >> 31);
  st->s = z ? z : 0x1234567890ABCDEFULL;  // never zero (xorshift fixed point)
}

// Uniform double in (0, 1] (matching cuRAND's curand_uniform_double convention).
__device__ inline double curand_uniform_double(curandState_t* st) {
  unsigned long long x = st->s;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  st->s = x;
  unsigned long long r = x * 0x2545F4914F6CDD1DULL;
  // top 53 bits -> {1, ..., 2^53} / 2^53  == (0, 1]
  return (double)((r >> 11) + 1ULL) * (1.0 / 9007199254740992.0);
}
