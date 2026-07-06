// subset_engine.cuh -- shared subset-enumeration utilities.
//
// The central design insight (docs/DESIGN.md §5): all four functions are a signed sum
// over 2^k subsets of a per-subset dense-linear-algebra kernel, followed by a
// reduction. This header holds the *utilities* for that structure.
//
// IMPLEMENTATION STATUS (be precise; the design goal is more than this):
// * The PERMANENT kernel uses the Gray-code delta walk below -- consecutive
// codes g(i) = i ^ (i>>1) differ in one bit (flipped_index, gray_bit_of), so
// each subset is an O(n) rank-update of the previous row-sum vector.
// * The hafnian, loop hafnian, and torontonian kernels currently enumerate the
// 2^k subsets *independently* (a plain `for mask` loop with __popcll), each
// rebuilding its per-subset matrix from scratch -- they do NOT yet share a
// single Gray-code delta-update engine. Unifying them onto this walk (so the
// per-subset state is updated incrementally) is intended but not done.
// * All kernels map ONE evaluation per THREAD and the batch across the grid.
// The warp/block-cooperative mapping in the product story (docs/DESIGN.md §5) is
// future work; this per-thread form is the correctness-first first cut.
//
// Canonical Gray-code walk (as used by the permanent):
// acc = reduce(empty_or_full_state);
// for (uint64_t i = 1; i < (1ull << k); ++i) {
// int j = flipped_index(i); // which element toggled (ruler function)
// bool now = gray_bit_of(i, j); // is it now IN the subset?
// state = delta_update(state, j, now);
// acc = combine(acc, subset_sign(i), reduce(state));
// }

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace gbs {

// Reflected binary Gray code of i.
__host__ __device__ inline uint64_t gray_code(uint64_t i) { return i ^ (i >> 1); }

// Index of the single bit that flips between Gray code i-1 and i: ctz(i).
__host__ __device__ inline int flipped_index(uint64_t i) {
#if defined(__CUDA_ARCH__)
  return __ffsll((long long)i) - 1; // 1-based ffs -> 0-based
#else
  return __builtin_ctzll(i);
#endif
}

// Is bit j set in the Gray code of i? (i.e. did element j just enter the subset)
__host__ __device__ inline bool gray_bit_of(uint64_t i, int j) {
  return ((gray_code(i) >> j) & 1ull) != 0ull;
}

// Population-count parity of the Gray code -> sign (-1)^|S| as +1/-1.
__host__ __device__ inline int subset_sign(uint64_t i) {
#if defined(__CUDA_ARCH__)
  return (__popcll((long long)gray_code(i)) & 1) ? -1 : 1;
#else
  return (__builtin_popcountll(gray_code(i)) & 1) ? -1 : 1;
#endif
}

} // namespace gbs

// One-dimensional batched kernel launch. Under nvcc this is the real
// <<<grid, block>>> launch; on a host pre-flight build (core/preflight, anchor
// sec.10) it emulates the grid by looping over every (block, thread) and setting
// the shim thread indices, so the same kernel source runs and is validated on
// CPU before any paid GPU session. ``kern`` is the __global__ kernel symbol.
#if defined(__CUDACC__)
#define GBS_LAUNCH_1D(kern, grid, block, stream, ...) \
  kern<<<(grid), (block), 0, (stream)>>>(__VA_ARGS__)
#else
#define GBS_LAUNCH_1D(kern, grid, block, stream, ...) \
  do { \
    for (int _bi = 0; _bi < (grid); ++_bi) \
      for (int _ti = 0; _ti < (block); ++_ti) { \
        ::gbs::shim::set_thread(_bi, _ti, (block)); \
        kern(__VA_ARGS__); \
      } \
  } while (0)
#endif
