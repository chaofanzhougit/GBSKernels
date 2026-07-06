// certified_rounding.cuh -- upward/downward-rounded scalar helpers shared by
// the certified kernels (certified.cu, certified_dd.cu). Device: IEEE-754
// per-instruction directed-rounding intrinsics. Host shim: nextafter beyond
// the round-to-nearest result (an over-approximation of ru / under- of rd),
// so bounds computed on the shim remain valid, one ulp looser.
#pragma once
#include <math.h>

namespace gbs {

__device__ inline double ru_add(double a, double b) {
#if defined(__CUDA_ARCH__)
  return __dadd_ru(a, b);
#else
  return nextafter(a + b, INFINITY);
#endif
}
__device__ inline double ru_mul(double a, double b) {
#if defined(__CUDA_ARCH__)
  return __dmul_ru(a, b);
#else
  return nextafter(a * b, INFINITY);
#endif
}
__device__ inline double ru_div(double a, double b) {
#if defined(__CUDA_ARCH__)
  return __ddiv_ru(a, b);
#else
  return nextafter(a / b, INFINITY);
#endif
}
__device__ inline double rd_mul(double a, double b) {
#if defined(__CUDA_ARCH__)
  return __dmul_rd(a, b);
#else
  return nextafter(a * b, -INFINITY);
#endif
}
__device__ inline double rd_sqrt(double a) {
#if defined(__CUDA_ARCH__)
  return __dsqrt_rd(a);
#else
  return nextafter(sqrt(a), -INFINITY);   // IEEE sqrt correctly rounded
#endif
}

}  // namespace gbs
