// core/preflight/cuda/cuComplex.h -- CPU shim of CUDA's <cuComplex.h>.
//
// PRE-FLIGHT ONLY. Lets the kernel sources in core/ compile and RUN on a host
// (clang++/g++) without a CUDA toolchain, so syntax/type/logic errors are caught
// before a paid rented-GPU session (docs/DESIGN.md §10). Faithfully reproduces the
// cuDoubleComplex semantics; not used in the real GPU build (which gets NVIDIA's
// real header). Put this directory first on the include path for the preflight.

#pragma once

#include <cmath>

struct cuDoubleComplex {
  double x; // real
  double y; // imag
};

inline cuDoubleComplex make_cuDoubleComplex(double r, double i) {
  return cuDoubleComplex{r, i};
}

inline double cuCreal(cuDoubleComplex z) { return z.x; }
inline double cuCimag(cuDoubleComplex z) { return z.y; }

inline cuDoubleComplex cuCadd(cuDoubleComplex a, cuDoubleComplex b) {
  return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}
inline cuDoubleComplex cuCsub(cuDoubleComplex a, cuDoubleComplex b) {
  return make_cuDoubleComplex(a.x - b.x, a.y - b.y);
}
inline cuDoubleComplex cuCmul(cuDoubleComplex a, cuDoubleComplex b) {
  return make_cuDoubleComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
inline cuDoubleComplex cuCdiv(cuDoubleComplex a, cuDoubleComplex b) {
  double d = b.x * b.x + b.y * b.y;
  return make_cuDoubleComplex((a.x * b.x + a.y * b.y) / d,
                              (a.y * b.x - a.x * b.y) / d);
}
inline double cuCabs(cuDoubleComplex z) { return std::hypot(z.x, z.y); }
