// dd.cuh -- double-double (DD) arithmetic via error-free transformations.
//
// STATUS: validated on-device (RTX 4090, CUDA 12.4, sm_89) AND via the CPU host pre-flight.
//   GPUs have no native quad precision (docs/DESIGN.md §6); DD represents each value
//   as an unevaluated sum of two FP64 (hi+lo, ~31 decimal digits) using
//   error-free transforms (Dekker/Knuth; Bailey's QD). This restores accuracy
//   where the FP64 alternating sums of these #P-hard functions cancel. Validated
//   against the highprec_ref/ (mpmath) boundary -- see bench/accuracy.py and the
//   check_*_dd gates (docs/DESIGN.md §8, Layer 5).
//
// References (reimplemented from public algorithms, not copied): Dekker 1971;
// Knuth TAOCP vol.2; Hida-Li-Bailey QD library design.

#pragma once

#include <cmath>  // host fma(); under nvcc fma is a device builtin (harmless include)
#include <cuda_runtime.h>

namespace gbs {

struct dd {        // value == hi + lo, with |lo| <= 0.5 ulp(hi)
  double hi;
  double lo;
};

// --- error-free transforms -------------------------------------------------

// Knuth TwoSum: s = fl(a+b), err = (a+b) - s exactly. No assumption on |a|,|b|.
__host__ __device__ inline double two_sum(double a, double b, double& err) {
  double s = a + b;
  double bb = s - a;
  err = (a - (s - bb)) + (b - bb);
  return s;
}

// Dekker FastTwoSum: requires |a| >= |b|; cheaper (3 flops).
__host__ __device__ inline double quick_two_sum(double a, double b, double& err) {
  double s = a + b;
  err = b - (s - a);
  return s;
}

// TwoProd via fused multiply-add: p = fl(a*b), err = a*b - p exactly.
__host__ __device__ inline double two_prod(double a, double b, double& err) {
  double p = a * b;
  err = fma(a, b, -p);
  return p;
}

// --- DD operations ---------------------------------------------------------

__host__ __device__ inline dd dd_from(double a) { return dd{a, 0.0}; }

__host__ __device__ inline dd dd_add(dd a, dd b) {
  double e;
  double s = two_sum(a.hi, b.hi, e);
  e += a.lo + b.lo;
  double hi = quick_two_sum(s, e, e);
  return dd{hi, e};
}

__host__ __device__ inline dd dd_add_d(dd a, double b) {
  double e;
  double s = two_sum(a.hi, b, e);
  e += a.lo;
  double hi = quick_two_sum(s, e, e);
  return dd{hi, e};
}

__host__ __device__ inline dd dd_mul(dd a, dd b) {
  double e;
  double p = two_prod(a.hi, b.hi, e);
  e += a.hi * b.lo + a.lo * b.hi;
  double hi = quick_two_sum(p, e, e);
  return dd{hi, e};
}

__host__ __device__ inline dd dd_mul_d(dd a, double b) {
  double e;
  double p = two_prod(a.hi, b, e);
  e += a.lo * b;
  double hi = quick_two_sum(p, e, e);
  return dd{hi, e};
}

__host__ __device__ inline dd dd_neg(dd a) { return dd{-a.hi, -a.lo}; }

// Complex double-double: real/imag each a dd. The kernels operate on these for
// the DD precision tier; FP64 tier uses cuDoubleComplex directly.
struct ddcomplex {
  dd re;
  dd im;
};

__host__ __device__ inline ddcomplex ddc_add(ddcomplex a, ddcomplex b) {
  return ddcomplex{dd_add(a.re, b.re), dd_add(a.im, b.im)};
}

__host__ __device__ inline ddcomplex ddc_mul(ddcomplex a, ddcomplex b) {
  // (a.re + i a.im)(b.re + i b.im)
  dd rr = dd_mul(a.re, b.re);
  dd ii = dd_mul(a.im, b.im);
  dd ri = dd_mul(a.re, b.im);
  dd ir = dd_mul(a.im, b.re);
  return ddcomplex{dd_add(rr, dd_neg(ii)), dd_add(ri, ir)};
}

__host__ __device__ inline ddcomplex ddc_mul_d(ddcomplex a, double s) {
  return ddcomplex{dd_mul_d(a.re, s), dd_mul_d(a.im, s)};
}

__host__ __device__ inline ddcomplex ddc_sub(ddcomplex a, ddcomplex b) {
  return ddcomplex{dd_add(a.re, dd_neg(b.re)), dd_add(a.im, dd_neg(b.im))};
}

// --- DD division by a double (QD-style; ~full DD precision) -----------------
// Needed by the loop-hafnian / hafnian power-trace recurrence, which divides by
// integers j (1/j is NOT exact, so ddc_mul_d would cap precision at FP64).

__host__ __device__ inline dd two_prod_dd(double a, double b) {
  double e;
  double p = two_prod(a, b, e);
  return dd{p, e};
}

__host__ __device__ inline dd dd_div_d(dd a, double b) {
  double q1 = a.hi / b;                          // first quotient digit
  dd r = dd_add(a, dd_neg(two_prod_dd(q1, b)));  // remainder a - q1*b
  double q2 = r.hi / b;                          // correction digit
  double e;
  double hi = quick_two_sum(q1, q2, e);
  return dd{hi, e};
}

__host__ __device__ inline ddcomplex ddc_div_d(ddcomplex a, double s) {
  return ddcomplex{dd_div_d(a.re, s), dd_div_d(a.im, s)};
}

// --- DD / DD and DD sqrt (real) --------------------------------------------
// Needed by the torontonian's per-subset determinant + 1/sqrt(det). The
// torontonian's physical domain is real O, so these real-DD ops suffice.

__host__ __device__ inline dd dd_div(dd a, dd b) {
  double q1 = a.hi / b.hi;
  dd r = dd_add(a, dd_neg(dd_mul_d(b, q1)));  // a - q1*b
  double q2 = r.hi / b.hi;
  r = dd_add(r, dd_neg(dd_mul_d(b, q2)));     // r - q2*b
  double q3 = r.hi / b.hi;
  double e;
  double hi = quick_two_sum(q1, q2, e);
  return dd_add(dd{hi, e}, dd{q3, 0.0});      // renormalize q1+q2+q3
}

// sqrt(a) for a >= 0, via one Newton step on a double seed (QD algorithm):
// sqrt(a) ~ y + (a - y^2)*(x/2) with x ~ 1/sqrt(a.hi), y = a*x.
__host__ __device__ inline dd dd_sqrt(dd a) {
  if (a.hi <= 0.0) return dd{0.0, 0.0};
  double x = 1.0 / sqrt(a.hi);
  dd y = dd_mul_d(a, x);                       // ~ sqrt(a)
  dd diff = dd_add(a, dd_neg(dd_mul(y, y)));   // a - y^2
  return dd_add(y, dd_mul_d(diff, 0.5 * x));
}

}  // namespace gbs
