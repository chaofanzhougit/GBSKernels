// certified_dd.cu -- batched CERTIFIED DOUBLE-DOUBLE evaluation: the DD value PLUS a rigorous bound, completing the proven
// escalation ladder certified-fp64 -> certified-dd -> mpmath.
//
// (check_certified_dd.cu) in the next rented-GPU session.
//
// Value paths mirror permanent_dd.cu / hafnian_dd.cu op-for-op (the same
// dd.cuh error-free transforms), so the certificate covers the value the DD
// tier already returns. The bound accumulators are plain doubles in
// UPWARD-ROUNDED arithmetic (certified.cu's ru_* helpers) with the per-op
// constants scaled to double-double:
//
// * per DD add/mul, relative error <= a few units of 2^-106 (Joldes, Muller
// & Popescu, "Tight and rigorous error bounds for basic building blocks
// of double-word arithmetic", ACM TOMS 2017: 3u^2-ish for DWPlusDW, 5u^2
// for DWTimesDW with FMA, u = 2^-53). We charge GBS_U_DD = 2^-100 per
// operation -- ~60x the proven constants, deliberately generous; validity
// is the claim, tightness is measured.
// * complex DD multiply (4 muls + 2 adds): GBS_C_MUL_DD = 2^-96.
// * |z| upper bounds from the hi components: |dd| <= |hi| (1 + 2^-52), and
// the 2-norm from hypot(hi_re, hi_im) is faithful, so a (1 + 2^-49)
// inflation covers both.
//
// The dominant bound term remains u_dd * sum|term| -- the rigorous kappa at
// double-double scale (~1e-30 relative on well-conditioned input), which is
// what lets the rtol ladder PROVE that a DD rerun met the tolerance instead
// of assuming it.

#include <cuComplex.h>
#include <math.h>

#include <cstdint>

#include "certified_rounding.cuh"
#include "dd.cuh"
#include "subset_engine.cuh"

namespace gbs {

constexpr int PERM_DDC_MAX_N = 28; // == permanent_dd.cu's cap
constexpr int HAF_DDC_MAX_N = 16; // == hafnian_dd.cu's cap

constexpr double GBS_U_DD = 7.888609052210118e-31; // 2^-100
constexpr double GBS_C_MUL_DD = 1.2621774483536189e-29; // 2^-96
constexpr double GBS_DD_ABS_SAFE = 1.0 + 1.7763568394002505e-15; // 1 + 2^-49
constexpr double GBS_DD_UFL = 8e-323;

// TU-local copies of permanent_dd.cu's inline conversions (identical bodies;
// inline + same definition keeps ODR happy when the TUs are linked together).
__host__ __device__ inline ddcomplex ddc_from_cu(cuDoubleComplex z) {
  return ddcomplex{dd{cuCreal(z), 0.0}, dd{cuCimag(z), 0.0}};
}
__host__ __device__ inline cuDoubleComplex ddc_to_cu(ddcomplex a) {
  return make_cuDoubleComplex(a.re.hi + a.re.lo, a.im.hi + a.im.lo);
}

__device__ inline ddcomplex ddc_neg_local(ddcomplex a) {
  return ddcomplex{dd_neg(a.re), dd_neg(a.im)};
}

__device__ inline double ddc_absu(ddcomplex z) {
  return ru_mul(hypot(z.re.hi, z.im.hi), GBS_DD_ABS_SAFE);
}

__device__ inline double cert_gamma_dd(double k) {
  double ku = k * GBS_U_DD;
  return ru_mul(ku / (1.0 - ku), 1.0 + 4.0 * GBS_U_DD);
}

// --------------------------------------------------------------------------
// certified DD permanent (value path == perm_glynn_dd_kernel)
// --------------------------------------------------------------------------

__global__ void perm_dd_certified_kernel(const cuDoubleComplex* __restrict__ mats,
                                         int n, int batch,
                                         cuDoubleComplex* __restrict__ out,
                                         double* __restrict__ bound) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  const cuDoubleComplex* A = mats + (size_t)b * n * n;

  if (n == 0) { out[b] = make_cuDoubleComplex(1.0, 0.0); bound[b] = 0.0; return; }
  if (n > PERM_DDC_MAX_N) {
    out[b] = make_cuDoubleComplex(NAN, NAN); bound[b] = INFINITY; return;
  }

  ddcomplex rowsum[PERM_DDC_MAX_N];
  double e_r[PERM_DDC_MAX_N];
  const double g_sum = cert_gamma_dd((double)n);
  const double g_prod = cert_gamma_dd(6.0 * (n > 1 ? n - 1 : 0));

  for (int r = 0; r < n; ++r) {
    ddcomplex s = ddc_from_cu(make_cuDoubleComplex(0.0, 0.0));
    double asum = 0.0;
    for (int c = 0; c < n; ++c) {
      s = ddc_add(s, ddc_from_cu(A[r * n + c]));
      asum = ru_add(asum, ru_mul(cuCabs(A[r * n + c]), GBS_DD_ABS_SAFE));
    }
    rowsum[r] = s;
    e_r[r] = ru_mul(g_sum, asum);
  }

  auto term_bound = [&](void) -> double {
    // polydisc + product-rounding, DD scale: p_hi = ru prod (|r|+e); the exact
    // lower product is within (1 - n*2^-49) of p of upper magnitudes, so the
    // spread is bounded by p_hi - p_hi*(1 - slack) + polydisc term; we keep the
    // simpler (and looser) form e_p ~ sum_i e_i/(|r_i|) * p_hi + g_prod * p_hi,
    // computed as ratios in ru arithmetic. Ratios guard division by zero.
    double p_hi = 1.0, ratio = 0.0;
    for (int r = 0; r < n; ++r) {
      double m = ddc_absu(rowsum[r]);
      p_hi = ru_mul(p_hi, ru_add(m, e_r[r]));
      double denom = m > 0.0 ? m : GBS_DD_UFL;
      ratio = ru_add(ratio, ru_div(e_r[r], denom));
    }
    return ru_add(ru_mul(p_hi, ratio),
                  ru_add(ru_mul(g_prod, p_hi), (n + 2) * GBS_DD_UFL));
  };

  ddcomplex prod = rowsum[0];
  for (int r = 1; r < n; ++r) prod = ddc_mul(prod, rowsum[r]);
  ddcomplex total = prod;
  double e_tot = term_bound();

  int sign = 1;
  const uint64_t terms = 1ull << (n - 1);
  for (uint64_t i = 1; i < terms; ++i) {
    int k = flipped_index(i);
    int col = k + 1;
    double step = gray_bit_of(i, k) ? -2.0 : +2.0; // exact scale in DD too
    for (int r = 0; r < n; ++r) {
      rowsum[r] = ddc_add(rowsum[r], ddc_mul_d(ddc_from_cu(A[r * n + col]), step));
      e_r[r] = ru_add(e_r[r], ru_mul(GBS_U_DD, ddc_absu(rowsum[r])));
    }
    sign = -sign;
    ddcomplex p = rowsum[0];
    for (int r = 1; r < n; ++r) p = ddc_mul(p, rowsum[r]);
    total = sign > 0 ? ddc_add(total, p) : ddc_sub(total, p);
    e_tot = ru_add(e_tot, ru_add(term_bound(), ru_mul(GBS_U_DD, ddc_absu(total))));
  }

  total = ddc_mul_d(total, 1.0 / (double)terms); // power of two: exact
  out[b] = ddc_to_cu(total);
  // the DD->double collapse on output adds one 2^-53-level rounding of the value
  bound[b] = ru_add(e_tot / (double)terms,
                    ru_mul(1.1102230246251565e-16, ddc_absu(total)));
}

extern "C" void gbs_perm_dd_certified_batched(const cuDoubleComplex* d_mats, int n,
                                              int batch, cuDoubleComplex* d_out,
                                              double* d_bound, cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 64;
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(perm_dd_certified_kernel, grid, block, stream, d_mats, n, batch, d_out, d_bound);
}

// --------------------------------------------------------------------------
// certified DD hafnian (value path == haf_powertrace_dd_kernel)
// --------------------------------------------------------------------------

__global__ void haf_dd_certified_kernel(const cuDoubleComplex* __restrict__ mats,
                                        int N, int batch,
                                        cuDoubleComplex* __restrict__ out,
                                        double* __restrict__ bound) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  const cuDoubleComplex* A = mats + (size_t)b * N * N;

  if (N == 0) { out[b] = make_cuDoubleComplex(1.0, 0.0); bound[b] = 0.0; return; }
  if (N & 1) { out[b] = make_cuDoubleComplex(0.0, 0.0); bound[b] = 0.0; return; }
  if (N > HAF_DDC_MAX_N) {
    out[b] = make_cuDoubleComplex(NAN, NAN); bound[b] = INFINITY; return;
  }
  int n = N / 2;

  ddcomplex total = ddc_from_cu(make_cuDoubleComplex(0.0, 0.0));
  double e_tot = 0.0;
  for (uint64_t mask = 0; mask < (1ull << n); ++mask) {
    int m = __popcll((long long)mask);
    int size = 2 * m;
    int pidx[HAF_DDC_MAX_N / 2];
    int pc = 0;
    for (int i = 0; i < n; ++i)
      if ((mask >> i) & 1ull) pidx[pc++] = i;
    ddcomplex BX[HAF_DDC_MAX_N * HAF_DDC_MAX_N];
    for (int r = 0; r < size; ++r) {
      int rr = 2 * pidx[r >> 1] + (r & 1);
      for (int c = 0; c < size; ++c) {
        int cc = 2 * pidx[c >> 1] + ((c & 1) ^ 1);
        BX[r * size + c] = ddc_from_cu(A[rr * N + cc]); // exact embed
      }
    }

    ddcomplex coeff;
    double cb;
    if (size == 0) {
      coeff = ddc_from_cu(make_cuDoubleComplex(n == 0 ? 1.0 : 0.0, 0.0));
      cb = 0.0;
    } else {
      const double c_dot = cert_gamma_dd(6.0 * size + 12.0);
      const double g_tr = cert_gamma_dd((double)size);
      ddcomplex p[HAF_DDC_MAX_N + 1];
      double e_tr[HAF_DDC_MAX_N + 1];
      ddcomplex P[HAF_DDC_MAX_N * HAF_DDC_MAX_N];
      double E[HAF_DDC_MAX_N * HAF_DDC_MAX_N];
      double Babs[HAF_DDC_MAX_N * HAF_DDC_MAX_N]; // |BX| hoisted (pure CSE)
      double Pabs[HAF_DDC_MAX_N * HAF_DDC_MAX_N];
      for (int i = 0; i < size * size; ++i) {
        P[i] = BX[i]; E[i] = 0.0; Babs[i] = ddc_absu(BX[i]);
      }
      for (int k = 1; k <= n; ++k) {
        ddcomplex tr = ddc_from_cu(make_cuDoubleComplex(0.0, 0.0));
        double etr = 0.0, atr = 0.0;
        for (int i = 0; i < size; ++i) {
          tr = ddc_add(tr, P[i * size + i]);
          etr = ru_add(etr, E[i * size + i]);
          atr = ru_add(atr, ddc_absu(P[i * size + i]));
        }
        p[k] = tr;
        e_tr[k] = ru_add(etr, ru_mul(g_tr, atr));
        if (k < n) {
          for (int i = 0; i < size * size; ++i) Pabs[i] = ddc_absu(P[i]);
          ddcomplex T[HAF_DDC_MAX_N * HAF_DDC_MAX_N];
          double TB[HAF_DDC_MAX_N * HAF_DDC_MAX_N];
          for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j) {
              ddcomplex s = ddc_from_cu(make_cuDoubleComplex(0.0, 0.0));
              double sb = 0.0, sm = 0.0;
              for (int t = 0; t < size; ++t) {
                double ca = Babs[t * size + j];
                s = ddc_add(s, ddc_mul(P[i * size + t], BX[t * size + j]));
                sb = ru_add(sb, ru_mul(E[i * size + t], ca));
                sm = ru_add(sm, ru_mul(Pabs[i * size + t], ca));
              }
              T[i * size + j] = s;
              TB[i * size + j] = ru_add(ru_add(sb, ru_mul(c_dot, sm)), size * GBS_DD_UFL);
            }
          for (int i = 0; i < size * size; ++i) { P[i] = T[i]; E[i] = TB[i]; }
        }
      }
      ddcomplex e[HAF_DDC_MAX_N + 1];
      double eb[HAF_DDC_MAX_N + 1];
      e[0] = ddc_from_cu(make_cuDoubleComplex(1.0, 0.0));
      eb[0] = 0.0;
      for (int j = 1; j <= n; ++j) {
        ddcomplex acc = ddc_from_cu(make_cuDoubleComplex(0.0, 0.0));
        double acc_b = 0.0;
        for (int k = 1; k <= j; ++k) {
          ddcomplex half_pk = ddc_mul_d(p[k], 0.5); // exact
          double A_ = ddc_absu(half_pk), B_ = ddc_absu(e[j - k]);
          double e_hk = 0.5 * e_tr[k];
          acc = ddc_add(acc, ddc_mul(half_pk, e[j - k]));
          double pb = ru_add(ru_add(ru_mul(A_, eb[j - k]), ru_mul(B_, e_hk)),
                             ru_add(ru_mul(e_hk, eb[j - k]),
                                    ru_add(ru_mul(GBS_C_MUL_DD, ru_mul(A_, B_)),
                                           GBS_DD_UFL)));
          acc_b = ru_add(acc_b, ru_add(pb, ru_mul(GBS_U_DD, ddc_absu(acc))));
        }
        e[j] = ddc_div_d(acc, (double)j); // proper DD divide
        eb[j] = ru_add(ru_div(acc_b, (double)j), ru_mul(GBS_U_DD, ddc_absu(e[j])));
      }
      coeff = e[n];
      cb = eb[n];
    }
    ddcomplex term = ((n - m) & 1) ? ddc_neg_local(coeff) : coeff;
    total = ddc_add(total, term);
    e_tot = ru_add(e_tot, ru_add(cb, ru_mul(GBS_U_DD, ddc_absu(total))));
  }
  out[b] = ddc_to_cu(total);
  // the D->double collapse on output adds one 2^-53-level rounding of the value
  bound[b] = ru_add(e_tot, ru_mul(1.1102230246251565e-16, ddc_absu(total)));
}

extern "C" void gbs_haf_dd_certified_batched(const cuDoubleComplex* d_mats, int N,
                                             int batch, cuDoubleComplex* d_out,
                                             double* d_bound, cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 64;
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(haf_dd_certified_kernel, grid, block, stream, d_mats, N, batch, d_out, d_bound);
}

} // namespace gbs
