// certified.cu -- batched CERTIFIED evaluation: the fp64 value PLUS a
// rigorous a-posteriori error bound, per element.
//
// session (check_certified.cu), like every kernel before it.
//
// The value path replicates the plain fp64 kernels' arithmetic (permanent.cu's
// Glynn walk; hafnian.cu's power-trace/Newton) expression-for-expression; the bound
// recipes are the device translation of the *validated* CPU derivation
// (cpu_ref/certified.py -- see its module docstring for the validity model), with
// one upgrade over the CPU reference : the BOUND
// accumulators run in UPWARD-ROUNDED arithmetic -- CUDA's per-instruction
// __dadd_ru/__dmul_ru/__ddiv_ru on the device (no rounding-mode switch, the reason
// GPUs suit certified arithmetic BETTER than CPUs), and nextafter(x, +inf) on the
// host shim (>= the true upward-rounded result, so host-shim runs stay valid, one
// ulp looser). Every accumulator operation is then individually an upper bound and
// the CPU reference's final "accumulator inflation" factor disappears. The
// standard-model gamma constants remain where they describe the VALUE path's
// rounding (they model the plain kernel's arithmetic, which stays round-to-nearest
// + FMA-contracted; an FMA performs fewer roundings than the two the model
// charges, so the bounds hold under contraction).
//
// Mapping: one evaluation per thread, batch across the grid, exactly like the
// plain kernels. Outputs: out[b] (complex value) and bound[b] (double,
// |out[b] - exact| <= bound[b]; +inf means "cannot certify", never a wrong claim).

#include <cuComplex.h>
#include <math.h>

#include <cstdint>

#include "certified_rounding.cuh"
#include "subset_engine.cuh"

namespace gbs {

constexpr int PERM_CERT_MAX_N = 28; // == permanent.cu's PERM_MAX_N
constexpr int HAF_CERT_MAX_N = 20; // == hafnian.cu's HAF_MAX_N

constexpr double GBS_U = 1.1102230246251565e-16; // 2^-53
constexpr double GBS_ABS_SAFE = 1.0 + 8.881784197001252e-16; // 1 + 2^-50
constexpr double GBS_UFL = 8e-323; // per-multiply underflow slop
constexpr double GBS_C_MUL = 3.0 * GBS_U;

__device__ inline double cert_gamma(double k) {
  double ku = k * GBS_U;
  return ru_mul(ku / (1.0 - ku), 1.0 + 4.0 * GBS_U); // ru slack on the constant itself
}

// |z| upper / lower bounds (cuCabs is a faithful hypot: within 1 ulp).
__device__ inline double cert_absu(cuDoubleComplex z) { return ru_mul(cuCabs(z), GBS_ABS_SAFE); }
__device__ inline double cert_absl(cuDoubleComplex z) { return rd_mul(cuCabs(z), 2.0 - GBS_ABS_SAFE); }

__device__ inline cuDoubleComplex cert_scale(cuDoubleComplex a, double s) {
  return make_cuDoubleComplex(cuCreal(a) * s, cuCimag(a) * s);
}

// --------------------------------------------------------------------------
// certified permanent: the Glynn walk of permanent.cu + the bound state
// (row-sum bounds e_r; per-term polydisc + product-rounding; signed-sum term)
// --------------------------------------------------------------------------

__global__ void perm_certified_kernel(const cuDoubleComplex* __restrict__ mats,
                                      int n, int batch,
                                      cuDoubleComplex* __restrict__ out,
                                      double* __restrict__ bound) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  const cuDoubleComplex* A = mats + (size_t)b * n * n;

  if (n == 0) { out[b] = make_cuDoubleComplex(1.0, 0.0); bound[b] = 0.0; return; }
  if (n > PERM_CERT_MAX_N) {
    out[b] = make_cuDoubleComplex(NAN, NAN); bound[b] = INFINITY; return;
  }

  cuDoubleComplex rowsum[PERM_CERT_MAX_N];
  double e_r[PERM_CERT_MAX_N];
  const double g_sum = cert_gamma((double)n);
  const double g_prod = cert_gamma(3.0 * (n > 1 ? n - 1 : 0));

  for (int r = 0; r < n; ++r) {
    cuDoubleComplex s = make_cuDoubleComplex(0.0, 0.0);
    double asum = 0.0;
    for (int c = 0; c < n; ++c) {
      s = cuCadd(s, A[r * n + c]);
      asum = ru_add(asum, cert_absu(A[r * n + c]));
    }
    rowsum[r] = s;
    e_r[r] = ru_mul(g_sum, asum);
  }

  // per-term bound: (prod(|r|+e) - prod|r|) + g_prod*prod(|r|+e) + underflow slop.
  // p_hi is an upward-rounded product of UPPER magnitudes, p_lo a downward-rounded
  // product of LOWER magnitudes, so (p_hi - p_lo) >= the exact polydisc spread.
  auto term_bound = [&](void) -> double {
    double p_lo = 1.0, p_hi = 1.0;
    for (int r = 0; r < n; ++r) {
      p_lo = rd_mul(p_lo, cert_absl(rowsum[r]));
      p_hi = ru_mul(p_hi, ru_add(cert_absu(rowsum[r]), e_r[r]));
    }
    double d = ru_add(p_hi, -p_lo);
    return ru_add(ru_add(d > 0.0 ? d : 0.0, ru_mul(g_prod, p_hi)), (n + 2) * GBS_UFL);
  };

  cuDoubleComplex prod = rowsum[0];
  for (int r = 1; r < n; ++r) prod = cuCmul(prod, rowsum[r]);
  cuDoubleComplex total = prod;
  double e_tot = term_bound();

  int sign = 1;
  const uint64_t terms = 1ull << (n - 1);
  for (uint64_t i = 1; i < terms; ++i) {
    int k = flipped_index(i);
    int col = k + 1;
    double step = gray_bit_of(i, k) ? -2.0 : +2.0; // power-of-two scale: exact
    for (int r = 0; r < n; ++r) {
      rowsum[r] = cuCadd(rowsum[r], cert_scale(A[r * n + col], step));
      e_r[r] = ru_add(e_r[r], ru_mul(GBS_U, cert_absu(rowsum[r]))); // one rounding/elem/step
    }
    sign = -sign;
    cuDoubleComplex p = rowsum[0];
    for (int r = 1; r < n; ++r) p = cuCmul(p, rowsum[r]);
    total = sign > 0 ? cuCadd(total, p) : cuCsub(total, p);
    e_tot = ru_add(e_tot, ru_add(term_bound(), ru_mul(GBS_U, cert_absu(total))));
  }

  out[b] = cert_scale(total, 1.0 / (double)terms); // exact power-of-two scale
  bound[b] = e_tot / (double)terms; // exact power-of-two scale
}

extern "C" void gbs_perm_certified_batched(const cuDoubleComplex* d_mats, int n,
                                           int batch, cuDoubleComplex* d_out,
                                           double* d_bound, cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 128;
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(perm_certified_kernel, grid, block, stream, d_mats, n, batch, d_out, d_bound);
}

// --------------------------------------------------------------------------
// certified hafnian: the power-trace/Newton of hafnian.cu + entrywise matmul
// bounds (E <- E|C| + c_dot|P||C|), trace-sum bounds, Newton pair recurrence,
// and the signed subset sum (whose bound is the rigorous kappa)
// --------------------------------------------------------------------------

// coeff + bound for one subset's [lambda^n] exp(sum_k tr(C^k)/(2k) lambda^k).
// Value path == hafnian.cu's exp_newton_coeff_t (same expression order).
template <int MAXN>
__device__ void exp_newton_coeff_cert(const cuDoubleComplex* C, int size, int n,
                                      cuDoubleComplex* coeff, double* cbound) {
  if (size == 0) {
    *coeff = make_cuDoubleComplex(n == 0 ? 1.0 : 0.0, 0.0);
    *cbound = 0.0;
    return;
  }
  const double c_dot = cert_gamma(3.0 * size + 6.0);
  const double g_tr = cert_gamma((double)size);

  cuDoubleComplex p[MAXN + 1];
  double e_tr[MAXN + 1];
  cuDoubleComplex P[MAXN * MAXN];
  double E[MAXN * MAXN];
  // |C| and |P| hoisted out of the dot loops (pure CSE -- identical doubles,
  // identical bounds; the CPU reference has always done this): removes an
  // O(size) hypot factor per matmul entry, the measured 8-11x haf-certified
  // cost driver on the A100.
  double Cabs[MAXN * MAXN];
  double Pabs[MAXN * MAXN];
  for (int i = 0; i < size * size; ++i) {
    P[i] = C[i]; E[i] = 0.0; Cabs[i] = cert_absu(C[i]);
  }

  for (int k = 1; k <= n; ++k) {
    cuDoubleComplex tr = make_cuDoubleComplex(0.0, 0.0);
    double etr = 0.0, atr = 0.0;
    for (int i = 0; i < size; ++i) {
      tr = cuCadd(tr, P[i * size + i]);
      etr = ru_add(etr, E[i * size + i]);
      atr = ru_add(atr, cert_absu(P[i * size + i]));
    }
    p[k] = tr;
    e_tr[k] = ru_add(etr, ru_mul(g_tr, atr));
    if (k < n) { // P <- P*C ; E <- E|C| + c_dot |P||C| + size*UFL, per entry
      for (int i = 0; i < size * size; ++i) Pabs[i] = cert_absu(P[i]);
      cuDoubleComplex T[MAXN * MAXN];
      double Tb[MAXN * MAXN];
      for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j) {
          cuDoubleComplex s = make_cuDoubleComplex(0.0, 0.0);
          double sb = 0.0, sm = 0.0;
          for (int t = 0; t < size; ++t) {
            double ca = Cabs[t * size + j];
            s = cuCadd(s, cuCmul(P[i * size + t], C[t * size + j]));
            sb = ru_add(sb, ru_mul(E[i * size + t], ca));
            sm = ru_add(sm, ru_mul(Pabs[i * size + t], ca));
          }
          T[i * size + j] = s;
          Tb[i * size + j] = ru_add(ru_add(sb, ru_mul(c_dot, sm)), size * GBS_UFL);
        }
      for (int i = 0; i < size * size; ++i) { P[i] = T[i]; E[i] = Tb[i]; }
    }
  }

  cuDoubleComplex e[MAXN + 1];
  double eb[MAXN + 1];
  e[0] = make_cuDoubleComplex(1.0, 0.0);
  eb[0] = 0.0;
  for (int j = 1; j <= n; ++j) {
    cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
    double acc_b = 0.0;
    for (int k = 1; k <= j; ++k) {
      cuDoubleComplex half_pk = make_cuDoubleComplex(0.5 * cuCreal(p[k]), 0.5 * cuCimag(p[k]));
      double A_ = cert_absu(half_pk), B_ = cert_absu(e[j - k]);
      double e_hk = 0.5 * e_tr[k]; // p/2 is exact
      acc = cuCadd(acc, cuCmul(half_pk, e[j - k]));
      double pb = ru_add(ru_add(ru_mul(A_, eb[j - k]), ru_mul(B_, e_hk)),
                         ru_add(ru_mul(e_hk, eb[j - k]),
                                ru_add(ru_mul(GBS_C_MUL, ru_mul(A_, B_)), GBS_UFL)));
      acc_b = ru_add(acc_b, ru_add(pb, ru_mul(GBS_U, cert_absu(acc))));
    }
    e[j] = make_cuDoubleComplex(cuCreal(acc) / j, cuCimag(acc) / j);
    eb[j] = ru_add(ru_div(acc_b, (double)j), ru_mul(GBS_U, cert_absu(e[j])));
  }
  *coeff = e[n];
  *cbound = eb[n];
}

__global__ void haf_certified_kernel(const cuDoubleComplex* __restrict__ mats,
                                     int N, int batch,
                                     cuDoubleComplex* __restrict__ out,
                                     double* __restrict__ bound) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  const cuDoubleComplex* A = mats + (size_t)b * N * N;

  if (N == 0) { out[b] = make_cuDoubleComplex(1.0, 0.0); bound[b] = 0.0; return; }
  if (N & 1) { out[b] = make_cuDoubleComplex(0.0, 0.0); bound[b] = 0.0; return; }
  if (N > HAF_CERT_MAX_N) {
    out[b] = make_cuDoubleComplex(NAN, NAN); bound[b] = INFINITY; return;
  }
  int n = N / 2;

  cuDoubleComplex total = make_cuDoubleComplex(0.0, 0.0);
  double e_tot = 0.0;
  for (uint64_t mask = 0; mask < (1ull << n); ++mask) {
    int m = __popcll((long long)mask);
    int size = 2 * m;
    cuDoubleComplex BX[HAF_CERT_MAX_N * HAF_CERT_MAX_N];
    int pidx[HAF_CERT_MAX_N / 2];
    int pc = 0;
    for (int i = 0; i < n; ++i)
      if ((mask >> i) & 1ull) pidx[pc++] = i;
    for (int r = 0; r < size; ++r) { // exact gather; diagonal ignored via X
      int pr = pidx[r >> 1], rr = 2 * pr + (r & 1);
      for (int c = 0; c < size; ++c) {
        int pc2 = pidx[c >> 1];
        int cc = 2 * pc2 + ((c & 1) ^ 1);
        BX[r * size + c] = A[rr * N + cc];
      }
    }
    cuDoubleComplex coeff;
    double cb;
    exp_newton_coeff_cert<HAF_CERT_MAX_N>(BX, size, n, &coeff, &cb);
    cuDoubleComplex term = ((n - m) & 1)
        ? make_cuDoubleComplex(-cuCreal(coeff), -cuCimag(coeff)) : coeff;
    total = cuCadd(total, term); // same accumulation as the plain kernel
    e_tot = ru_add(e_tot, ru_add(cb, ru_mul(GBS_U, cert_absu(total))));
  }
  out[b] = total;
  bound[b] = e_tot;
}

extern "C" void gbs_haf_certified_batched(const cuDoubleComplex* d_mats, int N,
                                          int batch, cuDoubleComplex* d_out,
                                          double* d_bound, cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 64; // large per-thread footprint, like the plain hafnian
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(haf_certified_kernel, grid, block, stream, d_mats, N, batch, d_out, d_bound);
}


// --------------------------------------------------------------------------
// certified loop hafnian: loop_hafnian.cu's per-subset chains (C with the
// diagonal, P = C^k, Q = C^{k-1}, v_k = (1/2) d^T X Q d) mirrored op-for-op,
// with entrywise bounds for BOTH chains and the two dot products
// --------------------------------------------------------------------------

constexpr int LHAF_CERT_MAX_N = 20; // == loop_hafnian.cu's LHAF_MAX_N

__global__ void lhaf_certified_kernel(const cuDoubleComplex* __restrict__ mats,
                                      int N, int batch,
                                      cuDoubleComplex* __restrict__ out,
                                      double* __restrict__ bound) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  const cuDoubleComplex* A = mats + (size_t)b * N * N;

  if (N == 0) { out[b] = make_cuDoubleComplex(1.0, 0.0); bound[b] = 0.0; return; }
  if ((N & 1) || N > LHAF_CERT_MAX_N) { // odd N is augmented by the caller
    out[b] = make_cuDoubleComplex(NAN, NAN); bound[b] = INFINITY; return;
  }
  int n = N / 2;

  cuDoubleComplex total = make_cuDoubleComplex(0.0, 0.0);
  double e_tot = 0.0;
  for (uint64_t mask = 0; mask < (1ull << n); ++mask) {
    int m = __popcll((long long)mask);
    if (m == 0) continue; // empty -> [lambda^n] exp(0) = 0 for n >= 1
    int size = 2 * m;
    int pidx[LHAF_CERT_MAX_N / 2];
    int pc = 0;
    for (int i = 0; i < n; ++i)
      if ((mask >> i) & 1ull) pidx[pc++] = i;

    cuDoubleComplex C[LHAF_CERT_MAX_N * LHAF_CERT_MAX_N]; // exact gathers
    cuDoubleComplex d[LHAF_CERT_MAX_N];
    for (int r = 0; r < size; ++r) {
      int rr = 2 * pidx[r >> 1] + (r & 1);
      for (int c = 0; c < size; ++c) {
        int cc = 2 * pidx[c >> 1] + ((c & 1) ^ 1);
        C[r * size + c] = A[rr * N + cc];
      }
      d[r] = A[rr * N + rr];
    }

    const double c_dot = cert_gamma(3.0 * size + 6.0);
    const double g_tr = cert_gamma((double)size);
    double Cabs[LHAF_CERT_MAX_N * LHAF_CERT_MAX_N];
    double Qabs[LHAF_CERT_MAX_N * LHAF_CERT_MAX_N];
    for (int i = 0; i < size * size; ++i) Cabs[i] = cert_absu(C[i]);

    cuDoubleComplex kg[LHAF_CERT_MAX_N + 1];
    double e_kg[LHAF_CERT_MAX_N + 1];
    cuDoubleComplex P[LHAF_CERT_MAX_N * LHAF_CERT_MAX_N];
    cuDoubleComplex Q[LHAF_CERT_MAX_N * LHAF_CERT_MAX_N];
    double EP[LHAF_CERT_MAX_N * LHAF_CERT_MAX_N];
    double EQ[LHAF_CERT_MAX_N * LHAF_CERT_MAX_N];
    for (int i = 0; i < size * size; ++i) {
      P[i] = C[i]; EP[i] = 0.0;
      Q[i] = make_cuDoubleComplex(0.0, 0.0); EQ[i] = 0.0;
    }
    for (int i = 0; i < size; ++i) Q[i * size + i] = make_cuDoubleComplex(1.0, 0.0);

    for (int k = 1; k <= n; ++k) {
      cuDoubleComplex pk = make_cuDoubleComplex(0.0, 0.0);
      double e_pk = 0.0, a_pk = 0.0;
      for (int i = 0; i < size; ++i) {
        pk = cuCadd(pk, P[i * size + i]);
        e_pk = ru_add(e_pk, EP[i * size + i]);
        a_pk = ru_add(a_pk, cert_absu(P[i * size + i]));
      }
      e_pk = ru_add(e_pk, ru_mul(g_tr, a_pk));

      // w = Q d (d exact); e_w tracks input error + the dot's own rounding
      for (int i = 0; i < size * size; ++i) Qabs[i] = cert_absu(Q[i]);
      cuDoubleComplex w[LHAF_CERT_MAX_N];
      double e_w[LHAF_CERT_MAX_N];
      for (int a = 0; a < size; ++a) {
        cuDoubleComplex sacc = make_cuDoubleComplex(0.0, 0.0);
        double sb = 0.0, sm = 0.0;
        for (int c = 0; c < size; ++c) {
          double da = cert_absu(d[c]);
          sacc = cuCadd(sacc, cuCmul(Q[a * size + c], d[c]));
          sb = ru_add(sb, ru_mul(EQ[a * size + c], da));
          sm = ru_add(sm, ru_mul(Qabs[a * size + c], da));
        }
        w[a] = sacc;
        e_w[a] = ru_add(ru_add(sb, ru_mul(c_dot, sm)), size * GBS_UFL);
      }
      // v_k = (1/2) sum_a d[a] * w[a^1]
      cuDoubleComplex vk = make_cuDoubleComplex(0.0, 0.0);
      double e_vk = 0.0, a_vk = 0.0;
      for (int a = 0; a < size; ++a) {
        int xa = a ^ 1;
        vk = cuCadd(vk, cuCmul(d[a], w[xa]));
        double da = cert_absu(d[a]);
        e_vk = ru_add(e_vk, ru_mul(da, e_w[xa]));
        a_vk = ru_add(a_vk, ru_mul(da, cert_absu(w[xa])));
      }
      e_vk = ru_add(e_vk, ru_add(ru_mul(c_dot, a_vk), size * GBS_UFL));
      vk = make_cuDoubleComplex(0.5 * cuCreal(vk), 0.5 * cuCimag(vk));
      e_vk = 0.5 * e_vk; // exact halving

      cuDoubleComplex half_pk = make_cuDoubleComplex(0.5 * cuCreal(pk), 0.5 * cuCimag(pk));
      cuDoubleComplex kvk = make_cuDoubleComplex(k * cuCreal(vk), k * cuCimag(vk));
      kg[k] = cuCadd(half_pk, kvk);
      double e_kvk = ru_add(ru_mul((double)k, e_vk), ru_mul(GBS_U, cert_absu(kvk)));
      e_kg[k] = ru_add(ru_add(0.5 * e_pk, e_kvk), ru_mul(GBS_U, cert_absu(kg[k])));

      if (k < n) { // Q <- P; P <- P*C (with bounds)
        for (int i = 0; i < size * size; ++i) { Q[i] = P[i]; EQ[i] = EP[i]; }
        for (int i = 0; i < size * size; ++i) Qabs[i] = cert_absu(P[i]); // |P| for the matmul
        cuDoubleComplex T[LHAF_CERT_MAX_N * LHAF_CERT_MAX_N];
        double TB[LHAF_CERT_MAX_N * LHAF_CERT_MAX_N];
        for (int i = 0; i < size; ++i)
          for (int j = 0; j < size; ++j) {
            cuDoubleComplex sacc = make_cuDoubleComplex(0.0, 0.0);
            double sb = 0.0, sm = 0.0;
            for (int t = 0; t < size; ++t) {
              double ca = Cabs[t * size + j];
              sacc = cuCadd(sacc, cuCmul(P[i * size + t], C[t * size + j]));
              sb = ru_add(sb, ru_mul(EP[i * size + t], ca));
              sm = ru_add(sm, ru_mul(Qabs[i * size + t], ca));
            }
            T[i * size + j] = sacc;
            TB[i * size + j] = ru_add(ru_add(sb, ru_mul(c_dot, sm)), size * GBS_UFL);
          }
        for (int i = 0; i < size * size; ++i) { P[i] = T[i]; EP[i] = TB[i]; }
      }
    }

    // Newton recurrence on (kg, e_kg), pair arithmetic (mirrors exp_coeff_from_kg)
    cuDoubleComplex e[LHAF_CERT_MAX_N + 1];
    double eb[LHAF_CERT_MAX_N + 1];
    e[0] = make_cuDoubleComplex(1.0, 0.0);
    eb[0] = 0.0;
    for (int j = 1; j <= n; ++j) {
      cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
      double acc_b = 0.0;
      for (int k = 1; k <= j; ++k) {
        double A_ = cert_absu(kg[k]), B_ = cert_absu(e[j - k]);
        acc = cuCadd(acc, cuCmul(kg[k], e[j - k]));
        double pb = ru_add(ru_add(ru_mul(A_, eb[j - k]), ru_mul(B_, e_kg[k])),
                           ru_add(ru_mul(e_kg[k], eb[j - k]),
                                  ru_add(ru_mul(GBS_C_MUL, ru_mul(A_, B_)), GBS_UFL)));
        acc_b = ru_add(acc_b, ru_add(pb, ru_mul(GBS_U, cert_absu(acc))));
      }
      e[j] = make_cuDoubleComplex(cuCreal(acc) / j, cuCimag(acc) / j);
      eb[j] = ru_add(ru_div(acc_b, (double)j), ru_mul(GBS_U, cert_absu(e[j])));
    }
    cuDoubleComplex term = ((n - m) & 1)
        ? make_cuDoubleComplex(-cuCreal(e[n]), -cuCimag(e[n])) : e[n];
    total = cuCadd(total, term);
    e_tot = ru_add(e_tot, ru_add(eb[n], ru_mul(GBS_U, cert_absu(total))));
  }
  out[b] = total;
  bound[b] = e_tot;
}

extern "C" void gbs_lhaf_certified_batched(const cuDoubleComplex* d_mats, int N,
                                           int batch, cuDoubleComplex* d_out,
                                           double* d_bound, cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 64;
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(lhaf_certified_kernel, grid, block, stream, d_mats, N, batch, d_out, d_bound);
}

// --------------------------------------------------------------------------
// certified torontonian: per-thread pair-arithmetic complex LU with partial
// pivoting + an a-posteriori inverse-sqrt bound (device translation of the
// cpu_ref/certified.py toolkit; the value is this kernel's own LU, agreeing
// with the plain kernel to tier tolerance -- the bound covers THIS value)
// --------------------------------------------------------------------------

constexpr int TOR_CERT_MAX_DIM = 24; // 2n, == torontonian.cu's cap

__device__ inline void tor_p_mul(cuDoubleComplex a, double ea, cuDoubleComplex b,
                                 double eb, cuDoubleComplex* v, double* e) {
  *v = cuCmul(a, b);
  double A_ = cert_absu(a), B_ = cert_absu(b);
  *e = ru_add(ru_add(ru_mul(A_, eb), ru_mul(B_, ea)),
              ru_add(ru_mul(ea, eb), ru_add(ru_mul(GBS_C_MUL, ru_mul(A_, B_)), GBS_UFL)));
}

// (a +- ea) / (b +- eb): textbook component division; bound derivation in
// cpu_ref/certified.py::_p_div. Guards degrade to bound = inf, never a claim.
__device__ inline void tor_p_div(cuDoubleComplex a, double ea, cuDoubleComplex b,
                                 double eb, cuDoubleComplex* v, double* e) {
  double Bm = cuCabs(b);
  if (!isfinite(Bm) || Bm == 0.0 || eb >= 0.5 * Bm) {
    *v = make_cuDoubleComplex(NAN, NAN); *e = INFINITY; return;
  }
  double c = cuCreal(b), dd_ = cuCimag(b);
  double den = c * c + dd_ * dd_;
  double vre = (cuCreal(a) * c + cuCimag(a) * dd_) / den;
  double vim = (cuCimag(a) * c - cuCreal(a) * dd_) / den;
  *v = make_cuDoubleComplex(vre, vim);
  if (!isfinite(den) || !isfinite(vre) || !isfinite(vim)) { *e = INFINITY; return; }
  double Am = cert_absu(a);
  double den_lo = rd_mul(rd_mul(Bm, Bm), 1.0 - 8.0 * GBS_U);
  double e_alg = ru_add(ru_div(ru_mul(8.0 * GBS_U, ru_mul(Am, Bm)), den_lo),
                        ru_add(ru_mul(8.0 * GBS_U, cert_absu(*v)), GBS_UFL));
  double e_pert = ru_div(ru_add(ru_mul(ea, Bm), ru_mul(eb, Am)),
                         rd_mul(Bm, Bm - eb));
  *e = ru_add(e_alg, e_pert);
}

// (1 / sqrt(z +- ez), bound), principal branch; a-posteriori via |s^2 - z|
// (cpu_ref/certified.py::_p_inv_sqrt derives the monotonicity argument).
__device__ inline void tor_p_inv_sqrt(cuDoubleComplex z, double ez,
                                      cuDoubleComplex* v, double* e) {
  double Zm = cuCabs(z);
  if (!isfinite(Zm) || ez >= 0.5 * Zm) { *v = make_cuDoubleComplex(NAN, NAN); *e = INFINITY; return; }
  if (cuCreal(z) < 0.0 && fabs(cuCimag(z)) <= ez) { // disc may cross the branch cut
    *v = make_cuDoubleComplex(NAN, NAN); *e = INFINITY; return;
  }
  // principal sqrt (any faithful value works; the bound is a-posteriori)
  double re = cuCreal(z), im = cuCimag(z);
  double sr, si;
  if (re >= 0.0) {
    sr = sqrt(0.5 * (Zm + re));
    si = (sr > 0.0) ? im / (2.0 * sr) : 0.0;
  } else {
    si = sqrt(0.5 * (Zm - re));
    if (im < 0.0) si = -si;
    sr = (si != 0.0) ? im / (2.0 * si) : 0.0;
  }
  cuDoubleComplex sc = make_cuDoubleComplex(sr, si);
  cuDoubleComplex sq;
  double e_sq;
  tor_p_mul(sc, 0.0, sc, 0.0, &sq, &e_sq);
  double delta = ru_add(ru_add(ru_mul(cuCabs(cuCsub(sq, z)), GBS_ABS_SAFE), e_sq),
                        ru_add(ru_mul(GBS_U, ru_add(cuCabs(sq), Zm)), ez));
  if (delta > 0.5 * Zm) { *v = sc; *e = INFINITY; return; }
  double t = ru_div(delta, sqrt(rd_mul(Zm, 1.0 - 4.0 * GBS_U)));
  tor_p_div(make_cuDoubleComplex(1.0, 0.0), 0.0, sc, t, v, e);
}

__global__ void tor_certified_kernel(const cuDoubleComplex* __restrict__ mats,
                                     int n, int batch,
                                     cuDoubleComplex* __restrict__ out,
                                     double* __restrict__ bound) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  const int dim = 2 * n;
  const cuDoubleComplex* O = mats + (size_t)b * dim * dim;

  if (n == 0) { out[b] = make_cuDoubleComplex(1.0, 0.0); bound[b] = 0.0; return; }
  if (dim > TOR_CERT_MAX_DIM) {
    out[b] = make_cuDoubleComplex(NAN, NAN); bound[b] = INFINITY; return;
  }

  cuDoubleComplex total = make_cuDoubleComplex(0.0, 0.0);
  double e_tot = 0.0;
  cuDoubleComplex M[TOR_CERT_MAX_DIM * TOR_CERT_MAX_DIM];
  double ME[TOR_CERT_MAX_DIM * TOR_CERT_MAX_DIM];
  int idx[TOR_CERT_MAX_DIM];

  for (uint64_t mask = 0; mask < (1ull << n); ++mask) {
    int m = __popcll((long long)mask);
    cuDoubleComplex term;
    double tb;
    if (m == 0) {
      term = make_cuDoubleComplex(1.0, 0.0);
      tb = 0.0;
    } else {
      int k = 2 * m, pc = 0;
      for (int i = 0; i < n; ++i)
        if ((mask >> i) & 1ull) idx[pc++] = i;
      for (int i = 0; i < n; ++i)
        if ((mask >> i) & 1ull) idx[pc++] = i + n;
      // I - O_S: off-diagonal negation exact; diagonal one fl subtract
      for (int r = 0; r < k; ++r)
        for (int c = 0; c < k; ++c) {
          cuDoubleComplex o = O[idx[r] * dim + idx[c]];
          if (r == c) {
            cuDoubleComplex vv = make_cuDoubleComplex(1.0 - cuCreal(o), -cuCimag(o));
            M[r * k + c] = vv;
            ME[r * k + c] = ru_mul(GBS_U, cert_absu(vv));
          } else {
            M[r * k + c] = make_cuDoubleComplex(-cuCreal(o), -cuCimag(o));
            ME[r * k + c] = 0.0;
          }
        }
      // LU with partial pivoting in pair arithmetic (pivot CHOICE needs no
      // certification; the arithmetic of the chosen sequence carries the bounds)
      cuDoubleComplex det = make_cuDoubleComplex(1.0, 0.0);
      double e_det = 0.0;
      for (int col = 0; col < k; ++col) {
        int piv = col;
        for (int r = col + 1; r < k; ++r)
          if (cuCabs(M[r * k + col]) > cuCabs(M[piv * k + col])) piv = r;
        if (piv != col) {
          for (int t = 0; t < k; ++t) {
            cuDoubleComplex tmp = M[col * k + t]; M[col * k + t] = M[piv * k + t]; M[piv * k + t] = tmp;
            double te = ME[col * k + t]; ME[col * k + t] = ME[piv * k + t]; ME[piv * k + t] = te;
          }
          det = make_cuDoubleComplex(-cuCreal(det), -cuCimag(det));
        }
        cuDoubleComplex pv = M[col * k + col];
        double pe = ME[col * k + col];
        cuDoubleComplex ndet;
        double nde;
        tor_p_mul(det, e_det, pv, pe, &ndet, &nde);
        det = ndet; e_det = nde;
        if (!isfinite(e_det)) break;
        if (cuCabs(pv) == 0.0) { det = make_cuDoubleComplex(0.0, 0.0); break; }
        for (int r = col + 1; r < k; ++r) {
          cuDoubleComplex f;
          double fe;
          tor_p_div(M[r * k + col], ME[r * k + col], pv, pe, &f, &fe);
          for (int c2 = col + 1; c2 < k; ++c2) {
            cuDoubleComplex mm;
            double me;
            tor_p_mul(f, fe, M[col * k + c2], ME[col * k + c2], &mm, &me);
            cuDoubleComplex v1 = cuCsub(M[r * k + c2], mm);
            M[r * k + c2] = v1;
            ME[r * k + c2] = ru_add(ME[r * k + c2],
                                    ru_add(me, ru_mul(GBS_U, cert_absu(v1))));
          }
        }
      }
      tor_p_inv_sqrt(det, e_det, &term, &tb);
    }
    if (((n - m) & 1) == 0) total = cuCadd(total, term);
    else total = cuCsub(total, term);
    e_tot = ru_add(e_tot, ru_add(tb, ru_mul(GBS_U, cert_absu(total))));
  }
  out[b] = total;
  bound[b] = e_tot;
}

extern "C" void gbs_tor_certified_batched(const cuDoubleComplex* d_mats, int n,
                                          int batch, cuDoubleComplex* d_out,
                                          double* d_bound, cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 64;
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(tor_certified_kernel, grid, block, stream, d_mats, n, batch, d_out, d_bound);
}

} // namespace gbs
