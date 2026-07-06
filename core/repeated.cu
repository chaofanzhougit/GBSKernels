// repeated.cu -- batched repeated-row loop hafnian: the finite-difference sieve
// (the sampling workload's own algorithm).
//
// in the next rented-GPU session, like every kernel before it.
//
// One (pattern, base-matrix) evaluation per thread. All threads in a launch
// share ONE base matrix A (M x M) and loop vector gamma (M) -- the GBS chain's
// shape: a fixed reduced A-matrix, many candidate patterns -- staged into
// shared memory once per block; each thread owns its reps row. Per-thread
// state is a length-M int vector + scalars (sub-KB!), so unlike the expanded
// power-trace kernels this one has NO footprint problem; its cost driver is
// the sieve term count prod(n_i + 1) (vs 2^(N/2) for the expansion).
//
// Mathematics (identical to cpu_ref/repeated.py, where it is derived and
// pinned against the expanded reference):
// lhaf = (1/(N! 2^N)) sum_{v <= n} (-1)^{|v|} prod C(n_i, v_i) * S_N(x_v),
// x_v = n - 2v, S_N(x) = sum_k C(N,2k)(2k-1)!! (x^T A x)^k (x . gamma)^{N-2k}.

#include <cuComplex.h>
#include <math.h>

#include <cstdint>

#include "certified_rounding.cuh"
#include "subset_engine.cuh"

namespace gbs {

constexpr int REP_MAX_M = 32; // base-matrix side; 32 covers 16-mode PNR
                                   // (A is 2M=32, reps doubled) on-GPU. Per-thread
                                   // Aabs[32*32]=8KB stack in the certified kernel
                                   // -- within the tor kernels' 9-27KB envelope.
constexpr int REP_MAX_PHOT = 64; // N = sum(reps): factorials/powers stay finite
// prod(n_i+1) guard: a per-thread sieve beyond this is a workload bug, not a batch.
constexpr double REP_MAX_TERMS = 1.099511627776e12; // 2^40

__device__ inline cuDoubleComplex rep_cmul(cuDoubleComplex a, cuDoubleComplex b) {
  return cuCmul(a, b);
}
__device__ inline cuDoubleComplex rep_cscale(cuDoubleComplex a, double s) {
  return make_cuDoubleComplex(cuCreal(a) * s, cuCimag(a) * s);
}

__global__ void lhaf_repeated_kernel(const cuDoubleComplex* __restrict__ d_A,
                                     const cuDoubleComplex* __restrict__ d_gamma,
                                     int M, const int* __restrict__ d_reps,
                                     int batch, cuDoubleComplex* __restrict__ out) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  if (M > REP_MAX_M) { out[b] = make_cuDoubleComplex(NAN, NAN); return; }
  // A and gamma are shared by the whole launch and read straight from global
  // memory (every thread reads the same entries -> L1/L2-cached broadcasts; a
  // __shared__ staging is a later tuning the host shim cannot emulate).
  const cuDoubleComplex* sA = d_A;
  const cuDoubleComplex* sg = d_gamma;

  const int* n = d_reps + (size_t)b * M;
  int N = 0;
  double terms = 1.0;
  for (int i = 0; i < M; ++i) { N += n[i]; terms *= (double)(n[i] + 1); }
  if (N == 0) { out[b] = make_cuDoubleComplex(1.0, 0.0); return; }
  if (N > REP_MAX_PHOT || terms > REP_MAX_TERMS) {
    out[b] = make_cuDoubleComplex(NAN, NAN); return;
  }
  const int kmax = N / 2;

  // C(N,2k)(2k-1)!! by the exact ratio recurrence c_{k+1} = c_k (N-2k)(N-2k-1)/(2k+2).
  double coeff[REP_MAX_PHOT / 2 + 1];
  coeff[0] = 1.0;
  for (int k = 0; k < kmax; ++k)
    coeff[k + 1] = coeff[k] * (double)(N - 2 * k) * (double)(N - 2 * k - 1) / (double)(2 * k + 2);

  int v[REP_MAX_M];
  double x[REP_MAX_M];
  for (int i = 0; i < M; ++i) { v[i] = 0; x[i] = (double)n[i]; }

  cuDoubleComplex total = make_cuDoubleComplex(0.0, 0.0);
  for (;;) {
    // v <-> n-v symmetry (equal contributions): keep v lexicographically
    // <= n-v, double the strict half, count the self-paired v = n/2 once.
    int cmp = 0; // -1: v < n-v, 0: equal, +1: v > n-v
    for (int i = 0; i < M; ++i) {
      int d = v[i] - (n[i] - v[i]);
      if (d != 0) { cmp = d < 0 ? -1 : 1; break; }
    }
    if (cmp > 0) goto advance;
    {
    // sign * prod C(n_i, v_i) for the current v (recomputed; O(M) per term).
    int vsum = 0;
    double c = (cmp < 0) ? 2.0 : 1.0;
    for (int i = 0; i < M; ++i) {
      vsum += v[i];
      // C(n_i, v_i) multiplicatively (exact in double for n_i <= 64)
      double bin = 1.0;
      for (int t = 1; t <= v[i]; ++t) bin = bin * (double)(n[i] - t + 1) / (double)t;
      c *= bin;
    }
    if (vsum & 1) c = -c;

    // sigma^2 = x^T A x (A symmetric not assumed here; full double loop),
    // mu = x . gamma
    cuDoubleComplex sig2 = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < M; ++i) {
      if (x[i] == 0.0) continue;
      cuDoubleComplex row = make_cuDoubleComplex(0.0, 0.0);
      for (int j = 0; j < M; ++j)
        if (x[j] != 0.0) row = cuCadd(row, rep_cscale(sA[i * M + j], x[j]));
      sig2 = cuCadd(sig2, rep_cscale(row, x[i]));
    }
    cuDoubleComplex mu = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < M; ++i)
      if (x[i] != 0.0) mu = cuCadd(mu, rep_cscale(sg[i], x[i]));

    // S_N = sum_k coeff[k] sig2^k mu^(N-2k): mu-powers descending from k=kmax.
    cuDoubleComplex mu2 = rep_cmul(mu, mu);
    cuDoubleComplex mupow[REP_MAX_PHOT / 2 + 1];
    mupow[kmax] = (N & 1) ? mu : make_cuDoubleComplex(1.0, 0.0);
    for (int k = kmax - 1; k >= 0; --k) mupow[k] = rep_cmul(mupow[k + 1], mu2);
    cuDoubleComplex SN = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex sk = make_cuDoubleComplex(1.0, 0.0);
    for (int k = 0; k <= kmax; ++k) {
      SN = cuCadd(SN, rep_cscale(rep_cmul(sk, mupow[k]), coeff[k]));
      if (k < kmax) sk = rep_cmul(sk, sig2);
    }
    total = cuCadd(total, rep_cscale(SN, c));
    }

  advance:
    // odometer over v <= n
    {
      int i = 0;
      while (i < M && v[i] == n[i]) {
        v[i] = 0; x[i] = (double)n[i]; ++i;
      }
      if (i == M) break;
      v[i] += 1; x[i] = (double)(n[i] - 2 * v[i]);
    }
  }

  // 1 / (N! 2^N)
  double scale = 1.0;
  for (int t = 2; t <= N; ++t) scale *= (double)t;
  scale *= exp2((double)N);
  out[b] = rep_cscale(total, 1.0 / scale);
}

extern "C" void gbs_lhaf_repeated_batched(const cuDoubleComplex* d_A,
                                          const cuDoubleComplex* d_gamma, int M,
                                          const int* d_reps, int batch,
                                          cuDoubleComplex* d_out, cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 128; // tiny per-thread state; no footprint pressure
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(lhaf_repeated_kernel, grid, block, stream, d_A, d_gamma, M, d_reps,
                batch, d_out);
}


// ---------------------------------------------------------------------------
// CERTIFIED sieve: the identical value walk with a running error
// bound in upward-rounded arithmetic (core/certified_rounding.cuh; standard
// model, u = 2^-53, complex multiply <= 3u, any-order dots via gamma).
// Integer data (x, the C(n_i, v_i) prefactors while < 2^53) is exact; the
// coefficient ratio recurrence, N!, and the prefactor loops charge their
// rounding explicitly. Values are bit-identical to lhaf_repeated_kernel
// (asserted by the gate); over-cap emits value NaN + bound INF.
// ---------------------------------------------------------------------------

constexpr double REPC_U = 1.1102230246251565e-16; // 2^-53
constexpr double REPC_ABS_SAFE = 1.0 + 8.881784197001252e-16;
constexpr double REPC_UFL = 8e-323;

__device__ inline double repc_gamma(double k) {
  double ku = k * REPC_U;
  return ru_mul(ku / (1.0 - ku), 1.0 + 4.0 * REPC_U);
}
__device__ inline double repc_absu(cuDoubleComplex z) {
  return hypot(cuCreal(z), cuCimag(z)) * REPC_ABS_SAFE;
}
// pair product bound (Lemma B1): |fl(ab) - a*b*| <= A eb + B ea + ea eb + 3u A B
__device__ inline double repc_pmul_b(double A, double ea, double B, double eb) {
  return ru_add(ru_add(ru_mul(A, eb), ru_mul(B, ea)),
                ru_add(ru_mul(ea, eb), ru_add(ru_mul(3.0 * REPC_U, ru_mul(A, B)), REPC_UFL)));
}

__global__ void lhaf_repeated_cert_kernel(const cuDoubleComplex* __restrict__ d_A,
                                          const cuDoubleComplex* __restrict__ d_gamma,
                                          int M, const int* __restrict__ d_reps,
                                          int batch, cuDoubleComplex* __restrict__ out,
                                          double* __restrict__ bounds) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  if (M > REP_MAX_M) { out[b] = make_cuDoubleComplex(NAN, NAN); bounds[b] = INFINITY; return; }
  const cuDoubleComplex* sA = d_A;
  const cuDoubleComplex* sg = d_gamma;

  const int* n = d_reps + (size_t)b * M;
  int N = 0;
  double terms = 1.0;
  for (int i = 0; i < M; ++i) { N += n[i]; terms *= (double)(n[i] + 1); }
  if (N == 0) { out[b] = make_cuDoubleComplex(1.0, 0.0); bounds[b] = 0.0; return; }
  if (N > REP_MAX_PHOT || terms > REP_MAX_TERMS) {
    out[b] = make_cuDoubleComplex(NAN, NAN); bounds[b] = INFINITY; return;
  }
  const int kmax = N / 2;

  // hoisted magnitudes (pure CSE -- the measured lesson from certified.cu)
  double Aabs[REP_MAX_M * REP_MAX_M];
  double gabs[REP_MAX_M];
  for (int i = 0; i < M * M; ++i) Aabs[i] = repc_absu(sA[i]);
  for (int i = 0; i < M; ++i) gabs[i] = repc_absu(sg[i]);
  const double c_dot = repc_gamma(3.0 * (double)(M * M + M) + 6.0);
  const double c_dotv = repc_gamma(3.0 * (double)M + 6.0);

  // coeff ratio recurrence. The VALUE line is byte-identical to the plain
  // kernel's (same operands, same left-to-right associativity ((c*a)*b)/c) so
  // the certified value is bit-for-bit the plain sieve's at ALL N -- not just
  // where a re-associated form (r=a*b/c; coeff*=r) happens to agree (it
  // diverges by N~16). The bound is computed separately: true c_{k+1} =
  // c_k*(a*b/c) exactly (a,b,c exact ints), so err = r_abs*ecoeff[k] plus the
  // 3 value-roundings, charged at gamma(4) (>= gamma(3)/(1-gamma(3))).
  double coeff[REP_MAX_PHOT / 2 + 1];
  double ecoeff[REP_MAX_PHOT / 2 + 1];
  coeff[0] = 1.0; ecoeff[0] = 0.0;
  for (int k = 0; k < kmax; ++k) {
    coeff[k + 1] = coeff[k] * (double)(N - 2 * k) * (double)(N - 2 * k - 1) / (double)(2 * k + 2);
    double r_abs = ru_div(ru_mul((double)(N - 2 * k), (double)(N - 2 * k - 1)),
                          (double)(2 * k + 2));
    ecoeff[k + 1] = ru_add(ru_mul(r_abs, ecoeff[k]),
                           ru_mul(repc_gamma(4.0), fabs(coeff[k + 1])));
  }

  int v[REP_MAX_M];
  double x[REP_MAX_M];
  for (int i = 0; i < M; ++i) { v[i] = 0; x[i] = (double)n[i]; }

  cuDoubleComplex total = make_cuDoubleComplex(0.0, 0.0);
  double e_tot = 0.0;
  for (;;) {
    int cmp = 0;
    for (int i = 0; i < M; ++i) {
      int d = v[i] - (n[i] - v[i]);
      if (d != 0) { cmp = d < 0 ? -1 : 1; break; }
    }
    if (cmp > 0) goto advance;
    {
    int vsum = 0;
    double c = (cmp < 0) ? 2.0 : 1.0;
    double ec_rel = 0.0; // relative bound on the prefactor loops
    for (int i = 0; i < M; ++i) {
      vsum += v[i];
      double bin = 1.0;
      for (int t = 1; t <= v[i]; ++t) bin = bin * (double)(n[i] - t + 1) / (double)t;
      c *= bin;
      ec_rel = ru_add(ec_rel, repc_gamma(3.0 * (double)v[i] + 1.0));
    }
    if (vsum & 1) c = -c;
    double e_c = ru_mul(ec_rel, fabs(c));

    cuDoubleComplex sig2 = make_cuDoubleComplex(0.0, 0.0);
    double m_sig = 0.0; // sum |x_i||x_j||A_ij| (rounding mass)
    for (int i = 0; i < M; ++i) {
      if (x[i] == 0.0) continue;
      cuDoubleComplex row = make_cuDoubleComplex(0.0, 0.0);
      double mrow = 0.0;
      for (int j = 0; j < M; ++j)
        if (x[j] != 0.0) {
          row = cuCadd(row, rep_cscale(sA[i * M + j], x[j]));
          mrow = ru_add(mrow, ru_mul(Aabs[i * M + j], fabs(x[j])));
        }
      sig2 = cuCadd(sig2, rep_cscale(row, x[i]));
      m_sig = ru_add(m_sig, ru_mul(mrow, fabs(x[i])));
    }
    double e_sig = ru_add(ru_mul(c_dot, m_sig), (double)(M * M) * REPC_UFL);
    cuDoubleComplex mu = make_cuDoubleComplex(0.0, 0.0);
    double m_mu = 0.0;
    for (int i = 0; i < M; ++i)
      if (x[i] != 0.0) {
        mu = cuCadd(mu, rep_cscale(sg[i], x[i]));
        m_mu = ru_add(m_mu, ru_mul(gabs[i], fabs(x[i])));
      }
    double e_mu = ru_add(ru_mul(c_dotv, m_mu), (double)M * REPC_UFL);

    cuDoubleComplex mu2 = rep_cmul(mu, mu);
    double a_mu = repc_absu(mu);
    double e_mu2 = repc_pmul_b(a_mu, e_mu, a_mu, e_mu);
    cuDoubleComplex mupow[REP_MAX_PHOT / 2 + 1];
    double e_mupow[REP_MAX_PHOT / 2 + 1];
    mupow[kmax] = (N & 1) ? mu : make_cuDoubleComplex(1.0, 0.0);
    e_mupow[kmax] = (N & 1) ? e_mu : 0.0;
    double a_mu2 = repc_absu(mu2);
    for (int k = kmax - 1; k >= 0; --k) {
      mupow[k] = rep_cmul(mupow[k + 1], mu2);
      e_mupow[k] = repc_pmul_b(repc_absu(mupow[k + 1]), e_mupow[k + 1], a_mu2, e_mu2);
    }
    cuDoubleComplex SN = make_cuDoubleComplex(0.0, 0.0);
    double e_SN = 0.0;
    cuDoubleComplex sk = make_cuDoubleComplex(1.0, 0.0);
    double e_sk = 0.0;
    double a_sig = repc_absu(sig2);
    for (int k = 0; k <= kmax; ++k) {
      cuDoubleComplex t1 = rep_cmul(sk, mupow[k]);
      double e1 = repc_pmul_b(repc_absu(sk), e_sk, repc_absu(mupow[k]), e_mupow[k]);
      SN = cuCadd(SN, rep_cscale(t1, coeff[k]));
      double e2 = ru_add(ru_mul(fabs(coeff[k]), e1),
                         ru_add(ru_mul(ecoeff[k], repc_absu(t1)),
                                ru_mul(REPC_U, ru_mul(fabs(coeff[k]), repc_absu(t1)))));
      e_SN = ru_add(e_SN, ru_add(e2, ru_mul(REPC_U, repc_absu(SN))));
      if (k < kmax) {
        double a_sk = repc_absu(sk);
        sk = rep_cmul(sk, sig2);
        e_sk = repc_pmul_b(a_sk, e_sk, a_sig, e_sig);
      }
    }
    total = cuCadd(total, rep_cscale(SN, c));
    e_tot = ru_add(e_tot,
                   ru_add(ru_add(ru_mul(fabs(c), e_SN), ru_mul(e_c, repc_absu(SN))),
                          ru_add(ru_mul(REPC_U, ru_mul(fabs(c), repc_absu(SN))),
                                 ru_mul(REPC_U, repc_absu(total)))));
    }

  advance:
    {
      int i = 0;
      while (i < M && v[i] == n[i]) {
        v[i] = 0; x[i] = (double)n[i]; ++i;
      }
      if (i == M) break;
      v[i] += 1; x[i] = (double)(n[i] - 2 * v[i]);
    }
  }

  double scale = 1.0;
  for (int t = 2; t <= N; ++t) scale *= (double)t; // rel err <= gamma(N)
  scale *= exp2((double)N); // exact
  cuDoubleComplex o = rep_cscale(total, 1.0 / scale);
  out[b] = o;
  bounds[b] = ru_add(ru_div(e_tot, scale * (1.0 - (double)N * 2.0 * REPC_U)),
                     ru_mul(repc_absu(o), ru_add(repc_gamma((double)N), 2.0 * REPC_U)));
}

extern "C" void gbs_lhaf_repeated_cert_batched(const cuDoubleComplex* d_A,
                                               const cuDoubleComplex* d_gamma, int M,
                                               const int* d_reps, int batch,
                                               cuDoubleComplex* d_out, double* d_bounds,
                                               cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 64; // heavier per-thread state than the plain sieve
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(lhaf_repeated_cert_kernel, grid, block, stream, d_A, d_gamma, M, d_reps,
                batch, d_out, d_bounds);
}

} // namespace gbs
