// tor_recursive.cu -- batched recursive prefix-Cholesky torontonian, physical
// (real O) domain.
//
// (check_tor_recursive.cu, bench_kernels.cu) in the next rented-GPU session.
// NOT dispatched into the public API until the device A/B says it wins
// (measured against the shipped complex-LU kernel before dispatch).
//
// The recursive algorithm of Kaposi, Kolarovszki et al. (arXiv:2109.04528),
// per thread: walk the mode-subset lattice as a DFS; *including* a mode appends
// two trailing rows to ONE Cholesky factor (O(size^2), prefix property);
// *backtracking* just forgets them (no downdate, no recompute); det(I - O_S)
// = prod diag(L)^2 rides along as a per-level running product. Per-subset
// amortized cost O(|S|^2) instead of the shipped kernel's O(|S|^3) fresh
// determinant -- AND the per-thread buffer is ONE real 24x24 factor (~4.6 KB),
// half the complex-LU kernel's footprint: this attacks both measured
// bottlenecks at once.
//
// Real-only (the torontonian's physical/validated domain, like candidate C's
// real-Cholesky kernel, which this composes with): a non-SPD leading minor
// (off-domain input) or non-real input surfaces as NaN so the host falls back
// to the complex-LU kernel -- never a silent wrong value.
//
// Reference: cpu_ref/tor_recursive.py (validated against cpu_ref.torontonian
// and, transitively, The Walrus + mpmath).

#include <cuComplex.h>
#include <math.h>

#include <cstdint>

#include "certified_rounding.cuh"
#include "dd.cuh"
#include "subset_engine.cuh"

namespace gbs {

constexpr int TORR_MAX_DIM = 24; // 2n, matches the shipped tor kernels
constexpr int TORR_MAX_MODES = TORR_MAX_DIM / 2;

// out[b] = tor(mats + b*(2n)^2) for REAL O (row-major 2n x 2n doubles).
__global__ void tor_recursive_real_kernel(const double* __restrict__ mats,
                                          int n, int batch,
                                          double* __restrict__ out) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  if (n == 0) { out[b] = 1.0; return; }
  if (2 * n > TORR_MAX_DIM) { out[b] = NAN; return; }
  const double* O = mats + (size_t)b * (2 * n) * (2 * n);
  const int dim = 2 * n;

  double L[TORR_MAX_DIM * TORR_MAX_DIM]; // the ONE factor buffer
  int modes[TORR_MAX_MODES]; // included modes, in order
  // DFS state per level: 0 = descend exclude next, 1 = descend include next,
  // 2 = both subtrees done (restore + ascend). detsave restores detprod when
  // an include-subtree unwinds.
  int state[TORR_MAX_MODES + 1];
  double detsave[TORR_MAX_MODES + 1];

  double total = 0.0;
  double detprod = 1.0;
  int count = 0;
  int lvl = 0;
  state[0] = 0;

  for (;;) {
    if (lvl == n) { // leaf: one subset S (the included modes)
      double c = 1.0 / sqrt(detprod); // (rsqrt is CUDA-only; the shim lacks it)
      total += ((n - count) & 1) ? -c : c;
      --lvl; // ascend; parent's state continues
      continue;
    }
    if (state[lvl] == 0) { // exclude `lvl`, descend
      state[lvl] = 1;
      state[lvl + 1] = 0;
      ++lvl;
      continue;
    }
    if (state[lvl] == 1) { // include `lvl`: append two rows
      state[lvl] = 2;
      detsave[lvl] = detprod;
      int j = lvl;
      int size = 2 * count;
      modes[count] = j;
      for (int t = 0; t < 2; ++t) { // t=0: x-row, t=1: p-row of mode j
        int r = size + t;
        int gj = (t == 0) ? j : j + n; // xxpp global row index
        for (int c = 0; c <= r; ++c) {
          int mc = ((c >> 1) == count) ? j : modes[c >> 1];
          int gc = (c & 1) ? mc + n : mc;
          double s = ((gj == gc) ? 1.0 : 0.0) - O[gj * dim + gc];
          for (int u = 0; u < c; ++u) s -= L[r * TORR_MAX_DIM + u] * L[c * TORR_MAX_DIM + u];
          if (c < r) {
            L[r * TORR_MAX_DIM + c] = s / L[c * TORR_MAX_DIM + c];
          } else {
            if (!(s > 0.0)) { out[b] = NAN; return; } // not SPD: off-domain
            L[r * TORR_MAX_DIM + r] = sqrt(s);
          }
        }
      }
      double dxx = L[size * TORR_MAX_DIM + size];
      double dpp = L[(size + 1) * TORR_MAX_DIM + size + 1];
      detprod *= (dxx * dpp) * (dxx * dpp);
      ++count;
      state[lvl + 1] = 0;
      ++lvl;
      continue;
    }
    // state == 2: include-subtree done -> restore and ascend
    detprod = detsave[lvl];
    --count;
    if (lvl == 0) break;
    --lvl;
  }
  out[b] = total;
}

extern "C" void gbs_tor_recursive_real_fp64_batched(const double* d_mats, int n,
                                                    int batch, double* d_out,
                                                    cudaStream_t stream) {
  if (batch <= 0) return;
  const int block = 64;
  const int grid = (batch + block - 1) / block;
  GBS_LAUNCH_1D(tor_recursive_real_kernel, grid, block, stream, d_mats, n, batch, d_out);
}


// --------------------------------------------------------------------------
// SINGLE-LARGE evaluation: one torontonian split across the whole GPU
// (single large evaluations past the batched-kernel dimension cap).
//
// Fix the include/exclude pattern of the first `g` modes: 2^g independent
// subtrees, one per thread. Each thread rebuilds its g-deep prefix factor
// (the appends for its included prefix modes -- O(g * dim^2), amortized away
// by the 2^(n-g)-leaf subtree it then walks) and runs the same append/pop DFS
// over the remaining modes, accumulating a partial signed sum. The host sums
// the 2^g partials. Any off-domain (non-SPD) minor writes NaN, which
// propagates through the host sum -- same never-silent contract.
//
// The factor cap is DIM 64 (n <= 32 modes): per-thread L is 64*64 doubles
// (32 KB local memory) -- heavy, but a single evaluation owns the whole GPU
// and the published validation ceiling this targets is 26 clicks (dim 52),
// with headroom to 32.
// --------------------------------------------------------------------------

constexpr int TORS_MAX_DIM = 64;
constexpr int TORS_MAX_MODES = TORS_MAX_DIM / 2;

__global__ void tor_recursive_single_kernel(const double* __restrict__ O,
                                            int n, int g,
                                            double* __restrict__ partials) {
  const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t nsub = 1ull << g;
  if (t >= nsub) return;
  const int dim = 2 * n;

  double L[TORS_MAX_DIM * TORS_MAX_DIM];
  int modes[TORS_MAX_MODES];
  int state[TORS_MAX_MODES + 1];
  double detsave[TORS_MAX_MODES + 1];

  double detprod = 1.0;
  int count = 0;

  // ---- prefix: append the included modes among the first g (bits of t) ----
  for (int j = 0; j < g; ++j) {
    if (!((t >> j) & 1ull)) continue;
    int size = 2 * count;
    modes[count] = j;
    for (int tt = 0; tt < 2; ++tt) {
      int r = size + tt;
      int gj = (tt == 0) ? j : j + n;
      for (int c = 0; c <= r; ++c) {
        int mc = ((c >> 1) == count) ? j : modes[c >> 1];
        int gc = (c & 1) ? mc + n : mc;
        double sacc = ((gj == gc) ? 1.0 : 0.0) - O[gj * dim + gc];
        for (int u = 0; u < c; ++u)
          sacc -= L[r * TORS_MAX_DIM + u] * L[c * TORS_MAX_DIM + u];
        if (c < r) {
          L[r * TORS_MAX_DIM + c] = sacc / L[c * TORS_MAX_DIM + c];
        } else {
          if (!(sacc > 0.0)) { partials[t] = NAN; return; }
          L[r * TORS_MAX_DIM + r] = sqrt(sacc);
        }
      }
    }
    double dxx = L[size * TORS_MAX_DIM + size];
    double dpp = L[(size + 1) * TORS_MAX_DIM + size + 1];
    detprod *= (dxx * dpp) * (dxx * dpp);
    ++count;
  }
  const int prefix_count = count;
  const double prefix_det = detprod;

  // ---- DFS over the remaining modes g..n-1 (same machine as the batched kernel)
  double total = 0.0;
  int lvl = 0;
  const int rem = n - g;
  state[0] = 0;
  for (;;) {
    if (lvl == rem) {
      double c = 1.0 / sqrt(detprod);
      total += ((n - count) & 1) ? -c : c;
      if (lvl == 0) break; // rem == 0: single leaf
      --lvl;
      continue;
    }
    if (state[lvl] == 0) {
      state[lvl] = 1;
      state[lvl + 1] = 0;
      ++lvl;
      continue;
    }
    if (state[lvl] == 1) {
      state[lvl] = 2;
      detsave[lvl] = detprod;
      int j = g + lvl;
      int size = 2 * count;
      modes[count] = j;
      for (int tt = 0; tt < 2; ++tt) {
        int r = size + tt;
        int gj = (tt == 0) ? j : j + n;
        for (int c = 0; c <= r; ++c) {
          int mc = ((c >> 1) == count) ? j : modes[c >> 1];
          int gc = (c & 1) ? mc + n : mc;
          double sacc = ((gj == gc) ? 1.0 : 0.0) - O[gj * dim + gc];
          for (int u = 0; u < c; ++u)
            sacc -= L[r * TORS_MAX_DIM + u] * L[c * TORS_MAX_DIM + u];
          if (c < r) {
            L[r * TORS_MAX_DIM + c] = sacc / L[c * TORS_MAX_DIM + c];
          } else {
            if (!(sacc > 0.0)) { partials[t] = NAN; return; }
            L[r * TORS_MAX_DIM + r] = sqrt(sacc);
          }
        }
      }
      double dxx = L[size * TORS_MAX_DIM + size];
      double dpp = L[(size + 1) * TORS_MAX_DIM + size + 1];
      detprod *= (dxx * dpp) * (dxx * dpp);
      ++count;
      state[lvl + 1] = 0;
      ++lvl;
      continue;
    }
    detprod = detsave[lvl];
    --count;
    if (lvl == 0) break;
    --lvl;
  }
  // silence unused warnings for degenerate g == n
  (void)prefix_count; (void)prefix_det;
  partials[t] = total;
}

extern "C" void gbs_tor_recursive_single_batched(const double* d_O, int n, int g,
                                                 double* d_partials,
                                                 cudaStream_t stream) {
  const uint64_t nsub = 1ull << g;
  const int block = 64;
  const uint64_t grid = (nsub + block - 1) / block;
  GBS_LAUNCH_1D(tor_recursive_single_kernel, (unsigned)grid, block, stream, d_O, n, g, d_partials);
}


// --------------------------------------------------------------------------
// CERTIFIED single-large evaluation (single large evaluation:
// certified torontonians AT and BEYOND the 26-click validation ceiling).
//
// The same 2^g-subtree split, with REAL pair-arithmetic bounds riding the
// prefix-Cholesky walk: EL mirrors L entrywise; e_det rides detprod with a
// save/restore stack; IEEE-correctly-rounded real sqrt gives the diagonal
// bound directly; each leaf's 1/sqrt(detprod +- e) uses the monotone real
// perturbation. Anything uncertifiable (non-SPD minor, bound overtaking a
// pivot) writes value NaN / bound +inf -- never a finite overclaim. Host sums
// value partials and upward-sums bound partials.
//
// Constants: standard model, u = 2^-53; any-order length-m sums via
// cert-gamma; the accumulator arithmetic is upward-rounded (certified.cu's
// helpers), so no global inflation. Memory: L + EL = 64 KB/thread at the
// dim-64 cap -- occupancy-poor but a single evaluation owns the GPU.
// --------------------------------------------------------------------------

constexpr double TORS_U = 1.1102230246251565e-16; // 2^-53

__device__ inline double tors_gamma(double kk) {
  double ku = kk * TORS_U;
  return ru_mul(ku / (1.0 - ku), 1.0 + 4.0 * TORS_U);
}

__global__ void tor_recursive_single_cert_kernel(const double* __restrict__ O,
                                                 int n, int g,
                                                 double* __restrict__ partials,
                                                 double* __restrict__ pbounds) {
  const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t nsub = 1ull << g;
  if (t >= nsub) return;
  const int dim = 2 * n;

  double L[TORS_MAX_DIM * TORS_MAX_DIM];
  double EL[TORS_MAX_DIM * TORS_MAX_DIM];
  int modes[TORS_MAX_MODES];
  int state[TORS_MAX_MODES + 1];
  double detsave[TORS_MAX_MODES + 1];
  double edetsave[TORS_MAX_MODES + 1];

  double detprod = 1.0, e_det = 0.0;
  int count = 0;

  // append mode j's two rows with bounds; returns false on breakdown/uncertifiable
  auto append = [&](int j) -> bool {
    int size = 2 * count;
    modes[count] = j;
    for (int tt = 0; tt < 2; ++tt) {
      int r = size + tt;
      int gj = (tt == 0) ? j : j + n;
      for (int c = 0; c <= r; ++c) {
        int mc = ((c >> 1) == count) ? j : modes[c >> 1];
        int gc = (c & 1) ? mc + n : mc;
        double m = ((gj == gc) ? 1.0 : 0.0) - O[gj * dim + gc];
        double e_m = (gj == gc) ? TORS_U * fabs(m) : 0.0; // one fl subtract on the diagonal
        double sacc = m;
        double sb = 0.0, sm = fabs(m);
        for (int u = 0; u < c; ++u) {
          double lr = L[r * TORS_MAX_DIM + u], lc = L[c * TORS_MAX_DIM + u];
          sacc -= lr * lc;
          sb = ru_add(sb, ru_add(ru_mul(EL[r * TORS_MAX_DIM + u], fabs(lc)),
                                 ru_add(ru_mul(fabs(lr), EL[c * TORS_MAX_DIM + u]),
                                        ru_mul(EL[r * TORS_MAX_DIM + u], EL[c * TORS_MAX_DIM + u]))));
          sm = ru_add(sm, ru_mul(fabs(lr), fabs(lc)));
        }
        double e_s = ru_add(ru_add(sb, e_m), ru_mul(tors_gamma((double)(c + 2)), sm));
        if (c < r) {
          double p = L[c * TORS_MAX_DIM + c], ep = EL[c * TORS_MAX_DIM + c];
          if (!(ep < 0.5 * p)) return false; // pivot uncertifiable
          double v = sacc / p;
          double p_lo = rd_mul(p - ep, 1.0 - 2.0 * TORS_U);
          L[r * TORS_MAX_DIM + c] = v;
          EL[r * TORS_MAX_DIM + c] =
              ru_add(ru_add(ru_div(e_s, p_lo), ru_div(ru_mul(ep, fabs(v)), p_lo)),
                     ru_mul(TORS_U, fabs(v)));
        } else {
          if (!(sacc > 0.0) || !(e_s < 0.5 * sacc)) return false; // not (certifiably) SPD
          double v = sqrt(sacc);
          L[r * TORS_MAX_DIM + r] = v;
          // |sqrt(s±e) - sqrt(s)| <= e / (2 sqrt(s-e)); fl(sqrt) adds u*v
          EL[r * TORS_MAX_DIM + r] =
              ru_add(ru_mul(TORS_U, v), ru_div(e_s, 2.0 * rd_sqrt(sacc - e_s)));
        }
      }
    }
    // detprod *= (dxx*dpp)^2 with real pair arithmetic
    double a = L[size * TORS_MAX_DIM + size], ea = EL[size * TORS_MAX_DIM + size];
    double b = L[(size + 1) * TORS_MAX_DIM + size + 1], eb = EL[(size + 1) * TORS_MAX_DIM + size + 1];
    double ab = a * b;
    double e_ab = ru_add(ru_add(ru_mul(fabs(a), eb), ru_mul(fabs(b), ea)),
                         ru_add(ru_mul(ea, eb), ru_mul(TORS_U, fabs(ab))));
    double d2 = ab * ab;
    double e_d2 = ru_add(ru_mul(2.0 * fabs(ab), e_ab),
                         ru_add(ru_mul(e_ab, e_ab), ru_mul(TORS_U, d2)));
    double nd = detprod * d2;
    e_det = ru_add(ru_add(ru_mul(detprod, e_d2), ru_mul(d2, e_det)),
                   ru_add(ru_mul(e_det, e_d2), ru_mul(TORS_U, fabs(nd))));
    detprod = nd;
    ++count;
    return true;
  };

  // ---- prefix ----
  for (int j = 0; j < g; ++j) {
    if (!((t >> j) & 1ull)) continue;
    if (!append(j)) { partials[t] = NAN; pbounds[t] = INFINITY; return; }
  }

  // ---- DFS over remaining modes ----
  double total = 0.0, e_tot = 0.0;
  int lvl = 0;
  const int rem = n - g;
  state[0] = 0;
  for (;;) {
    if (lvl == rem) {
      // leaf: term = 1/sqrt(detprod ± e_det)
      if (!(e_det < 0.5 * detprod)) { partials[t] = NAN; pbounds[t] = INFINITY; return; }
      double sroot = sqrt(detprod);
      double tS = ru_add(ru_mul(TORS_U, sroot),
                         ru_div(e_det, 2.0 * rd_sqrt(detprod - e_det)));
      double c = 1.0 / sroot;
      double s_lo = rd_mul(sroot - tS, 1.0 - 2.0 * TORS_U);
      // |1/(s±t) - fl(1/s)| <= t/(s*(s-t)) + u/s, all upward-rounded
      double e_c = ru_add(ru_div(tS, rd_mul(sroot, s_lo)), ru_mul(TORS_U, c));
      total += ((n - count) & 1) ? -c : c;
      e_tot = ru_add(e_tot, ru_add(e_c, ru_mul(TORS_U, fabs(total))));
      if (lvl == 0) break;
      --lvl;
      continue;
    }
    if (state[lvl] == 0) { state[lvl] = 1; state[lvl + 1] = 0; ++lvl; continue; }
    if (state[lvl] == 1) {
      state[lvl] = 2;
      detsave[lvl] = detprod;
      edetsave[lvl] = e_det;
      if (!append(g + lvl)) { partials[t] = NAN; pbounds[t] = INFINITY; return; }
      state[lvl + 1] = 0;
      ++lvl;
      continue;
    }
    detprod = detsave[lvl];
    e_det = edetsave[lvl];
    --count;
    if (lvl == 0) break;
    --lvl;
  }
  partials[t] = total;
  pbounds[t] = e_tot;
}

extern "C" void gbs_tor_recursive_single_cert_batched(const double* d_O, int n, int g,
                                                      double* d_partials,
                                                      double* d_pbounds,
                                                      cudaStream_t stream) {
  const uint64_t nsub = 1ull << g;
  const int block = 64;
  const uint64_t grid = (nsub + block - 1) / block;
  GBS_LAUNCH_1D(tor_recursive_single_cert_kernel, (unsigned)grid, block, stream,
                d_O, n, g, d_partials, d_pbounds);
}


// --------------------------------------------------------------------------
// CERTIFIED DOUBLE-DOUBLE single-large evaluation.
//
// Identical machine to tor_recursive_single_cert_kernel (Theorem 3'), but the
// VALUE path is carried in double-double (dd.cuh error-free transforms) while
// the bound path is the same real pair-arithmetic recurrence with the unit
// roundoff replaced by the double-word constant u_DD. Every dd operation obeys
// the same relative-error model fl(a.b)=(a.b)(1+d), |d|<=u_DD that the fp64
// bounds were derived from, so the derivation transfers verbatim with u->u_DD;
// magnitudes are read from the hi component with a non-overlap headroom. u_DD =
// 2^-100 is >~30x the proven double-word add/mul/div/sqrt constants
// (Joldes-Muller-Popescu, ACM TOMS 2017), so the charge is conservative and the
// enclosure is enforced as a test invariant against mpmath. Result value/bound
// collapse to fp64 on output (one final u*|.| charge folded into the leaf term).
//
// This is what makes the Jiuzhang frontier (past the fp64 wall near 15 clicks)
// return a TIGHT certified probability rather than only a proof that fp64 failed.
// Memory: dd L (64 KB) + fp64 EL (32 KB) per thread; a single evaluation owns
// the GPU. Off-domain / uncertifiable minors write NaN / +inf as before.
// --------------------------------------------------------------------------

constexpr double TORS_U_DD = 7.888609052210118e-31; // 2^-100 (charged), >~30x proven
constexpr double TORS_HDR = 1.0000000000000018;      // 1 + 2^-49, |dd| <= |hi|*HDR

__device__ inline double tors_gamma_dd(double kk) {
  double ku = kk * TORS_U_DD;
  return ru_mul(ku / (1.0 - ku), 1.0 + 4.0 * TORS_U_DD);
}
// upper / lower fp64 bounds on |x| for a double-double x
__device__ inline double md_hi(dd x) { return ru_mul(fabs(x.hi), TORS_HDR); }
__device__ inline double md_lo(dd x) { return rd_mul(fabs(x.hi), 1.0 - 2.0 * TORS_U_DD); }

__global__ void tor_recursive_single_ddcert_kernel(const double* __restrict__ O,
                                                   int n, int g,
                                                   double* __restrict__ partials,
                                                   double* __restrict__ pbounds) {
  const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t nsub = 1ull << g;
  if (t >= nsub) return;
  const int dim = 2 * n;

  dd L[TORS_MAX_DIM * TORS_MAX_DIM];   // double-double value factor
  double EL[TORS_MAX_DIM * TORS_MAX_DIM]; // fp64 entrywise error bound
  int modes[TORS_MAX_MODES];
  int state[TORS_MAX_MODES + 1];
  dd detsave[TORS_MAX_MODES + 1];
  double edetsave[TORS_MAX_MODES + 1];

  dd detprod = dd_from(1.0);
  double e_det = 0.0;
  int count = 0;

  auto append = [&](int j) -> bool {
    int size = 2 * count;
    modes[count] = j;
    for (int tt = 0; tt < 2; ++tt) {
      int r = size + tt;
      int gj = (tt == 0) ? j : j + n;
      for (int c = 0; c <= r; ++c) {
        int mc = ((c >> 1) == count) ? j : modes[c >> 1];
        int gc = (c & 1) ? mc + n : mc;
        double ind = (gj == gc) ? 1.0 : 0.0;
        double Ov = O[gj * dim + gc];
        dd sacc = dd_add(dd_from(ind), dd_from(-Ov)); // exact (1 or 0) - O
        double e_m = ru_mul(TORS_U_DD, md_hi(sacc));  // (charge; the fl subtract is exact in dd)
        double sb = 0.0, sm = md_hi(sacc);
        for (int u = 0; u < c; ++u) {
          dd lr = L[r * TORS_MAX_DIM + u], lc = L[c * TORS_MAX_DIM + u];
          sacc = dd_add(sacc, dd_neg(dd_mul(lr, lc)));
          double alr = md_hi(lr), alc = md_hi(lc);
          double elr = EL[r * TORS_MAX_DIM + u], elc = EL[c * TORS_MAX_DIM + u];
          sb = ru_add(sb, ru_add(ru_mul(elr, alc),
                                 ru_add(ru_mul(alr, elc), ru_mul(elr, elc))));
          sm = ru_add(sm, ru_mul(alr, alc));
        }
        double e_s = ru_add(ru_add(sb, e_m), ru_mul(tors_gamma_dd((double)(c + 2)), sm));
        if (c < r) {
          dd pv = L[c * TORS_MAX_DIM + c];
          double p = md_hi(pv), ep = EL[c * TORS_MAX_DIM + c];
          if (!(ep < 0.5 * md_lo(pv))) return false;   // pivot uncertifiable
          dd v = dd_div(sacc, pv);
          double av = md_hi(v);
          double p_lo = rd_mul(md_lo(pv) - ep, 1.0 - 2.0 * TORS_U_DD);
          L[r * TORS_MAX_DIM + c] = v;
          EL[r * TORS_MAX_DIM + c] =
              ru_add(ru_add(ru_div(e_s, p_lo), ru_div(ru_mul(ep, av), p_lo)),
                     ru_mul(TORS_U_DD, av));
        } else {
          if (!(sacc.hi > 0.0) || !(e_s < 0.5 * md_lo(sacc))) return false; // not (cert.) SPD
          dd v = dd_sqrt(sacc);
          double av = md_hi(v);
          L[r * TORS_MAX_DIM + r] = v;
          EL[r * TORS_MAX_DIM + r] =
              ru_add(ru_mul(TORS_U_DD, av), ru_div(e_s, 2.0 * rd_sqrt(md_lo(sacc) - e_s)));
        }
      }
    }
    dd a = L[size * TORS_MAX_DIM + size], b = L[(size + 1) * TORS_MAX_DIM + size + 1];
    double ea = EL[size * TORS_MAX_DIM + size], eb = EL[(size + 1) * TORS_MAX_DIM + size + 1];
    double aa = md_hi(a), ab_mag = md_hi(b);
    dd ab = dd_mul(a, b);
    double abm = md_hi(ab);
    double e_ab = ru_add(ru_add(ru_mul(aa, eb), ru_mul(ab_mag, ea)),
                         ru_add(ru_mul(ea, eb), ru_mul(TORS_U_DD, abm)));
    dd d2 = dd_mul(ab, ab);
    double d2m = md_hi(d2);
    double e_d2 = ru_add(ru_mul(2.0 * abm, e_ab),
                         ru_add(ru_mul(e_ab, e_ab), ru_mul(TORS_U_DD, d2m)));
    dd nd = dd_mul(detprod, d2);
    double dpm = md_hi(detprod), ndm = md_hi(nd);
    e_det = ru_add(ru_add(ru_mul(dpm, e_d2), ru_mul(d2m, e_det)),
                   ru_add(ru_mul(e_det, e_d2), ru_mul(TORS_U_DD, ndm)));
    detprod = nd;
    ++count;
    return true;
  };

  for (int j = 0; j < g; ++j) {
    if (!((t >> j) & 1ull)) continue;
    if (!append(j)) { partials[t] = NAN; pbounds[t] = INFINITY; return; }
  }

  dd total = dd_from(0.0);
  double e_tot = 0.0;
  int lvl = 0;
  const int rem = n - g;
  state[0] = 0;
  for (;;) {
    if (lvl == rem) {
      if (!(e_det < 0.5 * md_lo(detprod))) { partials[t] = NAN; pbounds[t] = INFINITY; return; }
      dd sroot = dd_sqrt(detprod);
      double srm = md_hi(sroot), srlo = md_lo(sroot);
      double tS = ru_add(ru_mul(TORS_U_DD, srm),
                         ru_div(e_det, 2.0 * rd_sqrt(md_lo(detprod) - e_det)));
      dd c = dd_div(dd_from(1.0), sroot);
      double cm = md_hi(c);
      double s_lo = rd_mul(srlo - tS, 1.0 - 2.0 * TORS_U_DD);
      double e_c = ru_add(ru_div(tS, rd_mul(srlo, s_lo)), ru_mul(TORS_U_DD, cm));
      total = dd_add(total, ((n - count) & 1) ? dd_neg(c) : c);
      e_tot = ru_add(e_tot, ru_add(e_c, ru_mul(TORS_U_DD, md_hi(total))));
      if (lvl == 0) break;
      --lvl;
      continue;
    }
    if (state[lvl] == 0) { state[lvl] = 1; state[lvl + 1] = 0; ++lvl; continue; }
    if (state[lvl] == 1) {
      state[lvl] = 2;
      detsave[lvl] = detprod;
      edetsave[lvl] = e_det;
      if (!append(g + lvl)) { partials[t] = NAN; pbounds[t] = INFINITY; return; }
      state[lvl + 1] = 0;
      ++lvl;
      continue;
    }
    detprod = detsave[lvl];
    e_det = edetsave[lvl];
    --count;
    if (lvl == 0) break;
    --lvl;
  }
  // collapse dd value to fp64 output with one final rounding charge
  double val = total.hi + total.lo;
  partials[t] = val;
  pbounds[t] = ru_add(e_tot, ru_mul(TORS_U_DD, fabs(val)));
}

extern "C" void gbs_tor_recursive_single_ddcert_batched(const double* d_O, int n, int g,
                                                        double* d_partials,
                                                        double* d_pbounds,
                                                        cudaStream_t stream) {
  const uint64_t nsub = 1ull << g;
  const int block = 64;
  const uint64_t grid = (nsub + block - 1) / block;
  GBS_LAUNCH_1D(tor_recursive_single_ddcert_kernel, (unsigned)grid, block, stream,
                d_O, n, g, d_partials, d_pbounds);
}

} // namespace gbs
