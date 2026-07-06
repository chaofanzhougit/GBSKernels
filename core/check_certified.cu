// check_certified.cu -- gate for the CERTIFIED kernels.
//
// Three checks per function (permanent, hafnian), over random complex inputs:
// (1) value consistency: the certified kernel's value == the plain fp64
// kernel's value to <= 1e-13 rel (same expression order; contraction may
// differ per kernel, so a tolerance rather than bitwise);
// (2) ENCLOSURE (the certificate itself): |value - ref_ld| <= bound + ld slack,
// where ref_ld is an independent long-double host evaluation (~1e-19);
// one violation fails the gate;
// (3) usefulness: bound finite and rel bound < 1e-8 on these well-conditioned
// inputs (a vacuous certificate also fails).
// The mpmath-grade validation runs through the host-shim Python bindings
// (tests/test_certified.py + test_gpu_bindings.py); this gate is the on-device
// counterpart.

#include <cuComplex.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace gbs {
extern "C" void gbs_perm_glynn_fp64_batched(const cuDoubleComplex*, int, int,
                                            cuDoubleComplex*, cudaStream_t);
extern "C" void gbs_perm_certified_batched(const cuDoubleComplex*, int, int,
                                           cuDoubleComplex*, double*, cudaStream_t);
extern "C" void gbs_haf_powertrace_fp64_batched(const cuDoubleComplex*, int, int,
                                                cuDoubleComplex*, cudaStream_t);
extern "C" void gbs_haf_certified_batched(const cuDoubleComplex*, int, int,
                                          cuDoubleComplex*, double*, cudaStream_t);
extern "C" void gbs_loop_haf_fp64_batched(const cuDoubleComplex*, int, int,
                                          cuDoubleComplex*, cudaStream_t);
extern "C" void gbs_lhaf_certified_batched(const cuDoubleComplex*, int, int,
                                           cuDoubleComplex*, double*, cudaStream_t);
extern "C" void gbs_tor_certified_batched(const cuDoubleComplex*, int, int,
                                          cuDoubleComplex*, double*, cudaStream_t);
extern "C" void gbs_perm_glynn_dd_batched(const cuDoubleComplex*, int, int,
                                          cuDoubleComplex*, cudaStream_t);
extern "C" void gbs_perm_dd_certified_batched(const cuDoubleComplex*, int, int,
                                              cuDoubleComplex*, double*, cudaStream_t);
extern "C" void gbs_haf_powertrace_dd_batched(const cuDoubleComplex*, int, int,
                                              cuDoubleComplex*, cudaStream_t);
extern "C" void gbs_haf_dd_certified_batched(const cuDoubleComplex*, int, int,
                                             cuDoubleComplex*, double*, cudaStream_t);
}

using cld = std::complex<long double>;

// Long-double Glynn permanent (independent reference, ~1e-19).
static cld ref_perm_ld(const std::complex<double>* A, int n) {
  if (n == 0) return {1.0L, 0.0L};
  std::vector<cld> rowsum(n);
  for (int r = 0; r < n; ++r) {
    cld s{0.0L, 0.0L};
    for (int c = 0; c < n; ++c) s += cld(A[r * n + c]);
    rowsum[r] = s;
  }
  cld prod = rowsum[0];
  for (int r = 1; r < n; ++r) prod *= rowsum[r];
  cld total = prod;
  int sign = 1;
  const uint64_t terms = 1ull << (n - 1);
  uint64_t prev = 0;
  for (uint64_t i = 1; i < terms; ++i) {
    uint64_t gray = i ^ (i >> 1);
    int k = __builtin_ctzll(gray ^ prev);
    int col = k + 1;
    long double step = ((gray >> k) & 1ull) ? -2.0L : +2.0L;
    for (int r = 0; r < n; ++r) rowsum[r] += step * cld(A[r * n + col]);
    sign = -sign;
    cld p = rowsum[0];
    for (int r = 1; r < n; ++r) p *= rowsum[r];
    total += (sign > 0 ? p : -p);
    prev = gray;
  }
  return total / (long double)terms;
}

// Long-double power-trace hafnian with the KERNEL's gather convention (no
// diagonal zeroing -- the diagonal cancels in the inclusion-exclusion; verified
// against the zeroing reference in the CPU suite).
static cld ref_haf_ld(const std::complex<double>* A, int N) {
  if (N == 0) return {1.0L, 0.0L};
  if (N & 1) return {0.0L, 0.0L};
  int n = N / 2;
  cld total{0.0L, 0.0L};
  for (uint64_t mask = 0; mask < (1ull << n); ++mask) {
    int m = __builtin_popcountll(mask), size = 2 * m;
    std::vector<int> pidx;
    for (int i = 0; i < n; ++i)
      if ((mask >> i) & 1ull) pidx.push_back(i);
    std::vector<cld> BX(size * size), P, T(size * size);
    for (int r = 0; r < size; ++r) {
      int rr = 2 * pidx[r >> 1] + (r & 1);
      for (int c = 0; c < size; ++c) {
        int cc = 2 * pidx[c >> 1] + ((c & 1) ^ 1);
        BX[r * size + c] = cld(A[rr * N + cc]);
      }
    }
    cld coeff;
    if (size == 0) {
      coeff = (n == 0) ? cld(1, 0) : cld(0, 0);
    } else {
      std::vector<cld> p(n + 1);
      P = BX;
      for (int k = 1; k <= n; ++k) {
        cld tr{0.0L, 0.0L};
        for (int i = 0; i < size; ++i) tr += P[i * size + i];
        p[k] = tr;
        if (k < n) {
          for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j) {
              cld s{0.0L, 0.0L};
              for (int t = 0; t < size; ++t) s += P[i * size + t] * BX[t * size + j];
              T[i * size + j] = s;
            }
          P = T;
        }
      }
      std::vector<cld> e(n + 1, cld(0, 0));
      e[0] = cld(1, 0);
      for (int j = 1; j <= n; ++j) {
        cld acc{0.0L, 0.0L};
        for (int k = 1; k <= j; ++k) acc += (p[k] * 0.5L) * e[j - k];
        e[j] = acc / (long double)j;
      }
      coeff = e[n];
    }
    total += ((n - m) & 1) ? -coeff : coeff;
  }
  return total;
}

// Long-double NAIVE loop hafnian (literal recursion over loop matchings) --
// algorithmic ground truth sharing nothing with the power-trace kernels.
static cld ref_lhaf_ld(const std::complex<double>* A, int N) {
  if (N == 0) return {1.0L, 0.0L};
  std::vector<int> all(N);
  for (int i = 0; i < N; ++i) all[i] = i;
  struct Rec {
    const std::complex<double>* A;
    int N;
    cld go(const std::vector<int>& r) {
      if (r.empty()) return {1.0L, 0.0L};
      int i = r[0];
      std::vector<int> rest(r.begin() + 1, r.end());
      cld total = cld(A[i * N + i]) * go(rest);
      for (size_t k = 0; k < rest.size(); ++k) {
        std::vector<int> rest2;
        for (size_t t = 0; t < rest.size(); ++t)
          if (t != k) rest2.push_back(rest[t]);
        total += cld(A[i * N + rest[k]]) * go(rest2);
      }
      return total;
    }
  } rec{A, N};
  return rec.go(all);
}

// Long-double torontonian by fresh subset determinants (complex LU), plus the
// |term|-mass used for the tightness criterion.
static void ref_tor_ld(const std::complex<double>* O, int n, cld* val, long double* mass) {
  int dim = 2 * n;
  cld total{0.0L, 0.0L};
  long double acc = 0.0L;
  for (uint64_t mask = 0; mask < (1ull << n); ++mask) {
    int m = __builtin_popcountll(mask);
    std::vector<int> idx;
    for (int i = 0; i < n; ++i) if ((mask >> i) & 1ull) idx.push_back(i);
    for (int i = 0; i < n; ++i) if ((mask >> i) & 1ull) idx.push_back(i + n);
    int k = 2 * m;
    cld det{1.0L, 0.0L};
    if (m > 0) {
      std::vector<cld> S((size_t)k * k);
      for (int r = 0; r < k; ++r)
        for (int c = 0; c < k; ++c)
          S[r * k + c] = ((r == c) ? cld(1, 0) : cld(0, 0)) - cld(O[idx[r] * dim + idx[c]]);
      for (int c = 0; c < k; ++c) {
        int piv = c;
        for (int r = c + 1; r < k; ++r)
          if (std::abs(S[r * k + c]) > std::abs(S[piv * k + c])) piv = r;
        if (piv != c) { for (int t = 0; t < k; ++t) std::swap(S[c * k + t], S[piv * k + t]); det = -det; }
        cld pv = S[c * k + c];
        det *= pv;
        if (pv == cld(0, 0)) break;
        for (int r = c + 1; r < k; ++r) {
          cld f = S[r * k + c] / pv;
          for (int t = c; t < k; ++t) S[r * k + t] -= f * S[c * k + t];
        }
      }
    }
    cld term = cld(1.0L, 0.0L) / std::sqrt(det);
    acc += std::abs(term);
    total += ((n - m) & 1) ? -term : term;
  }
  *val = total;
  *mass = acc;
}

struct Result { double consist, worst_slack, worst_rel_bound; bool enclosed; };

template <typename PlainFn, typename CertFn, typename RefFn>
static Result run_case(PlainFn plain, CertFn cert, RefFn ref, int dim, int batch,
                       uint64_t seed, bool haf_dims, double ld_slack = 1e-17) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  std::vector<std::complex<double>> host((size_t)batch * dim * dim);
  for (int b = 0; b < batch; ++b) {
    std::vector<std::complex<double>> G((size_t)dim * dim);
    for (auto& z : G) z = {U(rng), U(rng)};
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j)
        host[(size_t)b * dim * dim + i * dim + j] =
            haf_dims ? G[i * dim + j] + G[j * dim + i] : G[i * dim + j];
  }
  std::vector<cuDoubleComplex> hin(host.size());
  for (size_t i = 0; i < host.size(); ++i)
    hin[i] = make_cuDoubleComplex(host[i].real(), host[i].imag());

  cuDoubleComplex *d_in = nullptr, *d_v = nullptr, *d_vc = nullptr;
  double* d_b = nullptr;
  cudaMalloc(&d_in, hin.size() * sizeof(cuDoubleComplex));
  cudaMalloc(&d_v, batch * sizeof(cuDoubleComplex));
  cudaMalloc(&d_vc, batch * sizeof(cuDoubleComplex));
  cudaMalloc(&d_b, batch * sizeof(double));
  cudaMemcpy(d_in, hin.data(), hin.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

  plain(d_in, dim, batch, d_v, 0);
  cert(d_in, dim, batch, d_vc, d_b, 0);
  cudaDeviceSynchronize();

  std::vector<cuDoubleComplex> hv(batch), hvc(batch);
  std::vector<double> hb(batch);
  cudaMemcpy(hv.data(), d_v, batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  cudaMemcpy(hvc.data(), d_vc, batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), d_b, batch * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_in); cudaFree(d_v); cudaFree(d_vc); cudaFree(d_b);

  Result R{0.0, 1e300, 0.0, true};
  for (int b = 0; b < batch; ++b) {
    std::complex<double> v(cuCreal(hv[b]), cuCimag(hv[b]));
    std::complex<double> vc(cuCreal(hvc[b]), cuCimag(hvc[b]));
    cld r = ref(&host[(size_t)b * dim * dim], dim);
    double consist = std::abs(vc - v) / std::max(1.0, std::abs(v));
    R.consist = std::max(R.consist, consist);
    double err = (double)std::abs(cld(vc) - r);
    double slackroom = hb[b] + ld_slack * (double)std::abs(r) - err; // >= 0 required
    if (!(slackroom >= 0.0) || !std::isfinite(hb[b])) R.enclosed = false;
    R.worst_slack = std::min(R.worst_slack, slackroom);
    double relb = hb[b] / std::max(std::abs(vc), 1e-300);
    R.worst_rel_bound = std::max(R.worst_rel_bound, relb);
  }
  return R;
}

int main() {
  bool ok = true;
  double consist = 0.0, relb = 0.0;

  for (int n = 1; n <= 10; ++n) {
    Result r = run_case(gbs::gbs_perm_glynn_fp64_batched, gbs::gbs_perm_certified_batched,
                        ref_perm_ld, n, 64, 1000 + n, false);
    if (!r.enclosed) std::printf(" [perm fp64] enclosure violated at n=%d (slack %.2e)\n", n, r.worst_slack);
    ok = ok && r.enclosed && r.consist <= 1e-13;
    consist = std::max(consist, r.consist);
    relb = std::max(relb, r.worst_rel_bound);
  }
  std::printf("perm certified: value-consistency %.2e, worst rel bound %.2e\n", consist, relb);
  if (relb > 1e-8) ok = false;

  consist = 0.0; relb = 0.0;
  for (int N = 2; N <= 10; N += 2) {
    Result r = run_case(gbs::gbs_haf_powertrace_fp64_batched, gbs::gbs_haf_certified_batched,
                        ref_haf_ld, N, 64, 2000 + N, true);
    if (!r.enclosed) std::printf(" [haf fp64] enclosure violated at N=%d (slack %.2e)\n", N, r.worst_slack);
    ok = ok && r.enclosed && r.consist <= 1e-13;
    consist = std::max(consist, r.consist);
    relb = std::max(relb, r.worst_rel_bound);
  }
  std::printf("haf certified: value-consistency %.2e, worst rel bound %.2e\n", consist, relb);
  if (relb > 1e-8) ok = false;

  consist = 0.0; relb = 0.0;
  for (int N = 2; N <= 8; N += 2) {
    Result r = run_case(gbs::gbs_loop_haf_fp64_batched, gbs::gbs_lhaf_certified_batched,
                        ref_lhaf_ld, N, 32, 3000 + N, true);
    if (!r.enclosed) std::printf(" [lhaf] enclosure violated at N=%d (slack %.2e)\n", N, r.worst_slack);
    ok = ok && r.enclosed && r.consist <= 1e-13;
    consist = std::max(consist, r.consist);
    relb = std::max(relb, r.worst_rel_bound);
  }
  std::printf("lhaf certified: value-consistency %.2e, worst rel bound %.2e\n", consist, relb);
  if (relb > 1e-8) ok = false;

  // DD-certified (the proven escalation target): consistency vs the plain DD
  // kernels; the ld reference (~1e-19 on x86's 80-bit long double) is far
  // coarser than DD error (~1e-31), so the enclosure check runs with an ld
  // slack and the TIGHTNESS check is the informative one. On platforms where
  // long double == double (e.g. AArch64 macOS -- the host-shim dev box), the
  // reference itself has fp64-grade error, so the slack widens to 1e-10 and
  // the DD-grade enclosure is deferred to the device gate + the mpmath tests.
  const bool ld_is_double = sizeof(long double) == sizeof(double);
  const double dd_slack = ld_is_double ? 1e-10 : 3e-15;
  if (ld_is_double)
    std::printf(" (long double == double on this platform: DD enclosure at reduced strength)\n");
  consist = 0.0; relb = 0.0;
  for (int n = 2; n <= 8; ++n) {
    Result r = run_case(gbs::gbs_perm_glynn_dd_batched, gbs::gbs_perm_dd_certified_batched,
                        ref_perm_ld, n, 32, 4000 + n, false, dd_slack);
    ok = ok && r.enclosed && r.consist <= 1e-13;
    consist = std::max(consist, r.consist);
    relb = std::max(relb, r.worst_rel_bound);
  }
  // the DD->complex128 output collapse floors the bound at ~u*|v| (1.1e-16
  // relative): DD-certified PROVES full double precision -- vs fp64-certified's
  // kappa*u under cancellation. The interior DD bound (~1e-30) is what makes
  // that possible; the collapse term is the honest floor.
  std::printf("perm dd-cert : value-consistency %.2e, worst rel bound %.2e\n", consist, relb);
  if (relb > 3e-16) ok = false;
  consist = 0.0; relb = 0.0;
  for (int N = 2; N <= 8; N += 2) {
    Result r = run_case(gbs::gbs_haf_powertrace_dd_batched, gbs::gbs_haf_dd_certified_batched,
                        ref_haf_ld, N, 32, 5000 + N, true, dd_slack);
    if (!r.enclosed) std::printf(" [haf dd] enclosure violated at N=%d (slack %.2e)\n", N, r.worst_slack);
    ok = ok && r.enclosed && r.consist <= 1e-13;
    consist = std::max(consist, r.consist);
    relb = std::max(relb, r.worst_rel_bound);
  }
  std::printf("haf dd-cert : value-consistency %.2e, worst rel bound %.2e\n", consist, relb);
  if (relb > 3e-16) ok = false;

  // torontonian: its own LU value -- gate = ENCLOSURE vs ld + kappa-mass tightness
  // (tor(O->0) = 0, so a raw rel-bound criterion would measure the input family).
  {
    std::mt19937_64 rng(777);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    double worst_ratio = 0.0;
    bool enc = true;
    for (int n = 1; n <= 5; ++n) {
      const int batch = 32, dim = 2 * n;
      std::vector<std::complex<double>> host((size_t)batch * dim * dim);
      for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> G((size_t)dim * dim);
        for (auto& z : G) z = {0.1 * U(rng), 0.1 * U(rng)};
        for (int i = 0; i < dim; ++i)
          for (int j = 0; j < dim; ++j)
            host[(size_t)b * dim * dim + i * dim + j] = 0.5 * (G[i * dim + j] + G[j * dim + i]);
      }
      std::vector<cuDoubleComplex> hin(host.size());
      for (size_t i = 0; i < host.size(); ++i)
        hin[i] = make_cuDoubleComplex(host[i].real(), host[i].imag());
      cuDoubleComplex *d_in = nullptr, *d_v = nullptr;
      double* d_b = nullptr;
      cudaMalloc(&d_in, hin.size() * sizeof(cuDoubleComplex));
      cudaMalloc(&d_v, batch * sizeof(cuDoubleComplex));
      cudaMalloc(&d_b, batch * sizeof(double));
      cudaMemcpy(d_in, hin.data(), hin.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
      gbs::gbs_tor_certified_batched(d_in, n, batch, d_v, d_b, 0);
      cudaDeviceSynchronize();
      std::vector<cuDoubleComplex> hv(batch);
      std::vector<double> hb(batch);
      cudaMemcpy(hv.data(), d_v, batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
      cudaMemcpy(hb.data(), d_b, batch * sizeof(double), cudaMemcpyDeviceToHost);
      cudaFree(d_in); cudaFree(d_v); cudaFree(d_b);
      for (int b = 0; b < batch; ++b) {
        cld ref;
        long double mass;
        ref_tor_ld(&host[(size_t)b * dim * dim], n, &ref, &mass);
        cld got(cuCreal(hv[b]), cuCimag(hv[b]));
        double err = (double)std::abs(got - ref);
        if (!(err <= hb[b] + 1e-17 * (double)mass) || !std::isfinite(hb[b])) enc = false;
        worst_ratio = std::max(worst_ratio, hb[b] / (double)mass);
      }
    }
    std::printf("tor certified: enclosure %s, worst bound/|term|-mass %.2e\n",
                enc ? "holds" : "VIOLATED", worst_ratio);
    ok = ok && enc && worst_ratio <= 1e-10;
  }

  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
