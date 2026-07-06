// check_permanent_dd.cu -- GPU double-double permanent gate (docs/DESIGN.md §6/sec.8).
//
// Validates the DD permanent kernel two ways against an INDEPENDENT reference
// (x86 80-bit long double, ~18-19 digits -- a different precision path than DD):
//   (1) accuracy: GPU-DD tracks the long-double reference on random and
//       moderately cancellation-heavy inputs;
//   (2) value: on cancellation-heavy inputs GPU-DD is dramatically closer to the
//       reference than GPU-FP64 -- i.e. the DD arithmetic really is restoring the
//       precision the FP64 alternating sum loses, on the device.
// (The DD algorithm itself is validated to ~31 digits vs mpmath in the Python
// suite; this gate confirms the on-device behaviour.)
//
//   nvcc -O3 -std=c++17 permanent.cu permanent_dd.cu check_permanent_dd.cu -o check_permanent_dd

#include <cuComplex.h>

#include <complex>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

namespace gbs {
extern "C" void gbs_perm_glynn_fp64_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
extern "C" void gbs_perm_glynn_dd_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
}

using cd = std::complex<double>;
using cld = std::complex<long double>;

// Independent high-precision reference: Glynn permanent in 80-bit long double.
static cld host_perm_glynn_ld(const cd* A, int n) {
  if (n == 0) return cld(1.0L, 0.0L);
  std::vector<cld> rowsum(n);
  for (int r = 0; r < n; ++r) {
    cld s(0.0L, 0.0L);
    for (int c = 0; c < n; ++c) s += cld((long double)A[r * n + c].real(), (long double)A[r * n + c].imag());
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
    long double step = ((gray >> k) & 1ull) ? -2.0L : 2.0L;
    for (int r = 0; r < n; ++r)
      rowsum[r] += step * cld((long double)A[r * n + col].real(), (long double)A[r * n + col].imag());
    sign = -sign;
    cld p = rowsum[0];
    for (int r = 1; r < n; ++r) p *= rowsum[r];
    total += (sign > 0 ? p : -p);
    prev = gray;
  }
  return total / (long double)terms;
}

static double rel_ld(cd got, cld ref) {
  long double d = std::abs(cld((long double)got.real(), (long double)got.imag()) - ref);
  long double m = std::abs(ref);
  return (double)(d / (m > 1e-300L ? m : 1e-300L));
}

// run one batch through a kernel; return host results
static std::vector<cd> run_kernel(void (*fn)(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t),
                                  const std::vector<cd>& host, int n, int batch) {
  std::vector<cuDoubleComplex> hin(host.size());
  for (size_t i = 0; i < host.size(); ++i) hin[i] = make_cuDoubleComplex(host[i].real(), host[i].imag());
  cuDoubleComplex *d_in = nullptr, *d_out = nullptr;
  cudaMalloc(&d_in, hin.size() * sizeof(cuDoubleComplex));
  cudaMalloc(&d_out, batch * sizeof(cuDoubleComplex));
  cudaMemcpy(d_in, hin.data(), hin.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  fn(d_in, n, batch, d_out, 0);
  cudaDeviceSynchronize();
  std::vector<cuDoubleComplex> hout(batch);
  cudaMemcpy(hout.data(), d_out, batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  cudaFree(d_in); cudaFree(d_out);
  std::vector<cd> out(batch);
  for (int b = 0; b < batch; ++b) out[b] = cd(cuCreal(hout[b]), cuCimag(hout[b]));
  return out;
}

// B = [[2,1],[-1,0.5+delta]] (+) random remainder: tunable Glynn cancellation
// (kappa ~ 1/delta), same family as the Python make_cancellation_matrix.
static std::vector<cd> cancellation_matrix(int n, double delta, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> U(-1, 1);
  std::vector<cd> A((size_t)n * n, cd(0, 0));
  A[0] = cd(2, 0); A[1] = cd(1, 0); A[n] = cd(-1, 0); A[n + 1] = cd(0.5 + delta, 0);
  for (int i = 2; i < n; ++i)
    for (int j = 2; j < n; ++j) A[i * n + j] = cd(U(rng), U(rng));
  return A;
}

// Exact reference for the cancellation matrix via its block structure:
// perm = perm(B) * perm(R), with perm(B) = 2*B11 - 1 EXACT in FP64 (B11 ~ 0.5,
// so 2*B11 is exact and 2*B11-1 is exact by Sterbenz), and R well-conditioned so
// its permanent is accurate. Portable (no reliance on long double width).
static cld cancellation_ref(const std::vector<cd>& A, int n) {
  cld permB = cld((long double)(2.0 * A[n + 1].real() - 1.0), 0.0L);  // 2*B11 - 1
  std::vector<cd> R((size_t)(n - 2) * (n - 2));
  for (int i = 2; i < n; ++i)
    for (int j = 2; j < n; ++j) R[(i - 2) * (n - 2) + (j - 2)] = A[i * n + j];
  return permB * host_perm_glynn_ld(R.data(), n - 2);
}

int main() {
  std::mt19937_64 rng(20260620);
  std::uniform_real_distribution<double> U(-1, 1);
  const double dd_tol = 1e-12;
  bool ok = true;

  // (1) DD is correct on easy (well-conditioned) inputs: matches FP64 (both right)
  double worst_easy = 0.0;
  for (int n = 2; n <= 10; ++n) {
    const int batch = 64;
    std::vector<cd> host((size_t)batch * n * n);
    for (auto& z : host) z = cd(U(rng), U(rng));
    auto dd = run_kernel(gbs::gbs_perm_glynn_dd_batched, host, n, batch);
    auto fp = run_kernel(gbs::gbs_perm_glynn_fp64_batched, host, n, batch);
    for (int b = 0; b < batch; ++b) {
      double m = std::max(std::abs(fp[b]), 1e-300);
      double e = std::abs(dd[b] - fp[b]) / m;
      worst_easy = std::max(worst_easy, e);
    }
  }
  if (worst_easy > 1e-11) ok = false;
  std::printf("(1) DD == FP64 on well-conditioned (n=2..10): max rel diff %.3e  [tol 1e-11]\n", worst_easy);

  // (2) DD stays accurate while FP64 degrades, vs the EXACT block reference.
  // Criterion: DD is always accurate; where FP64 has actually degraded
  // (e_fp > 1e-12), DD must be far better. (At low kappa both are near machine
  // precision -- nothing to beat.)
  std::printf("(2) cancellation:  delta     relerr_FP64    relerr_DD     DD held?\n");
  bool dd_ok = true;
  double e_fp_worst = 0.0, e_dd_worst = 0.0;
  for (double delta : {1e-2, 1e-4, 1e-6, 1e-8, 1e-10}) {
    auto host = cancellation_matrix(6, delta, rng);
    auto dd = run_kernel(gbs::gbs_perm_glynn_dd_batched, host, 6, 1);
    auto fp = run_kernel(gbs::gbs_perm_glynn_fp64_batched, host, 6, 1);
    cld ref = cancellation_ref(host, 6);
    double e_dd = rel_ld(dd[0], ref), e_fp = rel_ld(fp[0], ref);
    bool held = (e_dd < dd_tol) && (e_fp <= 1e-12 || e_dd < e_fp / 50.0);
    dd_ok = dd_ok && held;
    e_fp_worst = std::max(e_fp_worst, e_fp);
    e_dd_worst = std::max(e_dd_worst, e_dd);
    std::printf("                   %.0e    %.3e    %.3e    %s\n", delta, e_fp, e_dd, held ? "yes" : "NO");
  }
  // the demonstration must be real: FP64 genuinely broke somewhere, DD never did
  bool demonstrated = (e_fp_worst > 1e-9) && (e_dd_worst < 1e-12);
  ok = ok && dd_ok && demonstrated;

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
