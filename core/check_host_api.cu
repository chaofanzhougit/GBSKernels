// check_host_api.cu -- smoke test for the host-facing wrappers (host_api.cu).
//
// Validates the H2D/launch/D2H plumbing (and batching) of every host wrapper
// against known closed-form values. The kernels themselves are validated by the
// check_*.cu gates; this confirms the host_api layer the Python bindings call.
// Builds + runs under the CPU pre-flight shim.
//
//   nvcc -O2 -std=c++17 *.cu check_host_api.cu -o check_host_api   (or via shim)

#include <cuComplex.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace gbs {
extern "C" {
int gbs_perm_host(const cuDoubleComplex*, int, int, cuDoubleComplex*);
int gbs_perm_dd_host(const cuDoubleComplex*, int, int, cuDoubleComplex*);
int gbs_haf_host(const cuDoubleComplex*, int, int, cuDoubleComplex*);
int gbs_lhaf_host(const cuDoubleComplex*, int, int, cuDoubleComplex*);
int gbs_tor_host(const cuDoubleComplex*, int, int, cuDoubleComplex*);
}
}

static int failures = 0;
static void check(const char* name, double got, double want) {
  bool ok = std::abs(got - want) < 1e-9 * (std::abs(want) + 1.0);
  std::printf("  %-10s got %.6f  want %.6f  %s\n", name, got, want, ok ? "ok" : "FAIL");
  if (!ok) failures++;
}

static std::vector<cuDoubleComplex> fill(std::initializer_list<double> re, int) {
  std::vector<cuDoubleComplex> v;
  for (double x : re) v.push_back(make_cuDoubleComplex(x, 0.0));
  return v;
}

// Independent host Glynn permanent (mirrors check_permanent.cu), to validate the
// gbs_perm_host auto-dispatch onto the cooperative kernel at n >= the crossover.
static std::complex<double> host_perm_glynn(const std::complex<double>* A, int n) {
  if (n == 0) return {1.0, 0.0};
  std::vector<std::complex<double>> rowsum(n);
  for (int r = 0; r < n; ++r) {
    std::complex<double> s{0.0, 0.0};
    for (int c = 0; c < n; ++c) s += A[r * n + c];
    rowsum[r] = s;
  }
  std::complex<double> prod = rowsum[0];
  for (int r = 1; r < n; ++r) prod *= rowsum[r];
  std::complex<double> total = prod;
  int sign = 1;
  const uint64_t terms = 1ull << (n - 1);
  uint64_t prev_gray = 0;
  for (uint64_t i = 1; i < terms; ++i) {
    uint64_t gray = i ^ (i >> 1);
    int k = __builtin_ctzll(gray ^ prev_gray);
    int col = k + 1;
    double step = ((gray >> k) & 1ull) ? -2.0 : +2.0;
    for (int r = 0; r < n; ++r) rowsum[r] += step * A[r * n + col];
    sign = -sign;
    std::complex<double> p = rowsum[0];
    for (int r = 1; r < n; ++r) p *= rowsum[r];
    total += (sign > 0 ? p : -p);
    prev_gray = gray;
  }
  return total / (double)terms;
}

int main() {
  std::printf("host_api smoke test (plumbing + batching):\n");

  // perm([[1,2],[3,4]]) = 10 ; batch of 2 (same matrix) -> [10, 10]
  {
    auto m = fill({1, 2, 3, 4}, 2);
    std::vector<cuDoubleComplex> two(m); two.insert(two.end(), m.begin(), m.end());
    std::vector<cuDoubleComplex> out(2);
    gbs::gbs_perm_host(two.data(), 2, 2, out.data());
    check("perm[0]", cuCreal(out[0]), 10.0);
    check("perm[1]", cuCreal(out[1]), 10.0);  // batching
  }
  // perm_dd([[1,2],[3,4]]) = 10
  {
    auto m = fill({1, 2, 3, 4}, 2);
    cuDoubleComplex out;
    gbs::gbs_perm_dd_host(m.data(), 2, 1, &out);
    check("perm_dd", cuCreal(out), 10.0);
  }
  // haf(all-ones 4x4) = (2*2-1)!! = 3
  {
    std::vector<cuDoubleComplex> m(16, make_cuDoubleComplex(1.0, 0.0));
    cuDoubleComplex out;
    gbs::gbs_haf_host(m.data(), 4, 1, &out);
    check("haf", cuCreal(out), 3.0);
  }
  // lhaf(all-ones 2x2) = b + a*c = 1 + 1 = 2  (telephone number T(2))
  {
    std::vector<cuDoubleComplex> m(4, make_cuDoubleComplex(1.0, 0.0));
    cuDoubleComplex out;
    gbs::gbs_lhaf_host(m.data(), 2, 1, &out);
    check("lhaf", cuCreal(out), 2.0);
  }
  // tor(zero 2x2, n=1) = 1/sqrt(det(I)) - 1 = 0
  {
    std::vector<cuDoubleComplex> m(4, make_cuDoubleComplex(0.0, 0.0));
    cuDoubleComplex out;
    gbs::gbs_tor_host(m.data(), 1, 1, &out);
    check("tor", cuCreal(out), 0.0);
  }
  // Dispatch: an all-real physical O routes to the recursive prefix-Cholesky
  // kernel -- closed form tor(a*I) = (a/(1-a))^n pins the dispatched value.
  {
    const int nm = 2, dim = 4;
    std::vector<cuDoubleComplex> m(dim * dim, make_cuDoubleComplex(0.0, 0.0));
    for (int i = 0; i < dim; ++i) m[i * dim + i] = make_cuDoubleComplex(0.2, 0.0);
    cuDoubleComplex out;
    gbs::gbs_tor_host(m.data(), nm, 1, &out);
    check("tor_rec", cuCreal(out), 0.0625);          // (0.2/0.8)^2, via recursion
    // complex input: must keep the complex-LU path and agree (imag ~ 0 here)
    m[1] = make_cuDoubleComplex(0.0, 1e-12);
    m[dim] = make_cuDoubleComplex(0.0, 1e-12);
    gbs::gbs_tor_host(m.data(), nm, 1, &out);
    check("tor_cplx", cuCreal(out), 0.0625);
    // off-domain real O (I - O_S not SPD): recursive NaNs -> complex-LU fallback
    std::vector<cuDoubleComplex> bad(dim * dim, make_cuDoubleComplex(0.0, 0.0));
    for (int i = 0; i < dim; ++i) bad[i * dim + i] = make_cuDoubleComplex(1.5, 0.0);
    gbs::gbs_tor_host(bad.data(), nm, 1, &out);
    bool finite = std::isfinite(cuCreal(out)) && std::isfinite(cuCimag(out));
    std::printf("  %-10s finite=%d (off-domain fell back to complex-LU)  %s\n",
                "tor_fb", (int)finite, finite ? "ok" : "FAIL");
    if (!finite) failures++;
  }
  // Auto-dispatch: n=12 (>= crossover) routes gbs_perm_host to the cooperative
  // kernel; the result must still equal the independent Glynn permanent.
  {
    const int n = 12, batch = 8;
    std::mt19937_64 rng(2024);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<std::complex<double>> host(batch * n * n);
    for (auto& z : host) z = {U(rng), U(rng)};
    std::vector<cuDoubleComplex> hin(host.size());
    for (size_t i = 0; i < host.size(); ++i)
      hin[i] = make_cuDoubleComplex(host[i].real(), host[i].imag());
    std::vector<cuDoubleComplex> out(batch);
    gbs::gbs_perm_host(hin.data(), n, batch, out.data());
    double worst = 0.0;
    for (int b = 0; b < batch; ++b) {
      std::complex<double> ref = host_perm_glynn(host.data() + (size_t)b * n * n, n);
      std::complex<double> got{cuCreal(out[b]), cuCimag(out[b])};
      worst = std::max(worst, std::abs(got - ref) / std::max(std::abs(ref), 1e-300));
    }
    bool ok = worst < 1e-9;
    std::printf("  %-10s rel err %.3e (coop dispatch, n=12)  %s\n", "perm@n12", worst, ok ? "ok" : "FAIL");
    if (!ok) failures++;
  }

  std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
