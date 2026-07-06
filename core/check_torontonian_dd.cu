// check_torontonian_dd.cu -- GPU double-double torontonian gate (docs/DESIGN.md §6).
//
// (1) DD == FP64 on well-conditioned real O (both correct on the device).
// (2) Cancellation: a single mode O = diag(a, a) has tor = 1/sqrt((1-a)^2) - 1,
//     which the kernel computes with a catastrophic "(1+a+...) - 1" cancellation
//     as a -> 0, but whose exact value a/(1-a) has none. So the reference a/(1-a)
//     is accurate while the FP64 kernel degrades and DD holds -- the torontonian's
//     accuracy boundary, on the device.
//
//   nvcc -O3 -std=c++17 torontonian.cu torontonian_dd.cu check_torontonian_dd.cu -o check_torontonian_dd

#include <cuComplex.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

namespace gbs {
extern "C" void gbs_tor_fp64_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
extern "C" void gbs_tor_dd_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
}

static std::vector<cuDoubleComplex> to_cu(const std::vector<double>& re) {
  std::vector<cuDoubleComplex> v(re.size());
  for (size_t i = 0; i < re.size(); ++i) v[i] = make_cuDoubleComplex(re[i], 0.0);
  return v;
}

static std::vector<double> run_kernel(void (*fn)(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t),
                                      const std::vector<double>& host, int n, int batch) {
  auto hin = to_cu(host);
  cuDoubleComplex *d_in = nullptr, *d_out = nullptr;
  cudaMalloc(&d_in, hin.size() * sizeof(cuDoubleComplex));
  cudaMalloc(&d_out, batch * sizeof(cuDoubleComplex));
  cudaMemcpy(d_in, hin.data(), hin.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  fn(d_in, n, batch, d_out, 0);
  cudaDeviceSynchronize();
  std::vector<cuDoubleComplex> hout(batch);
  cudaMemcpy(hout.data(), d_out, batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  cudaFree(d_in); cudaFree(d_out);
  std::vector<double> out(batch);
  for (int b = 0; b < batch; ++b) out[b] = cuCreal(hout[b]);
  return out;
}

static std::vector<double> sym_real(int n, std::mt19937_64& rng, double scale) {
  std::uniform_real_distribution<double> U(-1, 1);
  int N = 2 * n;
  std::vector<double> M((size_t)N * N, 0.0);
  for (int i = 0; i < N; ++i)
    for (int j = i; j < N; ++j) { double v = scale * U(rng); M[i * N + j] = v; M[j * N + i] = v; }
  return M;
}

int main() {
  std::mt19937_64 rng(20260620);
  const double dd_tol = 1e-12;
  bool ok = true;

  // (1) DD == FP64 on well-conditioned real O
  double worst_easy = 0.0;
  for (int n = 1; n <= 4; ++n) {
    const int batch = 64;
    std::vector<double> host;
    for (int b = 0; b < batch; ++b) { auto m = sym_real(n, rng, 0.12); host.insert(host.end(), m.begin(), m.end()); }
    auto dd = run_kernel(gbs::gbs_tor_dd_batched, host, n, batch);
    auto fp = run_kernel(gbs::gbs_tor_fp64_batched, host, n, batch);
    for (int b = 0; b < batch; ++b) {
      double mm = std::max(std::fabs(fp[b]), 1e-300);
      worst_easy = std::max(worst_easy, std::fabs(dd[b] - fp[b]) / mm);
    }
  }
  // The torontonian's alternating subset sum has mild cancellation even here, so
  // DD is often *more* accurate than FP64 -- they agree only to FP64's level
  // (~1e-9), not to machine precision. That agreement confirms DD correctness;
  // part (2) pins DD's accuracy against an exact reference.
  if (worst_easy > 1e-7) ok = false;
  std::printf("(1) DD ~= FP64 on well-conditioned (n=1..4): max rel diff %.3e  [tol 1e-7]\n", worst_easy);

  // (2) single-mode cancellation: O = diag(a,a), tor = a/(1-a) (exact ref)
  std::printf("(2) cancellation:  a         relerr_FP64    relerr_DD     DD held?\n");
  bool dd_ok = true;
  double e_fp_worst = 0.0, e_dd_worst = 0.0;
  for (double a : {1e-2, 1e-4, 1e-6, 1e-8, 1e-10}) {
    std::vector<double> O = {a, 0.0, 0.0, a};  // 2x2, 1 mode
    auto dd = run_kernel(gbs::gbs_tor_dd_batched, O, 1, 1);
    auto fp = run_kernel(gbs::gbs_tor_fp64_batched, O, 1, 1);
    long double ref = (long double)a / (1.0L - (long double)a);  // tor, no cancellation
    double e_dd = (double)(std::fabs((long double)dd[0] - ref) / std::fabs(ref));
    double e_fp = (double)(std::fabs((long double)fp[0] - ref) / std::fabs(ref));
    bool held = (e_dd < dd_tol) && (e_fp <= 1e-12 || e_dd < e_fp / 50.0);
    dd_ok = dd_ok && held;
    e_fp_worst = std::max(e_fp_worst, e_fp);
    e_dd_worst = std::max(e_dd_worst, e_dd);
    std::printf("                   %.0e    %.3e    %.3e    %s\n", a, e_fp, e_dd, held ? "yes" : "NO");
  }
  bool demonstrated = (e_fp_worst > 1e-9) && (e_dd_worst < 1e-12);
  ok = ok && dd_ok && demonstrated;

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
