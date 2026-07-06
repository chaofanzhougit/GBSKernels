// bench_kernels.cu -- GPU throughput timing for the four batched kernels.
//
// Built by CMake and run in a rented-GPU session (docs/DESIGN.md §9/sec.10). Times each
// batched kernel over a few (size, batch) points and emits one JSON object per
// line to stdout; bench/throughput_gpu.py wraps it with GPU/provenance metadata
// and writes the append-only artifact. Honesty guard (docs/DESIGN.md §8): after the
// device sync, every result is reduced to a checksum that is printed, so no
// async early-return can fake a fast time.
//
// Structured like the check_*.cu gates (device malloc/copy/launch/sync), so it
// also compiles and runs under the CPU pre-flight shim (where timing is
// meaningless but the build is verified). Real timing requires nvcc + a GPU.

#include <cuComplex.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace gbs {
extern "C" void gbs_perm_glynn_fp64_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
extern "C" void gbs_perm_glynn_dd_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
extern "C" void gbs_haf_powertrace_fp64_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
// size-specialized (small buffer-cap) hafnian -- perf-research A/B vs the full kernel.
extern "C" void gbs_haf_powertrace_fp64_small_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
extern "C" void gbs_haf_powertrace_dd_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
extern "C" void gbs_loop_haf_fp64_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
extern "C" void gbs_loop_haf_dd_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
extern "C" void gbs_tor_fp64_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
// real-Cholesky torontonian (physical real O) -- perf-research candidate C A/B vs `tor`.
extern "C" void gbs_tor_real_chol_fp64_batched(const double*, int, int, double*, cudaStream_t);
extern "C" void gbs_tor_recursive_real_fp64_batched(const double*, int, int, double*, cudaStream_t);
extern "C" void gbs_tor_recursive_single_batched(const double*, int, int, double*, cudaStream_t);
extern "C" void gbs_perm_certified_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*, cudaStream_t);
extern "C" void gbs_haf_certified_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*, cudaStream_t);
extern "C" void gbs_lhaf_certified_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*, cudaStream_t);
extern "C" void gbs_tor_certified_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*, cudaStream_t);
extern "C" void gbs_haf_dd_certified_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*, cudaStream_t);
extern "C" void gbs_tor_dd_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
// cooperative variants: extra `groups` arg (cooperation width); return cudaError_t.
extern "C" int gbs_perm_glynn_coop_batched(const cuDoubleComplex*, int, int, int, cuDoubleComplex*, cudaStream_t);
extern "C" int gbs_haf_coop_batched(const cuDoubleComplex*, int, int, int, cuDoubleComplex*, cudaStream_t);
extern "C" int gbs_loop_haf_coop_batched(const cuDoubleComplex*, int, int, int, cuDoubleComplex*, cudaStream_t);
extern "C" int gbs_tor_coop_batched(const cuDoubleComplex*, int, int, int, cuDoubleComplex*, cudaStream_t);
// fused warp permanent (permanent_warp.cu): GPU-only; benched when GBS_BENCH_WARP is set.
extern "C" int gbs_perm_glynn_warp_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
}

using Fn = void (*)(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);

// matrix_dim = side length of each input matrix (for allocation & symmetry).
// kernel_dim = the integer the kernel itself takes: n for perm (n x n) and tor
// (2n x 2n, arg n); N for haf/lhaf (N x N, arg N).
static double bench_one(Fn fn, const char* name, int matrix_dim, int kernel_dim,
                        int batch, int repeats, bool symmetric, double scale = 1.0) {
  std::mt19937_64 rng(1234 + matrix_dim);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  const int D = matrix_dim;
  std::vector<cuDoubleComplex> hin((size_t)batch * D * D);
  for (int b = 0; b < batch; ++b)
    for (int i = 0; i < D; ++i)
      for (int j = 0; j < D; ++j)
        hin[((size_t)b * D + i) * D + j] = make_cuDoubleComplex(scale * U(rng), scale * U(rng));
  if (symmetric) // haf/lhaf/tor expect symmetric inputs
    for (int b = 0; b < batch; ++b)
      for (int i = 0; i < D; ++i)
        for (int j = i + 1; j < D; ++j)
          hin[((size_t)b * D + j) * D + i] = hin[((size_t)b * D + i) * D + j];

  cuDoubleComplex *d_in = nullptr, *d_out = nullptr;
  cudaMalloc(&d_in, hin.size() * sizeof(cuDoubleComplex));
  cudaMalloc(&d_out, (size_t)batch * sizeof(cuDoubleComplex));
  cudaMemcpy(d_in, hin.data(), hin.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

  std::vector<cuDoubleComplex> hout(batch);
  std::vector<double> eps_all;
  eps_all.reserve(repeats);
  double checksum = 0.0;
  // Warm-up policy (docs/DESIGN.md §9): 2 untimed launches to reach steady-state GPU clocks
  // and prime the kernel before the timed repeats -- the cold first launch is never a
  // headline number. (Negligible on the host shim.)
  for (int w = 0; w < 2; ++w) { fn(d_in, kernel_dim, batch, d_out, 0); cudaDeviceSynchronize(); }
  for (int r = 0; r < repeats; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    fn(d_in, kernel_dim, batch, d_out, 0);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    // honesty guard: materialize every output post-sync into a checksum.
    cudaMemcpy(hout.data(), d_out, (size_t)batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    double s = 0.0;
    for (int b = 0; b < batch; ++b) s += cuCabs(hout[b]);
    checksum = s;
    eps_all.push_back(secs > 0 ? batch / secs : 0.0);
  }
  cudaFree(d_in);
  cudaFree(d_out);

  // median + interquartile range over the repeats (not just best-of-N).
  std::sort(eps_all.begin(), eps_all.end());
  auto quantile = [&](double q) {
    double x = q * (eps_all.size() - 1);
    size_t lo = (size_t)x;
    double frac = x - lo;
    return lo + 1 < eps_all.size() ? eps_all[lo] * (1 - frac) + eps_all[lo + 1] * frac
                                   : eps_all[lo];
  };
  double median = quantile(0.5), iqr = quantile(0.75) - quantile(0.25);
  std::printf(
      "{\"func\":\"%s\",\"matrix_dim\":%d,\"batch\":%d,\"repeats\":%d,"
      "\"evals_per_sec_median\":%.6e,\"evals_per_sec_iqr\":%.6e,"
      "\"evals_per_sec_best\":%.6e,\"checksum\":%.12e}\n",
      name, matrix_dim, batch, repeats, median, iqr, eps_all.back(), checksum);
  return median;
}

// Real-input variant of bench_one for the real-Cholesky torontonian (real O in,
// real out -- a different signature than the complex kernels). Same warm-up +
// median/IQR + post-sync checksum honesty guard. The checksum is the anti-cheat
// materialization, NOT a cross-kernel equality vs `tor` (the value equality on
// identical real O is the gate's job, check_torontonian_real_chol.cu).
using FnReal = void (*)(const double*, int, int, double*, cudaStream_t);

static double bench_one_real(FnReal fn, const char* name, int matrix_dim, int kernel_dim,
                             int batch, int repeats, double scale) {
  std::mt19937_64 rng(1234 + matrix_dim);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  const int D = matrix_dim;
  std::vector<double> hin((size_t)batch * D * D);
  for (int b = 0; b < batch; ++b)
    for (int i = 0; i < D; ++i)
      for (int j = 0; j < D; ++j)
        hin[((size_t)b * D + i) * D + j] = scale * U(rng);
  for (int b = 0; b < batch; ++b) // symmetric physical O (real)
    for (int i = 0; i < D; ++i)
      for (int j = i + 1; j < D; ++j)
        hin[((size_t)b * D + j) * D + i] = hin[((size_t)b * D + i) * D + j];

  double *d_in = nullptr, *d_out = nullptr;
  cudaMalloc(&d_in, hin.size() * sizeof(double));
  cudaMalloc(&d_out, (size_t)batch * sizeof(double));
  cudaMemcpy(d_in, hin.data(), hin.size() * sizeof(double), cudaMemcpyHostToDevice);

  std::vector<double> hout(batch);
  std::vector<double> eps_all;
  eps_all.reserve(repeats);
  double checksum = 0.0;
  for (int w = 0; w < 2; ++w) { fn(d_in, kernel_dim, batch, d_out, 0); cudaDeviceSynchronize(); } // warm-up
  for (int r = 0; r < repeats; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    fn(d_in, kernel_dim, batch, d_out, 0);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    cudaMemcpy(hout.data(), d_out, (size_t)batch * sizeof(double), cudaMemcpyDeviceToHost);
    double s = 0.0;
    for (int b = 0; b < batch; ++b) s += std::fabs(hout[b]);
    checksum = s;
    eps_all.push_back(secs > 0 ? batch / secs : 0.0);
  }
  cudaFree(d_in);
  cudaFree(d_out);

  std::sort(eps_all.begin(), eps_all.end());
  auto quantile = [&](double q) {
    double x = q * (eps_all.size() - 1);
    size_t lo = (size_t)x;
    double frac = x - lo;
    return lo + 1 < eps_all.size() ? eps_all[lo] * (1 - frac) + eps_all[lo + 1] * frac
                                   : eps_all[lo];
  };
  double median = quantile(0.5), iqr = quantile(0.75) - quantile(0.25);
  std::printf(
      "{\"func\":\"%s\",\"matrix_dim\":%d,\"batch\":%d,\"repeats\":%d,"
      "\"evals_per_sec_median\":%.6e,\"evals_per_sec_iqr\":%.6e,"
      "\"evals_per_sec_best\":%.6e,\"checksum\":%.12e}\n",
      name, matrix_dim, batch, repeats, median, iqr, eps_all.back(), checksum);
  return median;
}

// Cooperative-kernel timing: `groups` threads cooperate on each matrix (the
// subset sum is split + reduced). Same shape/inputs as bench_one but for the coop
// signature (extra `groups` arg); reported as `<name>` with the group width.
using CoopFn = int (*)(const cuDoubleComplex*, int, int, int, cuDoubleComplex*, cudaStream_t);

static double bench_coop(CoopFn fn, const char* name, int matrix_dim, int kernel_dim,
                         int batch, int repeats, bool symmetric, double scale, int groups) {
  std::mt19937_64 rng(1234 + matrix_dim);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  const int D = matrix_dim;
  std::vector<cuDoubleComplex> hin((size_t)batch * D * D);
  for (int b = 0; b < batch; ++b)
    for (int i = 0; i < D; ++i)
      for (int j = 0; j < D; ++j)
        hin[((size_t)b * D + i) * D + j] = make_cuDoubleComplex(scale * U(rng), scale * U(rng));
  if (symmetric)
    for (int b = 0; b < batch; ++b)
      for (int i = 0; i < D; ++i)
        for (int j = i + 1; j < D; ++j)
          hin[((size_t)b * D + j) * D + i] = hin[((size_t)b * D + i) * D + j];

  cuDoubleComplex *d_in = nullptr, *d_out = nullptr;
  cudaMalloc(&d_in, hin.size() * sizeof(cuDoubleComplex));
  cudaMalloc(&d_out, (size_t)batch * sizeof(cuDoubleComplex));
  cudaMemcpy(d_in, hin.data(), hin.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

  std::vector<cuDoubleComplex> hout(batch);
  std::vector<double> eps_all;
  eps_all.reserve(repeats);
  double checksum = 0.0;
  for (int w = 0; w < 2; ++w) { fn(d_in, kernel_dim, batch, groups, d_out, 0); cudaDeviceSynchronize(); } // warm-up
  for (int r = 0; r < repeats; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    fn(d_in, kernel_dim, batch, groups, d_out, 0);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    cudaMemcpy(hout.data(), d_out, (size_t)batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    double s = 0.0;
    for (int b = 0; b < batch; ++b) s += cuCabs(hout[b]);
    checksum = s;
    eps_all.push_back(secs > 0 ? batch / secs : 0.0);
  }
  cudaFree(d_in);
  cudaFree(d_out);

  std::sort(eps_all.begin(), eps_all.end());
  auto quantile = [&](double q) {
    double x = q * (eps_all.size() - 1);
    size_t lo = (size_t)x;
    double frac = x - lo;
    return lo + 1 < eps_all.size() ? eps_all[lo] * (1 - frac) + eps_all[lo + 1] * frac
                                   : eps_all[lo];
  };
  double median = quantile(0.5), iqr = quantile(0.75) - quantile(0.25);
  std::printf(
      "{\"func\":\"%s\",\"groups\":%d,\"matrix_dim\":%d,\"batch\":%d,\"repeats\":%d,"
      "\"evals_per_sec_median\":%.6e,\"evals_per_sec_iqr\":%.6e,"
      "\"evals_per_sec_best\":%.6e,\"checksum\":%.12e}\n",
      name, groups, matrix_dim, batch, repeats, median, iqr, eps_all.back(), checksum);
  return median;
}

#ifdef GBS_BENCH_WARP
// Adapter so the fused warp permanent (no `groups` arg) reuses bench_coop; the warp
// always uses 32 lanes, recorded as groups=32. Compiled only when GBS_BENCH_WARP is
// set (the bench target on the GPU) and permanent_warp.cu is linked.
static int warp_adapter(const cuDoubleComplex* m, int n, int b, int /*groups*/,
                        cuDoubleComplex* o, cudaStream_t s) {
  return gbs::gbs_perm_glynn_warp_batched(m, n, b, o, s);
}
#endif

int main(int argc, char** argv) {
  int batch = (argc > 1) ? std::atoi(argv[1]) : 1024;
  int reps = (argc > 2) ? std::atoi(argv[2]) : 5;
  for (int n : {8, 12, 16})
    bench_one(gbs::gbs_perm_glynn_fp64_batched, "perm", n, n, batch, reps, false);
  // cooperative variants (perf): same/larger sizes than per-thread for a direct
  // speedup + the regime where the per-thread serial subset sum collapses. Two
  // cooperation widths (8, 32) show the scaling. bench/throughput_gpu.py records
  // the "groups" field; the crossover is read off X vs X_coop at equal size.
  for (int g : {8, 32}) {
    for (int n : {12, 16, 20}) bench_coop(gbs::gbs_perm_glynn_coop_batched, "perm_coop", n, n, batch, reps, false, 1.0, g);
    for (int N : {12, 16, 20}) bench_coop(gbs::gbs_haf_coop_batched, "haf_coop", N, N, batch, reps, true, 1.0, g);
    for (int N : {12, 16}) bench_coop(gbs::gbs_loop_haf_coop_batched, "lhaf_coop", N, N, batch, reps, true, 1.0, g);
    for (int n : {6, 8, 10}) bench_coop(gbs::gbs_tor_coop_batched, "tor_coop", 2 * n, n, batch, reps, true, 0.05, g);
  }
#ifdef GBS_BENCH_WARP
  // fused __shfl permanent (32 lanes/matrix) vs the map/reduce coop above, equal n.
  for (int n : {12, 16, 20})
    bench_coop(warp_adapter, "perm_warp", n, n, batch, reps, false, 1.0, 32);
#endif
  for (int n : {8, 12, 16}) // DD tier -- the FP64<->DD throughput crossover (docs/DESIGN.md §9)
    bench_one(gbs::gbs_perm_glynn_dd_batched, "perm_dd", n, n, batch, reps, false);
  for (int N : {8, 12, 16})
    bench_one(gbs::gbs_haf_powertrace_fp64_batched, "haf", N, N, batch, reps, true);
  // perf research A/B: the size-specialized (small buffer-cap) hafnian vs the full
  // kernel at the SAME small N, where the footprint win should show (read off
  // haf vs haf_small at N=8,12). Same matrices/seed -> identical checksums.
  for (int N : {8, 12})
    bench_one(gbs::gbs_haf_powertrace_fp64_small_batched, "haf_small", N, N, batch, reps, true);
  for (int N : {8, 12, 16}) // DD hafnian tier
    bench_one(gbs::gbs_haf_powertrace_dd_batched, "haf_dd", N, N, batch, reps, true);
  for (int N : {8, 12, 16})
    bench_one(gbs::gbs_loop_haf_fp64_batched, "lhaf", N, N, batch, reps, true);
  for (int N : {8, 12}) // DD loop hafnian (heavy footprint; smaller grid)
    bench_one(gbs::gbs_loop_haf_dd_batched, "lhaf_dd", N, N, batch, reps, true);
  // torontonian needs physical O: small norm so I - O_S stays positive-definite
  // (real DD sqrt of a negative det is undefined; full-norm random O is not a
  // valid torontonian input). scale 0.05 keeps the spectral radius < 1.
  for (int n : {4, 6, 8})
    bench_one(gbs::gbs_tor_fp64_batched, "tor", 2 * n, n, batch, reps, true, 0.05);
  // perf research A/B candidate C: the real-Cholesky torontonian (physical real O,
  // real arithmetic ~1/4 the flops + half the per-thread buffer) vs the complex-LU
  // `tor` at the SAME n (read off tor vs tor_real_chol at n=4,6,8). Value equality on
  // identical real O is the gate's job (check_torontonian_real_chol.cu); this is timing.
  for (int n : {4, 6, 8}) {
    bench_one_real(gbs::gbs_tor_real_chol_fp64_batched, "tor_real_chol", 2 * n, n, batch, reps, 0.05);
    // R2 recursive prefix-Cholesky (real O): the A/B against tor + tor_real_chol at the same n.
    bench_one_real(gbs::gbs_tor_recursive_real_fp64_batched, "tor_recursive", 2 * n, n, batch, reps, 0.05);
  }
  // CERTIFIED kernels: value + rigorous bound in one pass. The
  // paper's acceptance (c): certified <= ~2x the plain fp64 kernel at batch.
  {
    auto bench_cert = [&](void (*fn)(const cuDoubleComplex*, int, int,
                                     cuDoubleComplex*, double*, cudaStream_t),
                          const char* name, int matrix_dim, int kernel_dim,
                          bool sym) {
      std::mt19937_64 rng(7777 + matrix_dim);
      std::uniform_real_distribution<double> U(-1.0, 1.0);
      const int D = matrix_dim;
      std::vector<cuDoubleComplex> h((size_t)batch * D * D);
      for (int b = 0; b < batch; ++b) {
        std::vector<double> Gre((size_t)D * D), Gim((size_t)D * D);
        for (size_t i = 0; i < Gre.size(); ++i) { Gre[i] = 0.3 * U(rng); Gim[i] = 0.3 * U(rng); }
        for (int i = 0; i < D; ++i)
          for (int j = 0; j < D; ++j) {
            double re = sym ? 0.5 * (Gre[i * D + j] + Gre[j * D + i]) : Gre[i * D + j];
            double im = sym ? 0.5 * (Gim[i * D + j] + Gim[j * D + i]) : Gim[i * D + j];
            h[(size_t)b * D * D + i * D + j] = make_cuDoubleComplex(re, im);
          }
      }
      cuDoubleComplex *d_in = nullptr, *d_out = nullptr;
      double* d_bnd = nullptr;
      cudaMalloc(&d_in, h.size() * sizeof(cuDoubleComplex));
      cudaMalloc(&d_out, (size_t)batch * sizeof(cuDoubleComplex));
      cudaMalloc(&d_bnd, (size_t)batch * sizeof(double));
      cudaMemcpy(d_in, h.data(), h.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
      fn(d_in, kernel_dim, batch, d_out, d_bnd, 0); // warm-up
      cudaDeviceSynchronize();
      std::vector<double> eps;
      double checksum = 0.0;
      for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        fn(d_in, kernel_dim, batch, d_out, d_bnd, 0);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();
        std::vector<cuDoubleComplex> ho(batch);
        cudaMemcpy(ho.data(), d_out, batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        double sum = 0.0;
        for (auto& z : ho) sum += cuCabs(z);
        checksum = sum;
        eps.push_back(batch / std::chrono::duration<double>(t1 - t0).count());
      }
      std::sort(eps.begin(), eps.end());
      double med = eps[eps.size() / 2];
      double iqr = eps[(3 * eps.size()) / 4] - eps[eps.size() / 4];
      std::printf(
          "{\"func\":\"%s\",\"matrix_dim\":%d,\"batch\":%d,\"repeats\":%d,"
          "\"evals_per_sec_median\":%.6e,\"evals_per_sec_iqr\":%.6e,"
          "\"evals_per_sec_best\":%.6e,\"checksum\":%.12e}\n",
          name, matrix_dim, batch, reps, med, iqr, eps.back(), checksum);
      cudaFree(d_in); cudaFree(d_out); cudaFree(d_bnd);
    };
    for (int n : {8, 12, 16}) bench_cert(gbs::gbs_perm_certified_batched, "perm_certified", n, n, false);
    for (int N : {8, 12, 16}) bench_cert(gbs::gbs_haf_certified_batched, "haf_certified", N, N, true);
    for (int N : {8, 12}) bench_cert(gbs::gbs_lhaf_certified_batched, "lhaf_certified", N, N, true);
    for (int n : {4, 6, 8}) bench_cert(gbs::gbs_tor_certified_batched, "tor_certified", 2 * n, n, true);
    for (int N : {8, 12}) bench_cert(gbs::gbs_haf_dd_certified_batched, "haf_dd_certified", N, N, true);
  }

  // SINGLE-LARGE recursive torontonian: one evaluation split into 2^g subtrees
  // across the grid -- the verification-frontier lever (dims beyond the batched
  // caps). Physical-ish random SPD-domain input
  // (a*I + small symmetric noise keeps I - O_S SPD); one JSON row per n with
  // evals_per_sec = 1/median_seconds (batch=1 by definition).
  for (int n : {16, 20, 24, 26}) {
    const int dim = 2 * n, g = (n >= 20 ? 16 : 12);
    std::mt19937_64 rngs(4321 + n);
    std::uniform_real_distribution<double> Us(-1.0, 1.0);
    std::vector<double> O((size_t)dim * dim);
    std::vector<double> G((size_t)dim * dim);
    for (auto& z : G) z = 0.02 * Us(rngs);
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j)
        O[i * dim + j] = 0.5 * (G[i * dim + j] + G[j * dim + i]) + (i == j ? 0.15 : 0.0);
    double *dO = nullptr, *dp = nullptr;
    cudaMalloc(&dO, O.size() * sizeof(double));
    cudaMalloc(&dp, (1ull << g) * sizeof(double));
    cudaMemcpy(dO, O.data(), O.size() * sizeof(double), cudaMemcpyHostToDevice);
    gbs::gbs_tor_recursive_single_batched(dO, n, g, dp, 0); // warm-up
    cudaDeviceSynchronize();
    std::vector<double> secs;
    double checksum = 0.0;
    for (int r = 0; r < reps; ++r) {
      auto t0 = std::chrono::steady_clock::now(); // chrono + sync: shim-portable
      gbs::gbs_tor_recursive_single_batched(dO, n, g, dp, 0);
      cudaDeviceSynchronize();
      auto t1 = std::chrono::steady_clock::now();
      std::vector<double> hp(1ull << g);
      cudaMemcpy(hp.data(), dp, hp.size() * sizeof(double), cudaMemcpyDeviceToHost);
      double sum = 0.0;
      for (double x : hp) sum += x;
      checksum = sum; // post-timing honesty checksum
      secs.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::sort(secs.begin(), secs.end());
    double med = secs[secs.size() / 2];
    double iqr = secs[(3 * secs.size()) / 4] - secs[secs.size() / 4];
    std::printf(
        "{\"func\":\"tor_single\",\"matrix_dim\":%d,\"batch\":1,\"repeats\":%d,"
        "\"evals_per_sec_median\":%.6e,\"evals_per_sec_iqr\":%.6e,"
        "\"evals_per_sec_best\":%.6e,\"checksum\":%.12e}\n",
        dim, reps, 1.0 / med, iqr > 0 ? (1.0 / med - 1.0 / (med + iqr)) : 0.0,
        1.0 / secs.front(), checksum);
    cudaFree(dO); cudaFree(dp);
  }
  for (int n : {4, 6, 8}) // DD torontonian
    bench_one(gbs::gbs_tor_dd_batched, "tor_dd", 2 * n, n, batch, reps, true, 0.05);
  return 0;
}
