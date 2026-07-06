// check_permanent_coop.cu -- differential gate for the COOPERATIVE permanent.
//
// Validates that the warp/block-cooperative map/reduce permanent (permanent_coop.cu)
// equals an independent host Glynn permanent (the same reference permanent.cu's gate
// uses, itself validated vs combinatorics + The Walrus on CPU) across a range of n
// AND of cooperation widths `groups`. groups>1 regroups the Glynn sum, so agreement
// is to FP64 tolerance (different summation associativity), not bit-exact. Runs on
// the CPU host shim before any GPU session.
//
//   nvcc -O3 -std=c++17 permanent_coop.cu check_permanent_coop.cu -o check_permanent_coop

#include <cuComplex.h>

#include <complex>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace gbs {
extern "C" int gbs_perm_glynn_coop_batched(const cuDoubleComplex*, int, int, int,
                                           cuDoubleComplex*, cudaStream_t);
}

// Independent host Glynn permanent (mirrors cpu_ref/permanent.py; identical to the
// reference in check_permanent.cu).
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
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  const int group_widths[] = {1, 8, 32};

  double worst = 0.0;
  for (int n = 1; n <= 12; ++n) {
    const int batch = 64;
    std::vector<std::complex<double>> host(batch * n * n);
    for (auto& z : host) z = {U(rng), U(rng)};
    std::vector<cuDoubleComplex> hin(host.size());
    for (size_t i = 0; i < host.size(); ++i)
      hin[i] = make_cuDoubleComplex(host[i].real(), host[i].imag());

    std::vector<std::complex<double>> ref(batch);
    for (int b = 0; b < batch; ++b)
      ref[b] = host_perm_glynn(host.data() + (size_t)b * n * n, n);

    cuDoubleComplex *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, hin.size() * sizeof(cuDoubleComplex));
    cudaMalloc(&d_out, batch * sizeof(cuDoubleComplex));
    cudaMemcpy(d_in, hin.data(), hin.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    for (int gw : group_widths) {
      gbs::gbs_perm_glynn_coop_batched(d_in, n, batch, gw, d_out, 0);
      cudaDeviceSynchronize();
      std::vector<cuDoubleComplex> hout(batch);
      cudaMemcpy(hout.data(), d_out, batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
      for (int b = 0; b < batch; ++b) {
        std::complex<double> got{cuCreal(hout[b]), cuCimag(hout[b])};
        worst = std::max(worst, std::abs(got - ref[b]) / std::max(std::abs(ref[b]), 1e-300));
      }
    }
    cudaFree(d_in);
    cudaFree(d_out);
  }

  const double tol = 1e-8;
  std::printf("max relative error (n=1..12, groups{1,8,32}, complex): %.3e  [tol %.1e]\n", worst, tol);
  std::printf("%s\n", worst < tol ? "PASS" : "FAIL");
  return worst < tol ? 0 : 1;
}
