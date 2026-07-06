// check_permanent.cu -- GPU-vs-CPU-reference differential check (docs/DESIGN.md §8 L5).
//
// Run this FIRST in any rented-GPU session, before timing anything: it compares
// the GPU Glynn kernel against an independent host (CPU) Glynn over random
// real/complex matrices and reports the max relative error. The host reference
// here mirrors cpu_ref/permanent.py, which is itself validated against
// combinatorial ground truth and The Walrus on CPU. PASS gate: max rel err below
// the FP64 tier tolerance.
//
//   nvcc -O3 -std=c++17 permanent.cu check_permanent.cu -o check_permanent
//   ./check_permanent

#include <cuComplex.h>

#include <complex>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

namespace gbs {
extern "C" void gbs_perm_glynn_fp64_batched(const cuDoubleComplex*, int, int,
                                            cuDoubleComplex*, cudaStream_t);
}

// Independent host Glynn permanent (mirrors cpu_ref/permanent.py).
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

  double worst = 0.0;
  for (int n = 1; n <= 12; ++n) {
    const int batch = 256;
    std::vector<std::complex<double>> host(batch * n * n);
    for (auto& z : host) z = {U(rng), U(rng)};

    std::vector<cuDoubleComplex> hin(batch * n * n);
    for (size_t i = 0; i < host.size(); ++i)
      hin[i] = make_cuDoubleComplex(host[i].real(), host[i].imag());

    cuDoubleComplex *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, hin.size() * sizeof(cuDoubleComplex));
    cudaMalloc(&d_out, batch * sizeof(cuDoubleComplex));
    cudaMemcpy(d_in, hin.data(), hin.size() * sizeof(cuDoubleComplex),
               cudaMemcpyHostToDevice);

    gbs::gbs_perm_glynn_fp64_batched(d_in, n, batch, d_out, 0);
    cudaDeviceSynchronize();

    std::vector<cuDoubleComplex> hout(batch);
    cudaMemcpy(hout.data(), d_out, batch * sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToHost);

    for (int b = 0; b < batch; ++b) {
      std::complex<double> ref = host_perm_glynn(host.data() + (size_t)b * n * n, n);
      std::complex<double> got{cuCreal(hout[b]), cuCimag(hout[b])};
      double rel = std::abs(got - ref) / std::max(std::abs(ref), 1e-300);
      worst = std::max(worst, rel);
    }
    cudaFree(d_in);
    cudaFree(d_out);
  }

  const double tol = 1e-8;
  std::printf("max relative error (n=1..12, complex): %.3e  [tol %.1e]\n", worst, tol);
  std::printf("%s\n", worst < tol ? "PASS" : "FAIL");
  return worst < tol ? 0 : 1;
}
