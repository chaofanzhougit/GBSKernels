// check_haf_small.cu -- differential gate for the SIZE-SPECIALIZED hafnian.
//
// The small-buffer-cap hafnian (gbs_haf_powertrace_fp64_small_batched, the perf-
// research candidate) must produce the SAME result as the full-cap per-thread kernel
// (itself validated vs an independent reference by check_hafnian.cu) for every N up to
// HAF_SMALL_N. Size specialization changes only the local-buffer footprint, never the
// value. Runs on the CPU host shim. The throughput A/B (small vs full at equal N) is
// measured on a device in bench_kernels.cu.
//
//   nvcc -O3 -std=c++17 hafnian.cu check_haf_small.cu -o check_haf_small

#include <cuComplex.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace gbs {
extern "C" void gbs_haf_powertrace_fp64_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
extern "C" void gbs_haf_powertrace_fp64_small_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
}

int main() {
  std::mt19937_64 rng(2024);
  std::normal_distribution<double> Z(0.0, 1.0);
  const int batch = 64;
  double worst = 0.0;

  for (int N = 2; N <= 12; N += 2) {  // even, within HAF_SMALL_N
    std::vector<cuDoubleComplex> hin((size_t)batch * N * N);
    for (int b = 0; b < batch; ++b)
      for (int i = 0; i < N; ++i)
        for (int j = i; j < N; ++j) {  // symmetric
          cuDoubleComplex v = make_cuDoubleComplex(Z(rng), Z(rng));
          hin[((size_t)b * N + i) * N + j] = v;
          hin[((size_t)b * N + j) * N + i] = v;
        }
    cuDoubleComplex *d_in = nullptr, *d_full = nullptr, *d_small = nullptr;
    cudaMalloc(&d_in, hin.size() * sizeof(cuDoubleComplex));
    cudaMalloc(&d_full, batch * sizeof(cuDoubleComplex));
    cudaMalloc(&d_small, batch * sizeof(cuDoubleComplex));
    cudaMemcpy(d_in, hin.data(), hin.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    gbs::gbs_haf_powertrace_fp64_batched(d_in, N, batch, d_full, 0);
    gbs::gbs_haf_powertrace_fp64_small_batched(d_in, N, batch, d_small, 0);
    cudaDeviceSynchronize();
    std::vector<cuDoubleComplex> full(batch), small(batch);
    cudaMemcpy(full.data(), d_full, batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(small.data(), d_small, batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    for (int b = 0; b < batch; ++b) {
      double dr = cuCreal(small[b]) - cuCreal(full[b]), di = cuCimag(small[b]) - cuCimag(full[b]);
      worst = std::max(worst, std::sqrt(dr * dr + di * di) / std::max(cuCabs(full[b]), 1e-300));
    }
    cudaFree(d_in); cudaFree(d_full); cudaFree(d_small);
  }

  const double tol = 1e-12;  // same algorithm + arithmetic -> bit-for-bit (only the buffer cap differs)
  std::printf("max relative error (N=2..12, small-cap vs full-cap hafnian): %.3e  [tol %.1e]\n", worst, tol);
  std::printf("%s\n", worst < tol ? "PASS" : "FAIL");
  return worst < tol ? 0 : 1;
}
