// check_lhaf_coop.cu -- differential gate for the COOPERATIVE loop hafnian.
//
// The cooperative map/reduce loop hafnian must equal the per-thread loop hafnian
// (validated independently by check_loop_hafnian.cu). Both call the same
// lhaf_subset_term, so this tests the regrouping (groups=1 bit-exact, groups>1 to
// FP64 tolerance). Runs on the CPU host shim.
//
//   nvcc -O3 -std=c++17 loop_hafnian.cu check_lhaf_coop.cu -o check_lhaf_coop

#include <cuComplex.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace gbs {
extern "C" void gbs_loop_haf_fp64_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
extern "C" int gbs_loop_haf_coop_batched(const cuDoubleComplex*, int, int, int, cuDoubleComplex*, cudaStream_t);
}

int main() {
  std::mt19937_64 rng(99);
  std::normal_distribution<double> Z(0.0, 1.0);
  const int group_widths[] = {1, 8, 32};
  const int batch = 64;
  double worst = 0.0;

  for (int N = 2; N <= 12; N += 2) {  // even sizes (the power-trace path)
    std::vector<cuDoubleComplex> hin((size_t)batch * N * N);
    for (int b = 0; b < batch; ++b)
      for (int i = 0; i < N; ++i)
        for (int j = i; j < N; ++j) {  // symmetric (diagonal carries the loop weights)
          cuDoubleComplex v = make_cuDoubleComplex(Z(rng), Z(rng));
          hin[((size_t)b * N + i) * N + j] = v;
          hin[((size_t)b * N + j) * N + i] = v;
        }
    cuDoubleComplex *d_in = nullptr, *d_ref = nullptr, *d_coop = nullptr;
    cudaMalloc(&d_in, hin.size() * sizeof(cuDoubleComplex));
    cudaMalloc(&d_ref, batch * sizeof(cuDoubleComplex));
    cudaMalloc(&d_coop, batch * sizeof(cuDoubleComplex));
    cudaMemcpy(d_in, hin.data(), hin.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    gbs::gbs_loop_haf_fp64_batched(d_in, N, batch, d_ref, 0);
    cudaDeviceSynchronize();
    std::vector<cuDoubleComplex> ref(batch);
    cudaMemcpy(ref.data(), d_ref, batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    for (int gw : group_widths) {
      gbs::gbs_loop_haf_coop_batched(d_in, N, batch, gw, d_coop, 0);
      cudaDeviceSynchronize();
      std::vector<cuDoubleComplex> got(batch);
      cudaMemcpy(got.data(), d_coop, batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
      for (int b = 0; b < batch; ++b) {
        double dr = cuCreal(got[b]) - cuCreal(ref[b]), di = cuCimag(got[b]) - cuCimag(ref[b]);
        worst = std::max(worst, std::sqrt(dr * dr + di * di) / std::max(cuCabs(ref[b]), 1e-300));
      }
    }
    cudaFree(d_in); cudaFree(d_ref); cudaFree(d_coop);
  }

  const double tol = 1e-8;
  std::printf("max relative error (N=2..12 even, groups{1,8,32}): %.3e  [tol %.1e]\n", worst, tol);
  std::printf("%s\n", worst < tol ? "PASS" : "FAIL");
  return worst < tol ? 0 : 1;
}
