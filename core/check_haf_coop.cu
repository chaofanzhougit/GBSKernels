// check_haf_coop.cu -- differential gate for the COOPERATIVE hafnian.
//
// The cooperative map/reduce hafnian (groups split the 2^(N/2) subset sum) must
// equal the per-thread hafnian, which is itself validated against an independent
// host reference (check_hafnian.cu) on-device. Both call the same haf_subset_term,
// so this directly tests the regrouping: groups=1 is the same order (bit-exact),
// groups>1 regroups (FP64 tolerance). Runs on the CPU host shim.
//
//   nvcc -O3 -std=c++17 hafnian.cu check_haf_coop.cu -o check_haf_coop

#include <cuComplex.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace gbs {
extern "C" void gbs_haf_powertrace_fp64_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
extern "C" int gbs_haf_coop_batched(const cuDoubleComplex*, int, int, int, cuDoubleComplex*, cudaStream_t);
}

int main() {
  std::mt19937_64 rng(2024);
  std::normal_distribution<double> Z(0.0, 1.0);
  const int group_widths[] = {1, 8, 32};
  const int batch = 64;
  double worst = 0.0;

  for (int N = 2; N <= 12; N += 2) {  // even sizes (haf of odd = 0)
    std::vector<cuDoubleComplex> hin((size_t)batch * N * N);
    for (int b = 0; b < batch; ++b)
      for (int i = 0; i < N; ++i)
        for (int j = i; j < N; ++j) {  // symmetric
          cuDoubleComplex v = make_cuDoubleComplex(Z(rng), Z(rng));
          hin[((size_t)b * N + i) * N + j] = v;
          hin[((size_t)b * N + j) * N + i] = v;
        }
    cuDoubleComplex *d_in = nullptr, *d_ref = nullptr, *d_coop = nullptr;
    cudaMalloc(&d_in, hin.size() * sizeof(cuDoubleComplex));
    cudaMalloc(&d_ref, batch * sizeof(cuDoubleComplex));
    cudaMalloc(&d_coop, batch * sizeof(cuDoubleComplex));
    cudaMemcpy(d_in, hin.data(), hin.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    gbs::gbs_haf_powertrace_fp64_batched(d_in, N, batch, d_ref, 0);
    cudaDeviceSynchronize();
    std::vector<cuDoubleComplex> ref(batch);
    cudaMemcpy(ref.data(), d_ref, batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    for (int gw : group_widths) {
      gbs::gbs_haf_coop_batched(d_in, N, batch, gw, d_coop, 0);
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
