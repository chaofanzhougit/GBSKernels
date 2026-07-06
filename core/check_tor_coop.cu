// check_tor_coop.cu -- differential gate for the COOPERATIVE torontonian.
//
// The cooperative map/reduce torontonian must equal the per-thread torontonian
// (validated independently by check_torontonian.cu). Both call the same
// tor_subset_term (same det_lu), so this tests only the regrouping -- conditioning
// is irrelevant here (unlike the cross-implementation check_torontonian gate).
// Physical real small-norm O keeps the terms O(1). Runs on the CPU host shim.
//
//   nvcc -O3 -std=c++17 torontonian.cu check_tor_coop.cu -o check_tor_coop

#include <cuComplex.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace gbs {
extern "C" void gbs_tor_fp64_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
extern "C" int gbs_tor_coop_batched(const cuDoubleComplex*, int, int, int, cuDoubleComplex*, cudaStream_t);
}

int main() {
  std::mt19937_64 rng(7);
  std::normal_distribution<double> Z(0.0, 1.0);
  const int group_widths[] = {1, 8, 32};
  const int batch = 64;
  double worst = 0.0;

  for (int n = 1; n <= 6; ++n) {  // modes; matrices are 2n x 2n
    const int N = 2 * n;
    std::vector<cuDoubleComplex> hin((size_t)batch * N * N);
    for (int b = 0; b < batch; ++b) {
      std::vector<double> M(N * N);
      for (auto& x : M) x = 0.10 * Z(rng);  // physical real O, small norm
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
          hin[((size_t)b * N + i) * N + j] =
              make_cuDoubleComplex(0.5 * (M[i * N + j] + M[j * N + i]), 0.0);  // symmetric, real
    }
    cuDoubleComplex *d_in = nullptr, *d_ref = nullptr, *d_coop = nullptr;
    cudaMalloc(&d_in, hin.size() * sizeof(cuDoubleComplex));
    cudaMalloc(&d_ref, batch * sizeof(cuDoubleComplex));
    cudaMalloc(&d_coop, batch * sizeof(cuDoubleComplex));
    cudaMemcpy(d_in, hin.data(), hin.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    gbs::gbs_tor_fp64_batched(d_in, n, batch, d_ref, 0);
    cudaDeviceSynchronize();
    std::vector<cuDoubleComplex> ref(batch);
    cudaMemcpy(ref.data(), d_ref, batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    for (int gw : group_widths) {
      gbs::gbs_tor_coop_batched(d_in, n, batch, gw, d_coop, 0);
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
  std::printf("max relative error (n=1..6 modes, real O, groups{1,8,32}): %.3e  [tol %.1e]\n", worst, tol);
  std::printf("%s\n", worst < tol ? "PASS" : "FAIL");
  return worst < tol ? 0 : 1;
}
