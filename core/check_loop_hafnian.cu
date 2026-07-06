// check_loop_hafnian.cu -- GPU-vs-CPU-reference differential gate (loop hafnian).
//
// Run in a rented-GPU session before timing. Compares the GPU loop hafnian
// against an independent host reference (mirrors cpu_ref/loop_hafnian.py, itself
// validated on CPU against naive enumeration, The Walrus loop=True, telephone
// numbers, and the diagonal-zero -> hafnian reduction). Even N only (the GPU
// power-trace path). PASS gate: max rel err below the FP64 tier tolerance.
//
//   nvcc -O3 -std=c++17 loop_hafnian.cu check_loop_hafnian.cu -o check_loop_hafnian

#include <cuComplex.h>

#include <complex>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

namespace gbs {
extern "C" void gbs_loop_haf_fp64_batched(const cuDoubleComplex*, int, int,
                                          cuDoubleComplex*, cudaStream_t);
}
using cd = std::complex<double>;

// Independent host loop hafnian: naive recursive sum over loop matchings.
static cd host_lhaf(const std::vector<cd>& A, int N, std::vector<int> rem) {
  if (rem.empty()) return cd(1, 0);
  int i = rem[0];
  std::vector<int> rest(rem.begin() + 1, rem.end());
  cd total = A[i * N + i] * host_lhaf(A, N, rest);  // i as a loop
  for (size_t k = 0; k < rest.size(); ++k) {
    int j = rest[k];
    std::vector<int> r2 = rest; r2.erase(r2.begin() + k);
    total += A[i * N + j] * host_lhaf(A, N, r2);
  }
  return total;
}

int main() {
  std::mt19937_64 rng(99);
  std::normal_distribution<double> Z(0.0, 1.0);
  double worst = 0.0;
  for (int N = 2; N <= 10; N += 2) {
    const int batch = 64;
    std::vector<cd> host(batch * N * N);
    for (int b = 0; b < batch; ++b) {
      std::vector<cd> G(N * N);
      for (auto& z : G) z = {Z(rng), Z(rng)};
      for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j)
        host[b * N * N + i * N + j] = G[i * N + j] + G[j * N + i];  // symmetric
    }
    std::vector<cuDoubleComplex> hin(batch * N * N);
    for (size_t i = 0; i < host.size(); ++i) hin[i] = make_cuDoubleComplex(host[i].real(), host[i].imag());
    cuDoubleComplex *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, hin.size() * sizeof(cuDoubleComplex));
    cudaMalloc(&d_out, batch * sizeof(cuDoubleComplex));
    cudaMemcpy(d_in, hin.data(), hin.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    gbs::gbs_loop_haf_fp64_batched(d_in, N, batch, d_out, 0);
    cudaDeviceSynchronize();
    std::vector<cuDoubleComplex> hout(batch);
    cudaMemcpy(hout.data(), d_out, batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    std::vector<int> all(N);
    for (int i = 0; i < N; ++i) all[i] = i;
    for (int b = 0; b < batch; ++b) {
      std::vector<cd> Ab(host.begin() + (size_t)b * N * N, host.begin() + (size_t)(b + 1) * N * N);
      cd ref = host_lhaf(Ab, N, all);
      cd got{cuCreal(hout[b]), cuCimag(hout[b])};
      worst = std::max(worst, std::abs(got - ref) / std::max(std::abs(ref), 1e-300));
    }
    cudaFree(d_in); cudaFree(d_out);
  }
  const double tol = 1e-8;
  std::printf("max relative error (N=2..10, complex symmetric): %.3e  [tol %.1e]\n", worst, tol);
  std::printf("%s\n", worst < tol ? "PASS" : "FAIL");
  return worst < tol ? 0 : 1;
}
