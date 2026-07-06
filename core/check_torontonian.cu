// check_torontonian.cu -- GPU-vs-CPU-reference differential gate (torontonian).
//
// Run in a rented-GPU session before timing. Compares the GPU torontonian
// against an independent host reference (mirrors cpu_ref/torontonian.py, itself
// validated on CPU against the n=1 closed form, multiplicativity, mpmath, and
// The Walrus on real O). Uses real O of small norm (the physical, unambiguous
// domain). PASS gate: max rel err below the FP64 tier tolerance.
//
//   nvcc -O3 -std=c++17 torontonian.cu check_torontonian.cu -o check_torontonian

#include <cuComplex.h>

#include <complex>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

namespace gbs {
extern "C" void gbs_tor_fp64_batched(const cuDoubleComplex*, int, int,
                                     cuDoubleComplex*, cudaStream_t);
}
using cd = std::complex<double>;

static cd host_det(std::vector<cd> M, int mm) {
  cd det(1, 0);
  for (int c = 0; c < mm; ++c) {
    int piv = c; double best = std::abs(M[c * mm + c]);
    for (int r = c + 1; r < mm; ++r) { double v = std::abs(M[r * mm + c]); if (v > best) { best = v; piv = r; } }
    if (best == 0) return cd(0, 0);
    if (piv != c) { for (int j = 0; j < mm; ++j) std::swap(M[c * mm + j], M[piv * mm + j]); det = -det; }
    cd p = M[c * mm + c]; det *= p;
    for (int r = c + 1; r < mm; ++r) { cd f = M[r * mm + c] / p; for (int j = c; j < mm; ++j) M[r * mm + j] -= f * M[c * mm + j]; }
  }
  return det;
}

static cd host_tor(const cd* O, int n) {
  int N = 2 * n; cd total(0, 0);
  for (uint64_t mask = 0; mask < (1ull << n); ++mask) {
    int m = __builtin_popcountll(mask);
    cd term;
    if (m == 0) term = cd(1, 0);
    else {
      std::vector<int> idx;
      for (int i = 0; i < n; ++i) if ((mask >> i) & 1ull) idx.push_back(i);
      for (int i = 0; i < n; ++i) if ((mask >> i) & 1ull) idx.push_back(i + n);
      int mm = 2 * m; std::vector<cd> sub(mm * mm);
      for (int r = 0; r < mm; ++r) for (int c = 0; c < mm; ++c)
        sub[r * mm + c] = (r == c ? cd(1, 0) : cd(0, 0)) - O[idx[r] * N + idx[c]];
      term = cd(1, 0) / std::sqrt(host_det(sub, mm));
    }
    total += ((n - m) & 1) ? -term : term;
  }
  return total;
}

int main() {
  std::mt19937_64 rng(7);
  std::normal_distribution<double> Z(0.0, 1.0);
  double worst = 0.0;
  // n <= 5 keeps every I - O_S reliably well-conditioned across platforms (the
  // gate compares the kernel's cuComplex determinant against an independent
  // std::complex one, which diverge ~1e-7 on a near-singular submatrix -- those
  // appear among the random draws at larger n). Larger-n GPU-vs-CPU agreement is
  // covered by bench/throughput_end_to_end.py (vs numpy, ~1e-11).
  for (int n = 1; n <= 5; ++n) {
    const int batch = 128, N = 2 * n;
    std::vector<cd> host(batch * N * N);
    for (int b = 0; b < batch; ++b) {
      std::vector<double> M(N * N);
      for (auto& x : M) x = 0.12 * Z(rng);
      for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j)
        host[b * N * N + i * N + j] = cd(0.5 * (M[i * N + j] + M[j * N + i]), 0.0);
    }
    std::vector<cuDoubleComplex> hin(batch * N * N);
    for (size_t i = 0; i < host.size(); ++i) hin[i] = make_cuDoubleComplex(host[i].real(), host[i].imag());
    cuDoubleComplex *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, hin.size() * sizeof(cuDoubleComplex));
    cudaMalloc(&d_out, batch * sizeof(cuDoubleComplex));
    cudaMemcpy(d_in, hin.data(), hin.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    gbs::gbs_tor_fp64_batched(d_in, n, batch, d_out, 0);
    cudaDeviceSynchronize();
    std::vector<cuDoubleComplex> hout(batch);
    cudaMemcpy(hout.data(), d_out, batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    for (int b = 0; b < batch; ++b) {
      cd ref = host_tor(host.data() + (size_t)b * N * N, n);
      cd got{cuCreal(hout[b]), cuCimag(hout[b])};
      worst = std::max(worst, std::abs(got - ref) / std::max(std::abs(ref), 1e-300));
    }
    cudaFree(d_in); cudaFree(d_out);
  }
  const double tol = 1e-8;
  std::printf("max relative error (n=1..5 modes, real O): %.3e  [tol %.1e]\n", worst, tol);
  std::printf("%s\n", worst < tol ? "PASS" : "FAIL");
  return worst < tol ? 0 : 1;
}
