// check_torontonian_real_chol.cu -- differential gate for the real-Cholesky
// torontonian (real-Cholesky, physical SPD domain).
//
// On real O the real-Cholesky value MUST equal the complex-LU kernel, which is
// already validated against cpu_ref/torontonian.py + The Walrus + mpmath by
// check_torontonian.cu. So this gate confirms the cheaper real-arithmetic path is
// bit-close on the physical domain (the same transitive-validation logic the varn
// hafnian uses against the single-size hafnian). PASS: max rel err below tol.
//
//   nvcc -O3 -std=c++17 torontonian.cu check_torontonian_real_chol.cu -o check_tor_real
//
// Robust across platforms BY CONSTRUCTION: O = -a*(R R^T) makes I - O_S = I + a*(R R^T)_S
// SPD with every eigenvalue >= 1 (a principal submatrix of a PSD matrix is PSD), so there are
// NO near-singular submatrices on which a real Cholesky and a complex LU could diverge; and the
// RNG is the PORTABLE raw 64-bit engine, not std::normal/uniform_distribution. Those two
// fragilities -- near-singular draws + non-portable std::distributions (libc++ vs libstdc++) --
// are exactly what made an earlier normal-draw version of this cross-algorithm gate pass on
// macOS but FAIL in Linux CI; both are removed here (see the differential-gate-conditioning
// note). n <= 5 just keeps the 2^n subset enumeration cheap.

#include <cuComplex.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace gbs {
extern "C" void gbs_tor_fp64_batched(const cuDoubleComplex*, int, int,
                                     cuDoubleComplex*, cudaStream_t);
extern "C" void gbs_tor_real_chol_fp64_batched(const double*, int, int,
                                               double*, cudaStream_t);
}

int main() {
  std::mt19937_64 rng(11);
  // Portable uniform in [-1, 1) straight from the raw 64-bit engine (53-bit mantissa) -- no
  // std::*_distribution, whose output differs across standard libraries.
  auto uni = [&]() { return (double)(rng() >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0; };
  double worst = 0.0;
  for (int n = 1; n <= 5; ++n) {
    const int batch = 128, N = 2 * n;
    std::vector<double> Oreal(batch * N * N);
    for (int b = 0; b < batch; ++b) {
      // O = -a*(R R^T): real symmetric, and I - O_S = I + a*(R R^T)_S is SPD with eigenvalues
      // >= 1 -- well-conditioned on every draw/platform (see the header note).
      std::vector<double> R(N * N);
      for (auto& x : R) x = uni();
      double* O = Oreal.data() + (size_t)b * N * N;
      const double a = 0.05;
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
          double g = 0.0;
          for (int k = 0; k < N; ++k) g += R[i * N + k] * R[j * N + k];   // (R R^T)_{ij}
          O[i * N + j] = -a * g;
        }
    }
    // complex-LU reference (the validated kernel) fed the same O with imag = 0.
    std::vector<cuDoubleComplex> cin(Oreal.size());
    for (size_t i = 0; i < Oreal.size(); ++i) cin[i] = make_cuDoubleComplex(Oreal[i], 0.0);
    cuDoubleComplex *d_cin = nullptr, *d_cout = nullptr;
    cudaMalloc(&d_cin, cin.size() * sizeof(cuDoubleComplex));
    cudaMalloc(&d_cout, batch * sizeof(cuDoubleComplex));
    cudaMemcpy(d_cin, cin.data(), cin.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    gbs::gbs_tor_fp64_batched(d_cin, n, batch, d_cout, 0);
    cudaDeviceSynchronize();
    std::vector<cuDoubleComplex> cout(batch);
    cudaMemcpy(cout.data(), d_cout, batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    // real-Cholesky kernel on the real O.
    double *d_rin = nullptr, *d_rout = nullptr;
    cudaMalloc(&d_rin, Oreal.size() * sizeof(double));
    cudaMalloc(&d_rout, batch * sizeof(double));
    cudaMemcpy(d_rin, Oreal.data(), Oreal.size() * sizeof(double), cudaMemcpyHostToDevice);
    gbs::gbs_tor_real_chol_fp64_batched(d_rin, n, batch, d_rout, 0);
    cudaDeviceSynchronize();
    std::vector<double> rout(batch);
    cudaMemcpy(rout.data(), d_rout, batch * sizeof(double), cudaMemcpyDeviceToHost);
    for (int b = 0; b < batch; ++b) {
      double ref = cuCreal(cout[b]);                 // imag ~ 1e-16 on real input
      worst = std::max(worst, std::fabs(rout[b] - ref) / std::max(std::fabs(ref), 1e-300));
    }
    cudaFree(d_cin); cudaFree(d_cout); cudaFree(d_rin); cudaFree(d_rout);
  }
  // I - O_S is SPD with eigenvalues >= 1 by construction, so the real Cholesky and the complex
  // LU agree to ~1e-13 on every draw and platform; the 1e-8 FP64 tier is a wide safety margin.
  const double tol = 1e-8;
  std::printf("max rel err real-Cholesky vs complex-LU (n=1..5, real O): %.3e  [tol %.1e]\n",
              worst, tol);
  std::printf("%s\n", worst < tol ? "PASS" : "FAIL");
  return worst < tol ? 0 : 1;
}
