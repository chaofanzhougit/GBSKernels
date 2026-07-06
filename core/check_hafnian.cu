// check_hafnian.cu -- GPU-vs-CPU-reference differential gate for the hafnian.
//
// Run in a rented-GPU session before any timing: compares the GPU power-trace
// hafnian against an independent host power-trace reference (mirroring
// cpu_ref/hafnian.py, itself validated against combinatorial truth + The Walrus
// on CPU) over random complex symmetric matrices. PASS gate: max rel err below
// the FP64 tier tolerance.
//
//   nvcc -O3 -std=c++17 hafnian.cu check_hafnian.cu -o check_hafnian && ./check_hafnian

#include <cuComplex.h>

#include <complex>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

namespace gbs {
extern "C" void gbs_haf_powertrace_fp64_batched(const cuDoubleComplex*, int, int,
                                                cuDoubleComplex*, cudaStream_t);
}

using cd = std::complex<double>;

// Independent host power-trace hafnian (mirrors cpu_ref/hafnian.py).
static cd host_exp_newton(const std::vector<cd>& C, int size, int n) {
  if (size == 0) return (n == 0) ? cd(1, 0) : cd(0, 0);
  std::vector<cd> p(n + 1), P = C, T(size * size);
  for (int k = 1; k <= n; ++k) {
    cd tr(0, 0);
    for (int i = 0; i < size; ++i) tr += P[i * size + i];
    p[k] = tr;
    if (k < n) {
      for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j) {
          cd s(0, 0);
          for (int t = 0; t < size; ++t) s += P[i * size + t] * C[t * size + j];
          T[i * size + j] = s;
        }
      P = T;
    }
  }
  std::vector<cd> e(n + 1, cd(0, 0));
  e[0] = cd(1, 0);
  for (int j = 1; j <= n; ++j) {
    cd acc(0, 0);
    for (int k = 1; k <= j; ++k) acc += (p[k] * 0.5) * e[j - k];
    e[j] = acc / (double)j;
  }
  return e[n];
}

static cd host_haf(const cd* A, int N) {
  if (N == 0) return cd(1, 0);
  if (N & 1) return cd(0, 0);
  int n = N / 2;
  cd total(0, 0);
  for (uint64_t mask = 0; mask < (1ull << n); ++mask) {
    int m = __builtin_popcountll(mask), size = 2 * m;
    std::vector<int> pidx;
    for (int i = 0; i < n; ++i) if ((mask >> i) & 1ull) pidx.push_back(i);
    std::vector<cd> BX(size * size);
    for (int r = 0; r < size; ++r) {
      int rr = 2 * pidx[r >> 1] + (r & 1);
      for (int c = 0; c < size; ++c) {
        int cc = 2 * pidx[c >> 1] + ((c & 1) ^ 1);
        BX[r * size + c] = A[rr * N + cc];
      }
    }
    cd coeff = host_exp_newton(BX, size, n);
    total += ((n - m) & 1) ? -coeff : coeff;
  }
  return total;
}

int main() {
  std::mt19937_64 rng(2024);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  double worst = 0.0;
  for (int N = 2; N <= 12; N += 2) {
    const int batch = 128;
    std::vector<cd> host(batch * N * N);
    for (int b = 0; b < batch; ++b) {
      std::vector<cd> G(N * N);
      for (auto& z : G) z = {U(rng), U(rng)};
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) host[b * N * N + i * N + j] = G[i * N + j] + G[j * N + i];
    }
    std::vector<cuDoubleComplex> hin(batch * N * N);
    for (size_t i = 0; i < host.size(); ++i)
      hin[i] = make_cuDoubleComplex(host[i].real(), host[i].imag());

    cuDoubleComplex *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, hin.size() * sizeof(cuDoubleComplex));
    cudaMalloc(&d_out, batch * sizeof(cuDoubleComplex));
    cudaMemcpy(d_in, hin.data(), hin.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    gbs::gbs_haf_powertrace_fp64_batched(d_in, N, batch, d_out, 0);
    cudaDeviceSynchronize();
    std::vector<cuDoubleComplex> hout(batch);
    cudaMemcpy(hout.data(), d_out, batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    for (int b = 0; b < batch; ++b) {
      cd ref = host_haf(host.data() + (size_t)b * N * N, N);
      cd got{cuCreal(hout[b]), cuCimag(hout[b])};
      worst = std::max(worst, std::abs(got - ref) / std::max(std::abs(ref), 1e-300));
    }
    cudaFree(d_in);
    cudaFree(d_out);
  }
  const double tol = 1e-8;
  std::printf("max relative error (N=2..12, complex symmetric): %.3e  [tol %.1e]\n", worst, tol);
  std::printf("%s\n", worst < tol ? "PASS" : "FAIL");
  return worst < tol ? 0 : 1;
}
