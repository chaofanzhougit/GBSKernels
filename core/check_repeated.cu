// check_repeated.cu -- gate for the repeated-row loop-hafnian sieve kernel (R4).
//
// Two independent references, both in long double:
//   (1) the naive expanded loop hafnian (literal recursion over loop matchings
//       of the expanded matrix) for small N -- algorithmic ground truth that
//       shares nothing with the sieve;
//   (2) the host sieve itself -- an implementation cross-check at larger N.
// PASS: kernel vs (1) <= 1e-10 rel on every small case AND kernel vs (2)
// <= 1e-11 rel on every case.

#include <cuComplex.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace gbs {
extern "C" void gbs_lhaf_repeated_cert_batched(const cuDoubleComplex*, const cuDoubleComplex*,
                                               int, const int*, int, cuDoubleComplex*, double*,
                                               cudaStream_t);
extern "C" void gbs_lhaf_repeated_batched(const cuDoubleComplex*, const cuDoubleComplex*,
                                          int, const int*, int, cuDoubleComplex*,
                                          cudaStream_t);
}

using cld = std::complex<long double>;

// (1) naive expanded loop hafnian, long double.
static cld naive_lhaf(const std::vector<cld>& E, int N) {
  if (N == 0) return {1.0L, 0.0L};
  std::vector<int> rem(N);
  for (int i = 0; i < N; ++i) rem[i] = i;
  struct Rec {
    const std::vector<cld>& E;
    int N;
    cld go(std::vector<int>& r) {
      if (r.empty()) return {1.0L, 0.0L};
      int i = r[0];
      std::vector<int> rest(r.begin() + 1, r.end());
      cld total = E[i * N + i] * go(rest);          // i as a loop
      for (size_t k = 0; k < rest.size(); ++k) {
        int j = rest[k];
        std::vector<int> rest2;
        for (size_t t = 0; t < rest.size(); ++t)
          if (t != k) rest2.push_back(rest[t]);
        total += E[i * N + j] * go(rest2);
      }
      return total;
    }
  } rec{E, N};
  return rec.go(rem);
}

// (2) host sieve, long double (same identities as the kernel).
static cld host_sieve(const std::vector<cld>& A, const std::vector<cld>& g, int M,
                      const std::vector<int>& n) {
  int N = 0;
  for (int i = 0; i < M; ++i) N += n[i];
  if (N == 0) return {1.0L, 0.0L};
  int kmax = N / 2;
  std::vector<long double> coeff(kmax + 1);
  coeff[0] = 1.0L;
  for (int k = 0; k < kmax; ++k)
    coeff[k + 1] = coeff[k] * (long double)(N - 2 * k) * (N - 2 * k - 1) / (2 * k + 2);

  std::vector<int> v(M, 0);
  cld total{0.0L, 0.0L};
  for (;;) {
    int vsum = 0;
    long double c = 1.0L;
    for (int i = 0; i < M; ++i) {
      vsum += v[i];
      long double bin = 1.0L;
      for (int t = 1; t <= v[i]; ++t) bin = bin * (n[i] - t + 1) / t;
      c *= bin;
    }
    if (vsum & 1) c = -c;
    cld sig2{0.0L, 0.0L}, mu{0.0L, 0.0L};
    for (int i = 0; i < M; ++i) {
      long double xi = n[i] - 2.0L * v[i];
      if (xi == 0.0L) continue;
      cld row{0.0L, 0.0L};
      for (int j = 0; j < M; ++j) {
        long double xj = n[j] - 2.0L * v[j];
        if (xj != 0.0L) row += A[i * M + j] * xj;
      }
      sig2 += row * xi;
      mu += g[i] * xi;
    }
    cld SN{0.0L, 0.0L}, sk{1.0L, 0.0L};
    for (int k = 0; k <= kmax; ++k) {
      cld mup{1.0L, 0.0L};
      for (int t = 0; t < N - 2 * k; ++t) mup *= mu;
      SN += coeff[k] * sk * mup;
      sk *= sig2;
    }
    total += c * SN;
    int i = 0;
    while (i < M && v[i] == n[i]) v[i++] = 0;
    if (i == M) break;
    ++v[i];
  }
  long double scale = 1.0L;
  for (int t = 2; t <= N; ++t) scale *= t;
  scale *= std::pow(2.0L, (long double)N);
  return total / scale;
}

int main() {
  std::mt19937_64 rng(777);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  const int M = 4;

  std::vector<std::complex<double>> A(M * M), g(M);
  std::vector<cld> Ald(M * M), gld(M);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j <= i; ++j) {
      std::complex<double> z{U(rng), U(rng)};
      A[i * M + j] = A[j * M + i] = z;
    }
  for (int i = 0; i < M; ++i) g[i] = {U(rng), U(rng)};
  for (int i = 0; i < M * M; ++i) Ald[i] = cld(A[i]);
  for (int i = 0; i < M; ++i) gld[i] = cld(g[i]);

  // patterns: small (vs naive) + larger (vs host sieve)
  std::vector<std::vector<int>> small = {{0,0,0,0},{1,1,0,0},{2,1,1,0},{2,0,2,0},
                                         {3,2,1,0},{1,1,1,1},{2,2,2,0},{3,3,1,1}};
  std::vector<std::vector<int>> big = {{4,4,3,3},{6,5,0,1},{5,5,5,5},{8,2,2,0}};

  std::vector<int> reps_flat;
  std::vector<std::vector<int>> all = small;
  all.insert(all.end(), big.begin(), big.end());
  for (auto& r : all) reps_flat.insert(reps_flat.end(), r.begin(), r.end());
  const int batch = (int)all.size();

  std::vector<cuDoubleComplex> hA(M * M), hg(M);
  for (int i = 0; i < M * M; ++i) hA[i] = make_cuDoubleComplex(A[i].real(), A[i].imag());
  for (int i = 0; i < M; ++i) hg[i] = make_cuDoubleComplex(g[i].real(), g[i].imag());

  cuDoubleComplex *dA = nullptr, *dg = nullptr, *dout = nullptr;
  int* dreps = nullptr;
  cudaMalloc(&dA, hA.size() * sizeof(cuDoubleComplex));
  cudaMalloc(&dg, hg.size() * sizeof(cuDoubleComplex));
  cudaMalloc(&dreps, reps_flat.size() * sizeof(int));
  cudaMalloc(&dout, batch * sizeof(cuDoubleComplex));
  cudaMemcpy(dA, hA.data(), hA.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  cudaMemcpy(dg, hg.data(), hg.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  cudaMemcpy(dreps, reps_flat.data(), reps_flat.size() * sizeof(int), cudaMemcpyHostToDevice);

  gbs::gbs_lhaf_repeated_batched(dA, dg, M, dreps, batch, dout, 0);
  cudaDeviceSynchronize();
  std::vector<cuDoubleComplex> hout(batch);
  cudaMemcpy(hout.data(), dout, batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  cudaFree(dA); cudaFree(dg); cudaFree(dreps); cudaFree(dout);

  double worst_naive = 0.0, worst_sieve = 0.0;
  for (int b = 0; b < batch; ++b) {
    const auto& n = all[b];
    cld kern(cuCreal(hout[b]), cuCimag(hout[b]));
    cld ref2 = host_sieve(Ald, gld, M, n);
    double d2 = (double)(std::abs(kern - ref2) / std::max((long double)1e-300L, std::abs(ref2)));
    worst_sieve = std::max(worst_sieve, d2);
    if (b < (int)small.size()) {
      int N = 0;
      for (int x : n) N += x;
      std::vector<int> idx;
      for (int i = 0; i < M; ++i)
        for (int t = 0; t < n[i]; ++t) idx.push_back(i);
      std::vector<cld> E((size_t)N * N);
      for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
          E[r * N + c] = (r == c) ? gld[idx[r]] : Ald[idx[r] * M + idx[c]];
      cld ref1 = naive_lhaf(E, N);
      double d1 = (double)(std::abs(kern - ref1) / std::max((long double)1e-300L, std::abs(ref1)));
      worst_naive = std::max(worst_naive, d1);
    }
  }
  std::printf("lhaf_repeated: vs naive expanded (ld) %.3e [tol 1e-10], "
              "vs host sieve (ld) %.3e [tol 1e-11]\n", worst_naive, worst_sieve);
  bool ok = worst_naive <= 1e-10 && worst_sieve <= 1e-11;
  // CERTIFIED sieve: values bit-identical to the plain kernel; the bound must
  // enclose the long-double expanded reference on every case.
  {
    const int M = 4, B = 9;
    std::mt19937_64 rng(77);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<cuDoubleComplex> A(M * M), g(M);
    for (int i = 0; i < M; ++i)
      for (int j = 0; j <= i; ++j) {
        cuDoubleComplex z = make_cuDoubleComplex(U(rng), U(rng));
        A[i * M + j] = A[j * M + i] = z;
      }
    for (int i = 0; i < M; ++i) g[i] = make_cuDoubleComplex(0.3 * U(rng), 0.3 * U(rng));
    std::vector<int> reps = {1, 2, 0, 3,  2, 2, 2, 2,  0, 1, 1, 0,
                             3, 3, 2, 2,  1, 1, 1, 1,  4, 0, 0, 4,
                             // larger N: exercises the coeff recurrence PAST the
                             // point (N~16) where a re-associated form diverges,
                             // pinning the bit-identity contract at supported scale
                             8, 8, 0, 0,  10, 10, 0, 0,  7, 7, 7, 7};
    cuDoubleComplex *dA = nullptr, *dg = nullptr, *dout = nullptr, *dout2 = nullptr;
    double* db = nullptr;
    int* dr = nullptr;
    cudaMalloc(&dA, A.size() * sizeof(cuDoubleComplex));
    cudaMalloc(&dg, g.size() * sizeof(cuDoubleComplex));
    cudaMalloc(&dr, reps.size() * sizeof(int));
    cudaMalloc(&dout, B * sizeof(cuDoubleComplex));
    cudaMalloc(&dout2, B * sizeof(cuDoubleComplex));
    cudaMalloc(&db, B * sizeof(double));
    cudaMemcpy(dA, A.data(), A.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dg, g.data(), g.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dr, reps.data(), reps.size() * sizeof(int), cudaMemcpyHostToDevice);
    gbs::gbs_lhaf_repeated_batched(dA, dg, M, dr, B, dout, 0);
    gbs::gbs_lhaf_repeated_cert_batched(dA, dg, M, dr, B, dout2, db, 0);
    cudaDeviceSynchronize();
    std::vector<cuDoubleComplex> h1(B), h2(B);
    std::vector<double> hb(B);
    cudaMemcpy(h1.data(), dout, B * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(h2.data(), dout2, B * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), db, B * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dg); cudaFree(dr); cudaFree(dout); cudaFree(dout2); cudaFree(db);
    double worst_id = 0.0, worst_ratio = 0.0;
    bool enc = true;
    for (int b = 0; b < B; ++b) {
      worst_id = std::max(worst_id, std::abs(cuCreal(h1[b]) - cuCreal(h2[b])) +
                                    std::abs(cuCimag(h1[b]) - cuCimag(h2[b])));
      // long-double sieve reference (80-bit on x86; on ld==double platforms
      // the reference has fp64-grade error itself -> platform-aware slack)
      std::vector<cld> Ac(M * M), gc(M);
      for (int i = 0; i < M * M; ++i) Ac[i] = cld(cuCreal(A[i]), cuCimag(A[i]));
      for (int i = 0; i < M; ++i) gc[i] = cld(cuCreal(g[i]), cuCimag(g[i]));
      std::vector<int> nb(reps.begin() + b * M, reps.begin() + (b + 1) * M);
      cld ref = host_sieve(Ac, gc, M, nb);
      cld got(cuCreal(h2[b]), cuCimag(h2[b]));
      double err = (double)std::abs(got - ref);
      const double sl = (sizeof(long double) == sizeof(double))
                            ? 1e-12 * (1.0 + (double)std::abs(ref)) : 1e-17;
      if (!(err <= hb[b] + sl)) enc = false;
      worst_ratio = std::max(worst_ratio, hb[b] > 0 ? err / hb[b] : 0.0);
    }
    std::printf("certified sieve: value-consistency %.2e (want 0), enclosure %s (worst err/bound %.2e)\n",
                worst_id, enc ? "holds" : "VIOLATED", worst_ratio);
    if (worst_id != 0.0 || !enc) ok = false;
  }
  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
