// check_tor_recursive.cu -- gate for the recursive prefix-Cholesky torontonian (R2).
//
// Reference: an independent long-double direct evaluation -- for every subset,
// det(I - O_S) by fresh Gaussian elimination (no prefix reuse, no Cholesky), so
// the two implementations share only the mathematical definition. Physical
// (real, SPD-domain) inputs; PASS gate <= 1e-10 max rel err (the conditioning
// floor of the other torontonian gates), plus the off-domain NaN contract.

#include <cuComplex.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace gbs {
extern "C" void gbs_tor_recursive_real_fp64_batched(const double*, int, int, double*,
                                                    cudaStream_t);
extern "C" void gbs_tor_recursive_single_batched(const double*, int, int, double*,
                                                 cudaStream_t);
extern "C" void gbs_tor_recursive_single_cert_batched(const double*, int, int, double*,
                                                      double*, cudaStream_t);
extern "C" void gbs_tor_recursive_single_ddcert_batched(const double*, int, int, double*,
                                                        double*, cudaStream_t);
}

// long-double det by Gaussian elimination with partial pivoting.
static long double det_ld(std::vector<long double> M, int k) {
  long double det = 1.0L;
  for (int c = 0; c < k; ++c) {
    int piv = c;
    for (int r = c + 1; r < k; ++r)
      if (fabsl(M[r * k + c]) > fabsl(M[piv * k + c])) piv = r;
    if (piv != c) {
      for (int t = 0; t < k; ++t) std::swap(M[c * k + t], M[piv * k + t]);
      det = -det;
    }
    long double p = M[c * k + c];
    det *= p;
    if (p == 0.0L) return 0.0L;
    for (int r = c + 1; r < k; ++r) {
      long double f = M[r * k + c] / p;
      for (int t = c; t < k; ++t) M[r * k + t] -= f * M[c * k + t];
    }
  }
  return det;
}

static long double abs_term_sum_ld(const double* O, int n);

static long double ref_tor_ld(const double* O, int n) {
  int dim = 2 * n;
  long double total = 0.0L;
  for (uint64_t mask = 0; mask < (1ull << n); ++mask) {
    int m = __builtin_popcountll(mask);
    std::vector<int> idx;
    for (int i = 0; i < n; ++i)
      if ((mask >> i) & 1ull) idx.push_back(i);
    for (int i = 0; i < n; ++i)
      if ((mask >> i) & 1ull) idx.push_back(i + n);
    int k = 2 * m;
    std::vector<long double> S((size_t)k * k);
    for (int r = 0; r < k; ++r)
      for (int c = 0; c < k; ++c)
        S[r * k + c] = ((r == c) ? 1.0L : 0.0L) - (long double)O[idx[r] * dim + idx[c]];
    long double d = (m == 0) ? 1.0L : det_ld(S, k);
    long double term = 1.0L / sqrtl(d);
    total += ((n - m) & 1) ? -term : term;
  }
  return total;
}

// sum_S |1/sqrt(det(I-O_S))| -- the cancellation normalizer.
static long double abs_term_sum_ld(const double* O, int n) {
  int dim = 2 * n;
  long double acc = 0.0L;
  for (uint64_t mask = 0; mask < (1ull << n); ++mask) {
    int m = __builtin_popcountll(mask);
    std::vector<int> idx;
    for (int i = 0; i < n; ++i)
      if ((mask >> i) & 1ull) idx.push_back(i);
    for (int i = 0; i < n; ++i)
      if ((mask >> i) & 1ull) idx.push_back(i + n);
    int k = 2 * m;
    std::vector<long double> S((size_t)k * k);
    for (int r = 0; r < k; ++r)
      for (int c = 0; c < k; ++c)
        S[r * k + c] = ((r == c) ? 1.0L : 0.0L) - (long double)O[idx[r] * dim + idx[c]];
    long double d = (m == 0) ? 1.0L : det_ld(S, k);
    acc += fabsl(1.0L / sqrtl(d));
  }
  return acc;
}

int main() {
  std::mt19937_64 rng(4242);
  std::uniform_real_distribution<double> U(-1.0, 1.0);

  double worst = 0.0;
  for (int n = 1; n <= 5; ++n) {
    const int batch = 64;
    const int dim = 2 * n;
    std::vector<double> host((size_t)batch * dim * dim);
    for (int b = 0; b < batch; ++b) {
      // physical-ish real O: small-norm symmetric -> I - O_S SPD
      std::vector<double> G((size_t)dim * dim);
      for (auto& z : G) z = 0.12 * U(rng);
      for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
          host[(size_t)b * dim * dim + i * dim + j] = 0.5 * (G[i * dim + j] + G[j * dim + i]);
    }
    double *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, host.size() * sizeof(double));
    cudaMalloc(&d_out, batch * sizeof(double));
    cudaMemcpy(d_in, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice);
    gbs::gbs_tor_recursive_real_fp64_batched(d_in, n, batch, d_out, 0);
    cudaDeviceSynchronize();
    std::vector<double> out(batch);
    cudaMemcpy(out.data(), d_out, batch * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);

    for (int b = 0; b < batch; ++b) {
      long double ref = ref_tor_ld(&host[(size_t)b * dim * dim], n);
      long double tsum = abs_term_sum_ld(&host[(size_t)b * dim * dim], n);
      // normalize by sum|term|: tor(O->0) == 0, so random small O is near-
      // totally cancelling and a raw rel-err tolerance would measure the input
      // family, not the kernel. err/sum|term| ~ u for a correct kernel.
      double scaled = std::abs((double)(((long double)out[b] - ref) / tsum));
      worst = std::max(worst, scaled);
    }
  }
  std::printf("tor_recursive vs long-double direct subset determinants, err/sum|term| "
              "(n=1..5, real O): %.3e  [tol 1e-12]\n", worst);

  // off-domain contract: I - O_S not SPD -> NaN (host falls back), never silent
  {
    const int n = 2, dim = 4;
    std::vector<double> bad((size_t)dim * dim, 0.0);
    for (int i = 0; i < dim; ++i) bad[i * dim + i] = 1.5;   // I - O = -0.5 I
    double *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, bad.size() * sizeof(double));
    cudaMalloc(&d_out, sizeof(double));
    cudaMemcpy(d_in, bad.data(), bad.size() * sizeof(double), cudaMemcpyHostToDevice);
    gbs::gbs_tor_recursive_real_fp64_batched(d_in, n, 1, d_out, 0);
    cudaDeviceSynchronize();
    double v = 0.0;
    cudaMemcpy(&v, d_out, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);
    if (!std::isnan(v)) {
      std::printf("off-domain input did NOT produce NaN\nFAIL\n");
      return 1;
    }
  }

  // ---- SINGLE-LARGE: 2^g-subtree split of one evaluation ----
  // (a) vs the batched kernel at n=5 (same values, different parallel split);
  // (b) closed form tor(a*I) = (a/(1-a))^n at n=16 and n=20 -- big-n validation
  //     with no reference enumeration; (c) off-domain -> NaN.
  {
    bool s_ok = true;
    {
      const int n = 5, dim = 2 * n, g = 3;
      std::vector<double> O((size_t)dim * dim);
      std::mt19937_64 r2(99);
      std::uniform_real_distribution<double> U2(-1.0, 1.0);
      std::vector<double> G((size_t)dim * dim);
      for (auto& z : G) z = 0.12 * U2(r2);
      for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) O[i * dim + j] = 0.5 * (G[i * dim + j] + G[j * dim + i]);
      double *dO = nullptr, *dp = nullptr, *db = nullptr;
      cudaMalloc(&dO, O.size() * sizeof(double));
      cudaMalloc(&dp, (1ull << g) * sizeof(double));
      cudaMalloc(&db, sizeof(double));
      cudaMemcpy(dO, O.data(), O.size() * sizeof(double), cudaMemcpyHostToDevice);
      gbs::gbs_tor_recursive_single_batched(dO, n, g, dp, 0);
      gbs::gbs_tor_recursive_real_fp64_batched(dO, n, 1, db, 0);
      cudaDeviceSynchronize();
      std::vector<double> hp(1ull << g);
      double hb = 0.0;
      cudaMemcpy(hp.data(), dp, hp.size() * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(&hb, db, sizeof(double), cudaMemcpyDeviceToHost);
      cudaFree(dO); cudaFree(dp); cudaFree(db);
      double sum = 0.0;
      for (double x : hp) sum += x;
      long double mass = abs_term_sum_ld(O.data(), n);
      double sc = std::abs((double)(((long double)sum - (long double)hb) / mass));
      if (sc > 1e-13) { std::printf("single vs batched: %.3e FAIL\n", sc); s_ok = false; }
    }
    for (int n : {16, 20}) {
      const int dim = 2 * n, g = 10;
      const double a = 0.2;
      std::vector<double> O((size_t)dim * dim, 0.0);
      for (int i = 0; i < dim; ++i) O[i * dim + i] = a;
      double *dO = nullptr, *dp = nullptr;
      cudaMalloc(&dO, O.size() * sizeof(double));
      cudaMalloc(&dp, (1ull << g) * sizeof(double));
      cudaMemcpy(dO, O.data(), O.size() * sizeof(double), cudaMemcpyHostToDevice);
      gbs::gbs_tor_recursive_single_batched(dO, n, g, dp, 0);
      cudaDeviceSynchronize();
      std::vector<double> hp(1ull << g);
      cudaMemcpy(hp.data(), dp, hp.size() * sizeof(double), cudaMemcpyDeviceToHost);
      cudaFree(dO); cudaFree(dp);
      long double sum = 0.0L;
      for (double x : hp) sum += (long double)x;
      long double exact = powl(a / (1.0L - a), (long double)n);
      // a*I is the WORST-cancelling torontonian input: kappa = ((1+1/(1-a))/
      // (a/(1-a)))^n ~ 9^n at a=0.2, so raw rel err is meaningless in fp64 at
      // these n (that cancellation is the certified tier's raison d'etre).
      // Gate on err / sum|term| = err / (1 + 1/(1-a))^n, the kernel-honest form.
      long double massc = powl(1.0L + 1.0L / (1.0L - a), (long double)n);
      double sc2 = std::abs((double)((sum - exact) / massc));
      std::printf("single n=%d closed form: err/mass %.3e (raw fp64 kappa ~ %.0Le)\n",
                  n, sc2, powl(9.0L, (long double)n));
      if (sc2 > 1e-12) s_ok = false;
    }
    {
      const int n = 8, dim = 16, g = 4;
      std::vector<double> bad((size_t)dim * dim, 0.0);
      for (int i = 0; i < dim; ++i) bad[i * dim + i] = 1.5;
      double *dO = nullptr, *dp = nullptr;
      cudaMalloc(&dO, bad.size() * sizeof(double));
      cudaMalloc(&dp, (1ull << g) * sizeof(double));
      cudaMemcpy(dO, bad.data(), bad.size() * sizeof(double), cudaMemcpyHostToDevice);
      gbs::gbs_tor_recursive_single_batched(dO, n, g, dp, 0);
      cudaDeviceSynchronize();
      std::vector<double> hp(1ull << g);
      cudaMemcpy(hp.data(), dp, hp.size() * sizeof(double), cudaMemcpyDeviceToHost);
      cudaFree(dO); cudaFree(dp);
      double sum = 0.0;
      for (double x : hp) sum += x;
      if (!std::isnan(sum)) { std::printf("single off-domain not NaN FAIL\n"); s_ok = false; }
    }
    // CERTIFIED single-large: the bound must ENCLOSE the ld reference at n=5
    // (physical) and the exact closed form at n=16 (the worst-cancelling input).
    {
      const int n = 5, dim = 2 * n, g = 3;
      std::vector<double> O((size_t)dim * dim);
      std::mt19937_64 r3(1234);
      std::uniform_real_distribution<double> U3(-1.0, 1.0);
      std::vector<double> G((size_t)dim * dim);
      for (auto& z : G) z = 0.12 * U3(r3);
      for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) O[i * dim + j] = 0.5 * (G[i * dim + j] + G[j * dim + i]);
      double *dO = nullptr, *dp = nullptr, *db = nullptr;
      cudaMalloc(&dO, O.size() * sizeof(double));
      cudaMalloc(&dp, (1ull << g) * sizeof(double));
      cudaMalloc(&db, (1ull << g) * sizeof(double));
      cudaMemcpy(dO, O.data(), O.size() * sizeof(double), cudaMemcpyHostToDevice);
      gbs::gbs_tor_recursive_single_cert_batched(dO, n, g, dp, db, 0);
      cudaDeviceSynchronize();
      std::vector<double> hp(1ull << g), hb(1ull << g);
      cudaMemcpy(hp.data(), dp, hp.size() * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(hb.data(), db, hb.size() * sizeof(double), cudaMemcpyDeviceToHost);
      cudaFree(dO); cudaFree(dp); cudaFree(db);
      long double sum = 0.0L;
      double eb = 0.0;
      for (size_t i = 0; i < hp.size(); ++i) { sum += (long double)hp[i]; eb += hb[i]; }
      long double ref = ref_tor_ld(O.data(), n);
      double err = std::abs((double)(sum - ref));
      std::printf("single-cert n=5: err %.2e bound %.2e %s\n", err, eb,
                  err <= eb ? "(enclosed)" : "ENCLOSURE VIOLATED");
      if (!(err <= eb) || !std::isfinite(eb)) s_ok = false;
    }
    {
      const int n = 16, dim = 2 * n, g = 10;
      const double a = 0.2;
      std::vector<double> O((size_t)dim * dim, 0.0);
      for (int i = 0; i < dim; ++i) O[i * dim + i] = a;
      double *dO = nullptr, *dp = nullptr, *db = nullptr;
      cudaMalloc(&dO, O.size() * sizeof(double));
      cudaMalloc(&dp, (1ull << g) * sizeof(double));
      cudaMalloc(&db, (1ull << g) * sizeof(double));
      cudaMemcpy(dO, O.data(), O.size() * sizeof(double), cudaMemcpyHostToDevice);
      gbs::gbs_tor_recursive_single_cert_batched(dO, n, g, dp, db, 0);
      cudaDeviceSynchronize();
      std::vector<double> hp(1ull << g), hb(1ull << g);
      cudaMemcpy(hp.data(), dp, hp.size() * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(hb.data(), db, hb.size() * sizeof(double), cudaMemcpyDeviceToHost);
      cudaFree(dO); cudaFree(dp); cudaFree(db);
      long double sum = 0.0L;
      double eb = 0.0;
      for (size_t i = 0; i < hp.size(); ++i) { sum += (long double)hp[i]; eb += hb[i]; }
      long double exact = powl(a / (1.0L - a), (long double)n);
      double err = std::abs((double)(sum - exact));
      std::printf("single-cert n=16 closed form: err %.2e bound %.2e %s\n", err, eb,
                  err <= eb ? "(enclosed at kappa~2e15)" : "ENCLOSURE VIOLATED");
      if (!(err <= eb) || !std::isfinite(eb)) s_ok = false;
    }
    // DD subtree outputs are collapsed to binary64 before the host reduction.
    // At n=1, g=1 the nonempty subtree is exactly 1/(1-a), so this gate
    // isolates that conversion and requires its residual to be in the bound.
    {
      const int n = 1, dim = 2, g = 1;
      const double a = 0.1;
      std::vector<double> O((size_t)dim * dim, 0.0);
      O[0] = a;
      O[3] = a;
      double *dO = nullptr, *dp = nullptr, *db = nullptr;
      cudaMalloc(&dO, O.size() * sizeof(double));
      cudaMalloc(&dp, (1ull << g) * sizeof(double));
      cudaMalloc(&db, (1ull << g) * sizeof(double));
      cudaMemcpy(dO, O.data(), O.size() * sizeof(double), cudaMemcpyHostToDevice);
      gbs::gbs_tor_recursive_single_ddcert_batched(dO, n, g, dp, db, 0);
      cudaDeviceSynchronize();
      std::vector<double> hp(1ull << g), hb(1ull << g);
      cudaMemcpy(hp.data(), dp, hp.size() * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(hb.data(), db, hb.size() * sizeof(double), cudaMemcpyDeviceToHost);
      cudaFree(dO); cudaFree(dp); cudaFree(db);
      long double exact_nonempty = 1.0L / (1.0L - (long double)a);
      double err = std::abs((double)((long double)hp[1] - exact_nonempty));
      std::printf("single-ddcert collapse: err %.2e bound %.2e %s\n", err, hb[1],
                  err <= hb[1] ? "(enclosed)" : "ENCLOSURE VIOLATED");
      if (!(err <= hb[1]) || !std::isfinite(hb[1])) s_ok = false;
    }
    if (!s_ok) { std::printf("FAIL\n"); return 1; }
    std::printf("single-large split: batched-agreement + closed forms + certified enclosure + off-domain NaN ok\n");
  }

  bool ok = worst <= 1e-12;
  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
