// check_loop_hafnian_dd.cu -- GPU double-double loop-hafnian gate (docs/DESIGN.md §6).
//
// Same structure as check_hafnian_dd. Cancellation family: a 2x2 block B with
// lhaf(B) = h + g*k = 2 + 1*(delta-2) = delta (the off-diagonal and the product
// of the two loop weights nearly cancel), direct-summed with a well-conditioned
// remainder R (nonzero diagonal -> loops), so lhaf(B (+) R) = lhaf(B)*lhaf(R) is
// tiny while the FP64 power-trace subset sum has O(1) terms. Reference computed
// from the block factorization (lhaf(B) exact by Sterbenz; R well-conditioned).
//
//   nvcc -O3 -std=c++17 loop_hafnian.cu loop_hafnian_dd.cu check_loop_hafnian_dd.cu -o check_loop_hafnian_dd

#include <cuComplex.h>

#include <complex>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

namespace gbs {
extern "C" void gbs_loop_haf_fp64_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
extern "C" void gbs_loop_haf_dd_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
}
using cd = std::complex<double>;
using cld = std::complex<long double>;

// Naive loop hafnian (recursive: each vertex loops or pairs) in long double.
static cld host_lhaf_ld(const std::vector<cld>& A, int N, std::vector<int> rem) {
  if (rem.empty()) return cld(1.0L, 0.0L);
  int i = rem[0];
  std::vector<int> rest(rem.begin() + 1, rem.end());
  cld total = A[(size_t)i * N + i] * host_lhaf_ld(A, N, rest);  // i as a loop
  for (size_t k = 0; k < rest.size(); ++k) {
    int j = rest[k];
    std::vector<int> r2 = rest;
    r2.erase(r2.begin() + k);
    total += A[(size_t)i * N + j] * host_lhaf_ld(A, N, r2);
  }
  return total;
}
static cld lhaf_ld(const cd* M, int N) {
  std::vector<cld> A((size_t)N * N);
  for (int i = 0; i < N * N; ++i) A[i] = cld((long double)M[i].real(), (long double)M[i].imag());
  std::vector<int> all(N);
  for (int i = 0; i < N; ++i) all[i] = i;
  return host_lhaf_ld(A, N, all);
}

static std::vector<cd> run_kernel(void (*fn)(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t),
                                  const std::vector<cd>& host, int N, int batch) {
  std::vector<cuDoubleComplex> hin(host.size());
  for (size_t i = 0; i < host.size(); ++i) hin[i] = make_cuDoubleComplex(host[i].real(), host[i].imag());
  cuDoubleComplex *d_in = nullptr, *d_out = nullptr;
  cudaMalloc(&d_in, hin.size() * sizeof(cuDoubleComplex));
  cudaMalloc(&d_out, batch * sizeof(cuDoubleComplex));
  cudaMemcpy(d_in, hin.data(), hin.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  fn(d_in, N, batch, d_out, 0);
  cudaDeviceSynchronize();
  std::vector<cuDoubleComplex> hout(batch);
  cudaMemcpy(hout.data(), d_out, batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  cudaFree(d_in); cudaFree(d_out);
  std::vector<cd> out(batch);
  for (int b = 0; b < batch; ++b) out[b] = cd(cuCreal(hout[b]), cuCimag(hout[b]));
  return out;
}
static double rel_ld(cd got, cld ref) {
  long double d = std::abs(cld((long double)got.real(), (long double)got.imag()) - ref);
  long double m = std::abs(ref);
  return (double)(d / (m > 1e-300L ? m : 1e-300L));
}

// symmetric real matrix WITH a random diagonal (loop weights)
static std::vector<cd> sym_diag(int N, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> U(-1, 1);
  std::vector<cd> A((size_t)N * N, cd(0, 0));
  for (int i = 0; i < N; ++i) {
    A[i * N + i] = cd(U(rng), 0);
    for (int j = i + 1; j < N; ++j) { cd v(U(rng), 0); A[i * N + j] = v; A[j * N + i] = v; }
  }
  return A;
}

// 2x2 block B (lhaf = 2 + (delta-2) = delta) (+) 4x4 well-conditioned R -> 6x6
static std::vector<cd> cancel_lhaf_matrix(double delta, std::mt19937_64& rng) {
  std::vector<cd> A(36, cd(0, 0));
  A[0] = cd(1, 0); A[1] = cd(2, 0); A[6] = cd(2, 0); A[7] = cd(delta - 2.0, 0);  // B
  auto R = sym_diag(4, rng);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) A[(i + 2) * 6 + (j + 2)] = R[i * 4 + j];
  return A;
}
static cld cancel_lhaf_ref(const std::vector<cd>& A) {
  std::vector<cd> B(4), R(16);
  B[0] = A[0]; B[1] = A[1]; B[2] = A[6]; B[3] = A[7];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) R[i * 4 + j] = A[(i + 2) * 6 + (j + 2)];
  return lhaf_ld(B.data(), 2) * lhaf_ld(R.data(), 4);
}

int main() {
  std::mt19937_64 rng(20260620);
  const double dd_tol = 1e-12;
  bool ok = true;

  // (1) DD == FP64 on well-conditioned symmetric-with-diagonal matrices
  double worst_easy = 0.0;
  for (int N = 2; N <= 10; N += 2) {
    const int batch = 32;
    std::vector<cd> host;
    for (int b = 0; b < batch; ++b) { auto m = sym_diag(N, rng); host.insert(host.end(), m.begin(), m.end()); }
    auto dd = run_kernel(gbs::gbs_loop_haf_dd_batched, host, N, batch);
    auto fp = run_kernel(gbs::gbs_loop_haf_fp64_batched, host, N, batch);
    for (int b = 0; b < batch; ++b) {
      double mm = std::max(std::abs(fp[b]), 1e-300);
      worst_easy = std::max(worst_easy, std::abs(dd[b] - fp[b]) / mm);
    }
  }
  if (worst_easy > 1e-11) ok = false;
  std::printf("(1) DD == FP64 on well-conditioned (N=2..10): max rel diff %.3e  [tol 1e-11]\n", worst_easy);

  // (2) DD beats FP64 on cancellation, vs the block-exact reference
  std::printf("(2) cancellation:  delta     relerr_FP64    relerr_DD     DD held?\n");
  bool dd_ok = true;
  double e_fp_worst = 0.0, e_dd_worst = 0.0;
  for (double delta : {1e-2, 1e-4, 1e-6, 1e-8, 1e-10}) {
    auto host = cancel_lhaf_matrix(delta, rng);
    auto dd = run_kernel(gbs::gbs_loop_haf_dd_batched, host, 6, 1);
    auto fp = run_kernel(gbs::gbs_loop_haf_fp64_batched, host, 6, 1);
    cld ref = cancel_lhaf_ref(host);
    double e_dd = rel_ld(dd[0], ref), e_fp = rel_ld(fp[0], ref);
    bool held = (e_dd < dd_tol) && (e_fp <= 1e-12 || e_dd < e_fp / 50.0);
    dd_ok = dd_ok && held;
    e_fp_worst = std::max(e_fp_worst, e_fp);
    e_dd_worst = std::max(e_dd_worst, e_dd);
    std::printf("                   %.0e    %.3e    %.3e    %s\n", delta, e_fp, e_dd, held ? "yes" : "NO");
  }
  bool demonstrated = (e_fp_worst > 1e-9) && (e_dd_worst < 1e-12);
  ok = ok && dd_ok && demonstrated;

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
