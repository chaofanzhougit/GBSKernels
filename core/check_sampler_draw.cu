// core/check_sampler_draw.cu -- gate for the v3 on-device draw kernel (increment A).
//
// The draw kernel turns each draw's (cutoff+1) conditional weights into a photon count via the
// inverse CDF, drawing u from a per-draw cuRAND stream. cuRAND != numpy RNG, so correctness is
// DISTRIBUTIONAL: feed a FIXED conditional to many draws and check the empirical photon
// histogram reproduces the multinomial the weights define (frequencies within sampling noise;
// chi-square sane). Also checks the per-mode normalisation handles the degenerate (all-zero)
// conditional -> vacuum.

#include <curand_kernel.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

extern "C" void gbs_sampler_init_states(unsigned long long, int, curandState_t*, cudaStream_t);
extern "C" void gbs_sampler_draw(const double*, const double*, int, int,
                                 curandState_t*, int*, cudaStream_t);

int main() {
  const int cutoff = 4;          // photon 0..4
  const int N = 200000;          // draws
  // A known conditional shape; we fold j! into the weights (invfac = 1) so the target is p/Z.
  double p[5] = {0.40, 0.25, 0.20, 0.10, 0.05};
  double invfac[5] = {1, 1, 1, 1, 1};
  double Z = 0; for (int j = 0; j <= cutoff; ++j) Z += p[j];
  double target[5]; for (int j = 0; j <= cutoff; ++j) target[j] = p[j] / Z;

  double *d_haf, *d_invfac; int* d_out; curandState_t* d_states;
  cudaMalloc(&d_haf, (size_t)N * (cutoff + 1) * sizeof(double));
  cudaMalloc(&d_invfac, (cutoff + 1) * sizeof(double));
  cudaMalloc(&d_out, (size_t)N * sizeof(int));
  cudaMalloc(&d_states, (size_t)N * sizeof(curandState_t));

  std::vector<double> haf((size_t)N * (cutoff + 1));
  for (int d = 0; d < N; ++d)
    for (int j = 0; j <= cutoff; ++j) haf[(size_t)d * (cutoff + 1) + j] = p[j];
  cudaMemcpy(d_haf, haf.data(), haf.size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_invfac, invfac, (cutoff + 1) * sizeof(double), cudaMemcpyHostToDevice);

  gbs_sampler_init_states(12345ULL, N, d_states, (cudaStream_t)0);
  gbs_sampler_draw(d_haf, d_invfac, cutoff, N, d_states, d_out, (cudaStream_t)0);

  std::vector<int> out((size_t)N);
  cudaMemcpy(out.data(), d_out, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost);

  long hist[5] = {0, 0, 0, 0, 0};
  for (int d = 0; d < N; ++d) {
    int v = out[d];
    if (v < 0 || v > cutoff) { printf("photon out of range: %d\nFAIL\n", v); return 1; }
    hist[v]++;
  }
  double chi2 = 0, maxdev = 0;
  for (int j = 0; j <= cutoff; ++j) {
    double e = target[j] * N, f = (double)hist[j] / N;
    chi2 += (hist[j] - e) * (hist[j] - e) / e;
    if (fabs(f - target[j]) > maxdev) maxdev = fabs(f - target[j]);
  }

  // Degenerate conditional (all-zero weights) -> every draw must be vacuum (0).
  std::vector<double> z((size_t)N * (cutoff + 1), 0.0);
  cudaMemcpy(d_haf, z.data(), z.size() * sizeof(double), cudaMemcpyHostToDevice);
  gbs_sampler_draw(d_haf, d_invfac, cutoff, N, d_states, d_out, (cudaStream_t)0);
  cudaMemcpy(out.data(), d_out, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost);
  bool degenerate_ok = true;
  for (int d = 0; d < N; ++d) if (out[d] != 0) { degenerate_ok = false; break; }

  printf("[check_sampler_draw] cutoff=%d N=%d  chi2=%.2f (dof %d)  max|freq-target|=%.4f  degenerate->0:%s\n",
         cutoff, N, chi2, cutoff, maxdev, degenerate_ok ? "ok" : "BAD");
  for (int j = 0; j <= cutoff; ++j)
    printf("  photon %d: freq %.4f  target %.4f\n", j, (double)hist[j] / N, target[j]);

  cudaFree(d_haf); cudaFree(d_invfac); cudaFree(d_out); cudaFree(d_states);
  // sampling noise per bin ~ 0.0011 at N=2e5; 0.01 is ~9 sigma. chi2 dof=4, 0.999-quantile ~18.5.
  bool ok = degenerate_ok && (maxdev < 0.01) && (chi2 < 30.0);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
