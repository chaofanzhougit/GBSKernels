// core/check_sampler_session.cu -- gate for the resident sampler chain (v3 increment C).
//
// The NEW code in sampler_session.cu is the device orchestration: the per-mode loop and the
// three helpers (compact prefix / scatter column / real extract). Validate it by running the
// resident chain (device loop) AND a HOST loop that uses the SAME gather / varn-hafnian / draw
// kernels but does the compact/extract/scatter with plain host memcpy, both starting from the
// same cuRAND states. They must produce IDENTICAL samples (the shared kernels advance the states
// identically, so any difference is an orchestration bug). The gather/haf/draw themselves are
// validated by their own gates; this isolates the orchestration. Also checks samples in range.

#include <cuComplex.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" void gbs_sampler_init_states(unsigned long long, int, curandState_t*, cudaStream_t);
// NOTE: the offsets argument is a HOST pointer (host-side pointer arithmetic only).
extern "C" void gbs_sampler_run(const cuDoubleComplex*, const int*, int, int, int, int, const double*,
                                curandState_t*, int*, int*, cuDoubleComplex*, int*, cuDoubleComplex*,
                                double*, int*, cudaStream_t);
extern "C" void gbs_sampler_gather(const cuDoubleComplex*, int, const int*, int, int, int,
                                   cuDoubleComplex*, int*, cudaStream_t);
extern "C" void gbs_haf_powertrace_fp64_varn_batched(const cuDoubleComplex*, int, int, const int*,
                                                     cuDoubleComplex*, cudaStream_t);
extern "C" void gbs_sampler_draw(const double*, const double*, int, int, curandState_t*, int*, cudaStream_t);

int main() {
  const int M = 3, cutoff = 3, maxn = 16, num_draws = 3000;
  const int batch = num_draws * (cutoff + 1);

  std::vector<int> off(M);
  std::vector<cuDoubleComplex> Ak;
  for (int k = 1; k <= M; ++k) {
    off[k - 1] = (int)Ak.size();
    int twok = 2 * k;
    for (int r = 0; r < twok; ++r)
      for (int c = 0; c < twok; ++c) {
        int lo = std::min(r, c), hi = std::max(r, c);
        Ak.push_back(make_cuDoubleComplex(0.30 * std::cos(0.5 * (lo + 1) * k) + 0.02 * hi,
                                          0.05 * std::sin(0.3 * (lo + hi + k))));
      }
  }
  std::vector<double> invfac(cutoff + 1);
  invfac[0] = 1.0; for (int j = 1; j <= cutoff; ++j) invfac[j] = invfac[j - 1] / j;

  // device (shim) buffers
  cuDoubleComplex *d_Ak, *d_sub, *d_hafc; int *d_state, *d_prefix_k, *d_n, *d_out_mode;
  double *d_invfac, *d_hafr; curandState_t *d_states;
  cudaMalloc(&d_Ak, Ak.size() * sizeof(cuDoubleComplex));
  cudaMalloc(&d_invfac, (cutoff + 1) * sizeof(double));
  cudaMalloc(&d_states, (size_t)num_draws * sizeof(curandState_t));
  cudaMalloc(&d_state, (size_t)num_draws * M * sizeof(int));
  cudaMalloc(&d_prefix_k, (size_t)num_draws * (M > 1 ? M - 1 : 1) * sizeof(int));
  cudaMalloc(&d_sub, (size_t)batch * maxn * maxn * sizeof(cuDoubleComplex));
  cudaMalloc(&d_n, (size_t)batch * sizeof(int));
  cudaMalloc(&d_hafc, (size_t)batch * sizeof(cuDoubleComplex));
  cudaMalloc(&d_hafr, (size_t)batch * sizeof(double));
  cudaMalloc(&d_out_mode, (size_t)num_draws * sizeof(int));
  cudaMemcpy(d_Ak, Ak.data(), Ak.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  cudaMemcpy(d_invfac, invfac.data(), (cutoff + 1) * sizeof(double), cudaMemcpyHostToDevice);

  // --- run 1: the resident chain (device orchestration) ---
  gbs_sampler_init_states(2024ULL, num_draws, d_states, (cudaStream_t)0);
  { std::vector<int> z((size_t)num_draws * M, 0); cudaMemcpy(d_state, z.data(), z.size() * sizeof(int), cudaMemcpyHostToDevice); }
  gbs_sampler_run(d_Ak, off.data(), M, num_draws, cutoff, maxn, d_invfac, d_states, d_state,
                  d_prefix_k, d_sub, d_n, d_hafc, d_hafr, d_out_mode, (cudaStream_t)0);
  std::vector<int> dev((size_t)num_draws * M);
  cudaMemcpy(dev.data(), d_state, dev.size() * sizeof(int), cudaMemcpyDeviceToHost);

  // --- run 2: the SAME chain, HOST-orchestrated (host compact/extract/scatter, same kernels) ---
  gbs_sampler_init_states(2024ULL, num_draws, d_states, (cudaStream_t)0);   // identical state seq
  std::vector<int> host((size_t)num_draws * M, 0);
  std::vector<int> prefix_k((size_t)num_draws * (M > 1 ? M - 1 : 1));
  std::vector<cuDoubleComplex> hafc(batch); std::vector<double> hafr(batch); std::vector<int> outm(num_draws);
  for (int k = 1; k <= M; ++k) {
    int Kc = k - 1;
    for (int d = 0; d < num_draws; ++d) for (int i = 0; i < Kc; ++i) prefix_k[(size_t)d * Kc + i] = host[(size_t)d * M + i];
    cudaMemcpy(d_prefix_k, prefix_k.data(), (size_t)num_draws * (Kc > 0 ? Kc : 1) * sizeof(int), cudaMemcpyHostToDevice);
    gbs_sampler_gather(d_Ak + off[k - 1], k, d_prefix_k, num_draws, cutoff, maxn, d_sub, d_n, (cudaStream_t)0);
    gbs_haf_powertrace_fp64_varn_batched(d_sub, maxn, batch, d_n, d_hafc, (cudaStream_t)0);
    cudaMemcpy(hafc.data(), d_hafc, batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    for (int t = 0; t < batch; ++t) hafr[t] = hafc[t].x;
    cudaMemcpy(d_hafr, hafr.data(), batch * sizeof(double), cudaMemcpyHostToDevice);
    gbs_sampler_draw(d_hafr, d_invfac, cutoff, num_draws, d_states, d_out_mode, (cudaStream_t)0);
    cudaMemcpy(outm.data(), d_out_mode, num_draws * sizeof(int), cudaMemcpyDeviceToHost);
    for (int d = 0; d < num_draws; ++d) host[(size_t)d * M + Kc] = outm[d];
  }

  bool ok = true; long mism = 0; int maxv = 0;
  for (size_t i = 0; i < dev.size(); ++i) {
    if (dev[i] < 0 || dev[i] > cutoff) { ok = false; }
    if (dev[i] > maxv) maxv = dev[i];
    if (dev[i] != host[i]) mism++;
  }
  // sanity: samples must actually vary (not all vacuum)
  long nonzero = 0; for (int v : dev) if (v) nonzero++;
  printf("[check_sampler_session] M=%d cutoff=%d draws=%d  device==host-orchestration mismatches=%ld  "
         "max photon=%d  nonzero=%ld/%zu\n", M, cutoff, num_draws, mism, maxv, nonzero, dev.size());
  ok = ok && (mism == 0) && (nonzero > 0);
  cudaFree(d_Ak); cudaFree(d_invfac); cudaFree(d_states); cudaFree(d_state);
  cudaFree(d_prefix_k); cudaFree(d_sub); cudaFree(d_n); cudaFree(d_hafc); cudaFree(d_hafr); cudaFree(d_out_mode);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
