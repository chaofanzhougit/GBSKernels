// core/check_sampler_haf_varn.cu -- gate for the variable-N hafnian (v3 sampler, increment C).
//
// The varn kernel computes haf of a batch where each matrix has its own even size N=d_n[b],
// stored padded in a maxn x maxn slot. It must equal the validated single-size hafnian kernel
// run on each matrix contiguously -- to machine precision (same power-trace, only the load
// differs). Covers the 0x0 (vacuum -> 1) and several even sizes, repeated sizes, in one batch.

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

extern "C" void gbs_haf_powertrace_fp64_varn_batched(const cuDoubleComplex*, int, int, const int*,
                                                     cuDoubleComplex*, cudaStream_t);
extern "C" void gbs_haf_powertrace_fp64_batched(const cuDoubleComplex*, int, int,
                                                cuDoubleComplex*, cudaStream_t);

int main() {
  const int maxn = 16;
  std::vector<int> sizes = {0, 2, 4, 6, 8, 4, 6};   // even; 0 = vacuum; repeats
  int batch = (int)sizes.size();

  std::vector<cuDoubleComplex> padded((size_t)batch * maxn * maxn, make_cuDoubleComplex(0, 0));
  std::vector<int> nn(batch);
  auto sym = [](int b, int r, int c) {
    int lo = std::min(r, c), hi = std::max(r, c);
    return make_cuDoubleComplex(std::sin(0.3 * (b + 1) * (lo + 1)) + 0.5 * std::cos(0.7 * (hi + 1)),
                                0.2 * std::cos(0.4 * (lo + hi + b)));
  };
  for (int b = 0; b < batch; ++b) {
    int N = sizes[b]; nn[b] = N;
    for (int r = 0; r < N; ++r)
      for (int c = 0; c < N; ++c) padded[(size_t)b * maxn * maxn + r * maxn + c] = sym(b, r, c);
  }

  cuDoubleComplex *d_pad, *d_out; int* d_n;
  cudaMalloc(&d_pad, padded.size() * sizeof(cuDoubleComplex));
  cudaMalloc(&d_out, (size_t)batch * sizeof(cuDoubleComplex));
  cudaMalloc(&d_n, (size_t)batch * sizeof(int));
  cudaMemcpy(d_pad, padded.data(), padded.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  cudaMemcpy(d_n, nn.data(), (size_t)batch * sizeof(int), cudaMemcpyHostToDevice);
  gbs_haf_powertrace_fp64_varn_batched(d_pad, maxn, batch, d_n, d_out, (cudaStream_t)0);
  std::vector<cuDoubleComplex> got(batch);
  cudaMemcpy(got.data(), d_out, (size_t)batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

  double maxrel = 0; bool ok = true;
  for (int b = 0; b < batch; ++b) {
    int N = nn[b];
    cuDoubleComplex ref = make_cuDoubleComplex(1, 0);
    if (N > 0) {
      std::vector<cuDoubleComplex> cont((size_t)N * N);
      for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c) cont[(size_t)r * N + c] = padded[(size_t)b * maxn * maxn + r * maxn + c];
      cuDoubleComplex *d_c, *d_o;
      cudaMalloc(&d_c, cont.size() * sizeof(cuDoubleComplex));
      cudaMalloc(&d_o, sizeof(cuDoubleComplex));
      cudaMemcpy(d_c, cont.data(), cont.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
      gbs_haf_powertrace_fp64_batched(d_c, N, 1, d_o, (cudaStream_t)0);
      cudaMemcpy(&ref, d_o, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
      cudaFree(d_c); cudaFree(d_o);
    }
    double denom = std::max(1e-300, std::hypot(ref.x, ref.y));
    double rel = std::hypot(got[b].x - ref.x, got[b].y - ref.y) / denom;
    if (rel > maxrel) maxrel = rel;
    if (rel > 1e-12) {
      printf("  b=%d N=%d varn=(%.6g,%.6g) ref=(%.6g,%.6g) rel=%.2e\n",
             b, N, got[b].x, got[b].y, ref.x, ref.y, rel); ok = false;
    }
  }
  printf("[check_sampler_haf_varn] batch=%d (sizes 0,2,4,6,8,4,6)  max rel diff vs single-size haf = %.2e\n",
         batch, maxrel);
  cudaFree(d_pad); cudaFree(d_out); cudaFree(d_n);
  ok = ok && (maxrel < 1e-12);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
