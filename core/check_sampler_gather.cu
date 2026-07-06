// core/check_sampler_gather.cu -- gate for the v3 on-device gather kernel (increment B).
//
// Independent check: a MARKER A_k with A[i][j] = i*(2k) + j (every entry uniquely identifies
// its (row,col)), gathered against HAND-DERIVED index sets for several (prefix, candidate j)
// cases -- so a passing gather provably matches the host sampler's np.ix_ index construction,
// not just its own re-statement. Also checks the recorded submatrix size n.

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

extern "C" void gbs_sampler_gather(const cuDoubleComplex*, int, const int*, int, int, int,
                                   cuDoubleComplex*, int*, cudaStream_t);

static bool check_case(const cuDoubleComplex* sub, int maxn, const std::vector<int>& idx, int twok) {
  int n = (int)idx.size();
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < n; ++c) {
      double got = sub[(size_t)r * maxn + c].x;
      double want = (double)(idx[r] * twok + idx[c]);
      if (got != want) { printf("  entry (%d,%d): got %.0f want %.0f\n", r, c, got, want); return false; }
    }
  return true;
}

int main() {
  const int k = 3, twok = 6, cutoff = 2, maxn = 16, num_draws = 2;
  std::vector<cuDoubleComplex> Ak((size_t)twok * twok);
  for (int i = 0; i < twok; ++i)
    for (int c = 0; c < twok; ++c) Ak[(size_t)i * twok + c] = make_cuDoubleComplex(i * twok + c, 0.0);
  std::vector<int> prefix = {1, 2,   0, 1};   // draw0 = [1,2], draw1 = [0,1]  (k-1 = 2 counts)

  int total = num_draws * (cutoff + 1);
  cuDoubleComplex *d_Ak, *d_sub; int *d_prefix, *d_n;
  cudaMalloc(&d_Ak, Ak.size() * sizeof(cuDoubleComplex));
  cudaMalloc(&d_prefix, prefix.size() * sizeof(int));
  cudaMalloc(&d_sub, (size_t)total * maxn * maxn * sizeof(cuDoubleComplex));
  cudaMalloc(&d_n, (size_t)total * sizeof(int));
  cudaMemcpy(d_Ak, Ak.data(), Ak.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  cudaMemcpy(d_prefix, prefix.data(), prefix.size() * sizeof(int), cudaMemcpyHostToDevice);

  gbs_sampler_gather(d_Ak, k, d_prefix, num_draws, cutoff, maxn, d_sub, d_n, (cudaStream_t)0);

  std::vector<cuDoubleComplex> sub((size_t)total * maxn * maxn);
  std::vector<int> nn((size_t)total);
  cudaMemcpy(sub.data(), d_sub, sub.size() * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.data(), d_n, (size_t)total * sizeof(int), cudaMemcpyDeviceToHost);

  struct Case { int d, j; std::vector<int> idx; };
  std::vector<Case> cases = {
    {0, 0, {0, 1, 1, 3, 4, 4}},
    {0, 1, {0, 1, 1, 2, 3, 4, 4, 5}},
    {0, 2, {0, 1, 1, 2, 2, 3, 4, 4, 5, 5}},
    {1, 0, {1, 4}},
    {1, 2, {1, 2, 2, 4, 5, 5}},
  };
  bool ok = true;
  for (auto& cs : cases) {
    int t = cs.d * (cutoff + 1) + cs.j;
    if (nn[t] != (int)cs.idx.size()) {
      printf("  n (d=%d,j=%d): got %d want %zu\n", cs.d, cs.j, nn[t], cs.idx.size()); ok = false; continue;
    }
    if (!check_case(&sub[(size_t)t * maxn * maxn], maxn, cs.idx, twok)) {
      printf("  case (d=%d,j=%d) FAILED\n", cs.d, cs.j); ok = false;
    }
  }
  printf("[check_sampler_gather] %zu hand-derived (prefix,j) cases vs marker A_k\n", cases.size());
  cudaFree(d_Ak); cudaFree(d_prefix); cudaFree(d_sub); cudaFree(d_n);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
