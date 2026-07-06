// check_session.cu -- device-resident session gate (docs/device_resident_contract.md).
//
// Validates the two residency guarantees on the CPU shim, before any GPU session:
//   (1) correctness: a bucket evaluated through gbs_session_evaluate equals the
//       one-shot gbs_*_host result (same kernels, so bit-for-bit) -- residency
//       changes buffer lifetime, never values;
//   (2) reuse: the input buffer grows MONOTONICALLY to the largest bucket and is
//       reused for every smaller/equal one -- a smaller bucket after a bigger one
//       must NOT reallocate (the gbs_session_reallocs witness stays put).
//
//   nvcc -O2 -std=c++17 *.cu check_session.cu -o check_session   (or via shim)

#include <cuComplex.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace gbs {
struct gbs_session;
extern "C" {
int gbs_session_open(gbs_session**);
int gbs_session_evaluate(gbs_session*, int, int, const cuDoubleComplex*, int, int, cuDoubleComplex*);
int gbs_session_evaluate_resident(gbs_session*, int, int, const cuDoubleComplex*, int, int, cuDoubleComplex**);
int gbs_dev_to_host(const cuDoubleComplex*, cuDoubleComplex*, int);
int gbs_dev_free(cuDoubleComplex*);
int gbs_session_close(gbs_session*);
std::size_t gbs_session_reallocs(const gbs_session*);
// one-shot references (the independent answer the session must match)
int gbs_perm_host(const cuDoubleComplex*, int, int, cuDoubleComplex*);
int gbs_perm_dd_host(const cuDoubleComplex*, int, int, cuDoubleComplex*);
int gbs_haf_host(const cuDoubleComplex*, int, int, cuDoubleComplex*);
int gbs_tor_host(const cuDoubleComplex*, int, int, cuDoubleComplex*);
}
}

// A random batch appropriate to `func` (1/2/3 symmetric; 3 real & small-norm).
static std::vector<cuDoubleComplex> make_batch(int func, int dim, int batch, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> Z(0.0, 1.0);
  const bool tor = (func == 3), sym = (func == 1 || func == 2 || func == 3);
  std::vector<cuDoubleComplex> v((size_t)batch * dim * dim);
  for (int b = 0; b < batch; ++b) {
    std::vector<double> re(dim * dim), im(dim * dim, 0.0);
    for (int i = 0; i < dim * dim; ++i) { re[i] = (tor ? 0.1 : 1.0) * Z(rng); im[i] = tor ? 0.0 : Z(rng); }
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j) {
        double rr = re[i * dim + j], ii = im[i * dim + j];
        if (sym) { rr = 0.5 * (re[i * dim + j] + re[j * dim + i]); ii = 0.5 * (im[i * dim + j] + im[j * dim + i]); }
        v[(size_t)b * dim * dim + i * dim + j] = make_cuDoubleComplex(rr, ii);
      }
  }
  return v;
}

static void ref_eval(int func, int prec, const cuDoubleComplex* m, int dim, int batch, cuDoubleComplex* o) {
  if (func == 0 && prec == 0) gbs::gbs_perm_host(m, dim, batch, o);
  else if (func == 0 && prec == 1) gbs::gbs_perm_dd_host(m, dim, batch, o);
  else if (func == 1 && prec == 0) gbs::gbs_haf_host(m, dim, batch, o);
  else if (func == 3 && prec == 0) gbs::gbs_tor_host(m, dim / 2, batch, o);  // tor: 2n x 2n
}

static double worst_diff(const std::vector<cuDoubleComplex>& a, const std::vector<cuDoubleComplex>& b) {
  double w = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double dr = cuCreal(a[i]) - cuCreal(b[i]), di = cuCimag(a[i]) - cuCimag(b[i]);
    w = std::max(w, std::sqrt(dr * dr + di * di));
  }
  return w;
}

int main() {
  std::printf("device-resident session gate (reuse + correctness):\n");
  struct Job { int func, prec, dim, batch; const char* name; size_t expect_reallocs; };
  // Grow (perm d4 -> haf d8), then a sequence of smaller/equal buckets that must
  // all REUSE the d8 buffer: reallocs witnesses 1, 2, then stays 2.
  const Job jobs[] = {
    {0, 0, 4, 32, "perm    d4 b32", 1},   // first alloc
    {1, 0, 8, 64, "haf     d8 b64", 2},   // grow
    {0, 0, 4, 32, "perm    d4 b32", 2},   // reuse (smaller)
    {3, 0, 6, 16, "tor     d6 b16", 2},   // reuse (different func)
    {0, 1, 4,  8, "perm_dd d4 b8 ", 2},   // reuse (dd precision)
    {0, 0, 16, 16, "perm d16 coop", 2},   // reuse (n>=12 -> cooperative dispatch)
  };

  gbs::gbs_session* s = nullptr;
  if (gbs::gbs_session_open(&s) != 0 || !s) { std::printf("  session open FAILED\nFAIL\n"); return 1; }

  int failures = 0;
  for (const Job& j : jobs) {
    auto in = make_batch(j.func, j.dim, j.batch, 0xC0FFEEu + j.dim * 131 + j.batch);
    std::vector<cuDoubleComplex> sout(j.batch), rout(j.batch);
    int rc = gbs::gbs_session_evaluate(s, j.func, j.prec, in.data(), j.dim, j.batch, sout.data());
    ref_eval(j.func, j.prec, in.data(), j.dim, j.batch, rout.data());
    const double w = worst_diff(sout, rout);
    const size_t re = gbs::gbs_session_reallocs(s);
    const bool ok = (rc == 0) && (w < 1e-12) && (re == j.expect_reallocs);
    std::printf("  %-14s  diff %.2e  reallocs %zu (want %zu)  %s\n",
                j.name, w, re, j.expect_reallocs, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
  }
  // v2 device-resident output: the result is left on the device (gbs_dev_to_host
  // brings it back) and must equal the one-shot reference -- residency changes where
  // the result lives, never its value.
  {
    const int func = 1, prec = 0, dim = 8, batch = 32;  // hafnian
    auto in = make_batch(func, dim, batch, 0xDEADBEEFu);
    cuDoubleComplex* d_res = nullptr;
    int rc = gbs::gbs_session_evaluate_resident(s, func, prec, in.data(), dim, batch, &d_res);
    std::vector<cuDoubleComplex> sout(batch), rout(batch);
    if (rc == 0 && d_res) rc = gbs::gbs_dev_to_host(d_res, sout.data(), batch);
    ref_eval(func, prec, in.data(), dim, batch, rout.data());
    const double w = worst_diff(sout, rout);
    const bool ok = (rc == 0) && (d_res != nullptr) && (w < 1e-12);
    std::printf("  %-14s  diff %.2e  (device-resident output)  %s\n", "haf d8 b32 res", w, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
    gbs::gbs_dev_free(d_res);
  }
  gbs::gbs_session_close(s);

  std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
