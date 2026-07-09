// host_api.cu -- host-facing batched entry points (numpy/ctypes-friendly).
//
// The kernels in permanent.cu / hafnian.cu / ... take *device* pointers; the
// Python bindings (bindings/gbskernels_ext.cpp) and any ctypes caller have *host*
// (numpy) data. These wrappers bridge the two: H2D copy -> launch -> sync -> D2H
// copy, with the device allocation managed here. Input/output buffers are plain
// contiguous `cuDoubleComplex` (== numpy complex128 memory layout), so the
// binding just hands over the array pointer.
//
// Structured to also build under the CPU pre-flight shim (cudaMalloc -> malloc,
// launch -> host grid emulation), so the plumbing is validated on CPU
// (check_host_api) before any GPU session.

#include <cuComplex.h>
#include <cstdint>
#include <cstddef>
#include <cstring> // std::memcpy (pageable <-> pinned staging in the v2 session)

namespace gbs {

extern "C" {
void gbs_perm_glynn_fp64_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
void gbs_perm_glynn_dd_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
void gbs_haf_powertrace_fp64_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
void gbs_haf_powertrace_fp64_small_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
void gbs_haf_powertrace_dd_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
void gbs_loop_haf_fp64_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
void gbs_loop_haf_dd_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
void gbs_tor_fp64_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
void gbs_tor_dd_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);
// cooperative permanent (permanent_coop.cu): `groups` threads split the Glynn sum.
int gbs_perm_glynn_coop_batched(const cuDoubleComplex*, int, int, int, cuDoubleComplex*, cudaStream_t);
// FP64 + cancellation indicator (precision="auto"): result + sum|term|, one pass.
void gbs_perm_glynn_fp64_kappa_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*, cudaStream_t);
void gbs_haf_powertrace_fp64_kappa_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*, cudaStream_t);
void gbs_loop_haf_fp64_kappa_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*, cudaStream_t);
void gbs_tor_fp64_kappa_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*, cudaStream_t);
// certified: fp64 value + rigorous |value - exact| bound, one pass.
void gbs_perm_certified_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*, cudaStream_t);
void gbs_haf_certified_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*, cudaStream_t);
void gbs_lhaf_certified_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*, cudaStream_t);
void gbs_tor_certified_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*, cudaStream_t);
void gbs_perm_dd_certified_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*, cudaStream_t);
void gbs_haf_dd_certified_batched(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*, cudaStream_t);
// recursive prefix-Cholesky torontonian, real O; NaN = off-domain.
void gbs_tor_recursive_real_fp64_batched(const double*, int, int, double*, cudaStream_t);
void gbs_tor_recursive_single_batched(const double*, int, int, double*, cudaStream_t);
void gbs_tor_recursive_single_cert_batched(const double*, int, int, double*, double*, cudaStream_t);
void gbs_tor_recursive_single_ddcert_batched(const double*, int, int, double*, double*, cudaStream_t);
// repeated-row loop-hafnian sieve: one shared A/gamma, per-element reps.
void gbs_lhaf_repeated_cert_batched(const cuDoubleComplex*, const cuDoubleComplex*, int,
                                    const int*, int, cuDoubleComplex*, double*, cudaStream_t);
void gbs_lhaf_repeated_batched(const cuDoubleComplex*, const cuDoubleComplex*, int, const int*, int, cuDoubleComplex*, cudaStream_t);
}

using KappaFn = void (*)(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*, cudaStream_t);

using Fn = void (*)(const cuDoubleComplex*, int, int, cuDoubleComplex*, cudaStream_t);

// FP64 permanent dispatch (measured on an RTX 4090, batch 4096): the cooperative
// kernel is ~5x faster from n=12 up, and groups=8 is the sweet spot (groups=32
// over-splits when the grid is already saturated). Below the crossover the
// per-thread kernel wins (no partials buffer / reduction overhead).
constexpr int PERM_COOP_CROSSOVER = 12;
constexpr int PERM_COOP_GROUPS = 8;

// Size-specialized hafnian dispatch (measured on an RTX 4090, batch 4096, matched
// block size): the small-buffer-cap kernel is ~1.07x (N=8) - 1.19x (N=12) faster than
// the full-cap kernel from the smaller per-thread footprint -- a modest but real win,
// and GBS submatrices are mostly small. N <= 12 (= HAF_SMALL_N) routes to it.
constexpr int HAF_SMALL_CROSSOVER = 12;

// alloc_dim = side length of each input matrix; kernel_dim = the integer the
// kernel takes (n for perm/tor, N for haf/lhaf; for tor alloc_dim = 2*kernel_dim).
// Returns 0 on success, otherwise the cudaError_t of the first failing call.
// Every CUDA call (alloc / copy / launch / sync) is checked; a failure
// short-circuits and the device buffers are freed before returning.
static int run_host(Fn fn, const cuDoubleComplex* h_mats, int alloc_dim,
                    int kernel_dim, int batch, cuDoubleComplex* h_out) {
  if (batch <= 0) return 0;
  size_t in_n = (size_t)batch * alloc_dim * alloc_dim;
  cuDoubleComplex *d_in = nullptr, *d_out = nullptr;
  cudaError_t err = cudaMalloc(&d_in, in_n * sizeof(cuDoubleComplex));
  if (err == cudaSuccess) err = cudaMalloc(&d_out, (size_t)batch * sizeof(cuDoubleComplex));
  if (err == cudaSuccess)
    err = cudaMemcpy(d_in, h_mats, in_n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  if (err == cudaSuccess) {
    fn(d_in, kernel_dim, batch, d_out, 0);
    err = cudaGetLastError(); // launch configuration / launch errors
  }
  if (err == cudaSuccess) err = cudaDeviceSynchronize(); // execution errors
  if (err == cudaSuccess)
    err = cudaMemcpy(h_out, d_out, (size_t)batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  cudaFree(d_in); // safe on nullptr
  cudaFree(d_out);
  return (int)err;
}

// Host-pointer wrapper for an FP64+indicator (kappa) kernel: H2D -> launch -> D2H of
// both the result and the per-element sum|term|. `alloc_dim` is the matrix side
// (= 2*kernel_dim for tor). Same error-checking discipline as run_host.
static int run_host_kappa(KappaFn fn, const cuDoubleComplex* h_mats, int alloc_dim,
                          int kernel_dim, int batch, cuDoubleComplex* h_out, double* h_abs) {
  if (batch <= 0) return 0;
  size_t in_n = (size_t)batch * alloc_dim * alloc_dim;
  cuDoubleComplex *d_in = nullptr, *d_out = nullptr;
  double* d_abs = nullptr;
  cudaError_t err = cudaMalloc(&d_in, in_n * sizeof(cuDoubleComplex));
  if (err == cudaSuccess) err = cudaMalloc(&d_out, (size_t)batch * sizeof(cuDoubleComplex));
  if (err == cudaSuccess) err = cudaMalloc(&d_abs, (size_t)batch * sizeof(double));
  if (err == cudaSuccess)
    err = cudaMemcpy(d_in, h_mats, in_n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  if (err == cudaSuccess) {
    fn(d_in, kernel_dim, batch, d_out, d_abs, 0);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  if (err == cudaSuccess)
    err = cudaMemcpy(h_out, d_out, (size_t)batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  if (err == cudaSuccess)
    err = cudaMemcpy(h_abs, d_abs, (size_t)batch * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_in); cudaFree(d_out); cudaFree(d_abs);
  return (int)err;
}

// Host-pointer wrapper for the cooperative permanent: H2D -> device map/reduce
// (which manages its own partials buffer + sync) -> D2H. Same shape as run_host
// but for the coop kernel's signature. Validated by check_host_api at n>=crossover.
static int run_host_perm_coop(const cuDoubleComplex* h_mats, int n, int batch,
                              cuDoubleComplex* h_out) {
  if (batch <= 0) return 0;
  size_t in_n = (size_t)batch * n * n;
  cuDoubleComplex *d_in = nullptr, *d_out = nullptr;
  cudaError_t err = cudaMalloc(&d_in, in_n * sizeof(cuDoubleComplex));
  if (err == cudaSuccess) err = cudaMalloc(&d_out, (size_t)batch * sizeof(cuDoubleComplex));
  if (err == cudaSuccess)
    err = cudaMemcpy(d_in, h_mats, in_n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  if (err == cudaSuccess) {
    int rc = gbs_perm_glynn_coop_batched(d_in, n, batch, PERM_COOP_GROUPS, d_out, 0);
    if (rc != 0) err = (cudaError_t)rc; // the coop wrapper has already synchronized
  }
  if (err == cudaSuccess)
    err = cudaMemcpy(h_out, d_out, (size_t)batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_out);
  return (int)err;
}

// Host-facing batched API. Returns 0 on success or a nonzero cudaError_t (the
// binding raises on nonzero). `n` is the matrix size for perm (n x n) and the
// mode count for tor (matrices are 2n x 2n); `N` is the matrix size for haf/lhaf.
extern "C" {
int gbs_perm_host(const cuDoubleComplex* m, int n, int b, cuDoubleComplex* o) {
  // auto-dispatch large permanents to the cooperative kernel (perf; transparent).
  if (n >= PERM_COOP_CROSSOVER) return run_host_perm_coop(m, n, b, o);
  return run_host(gbs_perm_glynn_fp64_batched, m, n, n, b, o);
}
int gbs_perm_dd_host(const cuDoubleComplex* m, int n, int b, cuDoubleComplex* o) {
  return run_host(gbs_perm_glynn_dd_batched, m, n, n, b, o);
}
int gbs_haf_host(const cuDoubleComplex* m, int N, int b, cuDoubleComplex* o) {
  // small N -> the size-specialized kernel (measured ~1.07-1.19x; same value).
  if (N >= 2 && !(N & 1) && N <= HAF_SMALL_CROSSOVER)
    return run_host(gbs_haf_powertrace_fp64_small_batched, m, N, N, b, o);
  return run_host(gbs_haf_powertrace_fp64_batched, m, N, N, b, o);
}
int gbs_haf_dd_host(const cuDoubleComplex* m, int N, int b, cuDoubleComplex* o) {
  return run_host(gbs_haf_powertrace_dd_batched, m, N, N, b, o);
}
int gbs_lhaf_host(const cuDoubleComplex* m, int N, int b, cuDoubleComplex* o) {
  return run_host(gbs_loop_haf_fp64_batched, m, N, N, b, o);
}
int gbs_lhaf_dd_host(const cuDoubleComplex* m, int N, int b, cuDoubleComplex* o) {
  return run_host(gbs_loop_haf_dd_batched, m, N, N, b, o);
}
int gbs_tor_host(const cuDoubleComplex* m, int n, int b, cuDoubleComplex* o) {
  // Dispatch: all-real (physical-domain) fp64 batches
  // go to the recursive prefix-Cholesky kernel -- MEASURED on a 4090 at 4.5x/7.0x/
  // 9.3x over this complex-LU kernel at 2n = 8/12/16 (results/throughput/, commit
  // 6b3f2e8) -- with the off-domain contract: any NaN out (a non-SPD minor) falls
  // back to the complex-LU kernel for the whole batch, never a silent wrong value.
  // Complex input keeps the complex-LU path. Routing status: the one-shot API
  // (here) and the NON-resident Session path (gbs_session_evaluate delegates to
  // this function) both dispatch; the RESIDENT Session tor keeps complex-LU
  // because its contract is a device-resident cuDoubleComplex output and the
  // recursive kernel emits real doubles -- a real->complex device conversion
  // kernel is deferred until a resident-tor consumer exists (outputs are one
  // value per matrix, so the skipped D2H is negligible; the dispatch win does
  // not apply to the copy).
  const int dim = 2 * n;
  const size_t total = (size_t)b * dim * dim;
  bool all_real = true;
  for (size_t i = 0; i < total; ++i)
    if (cuCimag(m[i]) != 0.0) { all_real = false; break; }
  if (all_real && b > 0 && n > 0) {
    double* h_real = (double*)malloc(total * sizeof(double));
    double* h_out = (double*)malloc((size_t)b * sizeof(double));
    if (h_real && h_out) {
      for (size_t i = 0; i < total; ++i) h_real[i] = cuCreal(m[i]);
      double *d_in = nullptr, *d_out = nullptr;
      cudaError_t err = cudaMalloc(&d_in, total * sizeof(double));
      if (err == cudaSuccess) err = cudaMalloc(&d_out, (size_t)b * sizeof(double));
      if (err == cudaSuccess)
        err = cudaMemcpy(d_in, h_real, total * sizeof(double), cudaMemcpyHostToDevice);
      if (err == cudaSuccess) {
        gbs_tor_recursive_real_fp64_batched(d_in, n, b, d_out, 0);
        err = cudaGetLastError();
      }
      if (err == cudaSuccess) err = cudaDeviceSynchronize();
      if (err == cudaSuccess)
        err = cudaMemcpy(h_out, d_out, (size_t)b * sizeof(double), cudaMemcpyDeviceToHost);
      cudaFree(d_in); cudaFree(d_out);
      if (err == cudaSuccess) {
        bool clean = true;
        for (int i = 0; i < b; ++i)
          if (h_out[i] != h_out[i]) { clean = false; break; } // NaN: off-domain
        if (clean) {
          for (int i = 0; i < b; ++i) o[i] = make_cuDoubleComplex(h_out[i], 0.0);
          free(h_real); free(h_out);
          return 0;
        }
      }
    }
    free(h_real); free(h_out); // fall through to the complex-LU kernel
  }
  return run_host(gbs_tor_fp64_batched, m, 2 * n, n, b, o); // matrices are 2n x 2n
}
int gbs_tor_dd_host(const cuDoubleComplex* m, int n, int b, cuDoubleComplex* o) {
  return run_host(gbs_tor_dd_batched, m, 2 * n, n, b, o);
}

// FP64 + cancellation indicator (precision="auto"): fill o (result) and absnorm
// (sum|term|) so the caller forms kappa = absnorm/|o| per element and reruns risky
// ones in DD. perm/haf/lhaf take the matrix side; tor takes n modes (matrices 2n x 2n).
int gbs_perm_kappa_host(const cuDoubleComplex* m, int n, int b, cuDoubleComplex* o, double* a) {
  return run_host_kappa(gbs_perm_glynn_fp64_kappa_batched, m, n, n, b, o, a);
}
int gbs_haf_kappa_host(const cuDoubleComplex* m, int N, int b, cuDoubleComplex* o, double* a) {
  return run_host_kappa(gbs_haf_powertrace_fp64_kappa_batched, m, N, N, b, o, a);
}
int gbs_lhaf_kappa_host(const cuDoubleComplex* m, int N, int b, cuDoubleComplex* o, double* a) {
  return run_host_kappa(gbs_loop_haf_fp64_kappa_batched, m, N, N, b, o, a);
}
int gbs_tor_kappa_host(const cuDoubleComplex* m, int n, int b, cuDoubleComplex* o, double* a) {
  return run_host_kappa(gbs_tor_fp64_kappa_batched, m, 2 * n, n, b, o, a);
}

// Certified: o = the fp64 value, e = a rigorous bound on |o - exact|
// (same marshalling shape as the kappa wrappers: complex values + double array).
int gbs_perm_certified_host(const cuDoubleComplex* m, int n, int b, cuDoubleComplex* o, double* e) {
  return run_host_kappa(gbs_perm_certified_batched, m, n, n, b, o, e);
}
int gbs_haf_certified_host(const cuDoubleComplex* m, int N, int b, cuDoubleComplex* o, double* e) {
  return run_host_kappa(gbs_haf_certified_batched, m, N, N, b, o, e);
}
int gbs_lhaf_certified_host(const cuDoubleComplex* m, int N, int b, cuDoubleComplex* o, double* e) {
  return run_host_kappa(gbs_lhaf_certified_batched, m, N, N, b, o, e);
}
int gbs_tor_certified_host(const cuDoubleComplex* m, int n, int b, cuDoubleComplex* o, double* e) {
  return run_host_kappa(gbs_tor_certified_batched, m, 2 * n, n, b, o, e);
}
int gbs_perm_dd_certified_host(const cuDoubleComplex* m, int n, int b, cuDoubleComplex* o, double* e) {
  return run_host_kappa(gbs_perm_dd_certified_batched, m, n, n, b, o, e);
}
int gbs_haf_dd_certified_host(const cuDoubleComplex* m, int N, int b, cuDoubleComplex* o, double* e) {
  return run_host_kappa(gbs_haf_dd_certified_batched, m, N, N, b, o, e);
}

// SINGLE-LARGE recursive torontonian (real O, dim <= 64): one evaluation split
// across 2^g subtrees; the host sums the partials (NaN propagates: off-domain).
int gbs_tor_recursive_single_host(const double* h_O, int n, int g, double* out) {
  const int dim = 2 * n;
  const uint64_t nsub = 1ull << g;
  double *d_O = nullptr, *d_p = nullptr;
  cudaError_t err = cudaMalloc(&d_O, (size_t)dim * dim * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_p, (size_t)nsub * sizeof(double));
  if (err == cudaSuccess)
    err = cudaMemcpy(d_O, h_O, (size_t)dim * dim * sizeof(double), cudaMemcpyHostToDevice);
  double* h_p = nullptr;
  if (err == cudaSuccess) {
    gbs_tor_recursive_single_batched(d_O, n, g, d_p, 0);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  if (err == cudaSuccess) {
    h_p = (double*)malloc((size_t)nsub * sizeof(double));
    if (h_p) err = cudaMemcpy(h_p, d_p, (size_t)nsub * sizeof(double), cudaMemcpyDeviceToHost);
    else { cudaFree(d_O); cudaFree(d_p); return 1; } // host OOM
  }
  cudaFree(d_O); cudaFree(d_p);
  if (err != cudaSuccess) { free(h_p); return (int)err; }
  double total = 0.0;
  for (uint64_t i = 0; i < nsub; ++i) total += h_p[i]; // NaN propagates
  free(h_p);
  *out = total;
  return 0;
}

// CERTIFIED single-large: value + rigorous bound. Host sums value partials and
// upward-sums bound partials (nextafter: each add's rounding is absorbed), plus
// one u*|running| term per add for the value summation's own rounding.
int gbs_tor_single_certified_host(const double* h_O, int n, int g,
                                  double* out, double* bound) {
  const int dim = 2 * n;
  const uint64_t nsub = 1ull << g;
  double *d_O = nullptr, *d_p = nullptr, *d_b = nullptr;
  cudaError_t err = cudaMalloc(&d_O, (size_t)dim * dim * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_p, (size_t)nsub * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_b, (size_t)nsub * sizeof(double));
  if (err == cudaSuccess)
    err = cudaMemcpy(d_O, h_O, (size_t)dim * dim * sizeof(double), cudaMemcpyHostToDevice);
  double *h_p = nullptr, *h_b = nullptr;
  if (err == cudaSuccess) {
    gbs_tor_recursive_single_cert_batched(d_O, n, g, d_p, d_b, 0);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  if (err == cudaSuccess) {
    h_p = (double*)malloc((size_t)nsub * sizeof(double));
    h_b = (double*)malloc((size_t)nsub * sizeof(double));
    if (h_p && h_b) {
      err = cudaMemcpy(h_p, d_p, (size_t)nsub * sizeof(double), cudaMemcpyDeviceToHost);
      if (err == cudaSuccess)
        err = cudaMemcpy(h_b, d_b, (size_t)nsub * sizeof(double), cudaMemcpyDeviceToHost);
    } else {
      cudaFree(d_O); cudaFree(d_p); cudaFree(d_b); free(h_p); free(h_b);
      return 1;
    }
  }
  cudaFree(d_O); cudaFree(d_p); cudaFree(d_b);
  if (err != cudaSuccess) { free(h_p); free(h_b); return (int)err; }
  double total = 0.0, e_tot = 0.0;
  const double u = 1.1102230246251565e-16;
  for (uint64_t i = 0; i < nsub; ++i) {
    total += h_p[i]; // NaN propagates
    e_tot = nextafter(e_tot + h_b[i], INFINITY); // upward bound sum
    e_tot = nextafter(e_tot + u * fabs(total), INFINITY);
  }
  free(h_p); free(h_b);
  *out = total;
  *bound = e_tot;
  return 0;
}

// Certified DOUBLE-DOUBLE single-large torontonian: same host plumbing as the
// fp64 certified path, but the device kernel carries the value in double-double.
// The host partial-sum stays fp64 (the kernel collapses each partial to fp64),
// so the host charges fp64 u per accumulation as before.
int gbs_tor_single_ddcertified_host(const double* h_O, int n, int g,
                                    double* out, double* bound) {
  const int dim = 2 * n;
  const uint64_t nsub = 1ull << g;
  double *d_O = nullptr, *d_p = nullptr, *d_b = nullptr;
  cudaError_t err = cudaMalloc(&d_O, (size_t)dim * dim * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_p, (size_t)nsub * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_b, (size_t)nsub * sizeof(double));
  if (err == cudaSuccess)
    err = cudaMemcpy(d_O, h_O, (size_t)dim * dim * sizeof(double), cudaMemcpyHostToDevice);
  double *h_p = nullptr, *h_b = nullptr;
  if (err == cudaSuccess) {
    gbs_tor_recursive_single_ddcert_batched(d_O, n, g, d_p, d_b, 0);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  if (err == cudaSuccess) {
    h_p = (double*)malloc((size_t)nsub * sizeof(double));
    h_b = (double*)malloc((size_t)nsub * sizeof(double));
    if (h_p && h_b) {
      err = cudaMemcpy(h_p, d_p, (size_t)nsub * sizeof(double), cudaMemcpyDeviceToHost);
      if (err == cudaSuccess)
        err = cudaMemcpy(h_b, d_b, (size_t)nsub * sizeof(double), cudaMemcpyDeviceToHost);
    } else {
      cudaFree(d_O); cudaFree(d_p); cudaFree(d_b); free(h_p); free(h_b);
      return 1;
    }
  }
  cudaFree(d_O); cudaFree(d_p); cudaFree(d_b);
  if (err != cudaSuccess) { free(h_p); free(h_b); return (int)err; }
  double total = 0.0, e_tot = 0.0;
  const double u = 1.1102230246251565e-16;
  for (uint64_t i = 0; i < nsub; ++i) {
    total += h_p[i];
    e_tot = nextafter(e_tot + h_b[i], INFINITY);
    e_tot = nextafter(e_tot + u * fabs(total), INFINITY);
  }
  free(h_p); free(h_b);
  *out = total;
  *bound = e_tot;
  return 0;
}

// Repeated-row loop-hafnian sieve: one M x M base matrix + gamma
// shared by the launch, a (batch x M) int reps table, one value per element.
int gbs_lhaf_repeated_host(const cuDoubleComplex* hA, const cuDoubleComplex* hg, int M,
                           const int* h_reps, int batch, cuDoubleComplex* h_out) {
  if (batch <= 0) return 0;
  cuDoubleComplex *dA = nullptr, *dg = nullptr, *dout = nullptr;
  int* dreps = nullptr;
  cudaError_t err = cudaMalloc(&dA, (size_t)M * M * sizeof(cuDoubleComplex));
  if (err == cudaSuccess) err = cudaMalloc(&dg, (size_t)M * sizeof(cuDoubleComplex));
  if (err == cudaSuccess) err = cudaMalloc(&dreps, (size_t)batch * M * sizeof(int));
  if (err == cudaSuccess) err = cudaMalloc(&dout, (size_t)batch * sizeof(cuDoubleComplex));
  if (err == cudaSuccess) err = cudaMemcpy(dA, hA, (size_t)M * M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  if (err == cudaSuccess) err = cudaMemcpy(dg, hg, (size_t)M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  if (err == cudaSuccess) err = cudaMemcpy(dreps, h_reps, (size_t)batch * M * sizeof(int), cudaMemcpyHostToDevice);
  if (err == cudaSuccess) {
    gbs_lhaf_repeated_batched(dA, dg, M, dreps, batch, dout, 0);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  if (err == cudaSuccess)
    err = cudaMemcpy(h_out, dout, (size_t)batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  cudaFree(dA); cudaFree(dg); cudaFree(dreps); cudaFree(dout);
  return (int)err;
}

// Certified sieve: values + rigorous bounds (kernel: lhaf_repeated_cert_kernel).
int gbs_lhaf_repeated_cert_host(const cuDoubleComplex* hA, const cuDoubleComplex* hg, int M,
                                const int* h_reps, int batch, cuDoubleComplex* h_out,
                                double* h_bounds) {
  if (batch <= 0) return 0;
  cuDoubleComplex *dA = nullptr, *dg = nullptr, *dout = nullptr;
  double* db = nullptr;
  int* dreps = nullptr;
  cudaError_t err = cudaMalloc(&dA, (size_t)M * M * sizeof(cuDoubleComplex));
  if (err == cudaSuccess) err = cudaMalloc(&dg, (size_t)M * sizeof(cuDoubleComplex));
  if (err == cudaSuccess) err = cudaMalloc(&dreps, (size_t)batch * M * sizeof(int));
  if (err == cudaSuccess) err = cudaMalloc(&dout, (size_t)batch * sizeof(cuDoubleComplex));
  if (err == cudaSuccess) err = cudaMalloc(&db, (size_t)batch * sizeof(double));
  if (err == cudaSuccess) err = cudaMemcpy(dA, hA, (size_t)M * M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  if (err == cudaSuccess) err = cudaMemcpy(dg, hg, (size_t)M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  if (err == cudaSuccess) err = cudaMemcpy(dreps, h_reps, (size_t)batch * M * sizeof(int), cudaMemcpyHostToDevice);
  if (err == cudaSuccess) {
    gbs_lhaf_repeated_cert_batched(dA, dg, M, dreps, batch, dout, db, 0);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  if (err == cudaSuccess)
    err = cudaMemcpy(h_out, dout, (size_t)batch * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  if (err == cudaSuccess)
    err = cudaMemcpy(h_bounds, db, (size_t)batch * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(dA); cudaFree(dg); cudaFree(dreps); cudaFree(dout); cudaFree(db);
  return (int)err;
}
}

// --- device-resident session (docs/device_resident_contract.md) -------------
// A session owns reusable d_in/d_out device buffers that GROW MONOTONICALLY to
// the largest bucket seen and are reused for every smaller/equal one -- so a
// sampler loop of many differently-sized launches pays allocation once, not per
// call. v1 still copies H2D/D2H per evaluate (the host assembles each step's
// submatrices); it amortizes allocation, not transfer. Same kernels as the
// one-shot host API above, so results are identical; validated on the CPU shim
// by check_session.cu before any GPU session.

// (func, precision) -> kernel. func: 0 perm, 1 haf, 2 lhaf, 3 tor. prec: 0 fp64, 1 dd.
static Fn kernel_for(int func, int prec) {
  switch (func) {
    case 0: return prec ? gbs_perm_glynn_dd_batched : gbs_perm_glynn_fp64_batched;
    case 1: return prec ? gbs_haf_powertrace_dd_batched : gbs_haf_powertrace_fp64_batched;
    case 2: return prec ? gbs_loop_haf_dd_batched : gbs_loop_haf_fp64_batched;
    case 3: return prec ? gbs_tor_dd_batched : gbs_tor_fp64_batched;
    default: return nullptr;
  }
}

struct gbs_session {
  cuDoubleComplex* d_in = nullptr;
  cuDoubleComplex* d_out = nullptr;
  cuDoubleComplex* h_pin_in = nullptr; // pinned host staging for H2D (v2)
  cuDoubleComplex* h_pin_out = nullptr; // pinned host staging for D2H (v2)
  size_t in_cap = 0; // elements currently allocated for d_in
  size_t out_cap = 0; // elements currently allocated for d_out
  size_t pin_in_cap = 0;
  size_t pin_out_cap = 0;
  cudaStream_t stream = nullptr; // persistent stream: async H2D / launch / D2H (v2)
  size_t reallocs = 0; // times d_in grew -- the residency-reuse witness (gate)
};

// Grow *buf to hold >= need elements, reusing it when it is already big enough.
// On growth, count it (the gate asserts a smaller bucket does NOT grow it). On
// failure, leave *buf freed and *cap=0 so the next call retries cleanly.
static int ensure_cap(cuDoubleComplex** buf, size_t* cap, size_t need, size_t* grew) {
  if (need <= *cap) return cudaSuccess;
  if (*buf) cudaFree(*buf);
  *buf = nullptr;
  cudaError_t err = cudaMalloc(buf, need * sizeof(cuDoubleComplex));
  if (err == cudaSuccess) { *cap = need; if (grew) ++(*grew); }
  else { *cap = 0; }
  return (int)err;
}

// Same, for PINNED host staging (cudaMallocHost): page-locked memory makes the H2D/
// D2H DMA asynchronous and overlappable. Reused/grown like the device buffers.
static int ensure_pinned(cuDoubleComplex** buf, size_t* cap, size_t need) {
  if (need <= *cap) return cudaSuccess;
  if (*buf) cudaFreeHost(*buf);
  *buf = nullptr;
  cudaError_t err = cudaMallocHost((void**)buf, need * sizeof(cuDoubleComplex));
  if (err == cudaSuccess) { *cap = need; } else { *cap = 0; }
  return (int)err;
}

extern "C" {
// Open a session. Returns 0 and *out a handle, or a nonzero cudaError_t. Creates the
// persistent stream the v2 async copy path runs on.
int gbs_session_open(gbs_session** out) {
  if (!out) return (int)cudaErrorInvalidValue;
  gbs_session* s = new gbs_session();
  cudaError_t err = cudaStreamCreate(&s->stream);
  if (err != cudaSuccess) { delete s; *out = nullptr; return (int)err; }
  *out = s;
  return 0;
}

// Evaluate one uniform bucket through the resident buffers. `dim` is the matrix
// side length (= 2*modes for tor); `func`/`prec` select the kernel. Buffers are
// reused/grown; a failure leaves the session usable (buffers intact). Returns 0
// or a nonzero cudaError_t.
int gbs_session_evaluate(gbs_session* s, int func, int prec,
                         const cuDoubleComplex* h_mats, int dim, int batch,
                         cuDoubleComplex* h_out) {
  if (!s) return (int)cudaErrorInvalidValue;
  if (batch <= 0) return 0;
  Fn fn = kernel_for(func, prec);
  if (!fn) return (int)cudaErrorInvalidValue;
  const int kernel_dim = (func == 3) ? dim / 2 : dim; // tor matrices are 2n x 2n
  if (func == 3 && prec == 0) {
    // Dispatch parity with the one-shot API: all-real physical batches go to
    // the recursive prefix-Cholesky kernel (measured 4.5-9.3x), complex or
    // off-domain falls back to the LU -- gbs_tor_host holds that logic. The
    // session's complex-buffer residency does not apply to the real-path
    // scratch; the kernel win dominates the allocation amortization it forgoes.
    // (The RESIDENT tor route below keeps the LU kernel: its contract is a
    // device-resident complex output, which the real-output recursive kernel
    // would need a conversion kernel to honor.)
    return gbs_tor_host(h_mats, kernel_dim, batch, h_out);
  }
  // Same FP64-permanent auto-dispatch as gbs_perm_host: large perms go cooperative;
  // and small FP64 hafnians go to the size-specialized kernel (the sampler's regime).
  const bool use_coop = (func == 0 && prec == 0 && dim >= PERM_COOP_CROSSOVER);
  const bool use_haf_small = (func == 1 && prec == 0 && !(dim & 1) && dim <= HAF_SMALL_CROSSOVER);
  const size_t in_n = (size_t)batch * dim * dim;
  const size_t out_n = (size_t)batch;
  // Resident device buffers + pinned host staging, all reused/grown across calls.
  cudaError_t err = (cudaError_t)ensure_cap(&s->d_in, &s->in_cap, in_n, &s->reallocs);
  if (err == cudaSuccess) err = (cudaError_t)ensure_cap(&s->d_out, &s->out_cap, out_n, nullptr);
  if (err == cudaSuccess) err = (cudaError_t)ensure_pinned(&s->h_pin_in, &s->pin_in_cap, in_n);
  if (err == cudaSuccess) err = (cudaError_t)ensure_pinned(&s->h_pin_out, &s->pin_out_cap, out_n);
  // pageable caller input -> pinned staging -> async H2D on the stream.
  if (err == cudaSuccess) {
    std::memcpy(s->h_pin_in, h_mats, in_n * sizeof(cuDoubleComplex));
    err = cudaMemcpyAsync(s->d_in, s->h_pin_in, in_n * sizeof(cuDoubleComplex),
                          cudaMemcpyHostToDevice, s->stream);
  }
  if (err == cudaSuccess) {
    if (use_coop) { // reuses the resident d_in/d_out; coop manages its own partials + sync
      int rc = gbs_perm_glynn_coop_batched(s->d_in, dim, batch, PERM_COOP_GROUPS, s->d_out, s->stream);
      if (rc != 0) err = (cudaError_t)rc;
    } else {
      if (use_haf_small) gbs_haf_powertrace_fp64_small_batched(s->d_in, dim, batch, s->d_out, s->stream);
      else fn(s->d_in, kernel_dim, batch, s->d_out, s->stream); // launch on the session stream
      err = cudaGetLastError();
    }
  }
  // async D2H into pinned staging, then sync the stream and copy out to the caller.
  if (err == cudaSuccess)
    err = cudaMemcpyAsync(s->h_pin_out, s->d_out, out_n * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToHost, s->stream);
  if (err == cudaSuccess) err = cudaStreamSynchronize(s->stream);
  if (err == cudaSuccess) std::memcpy(h_out, s->h_pin_out, out_n * sizeof(cuDoubleComplex));
  return (int)err;
}

// Free the resident buffers, pinned staging, and stream, and the handle. Safe on
// nullptr; returns 0.
int gbs_session_close(gbs_session* s) {
  if (!s) return 0;
  cudaFree(s->d_in); // safe on nullptr
  cudaFree(s->d_out);
  cudaFreeHost(s->h_pin_in);
  cudaFreeHost(s->h_pin_out);
  if (s->stream) cudaStreamDestroy(s->stream);
  delete s;
  return 0;
}

// Number of times the input buffer grew -- the gate uses this to witness reuse.
size_t gbs_session_reallocs(const gbs_session* s) { return s ? s->reallocs : 0; }

// --- device-resident output (Workspace v2) ---------------------------------
// Like gbs_session_evaluate, but the result is left ON THE DEVICE in a fresh buffer
// (owned by the caller via *d_result) and NOT copied back to the host -- so a
// downstream GPU consumer (CuPy/PyTorch via DLPack) reads it zero-copy. Input still
// uses the resident d_in + pinned staging; only the D2H of the result is skipped.
int gbs_session_evaluate_resident(gbs_session* s, int func, int prec,
                                  const cuDoubleComplex* h_mats, int dim, int batch,
                                  cuDoubleComplex** d_result) {
  if (!s || !d_result) return (int)cudaErrorInvalidValue;
  *d_result = nullptr;
  if (batch <= 0) return 0;
  Fn fn = kernel_for(func, prec);
  if (!fn) return (int)cudaErrorInvalidValue;
  const int kernel_dim = (func == 3) ? dim / 2 : dim;
  const bool use_coop = (func == 0 && prec == 0 && dim >= PERM_COOP_CROSSOVER);
  const bool use_haf_small = (func == 1 && prec == 0 && !(dim & 1) && dim <= HAF_SMALL_CROSSOVER);
  const size_t in_n = (size_t)batch * dim * dim, out_n = (size_t)batch;
  cudaError_t err = (cudaError_t)ensure_cap(&s->d_in, &s->in_cap, in_n, &s->reallocs);
  if (err == cudaSuccess) err = (cudaError_t)ensure_pinned(&s->h_pin_in, &s->pin_in_cap, in_n);
  cuDoubleComplex* d_res = nullptr; // fresh buffer owned by the returned handle
  if (err == cudaSuccess) err = cudaMalloc(&d_res, out_n * sizeof(cuDoubleComplex));
  if (err == cudaSuccess) {
    std::memcpy(s->h_pin_in, h_mats, in_n * sizeof(cuDoubleComplex));
    err = cudaMemcpyAsync(s->d_in, s->h_pin_in, in_n * sizeof(cuDoubleComplex),
                          cudaMemcpyHostToDevice, s->stream);
  }
  if (err == cudaSuccess) {
    if (use_coop) {
      int rc = gbs_perm_glynn_coop_batched(s->d_in, dim, batch, PERM_COOP_GROUPS, d_res, s->stream);
      if (rc != 0) err = (cudaError_t)rc;
    } else {
      if (use_haf_small) gbs_haf_powertrace_fp64_small_batched(s->d_in, dim, batch, d_res, s->stream);
      else fn(s->d_in, kernel_dim, batch, d_res, s->stream);
      err = cudaGetLastError();
    }
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(s->stream); // result ready on device, no D2H
  if (err != cudaSuccess) { cudaFree(d_res); return (int)err; }
  *d_result = d_res;
  return 0;
}

// Copy a device-resident result to host (the handle's materialize / .numpy()).
int gbs_dev_to_host(const cuDoubleComplex* d, cuDoubleComplex* h, int n) {
  if (n <= 0) return 0;
  return (int)cudaMemcpy(h, d, (size_t)n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
}

// Free a device-resident result buffer (the handle's owner deleter).
int gbs_dev_free(cuDoubleComplex* d) { cudaFree(d); return 0; }
}

} // namespace gbs
