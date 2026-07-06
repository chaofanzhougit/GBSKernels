// gbskernels_ext.cpp -- nanobind module exposing the GPU batched kernels.
//
// Thin numpy <-> host_api glue: takes a contiguous complex128 stack of matrices,
// hands its pointer to the host-facing wrappers in core/host_api.cu (which do the
// H2D/launch/D2H), and returns a complex128 vector of results. The device work
// and precision tiers live in core/; this file is only marshalling.
//
// Built by bindings/CMakeLists.txt (scikit-build-core + nanobind) against the
// CUDA kernels under nvcc in a rented-GPU session. The Python package
// (gbskernels) imports it lazily for the GPU backend and falls back to the CPU
// reference when it is absent.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <complex>
#include <cstddef>
#include <cstdint>

#include <cuComplex.h>

namespace nb = nanobind;

namespace gbs {
extern "C" {  // each returns 0 on success or a nonzero cudaError_t
int gbs_perm_host(const cuDoubleComplex*, int, int, cuDoubleComplex*);
int gbs_perm_dd_host(const cuDoubleComplex*, int, int, cuDoubleComplex*);
int gbs_haf_host(const cuDoubleComplex*, int, int, cuDoubleComplex*);
int gbs_haf_dd_host(const cuDoubleComplex*, int, int, cuDoubleComplex*);
int gbs_lhaf_host(const cuDoubleComplex*, int, int, cuDoubleComplex*);
int gbs_lhaf_dd_host(const cuDoubleComplex*, int, int, cuDoubleComplex*);
int gbs_tor_host(const cuDoubleComplex*, int, int, cuDoubleComplex*);
int gbs_tor_dd_host(const cuDoubleComplex*, int, int, cuDoubleComplex*);
// FP64 + cancellation indicator (precision="auto"): fills value + absnorm. tor takes
// n modes (matrices 2n x 2n); perm/haf/lhaf take the matrix side.
int gbs_perm_kappa_host(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*);
int gbs_haf_kappa_host(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*);
int gbs_lhaf_kappa_host(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*);
int gbs_tor_kappa_host(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*);
int gbs_perm_certified_host(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*);
int gbs_haf_certified_host(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*);
int gbs_lhaf_certified_host(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*);
int gbs_tor_certified_host(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*);
int gbs_perm_dd_certified_host(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*);
int gbs_haf_dd_certified_host(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*);
int gbs_tor_recursive_single_host(const double*, int, int, double*);
int gbs_tor_single_certified_host(const double*, int, int, double*, double*);
int gbs_lhaf_repeated_host(const cuDoubleComplex*, const cuDoubleComplex*, int, const int*, int, cuDoubleComplex*);
int gbs_lhaf_repeated_cert_host(const cuDoubleComplex*, const cuDoubleComplex*, int, const int*, int, cuDoubleComplex*, double*);

// Device-resident session (core/host_api.cu): reusable buffers across calls.
struct gbs_session;  // opaque
int gbs_session_open(gbs_session**);
int gbs_session_evaluate(gbs_session*, int, int, const cuDoubleComplex*, int, int, cuDoubleComplex*);
int gbs_session_close(gbs_session*);
std::size_t gbs_session_reallocs(const gbs_session*);
// v2 device-resident output: result stays on the device (DLPack handle).
int gbs_session_evaluate_resident(gbs_session*, int, int, const cuDoubleComplex*, int, int, cuDoubleComplex**);
int gbs_dev_to_host(const cuDoubleComplex*, cuDoubleComplex*, int);
int gbs_dev_free(cuDoubleComplex*);
// v3 fully on-device conditional sampler (core/sampler_session.cu): host reduced A-matrices
// {A_k} (concatenated; offsets) + 1/j! -> num_draws*M photon counts. The gather, variable-N
// hafnian, inverse-CDF + cuRAND draw, and the prefix state all stay on the device.
int gbs_sampler_sample(const cuDoubleComplex*, int, const int*, int, int, int, int,
                       const double*, unsigned long long, int*);
}
}


using Cd = std::complex<double>;
using HostFn = int (*)(const cuDoubleComplex*, int, int, cuDoubleComplex*);
using Stack = nb::ndarray<const Cd, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
using Vec = nb::ndarray<nb::numpy, Cd, nb::ndim<1>>;

// The v2 device-resident output is returned as a framework array that views the
// result buffer ZERO-COPY (no D2H in the binding) and is DLPack-exportable. The
// framework + device match the build: a host-shim build's "device" memory IS host
// memory, so it returns a numpy array on the kDLCPU device (consumable + validated on
// CPU); a real nvcc build returns a CuPy array on the kDLCUDA device, which PyTorch /
// JAX also consume zero-copy via DLPack (`torch.from_dlpack`, `jax.dlpack`).
#ifdef GBS_HOST_SHIM
using DevArr = nb::ndarray<nb::numpy, Cd, nb::ndim<1>>;
static constexpr int GBS_DLPACK_DEVICE = nb::device::cpu::value;
#else
using DevArr = nb::ndarray<nb::cupy, Cd, nb::ndim<1>>;
static constexpr int GBS_DLPACK_DEVICE = nb::device::cuda::value;
#endif

// Run `fn` over a (B, d, d) complex128 stack; `kernel_dim` is the integer the
// kernel takes (= d for perm/haf/lhaf, = d/2 for tor). Returns a (B,) vector.
// Per-function preconditions (size limits, even-N for the loop hafnian, real
// input for the DD torontonian) are validated in the Python layer
// (gbskernels._gpu_batched); here we validate squareness, propagate any CUDA
// error from the device call as a Python exception, and never leak the buffer.
static Vec eval(HostFn fn, const Stack& a, bool tor) {
  const size_t B = a.shape(0), d = a.shape(1);
  if (a.shape(2) != d)
    throw std::invalid_argument("gbskernels: each matrix must be square (B, d, d)");
  if (tor && (d % 2 != 0))
    throw std::invalid_argument("torontonian: matrix dimension must be even (2n x 2n)");

  const auto* in = reinterpret_cast<const cuDoubleComplex*>(a.data());
  auto* out = new Cd[B == 0 ? 1 : B];
  const int kernel_dim = tor ? static_cast<int>(d / 2) : static_cast<int>(d);
  int rc = fn(in, kernel_dim, static_cast<int>(B), reinterpret_cast<cuDoubleComplex*>(out));
  if (rc != 0) {
    delete[] out;
    throw std::runtime_error("gbskernels: CUDA error " + std::to_string(rc) +
                             " in GPU kernel launch/execution");
  }
  nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<Cd*>(p); });
  return Vec(out, {B}, owner);
}

using DVec = nb::ndarray<nb::numpy, double, nb::ndim<1>>;
using KappaHostFn = int (*)(const cuDoubleComplex*, int, int, cuDoubleComplex*, double*);

// FP64 + cancellation indicator (precision="auto"): returns (values, absnorms),
// absnorms[b] = sum|term| so the Python layer forms kappa = absnorm/|value| and reruns
// risky elements in DD. `tor` halves the kernel dim (matrices are 2n x 2n).
static nb::tuple kappa_eval(KappaHostFn fn, const Stack& a, bool tor) {
  const size_t B = a.shape(0), d = a.shape(1);
  if (a.shape(2) != d)
    throw std::invalid_argument("gbskernels: each matrix must be square (B, d, d)");
  if (tor && (d % 2 != 0))
    throw std::invalid_argument("torontonian: matrix dimension must be even (2n x 2n)");
  const int kernel_dim = tor ? static_cast<int>(d / 2) : static_cast<int>(d);
  const auto* in = reinterpret_cast<const cuDoubleComplex*>(a.data());
  auto* out = new Cd[B == 0 ? 1 : B];
  auto* absn = new double[B == 0 ? 1 : B];
  int rc = fn(in, kernel_dim, static_cast<int>(B),
              reinterpret_cast<cuDoubleComplex*>(out), absn);
  if (rc != 0) {
    delete[] out; delete[] absn;
    throw std::runtime_error("gbskernels: CUDA error " + std::to_string(rc) + " in kappa kernel");
  }
  nb::capsule own_v(out, [](void* p) noexcept { delete[] static_cast<Cd*>(p); });
  nb::capsule own_a(absn, [](void* p) noexcept { delete[] static_cast<double*>(p); });
  return nb::make_tuple(Vec(out, {B}, own_v), DVec(absn, {B}, own_a));
}

// v3 fully on-device conditional sampler. Marshals the host-side reduced A-matrices
// (concatenated complex128 + int32 offsets) and 1/j! to gbs_sampler_sample, which keeps the
// whole chain (gather, hafnian, inverse-CDF + cuRAND, prefix state) on the device and returns
// the photon counts. Out: (num_draws, M) int32. The Python layer (sampling.sampler) computes
// {A_k} and enforces the in-cap precondition.
using AkVec = nb::ndarray<const Cd, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using IVec = nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using DVec1 = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using IMat = nb::ndarray<nb::numpy, int32_t, nb::ndim<2>>;

static IMat sample_resident(AkVec ak, IVec off, int M, int num_draws, int cutoff, int maxn,
                            DVec1 invfac, uint64_t seed) {
  const int ak_total = static_cast<int>(ak.shape(0));
  auto* out = new int32_t[(size_t)num_draws * (M == 0 ? 1 : M)];
  int rc = gbs::gbs_sampler_sample(reinterpret_cast<const cuDoubleComplex*>(ak.data()), ak_total,
                                   reinterpret_cast<const int*>(off.data()), M, num_draws, cutoff,
                                   maxn, invfac.data(), seed, reinterpret_cast<int*>(out));
  if (rc != 0) {
    delete[] out;
    throw std::runtime_error("gbskernels: CUDA error " + std::to_string(rc) +
                             " in the resident sampler");
  }
  nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<int32_t*>(p); });
  return IMat(out, {(size_t)num_draws, (size_t)M}, owner);
}

// Device-resident session: the residency half of the v1 contract
// (docs/device_resident_contract.md). Reuses device buffers across calls so a
// sampler loop allocates once. Methods mirror the free functions; the Python
// Workspace validates per-bucket preconditions before dispatching here. `func`:
// 0 perm, 1 haf, 2 lhaf, 3 tor. `prec`: 0 fp64, 1 dd.
struct Session {
  gbs::gbs_session* s = nullptr;
  Session() {
    if (gbs::gbs_session_open(&s) != 0 || !s)
      throw std::runtime_error("gbskernels: failed to open device session");
  }
  ~Session() { if (s) gbs::gbs_session_close(s); }
  void close() { if (s) { gbs::gbs_session_close(s); s = nullptr; } }
  std::size_t reallocs() const { return s ? gbs::gbs_session_reallocs(s) : 0; }

  Vec call(int func, int prec, const Stack& a, bool tor) {
    if (!s) throw std::runtime_error("gbskernels: session is closed");
    const size_t B = a.shape(0), d = a.shape(1);
    if (a.shape(2) != d)
      throw std::invalid_argument("gbskernels: each matrix must be square (B, d, d)");
    if (tor && (d % 2 != 0))
      throw std::invalid_argument("torontonian: matrix dimension must be even (2n x 2n)");
    auto* out = new Cd[B == 0 ? 1 : B];
    int rc = gbs::gbs_session_evaluate(
        s, func, prec, reinterpret_cast<const cuDoubleComplex*>(a.data()),
        static_cast<int>(d), static_cast<int>(B),
        reinterpret_cast<cuDoubleComplex*>(out));
    if (rc != 0) {
      delete[] out;
      throw std::runtime_error("gbskernels: CUDA error " + std::to_string(rc) +
                               " in session evaluate");
    }
    nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<Cd*>(p); });
    return Vec(out, {B}, owner);
  }

  // v2: evaluate keeping the result ON THE DEVICE -- returns a framework-agnostic
  // nb::ndarray (so it exposes __dlpack__ / __dlpack_device__) over the device buffer,
  // with a deleter that frees it. A downstream CuPy/PyTorch pipeline consumes it
  // zero-copy via DLPack; no D2H happens here.
  DevArr call_resident(int func, int prec, const Stack& a, bool tor) {
    if (!s) throw std::runtime_error("gbskernels: session is closed");
    const size_t B = a.shape(0), d = a.shape(1);
    if (a.shape(2) != d)
      throw std::invalid_argument("gbskernels: each matrix must be square (B, d, d)");
    if (tor && (d % 2 != 0))
      throw std::invalid_argument("torontonian: matrix dimension must be even (2n x 2n)");
    cuDoubleComplex* d_res = nullptr;
    int rc = gbs::gbs_session_evaluate_resident(
        s, func, prec, reinterpret_cast<const cuDoubleComplex*>(a.data()),
        static_cast<int>(d), static_cast<int>(B), &d_res);
    if (rc != 0)
      throw std::runtime_error("gbskernels: CUDA error " + std::to_string(rc) +
                               " in resident evaluate");
    nb::capsule owner(d_res, [](void* p) noexcept {
      gbs::gbs_dev_free(static_cast<cuDoubleComplex*>(p));
    });
    size_t shape[1] = {B};
    return DevArr(d_res, 1, shape, owner, nullptr, nb::dtype<Cd>(), GBS_DLPACK_DEVICE, 0);
  }
};

NB_MODULE(gbskernels_ext, m) {
  m.doc() = "GBSKernels GPU batched kernels (FP64 + double-double).";
  m.def("perm", [](Stack a) { return eval(gbs::gbs_perm_host, a, false); },
        "Batched permanent (FP64).");
  m.def("perm_dd", [](Stack a) { return eval(gbs::gbs_perm_dd_host, a, false); },
        "Batched permanent (double-double).");
  m.def("perm_kappa", [](Stack a) { return kappa_eval(gbs::gbs_perm_kappa_host, a, false); },
        "Batched permanent (FP64) + cancellation indicator -> (values, absnorms).");
  m.def("haf_kappa", [](Stack a) { return kappa_eval(gbs::gbs_haf_kappa_host, a, false); },
        "Batched hafnian (FP64) + cancellation indicator -> (values, absnorms).");
  m.def("lhaf_kappa", [](Stack a) { return kappa_eval(gbs::gbs_lhaf_kappa_host, a, false); },
        "Batched loop hafnian (FP64) + cancellation indicator -> (values, absnorms).");
  m.def("tor_kappa", [](Stack a) { return kappa_eval(gbs::gbs_tor_kappa_host, a, true); },
        "Batched torontonian (FP64) + cancellation indicator -> (values, absnorms).");
  m.def("perm_certified", [](Stack a) { return kappa_eval(gbs::gbs_perm_certified_host, a, false); },
        "Batched permanent: fp64 values + rigorous error bounds -> (values, bounds).");
  m.def("haf_certified", [](Stack a) { return kappa_eval(gbs::gbs_haf_certified_host, a, false); },
        "Batched hafnian: fp64 values + rigorous error bounds -> (values, bounds).");
  m.def("lhaf_certified", [](Stack a) { return kappa_eval(gbs::gbs_lhaf_certified_host, a, false); },
        "Batched loop hafnian: fp64 values + rigorous error bounds -> (values, bounds).");
  m.def("tor_certified", [](Stack a) { return kappa_eval(gbs::gbs_tor_certified_host, a, true); },
        "Batched torontonian: certified-LU values + rigorous error bounds -> (values, bounds).");
  m.def("perm_dd_certified", [](Stack a) { return kappa_eval(gbs::gbs_perm_dd_certified_host, a, false); },
        "Batched permanent (double-double) + rigorous error bounds -> (values, bounds).");
  m.def("haf_dd_certified", [](Stack a) { return kappa_eval(gbs::gbs_haf_dd_certified_host, a, false); },
        "Batched hafnian (double-double) + rigorous error bounds -> (values, bounds).");
  m.def("tor_single",
        [](nb::ndarray<nb::numpy, const double, nb::ndim<2>> O, int g) {
          const size_t dim = O.shape(0);
          if (O.shape(1) != dim || dim % 2 != 0 || dim > 64)
            throw std::invalid_argument("tor_single: O must be (2n, 2n) real, 2n <= 64");
          double out = 0.0;
          int rc = gbs::gbs_tor_recursive_single_host(O.data(), (int)(dim / 2), g, &out);
          if (rc != 0)
            throw std::runtime_error("gbskernels: CUDA error " + std::to_string(rc) + " in tor_single");
          return out;
        },
        "SINGLE-LARGE recursive torontonian (real O, dim <= 64), one evaluation\n"
        "split across 2^g subtrees; NaN = off the physical (SPD) domain.");
  m.def("tor_single_certified",
        [](nb::ndarray<nb::numpy, const double, nb::ndim<2>> O, int g) {
          const size_t dim = O.shape(0);
          if (O.shape(1) != dim || dim % 2 != 0 || dim > 64)
            throw std::invalid_argument("tor_single_certified: O must be (2n, 2n) real, 2n <= 64");
          double out = 0.0, bound = 0.0;
          int rc = gbs::gbs_tor_single_certified_host(O.data(), (int)(dim / 2), g, &out, &bound);
          if (rc != 0)
            throw std::runtime_error("gbskernels: CUDA error " + std::to_string(rc) + " in tor_single_certified");
          return nb::make_tuple(out, bound);
        },
        "CERTIFIED single-large torontonian -> (value, rigorous |value-exact| bound);\n"
        "NaN/inf = off-domain or uncertifiable, never a finite overclaim.");
  m.def("lhaf_repeated",
        [](nb::ndarray<nb::numpy, const Cd, nb::ndim<2>> A,
           nb::ndarray<nb::numpy, const Cd, nb::ndim<1>> gamma,
           nb::ndarray<nb::numpy, const int32_t, nb::ndim<2>> reps) {
          const size_t M = A.shape(0);
          if (A.shape(1) != M) throw std::invalid_argument("lhaf_repeated: A must be (M, M)");
          if (gamma.shape(0) != M) throw std::invalid_argument("lhaf_repeated: gamma must be (M,)");
          if (reps.shape(1) != M) throw std::invalid_argument("lhaf_repeated: reps must be (B, M)");
          const size_t B = reps.shape(0);
          auto* out = new Cd[B == 0 ? 1 : B];
          int rc = gbs::gbs_lhaf_repeated_host(
              reinterpret_cast<const cuDoubleComplex*>(A.data()),
              reinterpret_cast<const cuDoubleComplex*>(gamma.data()), static_cast<int>(M),
              reinterpret_cast<const int*>(reps.data()), static_cast<int>(B),
              reinterpret_cast<cuDoubleComplex*>(out));
          if (rc != 0) {
            delete[] out;
            throw std::runtime_error("gbskernels: CUDA error " + std::to_string(rc) + " in lhaf_repeated");
          }
          nb::capsule own(out, [](void* p) noexcept { delete[] static_cast<Cd*>(p); });
          return Vec(out, {B}, own);
        },
        "Repeated-row loop hafnians: one (M,M) base matrix + (M,) loop vector, a (B,M)\n"
        "int32 repetition table -> (B,) values via the finite-difference sieve (R4).");
  m.def("lhaf_repeated_certified",
        [](nb::ndarray<nb::numpy, const Cd, nb::ndim<2>> A,
           nb::ndarray<nb::numpy, const Cd, nb::ndim<1>> gamma,
           nb::ndarray<nb::numpy, const int32_t, nb::ndim<2>> reps) {
          const size_t M = A.shape(0);
          if (A.shape(1) != M) throw std::invalid_argument("lhaf_repeated_certified: A must be (M, M)");
          if (gamma.shape(0) != M) throw std::invalid_argument("lhaf_repeated_certified: gamma must be (M,)");
          if (reps.shape(1) != M) throw std::invalid_argument("lhaf_repeated_certified: reps must be (B, M)");
          const size_t B = reps.shape(0);
          auto* out = new Cd[B == 0 ? 1 : B];
          auto* bnd = new double[B == 0 ? 1 : B];
          int rc = gbs::gbs_lhaf_repeated_cert_host(
              reinterpret_cast<const cuDoubleComplex*>(A.data()),
              reinterpret_cast<const cuDoubleComplex*>(gamma.data()), static_cast<int>(M),
              reinterpret_cast<const int*>(reps.data()), static_cast<int>(B),
              reinterpret_cast<cuDoubleComplex*>(out), bnd);
          if (rc != 0) {
            delete[] out;
            delete[] bnd;
            throw std::runtime_error("gbskernels: CUDA error " + std::to_string(rc) + " in lhaf_repeated_certified");
          }
          nb::capsule own(out, [](void* p) noexcept { delete[] static_cast<Cd*>(p); });
          nb::capsule ownb(bnd, [](void* p) noexcept { delete[] static_cast<double*>(p); });
          return nb::make_tuple(Vec(out, {B}, own),
                                nb::ndarray<nb::numpy, double>(bnd, {B}, ownb));
        },
        "CERTIFIED sieve: (values, rigorous |value-exact| bounds); values are\n"
        "bit-identical to lhaf_repeated; over-cap rows emit NaN + inf bound.");
  m.def("haf", [](Stack a) { return eval(gbs::gbs_haf_host, a, false); },
        "Batched hafnian (FP64).");
  m.def("haf_dd", [](Stack a) { return eval(gbs::gbs_haf_dd_host, a, false); },
        "Batched hafnian (double-double).");
  m.def("lhaf", [](Stack a) { return eval(gbs::gbs_lhaf_host, a, false); },
        "Batched loop hafnian (FP64).");
  m.def("lhaf_dd", [](Stack a) { return eval(gbs::gbs_lhaf_dd_host, a, false); },
        "Batched loop hafnian (double-double).");
  m.def("tor", [](Stack a) { return eval(gbs::gbs_tor_host, a, true); },
        "Batched torontonian (FP64); matrices are 2n x 2n.");
  m.def("tor_dd", [](Stack a) { return eval(gbs::gbs_tor_dd_host, a, true); },
        "Batched torontonian (double-double); real-O domain, 2n x 2n.");
  // v3 fully on-device conditional sampler (sampling.sampler.sample backend="gpu", resident=True).
  m.def("sample_resident", &sample_resident,
        "Fully on-device conditional GBS sampler: concatenated {A_k} + offsets + 1/j! -> "
        "(num_draws, M) int32 photon counts; the whole chain stays on the device.");
  // Provenance: True iff the kernels were compiled as the CPU host-shim (no
  // CUDA). Callers stamp results "host-shim" vs "gpu" from this, so an emulated
  // run is never mislabelled a real-device run (docs/DESIGN.md §8 honest benchmarking).
#ifdef GBS_HOST_SHIM
  m.attr("__host_shim__") = true;
#else
  m.attr("__host_shim__") = false;
#endif

  // Device-resident session (gbskernels.Workspace routes ragged buckets here).
  nb::class_<Session>(m, "Session")
      .def(nb::init<>())
      .def("perm",    [](Session& z, Stack a) { return z.call(0, 0, a, false); })
      .def("perm_dd", [](Session& z, Stack a) { return z.call(0, 1, a, false); })
      .def("haf",     [](Session& z, Stack a) { return z.call(1, 0, a, false); })
      .def("haf_dd",  [](Session& z, Stack a) { return z.call(1, 1, a, false); })
      .def("lhaf",    [](Session& z, Stack a) { return z.call(2, 0, a, false); })
      .def("lhaf_dd", [](Session& z, Stack a) { return z.call(2, 1, a, false); })
      .def("tor",     [](Session& z, Stack a) { return z.call(3, 0, a, true); })
      .def("tor_dd",  [](Session& z, Stack a) { return z.call(3, 1, a, true); })
      // v2 device-resident output -> DLPack handle (result stays on the device).
      .def("perm_resident", [](Session& z, Stack a) { return z.call_resident(0, 0, a, false); })
      .def("haf_resident",  [](Session& z, Stack a) { return z.call_resident(1, 0, a, false); })
      .def("lhaf_resident", [](Session& z, Stack a) { return z.call_resident(2, 0, a, false); })
      .def("tor_resident",  [](Session& z, Stack a) { return z.call_resident(3, 0, a, true); })
      .def("close",    &Session::close)
      .def("reallocs", &Session::reallocs);
}
