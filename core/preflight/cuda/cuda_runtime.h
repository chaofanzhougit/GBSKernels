// core/preflight/cuda/cuda_runtime.h  --  CPU shim of CUDA's <cuda_runtime.h>.
//
// PRE-FLIGHT ONLY (see cuComplex.h header). Provides just enough of the CUDA
// runtime/execution model to compile and *emulate* the kernels on a host:
// qualifier macros, a settable single-thread index, the warp intrinsics the
// kernels use, and no-op memory calls so the check_*.cu host drivers also build.
// __CUDACC__ is NOT defined here, so kernels take their host code paths.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

// --- qualifier macros (no-ops on host) -------------------------------------
#define __global__
#define __device__
#define __host__
#define __restrict__

// --- execution-configuration indices (settable for grid emulation) ---------
namespace gbs {
namespace shim {
struct Dim3 {
  unsigned x = 0, y = 1, z = 1;
};
inline Dim3 g_blockIdx;
inline Dim3 g_blockDim;
inline Dim3 g_threadIdx;
// Set the "current thread" before each emulated kernel invocation.
inline void set_thread(int blockIdx_x, int threadIdx_x, int blockDim_x) {
  g_blockIdx.x = (unsigned)blockIdx_x;
  g_threadIdx.x = (unsigned)threadIdx_x;
  g_blockDim.x = (unsigned)blockDim_x;
}
}  // namespace shim
}  // namespace gbs
#define blockIdx ::gbs::shim::g_blockIdx
#define blockDim ::gbs::shim::g_blockDim
#define threadIdx ::gbs::shim::g_threadIdx

// --- warp intrinsics used by the kernels (host equivalents) ----------------
inline int __popcll(long long x) { return __builtin_popcountll((unsigned long long)x); }
inline int __ffsll(long long x) { return __builtin_ffsll(x); }

// --- runtime API stubs (so check_*.cu host drivers compile on host) --------
using cudaStream_t = void*;
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorInvalidValue = 1;  // matches the real CUDA enum value
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
template <class T>  // CUDA's cudaMalloc is templated on T** (no cast at call sites)
inline cudaError_t cudaMalloc(T** p, size_t n) { *p = static_cast<T*>(std::malloc(n)); return cudaSuccess; }
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
  std::memcpy(dst, src, n); return cudaSuccess;
}
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "shim: no error"; }

// --- streams / async copies / pinned memory (Workspace v2) -----------------
// Degenerate on the host: a stream is a dummy token, async copies are synchronous
// memcpy, pinned alloc is plain malloc. The v2 Session code path (create stream ->
// pinned staging -> async H2D/D2H on the stream -> stream sync) is thus exercised
// and its RESULTS validated on CPU; the actual overlap/pinning speedups are
// device-only properties measured in a GPU session.
inline cudaError_t cudaStreamCreate(cudaStream_t* s) { *s = (cudaStream_t)1; return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind, cudaStream_t) {
  std::memcpy(dst, src, n); return cudaSuccess;
}
inline cudaError_t cudaMallocHost(void** p, size_t n) { *p = std::malloc(n); return cudaSuccess; }
inline cudaError_t cudaFreeHost(void* p) { std::free(p); return cudaSuccess; }
