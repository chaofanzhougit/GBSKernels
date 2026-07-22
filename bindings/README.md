# bindings/ — nanobind Python extension over the CUDA kernels

`gbskernels_ext.cpp` exposes the CUDA implementation to Python:

- FP64 and double-double batched permanent, hafnian, loop hafnian, and
  torontonian kernels;
- cancellation-indicator and certified-bound entry points, including
  certified double-double permanent/hafnian paths;
- FP64 and certified single-large recursive torontonians, plus the certified
  double-double variant;
- plain/certified repeated-row loop hafnians; and
- the resident sampler and reusable `Session`/DLPack workspace APIs.

The binding marshals NumPy stacks to the host-facing wrappers in
`core/host_api.cu` (H2D → launch → D2H). The Python package imports it lazily
and routes `backend="gpu"` through it; `backend="cpu"` remains the default.

```python
import numpy as np, gbskernels
A = np.ascontiguousarray(stack_of_matrices)         # (B, n, n) complex128
gbskernels.perm_batched(A, backend="gpu")           # FP64 on the GPU
gbskernels.perm_batched(A, precision="dd", backend="gpu")  # double-double on the GPU
gbskernels.gpu_available()                          # True iff the extension is built
```

## Build

**Real GPU build** (in a rented-GPU session, nvcc):
```bash
cmake -S bindings -B bindings/build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build bindings/build -j      # -> gbskernels_ext*.so
```

**Host pre-flight** (`-DGBS_HOST_SHIM=ON`): compiles the kernels as plain C++
against `core/preflight/cuda` (grid-emulated on CPU), so the *entire Python →
kernel* path is importable and validated **without a GPU**. This is what
`tests/test_gpu_bindings.py` exercises (the GPU backend's results are checked
against the CPU reference); the real GPU build of the same sources runs on-device.

Both modes are gated. The host-shim extension tests compare the public binding
surface with the independent CPU/reference implementations, including
precision, certified, recursive, repeated-row, resident, and workspace paths.
The real CUDA session hard-gates the underlying C++ kernels on-device and smoke-
tests the extension; it does not duplicate every host-shim API test on-device.
Both builds need `nanobind` + `cmake` (development dependencies).
