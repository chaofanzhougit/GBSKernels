# bindings/ — nanobind Python extension over the CUDA kernels

`gbskernels_ext.cpp` exposes the batched GPU kernels to Python (`perm`, `perm_dd`,
`haf`, `lhaf`, `tor`), marshalling numpy `complex128` stacks to the host-facing
wrappers in `core/host_api.cu` (which do the H2D → launch → D2H). The Python
package (`gbskernels`) imports it lazily and routes `backend="gpu"` through it;
`backend="cpu"` (default) uses the reference implementation.

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

Both modes are validated: the host-shim build's outputs match the CPU reference
(perm/haf/lhaf to ~1e-13, the double-double permanent **exactly**). Needs
`nanobind` + `cmake` (dev deps).
