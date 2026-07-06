# Device-resident + bucketing — the v1 contract

Status: **design + first increment** (this document is the contract; the Python
bucketing layer and the C++ session land against it). Scope is deliberately v1:
enough to make an *iterative sampler* drive the GPU efficiently, no more.

## 1. The problem (why this is architectural, not a benchmark)

The current GPU binding is a **one-shot** path. Every
`gbskernels.X_batched(stack, backend="gpu")` call does, in `core/host_api.cu`:

```
cudaMalloc(d_in) ; cudaMalloc(d_out) ; H2D ; launch ; sync ; D2H ; cudaFree ; cudaFree
```

with **no** state kept between calls and a **uniform** `(B, d, d)` precondition
(`gbskernels._gpu_batched` raises on a ragged batch). That is fine for a single
batched evaluation, and wrong for the workload the project actually targets.

The real workload (docs/DESIGN.md §2.3, §5) is a **sampler loop**: drawing GBS samples is
a chain of conditional probabilities, each proportional to a hafnian/permanent of
a **growing submatrix**. The chain produces *many* evaluations of *non-uniform*
size, repeatedly. It is already in the tree — `sampling/gbs.py`:

```python
hafs = gbskernels.haf_batched([_submatrix(B, p) for p in even], precision=precision)
```

`_submatrix(B, p)` returns a different size per Fock pattern `p`, so the list is
**ragged**. Today this only runs because the *CPU* backend loops over it; on the
GPU backend it raises. To put this workload on the GPU we need two things the
binding lacks (README "Known limitations"):

1. **Bucketing** — accept a ragged set of evaluations and group it into uniform
   per-size launches, then scatter results back to the caller's order.
2. **Device residency** — reuse device buffers (and a stream) across the many
   launches of a sampler loop instead of `cudaMalloc`/`cudaFree` per call.

## 2. v1 surface

A single new object, `gbskernels.Workspace`, a context manager that owns the
device-resident state and accepts ragged work:

```python
with gbskernels.Workspace(backend="gpu") as ws:
    # ragged list of differently-sized matrices -> one result vector, input order
    hafs = ws.haf_batched([_submatrix(B, p) for p in even], precision="fp64")
    perms = ws.perm_batched(list_of_varied_size_matrices)
    # ... many calls in a sampler loop reuse the same device buffers ...
```

* `Workspace.{perm,haf,lhaf,tor}_batched(matrices, precision=...)` mirror the
  free functions but **accept ragged input** and return a `complex128` vector
  aligned to the input order.
* `backend="cpu"` is supported too (buckets degenerate to the existing CPU loop),
  so the *same* sampler code is differential-tested CPU-vs-GPU.
* The free functions (`gbskernels.haf_batched(...)`) keep their current contract
  unchanged — uniform-only on GPU. The `Workspace` is the opt-in residency+ragged
  handle; nothing existing changes behaviour.

### Bucketing semantics

* **Group key** = `(func, precision, matrix_dim)`. Within a single `*_batched`
  call `func`/`precision` are fixed, so the key reduces to `matrix_dim`.
* Each bucket is one contiguous uniform `(Bᵢ, dᵢ, dᵢ)` launch through the resident
  session. Per-kernel size caps (`_GPU_MAX_DIM`) and preconditions (even-N loop
  hafnian, real-O DD torontonian) are enforced **per bucket**, with the original
  input index named in any error.
* **Order-preserving scatter**: results are written back to a single output vector
  at each evaluation's original position. The result for input `i` is identical
  (bit-for-bit, same kernel) to evaluating its matrix alone — bucketing only
  changes *grouping*, never *values*. This is the central correctness invariant.
* Empty input → empty vector. A single-size input → a single bucket (== today).

### Residency semantics (v1)

* The `Workspace` holds **one C++ session** (`core/host_api.cu`:
  `gbs_session_open/evaluate/close`) owning reusable `d_in`/`d_out` device buffers
  and a stream. Buffers **grow monotonically** to the largest bucket seen and are
  reused for every smaller/equal one — so a sampler loop pays allocation once, not
  per call.
* Inputs are still copied **H2D per evaluate** and results **D2H per evaluate**:
  v1 amortizes *allocation*, not *transfer*. The host (Python/numpy) still
  assembles each step's submatrices, so results must come back to the host. This
  is the honest v1 boundary; see §4.
* Close is explicit (context-manager exit) and idempotent; double-close is a
  no-op. A failed evaluate leaves the session usable (buffers intact, error
  surfaced as a Python exception) — the buffer is never leaked or left partial.

## 3. Verification (CPU-first, as always)

Everything here is validated on CPU **before** any GPU session, via the host shim
(`-DGBS_HOST_SHIM=ON`, where `cudaMalloc→malloc` and launches are grid-emulated):

* **L1/L3 — bucketing correctness** (`tests/test_workspace_bucketing.py`): for a
  ragged input, `ws.X_batched(ragged)` equals, element-by-element, the result of
  evaluating each matrix through the existing uniform path — across all four
  functions, both precisions, and including the degenerate single-size and empty
  cases. Per-bucket precondition errors name the offending input index.
* **L3 — residency reuse** (`core/check_session.cu`, a new host-shim gate in
  `run_preflight.sh`): open one session, evaluate a sequence of differently-sized
  buckets and several functions through it, and assert (a) every result matches
  the per-call `gbs_X_host` reference and (b) the buffers were *reused* (capacity
  grows monotonically; no re-alloc for a smaller bucket).
* **L5 — sampler on GPU path**: `sampling/` driven through a `Workspace(backend=
  "gpu")` (host-shim) reproduces the CPU sampler's probabilities (sum-to-one,
  per-pattern agreement), proving the ragged sampler workload runs end-to-end on
  the binding path.

## 4. v2 — genuinely device-resident (shipped)

v2 makes the Session genuinely device-resident, on top of v1's bucketing + buffer
reuse. All of it is validated on the CPU host shim (where the "device" is host
memory, so the semantics and the DLPack zero-copy are real); the device-only
*performance* (overlap, pinning) is measured in a GPU session.

* **CUDA streams + async copies.** The Session owns a persistent stream; H2D, the
  kernel launch, and D2H all run on it via `cudaMemcpyAsync` + `cudaStreamSynchronize`.
  (On the shim these degenerate to synchronous memcpy; the code path and results are
  validated. Overlap is a device property measured on a GPU.)
* **Pinned host staging.** Page-locked input/output staging buffers
  (`cudaMallocHost`), reused/grown like the device buffers, so the H2D/D2H DMA is
  asynchronous and overlappable.
* **Device-resident output handles.** `Workspace.{perm,haf,lhaf,tor}_resident(stack)`
  evaluates a uniform batch keeping the result **on the device** (a fresh buffer
  owned by the returned handle — *not* the reused `d_out`), with **no D2H**. The
  handle is a zero-copy, DLPack-exportable array so a downstream consumer reads it
  without a round trip. Gated on-device by `core/check_session.cu` (resident result
  == one-shot reference).
* **DLPack / CuPy / PyTorch interop.** The resident handle is returned as a framework
  array matching the build: a **CuPy** array on the kDLCUDA device for a real GPU
  build (PyTorch / JAX consume it zero-copy via `torch.from_dlpack` / `jax.dlpack`),
  and a **numpy** array on the kDLCPU device for the host-shim — where
  `numpy.from_dlpack` validates the zero-copy round trip on CPU.

Still future (v3): keeping outputs on-device to feed the next conditional step
**without leaving CUDA at all** (the chain-rule sampling + submatrix gather would move
into CUDA); compute/copy *overlap* across buckets; cross-function batching in one
launch; multi-GPU. The v2 resident path is uniform-batch only — a ragged resident
output would need an on-device scatter (use `*_batched` for ragged, host results).
