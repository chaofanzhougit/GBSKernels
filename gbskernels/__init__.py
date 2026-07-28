"""GBSKernels — GPU-native, batched photonic sampling matrix functions.

Public surface (docs/DESIGN.md §7): batched-first. The implementations in the
top-level ``cpu_ref`` (FP64) and ``highprec_ref`` (mpmath) packages provide the
CPU and reference backends; the optional CUDA extension is loaded behind the
same API. Differential and "batched-equals-looped" tests (docs/DESIGN.md §8,
Layer 5) exercise those paths through the public surface.

Precision is always an explicit tier:

* ``"fp64"`` (default) -- native double precision; the throughput workhorse.
  Correct for small-to-moderate ``n`` and well-conditioned matrices; carries a
  documented accuracy caveat for large / ill-conditioned / cancellation-heavy
  inputs (docs/DESIGN.md §6).
* ``"ref"`` -- arbitrary-precision mpmath ground truth. Slow, tiny
  ``n``; the source of truth for accuracy characterization, not for throughput.
* ``"dd"`` -- double-double; a GPU tier (error-free transforms with
  no native CPU/GPU quad). Not available on the CPU backend yet -> raises.
* ``"auto"`` -- adaptive: FP64 plus an a-posteriori cancellation
  indicator (kappa = sum|terms|/|result|); return the fast FP64 value when
  well-conditioned, else rerun in the high-precision tier. CPU backend reruns in
  mpmath; GPU backend reruns in DD -- for all four functions (each ``*_kappa``
  kernel emits the indicator in the FP64 pass; the host reruns only the risky
  elements in DD). Pass ``return_diagnostics=True`` to also get
  ``{"tier", "cancellation"}``. Turns the measured boundary (sec.6) into a feature.
  ``kappa`` is a **calibrated heuristic**, not an error certificate: ``bench.calibrate_auto``
  measures it vs the true error on physical/loss/adversarial ensembles, on the CPU and the
  GPU-kappa-kernel paths (zero false-trusts there; worst trusted error 1.15e-8 measured
  on-device, <= kappa_max·eps ~ 2e-8), but it is not a guarantee.
* ``"certified"`` -- the FP64 value (bit-for-bit the ``"fp64"`` reference
  result) together with a **rigorous** a-posteriori error bound: a running
  error bound under the standard floating-point model, the proven counterpart
  of ``kappa`` (``cpu_ref.certified``). Requires
  ``return_diagnostics=True`` (a certificate you cannot see is not a
  certificate): diagnostics are ``{"tier": "certified-fp64",
  "abs_error_bound", "rel_error_bound"}`` with ``|value - exact| <=
  abs_error_bound``. **All four functions, CPU and GPU** (the GPU bound
  accumulators run in per-instruction directed rounding); with ``rtol=`` the
  PROVEN ladder certified-fp64 -> certified-dd (GPU) -> mpmath escalates on
  proof, never heuristic. Single large torontonians:
  ``tor_single(..., certified=True)`` (real domain, dim <= 64).
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

import cpu_ref
import highprec_ref

__version__ = "0.2.2"
__all__ = [
    "perm", "perm_batched",
    "haf", "haf_batched",
    "lhaf", "lhaf_batched",
    "tor", "tor_batched",
    "lhaf_repeated", "tor_single",
    "gpu_available", "gpu_backend_kind", "Workspace", "PRECISIONS",
]

PRECISIONS = ("fp64", "ref", "dd", "auto", "certified")

# Each function maps a precision tier to its backend implementation. The
# single/batched wrappers below are shared across all four. The 'dd' tier is
# GPU-only (no CPU backend yet).
_BACKENDS = {
    "perm": {"fp64": cpu_ref.perm, "ref": highprec_ref.permanent_mp},
    "haf": {"fp64": cpu_ref.haf, "ref": highprec_ref.hafnian_mp},
    "lhaf": {"fp64": cpu_ref.lhaf, "ref": highprec_ref.loop_hafnian_mp},
    "tor": {"fp64": cpu_ref.tor, "ref": highprec_ref.torontonian_mp},
}


def _evaluate(func: str, A: Any, precision: str) -> complex:
    impls = _BACKENDS[func]
    if precision in impls:
        return complex(impls[precision](A))
    if precision == "dd":
        raise NotImplementedError(
            "double-double ('dd') is a GPU precision tier (error-free transforms); "
            "it is not available on the CPU backend. Use 'fp64' or 'ref'."
        )
    raise ValueError(f"unknown precision {precision!r}; expected one of {PRECISIONS}")


def _batched(func: str, matrices: Iterable[Any], precision: str) -> np.ndarray:
    """One evaluation per element on the CPU reference (loops, so the Layer-5
    "batched-equals-looped" invariant holds by construction). Accepts a stacked
    ``(B, n, n)`` array or any iterable of square matrices of *possibly different
    sizes* -- the GBS workload batches submatrices of many shapes (anchor
    sec.2.3)."""
    if isinstance(matrices, np.ndarray) and matrices.ndim == 3:
        batch: Any = matrices
    else:
        batch = list(matrices)
    out = np.empty(len(batch), dtype=np.complex128)
    for i, A in enumerate(batch):
        out[i] = _evaluate(func, A, precision)
    return out


# --- GPU backend (the compiled CUDA extension) ------------------------------

_GPU_EXT: Any = None # cached module or False once we know it is unavailable


def _load_gpu_ext() -> Any:
    """Lazily import the compiled CUDA extension (``gbskernels_ext``).

    Looks for an installed module first, then ``$GBSKERNELS_EXT_DIR`` and the
    in-repo build dirs (so the host-shim pre-flight build is usable in tests).
    Returns the module, or ``None`` if it was never built (CPU-only install)."""
    global _GPU_EXT
    if _GPU_EXT is not None:
        return _GPU_EXT or None
    import importlib
    import os
    import sys
    from pathlib import Path

    candidates = []
    env = os.environ.get("GBSKERNELS_EXT_DIR")
    if env:
        candidates.append(Path(env))
    repo = Path(__file__).resolve().parent.parent
    candidates += [repo / "bindings" / "build", repo / "bindings" / "build_host"]
    for d in candidates:
        if d.is_dir() and str(d) not in sys.path:
            sys.path.insert(0, str(d))
    try:
        _GPU_EXT = importlib.import_module("gbskernels_ext")
    except Exception:
        _GPU_EXT = False
    return _GPU_EXT or None


# (func, precision) -> name of the extension entry point.
_GPU_FNS = {
    ("perm", "fp64"): "perm", ("perm", "dd"): "perm_dd",
    ("haf", "fp64"): "haf", ("haf", "dd"): "haf_dd",
    ("lhaf", "fp64"): "lhaf", ("lhaf", "dd"): "lhaf_dd",
    ("tor", "fp64"): "tor", ("tor", "dd"): "tor_dd",
}

# Largest input matrix dimension d each GPU kernel supports, set by the fixed
# per-thread local-memory buffers in core/*.cu (PERM_MAX_N, HAF_MAX_N, ...).
# Requests beyond these are rejected here rather than overflowing device storage.
_GPU_MAX_DIM = {
    "perm": 28, "perm_dd": 28,
    "haf": 20, "haf_dd": 16,
    "lhaf": 20, "lhaf_dd": 14,
    "tor": 24, "tor_dd": 24, # d = 2n
}


def _gpu_prepare(func: str, matrices: Iterable[Any], precision: str):
    """Validate a *uniform* (B, d, d) bucket for the GPU backend.

    Returns ``(ext, name, stack)``: the loaded extension, its entry-point name for
    ``(func, precision)``, and the contiguous complex128 stack. Enforces every GPU
    precondition (size caps, even-N loop hafnian, real-O DD torontonian) so both
    the per-call path (:func:`_gpu_batched`) and the device-resident
    :class:`Workspace` session validate identically. Raises ValueError on a
    rejected input, NotImplementedError on an unimplemented (func, precision)."""
    ext = _load_gpu_ext()
    if ext is None:
        raise RuntimeError(
            "GPU backend requested but the compiled extension 'gbskernels_ext' is "
            "not available. Build it (bindings/CMakeLists.txt) in a CUDA environment, "
            "or set GBSKERNELS_EXT_DIR to its build directory."
        )
    key = (func, precision)
    if key not in _GPU_FNS:
        raise NotImplementedError(
            f"GPU backend has no {precision!r} kernel for {func!r}; the GPU precision "
            "tiers are 'fp64' and 'dd' (all four functions). 'ref' is the mpmath "
            "ground truth -- CPU only (backend='cpu')."
        )
    stack = np.ascontiguousarray(np.asarray(matrices, dtype=np.complex128))
    if stack.ndim != 3 or stack.shape[1] != stack.shape[2]:
        raise ValueError(
            "GPU backend needs a uniform (B, d, d) stack of square matrices; "
            "ragged batches are a CPU-backend feature (or use gbskernels.Workspace)."
        )
    name = _GPU_FNS[key]
    d = int(stack.shape[1])

    # Odd-N loop hafnian on the GPU: the power-trace kernel is even-N, but the loop
    # hafnian is invariant under adding a vertex with self-loop weight 1 and zero
    # off-diagonals (it can only loop, contributing a factor of 1) -- so augment each
    # odd matrix to (N+1) x (N+1) and run the validated even-N kernel (fp64 or DD).
    # lhaf(A_odd) == lhaf(A_odd (+) [1]); verified vs the any-N CPU reference.
    if func == "lhaf" and d % 2 != 0:
        aug = np.zeros((stack.shape[0], d + 1, d + 1), dtype=np.complex128)
        aug[:, :d, :d] = stack
        aug[:, d, d] = 1.0
        stack = np.ascontiguousarray(aug)
        d += 1

    # Size limit: the kernels use fixed per-thread buffers (see _GPU_MAX_DIM). For an
    # augmented odd loop hafnian this is the augmented dimension (so odd N <= cap-1).
    if d > _GPU_MAX_DIM[name]:
        raise ValueError(
            f"GPU {name}: matrix dimension {d} exceeds the kernel's limit "
            f"{_GPU_MAX_DIM[name]} (set by fixed per-thread local buffers; the "
            "cooperative kernels parallelize the subset sum but share that cap; an "
            "odd loop hafnian is augmented to N+1). Use the CPU backend for larger n."
        )
    # The DD torontonian works in real double-double (its physical domain is real
    # O); refuse complex input rather than silently dropping the imaginary part.
    if name == "tor_dd" and stack.size and float(np.max(np.abs(stack.imag))) > 1e-12:
        raise ValueError(
            "GPU torontonian (precision='dd') is real-domain only; the input has "
            "non-negligible imaginary parts. Use precision='fp64' for complex O "
            "(noting the sqrt-branch caveat) or pass a real matrix."
        )
    return ext, name, stack


def _gpu_batched(func: str, matrices: Iterable[Any], precision: str) -> np.ndarray:
    ext, name, stack = _gpu_prepare(func, matrices, precision)
    return np.asarray(getattr(ext, name)(stack))


# --- precision="auto": FP64 + cancellation indicator -> high-precision if risky ---
# The fast algorithms are alternating signed sums; FP64 loses ~log10(kappa) digits,
# where kappa = sum|terms|/|result| (cpu_ref.summation_condition_number), so
# rel_err_fp64 ~ kappa*eps. Below the threshold FP64 is trusted; above it the evaluation
# reruns in the high-precision tier (DD on GPU, mpmath on CPU).
#
# kappa is a HEURISTIC indicator, NOT a rigorous error certificate: it is a-posteriori
# (formed from the FP64-inexact |result|) and can mis-estimate under extreme cancellation.
# The 1e8 threshold is CALIBRATED, not assumed -- bench.calibrate_auto measures kappa vs the
# actual error vs mpmath on physical/loss/adversarial ensembles, on the CPU AND the GPU-kappa-
# kernel paths (log-log corr ~0.99; ZERO false-trusts; worst FP64 error among trusted cases
# 1.15e-8 measured on-device, <= kappa_max*eps ~ 2e-8). Treat it as a calibrated risk heuristic
# with a measured failure rate, not a guarantee.
_AUTO_KAPPA_MAX = 1e8


# GPU auto: which functions have an on-device cancellation kernel (FP64 + sum|term|
# in one pass) -> per-element FP64/DD rerun. All four ship.
_GPU_AUTO_FUNCS = ("perm", "haf", "lhaf", "tor")


def _gpu_auto_rerun_risky(ext, func: str, stack: np.ndarray, values: np.ndarray,
                          risky: np.ndarray) -> str:
    """Recompute the risky elements at a tier >= DD; return the tier actually used.

    The GPU 'auto' rerun tier is double-double, whose size cap is <= the FP64 cap (and,
    for the torontonian, is real-domain only). When the (uniform-size) batch is inside
    the DD domain, rerun the risky elements on the GPU in DD; when it is OUTSIDE -- a size
    above the DD cap, or a complex torontonian -- the GPU has no DD kernel for them, so
    rerun on the CPU in mpmath ('ref'), which is strictly more accurate than DD. Either
    way an element flagged risky is never returned as its unsafe FP64 value: the behavior
    is DEFINED for every size, not only within the DD cap (the old code raised, but only
    when an over-cap element *happened* to be risky -- data-dependent; docs/DESIGN.md §6)."""
    sub = np.ascontiguousarray(stack[risky])
    try:
        _gpu_prepare(func, sub, "dd") # DD cap + (tor) real-domain gate
    except (ValueError, NotImplementedError):
        import cpu_ref # noqa: F401 (mpmath rerun for the out-of-DD-domain elements)
        for i in np.nonzero(risky)[0]:
            values[i] = complex(_evaluate(func, stack[i], "ref"))
        return "ref"
    values[risky] = np.asarray(getattr(ext, f"{func}_dd")(sub))
    return "dd"


def _gpu_auto_batched(func: str, matrices: Iterable[Any], return_diagnostics: bool):
    """GPU auto: one FP64+indicator pass on the device, then rerun only the risky
    elements at a high-precision tier (DD on device when in the DD domain, else mpmath
    'ref' on the CPU -- see :func:`_gpu_auto_rerun_risky`). Per-element tier selection."""
    if func not in _GPU_AUTO_FUNCS:
        raise NotImplementedError(
            f"precision='auto' on backend='gpu' is implemented for {_GPU_AUTO_FUNCS} so far; "
            f"{func!r} has no on-device cancellation kernel yet (use backend='cpu', or "
            "precision='dd' explicitly on GPU)."
        )
    ext, _name, stack = _gpu_prepare(func, matrices, "fp64")
    if not hasattr(ext, f"{func}_kappa"):
        raise RuntimeError(
            f"the loaded extension has no {func}_kappa kernel; rebuild bindings/ (the "
            "GPU auto cancellation kernel postdates older builds)."
        )
    values, absnorm = (np.asarray(x) for x in getattr(ext, f"{func}_kappa")(stack))
    values = values.astype(np.complex128, copy=True)
    kappa = absnorm / np.maximum(np.abs(values), 1e-300)
    risky = kappa >= _AUTO_KAPPA_MAX
    rerun_tier = _gpu_auto_rerun_risky(ext, func, stack, values, risky) if risky.any() else None
    if not return_diagnostics:
        return values
    diags = [{"tier": rerun_tier if r else "fp64", "cancellation": float(k)}
             for r, k in zip(risky, kappa)]
    return values, diags


def _auto_single(func: str, A: Any, backend: str, return_diagnostics: bool):
    """FP64 with an a-posteriori cancellation check; rerun in the high-precision
    tier when risky. Returns the value, or (value, {tier, cancellation}) if asked."""
    if backend == "gpu":
        res = _gpu_auto_batched(func, np.asarray(A)[None, ...], return_diagnostics)
        if return_diagnostics:
            vals, diags = res
            return complex(vals[0]), diags[0]
        return complex(res[0])
    import cpu_ref
    fp = _evaluate(func, A, "fp64")
    kappa = cpu_ref.summation_condition_number(func, A, fp)
    if kappa < _AUTO_KAPPA_MAX:
        value, tier = complex(fp), "fp64"
    else:
        value, tier = complex(_evaluate(func, A, "ref")), "ref"
    return (value, {"tier": tier, "cancellation": float(kappa)}) if return_diagnostics else value


def _auto_batched(func: str, matrices: Iterable[Any], backend: str, return_diagnostics: bool):
    """Per-element auto: each evaluation independently keeps FP64 or reruns in the
    high-precision tier based on its own cancellation indicator."""
    if backend == "gpu":
        return _gpu_auto_batched(func, matrices, return_diagnostics)
    import cpu_ref
    batch = matrices if isinstance(matrices, np.ndarray) and matrices.ndim == 3 else list(matrices)
    out = np.empty(len(batch), dtype=np.complex128)
    diags: list[dict[str, Any]] = []
    for i, A in enumerate(batch):
        fp = _evaluate(func, A, "fp64")
        kappa = cpu_ref.summation_condition_number(func, A, fp)
        if kappa < _AUTO_KAPPA_MAX:
            out[i], tier = fp, "fp64"
        else:
            out[i], tier = _evaluate(func, A, "ref"), "ref"
        if return_diagnostics:
            diags.append({"tier": tier, "cancellation": float(kappa)})
    return (out, diags) if return_diagnostics else out


def _check_diag(precision, return_diagnostics):
    if return_diagnostics and precision not in ("auto", "certified"):
        raise ValueError("return_diagnostics=True requires precision='auto' or 'certified'")
    if precision == "certified" and not return_diagnostics:
        raise ValueError(
            "precision='certified' returns its error bound via diagnostics; call with "
            "return_diagnostics=True (a certificate you cannot see is not a certificate)"
        )


def _check_certified_input(A: Any) -> None:
    """Certified claims require finite binary64 input data."""
    try:
        finite = np.all(np.isfinite(np.asarray(A)))
    except (TypeError, ValueError):
        finite = False
    if not finite:
        raise ValueError("certified evaluation requires finite input entries")


def _finite_certificate(value: complex, bound: float) -> bool:
    return (np.isfinite(value.real) and np.isfinite(value.imag)
            and np.isfinite(bound) and bound >= 0.0)


def _certificate_rel(value: complex, bound: float) -> float:
    if not _finite_certificate(value, bound):
        return float("inf")
    magnitude = abs(value)
    if magnitude == 0.0:
        return 0.0 if bound == 0.0 else float("inf")
    return float(bound / magnitude)


def _certified_single(func: str, A: Any, backend: str, rtol: float | None = None):
    """(value, {tier, abs_error_bound, rel_error_bound, [escalated]}).

    The value is the certified evaluator's result (bit-for-bit the fp64
    reference for perm/haf/lhaf; the certified LU's own value for tor) with
    |value - exact| <= abs_error_bound rigorously. With ``rtol``, a bound that
    cannot prove rel error <= rtol escalates to the mpmath 'ref' tier: the
    *trigger* is rigorous (never trusts fp64 beyond proof — unlike kappa); the
    escalated value carries the same trust as precision='ref' everywhere else,
    and the diagnostics keep the fp64 bound that forced the escalation.

    ``backend="gpu"`` routes perm/haf through the on-device certified kernels
    (``core/certified.cu``: the plain kernels' value path + bound accumulators);
    lhaf/tor GPU-certified are the remaining R1 kernels."""
    _check_certified_input(A)
    if backend == "gpu":
        ext = _load_gpu_ext()
        kname = f"{func}_certified"
        if ext is None or not hasattr(ext, kname):
            raise NotImplementedError(
                f"precision='certified' on backend='gpu' needs the {kname} kernel "
                "(rebuild bindings/, or use backend='cpu')."
            )
        _, _, stack = _gpu_prepare(func, np.asarray(A)[None, ...], "fp64")
        values, bounds = (np.asarray(x) for x in getattr(ext, kname)(stack))
        value, bound = complex(values[0]), float(bounds[0])
    else:
        from cpu_ref import certified as _certified_eval
        value, bound = _certified_eval(func, A)
    value, bound = complex(value), float(bound)
    valid = _finite_certificate(value, bound)
    rel = _certificate_rel(value, bound)
    if rtol is None and not valid:
        raise FloatingPointError(
            f"certified {func} evaluator refused with a non-finite value or bound"
        )
    if rtol is not None and (not valid or not np.isfinite(rel) or rel > rtol):
        # the PROVEN ladder: certified-fp64 -> certified-dd (GPU) -> mpmath ref.
        # Each escalation is triggered by a rigorous bound, never a heuristic.
        if backend == "gpu":
            ext = _load_gpu_ext()
            dname = f"{func}_dd_certified"
            if ext is not None and hasattr(ext, dname):
                _, _, stack = _gpu_prepare(func, np.asarray(A)[None, ...], "fp64")
                v2, b2 = (np.asarray(x) for x in getattr(ext, dname)(stack))
                v2c, b2f = complex(v2[0]), float(b2[0])
                rel2 = _certificate_rel(v2c, b2f)
                if (_finite_certificate(v2c, b2f) and np.isfinite(rel2)
                        and rel2 <= rtol):
                    return v2c, {"tier": "certified-dd", "escalated": True,
                                 "abs_error_bound": b2f, "rel_error_bound": float(rel2)}
        ref = complex(_evaluate(func, A, "ref"))
        if not (np.isfinite(ref.real) and np.isfinite(ref.imag)):
            raise FloatingPointError(f"reference {func} evaluator returned a non-finite value")
        return ref, {"tier": "ref", "escalated": True,
                     "abs_error_bound": bound if valid else float("inf"),
                     "rel_error_bound": float(rel)}
    diag = {"tier": "certified-fp64", "abs_error_bound": float(bound),
            "rel_error_bound": float(rel)}
    if rtol is not None:
        diag["escalated"] = False
    return value, diag


def _certified_batched(func: str, matrices: Iterable[Any], backend: str,
                       rtol: float | None = None):
    if backend == "gpu": # one launch for the whole (uniform) stack
        ext = _load_gpu_ext()
        kname = f"{func}_certified"
        if ext is None or not hasattr(ext, kname):
            raise NotImplementedError(
                f"precision='certified' on backend='gpu' needs the {kname} kernel "
                "(rebuild bindings/, or use backend='cpu')."
            )
        _, _, stack = _gpu_prepare(func, matrices, "fp64")
        _check_certified_input(stack)
        values, bounds = (np.asarray(x) for x in getattr(ext, kname)(stack))
        out = values.astype(np.complex128, copy=True)
        diags = []
        for i in range(len(out)):
            v, e = complex(out[i]), float(bounds[i])
            valid = _finite_certificate(v, e)
            rel = _certificate_rel(v, e)
            if rtol is not None and (not valid or not np.isfinite(rel) or rel > rtol):
                # per-element proven ladder (certified-dd, then ref)
                out[i], d = _certified_single(func, stack[i], "gpu", rtol)
                diags.append(d)
            elif not valid:
                raise FloatingPointError(
                    f"certified {func} batch element {i} refused with a non-finite value or bound"
                )
            else:
                d = {"tier": "certified-fp64", "abs_error_bound": e,
                     "rel_error_bound": rel}
                if rtol is not None:
                    d["escalated"] = False
                diags.append(d)
        return out, diags
    batch = matrices if isinstance(matrices, np.ndarray) and matrices.ndim == 3 else list(matrices)
    out = np.empty(len(batch), dtype=np.complex128)
    diags: list[dict[str, Any]] = []
    for i, A in enumerate(batch):
        out[i], d = _certified_single(func, A, backend, rtol)
        diags.append(d)
    return out, diags


def _check_rtol(precision, rtol):
    if rtol is not None and precision != "certified":
        raise ValueError("rtol is a certified-tier parameter; it requires precision='certified'")
    if rtol is not None and (not np.isfinite(rtol) or rtol < 0.0):
        raise ValueError("rtol must be finite and non-negative")


def _dispatch_batched(func, matrices, precision, backend, return_diagnostics=False,
                      rtol=None):
    _check_diag(precision, return_diagnostics)
    _check_rtol(precision, rtol)
    if precision == "auto":
        return _auto_batched(func, matrices, backend, return_diagnostics)
    if precision == "certified":
        return _certified_batched(func, matrices, backend, rtol)
    if backend == "gpu":
        return _gpu_batched(func, matrices, precision)
    if backend == "cpu":
        return _batched(func, matrices, precision)
    raise ValueError(f"unknown backend {backend!r}; expected 'cpu' or 'gpu'")


def _dispatch_single(func, A, precision, backend, return_diagnostics=False, rtol=None):
    _check_diag(precision, return_diagnostics)
    _check_rtol(precision, rtol)
    if precision == "auto":
        return _auto_single(func, A, backend, return_diagnostics)
    if precision == "certified":
        return _certified_single(func, A, backend, rtol)
    if backend == "gpu":
        return complex(_gpu_batched(func, np.asarray(A)[None, ...], precision)[0])
    if backend == "cpu":
        return _evaluate(func, A, precision)
    raise ValueError(f"unknown backend {backend!r}; expected 'cpu' or 'gpu'")


def perm(A: Any, precision: str = "fp64", backend: str = "cpu", return_diagnostics: bool = False, rtol: float | None = None):
    """Permanent of a single square matrix ``A`` at the given precision/backend."""
    return _dispatch_single("perm", A, precision, backend, return_diagnostics, rtol)


def perm_batched(matrices: Iterable[Any], precision: str = "fp64", backend: str = "cpu", return_diagnostics: bool = False, rtol: float | None = None):
    """Permanents of a batch of square matrices -> ``complex128`` vector."""
    return _dispatch_batched("perm", matrices, precision, backend, return_diagnostics, rtol)


def haf(A: Any, precision: str = "fp64", backend: str = "cpu", return_diagnostics: bool = False, rtol: float | None = None):
    """Hafnian of a single symmetric matrix ``A`` at the given precision/backend."""
    return _dispatch_single("haf", A, precision, backend, return_diagnostics, rtol)


def haf_batched(matrices: Iterable[Any], precision: str = "fp64", backend: str = "cpu", return_diagnostics: bool = False, rtol: float | None = None):
    """Hafnians of a batch of symmetric matrices -> ``complex128`` vector.

    The batched surface for Gaussian boson sampling (docs/DESIGN.md §7): one hafnian
    per element, batched across the grid (one evaluation per thread; see the README
    "Known limitations" for the cooperative-kernel status)."""
    return _dispatch_batched("haf", matrices, precision, backend, return_diagnostics, rtol)


def lhaf(A: Any, precision: str = "fp64", backend: str = "cpu", return_diagnostics: bool = False, rtol: float | None = None):
    """Loop hafnian of a single symmetric matrix ``A`` (for nonzero displacement)."""
    return _dispatch_single("lhaf", A, precision, backend, return_diagnostics, rtol)


def lhaf_batched(matrices: Iterable[Any], precision: str = "fp64", backend: str = "cpu", return_diagnostics: bool = False, rtol: float | None = None):
    """Loop hafnians of a batch of symmetric matrices -> ``complex128`` vector."""
    return _dispatch_batched("lhaf", matrices, precision, backend, return_diagnostics, rtol)


def tor(O: Any, precision: str = "fp64", backend: str = "cpu", return_diagnostics: bool = False, rtol: float | None = None):
    """Torontonian of a single ``2n x 2n`` matrix ``O`` (threshold detectors)."""
    return _dispatch_single("tor", O, precision, backend, return_diagnostics, rtol)


def tor_batched(matrices: Iterable[Any], precision: str = "fp64", backend: str = "cpu", return_diagnostics: bool = False, rtol: float | None = None):
    """Torontonians of a batch of ``2n x 2n`` matrices -> ``complex128`` vector."""
    return _dispatch_batched("tor", matrices, precision, backend, return_diagnostics, rtol)


def lhaf_repeated(A: Any, gamma: Any, reps: Any, backend: str = "cpu",
                  certified: bool = False):
    """Loop hafnians of ``A`` expanded by repetition patterns.

    ``A`` is the (M, M) base matrix, ``gamma`` the length-M loop-weight vector
    (``None`` -> zeros: the ordinary hafnian of the expansion), and ``reps`` a
    (B, M) table of non-negative repetition counts (or a single length-M
    pattern). Returns the (B,) values (or a scalar for a single pattern) of the
    finite-difference sieve -- cost ``prod(n_i + 1)`` per pattern instead of
    ``2^(N/2)`` on the expansion (``cpu_ref.repeated`` derives and pins the
    identities). The GBS sampling workload in its native shape.

    ``certified=True`` runs the R1-machinery variant: values bit-identical to
    the plain sieve plus a rigorous per-pattern bound; returns ``(values,
    {"tier": "certified-fp64", "abs_error_bound": (B,) array})``. Enclosure
    (``|value - exact| <= bound``) is a hard test invariant.
    """
    A = np.ascontiguousarray(A, dtype=np.complex128)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"lhaf_repeated: A must be square (M, M), got {A.shape}")
    M = A.shape[0]
    g = (np.zeros(M, dtype=np.complex128) if gamma is None
         else np.ascontiguousarray(gamma, dtype=np.complex128))
    if certified:
        _check_certified_input(A)
        _check_certified_input(g)
    r = np.ascontiguousarray(reps, dtype=np.int32)
    single = r.ndim == 1
    if single:
        r = r[None, :]
    if r.ndim != 2 or r.shape[1] != M:
        raise ValueError(f"lhaf_repeated: reps must be (B, {M}) or ({M},), got {np.asarray(reps).shape}")
    if (r < 0).any(): # the GPU kernel's odometer assumes non-negative counts
        raise ValueError("lhaf_repeated: reps must be non-negative")
    bounds = None
    if backend == "gpu":
        ext = _load_gpu_ext()
        if ext is None or not hasattr(ext, "lhaf_repeated"):
            raise RuntimeError(
                "backend='gpu' needs the compiled extension with the lhaf_repeated "
                "kernel (rebuild bindings/), or use backend='cpu'.")
        if certified:
            if not hasattr(ext, "lhaf_repeated_certified"):
                raise RuntimeError("lhaf_repeated(certified=True) needs a rebuilt extension")
            v, bnd = ext.lhaf_repeated_certified(A, g, r)
            out = np.asarray(v).astype(np.complex128, copy=True)
            bounds = np.asarray(bnd).astype(np.float64, copy=True)
        else:
            out = np.asarray(ext.lhaf_repeated(A, g, r)).astype(np.complex128, copy=True)
    elif backend == "cpu":
        import cpu_ref
        if certified:
            from cpu_ref.repeated import lhaf_repeated_certified as _crt
            pairs = [_crt(A, g, row) for row in r]
            out = np.array([p[0] for p in pairs], dtype=np.complex128)
            bounds = np.array([p[1] for p in pairs], dtype=np.float64)
        else:
            out = np.array([cpu_ref.lhaf_repeated(A, g, row) for row in r], dtype=np.complex128)
    else:
        raise ValueError(f"unknown backend {backend!r}; expected 'cpu' or 'gpu'")
    if certified:
        valid = (np.all(np.isfinite(out.real)) and np.all(np.isfinite(out.imag))
                 and np.all(np.isfinite(bounds)) and np.all(bounds >= 0.0))
        if not valid:
            raise FloatingPointError(
                "certified lhaf_repeated evaluator refused with a non-finite "
                "value or invalid bound"
            )
        diag = {"tier": "certified-fp64",
                "abs_error_bound": (float(bounds[0]) if single else bounds)}
        return (complex(out[0]) if single else out), diag
    return complex(out[0]) if single else out


def tor_single(O: Any, groups: int | None = None, certified: bool = False,
               dd: bool = False):
    """SINGLE-LARGE torontonian of a real ``(2n, 2n)`` matrix, ``2n <= 64``.

    One evaluation split into ``2**groups`` prefix-Cholesky subtrees across the
    whole GPU (``core/tor_recursive.cu``) -- the path to torontonians BEYOND the
    batched kernels' dim-24 cap, up to 32 modes (the published validation
    ceiling is 26 clicks = dim 52). The matrix after binary64 conversion must
    be finite and exactly symmetric; the recursive Cholesky walk consumes one
    triangle, so approximate symmetry is not an unambiguous input contract.
    Physical-domain (SPD ``I - O_S``) inputs are required; off-domain raises.
    ``groups`` defaults to ``min(n, 14)``.

    With ``certified=True``, also returns a rigorous a-posteriori error bound.
    ``dd=True`` (implies certified) carries the value in double-double, giving a
    TIGHT certificate past the fp64 precision wall (where the fp64 certified
    bound exceeds the value); the returned tier is ``"certified-dd"``.
    """
    A = np.asarray(O)
    if np.iscomplexobj(A):
        if np.any(np.imag(A) != 0):
            raise ValueError("tor_single is real-domain only; use tor() for complex O (dim <= 24)")
        A = np.real(A)
    M = np.ascontiguousarray(A, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] % 2 != 0:
        raise ValueError(f"tor_single: O must be square with even dimension, got {M.shape}")
    if M.shape[0] > 64:
        raise ValueError(f"tor_single: dimension {M.shape[0]} exceeds the cap 64 (n <= 32 modes)")
    if not np.all(np.isfinite(M)):
        raise ValueError("tor_single: O must contain only finite binary64 values")
    if not np.array_equal(M, M.T):
        raise ValueError("tor_single: O must be exactly symmetric after conversion to binary64")
    n = M.shape[0] // 2
    g = min(n, 14) if groups is None else int(groups)
    if not 0 <= g <= n:
        raise ValueError(f"groups must be in [0, {n}]")
    ext = _load_gpu_ext()
    if ext is None or not hasattr(ext, "tor_single"):
        raise RuntimeError("tor_single needs the compiled extension with the single-large "
                           "kernel (rebuild bindings/)")
    if certified or dd:
        tier = "certified-dd" if dd else "certified-fp64"
        fn = "tor_single_ddcertified" if dd else "tor_single_certified"
        if not hasattr(ext, fn):
            raise RuntimeError(f"tor_single({tier}) needs a rebuilt extension")
        v, e = getattr(ext, fn)(M, g)
        v, e = float(v), float(e)
        if not (np.isfinite(v) and np.isfinite(e) and e >= 0.0):
            raise ValueError("tor_single: off the physical domain or uncertifiable "
                             "(the bound refuses rather than overclaims)")
        rel = e / abs(v) if v != 0 else float("inf")
        return v, {"tier": tier, "abs_error_bound": e, "rel_error_bound": rel}
    v = float(ext.tor_single(M, g))
    if not np.isfinite(v):
        raise ValueError("tor_single: input is off the physical domain (I - O_S not SPD); "
                         "no complex-LU fallback exists beyond dim 24")
    return v


def gpu_available() -> bool:
    """True if the compiled CUDA extension is importable (GPU backend usable)."""
    return _load_gpu_ext() is not None


def gpu_backend_kind() -> str:
    """Honest provenance of the loaded GPU extension, for labelling results.

    ``"gpu"`` is claimed only when the extension was compiled by nvcc for a real
    device; a CPU host-shim build (``-DGBS_HOST_SHIM=ON``, e.g. on macOS or a
    no-GPU CI runner) reports ``"host-shim"`` so an emulated run is never recorded
    as a real-device measurement. ``"none"`` if no extension is importable.

    The signal is the binary's own ``__host_shim__`` flag (set at compile time),
    not the host OS -- so a Linux CI box without a GPU is correctly ``"host-shim"``.
    """
    ext = _load_gpu_ext()
    if ext is None:
        return "none"
    return "host-shim" if getattr(ext, "__host_shim__", True) else "gpu"


# --- Device-resident + bucketing handle (docs/device_resident_contract.md) ---


def _as_square_list(matrices: Iterable[Any]) -> list[np.ndarray]:
    """Coerce a (possibly ragged) iterable of matrices to a list of contiguous
    complex128 square 2-D arrays, validating each. A 3-D stack iterates into its
    2-D slices, so a uniform batch is just the single-bucket degenerate case."""
    out: list[np.ndarray] = []
    for i, m in enumerate(matrices):
        a = np.ascontiguousarray(np.asarray(m, dtype=np.complex128))
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError(
                f"Workspace: matrix #{i} must be square 2-D; got shape {a.shape}"
            )
        out.append(a)
    return out


class Workspace:
    """Device-resident, bucketing handle for ragged, sampler-shaped workloads.

    The contract is ``docs/device_resident_contract.md``. A context manager that
    (1) accepts a **ragged** list of differently-sized matrices -- the shape a GBS
    sampler's growing-submatrix chain produces (docs/DESIGN.md §2.3) -- grouping it into
    uniform per-size launches and scattering results back to input order, and
    (2) on the GPU backend reuses one device-resident buffer set across every call
    (when the extension exposes a ``Session``), so a sampler *loop* pays allocation
    once, not per step.

        with gbskernels.Workspace(backend="gpu") as ws:
            hafs = ws.haf_batched([submatrix(B, p) for p in patterns])

    The result for input ``i`` is identical to evaluating its matrix alone --
    bucketing changes grouping, never values (the central invariant, tested in
    ``tests/test_workspace_bucketing.py``). ``backend="cpu"`` degenerates the
    buckets to the CPU reference loop, so the same sampler code is differential-
    tested CPU-vs-GPU.

    **v2 (device-resident).** The GPU session uses a persistent CUDA stream with
    pinned host staging and async copies, and :meth:`perm_resident` /
    :meth:`haf_resident` / :meth:`lhaf_resident` / :meth:`tor_resident` keep a uniform
    batch's result **on the device**, returning a zero-copy DLPack handle (a CuPy
    array on a real GPU build; a numpy array on the host-shim CPU "device") so a
    downstream CuPy/PyTorch pipeline consumes it without a D2H round trip. See
    ``docs/device_resident_contract.md`` for the full contract and the v3 boundary.
    """

    def __init__(self, backend: str = "gpu") -> None:
        if backend not in ("cpu", "gpu"):
            raise ValueError(f"unknown backend {backend!r}; expected 'cpu' or 'gpu'")
        self._backend = backend
        self._session: Any = None
        self._closed = False

    def __enter__(self) -> "Workspace":
        if self._backend == "gpu":
            ext = _load_gpu_ext()
            if ext is None:
                raise RuntimeError(
                    "Workspace(backend='gpu') needs the compiled extension "
                    "'gbskernels_ext'; build bindings/ or set GBSKERNELS_EXT_DIR."
                )
            # Use the device-resident session for buffer reuse when the extension
            # provides it; otherwise still bucket, via the per-call path.
            if hasattr(ext, "Session"):
                self._session = ext.Session()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False

    def close(self) -> None:
        """Release the device-resident session. Idempotent (double-close is a no-op)."""
        if self._closed:
            return
        if self._session is not None and hasattr(self._session, "close"):
            self._session.close()
        self._session = None
        self._closed = True

    def perm_batched(self, matrices: Iterable[Any], precision: str = "fp64") -> np.ndarray:
        """Permanents of a ragged batch -> complex128 vector, in input order."""
        return self._ragged("perm", matrices, precision)

    def haf_batched(self, matrices: Iterable[Any], precision: str = "fp64") -> np.ndarray:
        """Hafnians of a ragged batch -> complex128 vector, in input order."""
        return self._ragged("haf", matrices, precision)

    def lhaf_batched(self, matrices: Iterable[Any], precision: str = "fp64") -> np.ndarray:
        """Loop hafnians of a ragged batch -> complex128 vector, in input order."""
        return self._ragged("lhaf", matrices, precision)

    def tor_batched(self, matrices: Iterable[Any], precision: str = "fp64") -> np.ndarray:
        """Torontonians of a ragged batch -> complex128 vector, in input order."""
        return self._ragged("tor", matrices, precision)

    def _ragged(self, func: str, matrices: Iterable[Any], precision: str) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Workspace is closed")
        mats = _as_square_list(matrices)
        out = np.empty(len(mats), dtype=np.complex128)
        if not mats:
            return out
        # Group by matrix dimension, preserving first-seen order; one uniform
        # launch per bucket, scattered back to each input's original position.
        buckets: dict[int, list[int]] = {}
        for i, a in enumerate(mats):
            buckets.setdefault(a.shape[0], []).append(i)
        for d, idxs in buckets.items():
            stack = np.ascontiguousarray(np.stack([mats[i] for i in idxs]))
            try:
                res = self._dispatch_bucket(func, stack, precision)
            except ValueError as e: # name the first input in the rejected bucket
                raise ValueError(f"matrix #{idxs[0]} (dim {d}): {e}") from e
            for k, i in enumerate(idxs):
                out[i] = res[k]
        return out

    def _dispatch_bucket(self, func: str, stack: np.ndarray, precision: str) -> np.ndarray:
        if precision == "auto":
            # Auto is per-element (each element independently keeps its FP64 value or
            # reruns in a high-precision tier on its own cancellation indicator). Apply
            # it WITHIN the uniform bucket, so a ragged auto batch is bucket-wise auto and
            # element i's value is identical to evaluating its matrix alone -- the central
            # bucketing invariant, now honored for 'auto' too. (The GPU auto path uses the
            # per-call extension, not the resident session: correctness over buffer reuse;
            # fp64/dd buckets below still reuse the session.)
            if self._backend == "cpu":
                return _auto_batched(func, stack, "cpu", return_diagnostics=False)
            return _gpu_auto_batched(func, stack, return_diagnostics=False)
        if self._backend == "cpu":
            return _batched(func, stack, precision)
        ext, name, stack = _gpu_prepare(func, stack, precision)
        # Device-resident session (buffer reuse) when available, else per-call path.
        target = self._session if self._session is not None else ext
        return np.asarray(getattr(target, name)(stack))

    # --- v2: device-resident output (zero-copy DLPack handle) ----------------

    def perm_resident(self, matrices: Iterable[Any], precision: str = "fp64"):
        """Permanents of a *uniform* batch, left ON THE DEVICE -> DLPack handle."""
        return self._resident("perm", matrices, precision)

    def haf_resident(self, matrices: Iterable[Any], precision: str = "fp64"):
        """Hafnians of a *uniform* batch, left ON THE DEVICE -> DLPack handle."""
        return self._resident("haf", matrices, precision)

    def lhaf_resident(self, matrices: Iterable[Any], precision: str = "fp64"):
        """Loop hafnians of a *uniform* batch, left ON THE DEVICE -> DLPack handle."""
        return self._resident("lhaf", matrices, precision)

    def tor_resident(self, matrices: Iterable[Any], precision: str = "fp64"):
        """Torontonians of a *uniform* batch, left ON THE DEVICE -> DLPack handle."""
        return self._resident("tor", matrices, precision)

    def _resident(self, func: str, matrices: Iterable[Any], precision: str):
        """Evaluate a UNIFORM batch keeping the result on the device; return a
        zero-copy, DLPack-exportable handle (a CuPy array on a real GPU build, a numpy
        array on the host-shim CPU "device") so a downstream CuPy/PyTorch pipeline
        consumes it without a D2H round trip -- the v2 "genuinely device-resident"
        path. Ragged batches (which would need an on-device scatter) use the host-result
        ``*_batched`` instead. Precision is fp64 (a resident DD path is future work)."""
        if self._closed:
            raise RuntimeError("Workspace is closed")
        if self._backend != "gpu":
            raise NotImplementedError(
                "device-resident output requires backend='gpu' (on CPU the result is "
                "already a host array; use *_batched)."
            )
        if precision != "fp64":
            raise NotImplementedError(
                "device-resident output is fp64 only so far; use precision='fp64' "
                "(a resident DD path is future work)."
            )
        try:
            stack = np.ascontiguousarray(np.asarray(matrices, dtype=np.complex128))
            uniform = stack.ndim == 3
        except (ValueError, TypeError):
            uniform = False
        if not uniform:
            raise ValueError(
                "device-resident output needs a uniform (B, d, d) batch; a ragged batch "
                "would require an on-device scatter -- use *_batched (host results) instead."
            )
        ext, name, stack = _gpu_prepare(func, stack, precision) # validates; augments odd lhaf
        session = self._session
        if session is None or not hasattr(session, f"{name}_resident"):
            raise RuntimeError(
                "device-resident output needs the extension's Session.*_resident API; "
                "rebuild bindings/ (it postdates older builds)."
            )
        return getattr(session, f"{name}_resident")(stack)
