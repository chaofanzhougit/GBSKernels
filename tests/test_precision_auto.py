"""precision="auto" -- the adaptive tier that ships the work-precision result.

FP64 computes the fast result plus a cancellation indicator
(kappa = sum|terms|/|result|, cpu_ref.summation_condition_number); well-conditioned
evaluations return the FP64 value immediately, risky ones rerun in the
high-precision tier (mpmath on the CPU backend). The value is returned, plus optional
diagnostics {tier, cancellation}. This turns the measured FP64<->high-precision
boundary (docs/DESIGN.md §6) into a user-facing feature.
"""

from __future__ import annotations

import mpmath
import numpy as np
import pytest

import gbskernels
import highprec_ref as hp
from bench import _inputs as I

# func -> (single fn, physical input, adversarial input, mpmath reference)
_CASES = {
    "perm": (gbskernels.perm, lambda: I.physical_permanent(8, 1),
             lambda: I.make_cancellation_matrix(6, 1e-10, 1), hp.permanent_mp),
    "haf": (gbskernels.haf, lambda: I.physical_hafnian(8, 1),
            lambda: I.cancellation_hafnian(1e-10, 1), hp.hafnian_mp),
    "lhaf": (gbskernels.lhaf, lambda: I.physical_loop_hafnian(8, 1),
             lambda: I.cancellation_loop_hafnian(1e-10, 1), hp.loop_hafnian_mp),
    "tor": (gbskernels.tor, lambda: I.physical_torontonian(4, 1),
            lambda: I.cancellation_torontonian(1e-10), hp.torontonian_mp),
}


def _rel(v, exact_mp):
    return float(abs(mpmath.mpc(complex(v)) - exact_mp) / abs(exact_mp))


@pytest.mark.parametrize("func", list(_CASES))
def test_auto_returns_fast_fp64_when_well_conditioned(func):
    fn, physical, _, _ = _CASES[func]
    A = physical()
    value, diag = fn(A, precision="auto", return_diagnostics=True)
    assert diag["tier"] == "fp64", f"{func}: well-conditioned should stay FP64 (kappa {diag['cancellation']:.1e})"
    assert diag["cancellation"] < gbskernels._AUTO_KAPPA_MAX
    # the returned value is exactly the FP64 value (no rerun)
    assert value == complex(fn(A, precision="fp64"))


@pytest.mark.parametrize("func", list(_CASES))
def test_auto_reruns_and_recovers_when_ill_conditioned(func):
    fn, _, adversarial, mpref = _CASES[func]
    A = adversarial()
    value, diag = fn(A, precision="auto", return_diagnostics=True)
    with mpmath.workdps(60):
        exact = mpref(A, dps=60)
    assert diag["tier"] == "ref", f"{func}: heavy cancellation (kappa {diag['cancellation']:.1e}) should rerun"
    assert diag["cancellation"] >= gbskernels._AUTO_KAPPA_MAX
    # auto recovered accuracy where plain FP64 had lost many digits. (Assert the
    # *relative* recovery, not FP64's absolute error magnitude -- the latter sits
    # right at ~2e-7 and flips around any fixed threshold across platforms, the same
    # libc++/libstdc++ FP64-rounding sensitivity as the differential gates.)
    auto_rel = _rel(value, exact)
    fp64_rel = _rel(fn(A, precision="fp64"), exact)
    assert auto_rel < 1e-12                       # auto is essentially exact
    assert fp64_rel > 1e-9                         # FP64 genuinely degraded (robust both platforms)
    assert fp64_rel > 1e4 * max(auto_rel, 1e-18)   # auto is dramatically better (the recovery)


def test_auto_batched_selects_per_element():
    mats = [I.physical_hafnian(6, 1), I.cancellation_hafnian(1e-10, 1), I.physical_hafnian(8, 2)]
    vals, diags = gbskernels.haf_batched(mats, precision="auto", return_diagnostics=True)
    assert vals.shape == (3,) and vals.dtype == np.complex128
    assert [d["tier"] for d in diags] == ["fp64", "ref", "fp64"]
    # without diagnostics it is just the value array
    plain = gbskernels.haf_batched(mats, precision="auto")
    assert np.allclose(plain, vals)


def test_auto_diagnostics_guard_and_value_only():
    A = I.physical_permanent(6, 1)
    # value-only (no diagnostics) returns a bare complex
    assert isinstance(gbskernels.perm(A, precision="auto"), complex)
    # return_diagnostics only makes sense with auto
    with pytest.raises(ValueError, match="requires precision='auto'"):
        gbskernels.perm(A, precision="fp64", return_diagnostics=True)


def test_auto_gpu_backend_requires_the_extension():
    # GPU auto ships for all four functions (each *_kappa kernel emits sum|term| in
    # the FP64 pass; the FP64->DD rerun is validated in tests/test_gpu_bindings.py
    # against the host-shim build). Without a built extension it must raise clearly
    # rather than silently doing the wrong thing (this path needs no extension).
    if gbskernels.gpu_available():
        pytest.skip("a GPU extension is importable; the no-extension guard does not apply")
    for fn, A in [(gbskernels.perm, I.physical_permanent(4, 1)),
                  (gbskernels.haf, I.physical_hafnian(4, 1)),
                  (gbskernels.tor, I.physical_torontonian(2, 1))]:
        with pytest.raises(RuntimeError, match="extension"):
            fn(A, precision="auto", backend="gpu")
