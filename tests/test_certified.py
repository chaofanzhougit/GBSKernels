"""Certified tier: the enclosure contract, tested as a hard gate.

The certified tier's claim is falsifiable and this file falsifies-or-passes it:

1. **Bit-equality** — the certified value IS the fp64 reference value
   (``permanent_glynn`` / ``hafnian_powertrace``), so the certificate applies to
   the number the fast path returns.
2. **Enclosure** — ``|value - exact| <= abs_error_bound`` on EVERY input:
   closed forms (exact integers, no oracle needed), random real/complex vs the
   mpmath reference, and — the cases that matter — the adversarial cancellation
   families where plain FP64 silently loses digits. Unlike the ``kappa``
   calibration (a measured correlation), this is an invariant: one violation is
   a bug.
3. **Usefulness** — a valid-but-vacuous bound would also pass (2); assert the
   bound is tight enough to act on: small on well-conditioned inputs, and
   large exactly where FP64 is actually wrong (so the escalation logic built on
   it never over-trusts).
"""

from __future__ import annotations

from math import factorial

import numpy as np
import pytest

import cpu_ref
import gbskernels
from bench._inputs import adversarial_permanent, cancellation_hafnian
from cpu_ref.certified import certified_hafnian, certified_permanent
from highprec_ref import hafnian_mp, permanent_mp

pytestmark = pytest.mark.layer5

DPS = 50


def _rand(n: int, seed: int, cplx: bool = True) -> np.ndarray:
    g = np.random.default_rng(seed)
    A = g.standard_normal((n, n))
    if cplx:
        A = A + 1j * g.standard_normal((n, n))
    return A


def _sym(N: int, seed: int, cplx: bool = True) -> np.ndarray:
    A = _rand(N, seed, cplx)
    return A + A.T


# --------------------------------------------------------------------------
# 1. bit-equality: the certificate covers the shipped fp64 value
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 5, 7])
def test_certified_permanent_value_is_glynn_bitforbit(n):
    for seed in range(3):
        A = _rand(n, 10 * n + seed)
        v, _ = certified_permanent(A)
        assert v == cpu_ref.permanent_glynn(A)


@pytest.mark.parametrize("N", [2, 4, 6, 8])
def test_certified_hafnian_value_is_powertrace_bitforbit(N):
    for seed in range(3):
        A = _sym(N, 20 * N + seed)
        v, _ = certified_hafnian(A)
        assert v == cpu_ref.hafnian_powertrace(A)


# --------------------------------------------------------------------------
# 2. enclosure: closed forms (oracle-free, exact integers)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 4, 8, 10, 12])
def test_certified_permanent_encloses_factorial(n):
    v, e = certified_permanent(np.ones((n, n)))
    assert abs(v - factorial(n)) <= e
    assert e / factorial(n) < 1e-9 # and the bound is tight, not vacuous


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_certified_hafnian_encloses_double_factorial(n):
    dfact = 1
    for k in range(2 * n - 1, 0, -2):
        dfact *= k
    v, e = certified_hafnian(np.ones((2 * n, 2 * n)))
    assert abs(v - dfact) <= e
    assert e / dfact < 1e-9


def test_certified_permanent_identity_and_empty():
    v, e = certified_permanent(np.eye(6))
    assert abs(v - 1.0) <= e and e < 1e-12
    v, e = certified_permanent(np.empty((0, 0)))
    assert v == 1.0 and e == 0.0


def test_certified_hafnian_odd_and_empty():
    v, e = certified_hafnian(_sym(4, 0)[:3, :3])
    assert v == 0.0 and e == 0.0
    v, e = certified_hafnian(np.empty((0, 0)))
    assert v == 1.0 and e == 0.0


# --------------------------------------------------------------------------
# 2'. enclosure: random inputs vs the mpmath reference
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("cplx", [False, True])
def test_certified_permanent_encloses_mpmath(n, cplx):
    for seed in range(3):
        A = _rand(n, 100 * n + seed, cplx)
        v, e = certified_permanent(A)
        exact = complex(permanent_mp(A, dps=DPS))
        assert abs(v - exact) <= e


@pytest.mark.parametrize("N", [2, 4, 6])
@pytest.mark.parametrize("cplx", [False, True])
def test_certified_hafnian_encloses_mpmath(N, cplx):
    for seed in range(3):
        A = _sym(N, 200 * N + seed, cplx)
        v, e = certified_hafnian(A)
        exact = complex(hafnian_mp(A, dps=DPS))
        assert abs(v - exact) <= e


# --------------------------------------------------------------------------
# 2''. enclosure where it counts: the adversarial cancellation families
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(4))
def test_certified_permanent_encloses_adversarial(seed):
    A = adversarial_permanent(6, seed)
    v, e = certified_permanent(A)
    exact = complex(permanent_mp(A, dps=DPS))
    assert abs(v - exact) <= e


@pytest.mark.parametrize("delta", [1e-6, 1e-10, 1e-13])
def test_certified_hafnian_encloses_cancellation_family(delta):
    A = cancellation_hafnian(delta, seed=0)
    v, e = certified_hafnian(A)
    exact = complex(hafnian_mp(A, dps=DPS))
    err = abs(v - exact)
    assert err <= e
    # usefulness: where fp64 actually degrades, the bound must say so --
    # never certify more digits than are true.
    if abs(exact) > 0:
        assert e / abs(exact) >= err / abs(exact)


def test_certified_bound_signals_fp64_breakdown():
    """At delta=1e-10 the fp64 hafnian has ~6 wrong digits; the certificate must
    not claim better than ~that (measured rel bound ~1e-2 -- an escalation
    trigger, not an overclaim)."""
    A = cancellation_hafnian(1e-10, seed=0)
    _, e = certified_hafnian(A)
    exact = abs(complex(hafnian_mp(A, dps=DPS)))
    assert e / exact > 1e-6


# --------------------------------------------------------------------------
# 3. usefulness: tight on well-conditioned inputs
# --------------------------------------------------------------------------

def test_certified_bounds_are_tight_on_wellconditioned():
    for seed in range(3):
        A = _rand(6, 500 + seed)
        v, e = certified_permanent(A)
        assert e / abs(v) < 1e-11
        S = _sym(6, 600 + seed)
        v, e = certified_hafnian(S)
        assert e / abs(v) < 1e-10


# --------------------------------------------------------------------------
# public API surface
# --------------------------------------------------------------------------

def test_api_certified_single_and_batched():
    A = _rand(4, 1)
    v, d = gbskernels.perm(A, precision="certified", return_diagnostics=True)
    assert v == cpu_ref.permanent_glynn(A)
    assert d["tier"] == "certified-fp64"
    assert abs(v - complex(permanent_mp(A, dps=DPS))) <= d["abs_error_bound"]
    assert d["rel_error_bound"] == d["abs_error_bound"] / abs(v)

    stack = np.stack([_sym(4, s) for s in range(3)])
    vals, diags = gbskernels.haf_batched(stack, precision="certified",
                                         return_diagnostics=True)
    assert len(vals) == len(diags) == 3
    for i in range(3): # batched-equals-looped, values and bounds
        vi, di = gbskernels.haf(stack[i], precision="certified",
                                return_diagnostics=True)
        assert vals[i] == vi
        assert diags[i]["abs_error_bound"] == di["abs_error_bound"]


def test_api_certified_requires_diagnostics():
    with pytest.raises(ValueError, match="certificate"):
        gbskernels.perm(np.eye(3), precision="certified")


# --------------------------------------------------------------------------
# loop hafnian + torontonian (R1 completion)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("N", [2, 4, 6])
def test_certified_lhaf_even_bitforbit_and_enclosed(N):
    from cpu_ref.certified import certified_loop_hafnian
    from highprec_ref import loop_hafnian_mp
    for seed in range(3):
        A = _sym(N, 40 * N + seed)
        v, e = certified_loop_hafnian(A)
        assert v == cpu_ref.loop_hafnian_powertrace(A)
        assert abs(v - complex(loop_hafnian_mp(A, dps=DPS))) <= e


@pytest.mark.parametrize("N", [1, 3, 5])
def test_certified_lhaf_odd_bitforbit_and_enclosed(N):
    from cpu_ref.certified import certified_loop_hafnian
    from highprec_ref import loop_hafnian_mp
    A = _sym(N, 7 * N, cplx=False)
    v, e = certified_loop_hafnian(A)
    assert v == cpu_ref.loop_hafnian_naive(A)
    assert abs(v - complex(loop_hafnian_mp(A, dps=DPS))) <= e


def test_certified_lhaf_reduces_to_hafnian_when_diagonal_zero():
    from cpu_ref.certified import certified_hafnian, certified_loop_hafnian
    A = _sym(6, 99)
    np.fill_diagonal(A, 0.0)
    vl, el = certified_loop_hafnian(A)
    vh, eh = certified_hafnian(A)
    assert abs(vl - vh) <= el + eh # both enclose the same exact value


@pytest.mark.parametrize("n_modes", [2, 3, 4])
def test_certified_tor_enclosed_on_physical_domain(n_modes):
    from bench._inputs import physical_torontonian
    from cpu_ref.certified import certified_torontonian
    from highprec_ref import torontonian_mp
    for seed in range(3):
        O = physical_torontonian(n_modes, seed)
        v, e = certified_torontonian(O)
        exact = complex(torontonian_mp(O, dps=DPS))
        assert abs(v - exact) <= e
        assert np.isfinite(e) # certifiable on physical inputs
        # the certified LU's value agrees with the library reference to tier tol
        assert abs(v - cpu_ref.tor(O)) <= 1e-11 * max(1.0, abs(v))


def test_certified_tor_closed_form():
    a, n = 0.2, 3
    from cpu_ref.certified import certified_torontonian
    v, e = certified_torontonian(a * np.eye(2 * n))
    exact = (a / (1 - a)) ** n
    assert abs(v - exact) <= e and e / exact < 1e-9


def test_certified_tor_branch_risk_degrades_to_inf_not_silence():
    """A subset determinant whose disc could cross the branch cut must yield an
    inf bound (cannot certify), never a finite overclaim."""
    from cpu_ref.certified import _p_inv_sqrt
    _, e = _p_inv_sqrt(complex(-1.0, 0.0), 1e-12) # on the cut within ez
    assert e == float("inf")


# --------------------------------------------------------------------------
# rtol: the provably-safe escalation (the rigorous 'auto')
# --------------------------------------------------------------------------

def test_certified_rtol_escalates_only_when_unprovable():
    A_bad = cancellation_hafnian(1e-10, seed=0)
    v, d = gbskernels.haf(A_bad, precision="certified", return_diagnostics=True,
                          rtol=1e-8)
    assert d["tier"] == "ref" and d["escalated"] is True
    # the escalated value is the mpmath reference
    assert abs(v - complex(hafnian_mp(A_bad, dps=DPS))) < 1e-20 + 1e-10 * abs(v)

    S = _sym(6, 123)
    v, d = gbskernels.haf(S, precision="certified", return_diagnostics=True,
                          rtol=1e-8)
    assert d["tier"] == "certified-fp64" and d["escalated"] is False
    assert d["rel_error_bound"] <= 1e-8


def test_certified_rtol_requires_certified_precision():
    with pytest.raises(ValueError, match="rtol"):
        gbskernels.haf(_sym(4, 3), rtol=1e-8)


# --------------------------------------------------------------------------
# the GPU-kernel path (host-shim in CI; the same binary logic runs on-device)
# --------------------------------------------------------------------------

def _gpu_certified_available() -> bool:
    ext = gbskernels._load_gpu_ext()
    return ext is not None and hasattr(ext, "perm_certified")


needs_gpu_certified = pytest.mark.skipif(
    not _gpu_certified_available(),
    reason="gbskernels_ext with certified kernels not built (bindings/README.md)")


@needs_gpu_certified
def test_gpu_certified_enclosure_and_plain_kernel_equality():
    g = np.random.default_rng(7)
    stack = g.standard_normal((6, 5, 5)) + 1j * g.standard_normal((6, 5, 5))
    vals, diags = gbskernels.perm_batched(stack, precision="certified",
                                          backend="gpu", return_diagnostics=True)
    for i in range(len(vals)):
        exact = complex(permanent_mp(stack[i], dps=DPS))
        assert abs(vals[i] - exact) <= diags[i]["abs_error_bound"]
        assert diags[i]["rel_error_bound"] < 1e-10 # tight on well-conditioned

    S = g.standard_normal((4, 6, 6))
    S = S + S.transpose(0, 2, 1)
    vals, diags = gbskernels.haf_batched(S, precision="certified",
                                         backend="gpu", return_diagnostics=True)
    for i in range(len(vals)):
        exact = complex(hafnian_mp(S[i], dps=DPS))
        assert abs(vals[i] - exact) <= diags[i]["abs_error_bound"]
    # the certificate covers the very value the plain GPU kernel returns
    assert np.array_equal(vals, gbskernels.haf_batched(S, backend="gpu"))


@needs_gpu_certified
def test_gpu_certified_single_and_rtol():
    g = np.random.default_rng(8)
    A = g.standard_normal((5, 5)) + 1j * g.standard_normal((5, 5))
    v, d = gbskernels.perm(A, precision="certified", backend="gpu",
                           return_diagnostics=True, rtol=1e-8)
    assert d["tier"] == "certified-fp64" and d["escalated"] is False
    assert abs(v - complex(permanent_mp(A, dps=DPS))) <= d["abs_error_bound"]


@needs_gpu_certified
def test_gpu_certified_lhaf_and_tor_enclose_mpmath():
    if not hasattr(gbskernels._load_gpu_ext(), "lhaf_certified"):
        pytest.skip("extension predates lhaf/tor certified kernels")
    from bench._inputs import physical_torontonian
    from highprec_ref import loop_hafnian_mp, torontonian_mp

    for N in (4, 6, 5): # odd N exercises the augmentation
        S = _sym(N, 70 + N)
        v, d = gbskernels.lhaf(S, precision="certified", backend="gpu",
                               return_diagnostics=True)
        assert abs(v - complex(loop_hafnian_mp(S, dps=DPS))) <= d["abs_error_bound"]
        assert d["rel_error_bound"] < 1e-9
    for nm in (2, 3):
        O = physical_torontonian(nm, 5)
        v, d = gbskernels.tor(O, precision="certified", backend="gpu",
                              return_diagnostics=True)
        assert abs(v - complex(torontonian_mp(O, dps=DPS))) <= d["abs_error_bound"]
        assert np.isfinite(d["abs_error_bound"])


# --------------------------------------------------------------------------
# property-based: the enclosure survives whatever Hypothesis throws (slow tier)
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_certified_enclosure_property():
    from hypothesis import given, settings
    from hypothesis import strategies as st
    from hypothesis.extra.numpy import arrays

    @settings(max_examples=40, deadline=None)
    @given(
        n=st.integers(min_value=1, max_value=5),
        data=st.data(),
    )
    def check(n, data):
        elems = st.floats(min_value=-10, max_value=10, allow_nan=False,
                          allow_infinity=False, width=64)
        re = data.draw(arrays(np.float64, (n, n), elements=elems))
        im = data.draw(arrays(np.float64, (n, n), elements=elems))
        A = re + 1j * im
        v, e = certified_permanent(A)
        exact = complex(permanent_mp(A, dps=DPS))
        assert abs(v - exact) <= e

    check()


@needs_gpu_certified
def test_gpu_certified_dd_ladder_proves_under_cancellation():
    """The PROVEN escalation ladder: on a cancelling input the fp64-certified
    bound refuses rtol, the DD-certified pass proves ~u-level accuracy, and the
    enclosure holds vs mpmath. Well-conditioned inputs never escalate."""
    if not hasattr(gbskernels._load_gpu_ext(), "haf_dd_certified"):
        pytest.skip("extension predates dd-certified kernels")
    A = cancellation_hafnian(1e-10, seed=0)
    v, d = gbskernels.haf(A, precision="certified", backend="gpu",
                          return_diagnostics=True, rtol=1e-12)
    assert d["tier"] == "certified-dd" and d["escalated"] is True
    assert d["rel_error_bound"] <= 1e-12
    assert abs(v - complex(hafnian_mp(A, dps=60))) <= d["abs_error_bound"]

    S = _sym(A.shape[0], 321)
    v, d = gbskernels.haf(S, precision="certified", backend="gpu",
                          return_diagnostics=True, rtol=1e-8)
    assert d["tier"] == "certified-fp64" and d["escalated"] is False

    # batched: the risky element escalates through the same ladder
    stack = np.stack([S, A]).astype(np.complex128)
    vals, diags = gbskernels.haf_batched(stack, precision="certified",
                                         backend="gpu", return_diagnostics=True,
                                         rtol=1e-12)
    assert diags[0]["tier"] in ("certified-fp64", "certified-dd")
    assert diags[1]["tier"] == "certified-dd"
