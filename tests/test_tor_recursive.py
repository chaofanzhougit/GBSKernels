"""Recursive prefix-Cholesky torontonian — CPU reference tests.

The recursive walk must equal the direct subset-determinant reference (itself
validated against The Walrus and mpmath) on the physical domain, with the
cancellation-normalized criterion where the value itself cancels (tor(O→0)=0,
so small random O is near-totally cancelling by construction). The GPU kernel
(core/tor_recursive.cu) mirrors this reference and is gated by
check_tor_recursive.cu; its device A/B decides dispatch.
"""

from __future__ import annotations

import numpy as np
import pytest

import cpu_ref
from bench._inputs import physical_torontonian
from cpu_ref.diagnostics import torontonian_abs_term_sum
from cpu_ref.tor_recursive import torontonian_recursive

pytestmark = pytest.mark.layer5


@pytest.mark.parametrize("n_modes", [1, 2, 3, 4, 5])
def test_recursive_equals_direct_on_physical(n_modes):
    for seed in range(4):
        O = physical_torontonian(n_modes, seed)
        a = torontonian_recursive(O)
        b = cpu_ref.tor(O)
        scale = torontonian_abs_term_sum(O)
        assert abs(a - b) <= 1e-13 * scale


def test_recursive_equals_direct_mildly_complex():
    g = np.random.default_rng(0)
    for n_modes in (2, 3):
        Z = 0.05 * (g.standard_normal((2 * n_modes,) * 2)
                    + 1j * g.standard_normal((2 * n_modes,) * 2))
        O = (Z + Z.T) / 2
        a = torontonian_recursive(O)
        b = cpu_ref.tor(O)
        assert abs(a - b) <= 1e-12 * torontonian_abs_term_sum(O)


def test_recursive_closed_form_and_empty():
    a, n = 0.2, 3
    v = torontonian_recursive(a * np.eye(2 * n))
    assert abs(v - (a / (1 - a)) ** n) < 1e-12
    assert torontonian_recursive(np.empty((0, 0))) == 1.0


def test_recursive_breakdown_raises_not_silent():
    # I - O has a zero leading pivot -> the pivot-free tree walk must raise
    O = np.eye(4) # I - O_S = 0 for any nonempty S
    with pytest.raises(FloatingPointError, match="breakdown"):
        torontonian_recursive(O)


def test_tor_single_split_matches_reference_and_guards():
    """SINGLE-LARGE split (2^g subtrees of one evaluation): equals the CPU
    recursive reference (mass-normalized), the closed form at n=14 through the
    same criterion, and the off-domain guard raises instead of silently
    returning garbage."""
    import gbskernels

    ext = gbskernels._load_gpu_ext()
    if ext is None or not hasattr(ext, "tor_single"):
        pytest.skip("extension predates tor_single")
    O = physical_torontonian(8, 0)
    v = gbskernels.tor_single(np.real(O), groups=5)
    ref = torontonian_recursive(O)
    assert abs(v - ref.real) <= 1e-12 * torontonian_abs_term_sum(O)

    a, n = 0.2, 14
    v = gbskernels.tor_single(a * np.eye(2 * n), groups=8)
    assert abs(v - (a / (1 - a)) ** n) <= 1e-12 * (1 + 1 / (1 - a)) ** n

    with pytest.raises(ValueError, match="physical"):
        gbskernels.tor_single(1.5 * np.eye(8))
    with pytest.raises(ValueError, match="cap"):
        gbskernels.tor_single(0.1 * np.eye(66))


def test_tor_single_certified_enclosure():
    """The certified single-large walk: enclosure vs mpmath on physical input,
    enclosure on the worst-cancelling closed form, and refusal (never a finite
    overclaim) off-domain."""
    import gbskernels
    from highprec_ref import torontonian_mp

    ext = gbskernels._load_gpu_ext()
    if ext is None or not hasattr(ext, "tor_single_certified"):
        pytest.skip("extension predates tor_single_certified")
    O = np.real(physical_torontonian(5, 2))
    v, d = gbskernels.tor_single(O, groups=3, certified=True)
    exact = complex(torontonian_mp(O, dps=50)).real
    assert abs(v - exact) <= d["abs_error_bound"]
    assert d["tier"] == "certified-fp64" and np.isfinite(d["rel_error_bound"])

    a, n = 0.2, 14 # kappa ~ 9^14: the certificate must
    v, d = gbskernels.tor_single(a * np.eye(2 * n), groups=8, certified=True)
    assert abs(v - (a / (1 - a)) ** n) <= d["abs_error_bound"]

    with pytest.raises(ValueError, match="physical|uncertifiable"):
        gbskernels.tor_single(1.5 * np.eye(8), certified=True)
