"""Layer 5 -- numerical-accuracy characterization & engineering meta-tests.

The CPU-runnable subset of Layer 5. The GPU-specific invariants
(GPU-equals-CPU-reference within tier tolerance, DD paths, the benchmark-honesty
post-sync checksum) attach when the CUDA backend lands in a rented-GPU session;
they are marked ``skip`` here with that reason, so the contract is visible.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import complex_matrix, rel_error
from highprec_ref import permanent_mp

import gbskernels

pytestmark = pytest.mark.layer5


def test_batched_equals_looped_bit_identical():
    # A batch result must be bit-identical, per precision mode, to the same
    # inputs run singly (docs/DESIGN.md §8). On the CPU backend this is by construction;
    # the GPU backend must reproduce it.
    stack = np.stack([complex_matrix(5, seed=s) for s in range(16)])
    batched = gbskernels.perm_batched(stack, precision="fp64")
    looped = np.array([gbskernels.perm(A, precision="fp64") for A in stack])
    assert np.array_equal(batched, looped)  # exact, not approx


def test_determinism_fixed_input_reproducible_output():
    A = complex_matrix(7, seed=42)
    first = gbskernels.perm(A, precision="fp64")
    for _ in range(5):
        assert gbskernels.perm(A, precision="fp64") == first  # bit-identical


@pytest.mark.parametrize("n", range(2, 8))
@pytest.mark.parametrize("seed", range(5))
def test_fp64_accuracy_vs_reference_well_conditioned(n, seed):
    # For well-conditioned small matrices the FP64 tier should track the
    # arbitrary-precision reference to near machine precision. This is the seed
    # of the accuracy-vs-(size, conditioning) curves of docs/DESIGN.md §6/sec.9.
    A = complex_matrix(n, seed=314 * n + seed)
    approx = gbskernels.perm(A, precision="fp64")
    exact = complex(permanent_mp(A, dps=60))
    assert rel_error(approx, exact) < 1e-12


def test_reference_tier_routes_to_mpmath():
    A = complex_matrix(4, seed=5)
    assert gbskernels.perm(A, precision="ref") == pytest.approx(
        complex(permanent_mp(A, dps=60)), rel=1e-30
    )


def test_dd_tier_is_explicit_not_silent():
    # 'dd' is a GPU tier; the CPU backend must refuse it loudly, never pretend.
    with pytest.raises(NotImplementedError):
        gbskernels.perm(np.eye(3), precision="dd")


@pytest.mark.skip(reason="requires CUDA backend; runs in a rented-GPU session (docs/DESIGN.md §8/sec.10)")
def test_gpu_equals_cpu_reference():
    ...


@pytest.mark.skip(reason="requires CUDA backend; benchmark-honesty post-sync checksum (docs/DESIGN.md §8)")
def test_benchmark_honesty_checksum():
    ...
