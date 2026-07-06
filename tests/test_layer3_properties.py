"""Layer 3 -- mathematical invariants / property-based (Hypothesis).

Properties of the permanent that hold across whole classes of inputs, so they
catch bug families that point tests miss. Hypothesis draws the size, a seed and
an entry scale; the matrix itself is built with numpy to keep entries
well-behaved (the point is the invariant, not float pathologies).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import complex_matrix, rel_error
from cpu_ref import perm

pytestmark = pytest.mark.layer3

_size = st.integers(min_value=1, max_value=6)
_seed = st.integers(min_value=0, max_value=2**31 - 1)
_scale = st.floats(min_value=0.2, max_value=3.0)


@settings(max_examples=120, deadline=None)
@given(n=_size, seed=_seed, scale=_scale)
def test_transpose_invariance(n, seed, scale):
    A = complex_matrix(n, seed=seed, scale=scale)
    assert rel_error(perm(A.T), perm(A)) < 1e-9


@settings(max_examples=120, deadline=None)
@given(n=_size, seed=_seed, scale=_scale)
def test_row_and_column_permutation_invariance(n, seed, scale):
    A = complex_matrix(n, seed=seed, scale=scale)
    g = np.random.default_rng(seed ^ 0xABCD)
    P = np.eye(n)[g.permutation(n)]
    Q = np.eye(n)[g.permutation(n)]
    assert rel_error(perm(P @ A @ Q), perm(A)) < 1e-9


@settings(max_examples=120, deadline=None)
@given(n=_size, seed=_seed, scale=_scale, c=st.complex_numbers(max_magnitude=3, min_magnitude=0.2))
def test_scaling(n, seed, scale, c):
    # perm(cA) = c^n perm(A)
    A = complex_matrix(n, seed=seed, scale=scale)
    assert rel_error(perm(c * A), (c**n) * perm(A)) < 1e-8


@settings(max_examples=120, deadline=None)
@given(n=_size, seed=_seed, scale=_scale)
def test_conjugation(n, seed, scale):
    # perm(conj(A)) = conj(perm(A))
    A = complex_matrix(n, seed=seed, scale=scale)
    assert rel_error(perm(np.conjugate(A)), np.conjugate(perm(A))) < 1e-9


@settings(max_examples=120, deadline=None)
@given(n=_size, seed=_seed, scale=_scale)
def test_real_input_gives_real_output(n, seed, scale):
    g = np.random.default_rng(seed)
    A = g.uniform(-scale, scale, size=(n, n))
    assert abs(perm(A).imag) < 1e-9 * (abs(perm(A)) + 1.0)


@settings(max_examples=80, deadline=None)
@given(n=_size, seed=_seed)
def test_single_row_or_col_scales_linearly(n, seed):
    # Multiplying one row by alpha multiplies the permanent by alpha.
    A = complex_matrix(n, seed=seed)
    alpha = 2.5
    B = A.copy()
    B[0, :] *= alpha
    assert rel_error(perm(B), alpha * perm(A)) < 1e-9
