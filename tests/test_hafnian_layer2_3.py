"""Layers 2 & 3 (hafnian) -- The Walrus oracle + property-based invariants."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import rel_error, symmetric_matrix
from cpu_ref import haf

# --- Layer 2: The Walrus differential oracle --------------------------------

thewalrus = pytest.importorskip("thewalrus", reason="Layer 2 needs The Walrus")
walrus_haf = thewalrus.hafnian


@pytest.mark.layer2
@pytest.mark.parametrize("N", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("seed", range(4))
def test_matches_thewalrus_complex_symmetric(N, seed):
    A = symmetric_matrix(N, seed=17 * N + seed)
    assert haf(A) == pytest.approx(complex(walrus_haf(A)), rel=1e-7, abs=1e-10)


@pytest.mark.layer2
@pytest.mark.parametrize("N", [2, 4, 6, 8])
@pytest.mark.parametrize("seed", range(3))
def test_matches_thewalrus_real_symmetric(N, seed):
    g = np.random.default_rng(seed)
    G = g.uniform(-1, 1, (N, N))
    A = G + G.T
    assert haf(A) == pytest.approx(complex(walrus_haf(A)), rel=1e-7, abs=1e-10)


# --- Layer 3: mathematical invariants ---------------------------------------

_pairs = st.integers(min_value=1, max_value=4)
_seed = st.integers(min_value=0, max_value=2**31 - 1)
_scale = st.floats(min_value=0.2, max_value=3.0)


@pytest.mark.layer3
@settings(max_examples=80, deadline=None)
@given(n_pairs=_pairs, seed=_seed, scale=_scale)
def test_scaling(n_pairs, seed, scale):
    # haf(cA) = c^n haf(A) for a 2n x 2n matrix (n = n_pairs)
    N = 2 * n_pairs
    A = symmetric_matrix(N, seed=seed, scale=scale)
    c = 1.7 - 0.4j
    assert rel_error(haf(c * A), (c**n_pairs) * haf(A)) < 1e-8


@pytest.mark.layer3
@settings(max_examples=80, deadline=None)
@given(n_pairs=_pairs, seed=_seed, scale=_scale)
def test_permutation_invariance(n_pairs, seed, scale):
    # Simultaneous row/col permutation leaves the hafnian unchanged.
    N = 2 * n_pairs
    A = symmetric_matrix(N, seed=seed, scale=scale)
    g = np.random.default_rng(seed ^ 0x5555)
    p = g.permutation(N)
    assert rel_error(haf(A[np.ix_(p, p)]), haf(A)) < 1e-8


@pytest.mark.layer3
@settings(max_examples=60, deadline=None)
@given(n_pairs=_pairs, seed=_seed, scale=_scale)
def test_diagonal_ignored_by_ordinary_hafnian(n_pairs, seed, scale):
    # The ordinary hafnian ignores the diagonal: overwriting it changes nothing.
    N = 2 * n_pairs
    A = symmetric_matrix(N, seed=seed, scale=scale)
    g = np.random.default_rng(seed ^ 0xAAAA)
    B = A.copy()
    d = g.uniform(-5, 5, N) + 1j * g.uniform(-5, 5, N)
    np.fill_diagonal(B, d)
    assert rel_error(haf(B), haf(A)) < 1e-8


@pytest.mark.layer3
@settings(max_examples=60, deadline=None)
@given(n_pairs=_pairs, seed=_seed)
def test_real_symmetric_gives_real_output(n_pairs, seed):
    N = 2 * n_pairs
    g = np.random.default_rng(seed)
    G = g.uniform(-1, 1, (N, N))
    A = G + G.T
    assert abs(haf(A).imag) < 1e-9 * (abs(haf(A)) + 1.0)
