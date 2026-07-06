"""Layer 2 -- The Walrus as differential oracle.

Exact agreement (to FP64 precision) with the canonical library across random
real/complex matrices, sizes and seeds. Necessary but, per Layer 1, not
sufficient -- so this layer never stands alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import complex_matrix, real_matrix
from cpu_ref import perm

pytestmark = pytest.mark.layer2

thewalrus = pytest.importorskip("thewalrus", reason="Layer 2 needs The Walrus")
walrus_perm = thewalrus.perm


@pytest.mark.parametrize("n", range(1, 11))
@pytest.mark.parametrize("seed", range(4))
def test_matches_thewalrus_real(n, seed):
    A = real_matrix(n, seed=7 * n + seed)
    assert perm(A) == pytest.approx(complex(walrus_perm(A)), rel=1e-8, abs=1e-11)


@pytest.mark.parametrize("n", range(1, 11))
@pytest.mark.parametrize("seed", range(4))
def test_matches_thewalrus_complex(n, seed):
    A = complex_matrix(n, seed=13 * n + seed)
    assert perm(A) == pytest.approx(complex(walrus_perm(A)), rel=1e-8, abs=1e-11)
