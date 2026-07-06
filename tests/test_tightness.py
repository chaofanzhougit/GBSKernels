"""bench/tightness.py smoke: the enclosure invariant holds across ensembles and
the distribution stats are well-formed. Slow (mpmath references); tiny sample.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.layer5]


def test_tightness_enclosure_never_violated():
    from bench import tightness

    art = tightness.run(samples=5)
    # THE invariant: the certified bound never under-claims, on any ensemble.
    assert art["enclosure_violations_total"] == 0
    for func, regimes in art["by_function"].items():
        for regime, d in regimes.items():
            assert d["enclosure_fails"] == 0, (func, regime)
            if d["n"]:
                # tightness is >= 1 by construction (bound >= max(err, u|v|))
                assert d["tightness"]["min"] >= 1.0 - 1e-9
                # and the promised relative bound is finite and positive
                assert d["rel_bound"]["median"] > 0.0
