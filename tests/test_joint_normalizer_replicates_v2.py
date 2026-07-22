from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "examples" / "jiuzhang"
sys.path.insert(0, str(HERE))

import joint_normalizer_replicates as normalizers  # noqa: E402


def test_coherence_inputs_cross_the_classical_boundary(monkeypatch):
    nbar = np.asarray([0.25, 1.0])
    r25 = np.arcsinh(np.sqrt(nbar))
    transfer = np.eye(4)
    monkeypatch.setattr(normalizers.q7, "load_config", lambda exp_id: (r25, transfer))

    phn0, chn0, _ = normalizers.coherence_inputs(
        0.0, exp_id=0, parameterization="classical_excess")
    phn1, chn1, _ = normalizers.coherence_inputs(
        1.0, exp_id=0, parameterization="classical_excess")
    assert np.allclose(phn0[0::2], nbar)
    assert np.allclose(chn0[0::2], nbar)
    assert np.allclose(chn1[0::2], np.sqrt(nbar * (nbar + 1.0)))
    assert np.allclose(chn0[1::2], -nbar)
