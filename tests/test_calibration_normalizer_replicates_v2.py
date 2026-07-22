from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "examples" / "jiuzhang"
sys.path.insert(0, str(HERE))

from calibration_normalizer_replicates import generate  # noqa: E402


def test_normalizers_retain_draw_stratum_and_model_axes():
    calibration = {
        "r25": np.asarray([[0.2]]),
        "T": np.zeros((1, 2, 2), dtype=np.complex128),
        "block_drift": np.zeros((1, 2, 2, 2), dtype=np.complex128),
    }
    calls = []

    def fake_grouped(phn, chn, transfer, samples, groups, seed):
        calls.append((len(phn), samples, groups, seed))
        return np.asarray([1.0, 0.5, 0.25]), None

    out = generate(
        calibration, {"reference": 0.0, "alternative": 0.5, "middle": 1.0},
        parameterization="classical_excess", bands=[1, 2],
        samples_per_draw=10, seed=100, grouped_evaluator=fake_grouped)
    assert out["p_models"].shape == (1, 2, 3, 2)
    assert out["p_reference"].shape == (1, 2, 2)
    assert out["calibration_draw_sha256"].shape == (1,)
    assert out["paired_normalizer_draw_sha256"].shape == (1,)
    # Common random numbers: each stratum uses one seed for every model.
    assert [row[3] for row in calls] == [100, 100, 100, 101, 101, 101]
