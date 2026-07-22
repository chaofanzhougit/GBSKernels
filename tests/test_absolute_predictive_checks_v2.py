from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "examples" / "jiuzhang"
sys.path.insert(0, str(HERE))

from absolute_predictive_checks import (compare, empirical_observables,
                                        model_observables, scan_data)  # noqa: E402
from select_confirmatory_v2 import DET_POSITIONS  # noqa: E402


def test_vacuum_observables_and_click_distribution_agree():
    patterns = np.zeros((8, 2), dtype=bool)
    empirical = empirical_observables(patterns, [(0, 1)])
    model = model_observables({"Q": np.eye(4)}, [(0, 1)])
    metrics = compare(empirical, model, np.asarray([1.0, 0.0, 0.0]))
    assert metrics["click_count_tv"] == pytest.approx(0.0)
    assert metrics["marginal_rms"] == pytest.approx(0.0)
    assert metrics["pair_covariance_rms"] == pytest.approx(0.0)


def test_predictive_observables_reject_bad_detector_pairs():
    with pytest.raises(ValueError, match="outside"):
        empirical_observables(np.zeros((2, 2), dtype=bool), [(0, 2)])


def test_scan_data_applies_registered_exclusions(tmp_path):
    def record(clicks):
        bits = np.zeros(128, dtype=np.uint8)
        pattern = np.zeros(100, dtype=np.uint8)
        pattern[list(clicks)] = 1
        bits[DET_POSITIONS[::-1]] = pattern
        return np.packbits(bits).tobytes()

    path = tmp_path / "data.bin"
    path.write_bytes(record((0,)) + record((1,)))
    empirical, _ = scan_data(path, chunk_records=1, pairs=[(0, 1)], exclusions=[0])
    assert empirical["n"] == 1
    assert empirical["click_marginals"][1] == pytest.approx(1.0)
