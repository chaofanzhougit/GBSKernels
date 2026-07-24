from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_validation_data_preparer_stages_hash_bound_inputs(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "prepare_validation_data.py"),
         "--data-root", str(tmp_path)],
        check=True,
        cwd=ROOT,
    )
    expected = (
        "jiuzhang1/T_full.npy",
        "jiuzhang1/events_band13_32.npy",
        "jiuzhang1/squeezing parameters.txt",
        "jiuzhang1/empirical_click_rates.npy",
        "q7_1076_zenodo/click_probs/click_probs_squeezed_0.npy",
        "q7_1076_zenodo/click_probs/click_probs_squashed_0.npy",
    )
    assert all((tmp_path / path).is_file() for path in expected)
