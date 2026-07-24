from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1] / "examples" / "jiuzhang"
sys.path.insert(0, str(HERE))

import dd_adversarial_enclosure as enclosure  # noqa: E402


def _stub_reference(monkeypatch):
    monkeypatch.setattr(enclosure, "torontonian_mp", lambda matrix, dps: 0.0)
    monkeypatch.setattr(enclosure, "torontonian_abs_term_sum", lambda matrix: 1.0)


def test_all_refusals_fail_the_empirical_gate(tmp_path, monkeypatch):
    _stub_reference(monkeypatch)
    monkeypatch.setattr(enclosure, "REPO", tmp_path)
    monkeypatch.setattr(enclosure, "_try_load_states", lambda: None)
    monkeypatch.setitem(
        enclosure.BACKENDS, "cpu", lambda matrix: (float("nan"), float("inf"))
    )

    status = enclosure.run("cpu", 2, 1, 20, 7)

    assert status == 1
    artifact = next((tmp_path / "results" / "jiuzhang").glob("*.json"))
    summary = json.loads(artifact.read_text())["summary"]
    assert summary["n_checked"] == 0
    assert summary["gate_pass"] is False


def test_required_physical_family_cannot_be_silently_skipped(tmp_path, monkeypatch):
    _stub_reference(monkeypatch)
    monkeypatch.setattr(enclosure, "REPO", tmp_path)
    monkeypatch.setattr(enclosure, "_try_load_states", lambda: None)
    monkeypatch.setitem(enclosure.BACKENDS, "cpu", lambda matrix: (0.0, 0.0))

    status = enclosure.run(
        "cpu", 2, 1, 20, 7, require_physical=True, max_refusal_fraction=0.5
    )

    assert status == 1
    artifact = next((tmp_path / "results" / "jiuzhang").glob("*.json"))
    summary = json.loads(artifact.read_text())["summary"]
    assert summary["physical_cases"] == 0
    assert summary["physical_skipped"] == 1
    assert any("physical family incomplete" in item for item in summary["gate_failures"])
