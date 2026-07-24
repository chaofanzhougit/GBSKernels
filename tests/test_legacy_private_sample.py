from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1] / "examples" / "jiuzhang"
sys.path.insert(0, str(HERE))

import campaign_confirmatory as legacy_campaign  # noqa: E402
from campaign_confirmatory import aggregate, stratified_delta_H_band  # noqa: E402
from decode_events import (ABNORMAL_BIT, DET_POSITIONS, IGNORED, RECORD_BYTES,
                           audit_decoder_prefix)  # noqa: E402
from select_confirmatory import (_bind_historical_manifest, _parse_n,
                                 _select_equal_quota)  # noqa: E402


def test_historical_runner_loads_normalizers_without_private_modules(tmp_path, monkeypatch):
    click_probs = tmp_path / "click_probs"
    click_probs.mkdir()
    expected = {}
    for kind, stem in (("squeezed", "squeezed"), ("squashed", "squashed")):
        probabilities = np.linspace(0.0, 1.0, 31)
        uncertainties = np.full(31, 0.001 if kind == "squeezed" else 0.002)
        np.save(click_probs / f"click_probs_{stem}_0.npy", (probabilities, uncertainties))
        expected[kind] = (probabilities, uncertainties)

    monkeypatch.setattr(legacy_campaign.q7, "ZEN", tmp_path)
    loaded = legacy_campaign.their_normalizers()

    for kind in expected:
        assert np.array_equal(loaded[kind][0], expected[kind][0])
        assert np.array_equal(loaded[kind][1], expected[kind][1])


def test_historical_runner_has_no_private_module_dependency():
    source = Path(legacy_campaign.__file__).read_text()

    assert "from campaign import" not in source
    assert "from q7_conditioned import" not in source


def test_historical_selector_rejects_duplicate_band_targets():
    with pytest.raises(ValueError, match="duplicate"):
        _parse_n("27:800,27:900")


def test_historical_event_refuses_nonpositive_log_endpoint(monkeypatch):
    diagnostic = {"abs_error_bound": 1.0 - 1e-10}
    fake_module = types.SimpleNamespace(
        tor_single=lambda matrix, **kwargs: (1.0, diagnostic)
    )
    monkeypatch.setitem(sys.modules, "gbskernels", fake_module)
    states = {
        kind: {"O": np.zeros((200, 200)), "log_sqrt_detQ": 0.0}
        for kind in legacy_campaign.q7.KINDS
    }

    result = legacy_campaign.eval_event(states, np.array([], dtype=int), 27)

    assert result["refused"] is True
    assert result["reason"] == "nonpositive_log_endpoint"


def test_historical_selector_records_realized_equal_quota_design():
    ridx = np.array([0, 1, 2, 3, 100, 101, 102, 103], dtype=np.int64)
    pats = np.eye(8, 100, dtype=bool)

    first = _select_equal_quota(ridx, pats, target=4, n_strata=2, seed=17)
    second = _select_equal_quota(ridx, pats, target=4, n_strata=2, seed=17)

    assert np.array_equal(first["ridx"], second["ridx"])
    assert first["eligible_by_stratum"].tolist() == [4, 4]
    assert first["selected_by_stratum"].tolist() == [2, 2]
    assert np.array_equal(
        first["selected_by_stratum"],
        np.bincount(first["stratum"], minlength=2),
    )
    assert len(np.unique(first["keys"])) == 4


def test_regenerated_selection_is_cryptographically_bound_to_archived_npz(tmp_path):
    manifest = {
        27: {
            "ridx": np.array([10, 20], dtype=np.int64),
            "pats": np.eye(2, 100, dtype=bool),
        }
    }
    historical = tmp_path / "historical.npz"
    np.savez_compressed(
        historical,
        ridx_C27=manifest[27]["ridx"],
        pats_C27=manifest[27]["pats"],
    )

    binding = _bind_historical_manifest(historical, manifest, [27])
    assert binding["all_equal"] is True
    assert len(binding["sha256"]) == 64

    manifest[27]["ridx"] = np.array([10, 21], dtype=np.int64)
    with pytest.raises(ValueError, match="differs"):
        _bind_historical_manifest(historical, manifest, [27])


def test_decoder_mapping_audit_derives_dead_slots_and_order():
    n_normal = 101
    bits = np.zeros((n_normal + 1, RECORD_BYTES * 8), dtype=np.uint8)
    for detector_index, stored_position in enumerate(DET_POSITIONS[::-1]):
        bits[: detector_index + 1, stored_position] = 1
    bits[-1, DET_POSITIONS] = 1
    bits[-1, ABNORMAL_BIT] = 1
    raw = np.packbits(bits, axis=1).reshape(-1)
    expected = np.arange(1, 101, dtype=float) / n_normal

    audit = audit_decoder_prefix(raw, expected)

    assert audit["zero_rate_slots"] == IGNORED.tolist()
    assert audit["normal_records"] == n_normal
    assert audit["detector_rate_rms"] == 0.0
    assert audit["reversed_order_rms"] > 0.1


def test_stratified_estimator_uses_population_not_equal_stratum_weights():
    probabilities = np.ones(31)
    uncertainties = np.zeros(31)
    norms = {
        "squeezed": (probabilities, uncertainties),
        "squashed": (probabilities, uncertainties),
    }
    row = stratified_delta_H_band(
        np.array([0.0, 2.0, 10.0, 14.0]),
        np.array([0.1, 0.1, 0.2, 0.2]),
        np.array([0, 0, 1, 1]),
        np.array([2, 8]),
        27,
        norms,
    )

    assert np.isclose(row["delta_H"], 0.2 * 1.0 + 0.8 * 12.0)
    assert np.isclose(row["arithmetic_proxy"], 0.2 * 0.1 + 0.8 * 0.2)
    assert row["population_weights"] == [0.2, 0.8]
    assert row["sample_by_stratum"] == [2, 2]
    assert row["event_se"] > 0.0


def test_stratified_estimator_rejects_out_of_range_labels():
    probabilities = np.ones(31)
    norms = {
        "squeezed": (probabilities, np.zeros(31)),
        "squashed": (probabilities, np.zeros(31)),
    }
    with pytest.raises(ValueError, match="outside"):
        stratified_delta_H_band(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.zeros(4),
            np.array([0, 0, 2, 2]),
            np.array([10, 10]),
            27,
            norms,
        )


def test_aggregate_keeps_event_normalizer_and_arithmetic_scales_separate():
    rows = {
        band: {
            "delta_H": 0.01 * (band - 25),
            "event_se": 0.01,
            "normalizer_scale": 0.005,
            "arithmetic_proxy": 1e-5,
        }
        for band in (27, 28, 29, 30)
    }
    out = aggregate(rows)

    assert "gaussian_style_sensitivity_band" not in out
    assert "ci95_random" not in out
    assert "z_random" not in out
    assert "descriptive_standardized_ratio" not in out
    assert "normalizer_diagonal_sensitivity_scale" in out
    assert "arithmetic_proxy_interval" not in out
    assert "arithmetic_proxy_range" in out
    assert "post-first-4000" in out["estimand"]


def test_checkpoint_rows_are_bound_to_selection_record_indices(tmp_path, monkeypatch):
    monkeypatch.setattr(legacy_campaign, "OUT_DIR", tmp_path)
    path = tmp_path / "confirmatory_C27.jsonl"
    rows = [
        {"event": index, "ridx": ridx, "refused": False,
         "x_mid": float(index), "x_half": 1e-5}
        for index, ridx in enumerate((99, 11, 20, 21))
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    selection = {
        "ridx_C27": np.array([10, 11, 20, 21]),
        "stratum_C27": np.array([0, 0, 1, 1]),
        "eligible_C27": np.array([2, 8]),
    }
    probabilities = np.ones(31)
    norms = {
        "squeezed": (probabilities, np.zeros(31)),
        "squashed": (probabilities, np.zeros(31)),
    }

    with pytest.raises(ValueError, match="record index differs"):
        legacy_campaign.load_band_merged(27, norms, selection)

    rows[0]["ridx"] = 10
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = legacy_campaign.load_band_merged(27, norms, selection)
    assert result is not None
    assert result["selection_complete"] is True

    conflict = dict(rows[0])
    conflict["x_mid"] = 999.0
    path.write_text(
        "\n".join(json.dumps(row) for row in rows + [conflict]) + "\n"
    )
    with pytest.raises(ValueError, match="conflicting checkpoint"):
        legacy_campaign.load_band_merged(27, norms, selection)


def test_explicit_checkpoint_directory_is_loaded_and_recorded(tmp_path, monkeypatch):
    default_dir = tmp_path / "default"
    checkpoint_dir = tmp_path / "released" / "legacy_fixed_sample"
    default_dir.mkdir()
    checkpoint_dir.mkdir(parents=True)
    monkeypatch.setattr(legacy_campaign, "OUT_DIR", default_dir)

    checkpoint = checkpoint_dir / "confirmatory_C27.jsonl"
    rows = [
        {
            "event": index,
            "ridx": ridx,
            "refused": False,
            "x_mid": float(index),
            "x_half": 1e-5,
        }
        for index, ridx in enumerate((10, 11, 20, 21))
    ]
    checkpoint.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    selection = {
        "ridx_C27": np.array([10, 11, 20, 21]),
        "stratum_C27": np.array([0, 0, 1, 1]),
        "eligible_C27": np.array([2, 8]),
    }
    probabilities = np.ones(31)
    norms = {
        "squeezed": (probabilities, np.zeros(31)),
        "squashed": (probabilities, np.zeros(31)),
    }
    monkeypatch.setattr(legacy_campaign, "their_normalizers", lambda: norms)
    monkeypatch.setattr(legacy_campaign, "_sha256", lambda path: "0" * 64)

    manifest = tmp_path / "selection.npz"
    manifest.write_bytes(b"selection")
    output = tmp_path / "result.json"
    legacy_campaign.do_aggregate(
        manifest,
        {"targets": {"27": 4}},
        selection,
        output,
        checkpoint_dir=checkpoint_dir,
    )

    result = json.loads(output.read_text())
    assert result["bands"]["27"]["selection_complete"] is True
    assert result["inputs"]["checkpoints"] == [
        {"path": str(checkpoint.absolute()), "sha256": "0" * 64}
    ]
