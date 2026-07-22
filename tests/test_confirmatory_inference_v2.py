from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "examples" / "jiuzhang"
sys.path.insert(0, str(HERE))

import confirmatory_inference as inference_module  # noqa: E402
from confirmatory_inference import (analyze, analyze_coherence_grid,
                                    _centered_srswor_bootstrap,
                                    _confidence_interval_from_errors,
                                    _normalizer_estimate_and_errors,
                                    horvitz_thompson_band,
                                    load_joint_normalizer_replicates,
                                    load_reconstruction_replicates,
                                    predictive_model_gate,
                                    registered_nonclassical_decision,
                                    _resample_scalar_cells,
                                    restore_recovered_primary_rows,
                                    validate_absolute_predictive_checks)  # noqa: E402


def test_main_rejects_different_valid_container_digest(tmp_path, monkeypatch):
    contract_body = {"container_digest": "image@sha256:" + "a" * 64}
    contract = {**contract_body, "run_id": inference_module.hash_json(contract_body)}
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps({
        "schema": "gbskernels.verified-run.v2", "complete": True,
        "run_id": contract["run_id"], "contract": contract, "registration": {},
    }))
    plan = {
        "analysis_commit": "deadbeef",
        "external_requirements": {"container_digest_required": True},
    }
    monkeypatch.setattr(
        inference_module, "validate_registration",
        lambda registration, require_beacon: {"plan": plan},
    )
    monkeypatch.setattr(inference_module, "current_commit", lambda: "deadbeef")
    monkeypatch.setenv("GBS_CONTAINER_DIGEST", "image@sha256:" + "b" * 64)
    monkeypatch.setattr(sys, "argv", [
        "confirmatory_inference.py", "--verified-run", str(run_path),
        "--normalizer-replicates", str(tmp_path / "unused.npz"),
        "--out", str(tmp_path / "unused.json"),
    ])
    with pytest.raises(SystemExit, match="analysis container differs"):
        inference_module.main()


def _rows():
    rows = []
    for s in range(4):
        for band, shift in ((27, 0.01), (28, 0.03)):
            eligible = 10 + 5 * s
            for j in range(3):
                rows.append({"event_id": f"event-{s}-{band}-{j}",
                             "band": band, "stratum": s,
                             "log_pattern_ratio_mid": shift + 0.001 * (s + j),
                             "log_pattern_ratio_halfwidth": 1e-4,
                             "inclusion_probability": 3 / eligible,
                             "eligible_in_stratum": eligible})
    return rows


def _reconstruction(rows, joint, normalizer, joint_arith=None):
    joint = np.asarray(joint, dtype=float)
    bands = sorted({int(row["band"]) for row in rows})
    event_scores = np.zeros((len(joint), len(rows), 2), dtype=float)
    event_arith = np.zeros_like(event_scores)
    if joint_arith is None:
        joint_arith = np.zeros_like(joint)
    joint_arith = np.asarray(joint_arith, dtype=float)
    for index, row in enumerate(rows):
        bi = bands.index(int(row["band"]))
        event_scores[:, index, 1] = joint[:, bi]
        event_arith[:, index, 1] = joint_arith[:, bi]
    return {
        "joint": joint, "normalizer": np.asarray(normalizer, dtype=float),
        "model_names": ["reference", "alternative"],
        "event_model_scores": event_scores, "event_model_arith": event_arith,
        "event_ids": np.asarray([row["event_id"] for row in rows]),
        "event_bands": np.asarray([row["band"] for row in rows]),
        "event_strata": np.asarray([row["stratum"] for row in rows]),
        "event_inclusion_probability": np.asarray(
            [row["inclusion_probability"] for row in rows]),
        "event_eligible_in_stratum": np.asarray(
            [row["eligible_in_stratum"] for row in rows]),
    }


def test_design_weights_and_minkowski_interval():
    rows = _rows()
    normalizers = np.array([[0.002, 0.004], [0.001, 0.003],
                            [0.003, 0.005], [0.002, 0.003]])
    out = analyze(rows, band_weights={27: 0.4, 28: 0.6},
                  normalizer_replicates=normalizers,
                  reconstruction_replicates=None,
                  bootstrap_reps=2000, bootstrap_seed=7)
    pm = out["point_model"]["conditional_pattern"]
    assert pm["confidence_interval_with_arithmetic"][0] \
        == pytest.approx(pm["confidence_interval_random"][0] - 1e-4)
    assert pm["confidence_interval_with_arithmetic"][1] \
        == pytest.approx(pm["confidence_interval_random"][1] + 1e-4)
    cov = np.asarray(out["normalizer_covariance"])
    assert cov.shape == (2, 2) and cov[0, 1] != 0
    assert out["reconstruction_marginal"] is None


def test_reconstruction_is_separate_layer():
    normalizers = np.array([[0.0, 0.0], [0.001, -0.001], [-0.001, 0.001]])
    rows = _rows()
    reconstruction = _reconstruction(
        rows, [[0.01, 0.02], [0.02, 0.03], [0.0, 0.01]],
        [[0.001, 0.002], [0.002, 0.003], [0.0, 0.001]])
    out = analyze(rows, band_weights={27: 0.5, 28: 0.5},
                  normalizer_replicates=normalizers,
                  reconstruction_replicates=reconstruction,
                  bootstrap_reps=500, bootstrap_seed=11)
    assert out["reconstruction_marginal"] is not None
    assert out["point_model"]["joint"]["estimate"] \
        != out["reconstruction_marginal"]["joint"]["estimate"]


def test_reconstruction_arithmetic_replaces_nominal_arithmetic():
    rows = _rows()
    reconstruction = _reconstruction(
        rows, np.zeros((3, 2)), np.zeros((3, 2)), np.full((3, 2), 0.002))
    out = analyze(
        rows, band_weights={27: 0.5, 28: 0.5},
        normalizer_replicates=np.zeros((3, 2)),
        reconstruction_replicates=reconstruction,
        bootstrap_reps=100, bootstrap_seed=9,
    )
    assert out["reconstruction_marginal"]["joint"][
        "arithmetic_worstcase_displacement"] == pytest.approx(0.002)


def test_reconstruction_normalizer_displacement_cancels_nominal_mc_error():
    rows = _rows()
    reconstruction = _reconstruction(
        rows, np.zeros((3, 2)), np.zeros((3, 2)))
    nominal = np.asarray([[0.4, -0.3], [-0.5, 0.2], [0.1, 0.1]])
    out = analyze(
        rows, band_weights={27: 0.5, 28: 0.5},
        normalizer_replicates=nominal,
        reconstruction_replicates=reconstruction,
        bootstrap_reps=100, bootstrap_seed=13,
    )
    click = out["reconstruction_marginal"]["click_count"]
    assert click["confidence_interval_random"] == pytest.approx(
        [click["estimate"], click["estimate"]])


def test_reconstruction_sampling_is_conditional_on_calibration_draw():
    rows = [
        {"event_id": f"event-{index}", "band": 27, "stratum": index // 2,
         "eligible_in_stratum": 10, "inclusion_probability": 0.2,
         "log_pattern_ratio_mid": 0.0, "log_pattern_ratio_halfwidth": 0.0}
        for index in range(4)
    ]
    event_scores = np.zeros((2, 4, 2), dtype=float)
    event_scores[0, :, 1] = [0.0, 2.0, 0.0, 2.0]
    event_scores[1, :, 1] = [1.0, 1.0, 1.0, 1.0]
    reconstruction = {
        "joint": np.ones((2, 1)), "normalizer": np.zeros((2, 1)),
        "model_names": ["reference", "alternative"],
        "event_model_scores": event_scores,
        "event_model_arith": np.zeros_like(event_scores),
        "event_ids": np.asarray([row["event_id"] for row in rows]),
        "event_bands": np.asarray([27] * 4),
        "event_strata": np.asarray([0, 0, 1, 1]),
        "event_inclusion_probability": np.asarray([0.2] * 4),
        "event_eligible_in_stratum": np.asarray([10] * 4),
    }
    out = analyze(
        rows, band_weights={27: 1.0}, normalizer_replicates=np.zeros((2, 1)),
        reconstruction_replicates=reconstruction,
        bootstrap_reps=2000, bootstrap_seed=37,
    )
    interval = out["reconstruction_marginal"]["joint"]["confidence_interval_random"]
    assert out["reconstruction_marginal"]["joint"]["estimate"] == pytest.approx(1.0)
    assert interval[0] < 1.0 < interval[1]
    assert "same calibration draw" in out["reconstruction_marginal"][
        "sampling_calibration_pairing"]


def test_joint_decomposition_is_explicit():
    normalizers = np.array([[0.02, 0.03], [0.01, 0.02], [0.03, 0.04]])
    out = analyze(_rows(), band_weights={27: 0.5, 28: 0.5},
                  normalizer_replicates=normalizers,
                  reconstruction_replicates=None,
                  bootstrap_reps=200, bootstrap_seed=5)
    joint = out["point_model"]["joint"]["estimate"]
    click = out["point_model"]["click_count"]["estimate"]
    conditional = out["point_model"]["conditional_pattern"]["estimate"]
    assert joint == pytest.approx(click + conditional)


def test_invalid_inclusion_probability_fails():
    with pytest.raises(ValueError, match="inclusion"):
        horvitz_thompson_band([{"log_pattern_ratio_mid": 1.0,
                                "log_pattern_ratio_halfwidth": 0.1,
                                "inclusion_probability": 0.0}])


def test_coherence_grid_preserves_paired_block_scores():
    rows = []
    for s in range(4):
        for band in (27, 28):
            eligible = 10 + 2 * s
            for j in range(2):
                common = 0.01 * s + 0.001 * j
                rows.append({"stratum": s, "band": band,
                             "inclusion_probability": 2 / eligible,
                             "eligible_in_stratum": eligible,
                             "model_log_probability_proxy": {
                                 "classical": {"mid": -0.2 + common,
                                               "halfwidth": 1e-5},
                                 "middle": {"mid": -0.1 + common,
                                            "halfwidth": 1e-5},
                                 "squeezed": {"mid": -0.3 + common,
                                              "halfwidth": 1e-5},
                             }})
    out = analyze_coherence_grid(
        rows, {"classical": 0.0, "middle": 0.5, "squeezed": 1.0},
        band_weights={27: 0.4, 28: 0.6}, bootstrap_reps=100,
        bootstrap_seed=3)
    assert out["estimate"] == 0.5
    assert out["estimate_model"] == "middle"
    assert out["statistical_confidence_set_models"] == ["middle"]
    assert out["statistical_confidence_set_coordinate_interval"] == [0.5, 0.5]
    assert out["descriptive_argmax_bootstrap"]["coordinate_quantile_interval"] == [0.5, 0.5]


def test_coherence_grid_uses_draw_specific_event_sampling_under_reconstruction():
    rows = []
    for stratum in range(2):
        for index in range(2):
            rows.append({
                "event_id": f"event-{stratum}-{index}", "stratum": stratum,
                "band": 27, "eligible_in_stratum": 10,
                "inclusion_probability": 0.2,
                "model_log_probability_proxy": {
                    name: {"mid": 0.0, "halfwidth": 0.0}
                    for name in ("classical", "middle", "alternative")
                },
            })
    event_scores = np.zeros((2, 4, 3), dtype=float)
    # Artifact order is deliberately different from coordinate order.
    event_scores[:, :, 1] = 0.5
    event_scores[0, :, 2] = [0.0, 2.0, 0.0, 2.0]
    event_scores[1, :, 2] = 1.0
    reconstruction = {
        "model_names": ["classical", "alternative", "middle"],
        "event_model_scores": event_scores,
        "event_model_arith": np.zeros_like(event_scores),
        "event_ids": np.asarray([row["event_id"] for row in rows]),
        "event_bands": np.asarray([27] * 4),
        "event_strata": np.asarray([0, 0, 1, 1]),
        "event_inclusion_probability": np.asarray([0.2] * 4),
        "event_eligible_in_stratum": np.asarray([10] * 4),
    }
    out = analyze_coherence_grid(
        rows, {"classical": 0.0, "middle": 0.5, "alternative": 1.0},
        band_weights={27: 1.0}, bootstrap_reps=2000, bootstrap_seed=41,
        reconstruction_replicates=reconstruction,
    )
    assert out["estimate_model"] == "middle"
    assert out["mean_log_scores"] == pytest.approx([0.0, 1.0, 0.5])
    assert np.max(out["statistical_confidence_set"][
        "pairwise_standard_errors"]) > 0


def test_fixed_strata_use_population_weights_but_are_never_resampled():
    rows = [
        {"band": 27, "stratum": 0, "eligible_in_stratum": 2,
         "inclusion_probability": 1.0, "log_pattern_ratio_mid": value,
         "log_pattern_ratio_halfwidth": 0.0}
        for value in (0.0, 0.0)
    ] + [
        {"band": 27, "stratum": 1, "eligible_in_stratum": 18,
         "inclusion_probability": 2 / 18, "log_pattern_ratio_mid": value,
         "log_pattern_ratio_halfwidth": 0.0}
        for value in (10.0, 10.0)
    ]
    out = analyze(rows, band_weights={27: 1.0},
                  normalizer_replicates=np.zeros((3, 1)),
                  reconstruction_replicates=None,
                  bootstrap_reps=2000, bootstrap_seed=19)
    joint = out["point_model"]["joint"]
    assert joint["estimate"] == pytest.approx(9.0)
    assert joint["confidence_interval_random"] == pytest.approx([9.0, 9.0])
    assert out["bootstrap"]["strata"] == "fixed; never resampled"


def test_srswor_fpc_gives_zero_sampling_variance_for_census_cells():
    rows = [
        {"band": 27, "stratum": stratum, "eligible_in_stratum": 2,
         "inclusion_probability": 1.0, "log_pattern_ratio_mid": value,
         "log_pattern_ratio_halfwidth": 0.0}
        for stratum, values in enumerate(((0.0, 10.0), (100.0, 200.0)))
        for value in values
    ]
    rng = np.random.default_rng(23)
    observed = _resample_scalar_cells(rows, [27], [0, 1], 100, rng)
    assert np.array_equal(observed, np.zeros((100, 2, 1)))


def test_srswor_rescaling_has_exact_estimated_design_variance():
    class EnumeratingRng:
        @staticmethod
        def integers(low, high, size):
            assert (low, high, size) == (0, 2, (4, 2))
            return np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])

    values = np.asarray([0.0, 2.0])
    errors = _centered_srswor_bootstrap(values, eligible=4, reps=4,
                                        rng=EnumeratingRng())
    sample_variance = np.var(values, ddof=1)
    expected = (1 - len(values) / 4) * sample_variance / len(values)
    assert np.mean(errors * errors) == pytest.approx(expected)


def test_confidence_interval_inverts_asymmetric_estimation_errors():
    errors = np.asarray([-3.0, -1.0, 0.0, 2.0, 8.0])
    out = _confidence_interval_from_errors(10.0, errors, alpha=0.4)
    lo_error, hi_error = np.quantile(errors, [0.2, 0.8])
    assert out["confidence_interval_random"] == pytest.approx(
        [10.0 - hi_error, 10.0 - lo_error])

    shifts = np.asarray([0.0, 1.0, 1.0, 2.0, 4.0])
    shifted = _confidence_interval_from_errors(
        10.0, errors, alpha=0.4, model_displacements=shifts)
    expected = np.quantile(-errors + shifts, [0.2, 0.8]) + 10.0
    assert shifted["confidence_interval_random"] == pytest.approx(expected)


def test_normalizer_uses_log_of_pooled_means_and_pooled_mean_variance():
    class EnumeratingRng:
        @staticmethod
        def integers(low, high, size):
            assert (low, high, size) == (0, 2, (4, 2))
            return np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])

    point, errors, count, method = _normalizer_estimate_and_errors(
        {"p_reference": np.asarray([[1.0], [9.0]]),
         "p_alternative": np.asarray([[1.0], [1.0]])},
        bootstrap_reps=4, rng=EnumeratingRng())
    assert point == pytest.approx([np.log(5.0)])
    assert point != pytest.approx([(np.log(1.0) + np.log(9.0)) / 2])
    assert count == 2 and "pooled" in method
    assert np.mean(errors[:, 0] ** 2) < np.mean(
        (np.log(np.asarray([1.0, 9.0])) - np.log(5.0)) ** 2)


def test_cell_requires_two_events_and_equal_primary_inclusion_probability():
    rows = _rows()
    rows[0]["inclusion_probability"] *= 0.5
    with pytest.raises(ValueError, match="equal within each cell"):
        analyze(rows, band_weights={27: 0.5, 28: 0.5},
                normalizer_replicates=np.zeros((2, 2)),
                reconstruction_replicates=None,
                bootstrap_reps=100, bootstrap_seed=1)

    seen = set()
    one_per_cell = []
    for row in _rows():
        key = (row["stratum"], row["band"])
        if key not in seen:
            seen.add(key)
            one_per_cell.append(dict(row))
    with pytest.raises(ValueError, match="at least two sampled events"):
        analyze(one_per_cell, band_weights={27: 0.5, 28: 0.5},
                normalizer_replicates=np.zeros((2, 2)),
                reconstruction_replicates=None,
                bootstrap_reps=100, bootstrap_seed=1)


def test_recovered_primary_replaces_operational_reserve():
    primary = {"event_id": "primary", "band": 27, "stratum": 0,
               "record_index": 1, "input_sha256": "a", "selected_primary": True,
               "inclusion_probability": 0.1, "eligible_in_stratum": 10,
               "refused": True}
    reserve = {"event_id": "reserve", "band": 27, "stratum": 0,
               "record_index": 2, "input_sha256": "b", "selected_primary": True,
               "inclusion_probability": 0.1, "eligible_in_stratum": 10,
               "replacement_for_refusal_event_id": "primary",
               "log_pattern_ratio_mid": 9.0, "log_pattern_ratio_halfwidth": 0.1}
    run = {"rows": [reserve], "refusals": [primary]}
    refusal = {"recovered_scores": [{"event_id": "primary",
                                      "band": 27, "stratum": 0,
                                      "record_index": 1, "input_sha256": "a",
                                      "selected_primary": True,
                                      "log_pattern_ratio_mid": 1.0,
                                      "log_pattern_ratio_halfwidth": 0.01}]}
    rows = restore_recovered_primary_rows(run, refusal)
    assert [row["event_id"] for row in rows] == ["primary"]
    assert rows[0]["log_pattern_ratio_mid"] == 1.0

    refusal["recovered_scores"][0].pop("log_pattern_ratio_halfwidth")
    with pytest.raises(ValueError, match="score enclosure"):
        restore_recovered_primary_rows(run, refusal)

    refusal["recovered_scores"][0]["log_pattern_ratio_halfwidth"] = 0.01
    refusal["recovered_scores"][0]["selected_primary"] = False
    with pytest.raises(ValueError, match="registered role"):
        restore_recovered_primary_rows(run, refusal)


def test_reconstruction_loader_enforces_metadata_and_endpoint_invariants(tmp_path):
    path = tmp_path / "reconstruction.npz"
    models = np.asarray([
        [[0.0], [1.0], [0.5]],
        [[0.1], [1.2], [0.6]],
    ])
    model_arith = np.full_like(models, 0.01)
    event_rows = [
        {"event_id": f"event-{index}", "band": 27, "stratum": index,
         "position": index, "record_index": index,
         "input_sha256": f"input-{index}", "inclusion_probability": 0.5,
         "eligible_in_stratum": 4}
        for index in range(2)
    ]
    identity = [{key: row[key] for key in (
        "event_id", "band", "stratum", "position", "record_index", "input_sha256",
        "inclusion_probability", "eligible_in_stratum")}
        for row in event_rows]
    meta = {
        "schema": "gbskernels.reconstruction-replicates.v3",
        "bands": [27], "draws": 2, "run_id": "run",
        "registration_id": "registration", "manifest_id": "manifest",
        "event_count": 2,
        "event_identity_sha256": inference_module.hash_json(identity),
        "loss_variation_rtol": 1e-10, "loss_variation_atol": 1e-12,
        "model_names": ["reference", "alternative", "middle"],
        "coherence_points": [0.0, 1.0, 0.5],
        "calibration_sha256": "a" * 64,
        "normalizer_draws_sha256": "b" * 64,
        "nominal_normalizers_sha256": "c" * 64,
    }
    event_models = np.repeat(models.transpose(0, 2, 1), 2, axis=1)
    event_arith = np.repeat(model_arith.transpose(0, 2, 1), 2, axis=1)
    arrays = {
        "joint_log_score_band_draws": models[:, 1, :] - models[:, 0, :],
        "normalizer_log_ratio_band_draws": np.zeros((2, 1)),
        "model_log_score_band_draws": models,
        "model_arith_halfwidth_band_draws": model_arith,
        "joint_arith_halfwidth_band_draws": model_arith[:, 0, :] + model_arith[:, 1, :],
        "event_model_log_score_draws": event_models,
        "event_model_arith_halfwidth_draws": event_arith,
        "event_ids": np.asarray([row["event_id"] for row in event_rows]),
        "event_bands": np.asarray([row["band"] for row in event_rows]),
        "event_strata": np.asarray([row["stratum"] for row in event_rows]),
        "event_positions": np.asarray([row["position"] for row in event_rows]),
        "event_record_indices": np.asarray(
            [row["record_index"] for row in event_rows]),
        "event_input_sha256": np.asarray(
            [row["input_sha256"] for row in event_rows]),
        "event_inclusion_probability": np.asarray(
            [row["inclusion_probability"] for row in event_rows]),
        "event_eligible_in_stratum": np.asarray(
            [row["eligible_in_stratum"] for row in event_rows]),
    }
    np.savez(path, meta=json.dumps(meta), **arrays)
    loaded = load_reconstruction_replicates(
        path, [27], run_id="run", registration_id="registration",
        manifest_id="manifest", expected_event_rows=event_rows, expected_draws=2,
        expected_model_names=meta["model_names"],
        expected_coherence_points=meta["coherence_points"],
        calibration_sha256="a" * 64, nominal_normalizers_sha256="c" * 64,
    )
    assert loaded["joint"].shape == (2, 1)

    refusal_meta = {**meta, "refusal_analysis_sha256": "d" * 64}
    np.savez(path, meta=json.dumps(refusal_meta), **arrays)
    with pytest.raises(ValueError, match="unexpectedly consumed"):
        load_reconstruction_replicates(path, [27])
    load_reconstruction_replicates(
        path, [27], refusal_analysis_sha256="d" * 64)

    arrays["joint_log_score_band_draws"] = arrays[
        "joint_log_score_band_draws"] + 0.1
    np.savez(path, meta=json.dumps(meta), **arrays)
    with pytest.raises(ValueError, match="contradict"):
        load_reconstruction_replicates(path, [27])


def test_normalizer_loader_binds_registered_effort_seed_and_probability_range(tmp_path):
    path = tmp_path / "normalizers.npz"
    meta = {
        "schema": "gbskernels.joint-normalizer-replicates.v1",
        "bands": [27], "registration_id": "registration",
        "samples_per_replicate": 1000, "seed": 17,
    }
    np.savez(path, meta=json.dumps(meta),
             p_reference=np.asarray([[0.2], [0.3]]),
             p_alternative=np.asarray([[0.1], [0.2]]))
    loaded = load_joint_normalizer_replicates(
        path, [27], registration_id="registration", expected_replicates=2,
        expected_samples_per_replicate=1000, expected_seed=17)
    assert loaded["p_reference"].shape == (2, 1)

    with pytest.raises(ValueError, match="seed"):
        load_joint_normalizer_replicates(path, [27], expected_seed=18)
    np.savez(path, meta=json.dumps(meta),
             p_reference=np.asarray([[1.2], [0.3]]),
             p_alternative=np.asarray([[0.1], [0.2]]))
    with pytest.raises(ValueError, match=r"\(0,1\]"):
        load_joint_normalizer_replicates(path, [27])


def test_simultaneous_confidence_set_retains_exact_ties():
    rows = []
    for stratum in range(2):
        for j in range(2):
            common = 0.1 * stratum + 0.01 * j
            rows.append({
                "stratum": stratum, "band": 27, "eligible_in_stratum": 4,
                "inclusion_probability": 0.5,
                "model_log_probability_proxy": {
                    "classical": {"mid": common, "halfwidth": 0.0},
                    "middle": {"mid": common, "halfwidth": 0.0},
                    "squeezed": {"mid": common - 1.0, "halfwidth": 0.0},
                },
            })
    out = analyze_coherence_grid(
        rows, {"classical": 0.0, "middle": 0.5, "squeezed": 1.0},
        band_weights={27: 1.0}, bootstrap_reps=200, bootstrap_seed=31)
    assert out["statistical_confidence_set_models"] == ["classical", "middle"]
    assert out["statistical_confidence_set_coordinate_interval"] == [0.0, 0.5]


def _predictive_check_fixture():
    thresholds = {"click_count_tv_max": 0.1,
                  "marginal_rms_max": 0.1,
                  "pair_covariance_rms_max": 0.1}
    model_cfg = {
        "reference_model": "classical", "alternative_model": "squeezed",
        "coherence_points": {"classical": 0.0, "middle": 0.5, "squeezed": 1.0},
    }
    checks = {
        "schema": "gbskernels.absolute-predictive-checks.v1",
        "run_id": "run", "registration_id": "registration",
        "analysis_commit": "commit", "analysis_source_sha256": "source",
        "container_digest": "image@sha256:" + "a" * 64,
        "data_sha256": "b" * 64, "exclusion_sha256": "c" * 64,
        "detector_pairs": [[0, 1]], "thresholds": dict(thresholds),
        "models": {
            "classical": {"click_count_tv": 0.02, "marginal_rms": 0.03,
                          "pair_covariance_rms": 0.04, "pass": True},
            "middle": {"click_count_tv": 0.2, "marginal_rms": 0.03,
                       "pair_covariance_rms": 0.04, "pass": False},
            "squeezed": {"click_count_tv": 0.05, "marginal_rms": 0.06,
                         "pair_covariance_rms": 0.07, "pass": True},
        },
        "any_model_passes": True,
    }
    validation = {
        "run_id": "run", "registration_id": "registration",
        "selection_cfg": {"source_raw_sha256": "b" * 64,
                          "exclusion_sha256": "c" * 64},
        "predictive_cfg": {"detector_pairs": [[0, 1]], "thresholds": thresholds},
        "model_names": ["classical", "squeezed", "middle"],
        "expected_provenance": {
            "analysis_commit": "commit", "analysis_source_sha256": "source",
            "container_digest": "image@sha256:" + "a" * 64,
        },
    }
    return checks, validation, model_cfg


def test_predictive_gate_applies_to_every_statistical_confidence_set_model():
    result = {"coherence_grid": {
        "estimate_model": "middle",
        "statistical_confidence_set_models": ["middle", "squeezed"],
    }}
    checks, validation, model_cfg = _predictive_check_fixture()
    model_passes = validate_absolute_predictive_checks(checks, **validation)
    gate = predictive_model_gate(
        result, model_passes, policy="all_plausible_best", model_cfg=model_cfg)
    assert gate["pass"] is False
    checks["models"]["middle"]["click_count_tv"] = 0.08
    checks["models"]["middle"]["pass"] = True
    model_passes = validate_absolute_predictive_checks(checks, **validation)
    assert predictive_model_gate(
        result, model_passes, policy="all_plausible_best", model_cfg=model_cfg)["pass"]


@pytest.mark.parametrize(("mutation", "message"), [
    ("missing_model", "model set"),
    ("extra_model", "model set"),
    ("detector_pairs", "detector pairs"),
    ("boolean_detector_pairs", "detector pairs"),
    ("thresholds", "thresholds"),
    ("exclusions", "exclusions"),
    ("acquisition", "acquisition"),
])
def test_predictive_artifact_binds_exact_registered_inputs(mutation, message):
    checks, validation, _ = _predictive_check_fixture()
    if mutation == "missing_model":
        del checks["models"]["middle"]
    elif mutation == "extra_model":
        checks["models"]["unregistered"] = dict(checks["models"]["classical"])
    elif mutation == "detector_pairs":
        checks["detector_pairs"] = [[0, 2]]
    elif mutation == "boolean_detector_pairs":
        checks["detector_pairs"] = [[False, True]]
    elif mutation == "thresholds":
        checks["thresholds"]["click_count_tv_max"] = 1.0
    elif mutation == "exclusions":
        checks["exclusion_sha256"] = "d" * 64
    else:
        checks["data_sha256"] = "d" * 64
    with pytest.raises(ValueError, match=message):
        validate_absolute_predictive_checks(checks, **validation)


def test_predictive_artifact_rejects_forged_pass_booleans_and_metrics():
    checks, validation, _ = _predictive_check_fixture()
    checks["models"]["middle"]["pass"] = True
    with pytest.raises(ValueError, match="pass flag disagrees"):
        validate_absolute_predictive_checks(checks, **validation)

    checks, validation, _ = _predictive_check_fixture()
    checks["models"]["classical"]["marginal_rms"] = float("nan")
    with pytest.raises(ValueError, match="invalid marginal_rms"):
        validate_absolute_predictive_checks(checks, **validation)

    checks, validation, _ = _predictive_check_fixture()
    checks["any_model_passes"] = False
    with pytest.raises(ValueError, match="aggregate pass flag"):
        validate_absolute_predictive_checks(checks, **validation)


def test_registered_decision_uses_confidence_set_and_predictive_gate_only():
    rule = {
        "method": "simultaneous_paired_model_score_max_t",
        "claim_if_confidence_set_above_classical_boundary": True,
        "require_predictive_pass_for_all_confidence_set_models": True,
        "report_failure_without_suppressing_analysis": True,
    }
    config = {"primary_decision_rule": rule, "minimum_relevant_coherence": 0.25}
    result = {
        "coherence_grid": {
            "confidence_set_excludes_classical_region": True,
            "statistical_confidence_set_models": ["quantum_025", "quantum_050"],
            "statistical_confidence_set_coordinate_interval": [0.25, 0.5],
            "descriptive_argmax_bootstrap": {
                "probability_at_or_below_classical_boundary": 0.4},
        },
        "predictive_model_gate": {"pass": True},
    }
    decision = registered_nonclassical_decision(result, config)
    assert decision["claim_supported"] is True
    result["predictive_model_gate"]["pass"] = False
    decision = registered_nonclassical_decision(result, config)
    assert decision["claim_supported"] is False
    assert decision["failure_reasons"]
    result["predictive_model_gate"]["pass"] = True
    result["coherence_grid"]["confidence_set_excludes_classical_region"] = False
    assert registered_nonclassical_decision(result, config)["claim_supported"] is False
