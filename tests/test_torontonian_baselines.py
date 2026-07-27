"""Focused tests for the matched torontonian implementation artifact."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pytest

from bench.torontonian_baselines import (
    Baseline,
    EvaluationResult,
    freeze_workload,
    run,
    xxpp_to_xpxp,
)


def _small_spd_batch(func, dim, batch, regime, seed):
    assert func == "tor"
    assert regime == "loss"
    rng = np.random.default_rng(seed)
    matrices = []
    for _ in range(batch):
        candidate = rng.standard_normal((dim, dim))
        candidate = 0.08 * (candidate + candidate.T) / max(dim, 1)
        matrices.append(candidate.astype(np.complex128))
    return np.stack(matrices)


def _sum_evaluator(matrix):
    return float(np.sum(matrix))


class _TickClock:
    def __init__(self, step=0.25):
        self.value = 0.0
        self.step = step

    def __call__(self):
        self.value += self.step
        return self.value


def _baselines(seen=None):
    seen = seen if seen is not None else {"alpha": [], "beta": []}

    def record(name, prepare):
        def wrapped(matrix):
            seen[name].append(hashlib.sha256(np.asarray(matrix, dtype="<f8").tobytes()).hexdigest())
            return prepare(matrix)

        return wrapped

    return (
        Baseline(
            name="alpha",
            package="alpha-package",
            package_version="1.2.3",
            implementation="alpha.tor",
            input_order="xxpp",
            evaluate=_sum_evaluator,
            prepare=record("alpha", lambda matrix: np.array(matrix, copy=True)),
        ),
        Baseline(
            name="beta",
            package="beta-package",
            package_version="4.5.6",
            implementation="beta.tor",
            input_order="xpxp",
            evaluate=_sum_evaluator,
            prepare=record("beta", xxpp_to_xpxp),
        ),
    )


def _bounded_baselines():
    alpha, beta = _baselines()
    return (
        Baseline(
            name=alpha.name,
            package=alpha.package,
            package_version=alpha.package_version,
            implementation=alpha.implementation,
            input_order=alpha.input_order,
            evaluate=lambda matrix: EvaluationResult(
                _sum_evaluator(matrix), reported_abs_error_bound=0.125
            ),
            prepare=alpha.prepare,
        ),
        beta,
    )


def test_xxpp_to_xpxp_is_simultaneous_pair_permutation():
    matrix = np.arange(36, dtype=np.float64).reshape(6, 6)
    indices = [0, 3, 1, 4, 2, 5]
    expected = matrix[np.ix_(indices, indices)]
    assert np.array_equal(xxpp_to_xpxp(matrix), expected)
    assert xxpp_to_xpxp(matrix).flags.c_contiguous


def test_freeze_workload_is_deterministic_self_contained_and_read_only():
    first = freeze_workload(
        [1, 2], 2, "loss", 17, matrix_generator=_small_spd_batch
    )
    second = freeze_workload(
        [1, 2], 2, "loss", 17, matrix_generator=_small_spd_batch
    )
    different = freeze_workload(
        [1, 2], 2, "loss", 18, matrix_generator=_small_spd_batch
    )

    assert [case.sha256 for case in first] == [case.sha256 for case in second]
    assert [case.sha256 for case in first] != [case.sha256 for case in different]
    for case in first:
        raw = base64.b64decode(case.data_base64, validate=True)
        assert hashlib.sha256(raw).hexdigest() == case.sha256
        restored = np.frombuffer(raw, dtype="<f8").reshape(case.matrix.shape)
        assert np.array_equal(restored, case.matrix)
        assert not case.matrix.flags.writeable
        assert case.min_eigenvalue_i_minus_o > 0.0


def test_run_writes_strict_separated_matched_artifact(tmp_path):
    seen = {"alpha": [], "beta": []}
    output = tmp_path / "artifact.json"
    artifact, path = run(
        modes=[1, 2],
        matrices_per_size=2,
        repeats=3,
        warmups=1,
        regime="loss",
        seed=23,
        baselines=_baselines(seen),
        out_path=output,
        clock=_TickClock(),
        provenance_factory=lambda: {
            "commit": "abc123",
            "container_digest": "sha256:container",
        },
        now_factory=lambda: datetime(2026, 7, 26, tzinfo=timezone.utc),
        matrix_generator=_small_spd_batch,
    )

    assert path == output
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed == artifact
    assert parsed["kind"] == "torontonian_matched_implementations"
    assert parsed["schema_version"] == 3
    assert {engine["package_version"] for engine in parsed["engines"]} == {
        "1.2.3",
        "4.5.6",
    }

    inputs = parsed["inputs"]
    assert inputs["canonical_order"] == "xxpp"
    assert len(inputs["matrices"]) == 4
    assert len(inputs["workload_sha256"]) == 64
    assert all(len(row["sha256"]) == 64 for row in inputs["matrices"])
    # Every preprocessor was handed the same canonical matrices, in the same order.
    canonical_hashes = [row["sha256"] for row in inputs["matrices"]]
    assert seen["alpha"] == canonical_hashes
    assert seen["beta"] == canonical_hashes

    performance = parsed["performance"]
    agreement = parsed["numerical_agreement"]
    assert len(performance["raw"]) == 2 * 2 * 3
    assert len(performance["summary"]) == 2 * 2
    assert len(agreement["rows"]) == 4
    assert "do not establish accuracy" in performance["interpretation"]
    assert "ground-truth reference" in agreement["interpretation"]
    assert all(
        row["seconds_per_matrix_median"] == pytest.approx(0.125)
        for row in performance["summary"]
    )
    assert all(row["seconds_per_matrix_iqr"] == 0.0 for row in performance["summary"])
    assert all(
        pair["within_tolerance"]
        for row in agreement["rows"]
        for pair in row["pairwise"]
    )
    assert all(
        row["reported_abs_error_bounds"] == {"alpha": None, "beta": None}
        for row in agreement["rows"]
    )
    assert "NaN" not in path.read_text(encoding="utf-8")
    assert "Infinity" not in path.read_text(encoding="utf-8")


def test_implementation_reported_bounds_are_retained_but_not_called_an_oracle(tmp_path):
    artifact, _ = run(
        modes=[1],
        matrices_per_size=1,
        repeats=1,
        warmups=0,
        regime="loss",
        baselines=_bounded_baselines(),
        out_path=tmp_path / "bounded.json",
        clock=_TickClock(),
        provenance_factory=dict,
        matrix_generator=_small_spd_batch,
    )

    agreement = artifact["numerical_agreement"]
    assert agreement["rows"][0]["reported_abs_error_bounds"] == {
        "alpha": 0.125,
        "beta": None,
    }
    assert "not an independent oracle" in agreement[
        "reported_error_bound_interpretation"
    ]


def test_invalid_implementation_reported_bound_is_rejected(tmp_path):
    baseline = Baseline(
        name="bad-bound",
        package="bad-bound-package",
        package_version="0",
        implementation="bad_bound.tor",
        input_order="xxpp",
        evaluate=lambda matrix: EvaluationResult(
            _sum_evaluator(matrix), reported_abs_error_bound=float("inf")
        ),
        prepare=lambda matrix: np.array(matrix, copy=True),
    )
    with pytest.raises(ValueError, match="invalid absolute error bound"):
        run(
            modes=[1],
            matrices_per_size=1,
            repeats=1,
            warmups=0,
            regime="loss",
            baselines=[baseline],
            out_path=tmp_path / "must-not-exist.json",
            clock=_TickClock(),
            provenance_factory=dict,
            matrix_generator=_small_spd_batch,
        )


def test_randomized_timing_order_is_reproducible(tmp_path):
    kwargs = dict(
        modes=[1, 2, 3],
        matrices_per_size=1,
        repeats=3,
        warmups=0,
        regime="loss",
        seed=99,
        baselines=_baselines(),
        clock=_TickClock(),
        provenance_factory=dict,
        now_factory=lambda: datetime(2026, 7, 26, tzinfo=timezone.utc),
        matrix_generator=_small_spd_batch,
    )
    first, _ = run(out_path=tmp_path / "first.json", **kwargs)
    kwargs["clock"] = _TickClock()
    second, _ = run(out_path=tmp_path / "second.json", **kwargs)
    keys = lambda artifact: [
        (row["engine"], row["modes"], row["repeat"])
        for row in artifact["performance"]["raw"]
    ]
    assert keys(first) == keys(second)
    assert keys(first) != sorted(keys(first))


def test_nonfinite_results_are_rejected_before_json_write(tmp_path):
    bad = Baseline(
        name="bad",
        package="bad-package",
        package_version="0",
        implementation="bad.tor",
        input_order="xxpp",
        evaluate=lambda matrix: float("nan"),
        prepare=lambda matrix: np.array(matrix, copy=True),
    )
    with pytest.raises(ValueError, match="non-finite"):
        run(
            modes=[1],
            matrices_per_size=1,
            repeats=1,
            warmups=0,
            regime="loss",
            baselines=[bad],
            out_path=tmp_path / "must-not-exist.json",
            clock=_TickClock(),
            provenance_factory=dict,
            matrix_generator=_small_spd_batch,
        )
    assert not (tmp_path / "must-not-exist.json").exists()


def test_input_mutation_invalidates_the_artifact(tmp_path):
    def mutate(matrix):
        value = float(np.sum(matrix))
        matrix[0, 0] += 1.0
        return value

    mutating = Baseline(
        name="mutating",
        package="mutating-package",
        package_version="0",
        implementation="mutating.tor",
        input_order="xxpp",
        evaluate=mutate,
        prepare=lambda matrix: np.array(matrix, copy=True),
    )
    with pytest.raises(RuntimeError, match="modified its input matrix"):
        run(
            modes=[1],
            matrices_per_size=1,
            repeats=1,
            warmups=0,
            regime="loss",
            baselines=[mutating],
            out_path=tmp_path / "must-not-exist.json",
            clock=_TickClock(),
            provenance_factory=dict,
            matrix_generator=_small_spd_batch,
        )
    assert not (tmp_path / "must-not-exist.json").exists()


def test_official_run_binds_build_provenance_and_is_create_only(tmp_path, monkeypatch):
    archive_hash = "a" * 64
    build_hash = "b" * 64
    monkeypatch.setenv("GBS_SOURCE_ARCHIVE_SHA256", archive_hash)
    monkeypatch.setenv("GBS_BUILD_MANIFEST_SHA256", build_hash)
    output = tmp_path / "official.json"
    kwargs = dict(
        modes=[1],
        matrices_per_size=1,
        repeats=1,
        warmups=0,
        regime="loss",
        baselines=[_baselines()[0]],
        out_path=output,
        clock=_TickClock(),
        provenance_factory=lambda: {
            "commit": "c" * 40,
            "container_digest": "image@sha256:" + "d" * 64,
        },
        matrix_generator=_small_spd_batch,
        require_provenance=True,
    )
    artifact, _ = run(**kwargs)
    assert artifact["source_archive_sha256"] == archive_hash
    assert artifact["build_manifest_sha256"] == build_hash

    kwargs["clock"] = _TickClock()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run(**kwargs)
