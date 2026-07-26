"""Tests for exact endpoint comparison in the Arb validation campaign."""

from __future__ import annotations

import json
from fractions import Fraction

import numpy as np
import pytest

from examples.jiuzhang.arb_enclosure_campaign import (
    Case,
    cpu_candidate,
    evaluate_case,
    generate_structured_cases,
    run_campaign,
)


def test_exact_candidate_interval_contains_dense_arb_reference():
    matrix = 0.25 * np.eye(2, dtype=np.float64)
    case = Case("one-mode", "closed_form", matrix)
    row = evaluate_case(case, lambda _: (float(Fraction(1, 3)), 1e-15), target_bits=90)

    assert row["status"] == "proved"
    assert row["arb_reference"]["method"] == "dense-subset-determinants"
    assert row["candidate"]["value_hex"] == float(Fraction(1, 3)).hex()


def test_rounded_center_without_radius_is_not_misreported_as_proved():
    matrix = 0.25 * np.eye(2, dtype=np.float64)
    case = Case("one-mode", "closed_form", matrix)
    row = evaluate_case(case, lambda _: (float(Fraction(1, 3)), 0.0), target_bits=90)

    assert row["status"] in {"violation", "inconclusive"}


def test_structured_reference_records_full_subset_count():
    case = generate_structured_cases([8])[0]
    exact = float(Fraction(1, 3) ** 8)
    row = evaluate_case(case, lambda _: (exact, 1e-18), target_bits=90)

    assert row["status"] == "proved"
    assert row["arb_reference"]["method"] == "factorized-mode-blocks"
    assert row["arb_reference"]["subset_count"] == 1 << 8


def test_structured_reference_must_match_the_exact_hashed_matrix():
    case = Case(
        "mismatched",
        "structured_mode_blocks",
        0.25 * np.eye(2, dtype=np.float64),
        blocks=np.array([0.5 * np.eye(2, dtype=np.float64)]),
    )

    with pytest.raises(ValueError, match="exact hashed xxpp matrix"):
        evaluate_case(case, lambda _: (1.0, 0.0), target_bits=80)


def test_candidate_cannot_mutate_the_hashed_reference_input():
    matrix = 0.25 * np.eye(2, dtype=np.float64)

    def mutating_candidate(candidate_matrix):
        candidate_matrix[:] = 0.5 * np.eye(2, dtype=np.float64)
        return 1.0, 0.0

    with pytest.raises(RuntimeError, match="mutated"):
        evaluate_case(Case("mutating", "closed_form", matrix), mutating_candidate)
    assert np.array_equal(matrix, 0.25 * np.eye(2, dtype=np.float64))


def test_campaign_writes_strict_artifact_and_hashed_corpus(tmp_path):
    cases = [Case("one", "closed_form", 0.25 * np.eye(2, dtype=np.float64))]
    output = tmp_path / "campaign.json"
    corpus = tmp_path / "corpus.npz"
    payload, status = run_campaign(
        cases,
        lambda _: (float(Fraction(1, 3)), 1e-15),
        backend="test",
        output=output,
        corpus=corpus,
        target_bits=80,
    )

    assert status == 0
    assert payload["summary"]["gate_pass"] is True
    assert len(payload["corpus"]["sha256"]) == 64
    assert payload["reference"]["python_flint_version"] != "unknown"
    assert payload["reference"]["flint_version"] != "unknown"
    assert len(payload["reference"]["oracle_source_sha256"]) == 64
    assert len(payload["reference"]["campaign_source_sha256"]) == 64
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert "NaN" not in output.read_text(encoding="utf-8")
    assert "Infinity" not in output.read_text(encoding="utf-8")
    with np.load(corpus, allow_pickle=False) as archive:
        assert np.array_equal(archive["matrix_0000"], cases[0].matrix)
        assert archive["matrix_0000"].dtype == np.dtype("<f8")


def test_refusal_is_recorded_and_cannot_pass_the_gate(tmp_path):
    output = tmp_path / "refused.json"
    corpus = tmp_path / "refused.npz"
    payload, status = run_campaign(
        [Case("refused", "closed_form", 0.25 * np.eye(2))],
        lambda _: (float("nan"), float("inf")),
        backend="test",
        output=output,
        corpus=corpus,
    )

    assert status == 1
    assert payload["summary"]["gate_pass"] is False
    assert payload["summary"]["refused"] == 1
    assert payload["rows"][0]["status"] == "refused"
    assert "NaN" not in output.read_text(encoding="utf-8")
    assert "Infinity" not in output.read_text(encoding="utf-8")


def test_cpu_infinite_bound_is_normalized_to_a_refusal():
    value, bound = cpu_candidate(np.diag([0.5, 1.5]))
    assert np.isnan(value)
    assert np.isinf(bound)


def test_required_provenance_rejects_malformed_claims(tmp_path, monkeypatch):
    monkeypatch.setenv("GBS_SOURCE_ARCHIVE_SHA256", "not-a-sha")
    monkeypatch.setenv("GBS_BUILD_MANIFEST_SHA256", "b" * 64)

    with pytest.raises(ValueError, match="GBS_SOURCE_ARCHIVE_SHA256"):
        run_campaign(
            [Case("one", "closed_form", 0.25 * np.eye(2))],
            lambda _: (1 / 3, 1e-15),
            backend="test",
            output=tmp_path / "invalid.json",
            corpus=tmp_path / "invalid.npz",
            require_provenance=True,
        )


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_official_backend_labels_cannot_wrap_an_arbitrary_candidate(
    tmp_path, backend
):
    with pytest.raises(ValueError, match=f"backend='{backend}'"):
        run_campaign(
            [Case("one", "closed_form", 0.25 * np.eye(2))],
            lambda _: (1 / 3, 1e-15),
            backend=backend,
            output=tmp_path / f"{backend}.json",
            corpus=tmp_path / f"{backend}.npz",
        )
