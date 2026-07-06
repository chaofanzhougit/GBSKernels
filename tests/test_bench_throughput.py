"""Layer 5 -- the throughput harness produces well-formed, honest artifacts.

Runs the harness on a tiny config (into a tmp dir, not the append-only results/)
and checks structure, the benchmark-honesty checksum, and -- when The Walrus is
present -- the cross-engine checksum agreement that doubles as a correctness check.
"""

from __future__ import annotations

import pytest

from bench.throughput import run

pytestmark = pytest.mark.layer5


def test_harness_artifact_is_well_formed(tmp_path):
    artifact, path = run("perm", sizes=[4, 6], batch_size=8, repeats=3, out_dir=tmp_path)
    assert path.exists() and path.parent == tmp_path
    assert artifact["kind"] == "throughput"
    assert artifact["tier"] == "cpu-baseline"  # no GPU here
    assert {"env", "params", "summary", "raw"} <= artifact.keys()
    # raw has one row per (engine, size, repeat)
    engines = {r["engine"] for r in artifact["raw"]}
    assert "gbskernels" in engines
    assert len(artifact["raw"]) == len(engines) * 2 * 3
    for r in artifact["raw"]:
        assert r["seconds"] > 0
        assert r["evals_per_sec"] > 0
        assert "|" in r["checksum"]  # honesty checksum present


def test_no_composite_winner_score(tmp_path):
    # Hygiene: the artifact must not contain a single composite "winner" number.
    artifact, _ = run("perm", sizes=[4], batch_size=8, repeats=3, out_dir=tmp_path)
    assert "winner" not in artifact
    assert "score" not in artifact


def test_cross_engine_checksums_agree_when_walrus_present(tmp_path):
    # Same inputs -> same answers -> identical checksums (free correctness check).
    thewalrus = pytest.importorskip("thewalrus")
    artifact, _ = run("haf", sizes=[4, 6], batch_size=8, repeats=2, out_dir=tmp_path)
    for row in artifact["summary"]:
        assert row["checksums_agree"], f"checksum mismatch at n={row['n']}"
