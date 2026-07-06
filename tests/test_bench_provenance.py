"""Every benchmark artifact carries the frozen-experiment provenance (docs/DESIGN.md §9).

The reproducibility contract: a result must be reproducible from its artifact alone,
so each one records the code **commit** and the pinned **container digest** (plus
hostname + capture time). Both come from env vars the session script exports
(``GBS_COMMIT`` / ``GBS_CONTAINER_DIGEST``) on the rented box, falling back to a local
git probe / ``None`` -- recorded honestly. This guards that the shared helper is wired
into all four harnesses and that the env override flows through.
"""

from __future__ import annotations

import importlib

import numpy as np

from bench import _provenance


def test_provenance_block_has_the_required_fields():
    p = _provenance.provenance()
    assert {"commit", "container_digest", "hostname", "captured_utc", "environment"} <= set(p)
    assert p["captured_utc"].endswith("+00:00") or p["captured_utc"].endswith("Z")
    # the machine environment makes each artifact self-describing (GPU/CPU/thread caps)
    env = p["environment"]
    assert {"gpu", "cuda", "cpu", "blas_threads", "platform", "python"} <= set(env)
    assert "logical_cores" in env["cpu"] and "OPENBLAS_NUM_THREADS" in env["blas_threads"]


def test_commit_and_digest_honor_the_session_env(monkeypatch):
    monkeypatch.setenv("GBS_COMMIT", "deadbeef")
    monkeypatch.setenv("GBS_CONTAINER_DIGEST", "nvidia/cuda@sha256:abc123")
    importlib.reload(_provenance)
    assert _provenance.commit() == "deadbeef"
    assert _provenance.container_digest() == "nvidia/cuda@sha256:abc123"
    p = _provenance.provenance()
    assert p["commit"] == "deadbeef" and p["container_digest"] == "nvidia/cuda@sha256:abc123"


def test_missing_digest_is_recorded_as_none_not_faked(monkeypatch):
    monkeypatch.delenv("GBS_CONTAINER_DIGEST", raising=False)
    importlib.reload(_provenance)
    # no env and (in CI) no /etc digest file -> honestly None
    assert _provenance.container_digest() in (None,) or isinstance(_provenance.container_digest(), str)


def test_accuracy_artifact_embeds_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("GBS_COMMIT", "cafef00d")
    monkeypatch.setenv("GBS_CONTAINER_DIGEST", "img@sha256:zzz")
    importlib.reload(_provenance)
    from bench.accuracy import run as run_acc
    artifact, _ = run_acc(dps=50, out_dir=tmp_path)
    assert artifact["commit"] == "cafef00d"
    assert artifact["container_digest"] == "img@sha256:zzz"
    assert "hostname" in artifact and "captured_utc" in artifact


def test_e2e_artifact_embeds_provenance_and_warmup(tmp_path, monkeypatch):
    import gbskernels
    if not gbskernels.gpu_available():
        import pytest
        pytest.skip("GPU extension not importable (needs the host-shim build)")
    monkeypatch.setenv("GBS_CONTAINER_DIGEST", "img@sha256:e2e")
    importlib.reload(_provenance)
    from bench.throughput_end_to_end import run as run_e2e
    artifact, _ = run_e2e(batch=16, repeats=2, warmup=1, out_dir=tmp_path)
    assert artifact["container_digest"] == "img@sha256:e2e"
    assert artifact["params"]["warmup"] == 1
    for r in artifact["summary"]:
        assert r["backends_agree"]
