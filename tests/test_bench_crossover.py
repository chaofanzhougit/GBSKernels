"""Batch-size sweep + crossover harness and the (matplotlib-optional) plot.

Guards the §9 batched-throughput figures: the sweep produces a well-formed,
provenance-stamped artifact with median+IQR+raw per batch and a crossover batch per
(function, size); an official (strict) run is fatal on an e2e disagreement / missing
Walrus baseline; the plot script renders curves (or, without matplotlib, falls back to a
reproducible CSV). Skips if the GPU extension (host-shim) is not importable.
"""

from __future__ import annotations

import csv

import pytest

import gbskernels


@pytest.fixture(scope="module")
def _has_ext():
    if not gbskernels.gpu_available():
        pytest.skip("GPU extension not importable (needs the host-shim build)")


def test_crossover_sweep_artifact(tmp_path, _has_ext):
    from bench.crossover import run

    # strict=False so the structure test does not depend on thewalrus being importable.
    artifact, path = run([8, 16], repeats=3, warmup=1, error_n=1, error_dps=20,
                         strict=False, out_dir=tmp_path)
    assert path.exists() and artifact["kind"] == "crossover_batch_sweep"
    assert {"commit", "container_digest", "captured_utc", "environment"} <= set(artifact)
    assert artifact["params"]["batches"] == [8, 16]
    assert artifact["params"]["regime"] == "physical" and artifact["params"]["strict"] is False
    assert "walrus" in artifact                       # same-instance baseline metadata block
    assert artifact["series"], "per (func, dim) crossover series present"
    for s in artifact["series"]:
        assert {"func", "matrix_dim", "points", "crossover_batch_vs_cpu",
                "crossover_batch_vs_walrus", "achieved_rel_err_fp64", "precision_tier",
                "walrus_median"} <= set(s)
        assert s["precision_tier"] == "fp64"
        assert s["achieved_rel_err_fp64"] is None or s["achieved_rel_err_fp64"] >= 0.0
        assert [p["batch"] for p in s["points"]] == [8, 16]  # swept, sorted
        for p in s["points"]:
            # median + IQR + RAW repetitions retained per batch (the dispersion contract)
            assert {"batch", "gpu_median", "gpu_iqr", "gpu_raw",
                    "cpu_median", "cpu_iqr", "cpu_raw"} <= set(p)
            assert isinstance(p["gpu_raw"], list) and len(p["gpu_raw"]) == 3   # raw repeats kept


def test_strict_crossover_fails_on_missing_walrus(tmp_path, _has_ext, monkeypatch):
    # An official (strict) crossover must abort if the same-instance Walrus baseline is
    # unavailable -- never publish a crossover without its baseline.
    import bench.crossover as cx

    def boom(*a, **k):
        raise RuntimeError("thewalrus not importable")
    monkeypatch.setattr(cx.walrus_baseline, "run", boom)
    with pytest.raises(RuntimeError, match="Walrus baseline is unavailable"):
        cx.run([8], repeats=2, warmup=1, error_n=0, strict=True, out_dir=tmp_path)


def test_plot_crossover_emits_csv(tmp_path, _has_ext):
    from bench.crossover import run
    from bench.plot_crossover import plot

    artifact, _ = run([8, 16], repeats=2, warmup=1, error_n=0, strict=False, out_dir=tmp_path)
    outs = plot(artifact, tmp_path / "fig")          # PNGs if matplotlib, always a CSV
    csv_path = next(o for o in outs if o.suffix == ".csv")
    with csv_path.open() as f:
        header = next(csv.reader(f))
    assert header[:3] == ["func", "matrix_dim", "batch"]
    assert {"gpu_median", "gpu_iqr", "cpu_median", "walrus_median",
            "crossover_batch_vs_cpu", "achieved_rel_err_fp64"} <= set(header)
