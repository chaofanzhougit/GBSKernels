"""Matched torontonian implementations on frozen real inputs.

This harness times the GBSKernels public DD path and the recursive
implementations in The Walrus and Piquasso on one frozen mathematical
workload.  The canonical workload is stored in xxpp
ordering as little-endian binary64 data and embedded in the JSON artifact, so a
reader can reconstruct every input without relying on NumPy's random-number
generator or on a future version of this repository.

Piquasso's native torontonian entry point consumes xpxp ordering.  Its adapter
therefore applies the corresponding row/column permutation before timing.  The
permutation is deterministic, its output is hashed, and preprocessing is
explicitly excluded from both engines' timings.  Numerical agreement is
reported separately from performance; neither baseline is treated as a
ground-truth accuracy oracle.

Example::

    python -m bench.torontonian_baselines --modes 4,8,12,16,20 \
        --matrices-per-size 3 --warmups 2 --repeats 7
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import random
import re
import statistics
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from bench import _inputs, _provenance


DEFAULT_OUT = Path(__file__).resolve().parent.parent / "results" / "throughput"
_MATRIX_ENCODING = "base64 little-endian float64 C order"
_WORKLOAD_DOMAIN = b"gbskernels:torontonian-baselines:workload:v1\0"
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")

Evaluator = Callable[[np.ndarray], Any]
Preprocessor = Callable[[np.ndarray], np.ndarray]
MatrixGenerator = Callable[[str, int, int, str, int], np.ndarray]


@dataclass(frozen=True)
class Baseline:
    """One implementation and its native input representation.

    ``evaluate`` is called only with already-prepared matrices.  Keeping
    ``prepare`` separate makes the timed region unambiguous and lets tests inject
    lightweight stand-ins without importing either optional dependency.
    """

    name: str
    package: str
    package_version: str
    implementation: str
    input_order: str
    evaluate: Evaluator
    prepare: Preprocessor
    execution_device: str = "cpu"
    precision_tier: str = "fp64"


@dataclass(frozen=True)
class FrozenMatrix:
    matrix_id: str
    modes: int
    matrix: np.ndarray
    sha256: str
    data_base64: str
    min_eigenvalue_i_minus_o: float


def _identity_xxpp(matrix: np.ndarray) -> np.ndarray:
    return np.array(matrix, dtype=np.float64, order="C", copy=True)


def xxpp_to_xpxp(matrix: np.ndarray) -> np.ndarray:
    """Return the simultaneous xxpp-to-xpxp row/column permutation."""

    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % 2:
        raise ValueError("xxpp_to_xpxp requires a square matrix of even dimension")
    modes = matrix.shape[0] // 2
    indices = np.empty(2 * modes, dtype=np.intp)
    indices[0::2] = np.arange(modes)
    indices[1::2] = np.arange(modes, 2 * modes)
    return np.ascontiguousarray(matrix[np.ix_(indices, indices)], dtype=np.float64)


def load_default_baselines(
    version_getter: Callable[[str], str] = version,
) -> tuple[Baseline, Baseline]:
    """Load the two optional, official recursive baseline entry points."""

    try:
        from thewalrus import rec_torontonian
    except Exception as exc:  # pragma: no cover - depends on benchmark host
        raise RuntimeError(
            "The Walrus is unavailable; install thewalrus on the benchmark host"
        ) from exc

    try:
        from piquasso._math.torontonian import torontonian as piquasso_torontonian
    except Exception as exc:  # pragma: no cover - depends on benchmark host
        raise RuntimeError(
            "Piquasso is unavailable; install piquasso on the benchmark host"
        ) from exc

    return (
        Baseline(
            name="walrus",
            package="thewalrus",
            package_version=version_getter("thewalrus"),
            implementation="thewalrus.rec_torontonian",
            input_order="xxpp",
            evaluate=rec_torontonian,
            prepare=_identity_xxpp,
        ),
        Baseline(
            name="piquasso",
            package="piquasso",
            package_version=version_getter("piquasso"),
            implementation="piquasso._math.torontonian.torontonian",
            input_order="xpxp",
            evaluate=piquasso_torontonian,
            prepare=xxpp_to_xpxp,
        ),
    )


def load_gbskernels_dd_candidate(
    version_getter: Callable[[str], str] = version,
    *,
    require_gpu: bool = True,
) -> Baseline:
    """Load the end-to-end public DD candidate used by the manuscript."""
    import gbskernels

    if require_gpu and gbskernels.gpu_backend_kind() != "gpu":
        raise RuntimeError("the matched publication benchmark requires the GPU backend")

    def evaluate(matrix: np.ndarray) -> float:
        modes = matrix.shape[0] // 2
        value, _ = gbskernels.tor_single(
            matrix, groups=min(modes, 13), dd=True
        )
        return float(value)

    return Baseline(
        name="gbskernels_dd",
        package="gbskernels",
        package_version=version_getter("gbskernels"),
        implementation="gbskernels.tor_single(dd=True)",
        input_order="xxpp",
        evaluate=evaluate,
        prepare=_identity_xxpp,
        execution_device="gpu" if gbskernels.gpu_backend_kind() == "gpu" else "cpu",
        precision_tier="double-word",
    )


def _matrix_bytes(matrix: np.ndarray) -> bytes:
    canonical = np.asarray(matrix, dtype="<f8", order="C")
    return canonical.tobytes(order="C")


def _matrix_sha256(matrix: np.ndarray) -> str:
    return hashlib.sha256(_matrix_bytes(matrix)).hexdigest()


def _workload_sha256(cases: Sequence[FrozenMatrix]) -> str:
    digest = hashlib.sha256(_WORKLOAD_DOMAIN)
    for case in cases:
        identifier = case.matrix_id.encode("ascii")
        raw = _matrix_bytes(case.matrix)
        digest.update(struct.pack(">I", len(identifier)))
        digest.update(identifier)
        digest.update(struct.pack(">IIQ", case.modes, 2 * case.modes, len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def _canonical_real_xxpp(matrix: np.ndarray, modes: int) -> tuple[np.ndarray, float]:
    source = np.asarray(matrix)
    expected = (2 * modes, 2 * modes)
    if source.shape != expected:
        raise ValueError(f"expected a {expected} torontonian matrix, got {source.shape}")
    if np.iscomplexobj(source) and np.any(np.imag(source) != 0.0):
        raise ValueError("the matched baseline workload must be real")

    real = np.asarray(np.real(source), dtype=np.float64)
    if not np.all(np.isfinite(real)):
        raise ValueError("the matched baseline workload must be finite")

    # Both recursive implementations use Cholesky factorization.  Symmetrize at
    # binary64 once, before freezing, so they consume exactly the same
    # mathematical matrix rather than different triangles of roundoff-asymmetric
    # state-construction output.
    real = np.ascontiguousarray(0.5 * (real + real.T), dtype=np.float64)
    min_eigenvalue = float(np.linalg.eigvalsh(np.eye(2 * modes) - real)[0])
    if not math.isfinite(min_eigenvalue) or min_eigenvalue <= 0.0:
        raise ValueError(
            "I - O must be positive definite for every frozen baseline matrix; "
            f"minimum eigenvalue is {min_eigenvalue!r}"
        )
    real.setflags(write=False)
    return real, min_eigenvalue


def freeze_workload(
    modes: Sequence[int],
    matrices_per_size: int,
    regime: str,
    seed: int,
    matrix_generator: MatrixGenerator = _inputs.bench_batch,
) -> tuple[FrozenMatrix, ...]:
    """Generate and freeze a self-contained, deterministic xxpp workload."""

    mode_list = [int(item) for item in modes]
    if not mode_list or any(item <= 0 for item in mode_list):
        raise ValueError("modes must contain positive integers")
    if len(set(mode_list)) != len(mode_list):
        raise ValueError("modes must not contain duplicates")
    if matrices_per_size <= 0:
        raise ValueError("matrices_per_size must be positive")
    if regime not in _inputs.BENCH_REGIMES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {_inputs.BENCH_REGIMES}")

    cases: list[FrozenMatrix] = []
    for mode_count in mode_list:
        cell_seed = seed + 1_000_003 * mode_count
        batch = matrix_generator(
            "tor", 2 * mode_count, matrices_per_size, regime, cell_seed
        )
        if len(batch) != matrices_per_size:
            raise ValueError(
                "matrix generator returned an unexpected batch length: "
                f"{len(batch)} != {matrices_per_size}"
            )
        for index, candidate in enumerate(batch):
            matrix, min_eigenvalue = _canonical_real_xxpp(candidate, mode_count)
            raw = _matrix_bytes(matrix)
            cases.append(
                FrozenMatrix(
                    matrix_id=f"{regime}-m{mode_count:02d}-i{index:03d}",
                    modes=mode_count,
                    matrix=matrix,
                    sha256=hashlib.sha256(raw).hexdigest(),
                    data_base64=base64.b64encode(raw).decode("ascii"),
                    min_eigenvalue_i_minus_o=min_eigenvalue,
                )
            )
    return tuple(cases)


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = int(position)
    fraction = position - lower
    if lower + 1 == len(ordered):
        return ordered[lower]
    return ordered[lower] * (1.0 - fraction) + ordered[lower + 1] * fraction


def _finite_complex(value: Any, *, context: str) -> complex:
    array = np.asarray(value)
    if array.size != 1:
        raise TypeError(f"{context} returned a non-scalar value with shape {array.shape}")
    scalar = complex(array.reshape(()).item())
    if not (math.isfinite(scalar.real) and math.isfinite(scalar.imag)):
        raise ValueError(f"{context} returned a non-finite value")
    return scalar


def _complex_record(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _prepared_workloads(
    baselines: Sequence[Baseline], cases: Sequence[FrozenMatrix]
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, str]]]:
    prepared: dict[str, dict[str, np.ndarray]] = {}
    hashes: dict[str, dict[str, str]] = {}
    for baseline in baselines:
        engine_matrices: dict[str, np.ndarray] = {}
        engine_hashes: dict[str, str] = {}
        for case in cases:
            native = np.ascontiguousarray(baseline.prepare(case.matrix), dtype=np.float64)
            if native.shape != case.matrix.shape or not np.all(np.isfinite(native)):
                raise ValueError(
                    f"{baseline.name} preprocessing produced an invalid matrix for "
                    f"{case.matrix_id}"
                )
            engine_matrices[case.matrix_id] = native
            engine_hashes[case.matrix_id] = _matrix_sha256(native)
        prepared[baseline.name] = engine_matrices
        hashes[baseline.name] = engine_hashes
    return prepared, hashes


def _validate_baselines(baselines: Sequence[Baseline]) -> None:
    if not baselines:
        raise ValueError("at least one baseline is required")
    names = [baseline.name for baseline in baselines]
    if len(names) != len(set(names)):
        raise ValueError("baseline names must be unique")
    for baseline in baselines:
        if not all(
            (
                baseline.name,
                baseline.package,
                baseline.package_version,
                baseline.implementation,
                baseline.input_order,
                baseline.execution_device,
                baseline.precision_tier,
            )
        ):
            raise ValueError("baseline metadata fields must be non-empty")


def _input_artifact(
    cases: Sequence[FrozenMatrix], native_hashes: Mapping[str, Mapping[str, str]]
) -> dict[str, Any]:
    return {
        "canonical_order": "xxpp",
        "dtype": "<f8",
        "memory_order": "C",
        "encoding": _MATRIX_ENCODING,
        "workload_sha256": _workload_sha256(cases),
        "matrices": [
            {
                "matrix_id": case.matrix_id,
                "modes": case.modes,
                "shape": [2 * case.modes, 2 * case.modes],
                "sha256": case.sha256,
                "data_base64": case.data_base64,
                "min_eigenvalue_i_minus_o": case.min_eigenvalue_i_minus_o,
                "native_sha256": {
                    name: hashes[case.matrix_id]
                    for name, hashes in native_hashes.items()
                },
            }
            for case in cases
        ],
    }


def run(
    *,
    modes: Sequence[int],
    matrices_per_size: int = 3,
    repeats: int = 7,
    warmups: int = 2,
    regime: str = "loss",
    seed: int = 20260726,
    baselines: Sequence[Baseline] | None = None,
    out_path: Path | None = None,
    agreement_atol: float = 1e-12,
    agreement_rtol: float = 1e-10,
    require_provenance: bool = False,
    clock: Callable[[], float] = time.perf_counter,
    provenance_factory: Callable[[], Mapping[str, Any]] = _provenance.provenance,
    now_factory: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    matrix_generator: MatrixGenerator = _inputs.bench_batch,
) -> tuple[dict[str, Any], Path]:
    """Run matched baselines and write one strict, self-contained JSON artifact."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    if agreement_atol < 0.0 or agreement_rtol < 0.0:
        raise ValueError("agreement tolerances must be non-negative")

    engines = tuple(baselines) if baselines is not None else load_default_baselines()
    _validate_baselines(engines)
    cases = freeze_workload(
        modes, matrices_per_size, regime, seed, matrix_generator=matrix_generator
    )
    prepared, native_hashes = _prepared_workloads(engines, cases)

    cases_by_mode = {
        mode: [case for case in cases if case.modes == mode] for mode in modes
    }
    engine_by_name = {baseline.name: baseline for baseline in engines}

    # Warm every implementation/size cell before any timed observation.  A
    # warmup traverses every distinct matrix in the cell, including JIT work.
    for baseline in engines:
        for mode_count in modes:
            for _ in range(warmups):
                for case in cases_by_mode[mode_count]:
                    _finite_complex(
                        baseline.evaluate(prepared[baseline.name][case.matrix_id]),
                        context=baseline.name,
                    )

    tasks = [
        (baseline.name, int(mode_count), repeat)
        for baseline in engines
        for mode_count in modes
        for repeat in range(repeats)
    ]
    order_seed = seed ^ 0x544F524F4E544F4E
    random.Random(order_seed).shuffle(tasks)

    raw_rows: list[dict[str, Any]] = []
    for engine_name, mode_count, repeat in tasks:
        baseline = engine_by_name[engine_name]
        cell_cases = cases_by_mode[mode_count]
        values: list[complex] = []
        start = clock()
        for case in cell_cases:
            values.append(
                _finite_complex(
                    baseline.evaluate(prepared[engine_name][case.matrix_id]),
                    context=engine_name,
                )
            )
        elapsed = float(clock() - start)
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise ValueError(f"clock produced invalid elapsed time {elapsed!r}")
        count = len(cell_cases)
        checksum = sum(values, 0.0j)
        raw_rows.append(
            {
                "engine": engine_name,
                "modes": mode_count,
                "matrix_dimension": 2 * mode_count,
                "repeat": repeat,
                "matrices": count,
                "seconds": elapsed,
                "seconds_per_matrix": elapsed / count,
                "matrices_per_second": count / elapsed,
                "checksum": _complex_record(checksum),
            }
        )

    summaries: list[dict[str, Any]] = []
    for baseline in engines:
        for mode_count in modes:
            selected = [
                row
                for row in raw_rows
                if row["engine"] == baseline.name and row["modes"] == mode_count
            ]
            latencies = [row["seconds_per_matrix"] for row in selected]
            throughputs = [row["matrices_per_second"] for row in selected]
            summaries.append(
                {
                    "engine": baseline.name,
                    "modes": int(mode_count),
                    "matrix_dimension": 2 * int(mode_count),
                    "repeats": repeats,
                    "matrices_per_repeat": matrices_per_size,
                    "seconds_per_matrix_median": statistics.median(latencies),
                    "seconds_per_matrix_iqr": _quantile(latencies, 0.75)
                    - _quantile(latencies, 0.25),
                    "matrices_per_second_median": statistics.median(throughputs),
                    "matrices_per_second_iqr": _quantile(throughputs, 0.75)
                    - _quantile(throughputs, 0.25),
                }
            )

    # This pass is deliberately outside every timed region.  It is an
    # implementation-agreement diagnostic, not an accuracy oracle or a filter on
    # the performance rows above.
    value_rows: list[dict[str, Any]] = []
    for case in cases:
        values = {
            baseline.name: _finite_complex(
                baseline.evaluate(prepared[baseline.name][case.matrix_id]),
                context=baseline.name,
            )
            for baseline in engines
        }
        comparisons = []
        for left_index, left in enumerate(engines):
            for right in engines[left_index + 1 :]:
                left_value = values[left.name]
                right_value = values[right.name]
                absolute = abs(left_value - right_value)
                scale = max(abs(left_value), abs(right_value), 1e-300)
                comparisons.append(
                    {
                        "left": left.name,
                        "right": right.name,
                        "absolute_difference": float(absolute),
                        "relative_difference": float(absolute / scale),
                        "within_tolerance": bool(
                            absolute
                            <= agreement_atol
                            + agreement_rtol
                            * max(abs(left_value), abs(right_value))
                        ),
                    }
                )
        value_rows.append(
            {
                "matrix_id": case.matrix_id,
                "modes": case.modes,
                "values": {
                    name: _complex_record(value) for name, value in values.items()
                },
                "pairwise": comparisons,
            }
        )

    for baseline in engines:
        for case in cases:
            observed = _matrix_sha256(prepared[baseline.name][case.matrix_id])
            expected = native_hashes[baseline.name][case.matrix_id]
            if observed != expected:
                raise RuntimeError(
                    f"{baseline.name} modified its input matrix {case.matrix_id}; "
                    "the matched timing artifact is invalid"
                )

    now = now_factory()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    provenance = dict(provenance_factory())
    source_archive_sha256 = os.environ.get("GBS_SOURCE_ARCHIVE_SHA256")
    build_manifest_sha256 = os.environ.get("GBS_BUILD_MANIFEST_SHA256")
    if require_provenance:
        missing = []
        if not provenance.get("commit"):
            missing.append("commit")
        if not provenance.get("container_digest"):
            missing.append("container_digest")
        if not source_archive_sha256:
            missing.append("GBS_SOURCE_ARCHIVE_SHA256")
        if not build_manifest_sha256:
            missing.append("GBS_BUILD_MANIFEST_SHA256")
        if missing:
            raise RuntimeError(f"official benchmark provenance is incomplete: {missing}")
        if not _SHA256.fullmatch(source_archive_sha256):
            raise RuntimeError("GBS_SOURCE_ARCHIVE_SHA256 must be 64 lowercase hex")
        if not _SHA256.fullmatch(build_manifest_sha256):
            raise RuntimeError("GBS_BUILD_MANIFEST_SHA256 must be 64 lowercase hex")

    artifact: dict[str, Any] = {
        "schema_version": 2,
        "kind": "torontonian_matched_implementations",
        "created_utc": now.astimezone(timezone.utc).isoformat(),
        **provenance,
        "source_archive_sha256": source_archive_sha256,
        "build_manifest_sha256": build_manifest_sha256,
        "parameters": {
            "modes": [int(item) for item in modes],
            "matrices_per_size": matrices_per_size,
            "repeats": repeats,
            "warmups": warmups,
            "regime": regime,
            "seed": seed,
            "randomized_timing_order_seed": order_seed,
            "agreement_atol": agreement_atol,
            "agreement_rtol": agreement_rtol,
        },
        "engines": [
            {
                "name": baseline.name,
                "package": baseline.package,
                "package_version": baseline.package_version,
                "implementation": baseline.implementation,
                "execution_device": baseline.execution_device,
                "precision_tier": baseline.precision_tier,
                "native_input_order": baseline.input_order,
                "preprocessing": (
                    "none beyond a private writable float64 C-order copy"
                    if baseline.input_order == "xxpp"
                    else "canonical xxpp-to-native simultaneous row/column permutation"
                ),
                "preprocessing_timed": False,
            }
            for baseline in engines
        ],
        "inputs": _input_artifact(cases, native_hashes),
        "performance": {
            "metric": (
                "synchronous wall-clock time on each recorded execution device; "
                "native input preprocessing excluded"
            ),
            "summary_statistic": "median and interpolated IQR over randomized raw repeats",
            "interpretation": (
                "Performance rows are retained regardless of numerical agreement. "
                "They do not establish accuracy."
            ),
            "summary": summaries,
            "raw": raw_rows,
        },
        "numerical_agreement": {
            "interpretation": (
                "Pairwise implementation agreement only; neither baseline is a "
                "ground-truth reference, and agreement does not alter performance rows."
            ),
            "rows": value_rows,
        },
    }

    destination = Path(out_path) if out_path is not None else (
        DEFAULT_OUT / f"torontonian_baselines_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False)
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized + "\n")
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite matched benchmark artifact: {destination}"
        ) from exc
    return artifact, destination


def _parse_modes(value: str) -> list[int]:
    try:
        modes = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("modes must be comma-separated integers") from exc
    if not modes or any(item <= 0 for item in modes) or len(set(modes)) != len(modes):
        raise argparse.ArgumentTypeError("modes must be unique positive integers")
    return modes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes", type=_parse_modes, default=_parse_modes("4,8,12,16,20")
    )
    parser.add_argument("--matrices-per-size", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--regime", choices=_inputs.BENCH_REGIMES, default="loss")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--agreement-atol", type=float, default=1e-12)
    parser.add_argument("--agreement-rtol", type=float, default=1e-10)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--include-gbskernels-dd",
        action="store_true",
        help="include the public DD call; requires the compiled GPU backend",
    )
    parser.add_argument(
        "--require-provenance",
        action="store_true",
        help="fail unless commit, container, source archive, and build manifest are bound",
    )
    args = parser.parse_args()

    engines: Sequence[Baseline] = load_default_baselines()
    if args.include_gbskernels_dd:
        engines = (load_gbskernels_dd_candidate(), *engines)

    artifact, path = run(
        modes=args.modes,
        matrices_per_size=args.matrices_per_size,
        repeats=args.repeats,
        warmups=args.warmups,
        regime=args.regime,
        seed=args.seed,
        baselines=engines,
        out_path=args.out,
        agreement_atol=args.agreement_atol,
        agreement_rtol=args.agreement_rtol,
        require_provenance=args.require_provenance,
    )
    print(f"# matched torontonian implementations -> {path}")
    print(f"# workload sha256: {artifact['inputs']['workload_sha256']}")
    for row in artifact["performance"]["summary"]:
        print(
            f"  {row['engine']:>9} modes={row['modes']:>2} "
            f"median={row['seconds_per_matrix_median']:.6g}s "
            f"IQR={row['seconds_per_matrix_iqr']:.3g}s"
        )


if __name__ == "__main__":
    main()
