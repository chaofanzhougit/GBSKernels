#!/usr/bin/env python3
"""Rigorous Arb validation of the returned DD torontonian enclosures.

The small-matrix campaign reproduces the six families used by
``dd_adversarial_enclosure.py`` but replaces the rounded mpmath point with an
independent Arb interval.  The large-matrix campaign uses mode-independent
2x2 blocks: the production kernel still evaluates the full recursive
torontonian, while the reference uses the rigorously factorized identity.

Every matrix is retained in a compressed NumPy corpus and is identified by the
SHA-256 of its canonical little-endian binary64 bytes.  Certificate and Arb
endpoints are compared as exact rationals.  A case passes only when the whole
Arb interval is contained in the returned ``[value-bound, value+bound]``
interval.  An overlap is inconclusive, not a pass; disjoint intervals are a
violation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import flint

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bench._provenance import provenance
from highprec_ref.torontonian_arb import (
    TorontonianInterval,
    torontonian_arb,
    torontonian_block_diagonal_arb,
)

try:
    from .dd_adversarial_enclosure import (
        family_cancellation,
        family_near_refusal,
        family_physical,
        family_random_spd,
    )
except ImportError:  # direct script execution
    from dd_adversarial_enclosure import (  # type: ignore[no-redef]
        family_cancellation,
        family_near_refusal,
        family_physical,
        family_random_spd,
    )


SCHEMA = "gbskernels.arb-enclosure-campaign.v1"
DEFAULT_SEED = 20260714
Candidate = Callable[[np.ndarray], tuple[float, float]]
_LOWER_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class Case:
    matrix_id: str
    family: str
    matrix: np.ndarray
    blocks: np.ndarray | None = None

    @property
    def modes(self) -> int:
        return self.matrix.shape[0] // 2


def _canonical_matrix(matrix: Any) -> np.ndarray:
    value = np.asarray(matrix)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] % 2:
        raise ValueError(f"expected a square even-dimensional matrix, got {value.shape}")
    if np.iscomplexobj(value) and np.any(np.imag(value) != 0):
        raise ValueError("the rigorous real campaign does not accept complex matrices")
    # Always detach from caller-owned storage.  The candidate receives a
    # separate copy below, so neither it nor a later caller mutation can change
    # the bytes hashed for the reference evaluation.
    value = np.array(np.real(value), dtype=np.float64, order="C", copy=True)
    if not np.all(np.isfinite(value)):
        raise ValueError("campaign matrices must be finite")
    if not np.array_equal(value, value.T):
        raise ValueError("campaign matrices must be exactly symmetric binary64 arrays")
    return value


def _matrix_bytes(matrix: np.ndarray) -> bytes:
    return np.asarray(matrix, dtype="<f8", order="C").tobytes(order="C")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_environment(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if len(value) != 64 or any(character not in _LOWER_HEX for character in value):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    return value


def _official_provenance_errors(prov: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    commit = str(prov.get("commit") or "")
    if len(commit) not in (40, 64) or any(character not in _LOWER_HEX for character in commit):
        errors.append("commit is not a full lowercase Git object ID")
    container = str(prov.get("container_digest") or "")
    image, separator, digest = container.rpartition("@sha256:")
    if (
        not separator
        or not image
        or any(character.isspace() for character in container)
        or len(digest) != 64
        or any(character not in _LOWER_HEX for character in digest)
    ):
        errors.append("container_digest is not image@sha256:<64 lowercase hex>")
    git = prov.get("git")
    if isinstance(git, dict):
        if git.get("tracked_dirty") is True:
            errors.append("tracked source tree is dirty")
        git_commit = git.get("commit")
        if git_commit and git_commit != commit:
            errors.append("provenance commit fields disagree")
    return errors


def _fraction(value: Fraction) -> dict[str, str]:
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _candidate_interval(value: float, bound: float) -> tuple[Fraction, Fraction]:
    value = float(value)
    bound = float(bound)
    if not math.isfinite(value) or not math.isfinite(bound) or bound < 0:
        raise ValueError("a finite candidate requires finite value and nonnegative bound")
    center = Fraction.from_float(value)
    radius = Fraction.from_float(bound)
    return center - radius, center + radius


def _interval_relation(
    candidate_lower: Fraction,
    candidate_upper: Fraction,
    reference: TorontonianInterval,
) -> str:
    reference_lower = reference.lower.to_fraction()
    reference_upper = reference.upper.to_fraction()
    if candidate_lower <= reference_lower and reference_upper <= candidate_upper:
        return "proved"
    if candidate_upper < reference_lower or reference_upper < candidate_lower:
        return "violation"
    return "inconclusive"


def _center_error_bounds(value: float, reference: TorontonianInterval) -> tuple[Fraction, Fraction]:
    center = Fraction.from_float(float(value))
    lower = reference.lower.to_fraction()
    upper = reference.upper.to_fraction()
    distance = max(lower - center, center - upper, Fraction(0))
    maximum = max(abs(center - lower), abs(center - upper))
    return distance, maximum


def load_jiuzhang_states(data_dir: Path) -> dict[str, np.ndarray]:
    """Build the two public Jiuzhang 1.0 states from an explicit data directory."""
    try:
        from . import q7_construction as q7
    except ImportError:
        import q7_construction as q7  # type: ignore[no-redef]

    data = data_dir.expanduser().resolve()
    for name in ("T_full.npy", "squeezing parameters.txt"):
        if not (data / name).is_file():
            raise FileNotFoundError(f"Jiuzhang validation input is missing: {data / name}")
    previous = q7.DATA
    try:
        q7.DATA = data
        def threshold_matrix(kind: str) -> np.ndarray:
            cov = q7.build_cov(kind)
            sigma_q = (cov + np.eye(cov.shape[0], dtype=np.float64)) / 2.0
            sigma_q = _canonical_matrix((sigma_q + sigma_q.T) / 2.0)
            # Cholesky both validates positive definiteness and avoids the
            # unused slogdet diagnostic in threshold_O_xxpp.  NumPy 2.4's
            # Accelerate backend can emit spurious scaling warnings there for
            # this otherwise well-conditioned 200 x 200 matrix.
            np.linalg.cholesky(sigma_q)
            identity = np.eye(sigma_q.shape[0], dtype=np.float64)
            matrix = identity - np.linalg.solve(sigma_q, identity)
            return _canonical_matrix((matrix + matrix.T) / 2.0)

        return {
            "squeezed": threshold_matrix("squeezed"),
            "squashed": threshold_matrix("squashed"),
        }
    finally:
        q7.DATA = previous


def generate_small_cases(
    *,
    kmax: int = 14,
    per_cell: int = 4,
    seed: int = DEFAULT_SEED,
    states: dict[str, np.ndarray] | None,
) -> list[Case]:
    if kmax < 2 or per_cell <= 0:
        raise ValueError("kmax must be at least 2 and per_cell must be positive")
    rng = np.random.default_rng(seed)
    builders: list[tuple[str, Callable[[int], np.ndarray | None]]] = [
        ("random_spd", lambda k: family_random_spd(k, rng)),
        ("near_refusal_0.9", lambda k: family_near_refusal(k, rng, 0.9)),
        ("near_refusal_0.99", lambda k: family_near_refusal(k, rng, 0.99)),
        ("near_refusal_0.999", lambda k: family_near_refusal(k, rng, 0.999)),
        ("cancellation", lambda k: family_cancellation(k, rng)),
    ]
    if states is not None:
        builders.append(("physical", lambda k: family_physical(k, rng, states)))

    cases: list[Case] = []
    for modes in range(2, kmax + 1):
        for family, builder in builders:
            for index in range(per_cell):
                matrix = builder(modes)
                if matrix is None:
                    raise RuntimeError(f"builder unexpectedly skipped {family} at k={modes}")
                cases.append(Case(
                    matrix_id=f"small-{family}-k{modes:02d}-i{index:02d}",
                    family=family,
                    matrix=_canonical_matrix(matrix),
                ))
    return cases


def _xxpp_from_blocks(blocks: np.ndarray) -> np.ndarray:
    modes = len(blocks)
    matrix = np.zeros((2 * modes, 2 * modes), dtype=np.float64)
    for mode, block in enumerate(blocks):
        indices = (mode, mode + modes)
        matrix[np.ix_(indices, indices)] = block
    return matrix


def _validated_blocks_for_matrix(matrix: np.ndarray, supplied: Any) -> np.ndarray:
    """Bind a factorized reference to the exact matrix sent to the candidate."""
    raw = np.asarray(supplied)
    if np.iscomplexobj(raw) and np.any(np.imag(raw) != 0):
        raise ValueError("structured mode blocks must be real")
    blocks = np.array(np.real(raw), dtype=np.float64, order="C", copy=True)
    expected_shape = (matrix.shape[0] // 2, 2, 2)
    if blocks.shape != expected_shape or not np.all(np.isfinite(blocks)):
        raise ValueError(f"structured mode blocks must have shape {expected_shape} and be finite")
    reconstructed = _xxpp_from_blocks(blocks)
    if _matrix_bytes(reconstructed) != _matrix_bytes(matrix):
        raise ValueError(
            "structured mode blocks do not reconstruct the exact hashed xxpp matrix"
        )
    return blocks


def generate_structured_cases(modes: Iterable[int]) -> list[Case]:
    cases: list[Case] = []
    for count in modes:
        count = int(count)
        if count <= 0 or count > 32:
            raise ValueError("structured mode counts must lie in 1..32")
        # For each mode, det(I-B)=9/16 and the exact single-mode factor is 1/3.
        # The full torontonian is therefore (1/3)^count, while the production
        # evaluator still traverses the equivalent 2^count subset tree.
        blocks = np.repeat((0.25 * np.eye(2, dtype=np.float64))[None, :, :], count, axis=0)
        matrix = _canonical_matrix(_xxpp_from_blocks(blocks))
        cases.append(Case(
            matrix_id=f"structured-quarter-identity-k{count:02d}",
            family="structured_mode_blocks",
            matrix=matrix,
            blocks=blocks,
        ))
    return cases


def cpu_candidate(matrix: np.ndarray) -> tuple[float, float]:
    from cpu_ref.certified import certified_torontonian

    value, bound = certified_torontonian(matrix)
    scalar = complex(value)
    bound = float(bound)
    if (
        not math.isfinite(scalar.real)
        or not math.isfinite(scalar.imag)
        or not math.isfinite(bound)
        or bound < 0
    ):
        return float("nan"), float("inf")
    if scalar.imag != 0:
        raise ValueError("the real CPU certificate returned a complex value")
    return float(scalar.real), bound


def gpu_candidate(matrix: np.ndarray) -> tuple[float, float]:
    import gbskernels

    if gbskernels.gpu_backend_kind() != "gpu":
        raise RuntimeError("the official GPU campaign requires the compiled GPU backend")
    modes = matrix.shape[0] // 2
    try:
        value, diagnostics = gbskernels.tor_single(
            matrix, groups=min(modes, 13), dd=True
        )
    except ValueError:
        return float("nan"), float("inf")
    return float(value), float(diagnostics["abs_error_bound"])


def evaluate_case(
    case: Case,
    candidate: Candidate,
    *,
    target_bits: int = 80,
    max_precision_bits: int = 2048,
) -> dict[str, Any]:
    matrix = _canonical_matrix(case.matrix)
    modes = matrix.shape[0] // 2
    matrix_sha256 = hashlib.sha256(_matrix_bytes(matrix)).hexdigest()

    started = time.perf_counter()
    candidate_matrix = matrix.copy(order="C")
    value, bound = candidate(candidate_matrix)
    candidate_seconds = time.perf_counter() - started
    if (
        candidate_matrix.shape != matrix.shape
        or _matrix_bytes(candidate_matrix) != _matrix_bytes(matrix)
    ):
        raise RuntimeError("candidate mutated the matrix supplied by the campaign")
    refused = not (math.isfinite(value) and math.isfinite(bound) and bound >= 0)
    if refused:
        return {
            "matrix_id": case.matrix_id,
            "family": case.family,
            "modes": modes,
            "matrix_sha256": matrix_sha256,
            "candidate_seconds": candidate_seconds,
            "refused": True,
            "status": "refused",
        }

    started = time.perf_counter()
    if case.blocks is None:
        reference = torontonian_arb(
            matrix, target_bits=target_bits, max_precision_bits=max_precision_bits
        )
    else:
        blocks = _validated_blocks_for_matrix(matrix, case.blocks)
        reference = torontonian_block_diagonal_arb(
            blocks, target_bits=target_bits, max_precision_bits=max_precision_bits
        )
    reference_seconds = time.perf_counter() - started

    candidate_lower, candidate_upper = _candidate_interval(value, bound)
    relation = _interval_relation(candidate_lower, candidate_upper, reference)
    error_lower, error_upper = _center_error_bounds(value, reference)
    return {
        "matrix_id": case.matrix_id,
        "family": case.family,
        "modes": modes,
        "matrix_sha256": matrix_sha256,
        "candidate_seconds": candidate_seconds,
        "reference_seconds": reference_seconds,
        "refused": False,
        "status": relation,
        "candidate": {
            "value": value,
            "value_hex": value.hex(),
            "bound": bound,
            "bound_hex": bound.hex(),
            "lower": _fraction(candidate_lower),
            "upper": _fraction(candidate_upper),
        },
        "arb_reference": reference.to_dict(),
        "center_error_lower": _fraction(error_lower),
        "center_error_upper": _fraction(error_upper),
    }


def _strict_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite campaign artifact: {path}") from exc


def run_campaign(
    cases: Sequence[Case],
    candidate: Candidate,
    *,
    backend: str,
    output: Path,
    corpus: Path,
    target_bits: int = 80,
    max_precision_bits: int = 2048,
    require_provenance: bool = False,
) -> tuple[dict[str, Any], int]:
    if not cases:
        raise ValueError("the campaign requires at least one case")
    if backend == "gpu" and candidate is not gpu_candidate:
        raise ValueError("backend='gpu' requires the campaign's gpu_candidate wrapper")
    if backend == "cpu" and candidate is not cpu_candidate:
        raise ValueError("backend='cpu' requires the campaign's cpu_candidate wrapper")
    if output.exists() or corpus.exists():
        raise FileExistsError("campaign output and corpus paths must be new")

    prov = provenance()
    source_archive_sha256 = _sha256_environment("GBS_SOURCE_ARCHIVE_SHA256")
    build_manifest_sha256 = _sha256_environment("GBS_BUILD_MANIFEST_SHA256")
    if require_provenance:
        errors = _official_provenance_errors(prov)
        if source_archive_sha256 is None:
            errors.append("GBS_SOURCE_ARCHIVE_SHA256 is missing")
        if build_manifest_sha256 is None:
            errors.append("GBS_BUILD_MANIFEST_SHA256 is missing")
        if errors:
            raise RuntimeError(f"official campaign provenance is invalid: {errors}")

    arrays: dict[str, np.ndarray] = {}
    matrix_rows = []
    seen_ids: set[str] = set()
    for index, case in enumerate(cases):
        if case.matrix_id in seen_ids:
            raise ValueError(f"duplicate matrix id: {case.matrix_id}")
        seen_ids.add(case.matrix_id)
        matrix = _canonical_matrix(case.matrix)
        key = f"matrix_{index:04d}"
        arrays[key] = np.array(matrix, dtype="<f8", order="C", copy=True)
        matrix_rows.append({
            "matrix_id": case.matrix_id,
            "family": case.family,
            "modes": matrix.shape[0] // 2,
            "npz_key": key,
            "shape": list(matrix.shape),
            "dtype": "<f8",
            "sha256": hashlib.sha256(_matrix_bytes(matrix)).hexdigest(),
        })
    corpus.parent.mkdir(parents=True, exist_ok=True)
    with corpus.open("xb") as handle:
        np.savez_compressed(handle, **arrays)

    rows = [
        evaluate_case(
            case,
            candidate,
            target_bits=target_bits,
            max_precision_bits=max_precision_bits,
        )
        for case in cases
    ]
    for manifest_row, result_row in zip(matrix_rows, rows, strict=True):
        if manifest_row["sha256"] != result_row["matrix_sha256"]:
            raise RuntimeError(
                f"case {result_row['matrix_id']} changed after the corpus was frozen"
            )
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("proved", "inconclusive", "violation", "refused")
    }
    candidate_record: dict[str, Any] = {
        "callable_module": str(getattr(candidate, "__module__", "unknown")),
        "callable_qualname": str(getattr(candidate, "__qualname__", "unknown")),
    }
    if candidate is gpu_candidate:
        candidate_record.update({
            "precision_tier": "certified-dd",
            "groups": "min(modes, 13)",
            "backend_kind_required": "gpu",
        })
    elif candidate is cpu_candidate:
        candidate_record["precision_tier"] = "certified-fp64"

    payload = {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": backend,
        "candidate": candidate_record,
        "reference": {
            "library": "python-flint/Arb",
            "python_flint_version": str(getattr(flint, "__version__", "unknown")),
            "flint_version": str(getattr(flint, "__FLINT_VERSION__", "unknown")),
            "numpy_version": np.__version__,
            "method_small": "independent dense subset determinants",
            "method_structured": "independent factorized mode blocks",
            "exact_input_conversion": "binary64 as exact integer ratio",
            "endpoint_encoding": "exact dyadic mantissa times power of two",
            "oracle_source_sha256": _sha256_file(REPO / "highprec_ref" / "torontonian_arb.py"),
            "campaign_source_sha256": _sha256_file(Path(__file__).resolve()),
            "target_bits": target_bits,
            "max_precision_bits": max_precision_bits,
        },
        "provenance": prov,
        "source_archive_sha256": source_archive_sha256,
        "build_manifest_sha256": build_manifest_sha256,
        "corpus": {
            "path": corpus.name,
            "sha256": _sha256_file(corpus),
            "matrix_count": len(matrix_rows),
            "matrices": matrix_rows,
        },
        "summary": {
            "case_count": len(rows),
            **counts,
            "gate_pass": counts["proved"] == len(rows),
        },
        "rows": rows,
    }
    _strict_write(output, payload)
    return payload, 0 if payload["summary"]["gate_pass"] else 1


def _parse_modes(value: str) -> list[int]:
    if not value.strip():
        return []
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("mode counts must be comma-separated integers") from exc
    if len(result) != len(set(result)) or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("mode counts must be unique positive integers")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--small-kmax", type=int, default=14)
    parser.add_argument("--per-cell", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--structured-modes", type=_parse_modes, default=_parse_modes("25,26,27,28,29,30,31,32"))
    parser.add_argument("--jiuzhang-data", type=Path, default=Path("examples/jiuzhang/validation_data"))
    parser.add_argument("--skip-physical", action="store_true")
    parser.add_argument("--target-bits", type=int, default=80)
    parser.add_argument("--max-precision-bits", type=int, default=2048)
    parser.add_argument("--require-provenance", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()

    states = None if args.skip_physical else load_jiuzhang_states(args.jiuzhang_data)
    cases = generate_small_cases(
        kmax=args.small_kmax,
        per_cell=args.per_cell,
        seed=args.seed,
        states=states,
    )
    cases.extend(generate_structured_cases(args.structured_modes))
    if args.backend == "cpu" and args.structured_modes:
        raise ValueError("the exponential CPU candidate cannot run structured high-mode cases")
    payload, status = run_campaign(
        cases,
        gpu_candidate if args.backend == "gpu" else cpu_candidate,
        backend=args.backend,
        output=args.output,
        corpus=args.corpus,
        target_bits=args.target_bits,
        max_precision_bits=args.max_precision_bits,
        require_provenance=args.require_provenance,
    )
    summary = payload["summary"]
    print(
        "Arb enclosure campaign: "
        f"{summary['proved']}/{summary['case_count']} proved; "
        f"{summary['inconclusive']} inconclusive; "
        f"{summary['violation']} violations; {summary['refused']} refusals"
    )
    print(f"artifact: {args.output}")
    print(f"corpus: {args.corpus} ({payload['corpus']['sha256']})")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
