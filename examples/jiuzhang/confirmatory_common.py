"""Shared, dependency-light contracts for the Jiuzhang confirmatory v2 workflow.

All hashes in this module are SHA-256 over canonical bytes.  JSON objects use
sorted, compact JSON; NumPy arrays include their dtype and shape so two arrays
cannot collide merely because their raw buffers happen to match.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from confirmatory_contract import canonical_bytes, sha256_bytes, sha256_json


_CONTAINER_DIGEST = re.compile(r"\A[^\s@]+(?:/[^\s@]+)*@sha256:[0-9a-fA-F]{64}\Z")


def valid_container_digest(value: Any) -> bool:
    """Return whether *value* pins one named image by a SHA-256 digest."""
    return isinstance(value, str) and _CONTAINER_DIGEST.fullmatch(value) is not None


def canonical_json(value: Any) -> bytes:
    return canonical_bytes(value)


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk_size):
            h.update(block)
    return h.hexdigest()


def hash_json(value: Any) -> str:
    return sha256_json(value)


def placeholder_paths(value: object, prefix: str = "plan") -> list[str]:
    """Return every unresolved ``REPLACE-*`` field in a nested JSON value."""
    if isinstance(value, dict):
        return [path for key, item in value.items()
                for path in placeholder_paths(item, f"{prefix}.{key}")]
    if isinstance(value, list):
        return [path for index, item in enumerate(value)
                for path in placeholder_paths(item, f"{prefix}[{index}]")]
    if isinstance(value, str) and value.startswith("REPLACE-"):
        return [prefix]
    return []


def hash_array(value: np.ndarray) -> str:
    import hashlib

    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("ascii"))
    h.update(b"\0")
    h.update(canonical_json(list(a.shape)))
    h.update(b"\0")
    h.update(a.tobytes())
    return h.hexdigest()


def analysis_source_paths(root: str | Path | None = None) -> list[str]:
    repo = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    relative = [
        "examples/jiuzhang/confirmatory_contract.py",
        "examples/jiuzhang/confirmatory_common.py",
        "examples/jiuzhang/audit_selection_population.py",
        "examples/jiuzhang/build_exclusion_ledger.py",
        "examples/jiuzhang/confirmatory_design.py",
        "examples/jiuzhang/prepare_registration_v2.py",
        "examples/jiuzhang/registration_readiness_v2.py",
        "examples/jiuzhang/select_confirmatory_v2.py",
        "examples/jiuzhang/campaign_confirmatory_v2.py",
        "examples/jiuzhang/confirmatory_inference.py",
        "examples/jiuzhang/coherence_family.py",
        "examples/jiuzhang/joint_normalizer_replicates.py",
        "examples/jiuzhang/calibration_normalizer_replicates.py",
        "examples/jiuzhang/reconstruction_replicates.py",
        "examples/jiuzhang/absolute_predictive_checks.py",
        "examples/jiuzhang/analyze_refusals.py",
        "examples/jiuzhang/confirmatory_release.py",
        "examples/jiuzhang/q7_construction.py",
        "sampling/gbs.py",
    ]
    for directory, suffixes in (("core", {".cu", ".cuh", ".h", ".hpp", ".cpp"}),
                                ("bindings", {".cpp", ".h", ".hpp"}),
                                ("gbskernels", {".py"}),
                                ("sampling", {".py"}),
                                ("highprec_ref", {".py"}),
                                ("cpu_ref", {".py"})):
        base = repo / directory
        if base.exists():
            relative.extend(str(path.relative_to(repo)) for path in base.rglob("*")
                            if path.is_file() and path.suffix in suffixes
                            and "build" not in path.parts)
    return sorted(set(relative))


def analysis_source_hash(root: str | Path | None = None) -> str:
    """Hash the source files that define the v2 scientific calculation.

    Git's commit ID is insufficient on a checkout with uncommitted changes (or
    on the rsync-only GPU host), so the registration binds these exact bytes as
    well. Data, result, cache, and build trees are intentionally excluded.
    """
    import hashlib

    repo = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    relative = analysis_source_paths(repo)
    digest = hashlib.sha256()
    for name in relative:
        path = repo / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def analysis_sources_clean(root: str | Path | None = None) -> bool:
    """Return whether every analysis source is tracked and unmodified in Git."""
    repo = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", *analysis_source_paths(repo)],
            cwd=repo, capture_output=True, text=True, check=True, timeout=30)
    except Exception:
        return False
    return not status.stdout.strip()


def pattern_hash(pattern: np.ndarray) -> str:
    p = np.asarray(pattern, dtype=np.uint8)
    return sha256_bytes(np.packbits(p).tobytes())


def event_id(manifest_id: str, band: int, record_index: int,
             input_sha256: str) -> str:
    return hash_json({"manifest_id": manifest_id, "band": int(band),
                      "record_index": int(record_index),
                      "input_sha256": input_sha256})


def current_commit() -> str | None:
    value = os.environ.get("GBS_COMMIT", "").strip()
    if value:
        return value
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, timeout=10).stdout.strip()
    except Exception:
        return None


def load_json(path: str | Path) -> Any:
    with Path(path).open() as fh:
        return json.load(fh)


def write_json_exclusive(path: str | Path, value: Any) -> None:
    """Create a JSON object atomically and refuse to overwrite any prior record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=1, allow_nan=False) + "\n"
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def write_npz_exclusive(path: str | Path, **arrays: Any) -> None:
    """Create a compressed NPZ atomically and refuse an existing pathname."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as fh:
            os.fsync(fh.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
