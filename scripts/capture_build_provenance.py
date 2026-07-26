#!/usr/bin/env python3
"""Capture a fail-closed source-to-binary provenance manifest.

The source tree is treated as immutable input: builds, wheels, compile metadata,
and the output manifest must live outside it.  A canonical per-entry inventory
binds the extracted tree independently of timestamps and host path names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "gbskernels.build-provenance.v1"
REQUIRED_TOOLS = frozenset({"cmake", "cuobjdump", "cxx", "nvcc", "python"})
HEX_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
GIT_OBJECT_ID = re.compile(r"\A(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
CONTAINER_DIGEST = re.compile(
    r"\A[^\s@]+(?:/[^\s@]+)*@sha256:[0-9a-f]{64}\Z"
)
ENV_NAME = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]*\Z")

DEFAULT_ENV_KEYS = (
    "CC",
    "CXX",
    "CUDAHOSTCXX",
    "CFLAGS",
    "CXXFLAGS",
    "CUDAFLAGS",
    "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS",
    "CMAKE_GENERATOR",
    "CMAKE_BUILD_TYPE",
    "CMAKE_BUILD_PARALLEL_LEVEL",
    "SOURCE_DATE_EPOCH",
    "PYTHONHASHSEED",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "GBS_COMMIT",
    "GBS_CONTAINER_DIGEST",
    "GBSKERNELS_EXT_DIR",
)

CMAKE_KEYS = (
    "CMAKE_BUILD_TYPE",
    "CMAKE_GENERATOR",
    "CMAKE_CXX_COMPILER",
    "CMAKE_CXX_COMPILER_VERSION",
    "CMAKE_CXX_FLAGS",
    "CMAKE_CXX_FLAGS_RELEASE",
    "CMAKE_CUDA_ARCHITECTURES",
    "CMAKE_CUDA_COMPILER",
    "CMAKE_CUDA_COMPILER_VERSION",
    "CMAKE_CUDA_FLAGS",
    "CMAKE_CUDA_FLAGS_RELEASE",
    "CMAKE_EXPORT_COMPILE_COMMANDS",
)


class ProvenanceError(ValueError):
    """Raised when provenance cannot be captured without ambiguity."""


def _canonical_bytes(value: Any) -> bytes:
    _assert_strict_json(value)
    return (json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n").encode("ascii")


def _assert_strict_json(value: Any, location: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProvenanceError(f"non-finite JSON number at {location}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_strict_json(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProvenanceError(f"non-string JSON key at {location}")
            _assert_strict_json(item, f"{location}.{key}")
        return
    raise ProvenanceError(f"unsupported JSON value at {location}: {type(value).__name__}")


def _strict_json_load(path: Path) -> Any:
    def reject_constant(token: str) -> None:
        raise ProvenanceError(f"non-standard JSON constant {token!r} in {path}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProvenanceError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"cannot parse strict JSON {path}: {exc}") from exc


def _regular_file(path: Path, label: str, *, nonempty: bool = True) -> Path:
    candidate = path.expanduser().resolve()
    if path.is_symlink() or not candidate.is_file():
        raise ProvenanceError(f"{label} is missing or not a regular file: {path}")
    if nonempty and candidate.stat().st_size <= 0:
        raise ProvenanceError(f"{label} is empty: {path}")
    return candidate


def sha256_file(path: Path) -> str:
    """Hash a regular file and fail if it changes while being read."""
    candidate = _regular_file(path, "file")
    before = candidate.stat()
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = candidate.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ProvenanceError(f"file changed while hashing: {candidate}")
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    candidate = _regular_file(path, "artifact")
    info = candidate.stat()
    return {
        "path": str(candidate),
        "filename": candidate.name,
        "bytes": info.st_size,
        "executable": bool(info.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)),
        "sha256": sha256_file(candidate),
    }


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def source_tree_inventory(source_tree: Path) -> dict[str, Any]:
    """Return a deterministic inventory of every file and symbolic link."""
    root = source_tree.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ProvenanceError(f"source tree is missing or not a directory: {source_tree}")
    if (root / ".git").exists():
        raise ProvenanceError("source tree must be an extracted archive without .git metadata")

    entries: list[dict[str, Any]] = []
    paths = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            target = os.readlink(path)
            entries.append({"path": relative, "type": "symlink", "target": target})
        elif path.is_dir():
            continue
        elif path.is_file():
            info = path.stat()
            entries.append({
                "path": relative,
                "type": "file",
                "bytes": info.st_size,
                "executable": bool(
                    info.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                ),
                "sha256": sha256_file(path),
            })
        else:
            raise ProvenanceError(f"unsupported source-tree entry: {path}")
    if not entries:
        raise ProvenanceError("source tree contains no files")
    digest = hashlib.sha256(_canonical_bytes(entries)).hexdigest()
    return {"entry_count": len(entries), "tree_sha256": digest, "entries": entries}


def _resolve_executable(value: str, name: str) -> Path:
    located = shutil.which(value) if os.sep not in value else value
    if not located:
        raise ProvenanceError(f"required tool {name!r} was not found: {value}")
    path = Path(located).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ProvenanceError(f"required tool {name!r} is not executable: {path}")
    return path


def tool_record(name: str, executable: str, version_args: Sequence[str]) -> dict[str, Any]:
    path = _resolve_executable(executable, name)
    command = [str(path), *version_args]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProvenanceError(f"cannot query required tool {name!r}: {exc}") from exc
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0 or not (stdout or stderr):
        raise ProvenanceError(
            f"required tool {name!r} version query failed with rc={completed.returncode}"
        )
    record = file_record(path)
    record.update({
        "argv": command,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
    })
    return record


def compile_commands_record(path: Path) -> dict[str, Any]:
    candidate = _regular_file(path, "compile_commands.json")
    payload = _strict_json_load(candidate)
    if not isinstance(payload, list) or not payload:
        raise ProvenanceError("compile_commands.json must be a non-empty array")
    commands: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ProvenanceError(f"compile command {index} is not an object")
        if not isinstance(item.get("directory"), str) or not isinstance(item.get("file"), str):
            raise ProvenanceError(f"compile command {index} lacks directory/file strings")
        has_command = isinstance(item.get("command"), str) and bool(item["command"].strip())
        has_arguments = (
            isinstance(item.get("arguments"), list)
            and bool(item["arguments"])
            and all(isinstance(value, str) for value in item["arguments"])
        )
        if has_command == has_arguments:
            raise ProvenanceError(
                f"compile command {index} must contain exactly one of command or arguments"
            )
        if has_command:
            argv = shlex.split(item["command"], posix=True)
            raw_command: dict[str, Any] = {"command": item["command"]}
        else:
            argv = list(item["arguments"])
            raw_command = {"arguments": list(item["arguments"])}
        if not argv:
            raise ProvenanceError(f"compile command {index} has an empty argv")
        commands.append({
            "directory": item["directory"],
            "file": item["file"],
            "output": item.get("output") if isinstance(item.get("output"), str) else None,
            "argv": argv,
            **raw_command,
        })
    record = file_record(candidate)
    record.update({"count": len(commands), "commands": commands})
    return record


def cmake_cache_record(path: Path) -> dict[str, Any]:
    candidate = _regular_file(path, "CMakeCache.txt")
    selected: dict[str, str | None] = {key: None for key in CMAKE_KEYS}
    try:
        for line in candidate.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith(("#", "//")) or "=" not in line:
                continue
            typed_key, value = line.split("=", 1)
            key = typed_key.split(":", 1)[0]
            if key in selected:
                selected[key] = value
    except (OSError, UnicodeError) as exc:
        raise ProvenanceError(f"cannot read CMake cache {candidate}: {exc}") from exc
    if selected["CMAKE_CUDA_ARCHITECTURES"] is None:
        raise ProvenanceError("CMake cache does not record CMAKE_CUDA_ARCHITECTURES")
    record = file_record(candidate)
    record.update({"selected": selected})
    return record


def _artifact_group(
    label: str,
    paths: Iterable[Path],
    source_tree: Path,
    suffixes: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    values = list(paths)
    if not values:
        raise ProvenanceError(f"at least one {label} is required")
    records = []
    seen: set[Path] = set()
    for value in values:
        path = _regular_file(value, label)
        if path in seen:
            raise ProvenanceError(f"duplicate {label}: {path}")
        if _inside(path, source_tree):
            raise ProvenanceError(f"{label} must be outside the immutable source tree: {path}")
        if suffixes and not any(path.name.endswith(suffix) for suffix in suffixes):
            raise ProvenanceError(f"unexpected {label} filename: {path.name}")
        seen.add(path)
        records.append(file_record(path))
    return sorted(records, key=lambda item: (item["filename"], item["path"]))


def _metadata_group(
    label: str,
    paths: Iterable[Path],
    source_tree: Path,
    loader: Any,
) -> list[dict[str, Any]]:
    values = list(paths)
    if not values:
        raise ProvenanceError(f"at least one {label} is required")
    records = []
    seen: set[Path] = set()
    for value in values:
        path = _regular_file(value, label)
        if path in seen:
            raise ProvenanceError(f"duplicate {label}: {path}")
        if _inside(path, source_tree):
            raise ProvenanceError(f"{label} must be outside the immutable source tree")
        seen.add(path)
        records.append(loader(path))
    return sorted(records, key=lambda item: item["path"])


def selected_environment(keys: Iterable[str]) -> dict[str, str | None]:
    unique = sorted(set(keys))
    for key in unique:
        if not ENV_NAME.fullmatch(key):
            raise ProvenanceError(f"invalid environment-variable name: {key!r}")
    return {key: os.environ.get(key) for key in unique}


def capture_build_provenance(
    *,
    source_archive: Path,
    expected_source_archive_sha256: str,
    source_tree: Path,
    git_commit: str,
    git_tree: str,
    container_digest: str,
    compile_commands: Iterable[Path],
    cmake_caches: Iterable[Path],
    build_products: Iterable[Path],
    device_code: Iterable[Path],
    extensions: Iterable[Path],
    wheels: Iterable[Path],
    tools: Mapping[str, tuple[str, Sequence[str]]],
    env_keys: Iterable[str] = DEFAULT_ENV_KEYS,
    expected_source_tree_sha256: str | None = None,
) -> dict[str, Any]:
    """Capture and return one strict source-to-binary provenance record."""
    archive = _regular_file(source_archive, "source archive")
    root = source_tree.expanduser().resolve()
    if _inside(archive, root):
        raise ProvenanceError("source archive must be outside the extracted source tree")
    if not HEX_SHA256.fullmatch(expected_source_archive_sha256):
        raise ProvenanceError("expected source-archive SHA-256 must be 64 lowercase hex")
    if not GIT_OBJECT_ID.fullmatch(git_commit):
        raise ProvenanceError("git commit must be a 40- or 64-character lowercase object ID")
    if not GIT_OBJECT_ID.fullmatch(git_tree):
        raise ProvenanceError("git tree must be a 40- or 64-character lowercase object ID")
    if not CONTAINER_DIGEST.fullmatch(container_digest):
        raise ProvenanceError("container digest must be image@sha256:<64 lowercase hex>")
    env_commit = os.environ.get("GBS_COMMIT")
    if env_commit and env_commit != git_commit:
        raise ProvenanceError("GBS_COMMIT disagrees with the requested git commit")
    env_container = os.environ.get("GBS_CONTAINER_DIGEST")
    if env_container and env_container != container_digest:
        raise ProvenanceError("GBS_CONTAINER_DIGEST disagrees with the requested container")

    archive_record = file_record(archive)
    if archive_record["sha256"] != expected_source_archive_sha256:
        raise ProvenanceError("source archive SHA-256 does not match the expected release hash")
    tree_before = source_tree_inventory(root)
    if expected_source_tree_sha256 is not None:
        if not HEX_SHA256.fullmatch(expected_source_tree_sha256):
            raise ProvenanceError("expected source-tree SHA-256 must be 64 lowercase hex")
        if tree_before["tree_sha256"] != expected_source_tree_sha256:
            raise ProvenanceError("source tree SHA-256 does not match the expected release tree")

    compile_records = _metadata_group(
        "compile_commands.json", compile_commands, root, compile_commands_record
    )
    cache_records = _metadata_group("CMakeCache.txt", cmake_caches, root, cmake_cache_record)

    device_records = _artifact_group(
        "device-code artifact", device_code, root, (".ptx", ".cubin", ".fatbin")
    )
    if not any(item["filename"].endswith(".ptx") for item in device_records):
        raise ProvenanceError("device-code artifacts must include extracted PTX")
    if not any(
        item["filename"].endswith((".cubin", ".fatbin")) for item in device_records
    ):
        raise ProvenanceError("device-code artifacts must include a cubin or fatbin")
    artifact_records = {
        "build_products": _artifact_group("build product", build_products, root),
        "device_code": device_records,
        "extensions": _artifact_group(
            "compiled extension", extensions, root, (".so", ".pyd", ".dylib")
        ),
        "wheels": _artifact_group("wheel", wheels, root, (".whl",)),
    }
    missing_tools = REQUIRED_TOOLS - set(tools)
    if missing_tools:
        raise ProvenanceError(
            "missing required compiler/tool queries: " + ", ".join(sorted(missing_tools))
        )
    tool_records = {
        name: tool_record(name, executable, version_args)
        for name, (executable, version_args) in sorted(tools.items())
    }

    build = {
        "compile_commands": compile_records,
        "cmake_caches": cache_records,
        "tools": tool_records,
        "artifacts": artifact_records,
    }
    tree_after = source_tree_inventory(root)
    if tree_after != tree_before:
        raise ProvenanceError("source tree changed while provenance was being captured")
    if sha256_file(archive) != archive_record["sha256"]:
        raise ProvenanceError("source archive changed while provenance was being captured")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "git_commit": git_commit,
            "git_tree": git_tree,
            "archive": archive_record,
            "tree": tree_before,
        },
        "container_digest": container_digest,
        "build": build,
        "environment": {
            "selected": selected_environment(env_keys),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
        },
    }
    payload["manifest_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    _assert_strict_json(payload)
    return payload


def write_strict_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create a strict JSON artifact without overwriting prior evidence."""
    _assert_strict_json(payload)
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
    except FileExistsError as exc:
        raise ProvenanceError(f"refusing to overwrite provenance artifact: {output}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--expected-source-archive-sha256", required=True)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--expected-source-tree-sha256")
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--git-tree", required=True)
    parser.add_argument("--container-digest", required=True)
    parser.add_argument("--compile-commands", type=Path, action="append", required=True)
    parser.add_argument("--cmake-cache", type=Path, action="append", required=True)
    parser.add_argument("--build-product", type=Path, action="append", required=True)
    parser.add_argument("--device-code", type=Path, action="append", required=True)
    parser.add_argument("--extension", type=Path, action="append", required=True)
    parser.add_argument("--wheel", type=Path, action="append", required=True)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="additional non-secret environment variable to record",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_root = args.source_tree.expanduser().resolve()
    if _inside(args.output, source_root):
        raise ProvenanceError("output manifest must be outside the immutable source tree")
    tools = {
        "cmake": (args.cmake, ("--version",)),
        "cuobjdump": (args.cuobjdump, ("--version",)),
        "cxx": (args.cxx, ("--version",)),
        "nvcc": (args.nvcc, ("--version",)),
        "python": (sys.executable, ("--version",)),
    }
    payload = capture_build_provenance(
        source_archive=args.source_archive,
        expected_source_archive_sha256=args.expected_source_archive_sha256,
        source_tree=args.source_tree,
        expected_source_tree_sha256=args.expected_source_tree_sha256,
        git_commit=args.git_commit,
        git_tree=args.git_tree,
        container_digest=args.container_digest,
        compile_commands=args.compile_commands,
        cmake_caches=args.cmake_cache,
        build_products=args.build_product,
        device_code=args.device_code,
        extensions=args.extension,
        wheels=args.wheel,
        tools=tools,
        env_keys=(*DEFAULT_ENV_KEYS, *args.env),
    )
    write_strict_json(args.output, payload)
    print(f"wrote {args.output} ({payload['manifest_sha256']})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProvenanceError as exc:
        raise SystemExit(f"ABORT: {exc}") from exc
