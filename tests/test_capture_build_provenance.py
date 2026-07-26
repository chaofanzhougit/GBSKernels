from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.capture_build_provenance import (
    ProvenanceError,
    capture_build_provenance,
    source_tree_inventory,
    write_strict_json,
)


COMMIT = "a" * 40
TREE = "b" * 40
CONTAINER = "nvidia/cuda@sha256:" + "c" * 64


def _write(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _tool(path: Path, version: str) -> Path:
    path.write_text(f"#!/bin/sh\nprintf '%s\\n' '{version}'\n", encoding="ascii")
    path.chmod(0o755)
    return path


def _fixture(tmp_path: Path) -> dict:
    source = tmp_path / "source"
    _write(source / "alpha.txt", b"alpha\n")
    _write(source / "nested" / "beta.py", b"print('beta')\n").chmod(0o755)
    archive = _write(tmp_path / "release.tar", b"immutable release archive\n")
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()

    build = tmp_path / "build"
    compile_commands = _write(
        build / "compile_commands.json",
        json.dumps([
            {
                "directory": str(build),
                "file": str(source / "kernel.cu"),
                "arguments": ["nvcc", "-O3", "-arch=sm_89", "-c", "kernel.cu"],
                "output": "kernel.o",
            }
        ]).encode("ascii"),
    )
    cmake_cache = _write(
        build / "CMakeCache.txt",
        b"CMAKE_CUDA_ARCHITECTURES:STRING=89\n"
        b"CMAKE_CUDA_FLAGS:STRING=-lineinfo\n"
        b"CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++\n",
    )
    product = _write(build / "check_tor_recursive", b"ELF executable")
    ptx = _write(build / "tor_recursive.ptx", b"// PTX\n")
    cubin = _write(build / "tor_recursive.cubin", b"ELF cubin")
    sass = _write(build / "embedded.sass", b"Fatbin elf code:\nFunction : kernel\n")
    extension = _write(build / "gbskernels_ext.so", b"ELF extension")
    wheel = _write(tmp_path / "gbskernels-0.2.1-py3-none-any.whl", b"wheel bytes")
    tools = {
        name: (str(_tool(tmp_path / f"tool-{name}", f"{name} 1.2.3")), ("--version",))
        for name in ("nvcc", "cmake", "cxx", "cuobjdump", "python")
    }
    return {
        "source_archive": archive,
        "expected_source_archive_sha256": archive_hash,
        "source_tree": source,
        "git_commit": COMMIT,
        "git_tree": TREE,
        "container_digest": CONTAINER,
        "compile_commands": [compile_commands],
        "cmake_caches": [cmake_cache],
        "build_products": [product],
        "device_code": [ptx, cubin, sass],
        "extensions": [extension],
        "wheels": [wheel],
        "tools": tools,
        "env_keys": ("CXXFLAGS", "GBS_COMMIT", "GBS_CONTAINER_DIGEST"),
    }


def test_capture_binds_tree_commands_tools_and_artifacts(tmp_path, monkeypatch):
    args = _fixture(tmp_path)
    monkeypatch.setenv("CXXFLAGS", "-fno-fast-math")
    monkeypatch.setenv("GBS_COMMIT", COMMIT)
    monkeypatch.setenv("GBS_CONTAINER_DIGEST", CONTAINER)
    expected_tree = source_tree_inventory(args["source_tree"])["tree_sha256"]
    args["expected_source_tree_sha256"] = expected_tree

    payload = capture_build_provenance(**args)

    assert payload["schema"] == "gbskernels.build-provenance.v1"
    assert payload["source"]["tree"]["tree_sha256"] == expected_tree
    assert [row["path"] for row in payload["source"]["tree"]["entries"]] == [
        "alpha.txt",
        "nested/beta.py",
    ]
    assert payload["build"]["compile_commands"][0]["commands"][0]["argv"][1:3] == [
        "-O3",
        "-arch=sm_89",
    ]
    assert payload["build"]["cmake_caches"][0]["selected"]["CMAKE_CUDA_FLAGS"] == "-lineinfo"
    device_names = {
        item["filename"] for item in payload["build"]["artifacts"]["device_code"]
    }
    assert device_names == {
        "tor_recursive.ptx",
        "tor_recursive.cubin",
        "embedded.sass",
    }
    assert payload["build"]["artifacts"]["extensions"][0]["sha256"]
    assert payload["build"]["artifacts"]["wheels"][0]["sha256"]
    assert payload["build"]["tools"]["nvcc"]["sha256"]
    assert payload["environment"]["selected"]["CXXFLAGS"] == "-fno-fast-math"
    assert len(payload["manifest_sha256"]) == 64
    json.dumps(payload, allow_nan=False)


def test_tree_digest_ignores_mtime_but_covers_bytes_and_executable_bit(tmp_path):
    source = tmp_path / "source"
    file_a = _write(source / "a", b"same bytes")
    first = source_tree_inventory(source)["tree_sha256"]
    os.utime(file_a, (1_700_000_000, 1_700_000_000))
    assert source_tree_inventory(source)["tree_sha256"] == first
    file_a.chmod(0o755)
    executable = source_tree_inventory(source)["tree_sha256"]
    assert executable != first
    file_a.write_bytes(b"different bytes")
    assert source_tree_inventory(source)["tree_sha256"] != executable


def test_capture_fails_closed_for_missing_artifact_and_hash_mismatch(tmp_path):
    args = _fixture(tmp_path)
    args["device_code"] = []
    with pytest.raises(ProvenanceError, match="device-code artifact"):
        capture_build_provenance(**args)

    args = _fixture(tmp_path / "other")
    args["expected_source_archive_sha256"] = "0" * 64
    with pytest.raises(ProvenanceError, match="source archive SHA-256"):
        capture_build_provenance(**args)


def test_capture_accepts_cubin_without_embedded_ptx(tmp_path):
    args = _fixture(tmp_path)
    args["device_code"] = [
        artifact for artifact in args["device_code"] if artifact.suffix == ".cubin"
    ]

    payload = capture_build_provenance(**args)

    assert [
        item["filename"] for item in payload["build"]["artifacts"]["device_code"]
    ] == ["tor_recursive.cubin"]


def test_capture_accepts_embedded_sass_without_materialized_cubin(tmp_path):
    args = _fixture(tmp_path)
    args["device_code"] = [
        artifact for artifact in args["device_code"] if artifact.suffix == ".sass"
    ]

    payload = capture_build_provenance(**args)

    assert [
        item["filename"] for item in payload["build"]["artifacts"]["device_code"]
    ] == ["embedded.sass"]


def test_capture_rejects_ptx_without_executable_device_evidence(tmp_path):
    args = _fixture(tmp_path)
    args["device_code"] = [
        artifact for artifact in args["device_code"] if artifact.suffix == ".ptx"
    ]

    with pytest.raises(ProvenanceError, match="cubin, fatbin, or SASS"):
        capture_build_provenance(**args)


def test_capture_rejects_unrecognized_sass_dump(tmp_path):
    args = _fixture(tmp_path)
    sass = next(
        artifact for artifact in args["device_code"] if artifact.suffix == ".sass"
    )
    sass.write_text("not cuobjdump output\n", encoding="ascii")
    args["device_code"] = [sass]

    with pytest.raises(ProvenanceError, match="no recognizable embedded device code"):
        capture_build_provenance(**args)


def test_capture_rejects_nonstandard_compile_json(tmp_path):
    args = _fixture(tmp_path)
    args["compile_commands"][0].write_text("[NaN]\n", encoding="ascii")
    with pytest.raises(ProvenanceError, match="non-standard JSON constant"):
        capture_build_provenance(**args)


def test_strict_writer_is_create_only_and_rejects_nan(tmp_path):
    output = tmp_path / "provenance.json"
    write_strict_json(output, {"finite": 1.0})
    assert json.loads(output.read_text()) == {"finite": 1.0}
    with pytest.raises(ProvenanceError, match="overwrite"):
        write_strict_json(output, {"finite": 2.0})
    with pytest.raises(ProvenanceError, match="non-finite"):
        write_strict_json(tmp_path / "bad.json", {"bad": float("nan")})
