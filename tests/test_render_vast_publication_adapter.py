from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

import scripts.render_vast_publication_adapter as adapter_module
from scripts.render_vast_publication_adapter import (
    AdapterError,
    inspect_archive,
    render_adapter,
    write_adapter,
)


TREE_SHA = "c" * 64
CONTAINER = "nvidia/cuda@sha256:" + "d" * 64


def _archive(path: Path, *, unsafe: bool = False, duplicate: bool = False) -> Path:
    driver = b"#!/usr/bin/env bash\necho driver\n"
    with tarfile.open(path, "w:gz") as handle:
        info = tarfile.TarInfo("GBSKernels/scripts/publication_gpu_session.sh")
        info.mode = 0o755
        info.size = len(driver)
        handle.addfile(info, io.BytesIO(driver))
        if duplicate:
            repeated = tarfile.TarInfo("GBSKernels/scripts/publication_gpu_session.sh")
            repeated.mode = 0o755
            repeated.size = len(driver)
            handle.addfile(repeated, io.BytesIO(driver))
        if unsafe:
            bad = tarfile.TarInfo("../escape")
            bad.size = 1
            handle.addfile(bad, io.BytesIO(b"x"))
    return path


def _git_source(tmp_path: Path) -> tuple[Path, Path, str, str]:
    repository = tmp_path / "repository"
    driver = repository / "scripts" / "publication_gpu_session.sh"
    driver.parent.mkdir(parents=True)
    driver.write_bytes(b"#!/usr/bin/env bash\necho driver\n")
    driver.chmod(0o755)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git", "-C", str(repository),
            "-c", "user.name=Test",
            "-c", "user.email=test@example.invalid",
            "commit", "-q", "-m", "source",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    archive = tmp_path / "source.tar"
    with archive.open("xb") as output:
        subprocess.run(
            [
                "git", "-C", str(repository), "archive", "--format=tar",
                "--prefix=GBSKernels/", commit,
            ],
            check=True,
            stdout=output,
        )
    return repository, archive, commit, tree


def test_archive_inspection_binds_root_and_driver(tmp_path):
    archive = _archive(tmp_path / "source.tar.gz")
    root, driver_hash, tree_hash = inspect_archive(archive)
    assert root == "GBSKernels"
    assert driver_hash == hashlib.sha256(
        b"#!/usr/bin/env bash\necho driver\n"
    ).hexdigest()
    entries = [{
        "path": "scripts/publication_gpu_session.sh",
        "type": "file",
        "bytes": len(b"#!/usr/bin/env bash\necho driver\n"),
        "executable": True,
        "sha256": driver_hash,
    }]
    canonical = (json.dumps(
        entries, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n").encode("ascii")
    assert tree_hash == hashlib.sha256(canonical).hexdigest()


def test_rendered_adapter_contains_exact_contract_and_packages_evidence(tmp_path):
    repository, archive, commit, tree = _git_source(tmp_path)
    content = render_adapter(
        archive=archive,
        git_commit=commit,
        git_tree=tree,
        source_tree_sha256=None,
        container_digest=CONTAINER,
        repository=repository,
    )
    assert f"EXPECTED_ARCHIVE_SHA256={hashlib.sha256(archive.read_bytes()).hexdigest()}" in content
    assert f"GIT_COMMIT={commit}" in content
    assert f"GIT_TREE={tree}" in content
    assert f"CONTAINER_DIGEST={CONTAINER}" in content
    assert "--source-tree-sha256 \"$EXPECTED_SOURCE_TREE_SHA256\"" in content
    assert 'tar -czf "$PARTIAL_BUNDLE"' in content
    assert 'apt-get install -y --no-install-recommends' in content
    subprocess.run(["bash", "-n"], input=content, text=True, check=True)


def test_rejects_unsafe_archive_and_bad_metadata(tmp_path):
    with pytest.raises(AdapterError, match="unsafe archive member"):
        inspect_archive(_archive(tmp_path / "unsafe.tar.gz", unsafe=True))
    with pytest.raises(AdapterError, match="duplicate archive member"):
        inspect_archive(_archive(tmp_path / "duplicate.tar.gz", duplicate=True))
    repository, archive, commit, tree = _git_source(tmp_path)
    with pytest.raises(AdapterError, match="git commit"):
        render_adapter(
            archive=archive,
            git_commit="short",
            git_tree=tree,
            source_tree_sha256=TREE_SHA,
            container_digest=CONTAINER,
            repository=repository,
        )
    with pytest.raises(AdapterError, match="does not match the source archive"):
        render_adapter(
            archive=archive,
            git_commit=commit,
            git_tree=tree,
            source_tree_sha256=TREE_SHA,
            container_digest=CONTAINER,
            repository=repository,
        )


def test_archive_inspection_enforces_resource_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter_module, "MAX_ARCHIVE_MEMBERS", 1)
    with pytest.raises(AdapterError, match="member-count limit"):
        inspect_archive(_archive(tmp_path / "too-many.tar", duplicate=True))
    monkeypatch.setattr(adapter_module, "MAX_ARCHIVE_MEMBERS", 20_000)
    monkeypatch.setattr(adapter_module, "MAX_ARCHIVE_EXPANDED_BYTES", 1)
    with pytest.raises(AdapterError, match="expanded-size limit"):
        inspect_archive(_archive(tmp_path / "too-large.tar"))
def test_rejects_archive_that_is_not_the_supplied_git_commit(tmp_path):
    repository, archive, commit, tree = _git_source(tmp_path)
    with tarfile.open(archive, "a") as handle:
        extra = tarfile.TarInfo("GBSKernels/uncommitted.txt")
        extra.size = 1
        handle.addfile(extra, io.BytesIO(b"x"))
    with pytest.raises(AdapterError, match="do not match the supplied Git commit"):
        render_adapter(
            archive=archive,
            git_commit=commit,
            git_tree=tree,
            source_tree_sha256=None,
            container_digest=CONTAINER,
            repository=repository,
        )


def test_writer_is_create_only_and_owner_executable(tmp_path):
    output = tmp_path / "adapter.sh"
    write_adapter(output, "#!/bin/sh\n")
    assert output.stat().st_mode & 0o100
    assert output.stat().st_mode & 0o077 == 0
    with pytest.raises(AdapterError, match="refusing to overwrite"):
        write_adapter(output, "different\n")


def test_rendered_adapter_accepts_standard_sidecar_and_returns_bundle(tmp_path):
    repository, archive, commit, tree = _git_source(tmp_path)
    work = tmp_path / "remote"
    fake_bin = work / "bin"
    fake_bin.mkdir(parents=True)
    for name, body in {
        "apt-get": "#!/bin/sh\nexit 0\n",
        "nvidia-smi": "#!/bin/sh\nprintf '8.9\\n'\n",
    }.items():
        command = fake_bin / name
        command.write_text(body, encoding="ascii")
        command.chmod(0o755)
    adapter = work / "publication-session.sh"
    write_adapter(
        adapter,
        render_adapter(
            archive=archive,
            git_commit=commit,
            git_tree=tree,
            source_tree_sha256=None,
            container_digest=CONTAINER,
            repository=repository,
        ),
    )
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = work / "source.sha256"
    checksum.write_bytes(f"{digest.upper()}  source.archive".encode("ascii"))
    completed = subprocess.run(
        ["bash", str(adapter), str(archive), str(checksum), "output.tar.gz"],
        cwd=work,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    bundle = work / "output.tar.gz"
    assert bundle.is_file()
    with tarfile.open(bundle, "r:gz") as handle:
        assert set(handle.getnames()) == {
            "publication-session.sh",
            "adapter.log",
            "bootstrap.log",
            "session.log",
        }


def test_rendered_adapter_packages_logs_on_prebootstrap_failure(tmp_path):
    repository, archive, commit, tree = _git_source(tmp_path)
    work = tmp_path / "remote"
    work.mkdir()
    adapter = work / "publication-session.sh"
    write_adapter(
        adapter,
        render_adapter(
            archive=archive,
            git_commit=commit,
            git_tree=tree,
            source_tree_sha256=None,
            container_digest=CONTAINER,
            repository=repository,
        ),
    )
    checksum = work / "source.sha256"
    checksum.write_text(f"{'0' * 64}  source.archive\n", encoding="ascii")
    completed = subprocess.run(
        ["bash", str(adapter), str(archive), str(checksum), "failure.tar.gz"],
        cwd=work,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    with tarfile.open(work / "failure.tar.gz", "r:gz") as handle:
        assert set(handle.getnames()) == {
            "publication-session.sh",
            "adapter.log",
            "bootstrap.log",
            "session.log",
        }
        adapter_log = handle.extractfile("adapter.log")
        assert adapter_log is not None
        assert b"checksum sidecar disagrees with adapter" in adapter_log.read()
