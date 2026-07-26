#!/usr/bin/env python3
"""Render the immutable-archive adapter consumed by ``vast_publication.py``.

The Vast lifecycle runner intentionally accepts a simple three-positional
remote script.  The publication GPU driver intentionally accepts explicit,
fail-closed provenance flags.  This renderer bridges those contracts only
after the release archive and all Git metadata are known.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Sequence


HEX256 = re.compile(r"\A[0-9a-f]{64}\Z")
GIT_ID = re.compile(r"\A(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
CONTAINER = re.compile(r"\A[^\s@]+(?:/[^\s@]+)*@sha256:[0-9a-f]{64}\Z")
MAX_ARCHIVE_BYTES = 2 * 1024**3
MAX_ARCHIVE_MEMBERS = 20_000
MAX_ARCHIVE_MEMBER_BYTES = 2 * 1024**3
MAX_ARCHIVE_EXPANDED_BYTES = 8 * 1024**3
READ_CHUNK_BYTES = 1024 * 1024


class AdapterError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise AdapterError("file must not be a symbolic link")
    try:
        candidate = expanded.resolve(strict=True)
        before = candidate.stat()
    except OSError as exc:
        raise AdapterError(f"cannot stat file: {path}") from exc
    if not candidate.is_file():
        raise AdapterError("file must be regular")
    digest = hashlib.sha256()
    try:
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(READ_CHUNK_BYTES), b""):
                digest.update(chunk)
        after = candidate.stat()
    except OSError as exc:
        raise AdapterError(f"cannot hash file: {path}") from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise AdapterError("file changed while it was being hashed")
    return digest.hexdigest()


def inspect_archive(path: Path) -> tuple[str, str, str]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise AdapterError("source archive must be a regular non-symlink file")
    try:
        archive = expanded.resolve(strict=True)
        archive_size = archive.stat().st_size
    except OSError as exc:
        raise AdapterError(f"cannot stat source archive: {path}") from exc
    if not archive.is_file():
        raise AdapterError("source archive must be a regular non-symlink file")
    if archive_size > MAX_ARCHIVE_BYTES:
        raise AdapterError("compressed source archive exceeds the size limit")
    roots: set[str] = set()
    names: set[str] = set()
    file_hashes: dict[str, tuple[str, int]] = {}
    entries: list[dict[str, object]] = []
    expanded_bytes = 0
    member_count = 0
    try:
        with tarfile.open(archive, "r:*") as handle:
            for member in handle:
                member_count += 1
                if member_count > MAX_ARCHIVE_MEMBERS:
                    raise AdapterError("source archive exceeds the member-count limit")
                pure = PurePosixPath(member.name)
                if pure.is_absolute() or not pure.parts or ".." in pure.parts:
                    raise AdapterError(f"unsafe archive member: {member.name!r}")
                canonical_name = pure.as_posix()
                if canonical_name in names:
                    raise AdapterError(f"duplicate archive member: {member.name!r}")
                names.add(canonical_name)
                roots.add(pure.parts[0])
                if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                    raise AdapterError(f"unsupported archive member: {member.name!r}")
                if not member.isfile():
                    continue
                if member.size < 0 or member.size > MAX_ARCHIVE_MEMBER_BYTES:
                    raise AdapterError(f"archive member exceeds the size limit: {member.name!r}")
                expanded_bytes += member.size
                if expanded_bytes > MAX_ARCHIVE_EXPANDED_BYTES:
                    raise AdapterError("source archive exceeds the expanded-size limit")
                relative = PurePosixPath(*pure.parts[1:]).as_posix()
                if not relative or relative == ".":
                    raise AdapterError("source archive contains a file at its top-level root")
                extracted_file = handle.extractfile(member)
                if extracted_file is None:
                    raise AdapterError(f"cannot read archive member: {member.name!r}")
                digest = hashlib.sha256()
                bytes_read = 0
                for chunk in iter(lambda: extracted_file.read(READ_CHUNK_BYTES), b""):
                    bytes_read += len(chunk)
                    digest.update(chunk)
                if bytes_read != member.size:
                    raise AdapterError(
                        f"archive member size changed while reading: {member.name!r}"
                    )
                file_hash = digest.hexdigest()
                file_hashes[canonical_name] = (file_hash, bytes_read)
                entries.append({
                    "path": relative,
                    "type": "file",
                    "bytes": bytes_read,
                    "executable": bool(member.mode & 0o111),
                    "sha256": file_hash,
                })
    except (tarfile.TarError, OSError) as exc:
        raise AdapterError(f"cannot inspect source archive: {exc}") from exc
    if member_count == 0:
        raise AdapterError("source archive is empty")
    if len(roots) != 1:
        raise AdapterError("source archive must contain exactly one top-level directory")
    root = next(iter(roots))
    driver_name = f"{root}/scripts/publication_gpu_session.sh"
    driver_record = file_hashes.get(driver_name)
    if driver_record is None:
        raise AdapterError("source archive lacks scripts/publication_gpu_session.sh")
    driver_hash, driver_size = driver_record
    if driver_size == 0:
        raise AdapterError("publication GPU driver is empty")
    entries.sort(key=lambda item: str(item["path"]))
    canonical = (json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n").encode("ascii")
    return (
        root,
        driver_hash,
        hashlib.sha256(canonical).hexdigest(),
    )


def _git_output(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AdapterError("Git verification could not run") from exc
    if completed.returncode != 0:
        raise AdapterError("Git verification failed")
    return completed.stdout.strip()


def verify_git_archive(
    repository: Path,
    archive: Path,
    git_commit: str,
    git_tree: str,
) -> tuple[str, str, str]:
    """Prove that the supplied archive is the export of ``git_commit``."""

    repo = repository.expanduser().resolve()
    if not repo.is_dir():
        raise AdapterError("Git repository path is not a directory")
    resolved_commit = _git_output(repo, "rev-parse", "--verify", f"{git_commit}^{{commit}}")
    if resolved_commit != git_commit:
        raise AdapterError("Git commit does not resolve to the supplied object ID")
    resolved_tree = _git_output(repo, "rev-parse", f"{git_commit}^{{tree}}")
    if resolved_tree != git_tree:
        raise AdapterError("Git tree does not match the supplied commit")

    archive_root, driver_hash, archive_tree_hash = inspect_archive(archive)
    with tempfile.TemporaryDirectory(prefix="gbskernels-git-archive-") as temporary:
        expected_archive = Path(temporary) / "expected.tar"
        try:
            with expected_archive.open("xb") as output:
                completed = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(repo),
                        "archive",
                        "--format=tar",
                        f"--prefix={archive_root}/",
                        git_commit,
                    ],
                    check=False,
                    stdout=output,
                    stderr=subprocess.PIPE,
                    timeout=120,
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AdapterError("Git archive verification could not run") from exc
        if completed.returncode != 0:
            raise AdapterError("Git could not reproduce the source archive")
        _, expected_driver_hash, expected_tree_hash = inspect_archive(expected_archive)
    if archive_tree_hash != expected_tree_hash or driver_hash != expected_driver_hash:
        raise AdapterError("source archive contents do not match the supplied Git commit")
    return archive_root, driver_hash, archive_tree_hash


def _quoted(value: str) -> str:
    return shlex.quote(value)


def render_adapter(
    *,
    archive: Path,
    git_commit: str,
    git_tree: str,
    source_tree_sha256: str | None,
    container_digest: str,
    repository: Path,
) -> str:
    if not GIT_ID.fullmatch(git_commit):
        raise AdapterError("git commit must be a full lowercase Git object ID")
    if not GIT_ID.fullmatch(git_tree):
        raise AdapterError("git tree must be a full lowercase Git object ID")
    if source_tree_sha256 is not None and not HEX256.fullmatch(source_tree_sha256):
        raise AdapterError("source-tree SHA-256 must be 64 lowercase hex")
    if not CONTAINER.fullmatch(container_digest):
        raise AdapterError("container digest must be image@sha256:<64 lowercase hex>")

    archive_hash = sha256_file(archive)
    archive_root, driver_hash, archive_tree_hash = verify_git_archive(
        repository, archive, git_commit, git_tree
    )
    if source_tree_sha256 is not None and source_tree_sha256 != archive_tree_hash:
        raise AdapterError("source-tree SHA-256 does not match the source archive")
    values = {
        "EXPECTED_ARCHIVE_SHA256": archive_hash,
        "EXPECTED_SOURCE_TREE_SHA256": archive_tree_hash,
        "EXPECTED_DRIVER_SHA256": driver_hash,
        "ARCHIVE_ROOT": archive_root,
        "GIT_COMMIT": git_commit,
        "GIT_TREE": git_tree,
        "CONTAINER_DIGEST": container_digest,
    }
    assignments = "\n".join(f"{key}={_quoted(value)}" for key, value in values.items())
    return f"""#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\\n\\t'
umask 022
export COPYFILE_DISABLE=1

{assignments}

[ "$#" -eq 3 ] || {{ echo "usage: $0 ARCHIVE CHECKSUM OUTPUT" >&2; exit 2; }}
ARCHIVE="$1"
CHECKSUM="$2"
OUTPUT_NAME="$3"
[[ "$OUTPUT_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || {{ echo "invalid output filename" >&2; exit 2; }}
for reserved_name in adapter.log bootstrap.log session.log extracted publication-output \
  publication-session.sh source.archive source.sha256
do
  [ "$OUTPUT_NAME" != "$reserved_name" ] \
    || {{ echo "output filename collides with a publication-session path" >&2; exit 2; }}
done
WORK_DIR="$(pwd -P)"
OUTPUT_BUNDLE="$WORK_DIR/$OUTPUT_NAME"
PARTIAL_BUNDLE="$WORK_DIR/.${{OUTPUT_NAME}}.partial"
EXTRACT_ROOT="$WORK_DIR/extracted"
OUTPUT_ROOT="$WORK_DIR/publication-output"
BOOTSTRAP_LOG="$WORK_DIR/bootstrap.log"
SESSION_LOG="$WORK_DIR/session.log"
ADAPTER_LOG="$WORK_DIR/adapter.log"
[ ! -e "$OUTPUT_BUNDLE" ] && [ ! -e "$PARTIAL_BUNDLE" ] \
  && [ ! -e "$EXTRACT_ROOT" ] && [ ! -e "$OUTPUT_ROOT" ] \
  || {{ echo "remote publication paths already exist" >&2; exit 2; }}

: >"$BOOTSTRAP_LOG"
: >"$SESSION_LOG"
: >"$ADAPTER_LOG"
exec 3>&1 4>&2
exec >>"$ADAPTER_LOG" 2>&1

bundle_on_exit() {{
  local primary_rc=$?
  local bundle_rc=0
  local -a bundle_args
  trap - EXIT
  set +e
  exec 1>&3 2>&4
  exec 3>&- 4>&-
  if [ ! -e "$OUTPUT_BUNDLE" ] && [ ! -e "$PARTIAL_BUNDLE" ]; then
    bundle_args=(-C "$WORK_DIR" "$(basename "$0")")
    [ -f "$BOOTSTRAP_LOG" ] \
      && bundle_args+=(-C "$WORK_DIR" "$(basename "$BOOTSTRAP_LOG")")
    [ -f "$SESSION_LOG" ] \
      && bundle_args+=(-C "$WORK_DIR" "$(basename "$SESSION_LOG")")
    [ -f "$ADAPTER_LOG" ] \
      && bundle_args+=(-C "$WORK_DIR" "$(basename "$ADAPTER_LOG")")
    [ -d "$OUTPUT_ROOT/evidence" ] \
      && bundle_args+=(-C "$OUTPUT_ROOT" evidence)
    tar -czf "$PARTIAL_BUNDLE" "${{bundle_args[@]}}"
    bundle_rc=$?
    if [ "$bundle_rc" -eq 0 ]; then
      mv "$PARTIAL_BUNDLE" "$OUTPUT_BUNDLE"
      bundle_rc=$?
    fi
  fi
  if [ "$primary_rc" -eq 0 ] && [ "$bundle_rc" -ne 0 ]; then
    primary_rc=$bundle_rc
  fi
  exit "$primary_rc"
}}
trap bundle_on_exit EXIT

case "$ARCHIVE" in
  /*) ARCHIVE_PATH="$ARCHIVE" ;;
  *) ARCHIVE_PATH="$WORK_DIR/$ARCHIVE" ;;
esac
[ -f "$ARCHIVE_PATH" ] && [ ! -L "$ARCHIVE_PATH" ] \
  || {{ echo "source archive is not a regular non-symlink file" >&2; exit 2; }}
[ -f "$CHECKSUM" ] && [ ! -L "$CHECKSUM" ] \
  || {{ echo "checksum sidecar is not a regular non-symlink file" >&2; exit 2; }}
CHECKSUM_VALUE="$(LC_ALL=C awk \
  'NF {{ count++; value=tolower($1) }} END {{ if (count != 1) exit 2; print value }}' \
  "$CHECKSUM")" \
  || {{ echo "checksum sidecar must contain exactly one record" >&2; exit 2; }}
[[ "$CHECKSUM_VALUE" =~ ^[0-9a-f]{{64}}$ ]] \
  || {{ echo "checksum sidecar does not contain a SHA-256" >&2; exit 2; }}
[ "$CHECKSUM_VALUE" = "$EXPECTED_ARCHIVE_SHA256" ] \
  || {{ echo "checksum sidecar disagrees with adapter" >&2; exit 2; }}
ACTUAL_ARCHIVE_SHA256="$(sha256sum -- "$ARCHIVE_PATH" | awk '{{print $1}}')"
[ "$ACTUAL_ARCHIVE_SHA256" = "$EXPECTED_ARCHIVE_SHA256" ] \
  || {{ echo "source archive hash disagrees with adapter" >&2; exit 2; }}

bootstrap_rc=0
{{
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y --no-install-recommends \
    ca-certificates cmake ninja-build python3.12 python3.12-dev python3.12-venv
}} >"$BOOTSTRAP_LOG" 2>&1 || bootstrap_rc=$?
if [ "$bootstrap_rc" -ne 0 ]; then
  exit "$bootstrap_rc"
fi

mkdir "$EXTRACT_ROOT"
tar -xf "$ARCHIVE_PATH" -C "$EXTRACT_ROOT" --no-same-owner
SOURCE_ROOT="$EXTRACT_ROOT/$ARCHIVE_ROOT"
DRIVER="$SOURCE_ROOT/scripts/publication_gpu_session.sh"
[ -f "$DRIVER" ] && [ ! -L "$DRIVER" ] \
  || {{ echo "archive publication driver is missing" >&2; exit 2; }}
ACTUAL_DRIVER_SHA256="$(sha256sum "$DRIVER" | awk '{{print $1}}')"
[ "$ACTUAL_DRIVER_SHA256" = "$EXPECTED_DRIVER_SHA256" ] \
  || {{ echo "archive publication driver hash mismatch" >&2; exit 2; }}
CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
  | awk 'NR == 1 {{gsub(/\\./, "", $1); print $1}}')"
[[ "$CUDA_ARCH" =~ ^[0-9]+$ ]] \
  || {{ echo "could not determine CUDA architecture" >&2; exit 2; }}

session_rc=0
bash "$DRIVER" \
  --source-root "$SOURCE_ROOT" \
  --source-archive "$ARCHIVE_PATH" \
  --archive-sha256 "$EXPECTED_ARCHIVE_SHA256" \
  --source-tree-sha256 "$EXPECTED_SOURCE_TREE_SHA256" \
  --git-commit "$GIT_COMMIT" \
  --git-tree "$GIT_TREE" \
  --container-digest "$CONTAINER_DIGEST" \
  --output-root "$OUTPUT_ROOT" \
  --cuda-arch "$CUDA_ARCH" \
  --python python3.12 >"$SESSION_LOG" 2>&1 || session_rc=$?
exit "$session_rc"
"""


def write_adapter(path: Path, content: str) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o755)
    except FileExistsError as exc:
        raise AdapterError(f"refusing to overwrite adapter: {output}") from exc
    with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
        handle.write(content)
    output.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--git-tree", required=True)
    parser.add_argument("--source-tree-sha256")
    parser.add_argument("--container-digest", required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    content = render_adapter(
        archive=args.archive,
        git_commit=args.git_commit,
        git_tree=args.git_tree,
        source_tree_sha256=args.source_tree_sha256,
        container_digest=args.container_digest,
        repository=args.repository,
    )
    write_adapter(args.output, content)
    print(f"wrote {args.output} ({hashlib.sha256(content.encode('ascii')).hexdigest()})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AdapterError as exc:
        raise SystemExit(f"ABORT: {exc}") from exc
