#!/usr/bin/env python3
"""Fail-safe Vast.ai runner for one publication GPU session.

The runner uses the official ``vastai`` CLI for offer search, provisioning,
instance inspection, and destruction.  It never reads an API-key file, accepts
an API key, or prints subprocess output.  Authentication remains entirely the
CLI's responsibility.

The paid lifecycle has one owner and one instance ID.  Once creation returns an
ID, all normal errors and interrupts converge on the same exact-ID destruction
path.  A unique label is used only to recover that ID if the create command
returns ambiguously.  Local output and receipt paths are create-only.

The uploaded session script is invoked as::

    bash publication-session.sh source.archive source.sha256 publication-output.tar.gz

It must create the third argument in its working directory.  The source archive
is verified remotely against the caller-supplied checksum before execution.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


IMAGE_DIGEST_RE = re.compile(
    r"\A[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}\Z"
)
HEX_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
SAFE_REMOTE_NAME_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
SAFE_SSH_USER_RE = re.compile(r"\A[A-Za-z_][A-Za-z0-9_-]{0,31}\Z")
SAFE_SSH_HOST_RE = re.compile(r"\A[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?\Z")

REMOTE_ARCHIVE = "source.archive"
REMOTE_CHECKSUM = "source.sha256"
REMOTE_SESSION = "publication-session.sh"
REMOTE_RESERVED_NAMES = frozenset({
    REMOTE_ARCHIVE,
    REMOTE_CHECKSUM,
    REMOTE_SESSION,
    "adapter.log",
    "bootstrap.log",
    "session.log",
    "extracted",
    "publication-output",
})
ADAPTER_ASSIGNMENTS = (
    "EXPECTED_ARCHIVE_SHA256",
    "EXPECTED_SOURCE_TREE_SHA256",
    "EXPECTED_DRIVER_SHA256",
    "GIT_COMMIT",
    "GIT_TREE",
    "CONTAINER_DIGEST",
)

CommandExecutor = Callable[
    [Sequence[str], float | None], subprocess.CompletedProcess[str]
]


class PublicationError(RuntimeError):
    """A controlled failure carrying only a safe stage identifier."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


class TerminationRequested(KeyboardInterrupt):
    """Raised for SIGTERM/SIGHUP so Python unwinds through instance teardown."""


@dataclass(frozen=True)
class PublicationConfig:
    archive: Path
    checksum: Path
    session_script: Path
    output: Path
    receipt: Path
    image: str
    gpu_name: str = "RTX 4090"
    max_hourly_usd: float = 0.75
    max_total_usd: float = 4.00
    max_instance_seconds: float = 4.0 * 3600.0
    min_reliability: float = 0.98
    disk_gb: float = 50.0
    min_cuda: float = 12.4
    min_direct_ports: int = 1
    offer_limit: int = 50
    ssh_ready_timeout_seconds: float = 15.0 * 60.0
    ssh_poll_seconds: float = 10.0
    ssh_connect_timeout_seconds: float = 15.0
    transfer_timeout_seconds: float = 15.0 * 60.0
    retrieval_reserve_seconds: float = 10.0 * 60.0
    retrieval_attempts: int = 3
    retrieval_retry_seconds: float = 5.0
    destroy_attempt_timeout_seconds: float = 20.0
    destroy_attempts: int = 3
    ssh_user: str = "root"
    ssh_identity: Path | None = None
    remote_output_name: str = "publication-output.tar.gz"
    vastai_bin: str = "vastai"
    ssh_bin: str = "ssh"
    scp_bin: str = "scp"
    dry_run: bool = False
    confirm_spend: bool = False


@dataclass(frozen=True)
class InputRecord:
    archive_sha256: str
    archive_size: int
    checksum_sha256: str
    checksum_size: int
    session_sha256: str
    session_size: int
    adapter_contract: "AdapterContract"


@dataclass(frozen=True)
class AdapterContract:
    archive_sha256: str
    source_tree_sha256: str
    driver_sha256: str
    git_commit: str
    git_tree: str
    container_digest: str


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int


def default_executor(
    argv: Sequence[str], timeout: float | None
) -> subprocess.CompletedProcess[str]:
    """Run an argv vector without a shell and retain output privately."""

    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def sha256_file(path: Path, *, stage: str = "preflight") -> str:
    try:
        candidate = path.expanduser().resolve(strict=True)
        before = candidate.stat()
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = candidate.stat()
    except OSError as exc:
        raise PublicationError(stage, "file could not be hashed") from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise PublicationError(stage, "file changed while it was being hashed")
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_symlink():
        raise PublicationError("preflight", f"{label} must be a regular non-symlink file")
    try:
        candidate = expanded.resolve(strict=True)
    except OSError as exc:
        raise PublicationError(
            "preflight", f"{label} must be a regular non-symlink file"
        ) from exc
    if not candidate.is_file():
        raise PublicationError("preflight", f"{label} must be a regular non-symlink file")
    return candidate


def _adapter_contract(path: Path) -> AdapterContract:
    if path.stat().st_size > 1024 * 1024:
        raise PublicationError("preflight", "session adapter exceeds the size limit")
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PublicationError("preflight", "session adapter must be readable ASCII") from exc
    assignments: dict[str, str] = {}
    for line in lines:
        for name in ADAPTER_ASSIGNMENTS:
            if not line.startswith(name + "="):
                continue
            if name in assignments:
                raise PublicationError(
                    "preflight", f"session adapter repeats contract field {name}"
                )
            try:
                tokens = shlex.split(line, posix=True)
            except ValueError as exc:
                raise PublicationError(
                    "preflight", f"session adapter has invalid contract field {name}"
                ) from exc
            if len(tokens) != 1 or not tokens[0].startswith(name + "="):
                raise PublicationError(
                    "preflight", f"session adapter has invalid contract field {name}"
                )
            assignments[name] = tokens[0][len(name) + 1:]
    missing = sorted(set(ADAPTER_ASSIGNMENTS) - set(assignments))
    if missing:
        raise PublicationError(
            "preflight", f"session adapter lacks contract fields: {', '.join(missing)}"
        )
    for name in (
        "EXPECTED_ARCHIVE_SHA256",
        "EXPECTED_SOURCE_TREE_SHA256",
        "EXPECTED_DRIVER_SHA256",
    ):
        if not HEX_SHA256_RE.fullmatch(assignments[name]):
            raise PublicationError(
                "preflight", f"session adapter has invalid contract field {name}"
            )
    for name in ("GIT_COMMIT", "GIT_TREE"):
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", assignments[name]):
            raise PublicationError(
                "preflight", f"session adapter has invalid contract field {name}"
            )
    if not IMAGE_DIGEST_RE.fullmatch(assignments["CONTAINER_DIGEST"]):
        raise PublicationError(
            "preflight", "session adapter has invalid contract field CONTAINER_DIGEST"
        )
    return AdapterContract(
        archive_sha256=assignments["EXPECTED_ARCHIVE_SHA256"],
        source_tree_sha256=assignments["EXPECTED_SOURCE_TREE_SHA256"],
        driver_sha256=assignments["EXPECTED_DRIVER_SHA256"],
        git_commit=assignments["GIT_COMMIT"],
        git_tree=assignments["GIT_TREE"],
        container_digest=assignments["CONTAINER_DIGEST"],
    )


def _checksum_hash(path: Path) -> str:
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise PublicationError("preflight", "checksum file must be readable ASCII") from exc
    records = [line.strip() for line in text.splitlines() if line.strip()]
    if len(records) != 1:
        raise PublicationError("preflight", "checksum file must contain exactly one record")
    match = re.match(r"\A([0-9A-Fa-f]{64})(?:[ \t]+[* ]?.+)?\Z", records[0])
    if match is None:
        raise PublicationError("preflight", "checksum file does not contain a SHA-256 record")
    return match.group(1).lower()


def _exclusive_targets(config: PublicationConfig) -> None:
    output = config.output.absolute()
    receipt = config.receipt.absolute()
    if output == receipt:
        raise PublicationError("preflight", "output and receipt paths must differ")
    inputs = {
        config.archive.absolute(),
        config.checksum.absolute(),
        config.session_script.absolute(),
    }
    if output in inputs or receipt in inputs:
        raise PublicationError("preflight", "local artifact paths must not alias input files")
    for path, label in ((output, "output"), (receipt, "receipt")):
        if path.exists() or path.is_symlink():
            raise PublicationError("preflight", f"refusing to overwrite existing {label} path")
        path.parent.mkdir(parents=True, exist_ok=True)


def _validate_config(config: PublicationConfig) -> InputRecord:
    archive = _regular_file(config.archive, "archive")
    checksum = _regular_file(config.checksum, "checksum")
    session = _regular_file(config.session_script, "session script")
    if config.ssh_identity is not None:
        _regular_file(config.ssh_identity, "SSH identity")
    _exclusive_targets(config)

    if not IMAGE_DIGEST_RE.fullmatch(config.image):
        raise PublicationError(
            "preflight", "image must be pinned as repository@sha256:<64 lowercase hex>"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}", config.gpu_name):
        raise PublicationError("preflight", "GPU name contains unsupported characters")
    if not SAFE_SSH_USER_RE.fullmatch(config.ssh_user):
        raise PublicationError("preflight", "SSH user is invalid")
    if not SAFE_REMOTE_NAME_RE.fullmatch(config.remote_output_name):
        raise PublicationError("preflight", "remote output name is invalid")
    if config.remote_output_name in REMOTE_RESERVED_NAMES:
        raise PublicationError(
            "preflight", "remote output name collides with a publication-session path"
        )
    if Path(config.vastai_bin).name != "vastai":
        raise PublicationError("preflight", "the provider CLI executable must be vastai")

    positive = {
        "max hourly cost": config.max_hourly_usd,
        "max total cost": config.max_total_usd,
        "maximum instance lifetime": config.max_instance_seconds,
        "disk": config.disk_gb,
        "minimum CUDA": config.min_cuda,
        "SSH readiness timeout": config.ssh_ready_timeout_seconds,
        "SSH poll interval": config.ssh_poll_seconds,
        "SSH connection timeout": config.ssh_connect_timeout_seconds,
        "transfer timeout": config.transfer_timeout_seconds,
        "retrieval reserve": config.retrieval_reserve_seconds,
        "retrieval retry interval": config.retrieval_retry_seconds,
        "destroy timeout": config.destroy_attempt_timeout_seconds,
    }
    for label, value in positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise PublicationError("preflight", f"{label} must be finite and positive")
    if not 0.0 <= config.min_reliability <= 1.0:
        raise PublicationError("preflight", "minimum reliability must be in [0, 1]")
    if (
        config.min_direct_ports < 1
        or config.offer_limit < 1
        or config.retrieval_attempts < 1
        or config.destroy_attempts < 1
    ):
        raise PublicationError("preflight", "integer constraints must be positive")

    teardown_reserve = (
        config.destroy_attempt_timeout_seconds * config.destroy_attempts
    )
    if config.max_instance_seconds <= teardown_reserve + config.retrieval_reserve_seconds:
        raise PublicationError(
            "preflight",
            "maximum instance lifetime must exceed retrieval and teardown reserves",
        )
    if not config.dry_run and not config.confirm_spend:
        raise PublicationError(
            "preflight", "paid execution requires explicit --confirm-spend"
        )

    archive_hash = sha256_file(archive)
    expected_hash = _checksum_hash(checksum)
    if archive_hash != expected_hash:
        raise PublicationError("preflight", "archive SHA-256 does not match checksum file")
    contract = _adapter_contract(session)
    if contract.archive_sha256 != archive_hash:
        raise PublicationError(
            "preflight", "session adapter is not bound to the supplied archive"
        )
    if contract.container_digest != config.image:
        raise PublicationError(
            "preflight", "session adapter container digest differs from the launch image"
        )

    return InputRecord(
        archive_sha256=archive_hash,
        archive_size=archive.stat().st_size,
        checksum_sha256=sha256_file(checksum),
        checksum_size=checksum.stat().st_size,
        session_sha256=sha256_file(session),
        session_size=session.stat().st_size,
        adapter_contract=contract,
    )


def effective_hourly_ceiling(config: PublicationConfig) -> float:
    lifetime_hours = config.max_instance_seconds / 3600.0
    return min(config.max_hourly_usd, config.max_total_usd / lifetime_hours)


def build_offer_query(config: PublicationConfig) -> str:
    gpu = config.gpu_name.replace(" ", "_")
    hourly = effective_hourly_ceiling(config)
    return " ".join(
        (
            "external=false",
            "verified=true",
            "rentable=true",
            "rented=false",
            "num_gpus=1",
            f"gpu_name={gpu}",
            f"reliability>={config.min_reliability:.8g}",
            f"disk_space>={config.disk_gb:.8g}",
            f"cuda_vers>={config.min_cuda:.8g}",
            f"direct_port_count>={config.min_direct_ports}",
            f"dph<={hourly:.8g}",
        )
    )


def _finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return None
    return converted if converted > 0 else None


def _normalized_gpu(value: Any) -> str:
    return re.sub(r"[ _-]+", "", str(value)).upper()


def _offer_is_verified(offer: Mapping[str, Any]) -> bool:
    if offer.get("verified") is True:
        return True
    return str(offer.get("verification", "")).strip().lower() == "verified"


def offer_satisfies(offer: Mapping[str, Any], config: PublicationConfig) -> bool:
    price = _finite_float(offer.get("dph_total"))
    reliability = _finite_float(offer.get("reliability"))
    disk = _finite_float(offer.get("disk_space"))
    cuda = _finite_float(offer.get("cuda_max_good", offer.get("cuda_vers")))
    direct_ports = _positive_int(offer.get("direct_port_count"))
    num_gpus = _finite_float(offer.get("num_gpus"))
    offer_id = _positive_int(offer.get("id"))
    if None in (price, reliability, disk, cuda, direct_ports, num_gpus, offer_id):
        return False
    if not _offer_is_verified(offer):
        return False
    if offer.get("rentable") is not True or offer.get("rented") is not False:
        return False
    if offer.get("external") is True:
        return False
    if num_gpus != 1.0:
        return False
    if _normalized_gpu(offer.get("gpu_name")) != _normalized_gpu(config.gpu_name):
        return False
    return bool(
        price > 0.0
        and price <= effective_hourly_ceiling(config)
        and reliability >= config.min_reliability
        and disk >= config.disk_gb
        and cuda >= config.min_cuda
        and direct_ports >= config.min_direct_ports
    )


def select_offer(
    offers: Sequence[Mapping[str, Any]], config: PublicationConfig
) -> Mapping[str, Any]:
    eligible = [offer for offer in offers if offer_satisfies(offer, config)]
    if not eligible:
        raise PublicationError("search", "no offer passed all client-side constraints")
    return min(
        eligible,
        key=lambda offer: (
            float(offer["dph_total"]),
            -float(offer["reliability"]),
            int(offer["id"]),
        ),
    )


def _strict_json(text: str, stage: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value}")

    try:
        return json.loads(text, parse_constant=reject_constant)
    except (TypeError, json.JSONDecodeError, ValueError) as exc:
        raise PublicationError(stage, f"{stage} did not return strict JSON") from exc


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise PublicationError("receipt", "receipt contains non-JSON data") from exc
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise PublicationError("receipt", "refusing to overwrite receipt") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _safe_offer(offer: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if offer is None:
        return None
    fields = (
        "id",
        "machine_id",
        "gpu_name",
        "num_gpus",
        "reliability",
        "verified",
        "verification",
        "rentable",
        "rented",
        "disk_space",
        "cuda_max_good",
        "direct_port_count",
        "dph_total",
        "geolocation",
    )
    return {field: offer[field] for field in fields if field in offer}


def _safe_instance(
    instance_id: int | None,
    label: str,
    row: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if instance_id is None:
        return None
    record: dict[str, Any] = {"id": instance_id, "label": label}
    if row is not None:
        for field in (
            "actual_status",
            "gpu_name",
            "num_gpus",
            "dph_total",
            "start_date",
        ):
            if field in row:
                record[field] = row[field]
    return record


class VastPublicationRunner:
    def __init__(
        self,
        config: PublicationConfig,
        *,
        executor: CommandExecutor = default_executor,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        token_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ):
        self.config = config
        self.executor = executor
        self.monotonic = monotonic
        self.sleep = sleep
        self.utcnow = utcnow
        self.token_factory = token_factory
        self.stage = "preflight"

    def _run_command(
        self,
        argv: Sequence[str],
        *,
        timeout: float | None,
        stage: str,
        require_success: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        if "--api-key" in argv or "--api_key" in argv:
            raise PublicationError(stage, "API-key arguments are forbidden")
        try:
            result = self.executor(tuple(str(item) for item in argv), timeout)
        except subprocess.TimeoutExpired as exc:
            raise PublicationError(stage, f"{stage} command timed out") from exc
        except OSError as exc:
            raise PublicationError(stage, f"{stage} command could not start") from exc
        if require_success and result.returncode != 0:
            raise PublicationError(stage, f"{stage} command failed with exit {result.returncode}")
        return result

    def _json_command(
        self,
        argv: Sequence[str],
        *,
        timeout: float | None,
        stage: str,
    ) -> Any:
        result = self._run_command(argv, timeout=timeout, stage=stage)
        return _strict_json(result.stdout, stage)

    def _search(self) -> Mapping[str, Any]:
        query = build_offer_query(self.config)
        payload = self._json_command(
            (
                self.config.vastai_bin,
                "--raw",
                "search",
                "offers",
                query,
                "--type",
                "on-demand",
                "--limit",
                str(self.config.offer_limit),
                "--storage",
                str(self.config.disk_gb),
                "--order",
                "dph",
            ),
            timeout=60.0,
            stage="search",
        )
        if isinstance(payload, dict):
            payload = payload.get("offers")
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise PublicationError("search", "offer search returned an invalid JSON shape")
        return select_offer(payload, self.config)

    def _show_all_instances(self) -> list[Mapping[str, Any]]:
        payload = self._json_command(
            (self.config.vastai_bin, "--raw", "show", "instances"),
            timeout=30.0,
            stage="instance_recovery",
        )
        if isinstance(payload, dict):
            payload = payload.get("instances")
        if not isinstance(payload, list):
            raise PublicationError("instance_recovery", "instance list has invalid JSON shape")
        return [item for item in payload if isinstance(item, dict)]

    def _recover_instance_id(self, label: str) -> int | None:
        try:
            matches = [
                row for row in self._show_all_instances() if row.get("label") == label
            ]
        except BaseException:
            return None
        ids = {_positive_int(row.get("id")) for row in matches}
        ids.discard(None)
        return next(iter(ids)) if len(ids) == 1 else None

    def _show_instance(
        self, instance_id: int, timeout: float
    ) -> Mapping[str, Any]:
        payload = self._json_command(
            (
                self.config.vastai_bin,
                "--raw",
                "show",
                "instance",
                str(instance_id),
            ),
            timeout=timeout,
            stage="ssh_wait",
        )
        if isinstance(payload, dict) and set(payload) == {"instances"}:
            payload = payload["instances"]
        if not isinstance(payload, dict):
            raise PublicationError("ssh_wait", "instance inspection has invalid JSON shape")
        payload_id = payload.get("id")
        if payload_id is not None and _positive_int(payload_id) != instance_id:
            raise PublicationError("ssh_wait", "instance inspection returned the wrong ID")
        return payload

    def _create_argv(self, offer_id: int, label: str) -> tuple[str, ...]:
        return (
            self.config.vastai_bin,
            "--raw",
            "create",
            "instance",
            str(offer_id),
            "--image",
            self.config.image,
            "--disk",
            str(self.config.disk_gb),
            "--label",
            label,
            "--ssh",
            "--direct",
            "--cancel-unavail",
        )

    def _ssh_options(self, known_hosts: Path) -> list[str]:
        options = [
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={known_hosts}",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ServerAliveInterval=15",
        ]
        if self.config.ssh_identity is not None:
            options.extend(("-i", str(self.config.ssh_identity)))
        return options

    def _ssh_argv(
        self,
        endpoint: Endpoint,
        known_hosts: Path,
        remote_command: str,
    ) -> tuple[str, ...]:
        return tuple(
            [
                self.config.ssh_bin,
                *self._ssh_options(known_hosts),
                "-o",
                f"ConnectTimeout={max(1, int(self.config.ssh_connect_timeout_seconds))}",
                "-p",
                str(endpoint.port),
                f"{self.config.ssh_user}@{endpoint.host}",
                remote_command,
            ]
        )

    def _scp_argv(
        self,
        endpoint: Endpoint,
        known_hosts: Path,
        source: str,
        destination: str,
    ) -> tuple[str, ...]:
        return tuple(
            [
                self.config.scp_bin,
                "-q",
                *self._ssh_options(known_hosts),
                "-P",
                str(endpoint.port),
                "--",
                source,
                destination,
            ]
        )

    def _endpoint_from_row(self, row: Mapping[str, Any]) -> Endpoint | None:
        host = row.get("ssh_host")
        port = _positive_int(row.get("ssh_port"))
        if not isinstance(host, str) or not SAFE_SSH_HOST_RE.fullmatch(host):
            return None
        if port is None or port > 65535:
            return None
        return Endpoint(host=host, port=port)

    def _wait_for_ssh(
        self,
        instance_id: int,
        known_hosts: Path,
        work_deadline: float,
    ) -> tuple[Endpoint, Mapping[str, Any]]:
        ready_deadline = min(
            work_deadline,
            self.monotonic() + self.config.ssh_ready_timeout_seconds,
        )
        last_row: Mapping[str, Any] | None = None
        while self.monotonic() < ready_deadline:
            try:
                remaining = ready_deadline - self.monotonic()
                if remaining <= 0.0:
                    break
                last_row = self._show_instance(
                    instance_id, min(30.0, remaining)
                )
                endpoint = self._endpoint_from_row(last_row)
                if endpoint is not None:
                    remaining = ready_deadline - self.monotonic()
                    if remaining <= 0.0:
                        break
                    result = self._run_command(
                        self._ssh_argv(endpoint, known_hosts, "true"),
                        timeout=min(
                            remaining,
                            self.config.ssh_connect_timeout_seconds + 5.0,
                        ),
                        stage="ssh_probe",
                        require_success=False,
                    )
                    if result.returncode == 0:
                        return endpoint, last_row
            except PublicationError:
                pass
            remaining = ready_deadline - self.monotonic()
            if remaining <= 0.0:
                break
            self.sleep(min(self.config.ssh_poll_seconds, remaining))
        raise PublicationError("ssh_wait", "SSH did not become ready before the deadline")

    def _remaining(self, deadline: float, stage: str) -> float:
        remaining = deadline - self.monotonic()
        if remaining <= 0.0:
            raise PublicationError(stage, "paid work deadline exhausted")
        return remaining

    def _upload(
        self,
        endpoint: Endpoint,
        known_hosts: Path,
        remote_dir: str,
        work_deadline: float,
        inputs: InputRecord,
    ) -> None:
        if sha256_file(self.config.archive, stage="upload") != inputs.archive_sha256:
            raise PublicationError("upload", "archive changed after preflight")
        if sha256_file(self.config.checksum, stage="upload") != inputs.checksum_sha256:
            raise PublicationError("upload", "checksum changed after preflight")
        if sha256_file(self.config.session_script, stage="upload") != inputs.session_sha256:
            raise PublicationError("upload", "session script changed after preflight")

        self._run_command(
            self._ssh_argv(
                endpoint,
                known_hosts,
                shlex.join(("mkdir", "-p", "--", remote_dir)),
            ),
            timeout=min(
                self._remaining(work_deadline, "upload"),
                self.config.transfer_timeout_seconds,
            ),
            stage="upload",
        )
        target = f"{self.config.ssh_user}@{endpoint.host}"
        uploads = (
            (self.config.archive, REMOTE_ARCHIVE),
            (self.config.checksum, REMOTE_CHECKSUM),
            (self.config.session_script, REMOTE_SESSION),
        )
        for source, remote_name in uploads:
            canonical_source = source.expanduser().resolve()
            self._run_command(
                self._scp_argv(
                    endpoint,
                    known_hosts,
                    str(canonical_source),
                    f"{target}:{remote_dir}/{remote_name}",
                ),
                timeout=min(
                    self._remaining(work_deadline, "upload"),
                    self.config.transfer_timeout_seconds,
                ),
                stage="upload",
            )

    def _execute_remote(
        self,
        endpoint: Endpoint,
        known_hosts: Path,
        remote_dir: str,
        work_deadline: float,
    ) -> subprocess.CompletedProcess[str]:
        command = " && ".join(
            (
                "set -o pipefail",
                f"cd {shlex.quote(remote_dir)}",
                shlex.join(
                    (
                        "bash",
                        REMOTE_SESSION,
                        REMOTE_ARCHIVE,
                        REMOTE_CHECKSUM,
                        self.config.remote_output_name,
                    )
                ),
                shlex.join(("test", "-f", self.config.remote_output_name)),
            )
        )
        return self._run_command(
            self._ssh_argv(endpoint, known_hosts, f"bash -lc {shlex.quote(command)}"),
            timeout=self._remaining(work_deadline, "execute"),
            stage="execute",
            require_success=False,
        )

    def _retrieve_once(
        self,
        endpoint: Endpoint,
        known_hosts: Path,
        remote_dir: str,
        deadline: float,
    ) -> dict[str, Any]:
        parent = self.config.output.parent
        temporary_directory = Path(
            tempfile.mkdtemp(prefix=".vast-publication-", dir=parent)
        )
        temporary_output = temporary_directory / "retrieved-output"
        try:
            source = (
                f"{self.config.ssh_user}@{endpoint.host}:"
                f"{remote_dir}/{self.config.remote_output_name}"
            )
            self._run_command(
                self._scp_argv(
                    endpoint,
                    known_hosts,
                    source,
                    str(temporary_output),
                ),
                timeout=min(
                    self._remaining(deadline, "retrieve"),
                    self.config.transfer_timeout_seconds,
                ),
                stage="retrieve",
            )
            if temporary_output.is_symlink() or not temporary_output.is_file():
                raise PublicationError("retrieve", "retrieval produced no regular output file")
            digest = sha256_file(temporary_output, stage="retrieve")
            size = temporary_output.stat().st_size
            try:
                os.link(temporary_output, self.config.output)
            except FileExistsError as exc:
                raise PublicationError("retrieve", "refusing to overwrite local output") from exc
            return {
                "retrieved": True,
                "filename": self.config.output.name,
                "sha256": digest,
                "size": size,
            }
        finally:
            shutil.rmtree(temporary_directory, ignore_errors=True)

    def _retrieve(
        self,
        endpoint: Endpoint,
        known_hosts: Path,
        remote_dir: str,
        deadline: float,
    ) -> dict[str, Any]:
        last_error: PublicationError | None = None
        for attempt in range(1, self.config.retrieval_attempts + 1):
            try:
                return self._retrieve_once(endpoint, known_hosts, remote_dir, deadline)
            except PublicationError as exc:
                last_error = exc
                if attempt == self.config.retrieval_attempts:
                    break
                remaining = deadline - self.monotonic()
                if remaining <= 0.0:
                    break
                self.sleep(min(self.config.retrieval_retry_seconds, remaining))
        if last_error is not None:
            raise last_error
        raise PublicationError("retrieve", "retrieval exhausted without an attempt")

    def _destroy(self, instance_id: int) -> dict[str, Any]:
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, self.config.destroy_attempts + 1):
            record: dict[str, Any] = {"attempt": attempt, "success": False}
            try:
                result = self._run_command(
                    (
                        self.config.vastai_bin,
                        "--raw",
                        "destroy",
                        "instance",
                        str(instance_id),
                    ),
                    timeout=self.config.destroy_attempt_timeout_seconds,
                    stage="destroy",
                    require_success=False,
                )
                record["returncode"] = result.returncode
                if result.returncode == 0:
                    try:
                        payload = _strict_json(result.stdout, "destroy")
                    except PublicationError:
                        payload = None
                    if isinstance(payload, dict) and payload.get("success") is True:
                        record["success"] = True
                        attempts.append(record)
                        return {"attempted": True, "succeeded": True, "attempts": attempts}
            except BaseException as exc:
                record["error_type"] = type(exc).__name__
            attempts.append(record)
        return {"attempted": True, "succeeded": False, "attempts": attempts}

    def _dry_run_receipt(
        self,
        inputs: InputRecord,
        label: str,
        started_utc: datetime,
    ) -> dict[str, Any]:
        return self._receipt(
            status="dry_run",
            inputs=inputs,
            label=label,
            started_utc=started_utc,
            finished_utc=self.utcnow(),
            timings={"total_seconds": 0.0},
            offer=None,
            instance_id=None,
            instance_row=None,
            output=None,
            teardown={"attempted": False, "succeeded": None, "attempts": []},
            error=None,
        )

    def _receipt(
        self,
        *,
        status: str,
        inputs: InputRecord,
        label: str,
        started_utc: datetime,
        finished_utc: datetime,
        timings: Mapping[str, float],
        offer: Mapping[str, Any] | None,
        instance_id: int | None,
        instance_row: Mapping[str, Any] | None,
        output: Mapping[str, Any] | None,
        teardown: Mapping[str, Any],
        error: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        selected_rate = (
            _finite_float(offer.get("dph_total")) if offer is not None else None
        )
        total_seconds = float(timings.get("total_seconds", 0.0))
        return {
            "schema": "gbskernels.vast-publication-receipt.v1",
            "status": status,
            "dry_run": self.config.dry_run,
            "started_utc": started_utc.astimezone(timezone.utc).isoformat(),
            "finished_utc": finished_utc.astimezone(timezone.utc).isoformat(),
            "inputs": {
                "archive": {
                    "filename": self.config.archive.name,
                    "sha256": inputs.archive_sha256,
                    "size": inputs.archive_size,
                },
                "checksum": {
                    "filename": self.config.checksum.name,
                    "sha256": inputs.checksum_sha256,
                    "size": inputs.checksum_size,
                },
                "session_script": {
                    "filename": self.config.session_script.name,
                    "sha256": inputs.session_sha256,
                    "size": inputs.session_size,
                    "contract": {
                        "source_archive_sha256": inputs.adapter_contract.archive_sha256,
                        "source_tree_sha256": inputs.adapter_contract.source_tree_sha256,
                        "driver_sha256": inputs.adapter_contract.driver_sha256,
                        "git_commit": inputs.adapter_contract.git_commit,
                        "git_tree": inputs.adapter_contract.git_tree,
                        "container_digest": inputs.adapter_contract.container_digest,
                    },
                },
            },
            "plan": {
                "pricing_type": "on-demand",
                "search_query": build_offer_query(self.config),
                "image": self.config.image,
                "launch": {
                    "ssh": True,
                    "direct": True,
                    "cancel_unavailable": True,
                    "gpu_count": 1,
                    "disk_gb": self.config.disk_gb,
                },
                "label": label,
            },
            "constraints": {
                "gpu_name": self.config.gpu_name,
                "minimum_reliability": self.config.min_reliability,
                "minimum_cuda": self.config.min_cuda,
                "minimum_disk_gb": self.config.disk_gb,
                "minimum_direct_ports": self.config.min_direct_ports,
                "retrieval_reserve_seconds": self.config.retrieval_reserve_seconds,
                "retrieval_attempts": self.config.retrieval_attempts,
                "retrieval_retry_seconds": self.config.retrieval_retry_seconds,
            },
            "cost_ceiling": {
                "max_hourly_usd": self.config.max_hourly_usd,
                "effective_hourly_usd": effective_hourly_ceiling(self.config),
                "max_total_usd": self.config.max_total_usd,
                "max_instance_seconds": self.config.max_instance_seconds,
                "selected_hourly_usd": selected_rate,
                "selected_maximum_projected_usd": (
                    selected_rate * self.config.max_instance_seconds / 3600.0
                    if selected_rate is not None
                    else None
                ),
                "elapsed_cost_estimate_usd": (
                    selected_rate * total_seconds / 3600.0
                    if selected_rate is not None
                    else None
                ),
                "scope": "offer dph_total multiplied by bounded instance lifetime",
                "billing_note": (
                    "elapsed estimate only; variable network charges are excluded and "
                    "the provider invoice is authoritative"
                ),
            },
            "offer": _safe_offer(offer),
            "instance": _safe_instance(instance_id, label, instance_row),
            "timing": dict(timings),
            "output": dict(output) if output is not None else None,
            "teardown": dict(teardown),
            "error": dict(error) if error is not None else None,
        }

    def run(self) -> tuple[dict[str, Any], Path]:
        inputs = _validate_config(self.config)
        started_utc = self.utcnow()
        started_mono = self.monotonic()
        token = re.sub(r"[^a-z0-9]", "", self.token_factory().lower())[:20]
        if len(token) < 8:
            raise PublicationError("preflight", "run token is too short")
        label = f"gbskernels-publication-{token}"

        if self.config.dry_run:
            receipt = self._dry_run_receipt(inputs, label, started_utc)
            _exclusive_json(self.config.receipt, receipt)
            return receipt, self.config.receipt

        offer: Mapping[str, Any] | None = None
        instance_id: int | None = None
        instance_row: Mapping[str, Any] | None = None
        endpoint: Endpoint | None = None
        output: dict[str, Any] | None = None
        primary_error: BaseException | None = None
        primary_error_stage: str | None = None
        teardown: dict[str, Any] = {
            "attempted": False,
            "succeeded": None,
            "attempts": [],
        }
        timings: dict[str, float] = {}
        retrieval_attempted = False
        upload_completed = False
        remote_dir = f"/tmp/{label}"
        instance_deadline: float | None = None
        work_deadline: float | None = None
        retrieval_deadline: float | None = None

        with tempfile.TemporaryDirectory(prefix="gbskernels-vast-ssh-") as temporary:
            known_hosts = Path(temporary) / "known_hosts"
            try:
                self.stage = "search"
                phase = self.monotonic()
                offer = self._search()
                timings["search_seconds"] = self.monotonic() - phase

                self.stage = "create"
                offer_id = int(offer["id"])
                phase = self.monotonic()
                instance_deadline = phase + self.config.max_instance_seconds
                teardown_reserve = (
                    self.config.destroy_attempt_timeout_seconds
                    * self.config.destroy_attempts
                )
                retrieval_deadline = instance_deadline - teardown_reserve
                work_deadline = (
                    retrieval_deadline - self.config.retrieval_reserve_seconds
                )
                create_attempted = True
                try:
                    payload = self._json_command(
                        self._create_argv(offer_id, label),
                        timeout=min(120.0, self._remaining(work_deadline, "create")),
                        stage="create",
                    )
                    if isinstance(payload, dict):
                        instance_id = _positive_int(
                            payload.get("new_contract", payload.get("instance_id"))
                        )
                    if not isinstance(payload, dict) or payload.get("success") is not True:
                        raise PublicationError("create", "provider did not confirm creation")
                    if instance_id is None:
                        raise PublicationError("create", "creation response omitted instance ID")
                except BaseException:
                    if create_attempted and instance_id is None:
                        instance_id = self._recover_instance_id(label)
                    raise
                finally:
                    timings["create_seconds"] = self.monotonic() - phase

                self.stage = "ssh_wait"
                phase = self.monotonic()
                endpoint, instance_row = self._wait_for_ssh(
                    instance_id, known_hosts, work_deadline
                )
                timings["ssh_ready_seconds"] = self.monotonic() - phase

                self.stage = "upload"
                phase = self.monotonic()
                self._upload(
                    endpoint,
                    known_hosts,
                    remote_dir,
                    work_deadline,
                    inputs,
                )
                upload_completed = True
                timings["upload_seconds"] = self.monotonic() - phase

                self.stage = "execute"
                phase = self.monotonic()
                execution = self._execute_remote(
                    endpoint,
                    known_hosts,
                    remote_dir,
                    work_deadline,
                )
                timings["execute_seconds"] = self.monotonic() - phase

                self.stage = "retrieve"
                phase = self.monotonic()
                retrieval_attempted = True
                try:
                    output = self._retrieve(
                        endpoint, known_hosts, remote_dir, retrieval_deadline
                    )
                except BaseException as exc:
                    if execution.returncode != 0:
                        raise PublicationError(
                            "execute",
                            f"remote session failed with exit {execution.returncode}",
                        ) from exc
                    raise
                timings["retrieve_seconds"] = self.monotonic() - phase
                if execution.returncode != 0:
                    raise PublicationError(
                        "execute", f"remote session failed with exit {execution.returncode}"
                    )
            except BaseException as exc:
                primary_error = exc
                primary_error_stage = (
                    exc.stage if isinstance(exc, PublicationError) else self.stage
                )
            finally:
                if (
                    instance_id is not None
                    and endpoint is not None
                    and upload_completed
                    and not retrieval_attempted
                    and not self.config.output.exists()
                    and retrieval_deadline is not None
                    and self.monotonic() < retrieval_deadline
                ):
                    self.stage = "retrieve_after_failure"
                    phase = self.monotonic()
                    retrieval_attempted = True
                    try:
                        output = self._retrieve(
                            endpoint,
                            known_hosts,
                            remote_dir,
                            retrieval_deadline,
                        )
                    except BaseException:
                        pass
                    timings["retrieve_after_failure_seconds"] = self.monotonic() - phase

                if instance_id is not None:
                    self.stage = "destroy"
                    phase = self.monotonic()
                    teardown = self._destroy(instance_id)
                    timings["destroy_seconds"] = self.monotonic() - phase

        if primary_error is None and not teardown.get("succeeded"):
            primary_error = PublicationError(
                "destroy", "instance destruction could not be confirmed"
            )
            primary_error_stage = "destroy"

        finished_utc = self.utcnow()
        timings["total_seconds"] = self.monotonic() - started_mono
        interrupted = isinstance(primary_error, KeyboardInterrupt)
        status = "interrupted" if interrupted else ("failed" if primary_error else "succeeded")
        error_record = (
            {"type": type(primary_error).__name__, "stage": primary_error_stage}
            if primary_error is not None
            else None
        )
        receipt = self._receipt(
            status=status,
            inputs=inputs,
            label=label,
            started_utc=started_utc,
            finished_utc=finished_utc,
            timings=timings,
            offer=offer,
            instance_id=instance_id,
            instance_row=instance_row,
            output=output,
            teardown=teardown,
            error=error_record,
        )
        _exclusive_json(self.config.receipt, receipt)
        if primary_error is not None:
            raise primary_error
        return receipt, self.config.receipt


@contextlib.contextmanager
def termination_signals():
    """Translate process-termination signals into unwindable interrupts."""

    watched = [signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        watched.append(signal.SIGHUP)
    previous: dict[signal.Signals, Any] = {}

    def terminate(signum: int, frame: Any) -> None:
        del frame
        raise TerminationRequested(f"received signal {signum}")

    try:
        for watched_signal in watched:
            previous[watched_signal] = signal.getsignal(watched_signal)
            signal.signal(watched_signal, terminate)
        yield
    finally:
        for watched_signal, handler in previous.items():
            signal.signal(watched_signal, handler)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checksum", type=Path, required=True)
    parser.add_argument("--session-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpu-name", default="RTX 4090")
    parser.add_argument("--max-hourly-usd", type=float, default=0.75)
    parser.add_argument("--max-total-usd", type=float, default=4.0)
    parser.add_argument("--max-instance-seconds", type=float, default=4.0 * 3600.0)
    parser.add_argument("--min-reliability", type=float, default=0.98)
    parser.add_argument("--disk-gb", type=float, default=50.0)
    parser.add_argument("--min-cuda", type=float, default=12.4)
    parser.add_argument("--min-direct-ports", type=int, default=1)
    parser.add_argument("--offer-limit", type=int, default=50)
    parser.add_argument("--ssh-ready-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--ssh-poll-seconds", type=float, default=10.0)
    parser.add_argument("--ssh-connect-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--transfer-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--retrieval-reserve-seconds", type=float, default=600.0)
    parser.add_argument("--retrieval-attempts", type=int, default=3)
    parser.add_argument("--retrieval-retry-seconds", type=float, default=5.0)
    parser.add_argument("--destroy-attempt-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--destroy-attempts", type=int, default=3)
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--ssh-identity", type=Path, default=None)
    parser.add_argument("--remote-output-name", default="publication-output.tar.gz")
    parser.add_argument("--vastai-bin", default="vastai")
    parser.add_argument("--ssh-bin", default="ssh")
    parser.add_argument("--scp-bin", default="scp")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-spend", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> PublicationConfig:
    return PublicationConfig(
        archive=args.archive,
        checksum=args.checksum,
        session_script=args.session_script,
        output=args.output,
        receipt=args.receipt,
        image=args.image,
        gpu_name=args.gpu_name,
        max_hourly_usd=args.max_hourly_usd,
        max_total_usd=args.max_total_usd,
        max_instance_seconds=args.max_instance_seconds,
        min_reliability=args.min_reliability,
        disk_gb=args.disk_gb,
        min_cuda=args.min_cuda,
        min_direct_ports=args.min_direct_ports,
        offer_limit=args.offer_limit,
        ssh_ready_timeout_seconds=args.ssh_ready_timeout_seconds,
        ssh_poll_seconds=args.ssh_poll_seconds,
        ssh_connect_timeout_seconds=args.ssh_connect_timeout_seconds,
        transfer_timeout_seconds=args.transfer_timeout_seconds,
        retrieval_reserve_seconds=args.retrieval_reserve_seconds,
        retrieval_attempts=args.retrieval_attempts,
        retrieval_retry_seconds=args.retrieval_retry_seconds,
        destroy_attempt_timeout_seconds=args.destroy_attempt_timeout_seconds,
        destroy_attempts=args.destroy_attempts,
        ssh_user=args.ssh_user,
        ssh_identity=args.ssh_identity,
        remote_output_name=args.remote_output_name,
        vastai_bin=args.vastai_bin,
        ssh_bin=args.ssh_bin,
        scp_bin=args.scp_bin,
        dry_run=args.dry_run,
        confirm_spend=args.confirm_spend,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = config_from_args(args)
    runner = VastPublicationRunner(config)
    try:
        with termination_signals():
            receipt, path = runner.run()
    except KeyboardInterrupt:
        print("publication run interrupted; teardown was attempted", file=sys.stderr)
        return 130
    except PublicationError as exc:
        print(f"publication run failed at {exc.stage}", file=sys.stderr)
        return 1
    print(f"publication run {receipt['status']}; receipt: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
