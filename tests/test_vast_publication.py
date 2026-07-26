from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.vast_publication import (
    PublicationConfig,
    PublicationError,
    VastPublicationRunner,
    build_offer_query,
)


IMAGE = "nvidia/cuda@sha256:" + "c" * 64


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class FakeExecutor:
    def __init__(
        self,
        clock,
        *,
        interrupt_stage=None,
        interrupt_create=False,
        never_ready=False,
        retrieve_failures=0,
    ):
        self.clock = clock
        self.interrupt_stage = interrupt_stage
        self.interrupt_create = interrupt_create
        self.never_ready = never_ready
        self.retrieve_failures = retrieve_failures
        self.retrieval_reads = 0
        self.calls = []
        self.label = None

    @staticmethod
    def _result(argv, returncode=0, payload=None):
        stdout = "" if payload is None else json.dumps(payload, allow_nan=False)
        return subprocess.CompletedProcess(argv, returncode, stdout, "provider secret")

    def __call__(self, argv, timeout):
        argv = tuple(argv)
        self.calls.append((argv, timeout))
        self.clock.advance(0.25)

        if argv[:4] == ("vastai", "--raw", "search", "offers"):
            offers = [
                {
                    "id": 10,
                    "machine_id": 100,
                    "gpu_name": "RTX 4090",
                    "num_gpus": 1,
                    "reliability": 0.999,
                    "verified": False,
                    "rentable": True,
                    "rented": False,
                    "external": False,
                    "disk_space": 100,
                    "cuda_max_good": 12.8,
                    "direct_port_count": 4,
                    "dph_total": 0.20,
                },
                {
                    "id": 11,
                    "machine_id": 101,
                    "gpu_name": "RTX 4090",
                    "num_gpus": 1,
                    "reliability": 0.995,
                    "verified": True,
                    "rentable": True,
                    "rented": False,
                    "external": False,
                    "disk_space": 100,
                    "cuda_max_good": 12.8,
                    "direct_port_count": 4,
                    "dph_total": 0.50,
                },
                {
                    "id": 12,
                    "machine_id": 102,
                    "gpu_name": "RTX 4090",
                    "num_gpus": 1,
                    "reliability": 0.99,
                    "verified": True,
                    "rentable": True,
                    "rented": False,
                    "external": False,
                    "disk_space": 100,
                    "cuda_max_good": 12.8,
                    "direct_port_count": 4,
                    "dph_total": 0.40,
                    "api_key": "provider secret",
                },
            ]
            return self._result(argv, payload=offers)

        if argv[:4] == ("vastai", "--raw", "create", "instance"):
            self.label = argv[argv.index("--label") + 1]
            if self.interrupt_create:
                raise KeyboardInterrupt()
            return self._result(argv, payload={"success": True, "new_contract": 4321})

        if argv[:4] == ("vastai", "--raw", "show", "instances"):
            return self._result(
                argv,
                payload=[
                    {
                        "id": 4321,
                        "label": self.label,
                        "extra_env": [["KEY", "provider secret"]],
                    }
                ],
            )

        if argv[:4] == ("vastai", "--raw", "show", "instance"):
            if self.never_ready:
                return self._result(
                    argv,
                    payload={"id": 4321, "actual_status": "loading"},
                )
            return self._result(
                argv,
                payload={
                    "id": 4321,
                    "actual_status": "running",
                    "ssh_host": "203.0.113.7",
                    "ssh_port": 22022,
                    "gpu_name": "RTX 4090",
                    "num_gpus": 1,
                    "dph_total": 0.40,
                    "extra_env": [["TOKEN", "provider secret"]],
                },
            )

        if argv[:4] == ("vastai", "--raw", "destroy", "instance"):
            return self._result(argv, payload={"success": True})

        if argv[0] == "ssh":
            remote = argv[-1]
            if "publication-session.sh" in remote and self.interrupt_stage == "execute":
                raise KeyboardInterrupt()
            return self._result(argv)

        if argv[0] == "scp":
            source, destination = argv[-2:]
            if source.startswith("root@"):
                self.retrieval_reads += 1
                if (
                    self.interrupt_stage == "retrieve"
                    or self.retrieval_reads <= self.retrieve_failures
                ):
                    return self._result(argv, returncode=1)
                Path(destination).write_bytes(b"publication output\n")
            return self._result(argv)

        raise AssertionError(f"unexpected command: {argv}")


def _fixture(tmp_path, *, dry_run=False, confirm=True):
    archive = tmp_path / "release.tar"
    archive.write_bytes(b"immutable release archive\n")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = tmp_path / "release.tar.sha256"
    checksum.write_text(f"{digest}  release.tar\n", encoding="ascii")
    session = tmp_path / "publication.sh"
    session.write_text(
        "\n".join((
            "#!/bin/bash",
            f"EXPECTED_ARCHIVE_SHA256={digest}",
            f"EXPECTED_SOURCE_TREE_SHA256={'a' * 64}",
            f"EXPECTED_DRIVER_SHA256={'b' * 64}",
            f"GIT_COMMIT={'c' * 40}",
            f"GIT_TREE={'d' * 40}",
            f"CONTAINER_DIGEST={IMAGE}",
            "set -euo pipefail",
            'printf output > "$3"',
            "",
        )),
        encoding="ascii",
    )
    return PublicationConfig(
        archive=archive,
        checksum=checksum,
        session_script=session,
        output=tmp_path / "retrieved.tar.gz",
        receipt=tmp_path / "receipt.json",
        image=IMAGE,
        max_hourly_usd=0.60,
        max_total_usd=1.00,
        max_instance_seconds=600.0,
        min_reliability=0.98,
        disk_gb=50.0,
        min_cuda=12.4,
        ssh_ready_timeout_seconds=30.0,
        ssh_poll_seconds=1.0,
        retrieval_reserve_seconds=30.0,
        retrieval_retry_seconds=1.0,
        destroy_attempt_timeout_seconds=5.0,
        destroy_attempts=2,
        dry_run=dry_run,
        confirm_spend=confirm,
    )


def _runner(config, executor, clock):
    return VastPublicationRunner(
        config,
        executor=executor,
        monotonic=clock,
        sleep=clock.advance,
        utcnow=lambda: datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc),
        token_factory=lambda: "abcdef1234567890abcdef",
    )


def test_dry_run_writes_strict_plan_without_any_subprocess(tmp_path):
    config = _fixture(tmp_path, dry_run=True, confirm=False)
    calls = []

    def forbidden(argv, timeout):
        calls.append((argv, timeout))
        raise AssertionError("dry-run must not execute commands")

    clock = FakeClock()
    receipt, path = _runner(config, forbidden, clock).run()

    assert calls == []
    assert receipt["status"] == "dry_run"
    assert receipt["offer"] is None and receipt["instance"] is None
    assert receipt["teardown"]["attempted"] is False
    assert not config.output.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == receipt
    assert "verified=true" in build_offer_query(config)
    assert "num_gpus=1" in receipt["plan"]["search_query"]
    assert receipt["plan"]["launch"]["ssh"] is True
    assert receipt["plan"]["launch"]["direct"] is True
    assert "NaN" not in path.read_text(encoding="utf-8")
    assert "Infinity" not in path.read_text(encoding="utf-8")


def test_successful_lifecycle_selects_cheapest_valid_offer_and_destroys_exact_id(tmp_path):
    config = _fixture(tmp_path)
    clock = FakeClock()
    executor = FakeExecutor(clock)

    receipt, path = _runner(config, executor, clock).run()

    commands = [call[0] for call in executor.calls]
    create = next(command for command in commands if command[2:4] == ("create", "instance"))
    assert create[4] == "12"
    assert create[create.index("--image") + 1] == IMAGE
    assert {"--ssh", "--direct", "--cancel-unavail"} <= set(create)
    destroy = [command for command in commands if command[2:4] == ("destroy", "instance")]
    assert destroy == [("vastai", "--raw", "destroy", "instance", "4321")]
    assert all("--api-key" not in command and "--api_key" not in command for command in commands)

    scp_commands = [command for command in commands if command[0] == "scp"]
    assert len(scp_commands) == 4
    assert all("--" in command for command in scp_commands)
    assert all(
        Path(command[-2]).is_absolute()
        for command in scp_commands
        if not command[-2].startswith("root@")
    )
    assert config.output.read_bytes() == b"publication output\n"
    assert receipt["status"] == "succeeded"
    assert receipt["offer"]["id"] == 12
    assert receipt["instance"]["id"] == 4321
    assert receipt["teardown"]["succeeded"] is True
    assert receipt["output"]["sha256"] == hashlib.sha256(b"publication output\n").hexdigest()
    assert receipt["inputs"]["session_script"]["contract"]["container_digest"] == IMAGE
    assert receipt["cost_ceiling"]["selected_maximum_projected_usd"] <= 1.0
    serialized = path.read_text(encoding="utf-8")
    assert "provider secret" not in serialized
    assert json.loads(serialized) == receipt


def test_keyboard_interrupt_during_execution_still_retrieves_and_destroys(tmp_path):
    config = _fixture(tmp_path)
    clock = FakeClock()
    executor = FakeExecutor(clock, interrupt_stage="execute")

    with pytest.raises(KeyboardInterrupt):
        _runner(config, executor, clock).run()

    commands = [call[0] for call in executor.calls]
    assert ("vastai", "--raw", "destroy", "instance", "4321") in commands
    assert config.output.exists(), "best-effort retrieval should precede teardown"
    receipt = json.loads(config.receipt.read_text(encoding="utf-8"))
    assert receipt["status"] == "interrupted"
    assert receipt["error"] == {"stage": "execute", "type": "KeyboardInterrupt"}
    assert receipt["teardown"]["succeeded"] is True


def test_interrupt_during_ambiguous_create_recovers_label_and_destroys(tmp_path):
    config = _fixture(tmp_path)
    clock = FakeClock()
    executor = FakeExecutor(clock, interrupt_create=True)

    with pytest.raises(KeyboardInterrupt):
        _runner(config, executor, clock).run()

    commands = [call[0] for call in executor.calls]
    assert ("vastai", "--raw", "show", "instances") in commands
    assert ("vastai", "--raw", "destroy", "instance", "4321") in commands
    receipt = json.loads(config.receipt.read_text(encoding="utf-8"))
    assert receipt["instance"]["id"] == 4321
    assert receipt["status"] == "interrupted"
    assert receipt["teardown"]["succeeded"] is True


def test_retrieval_failure_records_failure_and_still_destroys(tmp_path):
    config = _fixture(tmp_path)
    clock = FakeClock()
    executor = FakeExecutor(clock, interrupt_stage="retrieve")

    with pytest.raises(PublicationError) as caught:
        _runner(config, executor, clock).run()

    assert caught.value.stage == "retrieve"
    commands = [call[0] for call in executor.calls]
    assert commands[-1] == ("vastai", "--raw", "destroy", "instance", "4321")
    receipt = json.loads(config.receipt.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["error"] == {"stage": "retrieve", "type": "PublicationError"}
    assert receipt["teardown"]["succeeded"] is True
    assert receipt["output"] is None
    assert not config.output.exists()


def test_transient_retrieval_failure_is_retried_before_teardown(tmp_path):
    config = _fixture(tmp_path)
    clock = FakeClock()
    executor = FakeExecutor(clock, retrieve_failures=1)

    receipt, _ = _runner(config, executor, clock).run()

    assert executor.retrieval_reads == 2
    assert receipt["status"] == "succeeded"
    assert receipt["teardown"]["succeeded"] is True
    assert config.output.read_bytes() == b"publication output\n"


def test_adapter_image_mismatch_refuses_before_any_command(tmp_path):
    config = replace(
        _fixture(tmp_path),
        image="nvidia/cuda@sha256:" + "e" * 64,
    )
    calls = []

    def forbidden(argv, timeout):
        calls.append((argv, timeout))
        raise AssertionError("contract mismatch must fail before provider commands")

    with pytest.raises(PublicationError, match="container digest"):
        _runner(config, forbidden, FakeClock()).run()
    assert calls == []


def test_reserved_remote_output_name_refuses_before_any_command(tmp_path):
    config = replace(_fixture(tmp_path), remote_output_name="bootstrap.log")
    calls = []

    def forbidden(argv, timeout):
        calls.append((argv, timeout))
        raise AssertionError("reserved output must fail before provider commands")

    with pytest.raises(PublicationError, match="collides"):
        _runner(config, forbidden, FakeClock()).run()
    assert calls == []


def test_ssh_readiness_wait_is_bounded_and_tears_down(tmp_path):
    config = replace(_fixture(tmp_path), ssh_ready_timeout_seconds=3.0)
    clock = FakeClock()
    executor = FakeExecutor(clock, never_ready=True)

    with pytest.raises(PublicationError) as caught:
        _runner(config, executor, clock).run()

    assert caught.value.stage == "ssh_wait"
    commands = [call[0] for call in executor.calls]
    show_calls = [command for command in commands if command[2:4] == ("show", "instance")]
    assert 1 <= len(show_calls) <= 4
    assert commands[-1] == ("vastai", "--raw", "destroy", "instance", "4321")
    receipt = json.loads(config.receipt.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["error"]["stage"] == "ssh_wait"
    assert receipt["teardown"]["succeeded"] is True


@pytest.mark.parametrize("existing", ["output", "receipt"])
def test_create_only_artifact_paths_refuse_before_any_command(tmp_path, existing):
    config = _fixture(tmp_path)
    target = config.output if existing == "output" else config.receipt
    target.write_bytes(b"do not overwrite")
    calls = []

    def forbidden(argv, timeout):
        calls.append((argv, timeout))
        raise AssertionError("preflight failure must not call subprocesses")

    with pytest.raises(PublicationError, match="overwrite"):
        _runner(config, forbidden, FakeClock()).run()
    assert calls == []
    assert target.read_bytes() == b"do not overwrite"
