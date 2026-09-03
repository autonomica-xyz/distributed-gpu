from __future__ import annotations

import json
import subprocess
import threading

import pytest

from gpu_orchestrator.agent import (
    AgentConfig,
    AgentError,
    AgentRunner,
    ExecutionError,
    _LeaseKeeper,
    _load_output_manifest,
    discover_nvidia_slots,
)
from gpu_orchestrator.models import AgentRegistration, SlotInventory


def test_nvidia_inventory_becomes_one_slot_per_gpu():
    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            [],
            0,
            stdout=(
                "0, GPU-a, NVIDIA RTX 4090, 24564, 8.9\n"
                "1, GPU-b, NVIDIA L4, 23034, 8.9\n"
            ),
            stderr="",
        )

    slots = discover_nvidia_slots(
        "tower-a",
        capabilities=["image-generate"],
        model_packs=["sdxl-core-v1"],
        runner=runner,
    )

    assert [slot.id for slot in slots] == ["tower-a-gpu-0", "tower-a-gpu-1"]
    assert slots[0].vram_mb == 24564
    assert slots[1].model == "NVIDIA L4"


def test_lease_keeper_renews_for_non_process_executors():
    heartbeat_seen = threading.Event()

    class Client:
        def heartbeat_job(self, *_args):
            heartbeat_seen.set()
            return False

    keeper = _LeaseKeeper(
        Client(),
        job_id="job-a",
        slot_id="slot-a",
        lease_token="secret",
        heartbeat_seconds=0.01,
        lease_seconds=1,
    )
    keeper.start()
    assert heartbeat_seen.wait(0.5)
    keeper.stop()
    keeper.check()


def test_lease_keeper_surfaces_cancel_and_renewal_failure():
    cancelled_seen = threading.Event()

    class CancelledClient:
        def heartbeat_job(self, *_args):
            cancelled_seen.set()
            return True

    cancelled = _LeaseKeeper(
        CancelledClient(),
        job_id="job-a",
        slot_id="slot-a",
        lease_token="secret",
        heartbeat_seconds=0.01,
        lease_seconds=1,
    )
    cancelled.start()
    assert cancelled_seen.wait(0.5)
    with pytest.raises(ExecutionError, match="cancellation") as exc_info:
        cancelled.check()
    assert exc_info.value.code == "cancelled"
    cancelled.stop()

    failure_seen = threading.Event()

    class FailedClient:
        def heartbeat_job(self, *_args):
            failure_seen.set()
            raise AgentError("control plane unavailable")

    failed = _LeaseKeeper(
        FailedClient(),
        job_id="job-b",
        slot_id="slot-b",
        lease_token="secret",
        heartbeat_seconds=0.01,
        lease_seconds=1,
    )
    failed.start()
    assert failure_seen.wait(0.5)
    with pytest.raises(ExecutionError, match="renewal failed") as exc_info:
        failed.check()
    assert exc_info.value.code == "worker_lost"
    failed.stop()


def test_output_manifest_cannot_escape_job_workdir(tmp_path):
    workdir = tmp_path / "job"
    workdir.mkdir()
    outside = tmp_path / "secret.json"
    outside.write_text("{}", encoding="utf-8")
    (workdir / "outputs.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "name": "result",
                        "path": "../secret.json",
                        "media_type": "application/json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ExecutionError, match="escapes workdir"):
        _load_output_manifest(workdir / "outputs.json", workdir)


def test_fatal_model_service_error_fails_job_then_drains_agent(tmp_path):
    slot = SlotInventory(
        id="agent-a-gpu-0", gpu_index=0, gpu_uuid="GPU-a", model="fake",
        vram_mb=8192, capabilities=["libretranslate"],
    )
    registration = AgentRegistration(
        agent_id="agent-a", provider="salad", hostname="host-a", slots=[slot]
    )

    class Client:
        def __init__(self):
            self.failed = []
            self.draining = []

        def confirm(self, *_args):
            return None

        def heartbeat_job(self, *_args):
            return False

        def fail(self, *args):
            self.failed.append(args)

        def heartbeat_agent(self, agent_id, *, draining=None):
            self.draining.append((agent_id, draining))

    client = Client()
    runner = AgentRunner(
        client,
        AgentConfig(registration=registration, workdir_root=tmp_path),
    )

    def unavailable(*_args, **_kwargs):
        raise ExecutionError("model_server_unavailable", "connection refused")

    runner._execute = unavailable
    runner._run_job(slot, {"id": "job-a", "lease_token": "lease"})

    assert client.failed[-1][-2:] == ("model_server_unavailable", "connection refused")
    assert client.draining == [("agent-a", True)]
    assert runner.stop.is_set()


def test_command_executor_uses_the_agent_python(tmp_path):
    slot = SlotInventory(
        id="agent-a-gpu-0",
        gpu_index=0,
        gpu_uuid="GPU-a",
        model="fake",
        vram_mb=1,
        capabilities=["demo-gpu"],
    )
    registration = AgentRegistration(
        agent_id="agent-a",
        provider="local",
        hostname="host-a",
        slots=[slot],
    )
    runner = AgentRunner(
        object(),
        AgentConfig(registration=registration, workdir_root=tmp_path),
    )

    class Lease:
        def check(self):
            return None

    outputs = runner._execute_process(
        slot,
        {
            "id": "job-a",
            "lease_token": "secret",
            "payload": {"message": "hello"},
            "executor": {
                "type": "command",
                "command": ["{agent_python}", "-m", "gpu_orchestrator.demo_executor"],
                "environment": {},
                "timeout_seconds": 10,
            },
        },
        tmp_path,
        Lease(),
        container=False,
    )

    assert outputs[0]["name"] == "result"
    result_path = tmp_path / "result.json"
    assert json.loads(result_path.read_text(encoding="utf-8"))["message"] == "hello"
