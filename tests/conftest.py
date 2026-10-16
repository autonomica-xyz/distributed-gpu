from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from sparehour.models import AgentRegistration, SlotInventory, WorkloadSpec
from sparehour.registry import WorkloadRegistry
from sparehour.service import Orchestrator
from sparehour.store import Store


def _load_demo_fixture() -> WorkloadSpec:
    """Load the no-GPU demo workload used by the local smoke path and tests.

    It is a test fixture, deliberately kept out of the operational
    ``workloads/`` directory so production deployments never advertise it.
    """
    path = Path(__file__).parent / "fixtures" / "workloads" / "demo.yaml"
    return WorkloadSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class MutableClock:
    def __init__(self):
        self.value = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)

    def __call__(self):
        return self.value


@pytest.fixture
def registry() -> WorkloadRegistry:
    reg = WorkloadRegistry.from_directory("workloads")
    reg.add(_load_demo_fixture())
    return reg


@pytest.fixture
def clock() -> MutableClock:
    return MutableClock()


@pytest.fixture
def store(tmp_path, clock) -> Store:
    result = Store(tmp_path / "control.sqlite3", clock=clock)
    yield result
    result.close()


@pytest.fixture
def orchestrator(registry, store) -> Orchestrator:
    return Orchestrator(registry, store)


def register_agent(
    store: Store,
    agent_id: str,
    *,
    slots: int = 1,
    vram_mb: int = 24576,
    capabilities: list[str] | None = None,
    model_packs: list[str] | None = None,
    provider: str = "local",
) -> AgentRegistration:
    registration = AgentRegistration(
        agent_id=agent_id,
        provider=provider,
        provider_resource_id=f"resource-{agent_id}",
        hostname=agent_id,
        cpu_cores=16,
        ram_mb=65536,
        disk_free_mb=500000,
        slots=[
            SlotInventory(
                id=f"{agent_id}-slot-{index}",
                gpu_index=index,
                gpu_uuid=f"GPU-{agent_id}-{index}",
                model="NVIDIA RTX 4090",
                vram_mb=vram_mb,
                compute_capability="8.9",
                capabilities=capabilities or ["demo-gpu"],
                model_packs=model_packs or [],
            )
            for index in range(slots)
        ],
    )
    store.register_agent(registration)
    return registration

