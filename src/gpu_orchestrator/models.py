"""Validated platform and workload contracts.

The queue persists a snapshot of these contracts with every job. That makes an
old job reproducible even after the registry is updated.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class JobState(str, Enum):
    PENDING = "pending"
    LEASED = "leased"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    DEAD_LETTER = "dead-letter"
    CANCELLED = "cancelled"


ACTIVE_JOB_STATES = {JobState.LEASED, JobState.RUNNING}
TERMINAL_JOB_STATES = {
    JobState.SUCCEEDED,
    JobState.DEAD_LETTER,
    JobState.CANCELLED,
}


class ArtifactSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,62}$")
    media_types: list[str] = Field(default_factory=list)
    required: bool = True
    multiple: bool = False
    max_bytes: int = Field(default=1 << 30, gt=0)
    validator: Literal[
        "none",
        "sha256",
        "json",
        "json_schema",
        "image",
        "media_probe",
    ] = "sha256"
    json_schema: dict[str, Any] | None = Field(default=None, alias="schema")

    @model_validator(mode="after")
    def validator_has_schema(self) -> "ArtifactSpec":
        if self.validator == "json_schema" and self.json_schema is None:
            raise ValueError("validator=json_schema requires schema")
        return self


class ResourceRequirements(BaseModel):
    """Resources and bounded inventory labels used at claim time.

    The first product version intentionally supports one GPU per independent
    job. ``gpu_count`` is retained in the contract so gang scheduling can be
    added later without changing every workload document.
    """

    model_config = ConfigDict(extra="forbid")

    gpu_count: int = Field(default=1, ge=1, le=1)
    vendor: str = "nvidia"
    min_vram_mb: int = Field(default=0, ge=0)
    min_compute_capability: str | None = None
    gpu_models: list[str] = Field(default_factory=list)
    excluded_gpu_models: list[str] = Field(default_factory=list)
    min_cpu_cores: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    min_ram_mb: int = Field(default=1024, ge=0)
    min_disk_mb: int = Field(default=1024, ge=0)
    required_labels: dict[str, str] = Field(default_factory=dict)
    allowed_providers: list[str] = Field(default_factory=list)


class CompatibilitySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability: str = Field(min_length=1)
    model_packs: list[str] = Field(default_factory=list)


class ExecutorSpec(BaseModel):
    """Trusted execution definition selected by admission, never by a job."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["command", "container", "http", "plugin"]
    command: list[str] = Field(default_factory=list)
    image: str | None = None
    endpoint: str | None = None
    handler: str | None = None
    timeout_seconds: int = Field(default=3600, gt=0)
    environment: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def required_executor_fields(self) -> "ExecutorSpec":
        if self.type in {"command", "container"} and not self.command:
            raise ValueError(f"executor type {self.type!r} requires command")
        if self.type == "container" and not self.image:
            raise ValueError("container executor requires image")
        if self.type == "container" and "@sha256:" not in str(self.image):
            raise ValueError("container executor image must be pinned by sha256 digest")
        if self.type == "http" and not self.endpoint:
            raise ValueError("http executor requires endpoint")
        if self.type == "plugin" and not self.handler:
            raise ValueError("plugin executor requires handler")
        return self


class CostSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit: str = Field(min_length=1)
    default_units_per_job: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    payload_units_field: str | None = None
    payload_items_field: str | None = None
    payload_units_multiplier: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    default_units_per_gpu_hour: float = Field(gt=0, allow_inf_nan=False)


class ScalingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    min_slots: int = Field(default=0, ge=0)
    max_slots: int = Field(default=10, ge=0)
    target_completion_hours: float = Field(default=6.0, gt=0, allow_inf_nan=False)
    startup_minutes: float = Field(default=15.0, ge=0, allow_inf_nan=False)
    scale_down_idle_minutes: float = Field(default=10.0, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid_slot_range(self) -> "ScalingSpec":
        if self.max_slots < self.min_slots:
            raise ValueError("max_slots must be >= min_slots")
        return self


class RetrySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(default=3, ge=1, le=100)
    retryable_error_codes: list[str] = Field(
        default_factory=lambda: [
            "worker_lost",
            "provider_interrupted",
            "artifact_transport_failed",
            "gpu_oom",
        ]
    )


class RouteSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    compatibility: CompatibilitySpec
    resources: ResourceRequirements = Field(default_factory=ResourceRequirements)
    executor: ExecutorSpec
    cost: CostSpec
    scaling: ScalingSpec = Field(default_factory=ScalingSpec)


class WorkloadSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_version: Literal["distributed-gpu.example/v1alpha1"] = "distributed-gpu.example/v1alpha1"
    name: str = Field(pattern=r"^[a-z][a-z0-9-]{1,62}$")
    version: str = Field(min_length=1)
    description: str = ""
    default_route: str
    payload_schema: dict[str, Any]
    inputs: list[ArtifactSpec] = Field(default_factory=list)
    outputs: list[ArtifactSpec] = Field(default_factory=list)
    routes: dict[str, RouteSpec]
    retry: RetrySpec = Field(default_factory=RetrySpec)

    @model_validator(mode="after")
    def cross_validate(self) -> "WorkloadSpec":
        if self.default_route not in self.routes:
            raise ValueError("default_route must name a route")
        if not self.routes:
            raise ValueError("at least one route is required")
        for group_name, artifacts in (("inputs", self.inputs), ("outputs", self.outputs)):
            names = [artifact.name for artifact in artifacts]
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate artifact name in {group_name}")
        return self


class SlotInventory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    gpu_index: int = Field(ge=0)
    gpu_uuid: str
    vendor: str = "nvidia"
    model: str
    vram_mb: int = Field(gt=0)
    compute_capability: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    model_packs: list[str] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)


class AgentRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str
    provider: str
    provider_resource_id: str | None = None
    hostname: str
    cpu_cores: float = Field(default=0, ge=0, allow_inf_nan=False)
    ram_mb: int = Field(default=0, ge=0)
    disk_free_mb: int = Field(default=0, ge=0)
    labels: dict[str, str] = Field(default_factory=dict)
    slots: list[SlotInventory] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_slot_inventory(self) -> "AgentRegistration":
        slot_ids = [slot.id for slot in self.slots]
        gpu_uuids = [slot.gpu_uuid for slot in self.slots]
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError("agent registration contains duplicate slot IDs")
        if len(gpu_uuids) != len(set(gpu_uuids)):
            raise ValueError("agent registration contains duplicate GPU UUIDs")
        return self


class OutputArtifact(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    uri: str
    media_type: str
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
