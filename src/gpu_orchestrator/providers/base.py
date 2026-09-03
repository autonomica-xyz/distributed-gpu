"""Provider interface and cross-provider billing safety invariants."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..models import ResourceRequirements


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        account_hard_stop: bool = False,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.account_hard_stop = account_hard_stop


@dataclass(frozen=True)
class ProviderOffer:
    provider: str
    offer_id: str
    gpu_model: str
    gpu_count: int
    vram_mb: int
    hourly_cost_usd: float
    reliability: float | None = None
    # None means the provider's optional availability preflight was
    # unavailable; zero is authoritative known-unavailable capacity.
    available_count: int | None = 1
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class ProviderResource:
    provider: str
    resource_id: str
    name: str
    state: str
    worker_id: str | None = None
    gpu_model: str | None = None
    hourly_cost_usd: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class WorkerLaunchSpec:
    name: str
    worker_id: str
    image: str
    command: list[str]
    environment: dict[str, str] = field(default_factory=dict, repr=False)
    disk_gb: int = 30
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    provider_template: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        # Mutable tags make unattended scale-out impossible to reproduce.
        if "@sha256:" not in self.image:
            raise ValueError("worker image must be pinned by sha256 digest")


class ProviderDriver(Protocol):
    name: str

    def search_offers(
        self,
        requirements: ResourceRequirements,
        *,
        max_hourly_cost_usd: float,
        limit: int = 25,
    ) -> list[ProviderOffer]: ...

    def launch(self, offer: ProviderOffer | None, spec: WorkerLaunchSpec) -> ProviderResource: ...

    def inventory(self) -> list[ProviderResource]: ...

    def destroy(self, resource_id: str) -> None: ...

    def account_credit(self) -> float | None: ...


def destroy_and_verify_absent(
    driver: ProviderDriver,
    resource_id: str,
    *,
    timeout_seconds: float = 60.0,
    poll_seconds: float = 2.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Destroy and require authoritative provider absence before returning.

    Callers must not remove worker credentials or local accounting rows until
    this function succeeds. This is the most important operational invariant
    carried over from the legacy system.
    """

    driver.destroy(resource_id)
    deadline = monotonic() + max(0.0, timeout_seconds)
    while True:
        visible = {resource.resource_id for resource in driver.inventory()}
        if resource_id not in visible:
            return
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise ProviderError(
                f"{driver.name} still lists resource {resource_id!r} after destroy",
                retryable=True,
            )
        sleep(min(max(0.05, poll_seconds), remaining))

