"""Route-scoped fleet planning across local, Vast, and Salad capacity."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .autoscaler import (
    ProviderCandidate as CostCandidate,
    ScalingInputs,
    estimate_target_slots,
    rank_candidates,
)
from .models import ResourceRequirements
from .providers.base import ProviderDriver, ProviderOffer
from .registry import WorkloadRegistry
from .store import Store


class ProviderPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_hourly_cost_usd: float = Field(gt=0, allow_inf_nan=False)
    expected_hourly_cost_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    require_known_credit: bool = True
    available_credit_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def expected_price_respects_cap(self) -> "ProviderPolicy":
        if (
            self.expected_hourly_cost_usd is not None
            and self.expected_hourly_cost_usd > self.max_hourly_cost_usd
        ):
            raise ValueError("expected_hourly_cost_usd must not exceed the hourly cap")
        return self


class PoolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_version: str = "sparehour.example/v1alpha1"
    id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,62}$")
    workload: str
    route: str
    worker_prefix: str
    image: str
    command: list[str]
    environment: dict[str, str] = Field(default_factory=dict)
    providers: dict[str, ProviderPolicy]
    # Salad launches need a full container-group template (GPU classes,
    # memory, probes). Templates live beside the pool so the trusted route
    # stays the only thing a job can influence.
    provider_templates: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def immutable_image(self) -> "PoolSpec":
        if "@sha256:" not in self.image:
            raise ValueError("pool worker image must be pinned by sha256 digest")
        if not self.command:
            raise ValueError("pool command is required")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PoolSpec":
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_directory(cls, directory: str | Path) -> list["PoolSpec"]:
        """Load every pool policy in a directory, sorted by id.

        A missing directory is an empty fleet rather than an error so the
        control plane can start before any provider pool exists.
        """
        root = Path(directory)
        if not root.is_dir():
            return []
        pools = []
        for path in sorted([*root.glob("*.yaml"), *root.glob("*.yml")]):
            pools.append(cls.from_yaml(path))
        return sorted(pools, key=lambda pool: pool.id)


@dataclass(frozen=True)
class ProviderPlan:
    provider: str
    credit_usd: float | None
    blocked_reason: str | None
    offers: list[dict[str, Any]]


@dataclass(frozen=True)
class FleetPlan:
    pool_id: str
    workload: str
    route: str
    queued_jobs: int
    queued_units: float
    running_jobs: int
    running_units: float
    current_slots: int
    target_slots: int
    requested_slots: int
    blocked_reason: str | None
    reason: str
    providers: list[ProviderPlan]
    ranked_candidates: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FleetPlanner:
    def __init__(
        self,
        registry: WorkloadRegistry,
        store: Store,
        *,
        agent_stale_seconds: float = 120.0,
    ):
        self.registry = registry
        self.store = store
        self.agent_stale_seconds = max(1.0, float(agent_stale_seconds))

    def plan(
        self,
        pool: PoolSpec,
        drivers: Mapping[str, ProviderDriver],
    ) -> FleetPlan:
        workload = self.registry.get(pool.workload)
        if pool.route not in workload.routes:
            raise ValueError(f"pool route {pool.route!r} does not exist")
        route = workload.routes[pool.route]
        requirements = route.resources
        current_slots = self._current_slots(pool, route.compatibility.capability)
        queued_jobs, queued_units, running_jobs, running_units = self._backlog(pool)

        provider_plans: list[ProviderPlan] = []
        provider_credits: dict[str, float | None] = {}
        offers: list[ProviderOffer] = []
        for provider_name, policy in pool.providers.items():
            if not policy.enabled or provider_name == "local":
                continue
            driver = drivers.get(provider_name)
            if driver is None:
                provider_plans.append(
                    ProviderPlan(provider_name, None, "driver_not_configured", [])
                )
                continue
            credit = policy.available_credit_usd
            blocked: str | None = None
            if credit is None:
                try:
                    credit = driver.account_credit()
                except Exception:
                    credit = None
            provider_credits[provider_name] = credit
            if policy.require_known_credit and credit is None:
                blocked = "provider_credit_unknown"
            elif credit is not None and credit <= 0:
                blocked = "provider_credit_insufficient"
            # Offer discovery is read-only, so keep showing price/availability
            # even when spend policy blocks deployment.
            try:
                rows = driver.search_offers(
                    requirements,
                    max_hourly_cost_usd=policy.max_hourly_cost_usd,
                )
                if policy.expected_hourly_cost_usd is not None:
                    rows = [
                        replace(
                            row,
                            hourly_cost_usd=policy.expected_hourly_cost_usd,
                        )
                        if row.raw.get("use_policy_price")
                        else row
                        for row in rows
                    ]
            except Exception as exc:
                blocked = blocked or f"offer_search_failed:{type(exc).__name__}"
                rows = []
            # Known-zero availability remains visible to operators but cannot
            # drive planning. Unknown availability is eligible because Salad's
            # preflight endpoint is optional and provider allocation remains
            # authoritative. Credit-blocked providers are display-only.
            if blocked is None:
                offers.extend(row for row in rows if row.available_count != 0)
            provider_plans.append(
                ProviderPlan(
                    provider=provider_name,
                    credit_usd=credit,
                    blocked_reason=blocked,
                    offers=[_public_offer(row) for row in rows],
                )
            )

        cost_candidates = [
            CostCandidate(
                provider=offer.provider,
                offer_id=offer.offer_id,
                gpu_model=offer.gpu_model,
                hourly_cost_usd=offer.hourly_cost_usd,
                startup_hours=route.scaling.startup_minutes / 60.0,
                units_per_gpu_hour=route.cost.default_units_per_gpu_hour,
                reliability=offer.reliability or 0.8,
                available_count=offer.available_count,
            )
            for offer in offers
        ]
        ranked = rank_candidates(
            cost_candidates, units_to_drain=queued_units + running_units
        )
        ranked_before_credit = ranked
        ranked = [
            row
            for row in ranked_before_credit
            if provider_credits.get(row.candidate.provider) is None
            or row.estimated_total_cost_usd
            <= float(provider_credits[row.candidate.provider] or 0.0)
        ]
        providers_before_credit = {
            row.candidate.provider for row in ranked_before_credit
        }
        providers_after_credit = {row.candidate.provider for row in ranked}
        insufficient_credit = {
            provider
            for provider in providers_before_credit - providers_after_credit
            if provider_credits.get(provider) is not None
        }
        if insufficient_credit:
            provider_plans = [
                replace(
                    plan,
                    blocked_reason=(
                        plan.blocked_reason or "provider_credit_insufficient"
                    ),
                )
                if plan.provider in insufficient_credit
                else plan
                for plan in provider_plans
            ]
        cheapest_hourly = ranked[0].candidate.hourly_cost_usd if ranked else 0.0
        decision = estimate_target_slots(
            ScalingInputs(
                queued_units=queued_units,
                running_units=running_units,
                current_slots=current_slots,
                units_per_gpu_hour=route.cost.default_units_per_gpu_hour,
                target_completion_hours=route.scaling.target_completion_hours,
                startup_hours=route.scaling.startup_minutes / 60.0,
                min_slots=route.scaling.min_slots,
                max_slots=route.scaling.max_slots,
                hourly_cost_usd=cheapest_hourly,
            )
        )
        target = decision.target_slots
        requested = decision.requested_slots
        blocker = decision.blocked_reason
        reason = decision.reason
        if not route.scaling.enabled:
            target = current_slots
            requested = 0
            blocker = "autoscaling_disabled"
            reason = "workload route has autonomous scaling disabled"
        elif requested > 0 and not ranked:
            target = current_slots
            requested = 0
            blocker = "provider_capacity_unavailable"
            reason = "no funded provider offer can satisfy this route"

        return FleetPlan(
            pool_id=pool.id,
            workload=pool.workload,
            route=pool.route,
            queued_jobs=queued_jobs,
            queued_units=queued_units,
            running_jobs=running_jobs,
            running_units=running_units,
            current_slots=current_slots,
            target_slots=target,
            requested_slots=requested,
            blocked_reason=blocker,
            reason=reason,
            providers=provider_plans,
            ranked_candidates=[
                {
                    "provider": row.candidate.provider,
                    "offer_id": row.candidate.offer_id,
                    "gpu_model": row.candidate.gpu_model,
                    "hourly_cost_usd": row.candidate.hourly_cost_usd,
                    "estimated_completion_hours": row.estimated_completion_hours,
                    "estimated_total_cost_usd": row.estimated_total_cost_usd,
                    "available_count": row.candidate.available_count,
                    "reliability": row.candidate.reliability,
                }
                for row in ranked
            ],
        )

    def _current_slots(self, pool: PoolSpec, capability: str) -> int:
        total = 0
        stale_before = self.store.clock() - timedelta(seconds=self.agent_stale_seconds)
        for agent in self.store.list_agents():
            if agent["labels"].get("pool") != pool.id:
                continue
            if agent["state"] != "ready":
                continue
            try:
                if datetime.fromisoformat(agent["heartbeat_at"]) < stale_before:
                    continue
            except (TypeError, ValueError):
                continue
            for slot in agent["slots"]:
                if slot["state"] == "offline":
                    continue
                if capability in set(slot["capabilities"]):
                    total += 1
        return total

    def _backlog(self, pool: PoolSpec) -> tuple[int, float, int, float]:
        queued_jobs = running_jobs = 0
        queued_units = running_units = 0.0
        for row in self.store.backlog():
            if row["workload"] != pool.workload or row["route"] != pool.route:
                continue
            if row["state"] == "pending":
                queued_jobs += int(row["jobs"])
                queued_units += float(row["units"])
            else:
                running_jobs += int(row["jobs"])
                running_units += float(row["units"])
        return queued_jobs, queued_units, running_jobs, running_units


def _public_offer(offer: ProviderOffer) -> dict[str, Any]:
    public = {
        "provider": offer.provider,
        "offer_id": offer.offer_id,
        "gpu_model": offer.gpu_model,
        "gpu_count": offer.gpu_count,
        "vram_mb": offer.vram_mb,
        "hourly_cost_usd": offer.hourly_cost_usd,
        "estimated_monthly_cost_usd": offer.hourly_cost_usd * 730.0,
        "reliability": offer.reliability,
        "available_count": offer.available_count,
    }
    # Explicit allowlist: never return arbitrary provider response bodies.
    for key in (
        "gpu_class_id",
        "priority",
        "compute_capability",
        "price_source",
        "availability_status",
        "on_call_count",
        "is_high_demand",
    ):
        if key in offer.raw:
            public[key] = offer.raw[key]
    return public
