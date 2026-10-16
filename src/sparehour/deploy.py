"""Operator-initiated provider worker deployment with a durable ledger.

Autonomous create/destroy remains deliberately disabled.
This service powers explicit, human-confirmed deployments from the workbench:
one pool policy + one provider + a bounded worker count, executed through the
existing provider drivers, recorded in the ``deployments`` ledger, and torn
down with the same safety ordering the verified Salad run used:

1. refuse teardown while the worker holds an active job lease;
2. destroy the provider resource and require authoritative inventory absence;
3. only then disable the worker credential and mark the ledger row destroyed.

Provider API keys never leave this process. Workers receive only the public
controller URL, an optional public wheel URL, a fixed identity, and a
short-lived agent token scoped to the pool's route capability.
"""

from __future__ import annotations

import os
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from .controller import FleetPlanner, PoolSpec
from .models import ResourceRequirements
from .providers.base import (
    ProviderDriver,
    ProviderError,
    ProviderOffer,
    ProviderResource,
    WorkerLaunchSpec,
    destroy_and_verify_absent,
)
from .registry import RegistryError, WorkloadRegistry
from .store import ConflictError, NotFoundError, Store


UTC = timezone.utc
TERMINAL_DEPLOYMENT_STATES = {"destroyed", "failed"}
MAX_WORKERS_PER_DEPLOY = 5
MIN_TOKEN_HOURS = 1
MAX_TOKEN_HOURS = 24 * 7
MONTHLY_BILLING_HOURS = 730.0
AGENT_STALE_SECONDS = 120.0


class DeploymentError(ValueError):
    """Operator-visible request error (maps to HTTP 422)."""


def build_drivers_from_env(
    environ: Mapping[str, str] | None = None,
) -> dict[str, ProviderDriver]:
    """Construct configured provider drivers from environment credentials."""

    env = dict(os.environ if environ is None else environ)
    drivers: dict[str, ProviderDriver] = {}
    vast_key = env.get("VAST_API_KEY", "").strip()
    if vast_key:
        from .providers.vast import VastProvider

        drivers["vast"] = VastProvider(
            vast_key, base_url=env.get("VAST_API_BASE", "https://console.vast.ai/api")
        )
    salad_key = env.get("SALAD_API_KEY", "").strip()
    if salad_key:
        from .providers.salad import SaladProvider

        drivers["salad"] = SaladProvider(
            salad_key,
            env.get("SALAD_ORGANIZATION", ""),
            env.get("SALAD_PROJECT", ""),
            base_url=env.get("SALAD_API_BASE", "https://api.salad.com/api/public"),
        )
    return drivers


@dataclass(frozen=True)
class DeploymentService:
    registry: WorkloadRegistry
    store: Store
    drivers: Mapping[str, ProviderDriver]
    pools_directory: str | Path
    public_controller_url: str
    wheel_url: str | None = None
    _reconcile_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    # ---- capacity (read-only planning surface for the workbench) ----

    def capacity(self) -> list[dict[str, Any]]:
        planner = FleetPlanner(self.registry, self.store)
        result = []
        for pool in self.pools():
            route = self.registry.get(pool.workload).routes[pool.route]
            try:
                plan = planner.plan(pool, self.drivers).to_dict()
            except Exception as exc:  # a broken pool must not hide the rest
                plan = {"error": f"{type(exc).__name__}: {exc}"}
            for provider in plan.get("providers", []):
                provider["offers"] = provider.get("offers", [])[:20]
            result.append(
                {
                    "pool": _public_pool(pool),
                    "capability": route.compatibility.capability,
                    "model_packs": list(route.compatibility.model_packs),
                    "plan": plan,
                }
            )
        return result

    def pools(self) -> list[PoolSpec]:
        return PoolSpec.from_directory(self.pools_directory)

    def _pool(self, pool_id: str) -> PoolSpec:
        pool = next((item for item in self.pools() if item.id == pool_id), None)
        if pool is None:
            raise NotFoundError(f"pool {pool_id!r} not found")
        return pool

    # ---- deploy ----

    def deploy(
        self,
        *,
        pool_id: str,
        provider: str,
        count: int = 1,
        expires_hours: int = 24,
        offer_id: str | None = None,
        actor: str = "operator",
        rule_id: str | None = None,
        action_id: str | None = None,
        startup_deadline: str | None = None,
        require_known_availability: bool = False,
    ) -> list[dict[str, Any]]:
        if not 1 <= int(count) <= MAX_WORKERS_PER_DEPLOY:
            raise DeploymentError(
                f"count must be between 1 and {MAX_WORKERS_PER_DEPLOY}"
            )
        if not MIN_TOKEN_HOURS <= int(expires_hours) <= MAX_TOKEN_HOURS:
            raise DeploymentError(
                f"expires_hours must be between {MIN_TOKEN_HOURS} and {MAX_TOKEN_HOURS}"
            )
        pool = self._pool(pool_id)
        policy = pool.providers.get(provider)
        if policy is None or provider == "local":
            raise DeploymentError(
                f"pool {pool_id!r} has no provider policy for {provider!r}"
            )
        if not policy.enabled:
            raise DeploymentError(f"provider {provider!r} is disabled for this pool")
        driver = self.drivers.get(provider)
        if driver is None:
            raise DeploymentError(
                f"provider driver {provider!r} is not configured on the control plane"
            )
        credit = policy.available_credit_usd
        if credit is None:
            try:
                credit = driver.account_credit()
            except Exception:
                credit = None
        if policy.require_known_credit and credit is None:
            raise DeploymentError(
                f"provider {provider!r} credit is unknown and this pool requires a known balance"
            )
        if credit is not None and credit <= 0:
            raise DeploymentError(f"provider {provider!r} has insufficient credit")

        workload = self.registry.get(pool.workload)
        route = workload.routes[pool.route]
        capability = route.compatibility.capability
        requirements = route.resources
        expires_at = datetime.now(UTC) + timedelta(hours=int(expires_hours))
        if not startup_deadline:
            startup_deadline = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        offer = self._select_offer(
            driver,
            requirements,
            policy.max_hourly_cost_usd,
            requested_offer_id=offer_id,
            require_known_availability=require_known_availability,
        )
        if offer is None:
            raise DeploymentError(
                f"provider {provider!r} has no eligible offer within the pool cost cap"
            )
        if offer.available_count is not None and int(count) > offer.available_count:
            raise DeploymentError(
                f"offer {offer.offer_id!r} has only {offer.available_count} "
                f"currently available worker(s)"
            )

        created: list[dict[str, Any]] = []
        for _ in range(int(count)):
            worker_id = f"{pool.worker_prefix}-{secrets.token_hex(3)}"
            raw_token = self.store.issue_token(
                role="agent",
                subject=worker_id,
                scopes={"routes": [capability]},
                expires_at=expires_at,
            )
            token_id = raw_token.removeprefix("sph_").split(".", 1)[0]
            environment = {
                **pool.environment,
                "SPAREHOUR_ENDPOINT": self.public_controller_url.rstrip("/"),
                "SPAREHOUR_TOKEN": raw_token,
                "SPAREHOUR_AGENT_ID": worker_id,
                "SPAREHOUR_PROVIDER_RESOURCE_ID": worker_id,
            }
            if self.wheel_url:
                environment["SPAREHOUR_WHEEL_URL"] = self.wheel_url
            command = [
                argument.replace("${SPAREHOUR_AGENT_ID}", worker_id)
                for argument in pool.command
            ]
            spec = WorkerLaunchSpec(
                name=worker_id,
                worker_id=worker_id,
                image=pool.image,
                command=command,
                environment=environment,
                disk_gb=max(1, requirements.min_disk_mb // 1024),
                resources=ResourceRequirements.model_validate(
                    requirements.model_dump(mode="json")
                ),
                provider_template=pool.provider_templates.get(provider, {}),
            )
            # Record the deployment before provider create so a controller crash
            # after the provider accepts the resource can still be correlated to
            # the action and worker_id during recovery.
            priority = str(offer.raw.get("priority") or "market")
            record = self.store.record_deployment(
                pool_id=pool.id,
                provider=provider,
                resource_id=worker_id,
                resource_name=worker_id,
                worker_id=worker_id,
                token_id=token_id,
                workload=pool.workload,
                route=pool.route,
                capability=capability,
                image=pool.image,
                hourly_cost_usd=offer.hourly_cost_usd,
                state="preparing",
                detail=f"preparing launch by {actor}; {offer.gpu_model} {priority} at ${offer.hourly_cost_usd:.4f}/h",
                offer_id=offer.offer_id,
                gpu_model=offer.gpu_model,
                priority=priority,
                region=offer.raw.get("region"),
                price_source=offer.raw.get("price_source"),
                availability_at_selection=offer.available_count,
                rule_id=rule_id,
                action_id=action_id,
                startup_deadline=startup_deadline,
            )
            try:
                resource = driver.launch(offer, spec)
                record = self.store.update_deployment(
                    record["id"],
                    state="requested",
                    resource_id=resource.resource_id,
                    resource_name=resource.name,
                    hourly_cost_usd=resource.hourly_cost_usd,
                    detail=(
                        f"launched by {actor}; {offer.gpu_model} {priority} at "
                        f"${offer.hourly_cost_usd:.4f}/h; provider state {resource.state}"
                    ),
                    provider_create_response={"resource_id": resource.resource_id, "name": resource.name, "state": resource.state},
                )
            except Exception as exc:
                # A timeout after create is ambiguous. Inventory is authoritative:
                # never revoke the credential if the named resource actually exists.
                inventory_error = False
                discovered = None
                try:
                    discovered = next(
                    (item for item in driver.inventory() if item.worker_id == worker_id or item.resource_id == worker_id or item.name == worker_id),
                    None,
                )
                except Exception:
                    inventory_error = True
                if discovered is not None:
                    record = self.store.update_deployment(
                        record["id"],
                        state="requested",
                        resource_id=discovered.resource_id,
                        resource_name=discovered.name,
                        hourly_cost_usd=discovered.hourly_cost_usd,
                        detail=f"launch response ambiguous; resource confirmed by inventory: {exc}",
                    )
                    created.append(record)
                    continue

                # Determine whether the failure is plausibly ambiguous (network
                # timeout, transient provider error, or inventory unavailable).
                # ProviderError with retryable=False or caller validation errors
                # are treated as definitive failures.
                ambiguous = inventory_error
                if isinstance(exc, ProviderError) and exc.retryable:
                    ambiguous = True
                if not isinstance(exc, (ProviderError, ValueError)):
                    ambiguous = True

                if ambiguous:
                    # We do not know whether the resource exists. Leave the
                    # deployment in the preparing state and the credential active;
                    # reconciliation/recovery will verify inventory later.
                    record = self.store.update_deployment(
                        record["id"],
                        state="preparing",
                        detail=f"launch response ambiguous; awaiting inventory: {exc}",
                    )
                    created.append(record)
                    continue

                # Confirmed absent: retire the credential and mark the deployment failed.
                self.store.disable_token(token_id)
                record = self.store.update_deployment(
                    record["id"],
                    state="failed",
                    detail=(
                        f"launch failed for offer {offer.offer_id} "
                        f"({offer.gpu_model}, priority "
                        f"{offer.raw.get('priority') or 'market'}, "
                        f"availability "
                        f"{offer.available_count if offer.available_count is not None else 'unknown'}, "
                        f"${offer.hourly_cost_usd:.4f}/h), requested by {actor}: {exc}"
                    ),
                    failure_classification="provider_launch_failed",
                )
            created.append(record)
        return created

    def _select_offer(
        self,
        driver: ProviderDriver,
        requirements: ResourceRequirements,
        max_hourly_cost_usd: float,
        *,
        requested_offer_id: str | None = None,
        require_known_availability: bool = False,
    ) -> ProviderOffer | None:
        offers = driver.search_offers(
            requirements, max_hourly_cost_usd=max_hourly_cost_usd
        )
        if requested_offer_id is not None:
            offer = next((row for row in offers if row.offer_id == requested_offer_id), None)
            if offer is None:
                raise DeploymentError(
                    f"offer {requested_offer_id!r} is no longer eligible; refresh capacity"
                )
            if offer.available_count == 0:
                raise DeploymentError(
                    f"offer {requested_offer_id!r} currently has no available capacity"
                )
            if require_known_availability and offer.available_count is None:
                raise DeploymentError(
                    f"offer {requested_offer_id!r} availability is unknown and this rule requires known availability"
                )
            return offer
        eligible = [
            row for row in offers
            if row.available_count != 0
            and (not require_known_availability or row.available_count is not None)
        ]
        return min(
            eligible,
            key=lambda row: (
                row.hourly_cost_usd,
                -(row.available_count or 0),
                row.offer_id,
            ),
            default=None,
        )

    # ---- ledger, reconciliation, teardown ----

    def list_deployments(self) -> list[dict[str, Any]]:
        agents = {row["id"]: row for row in self.store.list_agents()}
        now = self.store.clock()
        result = []
        for row in self.store.list_deployments():
            item = dict(row)
            agent_status = _agent_status(agents.get(row["worker_id"]), now)
            item["agent_status"] = agent_status
            agent = agents.get(row["worker_id"])
            item["agent_heartbeat_at"] = agent.get("heartbeat_at") if agent else None
            item["slot_count"] = len(agent.get("slots", [])) if agent else 0
            cost = item.get("hourly_cost_usd")
            item["estimated_monthly_cost_usd"] = (
                float(cost) * MONTHLY_BILLING_HOURS if cost is not None else None
            )
            result.append(item)
        return result

    def reconcile(self) -> dict[str, int]:
        """Refresh provider and agent states from authoritative inventory."""

        with self._reconcile_lock:
            return self._reconcile_locked()

    def _reconcile_locked(self) -> dict[str, int]:
        counts = {
            "checked": 0, "active": 0, "creating": 0, "allocating": 0,
            "starting": 0, "stale": 0, "absent": 0, "terminal": 0,
            "errors": 0,
        }
        rows = self.store.list_deployments()
        agents = {row["id"]: row for row in self.store.list_agents()}
        now = self.store.clock()
        inventories: dict[str, dict[str, ProviderResource] | None] = {}
        for provider in {row["provider"] for row in rows if row["state"] not in TERMINAL_DEPLOYMENT_STATES and row["state"] != "preparing"}:
            driver = self.drivers.get(provider)
            if driver is None:
                inventories[provider] = None
                continue
            try:
                resources = driver.inventory()
                inventories[provider] = {
                    key: resource
                    for resource in resources
                    for key in (resource.resource_id, resource.name)
                }
            except Exception:
                inventories[provider] = None

        for row in rows:
            if row["state"] in TERMINAL_DEPLOYMENT_STATES:
                counts["terminal"] += 1
                continue
            if row["state"] == "preparing":
                continue
            inventory = inventories.get(row["provider"])
            if inventory is None:
                counts["errors"] += 1
                continue
            counts["checked"] += 1
            resource = inventory.get(row["resource_id"]) or inventory.get(row["resource_name"])
            if resource is None:
                self.store.update_deployment(
                    row["id"], state="absent",
                    detail="not present in authoritative provider inventory",
                )
                self.store.disable_token(row["token_id"])
                counts["absent"] += 1
                continue

            agent_status = _agent_status(agents.get(row["worker_id"]), now)
            state = _deployment_state(resource.state, agent_status)
            detail_parts = [f"provider {resource.state}", f"agent {agent_status}"]
            if resource.gpu_model:
                detail_parts.append(resource.gpu_model)
            if resource.hourly_cost_usd is not None:
                detail_parts.append(f"${resource.hourly_cost_usd:.4f}/h")
            self.store.update_deployment(
                row["id"],
                state=state,
                detail="; ".join(detail_parts),
                hourly_cost_usd=resource.hourly_cost_usd,
            )
            if state in {"active", "draining"} and not row.get("first_heartbeat"):
                self.store.update_deployment(row["id"], first_heartbeat=now.isoformat(timespec="seconds"))
            counts[state] = counts.get(state, 0) + 1

        # Resolve ambiguous launches whose inventory check failed during deploy.
        for row in rows:
            if row["state"] != "preparing":
                continue
            deadline = row.get("startup_deadline")
            if deadline:
                try:
                    if now < datetime.fromisoformat(deadline):
                        continue
                except (TypeError, ValueError):
                    pass
            driver = self.drivers.get(row["provider"])
            if driver is None:
                continue
            try:
                resources = driver.inventory()
            except Exception:
                continue
            discovered = next(
                (item for item in resources if item.worker_id == row["worker_id"] or item.resource_id == row["worker_id"] or item.name == row["worker_id"]),
                None,
            )
            if discovered is not None:
                self.store.update_deployment(
                    row["id"],
                    state="requested",
                    resource_id=discovered.resource_id,
                    resource_name=discovered.name,
                    hourly_cost_usd=discovered.hourly_cost_usd,
                    detail="launch confirmed by inventory during reconciliation",
                )
                continue

            # The startup deadline has passed and inventory does not show the
            # resource. Before giving up, attempt to destroy it and require
            # provider confirmation of absence so a billable orphan is not left
            # behind.
            try:
                destroy_and_verify_absent(
                    driver,
                    row.get("resource_id") or row["worker_id"],
                    timeout_seconds=30,
                    poll_seconds=2,
                )
            except Exception:
                # The provider may reject the recorded id (e.g. Vast needs an
                # integer contract id). Fall back to the current inventory view.
                pass
            try:
                resources = driver.inventory()
            except Exception:
                continue
            if any(
                item.worker_id == row["worker_id"]
                or item.resource_id == row["worker_id"]
                or item.resource_id == (row.get("resource_id") or "")
                or item.name == row["worker_id"]
                or item.name == (row.get("resource_name") or "")
                for item in resources
            ):
                continue
            self.store.disable_token(row["token_id"])
            self.store.update_deployment(
                row["id"],
                state="failed",
                detail="launch failed; resource not confirmed in inventory after startup deadline",
                failure_classification="provider_launch_failed",
            )
        return counts

    def teardown(self, deployment_id: str, *, actor: str = "operator") -> dict[str, Any]:
        row = self.store.get_deployment(deployment_id)
        if row["state"] == "destroyed":
            raise ConflictError("deployment is already destroyed")
        active = self.store.active_jobs_for_worker(row["worker_id"])
        if active:
            raise ConflictError(
                "worker holds active jobs: "
                + ", ".join(sorted(job["id"] for job in active))
            )
        driver = self.drivers.get(row["provider"])
        if driver is None:
            raise DeploymentError(
                f"provider driver {row['provider']!r} is not configured; "
                "destroy the resource manually before retiring the credential"
            )
        # Provider absence is proven before the credential is disabled.
        destroy_and_verify_absent(driver, row["resource_id"])
        self.store.disable_token(row["token_id"])
        return self.store.update_deployment(
            deployment_id,
            state="destroyed",
            detail=f"destroyed by {actor}",
            destroyed_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )


def _agent_status(agent: dict[str, Any] | None, now: datetime) -> str:
    if agent is None:
        return "not_registered"
    try:
        heartbeat = datetime.fromisoformat(str(agent.get("heartbeat_at") or ""))
    except ValueError:
        return "stale"
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=UTC)
    if (now - heartbeat).total_seconds() > AGENT_STALE_SECONDS:
        return "stale"
    return str(agent.get("state") or "connected")


def _deployment_state(provider_state: str, agent_status: str) -> str:
    provider = provider_state.lower()
    if provider in {"running", "active"}:
        if agent_status == "not_registered":
            return "starting"
        if agent_status == "stale":
            return "stale"
        if agent_status == "draining":
            return "draining"
        return "active"
    if provider in {"pending", "deploying", "downloading", "creating", "allocating"}:
        return "allocating" if provider == "allocating" else "creating"
    if provider in {"failed", "error", "stopped", "stopping"}:
        return "provider-failed"
    return "starting"


def _public_pool(pool: PoolSpec) -> dict[str, Any]:
    document = pool.model_dump(mode="json")
    document.pop("provider_templates", None)
    return document


__all__ = [
    "DeploymentError",
    "DeploymentService",
    "RegistryError",
    "build_drivers_from_env",
]
