"""Bounded, report-first scaling evaluation engine.

Autonomous provider mutation is gated: ``allow_apply`` defaults to ``False``
and an explicit global pause check blocks unintended scale-out.  The engine
records every evaluation so the dashboard can explain why an action did or
did not happen.
"""

from __future__ import annotations

import os
import secrets
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .autoscaler import ProviderCandidate, ScalingInputs, estimate_target_slots
from .controller import FleetPlanner, PoolSpec
from .deploy import TERMINAL_DEPLOYMENT_STATES
from .fleet_economics import Candidate, rank_by_cost_to_target
from .providers.base import destroy_and_verify_absent
from .scaling_policy import load_rule
from .store import ConflictError, NotFoundError, Store


UTC = timezone.utc
DEFAULT_EVALUATION_INTERVAL_SECONDS = 60


@dataclass(frozen=True)
class ScalingEvaluation:
    rule_id: str
    rule_version: int
    decision: str
    blocked_reason: str | None
    current_slots: int
    in_flight_slots: int
    desired_slots: int
    proposed_additions: int
    selected_candidate: dict[str, Any] | None
    alternatives: list[dict[str, Any]]
    projected_eta_hours: float | None
    projected_cost_usd: float | None
    snapshot: dict[str, Any]
    evaluation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        def safe(value: Any) -> Any:
            if isinstance(value, float) and not math.isfinite(value): return None
            if isinstance(value, dict): return {key: safe(item) for key, item in value.items()}
            if isinstance(value, list): return [safe(item) for item in value]
            return value
        return safe({
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "decision": self.decision,
            "blocked_reason": self.blocked_reason,
            "current_slots": self.current_slots,
            "in_flight_slots": self.in_flight_slots,
            "desired_slots": self.desired_slots,
            "proposed_additions": self.proposed_additions,
            "selected_candidate": self.selected_candidate,
            "alternatives": self.alternatives,
            "projected_eta_hours": self.projected_eta_hours,
            "projected_cost_usd": self.projected_cost_usd,
            "snapshot": self.snapshot,
            "evaluation_id": self.evaluation_id,
        })


class ScalingEngine:
    """Evaluate one scaling rule against live queue and provider data."""

    def __init__(
        self,
        store: Store,
        registry: Any,
        *,
        pools_directory: str | Path | None = None,
        drivers: dict[str, Any] | None = None,
        public_controller_url: str = "",
        wheel_url: str | None = None,
        allow_apply: bool = False,
    ):
        self.store = store
        self.registry = registry
        self.pools_directory = Path(pools_directory) if pools_directory else None
        self.drivers = drivers or {}
        self.public_controller_url = public_controller_url
        self.wheel_url = wheel_url
        self.allow_apply = allow_apply
        # Read the durable switch at evaluation time; keeping construction
        # side-effect-free also permits API startup before migrations settle.
        self._global_pause = False
        self._recover_pending_deploys()

    def _recover_pending_deploys(self) -> None:
        """Resolve interrupted deploy actions from provider inventory before retrying."""
        terminal = {"failed", "destroyed", "absent"}
        now = self.store.clock()

        def release_reservation(action: dict[str, Any]) -> None:
            reservation = action.get("reservation") or {}
            if reservation.get("id"):
                self.store.release_budget_reservation(reservation["id"])

        def _confirm_in_inventory(deployment: dict[str, Any]) -> bool:
            driver = self.drivers.get(deployment["provider"])
            if not driver:
                return False
            try:
                return any(
                    resource.worker_id == deployment["worker_id"]
                    or resource.resource_id == deployment["worker_id"]
                    or resource.resource_id == deployment.get("resource_id")
                    or resource.name == deployment["worker_id"]
                    or resource.name == deployment.get("resource_name")
                    for resource in driver.inventory()
                )
            except Exception:
                return False

        def _startup_expired(deployment: dict[str, Any]) -> bool:
            deadline = deployment.get("startup_deadline")
            if not deadline:
                return False
            try:
                return now > datetime.fromisoformat(deadline)
            except (TypeError, ValueError):
                return False

        def _terminate_unconfirmed(deployment: dict[str, Any]) -> bool:
            """Attempt to destroy a deployment whose startup deadline has passed.

            Returns True only when the provider confirms the resource is now
            absent, so the credential can be safely disabled.
            """
            driver = self.drivers.get(deployment["provider"])
            if not driver:
                return False

            # Try to find the actual provider resource id first; for providers
            # like Vast the contract id is numeric and differs from the worker id.
            resource_id = deployment.get("resource_id") or deployment["worker_id"]
            try:
                for resource in driver.inventory():
                    if (
                        resource.worker_id == deployment["worker_id"]
                        or resource.resource_id == deployment["worker_id"]
                        or resource.resource_id == (deployment.get("resource_id") or "")
                        or resource.name == deployment["worker_id"]
                        or resource.name == (deployment.get("resource_name") or "")
                    ):
                        resource_id = resource.resource_id
                        if deployment["state"] == "preparing":
                            self.store.update_deployment(
                                deployment["id"],
                                resource_id=resource.resource_id,
                                resource_name=resource.name,
                            )
                        break
            except Exception:
                pass

            try:
                destroy_and_verify_absent(
                    driver,
                    resource_id,
                    timeout_seconds=30,
                    poll_seconds=2,
                )
            except Exception:
                # The provider may not accept the recorded resource id (e.g. Vast
                # needs an integer contract id). Fall back to an inventory check
                # before giving up.
                if _confirm_in_inventory(deployment):
                    return False

            # If destroy succeeded and the resource is gone, or inventory already
            # confirms absence, retire the credential and mark the deployment.
            if _confirm_in_inventory(deployment):
                return False
            if deployment.get("token_id"):
                self.store.disable_token(deployment["token_id"])
            self.store.update_deployment(
                deployment["id"],
                state="failed",
                detail="launch failed; resource not confirmed in inventory after startup deadline",
                failure_classification="provider_launch_failed",
            )
            return True

        def _promote_to_requested(deployment: dict[str, Any]) -> None:
            if deployment["state"] == "preparing":
                self.store.update_deployment(
                    deployment["id"],
                    state="requested",
                    resource_id=deployment.get("resource_id") or deployment["worker_id"],
                    resource_name=deployment.get("resource_name") or deployment["worker_id"],
                )

        for action in self.store.list_actions(limit=1000, state="pending"):
            if action["action"] == "drain":
                self._recover_pending_drain(action)
                continue
            if action["action"] != "deploy":
                continue
            try:
                # Gather every deployment associated with this action, including
                # the primary deployment recorded on the action and any siblings
                # created from the same multi-worker scale-up call.
                seen: set[str] = set()
                deployments: list[dict[str, Any]] = []
                if action.get("deployment_id"):
                    seen.add(action["deployment_id"])
                    deployments.append(self.store.get_deployment(action["deployment_id"]))
                for d in self.store.list_deployments(action_id=action["id"], limit=100):
                    if d["id"] not in seen:
                        seen.add(d["id"])
                        deployments.append(d)

                # No deployment rows yet: fall back to the recorded provider request.
                if not deployments:
                    provider_request = action.get("provider_request") or {}
                    deadline = provider_request.get("startup_deadline")
                    if deadline:
                        try:
                            if now > datetime.fromisoformat(deadline):
                                release_reservation(action)
                                self.store.update_action_state(
                                    action["id"],
                                    state="failed",
                                    provider_result={"failure": "startup_deadline_exceeded"},
                                )
                        except (TypeError, ValueError):
                            pass
                    continue

                confirmed: list[dict[str, Any]] = []
                failed: list[dict[str, Any]] = []
                unresolved: list[dict[str, Any]] = []
                for deployment in deployments:
                    if deployment["state"] in terminal:
                        failed.append(deployment)
                        continue
                    if _confirm_in_inventory(deployment):
                        confirmed.append(deployment)
                        _promote_to_requested(deployment)
                        continue
                    if _startup_expired(deployment):
                        if _terminate_unconfirmed(deployment):
                            failed.append(deployment)
                        else:
                            unresolved.append(deployment)
                    else:
                        unresolved.append(deployment)

                if unresolved:
                    # Some deployments are still ambiguous. Leave the action
                    # pending for the next recovery pass.
                    continue

                if failed:
                    release_reservation(action)
                    self.store.update_action_state(
                        action["id"],
                        state="failed",
                        deployment_id=confirmed[0]["id"] if confirmed else failed[0]["id"],
                        provider_result={"failure": "partial_deployment_failed" if confirmed else "deployment_failed", "failed": [d["id"] for d in failed]},
                    )
                    continue

                # All non-terminal deployments have been confirmed in inventory.
                self.store.update_action_state(
                    action["id"],
                    state="succeeded",
                    deployment_id=confirmed[0]["id"] if confirmed else None,
                    provider_result={"confirmed": [d["id"] for d in confirmed]},
                )
            except Exception:
                # Retain pending state for the bounded next evaluation; never
                # launch blindly after an ambiguous provider response.
                continue

    def _recover_pending_drain(self, action: dict[str, Any]) -> None:
        """Complete an interrupted scale-down action if the deployment is safe to destroy."""
        deployment_id = action.get("deployment_id")
        if not deployment_id:
            return
        try:
            deployment = self.store.get_deployment(deployment_id)
        except Exception:
            return
        if deployment["state"] in {"destroyed"}:
            self.store.update_action_state(
                action["id"],
                state="succeeded",
                deployment_id=deployment_id,
                provider_result={"recovered": "already_destroyed"},
            )
            return
        if deployment["state"] in TERMINAL_DEPLOYMENT_STATES:
            self.store.update_action_state(
                action["id"],
                state="failed",
                deployment_id=deployment_id,
                provider_result={"failure": "deployment_in_terminal_state", "state": deployment["state"]},
            )
            return
        active = self.store.active_jobs_for_worker(deployment["worker_id"])
        if active:
            # Worker still holds jobs; leave the drain pending and let the
            # next evaluation retry after the sustain window has passed.
            return
        try:
            from .deploy import DeploymentService

            service = DeploymentService(
                registry=self.registry,
                store=self.store,
                drivers=self.drivers,
                pools_directory=str(self.pools_directory) if self.pools_directory else ".",
                public_controller_url=self.public_controller_url,
                wheel_url=self.wheel_url,
            )
            result = service.teardown(deployment_id, actor=f"autoscaler:{action['rule_id']}")
            self.store.update_action_state(
                action["id"],
                state="succeeded",
                deployment_id=deployment_id,
                provider_result={"destroyed": result["id"]},
            )
        except (ConflictError, NotFoundError) as exc:
            if "already destroyed" in str(exc).lower() or isinstance(exc, NotFoundError):
                self.store.update_action_state(
                    action["id"],
                    state="succeeded",
                    deployment_id=deployment_id,
                    provider_result={"recovered": "already_destroyed_or_absent"},
                )
            elif "active jobs" in str(exc).lower():
                # Leave pending; recovery will retry once the worker is idle.
                pass
            else:
                self.store.update_action_state(
                    action["id"],
                    state="failed",
                    deployment_id=deployment_id,
                    provider_result={"error": f"{type(exc).__name__}: {exc}"},
                )
        except Exception as exc:
            # Transient or unclassified teardown failure; leave pending for retry.
            pass

    def evaluate_all(self) -> list[ScalingEvaluation]:
        """Evaluate enabled, non-deleted rules; one broken rule cannot stop fleet safety."""
        self._recover_pending_deploys()
        results = []
        for row in self.store.list_scaling_rules(enabled=True):
            try: results.append(self.evaluate(row["id"]))
            except Exception: continue
        return results

    def _pools(self) -> list[PoolSpec]:
        if self.pools_directory is None or not self.pools_directory.is_dir():
            return []
        return PoolSpec.from_directory(self.pools_directory)

    def _pool(self, pool_id: str) -> PoolSpec | None:
        return next((p for p in self._pools() if p.id == pool_id), None)

    def _is_globally_paused(self) -> bool:
        return bool(self.store.automation_state()["paused"])

    def evaluate(
        self,
        rule_id: str,
        *,
        dry_run: bool = False,
        owner: str | None = None,
    ) -> ScalingEvaluation:
        if owner is None:
            owner = f"autoscaler-{os.getpid()}-{secrets.token_hex(4)}"
        rule_doc = self.store.get_scaling_rule(rule_id)
        rule = load_rule(rule_doc)

        if not rule.enabled or rule.mode == "off":
            return self._record(
                rule,
                decision="no_action",
                blocked="rule_disabled_or_off",
                current_slots=0,
                in_flight=0,
                desired=0,
                proposed=0,
                selected=None,
                alternatives=[],
                eta=None,
                cost=None,
                snapshot={},
                dry_run=dry_run,
            )

        if self._is_globally_paused():
            return self._record(
                rule,
                decision="no_action",
                blocked="automation_globally_paused",
                current_slots=0,
                in_flight=0,
                desired=0,
                proposed=0,
                selected=None,
                alternatives=[],
                eta=None,
                cost=None,
                snapshot={"global_pause": True},
                dry_run=dry_run,
            )

        if not dry_run and not self.store.acquire_scaling_lease(
            rule_id, owner, ttl_seconds=DEFAULT_EVALUATION_INTERVAL_SECONDS
        ):
            return self._record(
                rule,
                decision="no_action",
                blocked="lease_held_by_another_controller",
                current_slots=0,
                in_flight=0,
                desired=0,
                proposed=0,
                selected=None,
                alternatives=[],
                eta=None,
                cost=None,
                snapshot={},
                dry_run=dry_run,
            )

        try:
            result = self._evaluate_rule(rule, dry_run=dry_run)
        finally:
            if not dry_run:
                self.store.release_scaling_lease(rule_id, owner)
        return result

    def _evaluate_rule(self, rule, *, dry_run: bool) -> ScalingEvaluation:
        pool = self._pool(rule.pool_id)
        if pool is None:
            return self._record(
                rule,
                decision="no_action",
                blocked="pool_not_found",
                current_slots=0,
                in_flight=0,
                desired=0,
                proposed=0,
                selected=None,
                alternatives=[],
                eta=None,
                cost=None,
                snapshot={"pool_id": rule.pool_id},
                dry_run=dry_run,
            )

        workload = self.registry.get(pool.workload)
        route = workload.routes[pool.route]
        capability = route.compatibility.capability

        planner = FleetPlanner(self.registry, self.store)
        current = planner._current_slots(pool, capability)
        in_flight = self._in_flight_slots(pool)
        _queued_jobs, queued_units, _running_jobs, running_units = planner._backlog(pool)
        total_units = queued_units + running_units

        # Observed throughput from the recent window
        stats = [
            row
            for row in self.store.queue_stats(window_hours=1)
            if row["workload"] == pool.workload and row["route"] == pool.route
        ]
        stat = stats[0] if stats else {}
        busy_ms = float(stat.get("recent_busy_ms") or 0.0)
        completed_units = float(stat.get("recent_completed_units") or 0.0)
        units_per_gpu_hour = route.cost.default_units_per_gpu_hour
        if busy_ms > 0:
            # busy time is accumulated per slot, so this is already per-GPU.
            units_per_gpu_hour = completed_units / (busy_ms / 3_600_000.0)
        elif rule.trigger.require_observed_throughput and total_units > 0:
            return self._record(
                rule,
                decision="no_action",
                blocked="insufficient_observed_throughput",
                current_slots=current,
                in_flight=in_flight,
                desired=0,
                proposed=0,
                selected=None,
                alternatives=[],
                eta=None,
                cost=None,
                snapshot=self._snapshot(pool, current, in_flight, total_units, units_per_gpu_hour),
                dry_run=dry_run,
            )

        current_throughput = (current + in_flight) * max(0.0, units_per_gpu_hour)
        current_eta = (
            total_units / current_throughput
            if current_throughput > 0
            else float("inf")
        )
        threshold_state = self._threshold_state(rule, current_eta)
        held = {"state": threshold_state, "since": None}
        if not dry_run:
            held = self.store.update_threshold_state(rule.id, threshold_state)
        sustain_seconds = (
            rule.trigger.sustain_seconds
            if threshold_state == "above"
            else rule.scale_down.sustain_seconds
        )
        if held["since"]:
            since = datetime.fromisoformat(held["since"])
            held_seconds = (self.store.clock() - since).total_seconds()
        else:
            held_seconds = 0

        # Scale-down is allowed even when the queue is empty, as long as
        # capacity is present and the below-threshold state has been sustained.
        if threshold_state == "below" and rule.scale_down.enabled and held_seconds >= sustain_seconds:
            return self._scale_down(rule, pool, current, in_flight, total_units, units_per_gpu_hour, current_eta, dry_run)

        if total_units <= 0:
            return self._record(
                rule,
                decision="no_action",
                blocked="queue_is_empty",
                current_slots=current,
                in_flight=in_flight,
                desired=0,
                proposed=0,
                selected=None,
                alternatives=[],
                eta=current_eta,
                cost=None,
                snapshot=self._snapshot(pool, current, in_flight, total_units, units_per_gpu_hour, current_eta=current_eta),
                dry_run=dry_run,
            )

        if threshold_state != "above" or held_seconds < sustain_seconds:
            return self._record(
                rule,
                decision="no_action",
                blocked=(f"eta_{threshold_state}_threshold" if threshold_state != "above" else "trigger_sustain_pending"),
                current_slots=current,
                in_flight=in_flight,
                desired=0,
                proposed=0,
                selected=None,
                alternatives=[],
                eta=current_eta,
                cost=None,
                snapshot=self._snapshot(pool, current, in_flight, total_units, units_per_gpu_hour, current_eta=current_eta),
                dry_run=dry_run,
            )

        # Ask the fleet planner for ranked, funded candidates
        try:
            plan = planner.plan(pool, self.drivers)
        except Exception as exc:
            return self._record(
                rule,
                decision="no_action",
                blocked=f"planning_error:{type(exc).__name__}",
                current_slots=current,
                in_flight=in_flight,
                desired=0,
                proposed=0,
                selected=None,
                alternatives=[],
                eta=current_eta,
                cost=None,
                snapshot=self._snapshot(pool, current, in_flight, total_units, units_per_gpu_hour, current_eta=current_eta),
                dry_run=dry_run,
            )

        # Apply rule-level provider and hourly bounds
        allowed = set(rule.providers.allowed)
        candidates: list[Candidate] = []
        for row in plan.ranked_candidates:
            if row["provider"] not in allowed:
                continue
            self.store.record_observation(kind="market", provider=row["provider"], offer_id=row["offer_id"], gpu_model=row.get("gpu_model"), route=pool.route, value=row.get("hourly_cost_usd"))
            if rule.providers.require_known_availability and row.get("available_count") is None:
                continue
            failures = self.store.recent_observations(kind="startup", provider=row["provider"], offer_id=row["offer_id"], seconds=900)
            if len(failures) >= 3 and not any(item["success_count"] for item in failures[-3:]):
                continue
            spec = getattr(rule.providers, row["provider"])
            if spec is None or row["hourly_cost_usd"] > spec.max_hourly_cost_usd:
                continue
            candidates.append(
                Candidate(
                    provider=row["provider"],
                    offer_id=row["offer_id"],
                    gpu_model=row["gpu_model"],
                    hourly_cost_usd=row["hourly_cost_usd"],
                    startup_hours=route.scaling.startup_minutes / 60.0,
                    units_per_gpu_hour=units_per_gpu_hour,
                    reliability=row.get("reliability") or 0.8,
                )
            )

        ranked = rank_by_cost_to_target(candidates, units_to_drain=total_units, current_slots=current + in_flight, current_units_per_gpu_hour=units_per_gpu_hour)
        if not ranked:
            return self._record(
                rule,
                decision="no_action",
                blocked="no_rule_eligible_candidate",
                current_slots=current,
                in_flight=in_flight,
                desired=0,
                proposed=0,
                selected=None,
                alternatives=[],
                eta=current_eta,
                cost=None,
                snapshot=self._snapshot(pool, current, in_flight, total_units, units_per_gpu_hour, current_eta=current_eta),
                dry_run=dry_run,
            )

        selected_ranked = ranked[0]
        selected = {
            "provider": selected_ranked.candidate.provider,
            "offer_id": selected_ranked.candidate.offer_id,
            "gpu_model": selected_ranked.candidate.gpu_model,
            "hourly_cost_usd": selected_ranked.candidate.hourly_cost_usd,
            "estimated_completion_hours": selected_ranked.completion_hours,
            "estimated_total_cost_usd": selected_ranked.total_cost_usd,
        }
        alternatives = [
            {
                "provider": r.candidate.provider,
                "offer_id": r.candidate.offer_id,
                "gpu_model": r.candidate.gpu_model,
                "hourly_cost_usd": r.candidate.hourly_cost_usd,
                "estimated_completion_hours": r.completion_hours,
                "estimated_total_cost_usd": r.total_cost_usd,
            }
            for r in ranked[1:]
        ]

        selected_provider_limit = getattr(rule.providers, selected["provider"])
        credit = None
        if selected_provider_limit and selected_provider_limit.require_known_credit:
            driver = self.drivers.get(selected["provider"])
            if driver is not None:
                try:
                    credit = driver.account_credit()
                except Exception:
                    credit = None

        startup_hours = route.scaling.startup_minutes / 60.0
        decision = estimate_target_slots(
            ScalingInputs(
                queued_units=total_units,
                running_units=0.0,
                current_slots=current + in_flight,
                units_per_gpu_hour=units_per_gpu_hour,
                target_completion_hours=rule.scale_up.target_eta_hours,
                startup_hours=startup_hours,
                min_slots=0,
                max_slots=rule.scale_up.max_provider_slots,
                hourly_cost_usd=selected["hourly_cost_usd"],
                hourly_budget_usd=rule.budgets.max_rule_hourly_usd,
                available_credit_usd=credit,
                require_known_credit=bool(
                    selected_provider_limit and selected_provider_limit.require_known_credit
                ),
            )
        )

        if decision.blocked_reason:
            return self._record(
                rule,
                decision="no_action",
                blocked=decision.blocked_reason,
                current_slots=current,
                in_flight=in_flight,
                desired=decision.target_slots,
                proposed=0,
                selected=selected,
                alternatives=alternatives,
                eta=decision.estimated_eta_hours_target,
                cost=decision.projected_drain_cost_usd,
                snapshot=self._snapshot(pool, current, in_flight, total_units, units_per_gpu_hour, current_eta=current_eta),
                dry_run=dry_run,
            )

        desired = decision.target_slots
        requested = max(0, desired - current - in_flight)
        proposed = min(requested, rule.scale_up.max_additions_per_evaluation)

        if rule.last_action_at and (self.store.clock() - datetime.fromisoformat(rule.last_action_at)).total_seconds() < rule.scale_up.cooldown_seconds:
            return self._record(rule, decision="no_action", blocked="scale_up_cooldown", current_slots=current, in_flight=in_flight, desired=desired, proposed=0, selected=selected, alternatives=alternatives, eta=decision.estimated_eta_hours_target, cost=decision.projected_drain_cost_usd, snapshot=self._snapshot(pool, current, in_flight, total_units, units_per_gpu_hour, current_eta=current_eta), dry_run=dry_run)
        if self.store.count_recent_actions(rule.id) >= rule.budgets.max_daily_actions:
            return self._record(rule, decision="no_action", blocked="daily_action_budget_exceeded", current_slots=current, in_flight=in_flight, desired=desired, proposed=0, selected=selected, alternatives=alternatives, eta=decision.estimated_eta_hours_target, cost=decision.projected_drain_cost_usd, snapshot=self._snapshot(pool, current, in_flight, total_units, units_per_gpu_hour, current_eta=current_eta), dry_run=dry_run)

        if proposed <= 0:
            return self._record(
                rule,
                decision="no_action",
                blocked="capacity_already_sufficient",
                current_slots=current,
                in_flight=in_flight,
                desired=desired,
                proposed=0,
                selected=selected,
                alternatives=alternatives,
                eta=decision.estimated_eta_hours_target,
                cost=decision.projected_drain_cost_usd,
                snapshot=self._snapshot(pool, current, in_flight, total_units, units_per_gpu_hour, current_eta=current_eta),
                dry_run=dry_run,
            )

        if rule.budgets.max_projected_drain_usd is not None and decision.projected_drain_cost_usd is not None:
            if decision.projected_drain_cost_usd > rule.budgets.max_projected_drain_usd:
                return self._record(
                    rule,
                    decision="no_action",
                    blocked="projected_drain_budget_exceeded",
                    current_slots=current,
                    in_flight=in_flight,
                    desired=desired,
                    proposed=0,
                    selected=selected,
                    alternatives=alternatives,
                    eta=decision.estimated_eta_hours_target,
                    cost=decision.projected_drain_cost_usd,
                    snapshot=self._snapshot(pool, current, in_flight, total_units, units_per_gpu_hour, current_eta=current_eta),
                    dry_run=dry_run,
                )

        # Report-only stops here. Apply only happens when explicitly allowed.
        if rule.mode == "report-only" or not self.allow_apply:
            return self._record(
                rule,
                decision="propose_scale_up",
                blocked=None,
                current_slots=current,
                in_flight=in_flight,
                desired=desired,
                proposed=proposed,
                selected=selected,
                alternatives=alternatives,
                eta=decision.estimated_eta_hours_target,
                cost=decision.projected_drain_cost_usd,
                snapshot=self._snapshot(pool, current, in_flight, total_units, units_per_gpu_hour, current_eta=current_eta),
                dry_run=dry_run,
            )

        # Apply path (only when allow_apply=True and mode=apply)
        # Persist the evaluation before the action: it is the idempotency/audit
        # anchor used to recover an interrupted controller.
        anchor = self._record(rule, decision="apply_scale_up_pending", blocked=None, current_slots=current, in_flight=in_flight, desired=desired, proposed=proposed, selected=selected, alternatives=alternatives, eta=decision.estimated_eta_hours_target, cost=decision.projected_drain_cost_usd, snapshot=self._snapshot(pool, current, in_flight, total_units, units_per_gpu_hour, current_eta=current_eta), dry_run=False)
        startup_deadline_seconds = max(1, rule.scale_up.startup_deadline_seconds or 3600)
        startup_deadline = (
            self.store.clock() + timedelta(seconds=startup_deadline_seconds)
        ).isoformat()

        # Enforce per-rule and per-provider credit budget ceilings against
        # already reserved but not yet reconciled amounts.
        automation_state = self.store.automation_state()
        if automation_state["paused"]:
            return self._record(rule, decision="no_action", blocked="automation_globally_paused", current_slots=current, in_flight=in_flight, desired=desired, proposed=0, selected=selected, alternatives=alternatives, eta=decision.estimated_eta_hours_target, cost=decision.projected_drain_cost_usd, snapshot=self._snapshot(pool, current, in_flight, total_units, units_per_gpu_hour, current_eta=current_eta), dry_run=dry_run)
        projected = float(decision.projected_drain_cost_usd or 0.0)
        rule_reserved = self.store.sum_unexpired_reservations(rule_id=rule.id)
        if rule.budgets.max_projected_drain_usd is not None and rule_reserved + projected > rule.budgets.max_projected_drain_usd:
            return self._record(rule, decision="no_action", blocked="rule_projected_drain_budget_reserved", current_slots=current, in_flight=in_flight, desired=desired, proposed=0, selected=selected, alternatives=alternatives, eta=decision.estimated_eta_hours_target, cost=decision.projected_drain_cost_usd, snapshot=self._snapshot(pool, current, in_flight, total_units, units_per_gpu_hour, current_eta=current_eta), dry_run=dry_run)
        if selected_provider_limit and selected_provider_limit.require_known_credit and credit is not None:
            provider_reserved = self.store.sum_unexpired_reservations(provider=selected["provider"])
            if provider_reserved + projected > credit:
                return self._record(rule, decision="no_action", blocked="provider_credit_reserved", current_slots=current, in_flight=in_flight, desired=desired, proposed=0, selected=selected, alternatives=alternatives, eta=decision.estimated_eta_hours_target, cost=decision.projected_drain_cost_usd, snapshot=self._snapshot(pool, current, in_flight, total_units, units_per_gpu_hour, current_eta=current_eta), dry_run=dry_run)

        try:
            reservation = self.store.reserve_budget(rule_id=rule.id, amount_usd=projected, ttl_seconds=startup_deadline_seconds, provider=selected["provider"], automation_generation=automation_state["generation"])
        except ConflictError as exc:
            return self._record(rule, decision="no_action", blocked=str(exc), current_slots=current, in_flight=in_flight, desired=desired, proposed=0, selected=selected, alternatives=alternatives, eta=decision.estimated_eta_hours_target, cost=decision.projected_drain_cost_usd, snapshot=self._snapshot(pool, current, in_flight, total_units, units_per_gpu_hour, current_eta=current_eta), dry_run=dry_run)
        action = self.store.record_action(
            rule_id=rule.id,
            evaluation_id=anchor.evaluation_id,
            idempotency_key=f"{rule.id}:{rule.version}:{anchor.evaluation_id}:0",
            action="deploy",
            state="pending",
            origin=f"autoscaler:{rule.id}",
            reservation=reservation,
            provider_request={
                "pool_id": pool.id,
                "provider": selected["provider"],
                "offer_id": selected["offer_id"],
                "count": proposed,
                "startup_deadline": startup_deadline,
            },
        )
        if self._is_globally_paused():
            self.store.release_budget_reservation(reservation["id"])
            self.store.update_action_state(action["id"], state="failed", provider_result={"blocked": "automation_globally_paused_before_mutate"})
            return self._record(rule, decision="apply_failed", blocked="automation_globally_paused", current_slots=current, in_flight=in_flight, desired=desired, proposed=proposed, selected=selected, alternatives=alternatives, eta=decision.estimated_eta_hours_target, cost=decision.projected_drain_cost_usd, snapshot=self._snapshot(pool, current, in_flight, total_units, units_per_gpu_hour, current_eta=current_eta), action_id=action["id"], dry_run=dry_run)

        try:
            from .deploy import DeploymentService

            service = DeploymentService(
                registry=self.registry,
                store=self.store,
                drivers=self.drivers,
                pools_directory=str(self.pools_directory) if self.pools_directory else ".",
                public_controller_url=self.public_controller_url,
                wheel_url=self.wheel_url,
            )
            created = service.deploy(
                pool_id=pool.id,
                provider=selected["provider"],
                count=proposed,
                expires_hours=24,
                offer_id=selected["offer_id"],
                actor=f"autoscaler:{rule.id}",
                rule_id=rule.id, action_id=action["id"],
                startup_deadline=startup_deadline,
                require_known_availability=rule.providers.require_known_availability,
            )
            failed = [row for row in created if row["state"] == "failed"]
            still_preparing = [row for row in created if row["state"] == "preparing"]
            for row in created:
                self.store.record_observation(kind="startup", provider=selected["provider"], offer_id=selected["offer_id"], gpu_model=selected.get("gpu_model"), route=pool.route, success=row["state"] not in ("failed", "preparing"))
            if failed:
                self.store.release_budget_reservation(reservation["id"])
                self.store.update_action_state(action["id"], state="failed", provider_result={"failed": [row["id"] for row in failed]})
                return self._record(rule, decision="apply_failed", blocked="deploy_failed", current_slots=current, in_flight=in_flight, desired=desired, proposed=proposed, selected=selected, alternatives=alternatives, eta=decision.estimated_eta_hours_target, cost=decision.projected_drain_cost_usd, snapshot=self._snapshot(pool, current, in_flight, total_units, units_per_gpu_hour, current_eta=current_eta), action_id=action["id"], dry_run=dry_run)
            if still_preparing:
                # Provider create response was ambiguous and inventory could not
                # be consulted. Leave the action pending for recovery.
                return self._record(
                    rule,
                    decision="apply_scale_up_pending",
                    blocked="launch_ambiguous_awaiting_inventory",
                    current_slots=current,
                    in_flight=in_flight,
                    desired=desired,
                    proposed=proposed,
                    selected=selected,
                    alternatives=alternatives,
                    eta=decision.estimated_eta_hours_target,
                    cost=decision.projected_drain_cost_usd,
                    snapshot=self._snapshot(pool, current, in_flight, total_units, units_per_gpu_hour, current_eta=current_eta),
                    action_id=action["id"],
                    dry_run=dry_run,
                )
            self.store.update_action_state(
                action["id"],
                state="succeeded",
                deployment_id=created[0]["id"] if created else None,
                provider_result={"created": [c["id"] for c in created]},
            )
            return self._record(
                rule,
                decision="apply_scale_up",
                blocked=None,
                current_slots=current,
                in_flight=in_flight,
                desired=desired,
                proposed=proposed,
                selected=selected,
                alternatives=alternatives,
                eta=decision.estimated_eta_hours_target,
                cost=decision.projected_drain_cost_usd,
                snapshot=self._snapshot(pool, current, in_flight, total_units, units_per_gpu_hour, current_eta=current_eta),
                action_id=action["id"],
                dry_run=dry_run,
            )
        except Exception as exc:
            self.store.release_budget_reservation(reservation["id"])
            self.store.update_action_state(
                action["id"],
                state="failed",
                provider_result={"error": f"{type(exc).__name__}: {exc}"},
            )
            return self._record(
                rule,
                decision="apply_failed",
                blocked=f"deploy:{type(exc).__name__}",
                current_slots=current,
                in_flight=in_flight,
                desired=desired,
                proposed=proposed,
                selected=selected,
                alternatives=alternatives,
                eta=decision.estimated_eta_hours_target,
                cost=decision.projected_drain_cost_usd,
                snapshot=self._snapshot(pool, current, in_flight, total_units, units_per_gpu_hour, current_eta=current_eta),
                action_id=action["id"],
                dry_run=dry_run,
            )

    def _in_flight_slots(self, pool: PoolSpec) -> int:
        count = 0
        for row in self.store.list_deployments():
            if row["state"] in {"preparing", "requested", "creating", "allocating", "starting"}:
                if row.get("pool_id") == pool.id:
                    count += 1
        return count

    def _scale_down(self, rule, pool, current, in_flight, units, uph, eta, dry_run: bool) -> ScalingEvaluation:
        owned = [row for row in self.store.list_deployments() if row.get("rule_id") == rule.id and row["state"] in {"active", "draining", "stale"}]
        owned_count = len(owned)
        if owned_count <= rule.scale_down.min_provider_slots or not owned:
            return self._record(rule, decision="no_action", blocked="scale_down_minimum_or_no_owned_deployment", current_slots=current, in_flight=in_flight, desired=current, proposed=0, selected=None, alternatives=[], eta=eta, cost=None, snapshot=self._snapshot(pool, current, in_flight, units, uph, current_eta=eta), dry_run=dry_run)
        now = self.store.clock()
        eligible = []
        for row in owned:
            if self.store.active_jobs_for_worker(row["worker_id"]): continue
            last_finish = self.store.last_job_finished_at(row["worker_id"])
            idle_ref = last_finish or row.get("first_heartbeat") or row.get("created_at")
            if not idle_ref: continue
            try: idle = (now - datetime.fromisoformat(idle_ref)).total_seconds()
            except (TypeError, ValueError): idle = 0
            if idle >= rule.scale_down.idle_seconds: eligible.append(row)
        if not eligible:
            return self._record(rule, decision="no_action", blocked="scale_down_idle_sustain_pending", current_slots=current, in_flight=in_flight, desired=current, proposed=0, selected=None, alternatives=[], eta=eta, cost=None, snapshot=self._snapshot(pool, current, in_flight, units, uph, current_eta=eta), dry_run=dry_run)
        target = sorted(eligible, key=lambda row: (-float(row.get("hourly_cost_usd") or 0), row.get("created_at") or ""))[0]
        if dry_run or rule.mode == "report-only" or not self.allow_apply:
            return self._record(rule, decision="propose_scale_down", blocked=None, current_slots=current, in_flight=in_flight, desired=owned_count - 1, proposed=0, selected={"deployment_id": target["id"]}, alternatives=[], eta=eta, cost=None, snapshot=self._snapshot(pool, current, in_flight, units, uph, current_eta=eta), dry_run=dry_run)

        # Do not create duplicate drain actions for the same deployment.
        pending_drain = next(
            (
                a for a in self.store.list_actions(rule_id=rule.id, state="pending", limit=100)
                if a.get("action") == "drain" and a.get("deployment_id") == target["id"]
            ),
            None,
        )
        if pending_drain:
            return self._record(rule, decision="no_action", blocked="scale_down_drain_in_progress", current_slots=current, in_flight=in_flight, desired=owned_count - 1, proposed=0, selected={"deployment_id": target["id"], "action_id": pending_drain["id"]}, alternatives=[], eta=eta, cost=None, snapshot=self._snapshot(pool, current, in_flight, units, uph, current_eta=eta), dry_run=dry_run)

        anchor = self._record(rule, decision="apply_scale_down_pending", blocked=None, current_slots=current, in_flight=in_flight, desired=owned_count - 1, proposed=0, selected={"deployment_id": target["id"]}, alternatives=[], eta=eta, cost=None, snapshot=self._snapshot(pool, current, in_flight, units, uph, current_eta=eta), dry_run=False)
        automation_state = self.store.automation_state()
        if automation_state["paused"]:
            return self._record(rule, decision="no_action", blocked="automation_globally_paused", current_slots=current, in_flight=in_flight, desired=owned_count - 1, proposed=0, selected={"deployment_id": target["id"]}, alternatives=[], eta=eta, cost=None, snapshot=self._snapshot(pool, current, in_flight, units, uph, current_eta=eta), dry_run=dry_run)
        action = self.store.record_action(rule_id=rule.id, evaluation_id=anchor.evaluation_id, idempotency_key=f"{rule.id}:{rule.version}:{anchor.evaluation_id}:0", action="drain", state="pending", deployment_id=target["id"], origin=f"autoscaler:{rule.id}")
        if self._is_globally_paused():
            self.store.update_action_state(action["id"], state="failed", provider_result={"blocked": "automation_globally_paused_before_mutate"})
            return self._record(rule, decision="apply_failed", blocked="automation_globally_paused", current_slots=current, in_flight=in_flight, desired=owned_count - 1, proposed=0, selected={"deployment_id": target["id"]}, alternatives=[], eta=eta, cost=None, snapshot=self._snapshot(pool, current, in_flight, units, uph, current_eta=eta), action_id=action["id"], dry_run=dry_run)
        try:
            self.store.heartbeat_agent(target["worker_id"], draining=True)
            from .deploy import DeploymentService
            service = DeploymentService(registry=self.registry, store=self.store, drivers=self.drivers, pools_directory=str(self.pools_directory), public_controller_url=self.public_controller_url, wheel_url=self.wheel_url)
            result = service.teardown(target["id"], actor=f"autoscaler:{rule.id}")
            self.store.update_action_state(action["id"], state="succeeded", deployment_id=target["id"], provider_result={"destroyed": result["id"]})
            return self._record(rule, decision="apply_scale_down", blocked=None, current_slots=current, in_flight=in_flight, desired=owned_count - 1, proposed=0, selected={"deployment_id": target["id"]}, alternatives=[], eta=eta, cost=None, snapshot=self._snapshot(pool, current, in_flight, units, uph, current_eta=eta), action_id=action["id"], dry_run=False)
        except (ConflictError, NotFoundError) as exc:
            if "already destroyed" in str(exc).lower() or isinstance(exc, NotFoundError):
                self.store.update_action_state(action["id"], state="succeeded", deployment_id=target["id"], provider_result={"recovered": "already_destroyed_or_absent"})
                return self._record(rule, decision="apply_scale_down", blocked=None, current_slots=current, in_flight=in_flight, desired=owned_count - 1, proposed=0, selected={"deployment_id": target["id"]}, alternatives=[], eta=eta, cost=None, snapshot=self._snapshot(pool, current, in_flight, units, uph, current_eta=eta), action_id=action["id"], dry_run=False)
            if "active jobs" in str(exc).lower():
                # Leave the drain pending; recovery will retry when the worker is idle.
                return self._record(rule, decision="apply_scale_down_pending", blocked="drain_worker_has_active_jobs", current_slots=current, in_flight=in_flight, desired=owned_count - 1, proposed=0, selected={"deployment_id": target["id"]}, alternatives=[], eta=eta, cost=None, snapshot=self._snapshot(pool, current, in_flight, units, uph, current_eta=eta), action_id=action["id"], dry_run=False)
            self.store.update_action_state(action["id"], state="failed", deployment_id=target["id"], provider_result={"error": str(exc)})
            return self._record(rule, decision="apply_failed", blocked="drain_failed", current_slots=current, in_flight=in_flight, desired=current, proposed=0, selected={"deployment_id": target["id"]}, alternatives=[], eta=eta, cost=None, snapshot=self._snapshot(pool, current, in_flight, units, uph, current_eta=eta), action_id=action["id"], dry_run=False)
        except Exception as exc:
            # Transient or unclassified teardown failure: leave pending so the
            # next recovery pass can retry before creating a duplicate drain.
            return self._record(rule, decision="apply_scale_down_pending", blocked=f"drain:{type(exc).__name__}", current_slots=current, in_flight=in_flight, desired=owned_count - 1, proposed=0, selected={"deployment_id": target["id"]}, alternatives=[], eta=eta, cost=None, snapshot=self._snapshot(pool, current, in_flight, units, uph, current_eta=eta), action_id=action["id"], dry_run=False)

    def _threshold_state(self, rule, eta_hours: float) -> str:
        if eta_hours > rule.trigger.above_hours:
            return "above"
        if rule.scale_down.enabled and eta_hours < rule.scale_down.below_eta_hours:
            return "below"
        return "normal"

    def _snapshot(
        self,
        pool: PoolSpec,
        current: int,
        in_flight: int,
        units: float,
        units_per_gpu_hour: float,
        current_eta: float | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        if isinstance(current_eta, float) and not math.isfinite(current_eta):
            current_eta = None
        return {
            "timestamp": now,
            "pool_id": pool.id,
            "workload": pool.workload,
            "route": pool.route,
            "current_slots": current,
            "in_flight_slots": in_flight,
            "backlog_units": units,
            "units_per_gpu_hour": units_per_gpu_hour,
            "current_eta_hours": current_eta,
        }

    @staticmethod
    def _safe_float(value: float | None) -> float | None:
        if isinstance(value, float) and math.isfinite(value):
            return value
        return None

    def _record(
        self,
        rule,
        *,
        decision: str,
        blocked: str | None,
        current_slots: int,
        in_flight: int,
        desired: int,
        proposed: int,
        selected: dict[str, Any] | None,
        alternatives: list[dict[str, Any]],
        eta: float | None,
        cost: float | None,
        snapshot: dict[str, Any],
        action_id: str | None = None,
        dry_run: bool = False,
    ) -> ScalingEvaluation:
        eta = self._safe_float(eta)
        cost = self._safe_float(cost)
        evaluation = ScalingEvaluation(
            rule_id=rule.id,
            rule_version=rule.version,
            decision=decision,
            blocked_reason=blocked,
            current_slots=current_slots,
            in_flight_slots=in_flight,
            desired_slots=desired,
            proposed_additions=proposed,
            selected_candidate=selected,
            alternatives=alternatives,
            projected_eta_hours=eta,
            projected_cost_usd=cost,
            snapshot=snapshot,
        )
        if not dry_run:
            record = self.store.record_evaluation(
                rule_id=rule.id,
                rule_version=rule.version,
                snapshot=evaluation.snapshot,
                threshold_state=None,
                current_slots=evaluation.current_slots,
                in_flight_slots=evaluation.in_flight_slots,
                desired_slots=evaluation.desired_slots,
                decision=evaluation.decision,
                blocked_reason=evaluation.blocked_reason,
                selected_candidate=evaluation.selected_candidate,
                alternatives=evaluation.alternatives,
                projected_eta_hours=evaluation.projected_eta_hours,
                projected_cost_usd=evaluation.projected_cost_usd,
            )
            evaluation = ScalingEvaluation(
                rule_id=rule.id,
                rule_version=rule.version,
                decision=decision,
                blocked_reason=blocked,
                current_slots=current_slots,
                in_flight_slots=in_flight,
                desired_slots=desired,
                proposed_additions=proposed,
                selected_candidate=selected,
                alternatives=alternatives,
                projected_eta_hours=eta,
                projected_cost_usd=cost,
                snapshot=snapshot,
                evaluation_id=record["id"],
            )
        return evaluation


__all__ = ["ScalingEngine", "ScalingEvaluation"]
