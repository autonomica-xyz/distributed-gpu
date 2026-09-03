"""Read-only operator workbench: overview aggregation and UI asset loading.

The workbench is a dependency-free HTML/JS page served by the control plane.
It never bypasses admission: every mutation still travels the authenticated
``/v1`` API. This module only assembles read-only dashboards and loads the UI
asset so the API layer stays thin.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from importlib import resources
from typing import Any

from .service import Orchestrator


UTC = timezone.utc
AGENT_STALE_SECONDS = 120.0


def dashboard_html() -> str:
    """Return the workbench HTML, reading the packaged asset once and caching."""

    cache = dashboard_html.__dict__
    text = cache.get("text")
    if text is None:
        text = (resources.files(__package__) / "dashboard.html").read_text(encoding="utf-8")
        cache["text"] = text
    return text


HISTORICAL_DEPLOYMENT_STATES = {"absent", "destroyed", "failed"}


def fleet_agents(orchestrator: Orchestrator) -> list[dict[str, Any]]:
    """Annotate retained agents as current or historical provider inventory.

    Agent rows are durable audit records, so a stale heartbeat alone must not
    imply that a provider GPU still exists.  For controller-managed workers,
    the latest deployment ledger state is the lifecycle authority.
    """

    latest_deployments: dict[str, dict[str, Any]] = {}
    for deployment in orchestrator.store.list_deployments():
        latest_deployments.setdefault(deployment["worker_id"], deployment)

    stale_threshold = orchestrator.store.clock() - timedelta(seconds=AGENT_STALE_SECONDS)
    documents = []
    for raw_agent in orchestrator.store.list_agents():
        agent = dict(raw_agent)
        deployment = latest_deployments.get(agent["id"])
        deployment_state = deployment.get("state") if deployment else None
        historical = deployment_state in HISTORICAL_DEPLOYMENT_STATES
        heartbeat_at = _parse_iso(agent.get("heartbeat_at"))
        stale = heartbeat_at is None or heartbeat_at < stale_threshold
        agent.update(
            {
                "deployment_id": deployment.get("id") if deployment else None,
                "deployment_state": deployment_state,
                "destroyed_at": deployment.get("destroyed_at") if deployment else None,
                "historical": historical,
                "fleet_status": "past" if historical else ("stale" if stale else "active"),
            }
        )
        documents.append(agent)
    return documents


def _fleet_summary(orchestrator: Orchestrator) -> dict[str, Any]:
    slots_by_state: dict[str, int] = {}
    slots_by_provider: dict[str, int] = {}
    slots_by_model: dict[str, int] = {}
    slots_by_capability: dict[str, int] = {}
    total_vram_mb = 0
    slot_total = 0
    agents_stale = 0
    agents_history = 0
    agents = fleet_agents(orchestrator)
    for agent in agents:
        if agent["historical"]:
            agents_history += 1
            continue
        if agent["fleet_status"] == "stale":
            agents_stale += 1
        for slot in agent["slots"]:
            slot_total += 1
            state = "offline" if agent["fleet_status"] == "stale" else slot["state"]
            slots_by_state[state] = slots_by_state.get(state, 0) + 1
            slots_by_provider[agent["provider"]] = slots_by_provider.get(agent["provider"], 0) + 1
            slots_by_model[slot["model"]] = slots_by_model.get(slot["model"], 0) + 1
            for capability in slot["capabilities"]:
                slots_by_capability[capability] = slots_by_capability.get(capability, 0) + 1
            total_vram_mb += int(slot["vram_mb"])
    return {
        "agents": len(agents) - agents_history,
        "agents_stale": agents_stale,
        "agents_history": agents_history,
        "slots_total": slot_total,
        "slots_by_state": slots_by_state,
        "slots_by_provider": slots_by_provider,
        "slots_by_model": slots_by_model,
        "slots_by_capability": slots_by_capability,
        "total_vram_mb": total_vram_mb,
    }


def build_overview(orchestrator: Orchestrator, principal: dict[str, Any]) -> dict[str, Any]:
    """Assemble one overview document, scoped by token role.

    Clients see their own project's job histogram and aggregate fleet capacity
    (counts only, no identifiers). Controllers and admins additionally see the
    route-grouped backlog across the whole fleet.
    """

    role = principal["role"]
    subject = principal["subject"]
    project_scoped = role == "client"
    project_id = subject if project_scoped else None

    raw_stats = orchestrator.store.job_stats(project_id=project_id)
    states = [
        "pending",
        "leased",
        "running",
        "succeeded",
        "dead-letter",
        "cancelled",
    ]
    jobs_by_state = {state: int(raw_stats.get(state, 0)) for state in states}
    jobs_by_state["total"] = sum(jobs_by_state.values())

    backlog: list[dict[str, Any]] = []
    if not project_scoped:
        backlog = orchestrator.store.backlog()

    return {
        "server_time": datetime.now(UTC).isoformat(timespec="seconds"),
        "role": role,
        "subject": subject,
        "project_id": project_id,
        "jobs_by_state": jobs_by_state,
        "backlog": backlog,
        "fleet": _fleet_summary(orchestrator),
        "workloads": list(orchestrator.registry.names()),
        "can_administer": role in {"admin", "controller"},
    }


def build_queues(
    orchestrator: Orchestrator,
    principal: dict[str, Any],
    *,
    window_hours: int = 24,
) -> list[dict[str, Any]]:
    """Build generic workload/route queue health and drain estimates."""
    project_id = principal["subject"] if principal["role"] == "client" else None
    observed = {
        (row["workload"], row["route"]): row
        for row in orchestrator.store.queue_stats(
            project_id=project_id, window_hours=window_hours
        )
    }
    documents: list[dict[str, Any]] = []
    for workload_name in orchestrator.registry.names():
        workload = orchestrator.registry.get(workload_name)
        for route_name, route in workload.routes.items():
            row = observed.get((workload_name, route_name), {})
            pending = int(row.get("pending_jobs") or 0)
            leased = int(row.get("leased_jobs") or 0)
            running = int(row.get("running_jobs") or 0)
            backlog_jobs = pending + leased + running
            backlog_units = float(row.get("backlog_units") or 0)
            completed = int(row.get("recent_completed_jobs") or 0)
            failed = int(row.get("recent_failed_jobs") or 0)
            completed_units = float(row.get("recent_completed_units") or 0)
            busy_hours = float(row.get("recent_busy_ms") or 0) / 3_600_000
            units_per_worker_hour = completed_units / busy_hours if busy_hours > 0 else None

            active_workers = row.get("active_workers") or []
            recent_workers = row.get("workers") or []
            active_slots = {worker.get("slot_id") for worker in active_workers}
            recent_slots = {worker.get("slot_id") for worker in recent_workers}
            serving_slots = len(active_slots) or len(recent_slots)
            effective_units_per_hour = (
                units_per_worker_hour * serving_slots
                if units_per_worker_hour is not None and serving_slots
                else None
            )
            eta_seconds: float | None
            if backlog_jobs == 0:
                eta_seconds = 0.0
            elif effective_units_per_hour and effective_units_per_hour > 0:
                eta_seconds = backlog_units / effective_units_per_hour * 3600
            else:
                eta_seconds = None

            active_by_slot = {worker.get("slot_id"): worker for worker in active_workers}
            workers = []
            seen: set[str] = set()
            for worker in [*active_workers, *recent_workers]:
                slot_id = str(worker.get("slot_id") or "")
                if not slot_id or slot_id in seen:
                    continue
                seen.add(slot_id)
                recent = next(
                    (item for item in recent_workers if item.get("slot_id") == slot_id), {}
                )
                current = active_by_slot.get(slot_id, {})
                workers.append(
                    {
                        "worker_id": worker.get("worker_id"),
                        "slot_id": slot_id,
                        "model": worker.get("model"),
                        "provider": worker.get("provider"),
                        "active": bool(current),
                        "job_id": current.get("job_id"),
                        "state": current.get("state"),
                        "completed_jobs": int(recent.get("completed_jobs") or 0),
                        "avg_duration_ms": float(recent.get("avg_duration_ms") or 0),
                    }
                )

            documents.append(
                {
                    "workload": workload_name,
                    "route": route_name,
                    "capability": route.compatibility.capability,
                    "cost_unit": row.get("cost_unit") or route.cost.unit,
                    "pending_jobs": pending,
                    "leased_jobs": leased,
                    "running_jobs": running,
                    "backlog_jobs": backlog_jobs,
                    "backlog_units": backlog_units,
                    "succeeded_jobs": int(row.get("succeeded_jobs") or 0),
                    "dead_letter_jobs": int(row.get("dead_letter_jobs") or 0),
                    "recent_completed_jobs": completed,
                    "recent_failed_jobs": failed,
                    "failure_rate": failed / (completed + failed) if completed + failed else None,
                    "units_per_worker_hour": units_per_worker_hour,
                    "effective_units_per_hour": effective_units_per_hour,
                    "eta_seconds": eta_seconds,
                    "window_hours": window_hours,
                    "workers": workers,
                }
            )
    return sorted(
        documents,
        key=lambda row: (
            -row["backlog_jobs"],
            -row["recent_completed_jobs"],
            row["workload"],
            row["route"],
        ),
    )


def build_recent(
    orchestrator: Orchestrator,
    principal: dict[str, Any],
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    project_id = principal["subject"] if principal["role"] == "client" else None
    return orchestrator.store.recent_events(limit=limit, project_id=project_id)


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None
