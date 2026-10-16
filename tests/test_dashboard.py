from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from sparehour.api import create_app

from conftest import register_agent


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def _app(orchestrator):
    return TestClient(create_app(orchestrator, enable_dashboard=True))


def _seed(orchestrator, store):
    register_agent(store, "agent-a", slots=2)
    client_token = store.issue_token(role="client", subject="project-a")
    other_token = store.issue_token(role="client", subject="project-b")
    admin_token = store.issue_token(role="admin", subject="operator")
    controller_token = store.issue_token(role="controller", subject="fleet-ops")
    job_a = orchestrator.submit(
        project_id="project-a", workload="demo-gpu", payload={"message": "hello a"}
    )
    job_b = orchestrator.submit(
        project_id="project-b", workload="demo-gpu", payload={"message": "hello b"}
    )
    return {
        "client": client_token,
        "other": other_token,
        "admin": admin_token,
        "controller": controller_token,
        "job_a": job_a["id"],
        "job_b": job_b["id"],
    }


def test_dashboard_page_is_served_when_enabled(orchestrator):
    api = _app(orchestrator)
    page = api.get("/dashboard")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]


def test_dashboard_is_not_mounted_by_default(orchestrator):
    api = TestClient(create_app(orchestrator))
    assert api.get("/dashboard").status_code == 404
    assert api.get("/v1/dashboard/overview").status_code == 404


def test_dashboard_endpoints_require_auth(orchestrator):
    api = _app(orchestrator)
    assert api.get("/v1/dashboard/overview").status_code == 401
    assert api.get("/v1/dashboard/recent").status_code == 401
    assert api.get("/v1/dashboard/queues").status_code == 401


def test_agent_tokens_cannot_use_dashboard(orchestrator, store):
    agent_token = store.issue_token(role="agent", subject="agent-a")
    api = _app(orchestrator)
    res = api.get("/v1/dashboard/overview", headers=bearer(agent_token))
    assert res.status_code == 403


def test_overview_aggregates_fleet_and_job_counts(orchestrator, store):
    ctx = _seed(orchestrator, store)
    api = _app(orchestrator)
    res = api.get("/v1/dashboard/overview", headers=bearer(ctx["admin"]))
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["role"] == "admin"
    assert body["can_administer"] is True
    assert body["jobs_by_state"]["pending"] == 2
    assert body["jobs_by_state"]["total"] == 2
    assert body["fleet"]["slots_total"] == 2
    assert body["fleet"]["slots_by_state"]["ready"] == 2
    assert body["fleet"]["total_vram_mb"] == 24576 * 2
    # admin sees the full cross-tenant backlog, grouped by workload/route/state
    pending = [row for row in body["backlog"] if row["state"] == "pending"]
    assert sum(row["jobs"] for row in pending) == 2
    assert sorted(body["workloads"])


def test_destroyed_provider_agent_is_history_not_live_capacity(orchestrator, store):
    register_agent(store, "past-gpu")
    store.record_deployment(
        pool_id="pool-a", provider="salad", resource_id="gone-resource",
        resource_name="gone-resource", worker_id="past-gpu", token_id="past-token",
        workload="demo-gpu", route="default", capability="demo-gpu",
        image="example.invalid/worker@sha256:" + "a" * 64,
        hourly_cost_usd=0.06, state="absent",
        detail="not present in authoritative provider inventory",
    )
    admin = store.issue_token(role="admin", subject="operator")
    api = _app(orchestrator)

    overview = api.get("/v1/dashboard/overview", headers=bearer(admin)).json()
    assert overview["fleet"]["agents"] == 0
    assert overview["fleet"]["agents_history"] == 1
    assert overview["fleet"]["slots_total"] == 0

    agents = api.get("/v1/control/agents", headers=bearer(admin)).json()
    assert agents[0]["historical"] is True
    assert agents[0]["fleet_status"] == "past"
    assert agents[0]["deployment_state"] == "absent"


def test_client_overview_is_project_scoped(orchestrator, store):
    ctx = _seed(orchestrator, store)
    api = _app(orchestrator)
    res = api.get("/v1/dashboard/overview", headers=bearer(ctx["client"]))
    assert res.status_code == 200
    body = res.json()
    assert body["role"] == "client"
    assert body["project_id"] == "project-a"
    assert body["jobs_by_state"]["pending"] == 1
    assert body["jobs_by_state"]["total"] == 1
    assert body["can_administer"] is False
    # clients do not see other projects' backlog
    assert body["backlog"] == []
    # clients still see aggregate fleet capacity
    assert body["fleet"]["slots_total"] == 2


def test_queue_stats_include_depth_eta_and_workers(orchestrator, store, clock):
    ctx = _seed(orchestrator, store)
    claim = orchestrator.claim("agent-a-slot-0")
    orchestrator.confirm(claim["id"], "agent-a-slot-0", claim["lease_token"])
    clock.value += timedelta(seconds=10)
    orchestrator.complete(
        claim["id"],
        "agent-a-slot-0",
        claim["lease_token"],
        [{
            "name": "result",
            "uri": "artifact://" + "a" * 64,
            "media_type": "application/json",
            "size": 10,
            "sha256": "a" * 64,
        }],
    )
    api = _app(orchestrator)
    response = api.get(
        "/v1/dashboard/queues?window_hours=24", headers=bearer(ctx["admin"])
    )
    assert response.status_code == 200
    queue = next(row for row in response.json() if row["workload"] == "demo-gpu")
    assert queue["pending_jobs"] == 1
    assert queue["recent_completed_jobs"] == 1
    assert queue["effective_units_per_hour"] == pytest.approx(360.0, rel=1e-4)
    assert queue["eta_seconds"] == pytest.approx(10.0, rel=1e-4)
    assert queue["workers"][0]["slot_id"] == "agent-a-slot-0"
    assert queue["workers"][0]["completed_jobs"] == 1

    project = api.get("/v1/dashboard/queues", headers=bearer(ctx["client"])).json()
    project_queue = next(row for row in project if row["workload"] == "demo-gpu")
    assert project_queue["backlog_jobs"] == 0
    assert project_queue["eta_seconds"] == 0


def test_recent_activity_is_role_scoped(orchestrator, store):
    ctx = _seed(orchestrator, store)
    api = _app(orchestrator)

    admin_recent = api.get("/v1/dashboard/recent", headers=bearer(ctx["admin"]))
    assert admin_recent.status_code == 200
    admin_jobs = {row["job_id"] for row in admin_recent.json()}
    assert {ctx["job_a"], ctx["job_b"]} <= admin_jobs

    client_recent = api.get("/v1/dashboard/recent", headers=bearer(ctx["client"]))
    assert client_recent.status_code == 200
    client_jobs = {row["job_id"] for row in client_recent.json()}
    assert client_jobs == {ctx["job_a"]}


def test_dashboard_submit_flows_through_bounded_admission(orchestrator, store):
    ctx = _seed(orchestrator, store)
    api = _app(orchestrator)
    res = api.post(
        "/v1/jobs",
        headers=bearer(ctx["client"]),
        json={
            "project_id": "project-a",
            "workload": "demo-gpu",
            "payload": {"message": "from the workbench"},
        },
    )
    assert res.status_code == 201, res.text
    assert res.json()["state"] == "pending"

    # the trust boundary holds: extra payload fields are rejected
    bad = api.post(
        "/v1/jobs",
        headers=bearer(ctx["client"]),
        json={
            "project_id": "project-a",
            "workload": "demo-gpu",
            "payload": {"message": "x", "untrusted_field": "nope"},
        },
    )
    assert bad.status_code == 422
