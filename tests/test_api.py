from __future__ import annotations

from fastapi.testclient import TestClient

from sparehour.api import create_app
from sparehour.artifacts import ArtifactStore

from conftest import register_agent


SHA = "b" * 64


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def test_authenticated_http_lifecycle(orchestrator, store):
    client_token = store.issue_token(role="client", subject="project-a")
    other_token = store.issue_token(role="client", subject="project-b")
    agent_token = store.issue_token(
        role="agent", subject="agent-a", scopes={"routes": ["demo-gpu"]}
    )
    api = TestClient(create_app(orchestrator))

    assert api.get("/healthz").status_code == 200
    assert api.get("/v1/workloads").status_code == 401

    registered = api.post(
        "/v1/agents/register",
        headers=bearer(agent_token),
        json={
            "agent_id": "agent-a",
            "provider": "local",
            "provider_resource_id": "tower-a",
            "hostname": "tower-a",
            "cpu_cores": 16,
            "ram_mb": 65536,
            "disk_free_mb": 100000,
            "slots": [
                {
                    "id": "slot-a",
                    "gpu_index": 0,
                    "gpu_uuid": "GPU-1",
                    "vendor": "nvidia",
                    "model": "RTX 4090",
                    "vram_mb": 24576,
                    "compute_capability": "8.9",
                    "capabilities": ["demo-gpu"],
                }
            ],
        },
    )
    assert registered.status_code == 200, registered.text

    submitted = api.post(
        "/v1/jobs",
        headers={**bearer(client_token), "Idempotency-Key": "api-test"},
        json={
            "project_id": "project-a",
            "workload": "demo-gpu",
            "payload": {"message": "hello"},
        },
    )
    assert submitted.status_code == 201, submitted.text
    job_id = submitted.json()["id"]

    assert api.get(f"/v1/jobs/{job_id}", headers=bearer(other_token)).status_code == 403

    claimed = api.post(
        "/v1/slots/slot-a/claim",
        headers=bearer(agent_token),
        json={"lease_seconds": 90},
    )
    assert claimed.status_code == 200, claimed.text
    lease = claimed.json()["lease_token"]
    lease_headers = {**bearer(agent_token), "X-SPAREHOUR-Lease-Token": lease}

    confirmed = api.post(
        f"/v1/jobs/{job_id}/confirm?slot_id=slot-a", headers=lease_headers
    )
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["state"] == "running"

    completed = api.post(
        f"/v1/jobs/{job_id}/complete?slot_id=slot-a",
        headers=lease_headers,
        json={
            "outputs": [
                {
                    "name": "result",
                    "uri": f"artifact://{SHA}",
                    "media_type": "application/json",
                    "size": 2,
                    "sha256": SHA,
                }
            ]
        },
    )
    assert completed.status_code == 200, completed.text
    assert completed.json()["state"] == "succeeded"

    fetched = api.get(f"/v1/jobs/{job_id}", headers=bearer(client_token))
    assert fetched.status_code == 200
    assert fetched.json()["outputs"][0]["name"] == "result"

    by_slot = api.get("/v1/jobs?slot_id=slot-a", headers=bearer(client_token))
    assert [row["id"] for row in by_slot.json()] == [job_id]
    by_worker = api.get("/v1/jobs?worker_id=agent-a", headers=bearer(client_token))
    assert [row["id"] for row in by_worker.json()] == [job_id]
    other_slot = api.get("/v1/jobs?slot_id=slot-b", headers=bearer(client_token))
    assert other_slot.json() == []
    searched = api.get("/v1/jobs?q=hello", headers=bearer(client_token))
    assert [row["id"] for row in searched.json()] == [job_id]
    assert api.get("/v1/jobs?q=absent", headers=bearer(client_token)).json() == []
    assert api.get("/v1/jobs?limit=1&offset=1", headers=bearer(client_token)).json() == []


def test_client_submission_token_can_be_workload_and_route_scoped(orchestrator, store):
    token = store.issue_token(
        role="client",
        subject="overwatch",
        scopes={
            "projects": ["overwatch"],
            "workloads": ["translate-segment"],
            "routes": ["libretranslate-v1"],
        },
    )
    api = TestClient(create_app(orchestrator))

    wrong_workload = api.post(
        "/v1/jobs", headers=bearer(token),
        json={"project_id": "overwatch", "workload": "demo-gpu",
              "payload": {"message": "no"}},
    )
    assert wrong_workload.status_code == 403

    wrong_route = api.post(
        "/v1/jobs", headers=bearer(token),
        json={"project_id": "overwatch", "workload": "translate-segment",
              "route": "not-allowed",
              "payload": {"text": "Merhaba", "source": "tr", "target": "en"}},
    )
    assert wrong_route.status_code == 403

    allowed = api.post(
        "/v1/jobs", headers=bearer(token),
        json={"project_id": "overwatch", "workload": "translate-segment",
              "route": "libretranslate-v1",
              "payload": {"text": "Merhaba", "source": "tr", "target": "en"}},
    )
    assert allowed.status_code == 201, allowed.text


def test_agent_cannot_advertise_capability_outside_token_scope(orchestrator, store):
    token = store.issue_token(
        role="agent", subject="agent-a", scopes={"routes": ["demo-gpu"]}
    )
    api = TestClient(create_app(orchestrator))
    response = api.post(
        "/v1/agents/register",
        headers=bearer(token),
        json={
            "agent_id": "agent-a",
            "provider": "vast",
            "hostname": "provider-host",
            "slots": [
                {
                    "id": "slot-a",
                    "gpu_index": 0,
                    "gpu_uuid": "GPU-1",
                    "model": "RTX 4090",
                    "vram_mb": 24576,
                    "capabilities": ["image-generate"],
                }
            ],
        },
    )
    assert response.status_code == 403


def test_admin_can_audit_edit_and_revoke_tokens(orchestrator, store):
    admin = store.issue_token(role="admin", subject="operator", token_id="admin-token")
    controller = store.issue_token(role="controller", subject="observer")
    target = store.issue_token(
        role="client", subject="project-a", scopes={"projects": ["project-a"]},
        token_id="editable-token",
    )
    api = TestClient(create_app(orchestrator))

    assert api.get("/v1/control/tokens", headers=bearer(controller)).status_code == 403
    listing = api.get("/v1/control/tokens", headers=bearer(admin))
    assert listing.status_code == 200
    row = next(item for item in listing.json() if item["id"] == "editable-token")
    assert row["status"] == "active"
    assert row["last_used_at"] is None
    assert row["scopes"] == {"projects": ["project-a"]}
    assert "secret_hash" not in row and "salt_hex" not in row

    edited = api.put(
        "/v1/control/tokens/editable-token", headers=bearer(admin),
        json={"role": "client", "subject": "project-b",
              "scopes": {"projects": ["project-b"], "workloads": ["demo-gpu"]},
              "expires_at": None},
    )
    assert edited.status_code == 200, edited.text
    principal = store.authenticate(target)
    assert principal["subject"] == "project-b"
    assert principal["scopes"]["workloads"] == ["demo-gpu"]

    revoked = api.delete(
        "/v1/control/tokens/editable-token", headers=bearer(admin)
    )
    assert revoked.status_code == 200
    assert revoked.json()["status"] == "revoked"
    assert revoked.json()["revoked_at"] is not None
    assert revoked.json()["last_used_at"] is not None
    assert api.get("/v1/workloads", headers=bearer(target)).status_code == 401
    retained = api.get("/v1/control/tokens", headers=bearer(admin)).json()
    assert next(item for item in retained if item["id"] == "editable-token")["status"] == "revoked"


def test_output_is_validated_brokered_and_downloadable(orchestrator, store, tmp_path):
    register_agent(store, "agent-a")
    client_token = store.issue_token(role="client", subject="project-a")
    agent_token = store.issue_token(role="agent", subject="agent-a")
    job = orchestrator.submit(
        project_id="project-a", workload="demo-gpu", payload={"message": "hello"}
    )
    claim = orchestrator.claim("agent-a-slot-0")
    orchestrator.confirm(job["id"], "agent-a-slot-0", claim["lease_token"])

    api = TestClient(create_app(orchestrator, ArtifactStore(tmp_path / "artifacts")))
    lease_headers = {
        **bearer(agent_token),
        "X-SPAREHOUR-Lease-Token": claim["lease_token"],
        "Content-Type": "application/json",
    }
    invalid = api.put(
        f"/v1/jobs/{job['id']}/outputs/result?slot_id=agent-a-slot-0",
        headers=lease_headers,
        content=b"not-json",
    )
    assert invalid.status_code == 422

    uploaded = api.put(
        f"/v1/jobs/{job['id']}/outputs/result?slot_id=agent-a-slot-0",
        headers=lease_headers,
        content=b'{"ok":true}',
    )
    assert uploaded.status_code == 200, uploaded.text
    artifact = uploaded.json()
    assert artifact["uri"].startswith(f"/v1/jobs/{job['id']}/artifacts/")

    completed = api.post(
        f"/v1/jobs/{job['id']}/complete?slot_id=agent-a-slot-0",
        headers=lease_headers,
        json={"outputs": [artifact]},
    )
    assert completed.status_code == 200, completed.text

    downloaded = api.get(artifact["uri"], headers=bearer(client_token))
    assert downloaded.status_code == 200
    assert downloaded.content == b'{"ok":true}'


def test_optional_public_bootstrap_serves_only_wheels(orchestrator, tmp_path):
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    wheel = bootstrap / "sparehour-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel-content")
    (bootstrap / "secret.txt").write_text("not public", encoding="utf-8")
    api = TestClient(create_app(orchestrator, bootstrap_directory=bootstrap))

    served = api.get(f"/bootstrap/{wheel.name}")
    assert served.status_code == 200
    assert served.content == b"wheel-content"
    assert api.get("/bootstrap/secret.txt").status_code == 404


def test_workload_depths_require_dynamic_permission(orchestrator, store):
    client_token = store.issue_token(role="client", subject="project-a")
    plain_agent = store.issue_token(role="agent", subject="agent-a")
    dynamic_agent = store.issue_token(
        role="agent", subject="agent-a", scopes={"perms": ["dynamic"]}
    )
    api = TestClient(create_app(orchestrator))

    from conftest import register_agent

    register_agent(store, "agent-a")

    submitted = api.post(
        "/v1/jobs",
        headers={**bearer(client_token), "Idempotency-Key": "depth-test"},
        json={
            "project_id": "project-a",
            "workload": "demo-gpu",
            "payload": {"message": "hello"},
        },
    )
    assert submitted.status_code == 201, submitted.text

    def routes_for(token) -> dict:
        document = next(
            item
            for item in api.get("/v1/workloads", headers=bearer(token)).json()
            if item["name"] == "demo-gpu"
        )
        return document["routes"]

    # Plain agent tokens keep the historical shape: no pending counts at all.
    assert "pending_jobs" not in routes_for(plain_agent)["command"]
    # perms=dynamic tokens see per-route queue depth.
    assert routes_for(dynamic_agent)["command"]["pending_jobs"] == 1
    # Depths are real-time: draining the queue drops the count back to zero.
    claimed = api.post(
        "/v1/slots/agent-a-slot-0/claim",
        headers=bearer(dynamic_agent),
        json={"lease_seconds": 90},
    )
    assert claimed.status_code == 200 and claimed.json() is not None
    assert routes_for(dynamic_agent)["command"]["pending_jobs"] == 0
