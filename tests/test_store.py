from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest

from gpu_orchestrator.service import OutputContractError
from gpu_orchestrator.store import ConflictError, LeaseError

from conftest import register_agent


SHA = "a" * 64


def output(name="result", media_type="application/json"):
    return {
        "name": name,
        "uri": f"artifact://{SHA}",
        "media_type": media_type,
        "size": 10,
        "sha256": SHA,
    }


def test_idempotent_submission(orchestrator):
    first = orchestrator.submit(
        project_id="project-a",
        workload="demo-gpu",
        payload={"message": "hello"},
        idempotency_key="same-request",
    )
    second = orchestrator.submit(
        project_id="project-a",
        workload="demo-gpu",
        payload={"message": "hello"},
        idempotency_key="same-request",
    )
    with pytest.raises(ConflictError, match="different request"):
        orchestrator.submit(
            project_id="project-a",
            workload="demo-gpu",
            payload={"message": "different"},
            idempotency_key="same-request",
        )
    other_project = orchestrator.submit(
        project_id="project-b",
        workload="demo-gpu",
        payload={"message": "hello"},
        idempotency_key="same-request",
    )

    assert second["id"] == first["id"]
    assert second["payload"] == {"message": "hello"}
    assert other_project["id"] != first["id"]


def test_agent_cannot_take_over_another_agents_slot(store):
    first = register_agent(store, "owner")
    attacker = first.model_copy(
        update={"agent_id": "attacker", "hostname": "attacker"}
    )

    with pytest.raises(ConflictError, match="already owned"):
        store.register_agent(attacker)

    assert store.get_slot("owner-slot-0")["agent_id"] == "owner"


def test_claim_matches_vram_capability_and_model_pack(orchestrator, store):
    register_agent(
        store,
        "low",
        vram_mb=8192,
        capabilities=["image-generate"],
        model_packs=["sdxl-base-1.0-fp16-76d28af"],
    )
    register_agent(
        store,
        "missing-pack",
        vram_mb=24576,
        capabilities=["image-generate"],
        model_packs=[],
    )
    register_agent(
        store,
        "ready",
        vram_mb=24576,
        capabilities=["image-generate"],
        model_packs=["sdxl-base-1.0-fp16-76d28af"],
    )
    job = orchestrator.submit(
        project_id="project-a",
        workload="image-generate",
        payload={"prompt": "test", "seed": 42},
    )

    assert orchestrator.claim("low-slot-0") is None
    assert orchestrator.claim("missing-pack-slot-0") is None
    claim = orchestrator.claim("ready-slot-0")
    assert claim["id"] == job["id"]
    assert claim["state"] == "leased"
    assert claim["lease_token"]


def test_full_lifecycle_and_output_contract(orchestrator, store):
    register_agent(store, "worker")
    job = orchestrator.submit(
        project_id="project-a", workload="demo-gpu", payload={"message": "hello"}
    )
    claim = orchestrator.claim("worker-slot-0")
    token = claim["lease_token"]

    running = orchestrator.confirm(job["id"], "worker-slot-0", token)
    assert running["state"] == "running"
    assert running["attempts"] == 1
    assert orchestrator.heartbeat(job["id"], "worker-slot-0", token) is False

    with pytest.raises(OutputContractError, match="missing required"):
        orchestrator.complete(job["id"], "worker-slot-0", token, [])

    done = orchestrator.complete(job["id"], "worker-slot-0", token, [output()])
    assert done["state"] == "succeeded"
    assert done["outputs"][0]["sha256"] == SHA
    assert store.get_slot("worker-slot-0")["state"] == "ready"
    assert store.job_attempts(job["id"])[0]["state"] == "succeeded"


def test_retry_policy_is_central_not_worker_controlled(orchestrator, store):
    register_agent(store, "worker")
    job = orchestrator.submit(
        project_id="project-a", workload="demo-gpu", payload={"message": "hello"}
    )
    claim = orchestrator.claim("worker-slot-0")
    orchestrator.confirm(job["id"], "worker-slot-0", claim["lease_token"])
    retried = orchestrator.fail(
        job["id"],
        "worker-slot-0",
        claim["lease_token"],
        error_code="worker_lost",
        error_message="network vanished",
    )
    assert retried["state"] == "pending"

    claim = orchestrator.claim("worker-slot-0")
    orchestrator.confirm(job["id"], "worker-slot-0", claim["lease_token"])
    terminal = orchestrator.fail(
        job["id"],
        "worker-slot-0",
        claim["lease_token"],
        error_code="input_invalid",
        error_message="bad request",
    )
    assert terminal["state"] == "dead-letter"


def test_expired_lease_is_requeued_and_old_worker_is_fenced(orchestrator, store, clock):
    register_agent(store, "old")
    register_agent(store, "new")
    job = orchestrator.submit(
        project_id="project-a", workload="demo-gpu", payload={"message": "hello"}
    )
    old_claim = orchestrator.claim("old-slot-0", lease_seconds=30)
    orchestrator.confirm(job["id"], "old-slot-0", old_claim["lease_token"])
    clock.value += timedelta(seconds=31)

    assert store.reap_expired_leases() == {"released": 0, "retried": 1, "dead_lettered": 0}
    new_claim = orchestrator.claim("new-slot-0", lease_seconds=30)
    assert new_claim["id"] == job["id"]

    with pytest.raises(LeaseError):
        orchestrator.complete(job["id"], "old-slot-0", old_claim["lease_token"], [output()])


def test_cancel_running_job_propagates_on_heartbeat(orchestrator, store):
    register_agent(store, "worker")
    job = orchestrator.submit(
        project_id="project-a", workload="demo-gpu", payload={"message": "hello"}
    )
    claim = orchestrator.claim("worker-slot-0")
    orchestrator.confirm(job["id"], "worker-slot-0", claim["lease_token"])
    requested = store.cancel(job["id"])
    assert requested["cancel_requested"] is True
    assert orchestrator.heartbeat(job["id"], "worker-slot-0", claim["lease_token"]) is True

    cancelled = orchestrator.fail(
        job["id"],
        "worker-slot-0",
        claim["lease_token"],
        error_code="cancelled",
        error_message="cancelled by client",
    )
    assert cancelled["state"] == "cancelled"


def test_atomic_claim_gives_one_job_to_exactly_one_slot(orchestrator, store):
    slots = 20
    register_agent(store, "fleet", slots=slots)
    job = orchestrator.submit(
        project_id="project-a", workload="demo-gpu", payload={"message": "race"}
    )
    barrier = threading.Barrier(slots)

    def claim(index):
        barrier.wait()
        return orchestrator.claim(f"fleet-slot-{index}")

    with ThreadPoolExecutor(max_workers=slots) as pool:
        claims = list(pool.map(claim, range(slots)))

    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1
    assert winners[0]["id"] == job["id"]


def test_slot_cannot_claim_twice(orchestrator, store):
    register_agent(store, "worker")
    for value in ("one", "two"):
        orchestrator.submit(
            project_id="project-a", workload="demo-gpu", payload={"message": value}
        )
    assert orchestrator.claim("worker-slot-0") is not None
    with pytest.raises(ConflictError, match="not available"):
        orchestrator.claim("worker-slot-0")


def test_scoped_tokens_are_hashed_and_revocable(store):
    raw = store.issue_token(
        role="agent", subject="worker", scopes={"routes": ["demo-gpu"]}, token_id="worker-token"
    )
    assert raw.startswith("dgpu.worker-token.")
    assert next(row for row in store.list_tokens() if row["id"] == "worker-token")["last_used_at"] is None
    with pytest.raises(PermissionError):
        store.authenticate("dgpu.worker-token.wrong-secret")
    assert next(row for row in store.list_tokens() if row["id"] == "worker-token")["last_used_at"] is None
    principal = store.authenticate(raw)
    assert principal["subject"] == "worker"
    assert next(row for row in store.list_tokens() if row["id"] == "worker-token")["last_used_at"] is not None
    assert principal["scopes"] == {"routes": ["demo-gpu"]}
    assert store.disable_token("worker-token") is True
    with pytest.raises(PermissionError):
        store.authenticate(raw)
