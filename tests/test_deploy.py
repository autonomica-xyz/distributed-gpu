from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import time

import pytest
import yaml
from fastapi.testclient import TestClient

from sparehour.api import create_app
from sparehour.deploy import DeploymentService
from sparehour.models import ResourceRequirements
from sparehour.providers.base import (
    ProviderOffer,
    ProviderResource,
    WorkerLaunchSpec,
)

from conftest import register_agent


DIGEST = "ghcr.io/example/worker@sha256:" + "a" * 64


class FakeDriver:
    name = "vast"

    def __init__(self):
        self.offers = [
            ProviderOffer(
                provider=self.name,
                offer_id="offer-1",
                gpu_model="RTX 4090",
                gpu_count=1,
                vram_mb=24576,
                hourly_cost_usd=0.30,
                reliability=0.99,
                available_count=5,
            )
        ]
        self.launched: list[WorkerLaunchSpec] = []
        self.selected_offers: list[ProviderOffer] = []
        self.resources: dict[str, ProviderResource] = {}
        self.destroyed: list[str] = []
        self.still_visible_after_destroy = False
        self.credit = 100.0

    def search_offers(self, requirements, *, max_hourly_cost_usd, limit=25):
        return [o for o in self.offers if o.hourly_cost_usd <= max_hourly_cost_usd]

    def launch(self, offer, spec):
        self.selected_offers.append(offer)
        self.launched.append(spec)
        resource = ProviderResource(
            provider=self.name,
            resource_id=f"res-{len(self.launched)}",
            name=spec.name,
            worker_id=spec.worker_id,
            state="creating",
            gpu_model=offer.gpu_model if offer else None,
            hourly_cost_usd=offer.hourly_cost_usd if offer else None,
        )
        self.resources[resource.resource_id] = resource
        return resource

    def inventory(self):
        if self.still_visible_after_destroy:
            return list(self.resources.values())
        return [r for rid, r in self.resources.items() if rid not in self.destroyed]

    def destroy(self, resource_id):
        self.destroyed.append(resource_id)

    def account_credit(self):
        return self.credit


class FakeSaladDriver(FakeDriver):
    name = "salad"

    def launch(self, offer, spec):
        if not spec.provider_template:
            raise ValueError("Salad launch requires provider_template")
        return super().launch(offer, spec)


def pool_yaml(tmp_path, *, image=DIGEST, with_salad_template=False):
    pool = {
        "api_version": "sparehour.example/v1alpha1",
        "id": "test-pool",
        "workload": "demo-gpu",
        "route": "command",
        "worker_prefix": "testpool",
        "image": image,
        "command": ["sparehour", "agent", "--agent-id", "${SPAREHOUR_AGENT_ID}",
                    "--capability", "demo-gpu"],
        "environment": {"EXTRA": "1"},
        "providers": {
            "vast": {"enabled": True, "max_hourly_cost_usd": 0.50},
            "salad": {"enabled": True, "max_hourly_cost_usd": 0.50},
        },
    }
    if with_salad_template:
        pool["provider_templates"] = {
            "salad": {"container": {"command": ["x"], "image": DIGEST},
                      "resources": {"cpu": 4}},
        }
    (tmp_path / "test-pool.yaml").write_text(yaml.safe_dump(pool), encoding="utf-8")
    return tmp_path


@pytest.fixture
def driver():
    return FakeDriver()


@pytest.fixture
def service(orchestrator, tmp_path, driver):
    return DeploymentService(
        registry=orchestrator.registry,
        store=orchestrator.store,
        drivers={"vast": driver, "salad": driver},
        pools_directory=str(pool_yaml(tmp_path)),
        public_controller_url="https://public.example.com/",
        wheel_url="https://public.example.com/bootstrap/sparehour-0.1.0-py3-none-any.whl",
    )


def _api(orchestrator, service):
    return TestClient(create_app(orchestrator, deployment_service=service))


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def _admin(store):
    return store.issue_token(role="admin", subject="operator")


def test_deploy_issues_scoped_worker_credentials_and_records_ledger(service, orchestrator):
    rows = service.deploy(pool_id="test-pool", provider="vast", count=2, expires_hours=6)
    assert len(rows) == 2
    assert all(r["state"] == "requested" for r in rows)

    # each worker got a unique identity and a route-scoped, expiring token
    worker_ids = [r["worker_id"] for r in rows]
    assert len(set(worker_ids)) == 2
    assert all(w.startswith("testpool-") for w in worker_ids)
    for spec in service.drivers["vast"].launched:
        token = spec.environment["SPAREHOUR_TOKEN"]
        principal = orchestrator.store.authenticate(token)
        assert principal["role"] == "agent"
        assert principal["subject"] == spec.worker_id
        assert principal["scopes"] == {"routes": ["demo-gpu"]}
        expiry = datetime.fromisoformat(principal["expires_at"])
        assert datetime.now(UTC) < expiry <= datetime.now(UTC) + timedelta(hours=6, minutes=1)
        # workers see only the public URL, identity, and their own token
        assert spec.environment["SPAREHOUR_ENDPOINT"] == "https://public.example.com"
        assert spec.environment["SPAREHOUR_AGENT_ID"] == spec.worker_id
        assert spec.environment["SPAREHOUR_WHEEL_URL"].endswith(".whl")
        assert spec.environment["EXTRA"] == "1"
        assert spec.image == DIGEST
        assert spec.command[3] == spec.worker_id  # ${SPAREHOUR_AGENT_ID} substituted
    assert [r["resource_id"] for r in rows] == ["res-1", "res-2"]


def test_deploy_rejects_unknown_pool_disabled_provider_and_bad_counts(service):
    from sparehour.deploy import DeploymentError
    from sparehour.store import NotFoundError

    with pytest.raises(NotFoundError):  # maps to HTTP 404
        service.deploy(pool_id="nope", provider="vast")
    with pytest.raises(DeploymentError):
        service.deploy(pool_id="test-pool", provider="local")
    with pytest.raises(DeploymentError):
        service.deploy(pool_id="test-pool", provider="vast", count=0)
    with pytest.raises(DeploymentError):
        service.deploy(pool_id="test-pool", provider="vast", expires_hours=999)

    driver = service.drivers["vast"]
    driver.credit = None
    with pytest.raises(DeploymentError, match="credit is unknown"):
        service.deploy(pool_id="test-pool", provider="vast")
    driver.credit = 0
    with pytest.raises(DeploymentError, match="insufficient credit"):
        service.deploy(pool_id="test-pool", provider="vast")
    driver.credit = 100


def test_deploy_selects_requested_offer_and_rejects_stale_or_insufficient_capacity(
    service, driver
):
    from sparehour.deploy import DeploymentError

    driver.offers.append(
        ProviderOffer("vast", "offer-cheap", "RTX 3070", 1, 8192, 0.10,
                      available_count=1)
    )
    row = service.deploy(
        pool_id="test-pool", provider="vast", offer_id="offer-cheap"
    )[0]
    assert row["hourly_cost_usd"] == 0.10
    assert driver.selected_offers[-1].offer_id == "offer-cheap"

    with pytest.raises(DeploymentError, match="only 1"):
        service.deploy(
            pool_id="test-pool", provider="vast", offer_id="offer-cheap", count=2
        )
    driver.offers.append(
        ProviderOffer("vast", "offer-empty", "RTX 3050", 1, 8192, 0.05,
                      available_count=0)
    )
    with pytest.raises(DeploymentError, match="no available capacity"):
        service.deploy(
            pool_id="test-pool", provider="vast", offer_id="offer-empty"
        )
    with pytest.raises(DeploymentError, match="no longer eligible"):
        service.deploy(
            pool_id="test-pool", provider="vast", offer_id="missing"
        )


def test_failed_launch_audits_selected_offer_and_retires_token(
    service, orchestrator, driver
):
    from sparehour.providers.base import ProviderError

    def reject(_offer, _spec):
        raise ProviderError("provider rejected create")

    driver.launch = reject
    row = service.deploy(
        pool_id="test-pool", provider="vast", offer_id="offer-1", actor="alice"
    )[0]
    assert row["state"] == "failed"
    assert "offer offer-1" in row["detail"]
    assert "RTX 4090" in row["detail"]
    assert "availability 5" in row["detail"]
    assert "$0.3000/h" in row["detail"]
    assert "requested by alice" in row["detail"]
    # A failed provider create must never leave a usable worker credential.
    token_row = next(
        item for item in orchestrator.store.list_tokens()
        if item["id"] == row["token_id"]
    )
    assert token_row["status"] == "revoked"


def test_reconcile_tracks_provider_then_agent_and_absence(service, orchestrator):
    rows = service.deploy(pool_id="test-pool", provider="vast")
    row = rows[0]
    counts = service.reconcile()
    assert counts["creating"] == 1
    assert orchestrator.store.get_deployment(row["id"])["state"] == "creating"

    driver = service.drivers["vast"]
    driver.resources[row["resource_id"]] = replace(
        driver.resources[row["resource_id"]], state="running"
    )
    register_agent(orchestrator.store, row["worker_id"])
    counts = service.reconcile()
    assert counts["active"] == 1
    assert orchestrator.store.get_deployment(row["id"])["state"] == "active"
    listing = service.list_deployments()[0]
    assert listing["agent_status"] == "ready"
    assert listing["slot_count"] == 1
    assert listing["estimated_monthly_cost_usd"] == pytest.approx(0.30 * 730)

    orchestrator.store.heartbeat_agent(row["worker_id"], draining=True)
    counts = service.reconcile()
    assert counts["draining"] == 1
    assert orchestrator.store.get_deployment(row["id"])["state"] == "draining"

    driver.destroyed.append(row["resource_id"])  # vanished externally
    service.reconcile()
    assert orchestrator.store.get_deployment(row["id"])["state"] == "absent"


def test_background_reconciler_updates_deployments_without_manual_post(
    service, orchestrator
):
    row = service.deploy(pool_id="test-pool", provider="vast")[0]
    app = create_app(
        orchestrator,
        deployment_service=service,
        deployment_reconcile_seconds=0.01,
    )
    with TestClient(app):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if orchestrator.store.get_deployment(row["id"])["state"] == "creating":
                break
            time.sleep(0.01)
    assert orchestrator.store.get_deployment(row["id"])["state"] == "creating"


def test_teardown_refuses_active_jobs_then_verifies_absence_before_retiring_token(
    service, orchestrator, driver
):
    from sparehour.store import ConflictError

    row = service.deploy(pool_id="test-pool", provider="vast")[0]
    token = driver.launched[0].environment["SPAREHOUR_TOKEN"]
    # worker holds a running job -> teardown must refuse
    register_agent(store=orchestrator.store, agent_id=row["worker_id"])
    job = orchestrator.submit(
        project_id="project-p", workload="demo-gpu", payload={"message": "hi"}
    )
    slot = f"{row['worker_id']}-slot-0"
    claim = orchestrator.claim(slot)
    orchestrator.confirm(job["id"], slot, claim["lease_token"])

    with pytest.raises(ConflictError):
        service.teardown(row["id"])
    assert orchestrator.store.authenticate(token) is not None  # credential intact

    # release the job so teardown can proceed
    sha = "c" * 64
    orchestrator.complete(
        job["id"], slot, claim["lease_token"],
        [{"name": "result", "uri": f"artifact://{sha}", "media_type": "application/json",
          "size": 2, "sha256": sha}],
    )

    # provider still lists the resource -> destroy verification fails and the
    # credential must NOT be retired until provider inventory is absent
    driver.still_visible_after_destroy = True
    from sparehour.providers.base import ProviderError

    with pytest.raises(ProviderError):
        service.teardown(row["id"])
    assert orchestrator.store.authenticate(token) is not None
    assert orchestrator.store.get_deployment(row["id"])["state"] != "destroyed"

    # clean path: verified absence, then token disabled, then ledger closed
    driver.still_visible_after_destroy = False
    finished = service.teardown(row["id"])
    assert finished["state"] == "destroyed"
    assert finished["destroyed_at"]
    with pytest.raises(PermissionError):
        orchestrator.store.authenticate(token)


def test_capacity_lists_pool_plans_with_trimmed_offers(service, orchestrator):
    orchestrator.submit(project_id="project-p", workload="demo-gpu", payload={"message": "hi"})
    caps = service.capacity()
    assert len(caps) == 1
    cap = caps[0]
    assert cap["pool"]["id"] == "test-pool"
    assert cap["capability"] == "demo-gpu"
    assert "provider_templates" not in cap["pool"]
    assert cap["plan"]["queued_jobs"] == 1
    assert all(len(p["offers"]) <= 20 for p in cap["plan"]["providers"])


def test_deploy_endpoints_are_controller_only_and_flow_over_http(service, orchestrator, store):
    api = _api(orchestrator, service)
    client_token = store.issue_token(role="client", subject="p")
    admin_token = _admin(store)

    assert api.get("/v1/control/capacity", headers=bearer(client_token)).status_code == 403
    assert api.get("/v1/control/deployments").status_code == 401

    deployed = api.post(
        "/v1/control/deployments",
        headers=bearer(admin_token),
        json={"pool_id": "test-pool", "provider": "vast", "offer_id": "offer-1",
              "count": 1, "expires_hours": 8},
    )
    assert deployed.status_code == 201, deployed.text
    row = deployed.json()[0]
    assert row["state"] == "requested"

    listing = api.get("/v1/control/deployments", headers=bearer(admin_token))
    assert listing.status_code == 200
    assert [d["id"] for d in listing.json()] == [row["id"]]

    reconciled = api.post("/v1/control/deployments/reconcile", headers=bearer(admin_token))
    assert reconciled.status_code == 200
    assert reconciled.json()["creating"] >= 1

    destroyed = api.delete(
        f"/v1/control/deployments/{row['id']}", headers=bearer(admin_token)
    )
    assert destroyed.status_code == 200
    assert destroyed.json()["state"] == "destroyed"


def test_deploy_without_public_url_service_mounts_nothing(orchestrator):
    api = TestClient(create_app(orchestrator))
    admin = orchestrator.store.issue_token(role="admin", subject="operator")
    assert api.get("/v1/control/capacity", headers=bearer(admin)).status_code == 404


def test_salad_launch_requires_pool_provider_template(orchestrator, tmp_path):
    from sparehour.deploy import DeploymentError

    driver = FakeSaladDriver()
    service = DeploymentService(
        registry=orchestrator.registry,
        store=orchestrator.store,
        drivers={"salad": driver},
        pools_directory=str(pool_yaml(tmp_path, with_salad_template=False)),
        public_controller_url="https://public.example.com",
    )
    rows = service.deploy(pool_id="test-pool", provider="salad")
    assert rows[0]["state"] == "failed"
    assert "provider_template" in rows[0]["detail"]

    driver2 = FakeSaladDriver()
    service2 = DeploymentService(
        registry=orchestrator.registry,
        store=orchestrator.store,
        drivers={"salad": driver2},
        pools_directory=str(pool_yaml(tmp_path, with_salad_template=True)),
        public_controller_url="https://public.example.com",
    )
    rows2 = service2.deploy(pool_id="test-pool", provider="salad")
    assert rows2[0]["state"] == "requested"
    assert driver2.launched[-1].provider_template["resources"]["cpu"] == 4


def test_pool_loader_rejects_mutable_image_tags(tmp_path):
    from sparehour.controller import PoolSpec

    pool_yaml(tmp_path, image="ghcr.io/example/worker:latest")
    with pytest.raises(Exception, match="digest"):
        PoolSpec.from_directory(tmp_path)
