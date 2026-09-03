from __future__ import annotations

import io
import json
import urllib.error

import pytest

from gpu_orchestrator.models import ResourceRequirements
from gpu_orchestrator.providers.base import (
    ProviderError,
    ProviderOffer,
    ProviderResource,
    WorkerLaunchSpec,
    destroy_and_verify_absent,
)
from gpu_orchestrator.providers.vast import VastProvider
from gpu_orchestrator.providers.salad import SaladProvider


class Response:
    def __init__(self, payload, status=200):
        self.status = status
        self.body = b"" if payload is None else json.dumps(payload).encode()

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_vast_search_normalizes_and_sorts_offers():
    seen = []

    def opener(request, timeout):
        seen.append((request.method, request.full_url, json.loads(request.data)))
        return Response(
            {
                "offers": [
                    {"id": 2, "gpu_name": "RTX 4090", "gpu_ram": 24576, "dph_total": 0.4, "reliability": 0.99},
                    {"id": 1, "gpu_name": "RTX 3090", "gpu_ram": 24576, "dph_total": 0.2, "reliability": 0.98},
                    {"id": 3, "gpu_name": "too expensive", "gpu_ram": 24576, "dph_total": 9},
                ]
            }
        )

    provider = VastProvider("secret", opener=opener)
    offers = provider.search_offers(
        ResourceRequirements(min_vram_mb=16000, min_ram_mb=8000, min_disk_mb=20000),
        max_hourly_cost_usd=0.5,
    )

    assert [offer.offer_id for offer in offers] == ["1", "2"]
    assert seen[0][0] == "POST"
    assert seen[0][2]["gpu_ram"] == {"gte": 16000}
    assert seen[0][2]["num_gpus"] == {"eq": 1}


def test_worker_images_must_be_immutable():
    with pytest.raises(ValueError, match="pinned"):
        WorkerLaunchSpec(
            name="worker-1",
            worker_id="worker-1",
            image="example/worker:latest",
            command=["agent"],
        )


def test_vast_launch_uses_documented_docker_flag_environment_string():
    seen = []

    def opener(request, timeout):
        seen.append(json.loads(request.data))
        return Response({"success": True, "new_contract": 123})

    provider = VastProvider("secret", opener=opener)
    resource = provider.launch(
        ProviderOffer("vast", "42", "RTX 4090", 1, 24576, 0.25),
        WorkerLaunchSpec(
            name="worker-1",
            worker_id="worker-1",
            image="example/worker@sha256:" + "0" * 64,
            command=["dgpu", "agent"],
            environment={"DGPU_AGENT_ID": "worker-1", "TOKEN": "has spaces"},
        ),
    )

    assert resource.resource_id == "123"
    assert isinstance(seen[0]["env"], str)
    assert "-e DGPU_AGENT_ID=worker-1" in seen[0]["env"]
    assert "'TOKEN=has spaces'" in seen[0]["env"]


def _salad_price_rows():
    return {
        "items": [
            {
                "id": "2aec4fc1-2270-4e40-b8cc-6e69fae61d4f",
                "name": "RTX 2070 (8 GB)",
                "prices": [
                    {"priority": "high", "price": "0.06"},
                    {"priority": "low", "price": "0.033"},
                    {"priority": "batch", "price": "0.02"},
                ],
            },
            {
                "id": "6b17a5e7-9a6a-4f1c-9782-243277f4fa00",
                "name": "GTX 1070, 1080, 1080Ti (8 GB)",
                "prices": [{"priority": "batch", "price": "0.01"}],
            },
        ]
    }


def test_salad_search_returns_live_priced_available_class_priority_offers():
    seen = []

    def opener(request, timeout):
        seen.append((request.method, request.full_url, json.loads(request.data) if request.data else None))
        if request.method == "GET":
            return Response(_salad_price_rows())
        return Response({
            "available_gpu_batch": 1,
            "available_gpu_low": 8,
            "available_gpu_high": 15,
            "on_call_gpu": 97,
        })

    provider = SaladProvider("secret", "org", "project", opener=opener)
    offers = provider.search_offers(
        ResourceRequirements(
            min_vram_mb=8192,
            min_compute_capability="7.0",
            gpu_models=["*RTX*"],
            min_cpu_cores=4,
            min_ram_mb=12288,
            min_disk_mb=30720,
        ),
        max_hourly_cost_usd=0.10,
    )

    assert [(row.raw["priority"], row.hourly_cost_usd, row.available_count)
            for row in offers] == [
        ("batch", 0.02, 1), ("low", 0.033, 8), ("high", 0.06, 15)
    ]
    assert all(row.gpu_model == "RTX 2070 (8 GB)" for row in offers)
    assert all(row.raw["price_source"] == "salad_api" for row in offers)
    assert seen[1][2] == {
        "gpu_classes": ["2aec4fc1-2270-4e40-b8cc-6e69fae61d4f"],
        "cpu": 4,
        "memory": 12288,
        "storage_amount": 30720 * 1024 * 1024,
    }


def test_salad_search_includes_gtx_1080_class_for_cc61_route():
    def opener(request, timeout):
        if request.method == "GET":
            return Response(_salad_price_rows())
        return Response({
            "available_gpu_batch": 5, "available_gpu_low": 7,
            "available_gpu_medium": 7, "available_gpu_high": 7,
        })

    provider = SaladProvider("secret", "org", "project", opener=opener)
    offers = provider.search_offers(
        ResourceRequirements(
            min_vram_mb=8192, min_compute_capability="6.1",
            gpu_models=["*GTX 1070*", "*GTX 1080*"],
        ),
        max_hourly_cost_usd=0.10,
    )

    assert offers
    assert offers[0].gpu_model == "GTX 1070, 1080, 1080Ti (8 GB)"
    assert offers[0].raw["compute_capability"] == "6.1"
    assert offers[0].available_count == 5


def test_salad_availability_failure_is_unknown_not_false_zero():
    def opener(request, timeout):
        if request.method == "GET":
            return Response(_salad_price_rows())
        raise urllib.error.HTTPError(
            request.full_url, 403, "Forbidden", {}, io.BytesIO(b"forbidden")
        )

    provider = SaladProvider("secret", "org", "project", opener=opener)
    offers = provider.search_offers(
        ResourceRequirements(min_vram_mb=8192, min_compute_capability="7.0",
                             gpu_models=["*RTX*"]),
        max_hourly_cost_usd=0.10,
    )

    assert offers
    assert all(row.available_count is None for row in offers)
    assert all(row.raw["availability_status"] == "unknown" for row in offers)


def test_salad_launch_applies_selected_gpu_class_priority_and_price():
    seen = []

    def opener(request, timeout):
        seen.append(json.loads(request.data))
        return Response({"id": "group-id"}, status=201)

    provider = SaladProvider("secret", "org", "project", opener=opener)
    template = {
        "container": {
            "image": "old",
            "priority": "high",
            "resources": {"cpu": 4, "gpu_classes": ["old-class"]},
        }
    }
    offer = ProviderOffer(
        "salad", "class-id:batch", "RTX 2070 (8 GB)", 1, 8192, 0.02,
        available_count=1,
        raw={"gpu_class_id": "class-id", "priority": "batch"},
    )
    resource = provider.launch(
        offer,
        WorkerLaunchSpec(
            name="worker-1", worker_id="worker-1",
            image="example/worker@sha256:" + "0" * 64,
            command=["dgpu", "agent"], environment={"DGPU_AGENT_ID": "worker-1"},
            provider_template=template,
        ),
    )

    assert resource.hourly_cost_usd == 0.02
    assert resource.gpu_model == "RTX 2070 (8 GB)"
    assert seen[0]["container"]["priority"] == "batch"
    assert seen[0]["container"]["resources"]["gpu_classes"] == ["class-id"]
    assert template["container"]["priority"] == "high"  # caller template unchanged


def test_salad_inventory_reads_current_state_status():
    def opener(request, timeout):
        return Response(
            {
                "items": [
                    {
                        "id": "group-id",
                        "name": "worker-1",
                        "current_state": {"status": "running"},
                        "container": {
                            "environment_variables": {"DGPU_AGENT_ID": "agent-1"}
                        },
                    }
                ]
            }
        )

    provider = SaladProvider("secret", "org", "project", opener=opener)
    resources = provider.inventory()

    assert resources[0].state == "running"
    assert resources[0].worker_id == "agent-1"


def test_salad_inventory_uses_instance_counts_and_selected_live_price():
    def opener(request, timeout):
        if request.full_url.endswith("/gpu-classes"):
            return Response(_salad_price_rows())
        group = {
            "id": "group-id",
            "name": "worker-1",
            "current_state": {
                "status": "running",
                "instance_status_counts": {
                    "running_count": 0, "creating_count": 1,
                    "allocating_count": 0, "stopping_count": 0,
                },
            },
            "container": {
                "priority": "batch",
                "resources": {"gpu_classes": ["2aec4fc1-2270-4e40-b8cc-6e69fae61d4f"]},
                "environment_variables": {"DGPU_AGENT_ID": "agent-1"},
            },
        }
        if request.full_url.endswith("/worker-1"):
            return Response(group)
        return Response({"items": [group]})

    provider = SaladProvider("secret", "org", "project", opener=opener)
    resource = provider.inventory()[0]

    assert resource.state == "creating"
    assert resource.gpu_model == "RTX 2070 (8 GB)"
    assert resource.hourly_cost_usd == 0.02


class FakeDriver:
    name = "fake"

    def __init__(self, disappears=True):
        self.visible = True
        self.disappears = disappears
        self.destroyed = []

    def destroy(self, resource_id):
        self.destroyed.append(resource_id)
        if self.disappears:
            self.visible = False

    def inventory(self):
        return [ProviderResource("fake", "r-1", "r-1", "running")] if self.visible else []


def test_destroy_requires_authoritative_inventory_absence():
    driver = FakeDriver()
    destroy_and_verify_absent(driver, "r-1", timeout_seconds=0)
    assert driver.destroyed == ["r-1"]

    stuck = FakeDriver(disappears=False)
    with pytest.raises(ProviderError, match="still lists"):
        destroy_and_verify_absent(stuck, "r-1", timeout_seconds=0)
