from __future__ import annotations

from datetime import timedelta

from sparehour.controller import FleetPlanner, PoolSpec
from sparehour.providers.base import ProviderOffer

from conftest import register_agent


class FakeProvider:
    name = "vast"

    def account_credit(self):
        return 50.0

    def search_offers(self, requirements, *, max_hourly_cost_usd, limit=25):
        return [
            ProviderOffer(
                provider="vast",
                offer_id="offer-1",
                gpu_model="RTX 4090",
                gpu_count=1,
                vram_mb=24576,
                hourly_cost_usd=0.30,
                reliability=0.99,
            )
        ]


def pool():
    return PoolSpec(
        id="image-standard",
        workload="image-generate",
        route="diffusers-sdxl",
        worker_prefix="image-standard",
        image="example/worker@sha256:" + "0" * 64,
        command=["sparehour", "agent"],
        providers={
            "local": {"max_hourly_cost_usd": 0.01, "require_known_credit": False},
            "vast": {"max_hourly_cost_usd": 0.5, "require_known_credit": True},
        },
    )


def test_planner_counts_local_pool_and_recommends_provider_capacity(orchestrator, store):
    registration = register_agent(
        store,
        "local-image",
        capabilities=["image-generate"],
        model_packs=["sdxl-base-1.0-fp16-76d28af"],
    )
    # Pool membership is explicit so one shared slot is never accidentally
    # counted by two independent autoscalers.
    registration.labels["pool"] = "image-standard"
    store.register_agent(registration)
    for index in range(100):
        orchestrator.submit(
            project_id="project-a",
            workload="image-generate",
            payload={"prompt": f"image {index}", "seed": index},
        )

    plan = FleetPlanner(orchestrator.registry, store).plan(
        pool(), {"vast": FakeProvider()}
    )

    assert plan.current_slots == 1
    assert plan.queued_units == 100
    assert plan.target_slots > 1
    assert plan.requested_slots == plan.target_slots - 1
    assert plan.ranked_candidates[0]["provider"] == "vast"


def test_unknown_provider_credit_blocks_autonomous_capacity(orchestrator, store):
    class UnknownCredit(FakeProvider):
        def account_credit(self):
            return None

    orchestrator.submit(
        project_id="project-a",
        workload="image-generate",
        payload={"prompt": "image", "seed": 1},
    )
    plan = FleetPlanner(orchestrator.registry, store).plan(
        pool(), {"vast": UnknownCredit()}
    )
    assert plan.target_slots == 0
    assert plan.blocked_reason == "provider_capacity_unavailable"
    assert plan.providers[0].blocked_reason == "provider_credit_unknown"


def test_configured_expected_price_is_used_for_explicitly_unpriced_offer(
    orchestrator, store
):
    class UnpricedProvider(FakeProvider):
        def search_offers(self, requirements, *, max_hourly_cost_usd, limit=25):
            rows = super().search_offers(
                requirements, max_hourly_cost_usd=max_hourly_cost_usd, limit=limit
            )
            rows[0].raw["use_policy_price"] = True
            return rows

    spec = pool().model_copy(
        update={
            "providers": {
                "local": pool().providers["local"],
                "vast": pool().providers["vast"].model_copy(
                    update={"expected_hourly_cost_usd": 0.12}
                ),
            }
        }
    )
    orchestrator.submit(
        project_id="project-a",
        workload="image-generate",
        payload={"prompt": "image", "seed": 2},
    )

    plan = FleetPlanner(orchestrator.registry, store).plan(
        spec, {"vast": UnpricedProvider()}
    )

    assert plan.providers[0].offers[0]["hourly_cost_usd"] == 0.12
    assert plan.ranked_candidates[0]["hourly_cost_usd"] == 0.12


def test_live_provider_price_is_not_replaced_by_policy_expectation(orchestrator, store):
    spec = pool().model_copy(
        update={
            "providers": {
                "local": pool().providers["local"],
                "vast": pool().providers["vast"].model_copy(
                    update={"expected_hourly_cost_usd": 0.12}
                ),
            }
        }
    )
    orchestrator.submit(
        project_id="project-a", workload="image-generate",
        payload={"prompt": "image", "seed": 3},
    )

    plan = FleetPlanner(orchestrator.registry, store).plan(
        spec, {"vast": FakeProvider()}
    )

    assert plan.providers[0].offers[0]["hourly_cost_usd"] == 0.30


def test_stale_agent_is_not_counted_as_current_capacity(orchestrator, store, clock):
    registration = register_agent(
        store,
        "local-image",
        capabilities=["image-generate"],
        model_packs=["sdxl-base-1.0-fp16-76d28af"],
    )
    registration.labels["pool"] = "image-standard"
    store.register_agent(registration)
    clock.value += timedelta(seconds=121)

    plan = FleetPlanner(orchestrator.registry, store).plan(
        pool(), {"vast": FakeProvider()}
    )

    assert plan.current_slots == 0


def test_credit_must_cover_estimated_drain_cost(orchestrator, store):
    spec = pool().model_copy(
        update={
            "providers": {
                "local": pool().providers["local"],
                "vast": pool().providers["vast"].model_copy(
                    update={"available_credit_usd": 0.01}
                ),
            }
        }
    )
    orchestrator.submit(
        project_id="project-a",
        workload="image-generate",
        payload={"prompt": "image", "seed": 3},
    )

    plan = FleetPlanner(orchestrator.registry, store).plan(
        spec, {"vast": FakeProvider()}
    )

    assert plan.requested_slots == 0
    assert plan.ranked_candidates == []
    assert plan.providers[0].blocked_reason == "provider_credit_insufficient"
