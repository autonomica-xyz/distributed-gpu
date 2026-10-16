"""Vast.ai marketplace driver using its public REST API."""

from __future__ import annotations

import math
import re
import shlex
from typing import Any

from ..models import ResourceRequirements
from .base import ProviderError, ProviderOffer, ProviderResource, WorkerLaunchSpec
from .http import JsonHttpClient


class VastProvider:
    name = "vast"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://console.vast.ai/api",
        timeout: float = 60.0,
        opener=None,
    ):
        kwargs = {} if opener is None else {"opener": opener}
        self.http = JsonHttpClient(
            base_url,
            {
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "sparehour/0.1",
            },
            timeout=timeout,
            **kwargs,
        )

    def search_offers(
        self,
        requirements: ResourceRequirements,
        *,
        max_hourly_cost_usd: float,
        limit: int = 25,
    ) -> list[ProviderOffer]:
        payload: dict[str, Any] = {
            "limit": max(1, min(limit, 1000)),
            "type": "ondemand",
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "num_gpus": {"eq": 1},
            "gpu_ram": {"gte": requirements.min_vram_mb},
            "cpu_cores_effective": {"gte": requirements.min_cpu_cores},
            "cpu_ram": {"gte": requirements.min_ram_mb},
            "dph_total": {"lte": max_hourly_cost_usd},
            "allocated_storage": max(1, requirements.min_disk_mb // 1024),
            "verified": {"eq": True},
            "reliability": {"gte": 0.95},
            "order": [["dph_total", "asc"]],
        }
        if requirements.gpu_models:
            # Vast expects concrete names here; wildcard enforcement is still
            # repeated locally below.
            concrete = [name for name in requirements.gpu_models if "*" not in name and "?" not in name]
            if concrete:
                payload["gpu_name"] = {"in": concrete}
        data = self.http.request("POST", "/v0/bundles/", payload)
        rows = data.get("offers", []) if isinstance(data, dict) else []
        offers: list[ProviderOffer] = []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            try:
                offer_id = row.get("id") or row.get("ask_contract_id")
                cost = _first_float(row, "dph_total", "discounted_dph_total", "discounted_hourly")
                vram = int(row.get("gpu_ram") or 0)
                if offer_id is None or cost is None or cost > max_hourly_cost_usd:
                    continue
                offers.append(
                    ProviderOffer(
                        provider=self.name,
                        offer_id=str(offer_id),
                        gpu_model=str(row.get("gpu_name") or "unknown"),
                        gpu_count=int(row.get("num_gpus") or 1),
                        vram_mb=vram,
                        hourly_cost_usd=cost,
                        reliability=(
                            float(row["reliability"])
                            if row.get("reliability") is not None
                            and math.isfinite(float(row["reliability"]))
                            else None
                        ),
                        available_count=None,
                        raw=row,
                    )
                )
            except (TypeError, ValueError):
                continue
        return sorted(offers, key=lambda offer: (offer.hourly_cost_usd, -(offer.reliability or 0)))

    def launch(self, offer: ProviderOffer | None, spec: WorkerLaunchSpec) -> ProviderResource:
        if offer is None or offer.provider != self.name:
            raise ValueError("Vast launch requires a Vast offer")
        payload: dict[str, Any] = {
            "image": spec.image,
            "label": f"{spec.name} worker_id={spec.worker_id}",
            "disk": spec.disk_gb,
            "runtype": "args",
            "target_state": "running",
            "cancel_unavail": True,
            "env": _docker_environment(spec.environment),
            "args": list(spec.command),
        }
        data = self.http.request("PUT", f"/v0/asks/{offer.offer_id}/", payload)
        if not isinstance(data, dict) or not data.get("success") or not data.get("new_contract"):
            raise ProviderError(f"Vast create did not return success/new_contract: {data!r}")
        resource_id = str(data["new_contract"])
        return ProviderResource(
            provider=self.name,
            resource_id=resource_id,
            name=spec.name,
            worker_id=spec.worker_id,
            state="creating",
            gpu_model=offer.gpu_model,
            hourly_cost_usd=offer.hourly_cost_usd,
            raw=data,
        )

    def inventory(self) -> list[ProviderResource]:
        data = self.http.request("GET", "/v1/instances/")
        if isinstance(data, dict):
            rows = data.get("instances", data.get("items", []))
        else:
            rows = data
        if not isinstance(rows, list):
            raise ProviderError("Vast inventory did not contain an instance list")
        resources = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            resource_id = row.get("id") or row.get("instance_id")
            if resource_id is None:
                continue
            label = str(row.get("label") or row.get("name") or resource_id)
            resources.append(
                ProviderResource(
                    provider=self.name,
                    resource_id=str(resource_id),
                    name=label,
                    worker_id=_worker_id(row, label),
                    state=str(
                        row.get("actual_status") or row.get("status") or row.get("cur_state") or "unknown"
                    ).lower(),
                    gpu_model=str(row.get("gpu_name")) if row.get("gpu_name") else None,
                    hourly_cost_usd=_first_float(row, "dph_total"),
                    raw=row,
                )
            )
        return resources

    def destroy(self, resource_id: str) -> None:
        self.http.request("DELETE", f"/v0/instances/{int(resource_id)}/")

    def account_credit(self) -> float | None:
        # Vast has changed account response shapes over time. Unknown credit is
        # safer than guessing when the response does not expose a known field.
        data = self.http.request("GET", "/v0/users/current/")
        if isinstance(data, dict):
            for key in ("credit", "balance", "current_balance"):
                try:
                    if data.get(key) is not None:
                        value = float(data[key])
                        if math.isfinite(value):
                            return value
                except (TypeError, ValueError):
                    pass
        return None


def _first_float(row: dict[str, Any], *keys: str) -> float | None:
    sources = [row]
    search = row.get("search") if isinstance(row.get("search"), dict) else {}
    instance = row.get("instance") if isinstance(row.get("instance"), dict) else {}
    for source in (*sources, {**search, **instance}):
        for key in (keys if source is row else ("discountedTotalPerHour", "totalHour")):
            try:
                if source.get(key) is not None:
                    value = float(source[key])
                    if math.isfinite(value):
                        return value
            except (TypeError, ValueError):
                pass
    return None


def _worker_id(row: dict[str, Any], label: str) -> str | None:
    env = row.get("env")
    if isinstance(env, dict) and env.get("SPAREHOUR_AGENT_ID"):
        return str(env["SPAREHOUR_AGENT_ID"])
    if "worker_id=" in label:
        return label.split("worker_id=", 1)[1].split()[0]
    return None


def _docker_environment(environment: dict[str, str]) -> str:
    arguments = []
    for name, value in sorted(environment.items()):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"invalid environment variable name {name!r}")
        arguments.extend(["-e", shlex.quote(f"{name}={value}")])
    return " ".join(arguments)
