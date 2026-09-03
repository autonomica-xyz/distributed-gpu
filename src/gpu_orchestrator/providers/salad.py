"""SaladCloud container-group provider driver with priced GPU-class offers."""

from __future__ import annotations

import copy
import fnmatch
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from ..models import ResourceRequirements
from .base import ProviderError, ProviderOffer, ProviderResource, WorkerLaunchSpec
from .http import JsonHttpClient

_PRIORITY_COUNT_KEYS = {
    "batch": "available_gpu_batch",
    "low": "available_gpu_low",
    "medium": "available_gpu_medium",
    "high": "available_gpu_high",
}
_PRIORITY_RELIABILITY = {"batch": 0.70, "low": 0.80, "medium": 0.90, "high": 0.95}
_PRIORITY_ORDER = {"batch": 0, "low": 1, "medium": 2, "high": 3}


def _fallback_gpu_class(
    class_id: str,
    name: str,
    batch: float,
    low: float,
    medium: float,
    high: float,
) -> dict[str, Any]:
    return {
        "id": class_id,
        "name": name,
        "prices": {"batch": batch, "low": low, "medium": medium, "high": high},
        "is_high_demand": False,
        "price_source": "reviewed_fallback",
    }


# Reviewed fallback from Salad's GPU-class API. Live prices replace these when
# the account can read /organizations/{org}/gpu-classes. Unknown classes are
# excluded rather than guessed.
_FALLBACK_GPU_CLASSES = (
    _fallback_gpu_class("6b17a5e7-9a6a-4f1c-9782-243277f4fa00", "GTX 1070, 1080, 1080Ti (8 GB)", 0.020, 0.027, 0.033, 0.040),
    _fallback_gpu_class("2aec4fc1-2270-4e40-b8cc-6e69fae61d4f", "RTX 2070 (8 GB)", 0.020, 0.033, 0.047, 0.060),
    _fallback_gpu_class("99305b63-f7cf-4b8d-ae33-2d6b6ff0b085", "RTX 3050 (8 GB)", 0.030, 0.043, 0.057, 0.070),
    _fallback_gpu_class("69071b6c-3d2c-4181-b3ce-de03aec04b9f", "RTX 3060 (8 GB)", 0.030, 0.047, 0.063, 0.080),
    _fallback_gpu_class("f51baccc-dc95-40fb-a5d1-6d0ee0db31d2", "RTX 3060 (12 GB)", 0.040, 0.053, 0.067, 0.080),
    _fallback_gpu_class("951131f6-5acf-489c-b303-0906be8b26ef", "RTX 3070 (8 GB)", 0.040, 0.060, 0.080, 0.100),
    _fallback_gpu_class("ffc51032-64d2-4df3-855a-f3a649895c0f", "RTX 2080 (8 GB)", 0.050, 0.060, 0.070, 0.080),
    _fallback_gpu_class("43a49c0c-f860-40e9-a509-702d0dba0902", "RTX 3080 (10 GB)", 0.060, 0.100, 0.140, 0.180),
    _fallback_gpu_class("0798d5aa-2d17-42ee-81b8-ea92e3bc088e", "RTX 4070 (12 GB)", 0.070, 0.120, 0.170, 0.220),
    _fallback_gpu_class("2b73eef8-be49-4667-8fc0-5c0cb127cfe0", "RTX 4060 Ti (16 GB)", 0.080, 0.127, 0.173, 0.220),
    _fallback_gpu_class("5d6b104d-c029-4357-b179-8b662d0a76b2", "RTX 5060 Ti (16 GB)", 0.070, 0.107, 0.143, 0.180),
    _fallback_gpu_class("1b8747be-e789-475b-a339-3c1028010d84", "RTX 5070 Ti (16 GB)", 0.100, 0.160, 0.220, 0.280),
    _fallback_gpu_class("0d062939-7c01-4aae-a2b1-30e315124e51", "RTX 4080 (16 GB)", 0.110, 0.167, 0.223, 0.280),
    _fallback_gpu_class("a5db5c50-cbcb-4596-ae80-6a0c8090d80f", "RTX 3090 (24 GB)", 0.090, 0.143, 0.197, 0.250),
    _fallback_gpu_class("6d4e9e99-d27e-4751-8d7d-393f7d8ea949", "RTX A5000 (24 GB)", 0.090, 0.143, 0.197, 0.250),
    _fallback_gpu_class("ed563892-aacd-40f5-80b7-90c9be6c759b", "RTX 4090 (24 GB)", 0.160, 0.207, 0.253, 0.300),
)


class SaladProvider:
    name = "salad"

    def __init__(
        self,
        api_key: str,
        organization: str,
        project: str,
        *,
        base_url: str = "https://api.salad.com/api/public",
        timeout: float = 60.0,
        opener=None,
        monotonic: Callable[[], float] = time.monotonic,
        catalog_cache_seconds: float = 300.0,
        availability_cache_seconds: float = 30.0,
        inventory_detail_cache_seconds: float = 30.0,
        availability_class_limit: int = 4,
    ):
        kwargs = {} if opener is None else {"opener": opener}
        self.organization = organization
        self.project = project
        self.http = JsonHttpClient(
            base_url,
            {
                "Salad-Api-Key": api_key,
                "User-Agent": "distributed-gpu/0.1",
            },
            timeout=timeout,
            **kwargs,
        )
        self.monotonic = monotonic
        self.catalog_cache_seconds = max(0.0, catalog_cache_seconds)
        self.availability_cache_seconds = max(0.0, availability_cache_seconds)
        self.inventory_detail_cache_seconds = max(0.0, inventory_detail_cache_seconds)
        self.availability_class_limit = max(1, availability_class_limit)
        self._catalog_cache: tuple[float, list[dict[str, Any]]] | None = None
        self._availability_cache: dict[
            tuple[str, int, int, int], tuple[float, dict[str, Any] | None]
        ] = {}
        self._inventory_detail_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._cache_lock = threading.Lock()

    @property
    def groups_path(self) -> str:
        return f"/organizations/{self.organization}/projects/{self.project}/containers"

    @property
    def gpu_classes_path(self) -> str:
        return f"/organizations/{self.organization}/gpu-classes"

    @property
    def availability_path(self) -> str:
        return f"/organizations/{self.organization}/availability/sce-gpu-availability"

    def search_offers(
        self,
        requirements: ResourceRequirements,
        *,
        max_hourly_cost_usd: float,
        limit: int = 25,
    ) -> list[ProviderOffer]:
        if requirements.gpu_count != 1 or requirements.vendor.casefold() != "nvidia":
            return []
        allowed_providers = {value.casefold() for value in requirements.allowed_providers}
        if allowed_providers and self.name not in allowed_providers:
            return []

        eligible: list[tuple[float, dict[str, Any], int, str]] = []
        for row in self._gpu_classes():
            profile = _gpu_profile(row)
            if profile is None:
                continue
            vram_mb, compute_capability = profile
            if not _matches_requirements(row["name"], vram_mb, compute_capability, requirements):
                continue
            prices = _priority_prices(row)
            usable = [price for price in prices.values() if 0 < price <= max_hourly_cost_usd]
            if usable:
                eligible.append((min(usable), row, vram_mb, compute_capability))
        eligible.sort(key=lambda item: (item[0], str(item[1]["name"]), str(item[1]["id"])))

        selected = eligible[: self.availability_class_limit]
        availability_by_class: dict[str, dict[str, Any] | None] = {}
        if selected:
            workers = min(6, len(selected))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    str(row["id"]): executor.submit(
                        self._availability, str(row["id"]), requirements
                    )
                    for _, row, _, _ in selected
                }
                for class_id, future in futures.items():
                    availability_by_class[class_id] = future.result()

        offers: list[ProviderOffer] = []
        for _, row, vram_mb, compute_capability in selected:
            availability = availability_by_class.get(str(row["id"]))
            prices = _priority_prices(row)
            for priority, price in prices.items():
                if price <= 0 or price > max_hourly_cost_usd:
                    continue
                count_key = _PRIORITY_COUNT_KEYS.get(priority)
                available_count = (
                    _optional_nonnegative_int(availability.get(count_key))
                    if availability is not None and count_key
                    else None
                )
                offers.append(
                    ProviderOffer(
                        provider=self.name,
                        offer_id=f"{row['id']}:{priority}",
                        gpu_model=str(row["name"]).strip(),
                        gpu_count=1,
                        vram_mb=vram_mb,
                        hourly_cost_usd=price,
                        reliability=_PRIORITY_RELIABILITY.get(priority, 0.8),
                        available_count=available_count,
                        raw={
                            "gpu_class_id": str(row["id"]),
                            "priority": priority,
                            "compute_capability": compute_capability,
                            "price_source": row.get("price_source", "salad_api"),
                            "availability_status": "live" if availability is not None else "unknown",
                            "on_call_count": (
                                _optional_nonnegative_int(availability.get("on_call_gpu"))
                                if availability is not None
                                else None
                            ),
                            "is_high_demand": bool(row.get("is_high_demand", False)),
                        },
                    )
                )
        offers.sort(
            key=lambda row: (
                row.hourly_cost_usd,
                row.available_count == 0,
                -1 if row.available_count is None else -row.available_count,
                row.gpu_model,
                _PRIORITY_ORDER.get(str(row.raw.get("priority")), 99),
            )
        )
        return offers[: max(0, int(limit))]

    def _gpu_classes(self) -> list[dict[str, Any]]:
        now = self.monotonic()
        with self._cache_lock:
            if self._catalog_cache and self._catalog_cache[0] > now:
                return copy.deepcopy(self._catalog_cache[1])
        rows: list[dict[str, Any]]
        try:
            data = self.http.request("GET", self.gpu_classes_path)
            items = data.get("items") if isinstance(data, dict) else None
            if not isinstance(items, list):
                raise ProviderError("Salad GPU class response did not contain an items array")
            rows = []
            for item in items:
                if not isinstance(item, dict) or not item.get("id") or not item.get("name"):
                    continue
                row = copy.deepcopy(item)
                row["price_source"] = "salad_api"
                rows.append(row)
            if not rows:
                raise ProviderError("Salad GPU class response was empty")
        except ProviderError:
            rows = [copy.deepcopy(row) for row in _FALLBACK_GPU_CLASSES]
        with self._cache_lock:
            self._catalog_cache = (now + self.catalog_cache_seconds, copy.deepcopy(rows))
        return rows

    def _availability(
        self, class_id: str, requirements: ResourceRequirements
    ) -> dict[str, Any] | None:
        cpu = max(1, int(requirements.min_cpu_cores))
        memory = max(1024, int(requirements.min_ram_mb))
        storage = max(1, int(requirements.min_disk_mb)) * 1024 * 1024
        key = (class_id, cpu, memory, storage)
        now = self.monotonic()
        with self._cache_lock:
            cached = self._availability_cache.get(key)
            if cached and cached[0] > now:
                return copy.deepcopy(cached[1])
        try:
            data = self.http.request(
                "POST",
                self.availability_path,
                {
                    "gpu_classes": [class_id],
                    "cpu": cpu,
                    "memory": memory,
                    "storage_amount": storage,
                },
            )
            result = data if isinstance(data, dict) else None
        except ProviderError:
            # Availability is an optional preflight signal, not a dependency.
            result = None
        with self._cache_lock:
            self._availability_cache[key] = (
                now + self.availability_cache_seconds,
                copy.deepcopy(result),
            )
        return result

    def launch(self, offer: ProviderOffer | None, spec: WorkerLaunchSpec) -> ProviderResource:
        payload = copy.deepcopy(spec.provider_template)
        if not payload:
            raise ValueError("Salad launch requires provider_template")
        if offer is None:
            raise ValueError("Salad launch requires a priced GPU-class offer")
        class_id = str(offer.raw.get("gpu_class_id") or "")
        priority = str(offer.raw.get("priority") or "").lower()
        if not class_id or priority not in _PRIORITY_COUNT_KEYS:
            raise ValueError("Salad offer is missing a trusted GPU class or priority")
        payload["name"] = spec.name
        payload["display_name"] = spec.name
        payload["replicas"] = 1
        container = payload.setdefault("container", {})
        container["image"] = spec.image
        container["command"] = list(spec.command)
        container["priority"] = priority
        resources = container.setdefault("resources", {})
        resources["gpu_classes"] = [class_id]
        environment = container.setdefault("environment_variables", {})
        environment.update(spec.environment)
        data = self.http.request(
            "POST", self.groups_path, payload, ok_statuses={200, 201, 202}
        )
        resource_id = str(data.get("id") if isinstance(data, dict) else spec.name)
        return ProviderResource(
            provider=self.name,
            resource_id=resource_id,
            name=spec.name,
            worker_id=spec.worker_id,
            state="creating",
            gpu_model=offer.gpu_model,
            hourly_cost_usd=offer.hourly_cost_usd,
            raw=data if isinstance(data, dict) else {},
        )

    def inventory(self) -> list[ProviderResource]:
        data = self.http.request("GET", self.groups_path)
        rows = data.get("items", []) if isinstance(data, dict) else []
        details: list[tuple[dict[str, Any], dict[str, Any], str]] = []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or row.get("display_name") or row.get("id") or "unknown")
            details.append((row, self._inventory_detail(name, row), name))

        # The list endpoint may omit priority. Individual group details retain
        # it, allowing reconciliation to repair old synthetic ledger prices.
        catalog_by_id: dict[str, dict[str, Any]] = {}
        if any(_selected_gpu_class(detail) for _, detail, _ in details):
            catalog_by_id = {str(row.get("id")): row for row in self._gpu_classes()}

        resources = []
        for summary, detail, name in details:
            container = detail.get("container") if isinstance(detail.get("container"), dict) else {}
            environment = (
                container.get("environment_variables")
                if isinstance(container.get("environment_variables"), dict)
                else {}
            )
            current_state = (
                detail.get("current_state")
                if isinstance(detail.get("current_state"), dict)
                else {}
            )
            class_id = _selected_gpu_class(detail)
            priority = str(container.get("priority") or detail.get("priority") or "").lower()
            catalog_row = catalog_by_id.get(class_id or "", {})
            hourly_cost = _priority_prices(catalog_row).get(priority)
            resources.append(
                ProviderResource(
                    provider=self.name,
                    resource_id=str(detail.get("id") or summary.get("id") or name),
                    name=name,
                    worker_id=str(environment.get("DGPU_AGENT_ID"))
                    if environment.get("DGPU_AGENT_ID")
                    else None,
                    state=_salad_instance_state(current_state, detail),
                    gpu_model=str(catalog_row.get("name")) if catalog_row.get("name") else None,
                    hourly_cost_usd=hourly_cost,
                    raw=detail,
                )
            )
        return resources

    def _inventory_detail(self, name: str, fallback: dict[str, Any]) -> dict[str, Any]:
        now = self.monotonic()
        with self._cache_lock:
            cached = self._inventory_detail_cache.get(name)
            if cached is not None and cached[0] > now:
                return copy.deepcopy(cached[1])
        try:
            data = self.http.request("GET", f"{self.groups_path}/{name}")
            detail = (
                data
                if isinstance(data, dict) and (data.get("id") or data.get("name"))
                else fallback
            )
        except ProviderError:
            detail = fallback
        with self._cache_lock:
            self._inventory_detail_cache[name] = (
                now + self.inventory_detail_cache_seconds,
                copy.deepcopy(detail),
            )
        return detail

    def destroy(self, resource_id: str) -> None:
        resource = next(
            (
                item
                for item in self.inventory()
                if item.resource_id == resource_id or item.name == resource_id
            ),
            None,
        )
        if resource is None:
            return
        self.http.request(
            "DELETE",
            f"{self.groups_path}/{resource.name}",
            ok_statuses={200, 202, 204},
        )

    def account_credit(self) -> float | None:
        return None


def _selected_gpu_class(group: dict[str, Any]) -> str | None:
    container = group.get("container") if isinstance(group.get("container"), dict) else {}
    resources = container.get("resources") if isinstance(container.get("resources"), dict) else {}
    classes = resources.get("gpu_classes")
    if isinstance(classes, list) and classes:
        return str(classes[0])
    return None


def _salad_instance_state(current_state: dict[str, Any], group: dict[str, Any]) -> str:
    counts = current_state.get("instance_status_counts")
    counts = counts if isinstance(counts, dict) else {}
    running = _optional_nonnegative_int(counts.get("running_count")) or 0
    creating = _optional_nonnegative_int(counts.get("creating_count")) or 0
    allocating = _optional_nonnegative_int(counts.get("allocating_count")) or 0
    stopping = _optional_nonnegative_int(counts.get("stopping_count")) or 0
    if running > 0:
        return "running"
    if creating > 0:
        return "creating"
    if allocating > 0:
        return "allocating"
    if stopping > 0:
        return "stopping"
    return str(
        current_state.get("status")
        or group.get("status")
        or group.get("state")
        or "unknown"
    ).lower()


def _priority_prices(row: dict[str, Any]) -> dict[str, float]:
    raw = row.get("prices")
    result: dict[str, float] = {}
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = (
            (item.get("priority"), item.get("price"))
            for item in raw
            if isinstance(item, dict)
        )
    else:
        items = ()
    for priority, value in items:
        key = str(priority or "").lower()
        if key not in _PRIORITY_COUNT_KEYS:
            continue
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(price) and price > 0:
            result[key] = price
    return result


def _gpu_profile(row: dict[str, Any]) -> tuple[int, str] | None:
    name = str(row.get("name") or "")
    match = re.search(r"\((\d+)\s*GB\)", name, re.IGNORECASE)
    if not match:
        return None
    compute_capability = _infer_compute_capability(name)
    if compute_capability is None:
        return None
    return int(match.group(1)) * 1024, compute_capability


def _infer_compute_capability(name: str) -> str | None:
    text = name.upper()
    if re.search(r"GTX\s+10(?:50|60|70|80)", text):
        return "6.1"
    if re.search(r"GTX\s+16\d{2}", text) or re.search(r"(?:QUADRO\s+)?RTX\s+20\d{2}", text):
        return "7.5"
    if re.search(r"RTX\s+30\d{2}", text) or re.search(r"RTX\s+A[456]\d{3}", text):
        return "8.6"
    if re.search(r"RTX\s+40\d{2}", text):
        return "8.9"
    if re.search(r"RTX\s+50\d{2}", text):
        return "12.0"
    return None


def _matches_requirements(
    name: str,
    vram_mb: int,
    compute_capability: str,
    requirements: ResourceRequirements,
) -> bool:
    if vram_mb < requirements.min_vram_mb:
        return False
    if _version_tuple(compute_capability) < _version_tuple(requirements.min_compute_capability):
        return False
    model = name.casefold()
    allowed = [value.casefold() for value in requirements.gpu_models]
    excluded = [value.casefold() for value in requirements.excluded_gpu_models]
    if allowed and not any(fnmatch.fnmatch(model, pattern) for pattern in allowed):
        return False
    if any(fnmatch.fnmatch(model, pattern) for pattern in excluded):
        return False
    return True


def _version_tuple(value: str | None) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in str(value or "").split("."))
    except ValueError:
        return ()


def _optional_nonnegative_int(value: Any) -> int | None:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None
