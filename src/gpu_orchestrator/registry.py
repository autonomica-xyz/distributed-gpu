"""Workload registry, admission validation, and immutable job snapshots."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .models import RouteSpec, WorkloadSpec


class RegistryError(ValueError):
    pass


class AdmissionError(ValueError):
    pass


@dataclass(frozen=True)
class AdmittedJob:
    workload: str
    workload_version: str
    route: str
    payload: dict[str, Any]
    requirements: dict[str, Any]
    executor: dict[str, Any]
    input_contract: list[dict[str, Any]]
    output_contract: list[dict[str, Any]]
    cost_unit: str
    estimated_units: float
    max_attempts: int
    retryable_error_codes: frozenset[str]


def _lookup_number(document: dict[str, Any], dotted_path: str) -> float:
    current = _lookup_value(document, dotted_path)
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise AdmissionError(f"cost field {dotted_path!r} must be numeric")
    value = float(current)
    if not math.isfinite(value):
        raise AdmissionError(f"cost field {dotted_path!r} must be finite")
    return value


def _ensure_finite_payload(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AdmissionError(f"payload {path} must be a finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _ensure_finite_payload(item, f"{path}.{key}")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_finite_payload(item, f"{path}[{index}]")


def _lookup_value(document: dict[str, Any], dotted_path: str) -> Any:
    current: Any = document
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise AdmissionError(f"cost field {dotted_path!r} is missing")
        current = current[part]
    return current


class WorkloadRegistry:
    def __init__(self, workloads: Iterable[WorkloadSpec] = ()):
        self._workloads: dict[str, WorkloadSpec] = {}
        self._validators: dict[str, Draft202012Validator] = {}
        for workload in workloads:
            self.add(workload)

    @classmethod
    def from_directory(cls, directory: str | Path) -> "WorkloadRegistry":
        root = Path(directory)
        if not root.exists():
            raise RegistryError(f"workload directory does not exist: {root}")
        registry = cls()
        files = sorted([*root.glob("*.yaml"), *root.glob("*.yml")])
        if not files:
            raise RegistryError(f"no workload YAML files found in {root}")
        for path in files:
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                workload = WorkloadSpec.model_validate(raw)
            except Exception as exc:
                raise RegistryError(f"invalid workload {path}: {exc}") from exc
            registry.add(workload)
        return registry

    def add(self, workload: WorkloadSpec) -> None:
        if workload.name in self._workloads:
            raise RegistryError(f"duplicate workload {workload.name!r}")
        try:
            Draft202012Validator.check_schema(workload.payload_schema)
        except SchemaError as exc:
            raise RegistryError(f"invalid payload schema for {workload.name}: {exc.message}") from exc
        self._workloads[workload.name] = workload
        self._validators[workload.name] = Draft202012Validator(workload.payload_schema)

    def names(self) -> list[str]:
        return sorted(self._workloads)

    def get(self, name: str) -> WorkloadSpec:
        try:
            return self._workloads[name]
        except KeyError as exc:
            raise AdmissionError(f"unknown workload {name!r}") from exc

    def admit(
        self,
        workload_name: str,
        payload: dict[str, Any],
        route_name: str | None = None,
    ) -> AdmittedJob:
        workload = self.get(workload_name)
        errors = sorted(
            self._validators[workload_name].iter_errors(payload),
            key=lambda error: list(error.absolute_path),
        )
        if errors:
            error = errors[0]
            path = ".".join(str(item) for item in error.absolute_path) or "$"
            raise AdmissionError(f"payload {path}: {error.message}")

        _ensure_finite_payload(payload)

        route_id = route_name or workload.default_route
        try:
            route: RouteSpec = workload.routes[route_id]
        except KeyError as exc:
            raise AdmissionError(
                f"unknown route {route_id!r} for workload {workload_name!r}"
            ) from exc

        cost = route.cost
        units = cost.default_units_per_job
        if cost.payload_units_field:
            units = _lookup_number(payload, cost.payload_units_field) * cost.payload_units_multiplier
        if cost.payload_items_field:
            items = _lookup_value(payload, cost.payload_items_field)
            if not isinstance(items, list):
                raise AdmissionError(
                    f"cost items field {cost.payload_items_field!r} must be an array"
                )
            units *= len(items)
        if not math.isfinite(units) or units <= 0:
            raise AdmissionError("estimated workload units must be a finite positive number")

        requirements = route.resources.model_dump(mode="json")
        requirements["capability"] = route.compatibility.capability
        requirements["model_packs"] = list(route.compatibility.model_packs)
        requirements["route"] = f"{workload.name}/{route_id}"

        return AdmittedJob(
            workload=workload.name,
            workload_version=workload.version,
            route=route_id,
            payload=copy.deepcopy(payload),
            requirements=requirements,
            executor=route.executor.model_dump(mode="json"),
            input_contract=[item.model_dump(mode="json") for item in workload.inputs],
            output_contract=[item.model_dump(mode="json") for item in workload.outputs],
            cost_unit=cost.unit,
            estimated_units=float(units),
            max_attempts=workload.retry.max_attempts,
            retryable_error_codes=frozenset(workload.retry.retryable_error_codes),
        )
