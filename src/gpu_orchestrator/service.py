"""Application service joining admission policy with the durable queue."""

from __future__ import annotations

import re
from typing import Any, Sequence

from .models import AgentRegistration, OutputArtifact
from .registry import WorkloadRegistry
from .store import Store


PROJECT_RE = re.compile(r"^[a-z][a-z0-9_-]{1,62}$")


class OutputContractError(ValueError):
    pass


class Orchestrator:
    def __init__(self, registry: WorkloadRegistry, store: Store):
        self.registry = registry
        self.store = store

    def submit(
        self,
        *,
        project_id: str,
        workload: str,
        payload: dict[str, Any],
        route: str | None = None,
        priority: int = 0,
        idempotency_key: str | None = None,
        inputs: Sequence[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        if not PROJECT_RE.fullmatch(project_id):
            raise ValueError("project_id must match ^[a-z][a-z0-9_-]{1,62}$")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        admitted = self.registry.admit(workload, payload, route)
        self._validate_inputs(admitted.input_contract, inputs)
        return self.store.submit_job(
            admitted,
            project_id=project_id,
            priority=priority,
            idempotency_key=idempotency_key,
            inputs=inputs,
        )

    def register_agent(self, registration: AgentRegistration) -> dict[str, Any]:
        return self.store.register_agent(registration)

    def claim(self, slot_id: str, *, lease_seconds: int = 90) -> dict[str, Any] | None:
        return self.store.claim_next(slot_id, lease_seconds=lease_seconds)

    def confirm(self, job_id: str, slot_id: str, lease_token: str) -> dict[str, Any]:
        return self.store.confirm(job_id, slot_id, lease_token)

    def heartbeat(
        self,
        job_id: str,
        slot_id: str,
        lease_token: str,
        *,
        lease_seconds: int = 90,
    ) -> bool:
        return self.store.heartbeat(
            job_id, slot_id, lease_token, lease_seconds=lease_seconds
        )

    def complete(
        self,
        job_id: str,
        slot_id: str,
        lease_token: str,
        outputs: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        validated = [OutputArtifact.model_validate(output).model_dump(mode="json") for output in outputs]
        self._validate_outputs(job["output_contract"], validated)
        return self.store.complete(job_id, slot_id, lease_token, validated)

    def fail(
        self,
        job_id: str,
        slot_id: str,
        lease_token: str,
        *,
        error_code: str,
        error_message: str,
    ) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        retryable = error_code in set(job["retryable_error_codes"])
        return self.store.fail(
            job_id,
            slot_id,
            lease_token,
            error_code=error_code,
            error_message=error_message,
            retryable=retryable,
        )

    @staticmethod
    def _validate_inputs(
        contract: Sequence[dict[str, Any]], inputs: Sequence[dict[str, Any]]
    ) -> None:
        declared = {item["name"]: item for item in contract}
        counts: dict[str, int] = {}
        for artifact in inputs:
            name = str(artifact.get("name") or "")
            if name not in declared:
                raise ValueError(f"undeclared input artifact {name!r}")
            counts[name] = counts.get(name, 0) + 1
            if counts[name] > 1 and not declared[name].get("multiple"):
                raise ValueError(f"input artifact {name!r} does not allow multiple values")
        missing = [
            name for name, spec in declared.items() if spec.get("required") and counts.get(name, 0) == 0
        ]
        if missing:
            raise ValueError(f"missing required input artifacts: {', '.join(sorted(missing))}")

    @staticmethod
    def _validate_outputs(
        contract: Sequence[dict[str, Any]], outputs: Sequence[dict[str, Any]]
    ) -> None:
        declared = {item["name"]: item for item in contract}
        counts: dict[str, int] = {}
        for artifact in outputs:
            name = artifact["name"]
            spec = declared.get(name)
            if spec is None:
                raise OutputContractError(f"undeclared output artifact {name!r}")
            counts[name] = counts.get(name, 0) + 1
            if counts[name] > 1 and not spec.get("multiple"):
                raise OutputContractError(f"output artifact {name!r} does not allow multiple values")
            if artifact["size"] > int(spec["max_bytes"]):
                raise OutputContractError(
                    f"output artifact {name!r} exceeds max_bytes={spec['max_bytes']}"
                )
            media_types = set(spec.get("media_types") or [])
            if media_types and artifact["media_type"] not in media_types:
                raise OutputContractError(
                    f"output artifact {name!r} has disallowed media type {artifact['media_type']!r}"
                )
        missing = [
            name for name, spec in declared.items() if spec.get("required") and counts.get(name, 0) == 0
        ]
        if missing:
            raise OutputContractError(
                f"missing required output artifacts: {', '.join(sorted(missing))}"
            )

