"""Authenticated HTTP API for clients, agents, and the fleet controller."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import re
from typing import Annotated, Any, Callable, Sequence

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .artifacts import ArtifactError, ArtifactStore
from .dashboard import build_overview, build_queues, build_recent, dashboard_html, fleet_agents
from .deploy import DeploymentError, DeploymentService
from .models import AgentRegistration
from .registry import AdmissionError, RegistryError
from .scaling_engine import ScalingEngine
from . import scaling_policy
from .providers.base import ProviderError
from .service import Orchestrator, OutputContractError
from .store import ConflictError, LeaseError, NotFoundError, StoreError


LeaseHeader = Annotated[str, Header(alias="X-DGPU-Lease-Token")]
LOGGER = logging.getLogger(__name__)


class SubmitJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    workload: str
    route: str | None = None
    payload: dict[str, Any]
    inputs: list[dict[str, Any]] = Field(default_factory=list)
    priority: int = 0
    idempotency_key: str | None = None


class AgentHeartbeatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disk_free_mb: int | None = Field(default=None, ge=0)
    draining: bool | None = None


class ClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lease_seconds: int = Field(default=90, ge=10, le=3600)


class CompleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outputs: list[dict[str, Any]]


class FailRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_code: str = Field(pattern=r"^[a-z][a-z0-9_]{1,62}$")
    error_message: str = Field(max_length=4000)


class ReleaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1, max_length=500)


class IssueTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    subject: str
    scopes: dict[str, Any] = Field(default_factory=dict)
    expires_at: datetime | None = None
    token_id: str | None = None


class UpdateTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    subject: str
    scopes: dict[str, Any] = Field(default_factory=dict)
    expires_at: datetime | None = None


class DeployRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pool_id: str
    provider: str
    count: int = Field(default=1, ge=1, le=5)
    expires_hours: int = Field(default=24, ge=1, le=168)
    offer_id: str | None = Field(default=None, min_length=1, max_length=200)


def create_app(
    orchestrator: Orchestrator,
    artifact_store: ArtifactStore | None = None,
    bootstrap_directory: str | Path | None = None,
    deployment_service: DeploymentService | None = None,
    scaling_engine: ScalingEngine | None = None,
    *,
    enable_dashboard: bool = False,
    deployment_reconcile_seconds: float | None = None,
    scaling_evaluate_seconds: float | None = None,
    autonomous_apply: bool | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        interval = float(deployment_reconcile_seconds or 0)
        scaling_interval = float(scaling_evaluate_seconds or 0)

        if (deployment_service is None or interval <= 0) and (scaling_engine is None or scaling_interval <= 0):
            yield
            return
        stop = asyncio.Event()
        async def reconcile_loop() -> None:
            while not stop.is_set():
                try:
                    app.state.deployment_reconcile_last = await asyncio.to_thread(
                        deployment_service.reconcile
                    )
                except Exception:
                    LOGGER.exception("background provider deployment reconciliation failed")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass

        tasks: list[asyncio.Task[Any]] = []
        if deployment_service is not None and interval > 0:
            task = asyncio.create_task(reconcile_loop(), name="deployment-reconciler")
            tasks.append(task); app.state.deployment_reconciler_task = task
        if scaling_engine is not None and scaling_interval > 0:
            async def scaling_loop() -> None:
                while not stop.is_set():
                    try: await asyncio.to_thread(scaling_engine.evaluate_all)
                    except Exception: LOGGER.exception("background scaling evaluation failed")
                    try: await asyncio.wait_for(stop.wait(), timeout=scaling_interval)
                    except asyncio.TimeoutError: pass
            tasks.append(asyncio.create_task(scaling_loop(), name="scaling-evaluator"))
        try:
            yield
        finally:
            stop.set()
            for task in tasks: task.cancel()
            for task in tasks:
                with suppress(asyncio.CancelledError): await task

    app = FastAPI(
        title="Distributed GPU Control Plane",
        version="0.1.0",
        description="Distribute independent jobs across inventory-matched GPU slots.",
        lifespan=lifespan,
    )
    app.state.orchestrator = orchestrator
    if scaling_engine is None and deployment_service is not None:
        scaling_engine = ScalingEngine(
            store=orchestrator.store,
            registry=orchestrator.registry,
            pools_directory=deployment_service.pools_directory,
            drivers=dict(deployment_service.drivers),
            public_controller_url=deployment_service.public_controller_url,
            wheel_url=deployment_service.wheel_url,
            allow_apply=(autonomous_apply if autonomous_apply is not None else os.environ.get("DGPU_AUTONOMOUS_APPLY", "").lower() in {"1", "true", "yes"}),
        )
    app.state.scaling_engine = scaling_engine
    app.state.automation_paused = False
    bootstrap_root = (
        Path(bootstrap_directory).resolve() if bootstrap_directory is not None else None
    )

    @app.exception_handler(NotFoundError)
    async def not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": str(exc)})

    @app.exception_handler(ConflictError)
    async def conflict(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=409, content={"error": "conflict", "detail": str(exc)})

    @app.exception_handler(ValueError)
    async def invalid(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=422, content={"error": "invalid_request", "detail": str(exc)})

    @app.exception_handler(StoreError)
    async def store_error(_: Request, exc: StoreError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"error": "store_error", "detail": str(exc)})

    @app.exception_handler(ProviderError)
    async def provider_error(_: Request, exc: ProviderError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"error": "provider_error", "detail": str(exc)})

    def authenticate(authorization: Annotated[str | None, Header()] = None) -> dict[str, Any]:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        try:
            return orchestrator.store.authenticate(authorization.removeprefix("Bearer ").strip())
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    def roles(*allowed: str) -> Callable[..., dict[str, Any]]:
        def dependency(principal: dict[str, Any] = Depends(authenticate)) -> dict[str, Any]:
            if principal["role"] not in allowed:
                raise HTTPException(status_code=403, detail="token role is not allowed")
            return principal

        return dependency

    client_principal = roles("admin", "client")
    agent_principal = roles("admin", "agent")
    controller_principal = roles("admin", "controller")
    any_principal = roles("admin", "client", "agent", "controller")
    dashboard_principal = roles("admin", "client", "controller")
    admin_principal = roles("admin")

    def require_project(principal: dict[str, Any], project_id: str) -> None:
        if principal["role"] == "admin":
            return
        projects = set(principal["scopes"].get("projects") or [])
        if project_id != principal["subject"] and "*" not in projects and project_id not in projects:
            raise HTTPException(status_code=403, detail="token is not scoped to this project")

    def require_submission_scope(
        principal: dict[str, Any], workload: str, route: str | None
    ) -> None:
        if principal["role"] == "admin":
            return
        scopes = principal.get("scopes") or {}
        workloads = set(scopes.get("workloads") or [])
        if workloads and "*" not in workloads and workload not in workloads:
            raise HTTPException(status_code=403, detail="token is not scoped to this workload")
        selected_route = route or orchestrator.registry.get(workload).default_route
        routes = set(scopes.get("routes") or [])
        if routes and "*" not in routes and selected_route not in routes:
            raise HTTPException(status_code=403, detail="token is not scoped to this route")

    def require_agent(principal: dict[str, Any], agent_id: str) -> None:
        if principal["role"] != "admin" and principal["subject"] != agent_id:
            raise HTTPException(status_code=403, detail="token is not scoped to this agent")

    def require_slot(principal: dict[str, Any], slot_id: str) -> None:
        slot = orchestrator.store.get_slot(slot_id)
        require_agent(principal, slot["agent_id"])

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "workloads": len(orchestrator.registry.names())}

    @app.get("/bootstrap/{filename}")
    def bootstrap_package(filename: str) -> FileResponse:
        """Serve non-secret worker wheels during a temporary bootstrap.

        The package contains no worker credential. All control-plane actions
        still require the separately scoped bearer token injected by the
        provider. Restricting names to wheel files prevents this endpoint from
        becoming a general local-file server.
        """

        if bootstrap_root is None:
            raise NotFoundError("worker bootstrap is not configured")
        if not re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", filename):
            raise NotFoundError("bootstrap package not found")
        path = (bootstrap_root / filename).resolve()
        try:
            path.relative_to(bootstrap_root)
        except ValueError as exc:
            raise NotFoundError("bootstrap package not found") from exc
        if not path.is_file():
            raise NotFoundError("bootstrap package not found")
        return FileResponse(path, media_type="application/octet-stream", filename=filename)

    @app.get("/v1/workloads")
    def workloads(_: dict[str, Any] = Depends(any_principal)) -> list[dict[str, Any]]:
        return [
            orchestrator.registry.get(name).model_dump(mode="json")
            for name in orchestrator.registry.names()
        ]

    @app.post("/v1/jobs", status_code=status.HTTP_201_CREATED)
    def submit_job(
        body: SubmitJobRequest,
        principal: dict[str, Any] = Depends(client_principal),
        idempotency_header: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> dict[str, Any]:
        require_project(principal, body.project_id)
        require_submission_scope(principal, body.workload, body.route)
        return orchestrator.submit(
            project_id=body.project_id,
            workload=body.workload,
            route=body.route,
            payload=body.payload,
            inputs=body.inputs,
            priority=body.priority,
            idempotency_key=idempotency_header or body.idempotency_key,
        )

    @app.get("/v1/jobs")
    def list_jobs(
        project_id: str | None = None,
        workload: str | None = None,
        route: str | None = None,
        state_filter: Annotated[str | None, Query(alias="state")] = None,
        worker_id: str | None = None,
        slot_id: str | None = None,
        query: Annotated[str | None, Query(alias="q", max_length=200)] = None,
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        principal: dict[str, Any] = Depends(client_principal),
    ) -> list[dict[str, Any]]:
        if principal["role"] != "admin":
            if project_id is None:
                project_id = principal["subject"]
            require_project(principal, project_id)
        return orchestrator.store.list_jobs(
            project_id=project_id,
            workload=workload,
            route=route,
            state=state_filter,
            worker_id=worker_id,
            slot_id=slot_id,
            query=query,
            limit=limit,
            offset=offset,
        )

    @app.get("/v1/jobs/{job_id}")
    def get_job(
        job_id: str, principal: dict[str, Any] = Depends(client_principal)
    ) -> dict[str, Any]:
        job = orchestrator.store.get_job(job_id)
        require_project(principal, job["project_id"])
        return job

    @app.get("/v1/jobs/{job_id}/attempts")
    def attempts(
        job_id: str, principal: dict[str, Any] = Depends(client_principal)
    ) -> list[dict[str, Any]]:
        job = orchestrator.store.get_job(job_id)
        require_project(principal, job["project_id"])
        return orchestrator.store.job_attempts(job_id)

    @app.get("/v1/jobs/{job_id}/events")
    def events(
        job_id: str, principal: dict[str, Any] = Depends(client_principal)
    ) -> list[dict[str, Any]]:
        job = orchestrator.store.get_job(job_id)
        require_project(principal, job["project_id"])
        return orchestrator.store.job_events(job_id)

    @app.delete("/v1/jobs/{job_id}")
    def cancel_job(
        job_id: str, principal: dict[str, Any] = Depends(client_principal)
    ) -> dict[str, Any]:
        job = orchestrator.store.get_job(job_id)
        require_project(principal, job["project_id"])
        return orchestrator.store.cancel(job_id, actor=principal["subject"])

    @app.post("/v1/agents/register")
    def register_agent(
        body: AgentRegistration,
        principal: dict[str, Any] = Depends(agent_principal),
    ) -> dict[str, Any]:
        require_agent(principal, body.agent_id)
        route_scopes = set(principal["scopes"].get("routes") or [])
        if principal["role"] != "admin" and route_scopes and "*" not in route_scopes:
            for slot in body.slots:
                if not set(slot.capabilities).issubset(route_scopes):
                    raise HTTPException(status_code=403, detail="agent advertised an unscoped capability")
        return orchestrator.register_agent(body)

    @app.post("/v1/agents/{agent_id}/heartbeat")
    def heartbeat_agent(
        agent_id: str,
        body: AgentHeartbeatRequest,
        principal: dict[str, Any] = Depends(agent_principal),
    ) -> dict[str, Any]:
        require_agent(principal, agent_id)
        return orchestrator.store.heartbeat_agent(
            agent_id,
            disk_free_mb=body.disk_free_mb,
            draining=body.draining,
        )

    @app.post("/v1/slots/{slot_id}/claim")
    def claim(
        slot_id: str,
        body: ClaimRequest,
        principal: dict[str, Any] = Depends(agent_principal),
    ) -> dict[str, Any] | None:
        require_slot(principal, slot_id)
        return orchestrator.claim(slot_id, lease_seconds=body.lease_seconds)

    @app.post("/v1/jobs/{job_id}/confirm")
    def confirm(
        job_id: str,
        slot_id: str,
        lease_token: LeaseHeader,
        principal: dict[str, Any] = Depends(agent_principal),
    ) -> dict[str, Any]:
        require_slot(principal, slot_id)
        return orchestrator.confirm(job_id, slot_id, lease_token)

    @app.post("/v1/jobs/{job_id}/heartbeat")
    def heartbeat_job(
        job_id: str,
        slot_id: str,
        lease_token: LeaseHeader,
        lease_seconds: int = Query(default=90, ge=10, le=3600),
        principal: dict[str, Any] = Depends(agent_principal),
    ) -> dict[str, Any]:
        require_slot(principal, slot_id)
        return {
            "cancel_requested": orchestrator.heartbeat(
                job_id, slot_id, lease_token, lease_seconds=lease_seconds
            )
        }

    @app.post("/v1/jobs/{job_id}/complete")
    def complete(
        job_id: str,
        slot_id: str,
        body: CompleteRequest,
        lease_token: LeaseHeader,
        principal: dict[str, Any] = Depends(agent_principal),
    ) -> dict[str, Any]:
        require_slot(principal, slot_id)
        return orchestrator.complete(job_id, slot_id, lease_token, body.outputs)

    @app.put("/v1/jobs/{job_id}/outputs/{artifact_name}")
    async def upload_output(
        job_id: str,
        artifact_name: str,
        request: Request,
        slot_id: str,
        lease_token: LeaseHeader,
        content_type: Annotated[str | None, Header(alias="Content-Type")] = None,
        principal: dict[str, Any] = Depends(agent_principal),
    ) -> dict[str, Any]:
        if artifact_store is None:
            raise HTTPException(status_code=503, detail="artifact store is not configured")
        require_slot(principal, slot_id)
        job = orchestrator.store.verify_lease(
            job_id, slot_id, lease_token, allowed_states={"running"}
        )
        spec = next(
            (item for item in job["output_contract"] if item["name"] == artifact_name),
            None,
        )
        if spec is None:
            raise HTTPException(status_code=422, detail="output is not declared by the workload")
        media_type = (content_type or "application/octet-stream").split(";", 1)[0].strip()
        allowed_types = set(spec.get("media_types") or [])
        if allowed_types and media_type not in allowed_types:
            raise HTTPException(status_code=422, detail="output media type is not allowed")
        try:
            digest, size, _ = await artifact_store.ingest(
                request.stream(),
                max_bytes=int(spec["max_bytes"]),
                validator=str(spec.get("validator") or "sha256"),
                json_schema=spec.get("json_schema"),
            )
        except ArtifactError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "name": artifact_name,
            "uri": f"/v1/jobs/{job_id}/artifacts/{digest}",
            "media_type": media_type,
            "size": size,
            "sha256": digest,
        }

    @app.get("/v1/jobs/{job_id}/artifacts/{sha256}")
    def download_output(
        job_id: str,
        sha256: str,
        principal: dict[str, Any] = Depends(client_principal),
    ) -> FileResponse:
        if artifact_store is None:
            raise HTTPException(status_code=503, detail="artifact store is not configured")
        job = orchestrator.store.get_job(job_id)
        require_project(principal, job["project_id"])
        artifact = next(
            (item for item in job["outputs"] if item.get("sha256") == sha256), None
        )
        if artifact is None:
            raise NotFoundError("artifact is not attached to this job")
        path = artifact_store.path_for(sha256)
        if not path.is_file():
            raise NotFoundError("artifact content is missing")
        return FileResponse(
            path,
            media_type=artifact["media_type"],
            filename=f"{artifact['name']}-{sha256[:12]}",
        )

    @app.post("/v1/jobs/{job_id}/fail")
    def fail(
        job_id: str,
        slot_id: str,
        body: FailRequest,
        lease_token: LeaseHeader,
        principal: dict[str, Any] = Depends(agent_principal),
    ) -> dict[str, Any]:
        require_slot(principal, slot_id)
        return orchestrator.fail(
            job_id,
            slot_id,
            lease_token,
            error_code=body.error_code,
            error_message=body.error_message,
        )

    @app.post("/v1/jobs/{job_id}/release")
    def release(
        job_id: str,
        slot_id: str,
        body: ReleaseRequest,
        lease_token: LeaseHeader,
        principal: dict[str, Any] = Depends(agent_principal),
    ) -> dict[str, Any]:
        require_slot(principal, slot_id)
        return orchestrator.store.release(
            job_id, slot_id, lease_token, reason=body.reason
        )

    @app.get("/v1/control/agents")
    def agents(
        provider: str | None = None,
        _: dict[str, Any] = Depends(controller_principal),
    ) -> list[dict[str, Any]]:
        rows = fleet_agents(orchestrator)
        return [row for row in rows if provider is None or row["provider"] == provider]

    @app.get("/v1/control/backlog")
    def backlog(_: dict[str, Any] = Depends(controller_principal)) -> list[dict[str, Any]]:
        return orchestrator.store.backlog()

    @app.post("/v1/control/reap-expired")
    def reap(_: dict[str, Any] = Depends(controller_principal)) -> dict[str, int]:
        return orchestrator.store.reap_expired_leases()

    @app.post("/v1/control/tokens", status_code=status.HTTP_201_CREATED)
    def issue_token(
        body: IssueTokenRequest,
        principal: dict[str, Any] = Depends(controller_principal),
    ) -> dict[str, str]:
        if principal["role"] != "admin":
            raise HTTPException(status_code=403, detail="only admin tokens can issue credentials")
        raw = orchestrator.store.issue_token(
            role=body.role,
            subject=body.subject,
            scopes=body.scopes,
            expires_at=body.expires_at,
            token_id=body.token_id,
        )
        return {"token": raw}

    @app.get("/v1/control/tokens")
    def list_tokens(
        principal: dict[str, Any] = Depends(controller_principal),
    ) -> list[dict[str, Any]]:
        if principal["role"] != "admin":
            raise HTTPException(status_code=403, detail="only admin tokens can list credentials")
        return orchestrator.store.list_tokens()

    @app.put("/v1/control/tokens/{token_id}")
    def update_token(
        token_id: str,
        body: UpdateTokenRequest,
        principal: dict[str, Any] = Depends(controller_principal),
    ) -> dict[str, Any]:
        if principal["role"] != "admin":
            raise HTTPException(status_code=403, detail="only admin tokens can edit credentials")
        return orchestrator.store.update_token(
            token_id,
            role=body.role,
            subject=body.subject,
            scopes=body.scopes,
            expires_at=body.expires_at,
        )

    @app.delete("/v1/control/tokens/{token_id}")
    def revoke_token(
        token_id: str,
        principal: dict[str, Any] = Depends(controller_principal),
    ) -> dict[str, Any]:
        if principal["role"] != "admin":
            raise HTTPException(status_code=403, detail="only admin tokens can revoke credentials")
        if not orchestrator.store.disable_token(token_id):
            existing = next(
                (row for row in orchestrator.store.list_tokens() if row["id"] == token_id),
                None,
            )
            if existing is None:
                raise NotFoundError(f"token {token_id!r} not found")
            return existing
        return next(
            row for row in orchestrator.store.list_tokens() if row["id"] == token_id
        )

    if deployment_service is not None:

        @app.get("/v1/control/capacity")
        def capacity(_: dict[str, Any] = Depends(controller_principal)) -> list[dict[str, Any]]:
            return deployment_service.capacity()

        @app.post("/v1/control/deployments", status_code=status.HTTP_201_CREATED)
        def deploy_workers(
            body: DeployRequest,
            principal: dict[str, Any] = Depends(admin_principal),
        ) -> list[dict[str, Any]]:
            return deployment_service.deploy(
                pool_id=body.pool_id,
                provider=body.provider,
                count=body.count,
                expires_hours=body.expires_hours,
                offer_id=body.offer_id,
                actor=principal["subject"],
            )

        @app.get("/v1/control/deployments")
        def list_deployments(
            _: dict[str, Any] = Depends(controller_principal),
        ) -> list[dict[str, Any]]:
            return deployment_service.list_deployments()

        @app.post("/v1/control/deployments/reconcile")
        def reconcile_deployments(
            _: dict[str, Any] = Depends(admin_principal),
        ) -> dict[str, int]:
            return deployment_service.reconcile()

        @app.delete("/v1/control/deployments/{deployment_id}")
        def teardown_deployment(
            deployment_id: str,
            principal: dict[str, Any] = Depends(admin_principal),
        ) -> dict[str, Any]:
            return deployment_service.teardown(deployment_id, actor=principal["subject"])

    if enable_dashboard:
        @app.get("/dashboard", include_in_schema=False)
        def dashboard_page() -> HTMLResponse:
            return HTMLResponse(dashboard_html())

        @app.get("/v1/dashboard/overview", include_in_schema=False)
        def dashboard_overview(
            principal: dict[str, Any] = Depends(dashboard_principal),
        ) -> dict[str, Any]:
            return build_overview(orchestrator, principal)

        @app.get("/v1/dashboard/queues", include_in_schema=False)
        def dashboard_queues(
            window_hours: int = Query(default=24, ge=1, le=168),
            principal: dict[str, Any] = Depends(dashboard_principal),
        ) -> list[dict[str, Any]]:
            return build_queues(
                orchestrator, principal, window_hours=window_hours
            )

        @app.get("/v1/dashboard/recent", include_in_schema=False)
        def dashboard_recent(
            limit: int = Query(default=50, ge=1, le=500),
            principal: dict[str, Any] = Depends(dashboard_principal),
        ) -> list[dict[str, Any]]:
            return build_recent(orchestrator, principal, limit=limit)

    @app.get("/v1/control/scaling-rules")
    def list_scaling_rules(
        pool_id: str | None = None,
        principal: dict[str, Any] = Depends(controller_principal),
    ) -> list[dict[str, Any]]:
        return orchestrator.store.list_scaling_rules(pool_id=pool_id)

    @app.post("/v1/control/scaling-rules", status_code=status.HTTP_201_CREATED)
    def create_scaling_rule(
        body: dict[str, Any],
        principal: dict[str, Any] = Depends(admin_principal),
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        draft = {
            **body,
            "id": "temp",
            "version": 1,
            "created_at": now,
            "updated_at": now,
            "actor": principal["subject"],
            "last_evaluation_at": None,
            "last_action_at": None,
        }
        rule = scaling_policy.load_rule(draft)
        return orchestrator.store.create_scaling_rule(
            name=rule.name,
            pool_id=rule.pool_id,
            mode=rule.mode,
            enabled=rule.enabled,
            trigger=rule.trigger.model_dump(mode="json"),
            scale_up=rule.scale_up.model_dump(mode="json"),
            providers=rule.providers.model_dump(mode="json"),
            budgets=rule.budgets.model_dump(mode="json"),
            scale_down=rule.scale_down.model_dump(mode="json"),
            actor=principal["subject"],
        )

    @app.get("/v1/control/scaling-rules/{rule_id}")
    def get_scaling_rule(
        rule_id: str,
        principal: dict[str, Any] = Depends(admin_principal),
    ) -> dict[str, Any]:
        return orchestrator.store.get_scaling_rule(rule_id)

    @app.put("/v1/control/scaling-rules/{rule_id}")
    def update_scaling_rule(
        rule_id: str,
        body: dict[str, Any],
        principal: dict[str, Any] = Depends(admin_principal),
    ) -> dict[str, Any]:
        existing = orchestrator.store.get_scaling_rule(rule_id)
        merged = {**existing, **{key: value for key, value in body.items() if key != "expected_version"}}
        merged["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        merged["actor"] = principal["subject"]
        rule = scaling_policy.load_rule(merged)
        return orchestrator.store.update_scaling_rule(
            rule_id,
            name=rule.name, mode=rule.mode, enabled=rule.enabled,
            trigger=rule.trigger.model_dump(mode="json"), scale_up=rule.scale_up.model_dump(mode="json"), providers=rule.providers.model_dump(mode="json"), budgets=rule.budgets.model_dump(mode="json"), scale_down=rule.scale_down.model_dump(mode="json"),
            actor=principal["subject"],
            expected_version=body.get("expected_version"),
        )

    @app.delete("/v1/control/scaling-rules/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_scaling_rule(
        rule_id: str,
        principal: dict[str, Any] = Depends(admin_principal),
    ) -> None:
        orchestrator.store.delete_scaling_rule(rule_id)

    @app.post("/v1/control/scaling-rules/{rule_id}/simulate")
    def simulate_scaling_rule(
        rule_id: str,
        principal: dict[str, Any] = Depends(controller_principal),
    ) -> dict[str, Any]:
        if app.state.scaling_engine is None:
            raise HTTPException(status_code=503, detail="scaling engine is not configured")
        return app.state.scaling_engine.evaluate(
            rule_id, dry_run=True, owner=principal["subject"]
        ).to_dict()

    @app.post("/v1/control/scaling-rules/{rule_id}/evaluate")
    def evaluate_scaling_rule(
        rule_id: str,
        principal: dict[str, Any] = Depends(admin_principal),
    ) -> dict[str, Any]:
        if app.state.scaling_engine is None:
            raise HTTPException(status_code=503, detail="scaling engine is not configured")
        return app.state.scaling_engine.evaluate(
            rule_id, dry_run=False, owner=principal["subject"]
        ).to_dict()

    @app.get("/v1/control/scaling-evaluations")
    def list_scaling_evaluations(
        rule_id: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
        principal: dict[str, Any] = Depends(controller_principal),
    ) -> list[dict[str, Any]]:
        return orchestrator.store.list_evaluations(rule_id=rule_id, limit=limit)

    @app.get("/v1/control/scaling-actions")
    def list_scaling_actions(
        rule_id: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
        principal: dict[str, Any] = Depends(controller_principal),
    ) -> list[dict[str, Any]]:
        return orchestrator.store.list_actions(rule_id=rule_id, limit=limit)

    @app.post("/v1/control/automation/pause")
    def pause_automation(
        principal: dict[str, Any] = Depends(admin_principal),
    ) -> dict[str, bool]:
        state = orchestrator.store.set_automation_paused(True, principal["subject"])
        app.state.automation_paused = True
        return state

    @app.post("/v1/control/automation/resume")
    def resume_automation(principal: dict[str, Any] = Depends(admin_principal)) -> dict[str, Any]:
        app.state.automation_paused = False
        return orchestrator.store.set_automation_paused(False, principal["subject"])

    @app.get("/v1/control/automation/state")
    def automation_state(principal: dict[str, Any] = Depends(controller_principal)) -> dict[str, Any]:
        return orchestrator.store.automation_state()

    return app
