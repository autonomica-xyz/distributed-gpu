"""SQLite persistence and atomic GPU-slot job leasing.

SQLite is used because a small control plane is easy to audit, back up, and
operate. ``BEGIN IMMEDIATE`` serializes the short claim transaction so two GPU
slots can never receive the same job.
"""

from __future__ import annotations

import fnmatch
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
import math
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from .models import AgentRegistration, JobState
from .registry import AdmittedJob


class StoreError(RuntimeError):
    pass


class NotFoundError(StoreError):
    pass


class ConflictError(StoreError):
    pass


class LeaseError(ConflictError):
    pass


UTC = timezone.utc
AGENT_STALE_SECONDS = 120.0


SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS agents (
    id                   TEXT PRIMARY KEY,
    provider             TEXT NOT NULL,
    provider_resource_id TEXT,
    hostname             TEXT NOT NULL,
    cpu_cores            REAL NOT NULL,
    ram_mb               INTEGER NOT NULL,
    disk_free_mb         INTEGER NOT NULL,
    labels_json          TEXT NOT NULL,
    state                TEXT NOT NULL,
    registered_at        TEXT NOT NULL,
    heartbeat_at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS agents_provider_resource
    ON agents(provider, provider_resource_id);

CREATE TABLE IF NOT EXISTS slots (
    id                   TEXT PRIMARY KEY,
    agent_id             TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    gpu_index            INTEGER NOT NULL,
    gpu_uuid             TEXT NOT NULL,
    vendor               TEXT NOT NULL,
    model                TEXT NOT NULL,
    vram_mb              INTEGER NOT NULL,
    compute_capability   TEXT,
    capabilities_json    TEXT NOT NULL,
    model_packs_json     TEXT NOT NULL,
    labels_json          TEXT NOT NULL,
    state                TEXT NOT NULL,
    active_job_id        TEXT,
    heartbeat_at         TEXT NOT NULL,
    UNIQUE(agent_id, gpu_uuid)
);

CREATE INDEX IF NOT EXISTS slots_schedulable
    ON slots(state, active_job_id, heartbeat_at);

CREATE TABLE IF NOT EXISTS jobs (
    id                         TEXT PRIMARY KEY,
    project_id                 TEXT NOT NULL,
    workload                   TEXT NOT NULL,
    workload_version           TEXT NOT NULL,
    route                      TEXT NOT NULL,
    payload_json               TEXT NOT NULL,
    inputs_json                TEXT NOT NULL,
    requirements_json          TEXT NOT NULL,
    executor_json              TEXT NOT NULL,
    input_contract_json        TEXT NOT NULL,
    output_contract_json       TEXT NOT NULL,
    retryable_error_codes_json TEXT NOT NULL,
    cost_unit                  TEXT NOT NULL,
    estimated_units            REAL NOT NULL,
    priority                   INTEGER NOT NULL,
    state                      TEXT NOT NULL,
    attempts                   INTEGER NOT NULL DEFAULT 0,
    max_attempts               INTEGER NOT NULL,
    worker_id                  TEXT,
    slot_id                    TEXT,
    lease_token_hash           TEXT,
    lease_expires_at           TEXT,
    heartbeat_at               TEXT,
    cancel_requested           INTEGER NOT NULL DEFAULT 0,
    outputs_json               TEXT NOT NULL DEFAULT '[]',
    error_code                 TEXT,
    error_message              TEXT,
    idempotency_key            TEXT,
    created_at                 TEXT NOT NULL,
    leased_at                  TEXT,
    started_at                 TEXT,
    finished_at                TEXT,
    UNIQUE(project_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS jobs_pending
    ON jobs(state, priority DESC, created_at)
    WHERE state = 'pending';
CREATE INDEX IF NOT EXISTS jobs_route_state
    ON jobs(workload, route, state);
CREATE INDEX IF NOT EXISTS jobs_active_lease
    ON jobs(state, lease_expires_at)
    WHERE state IN ('leased', 'running');
CREATE INDEX IF NOT EXISTS jobs_finished_route
    ON jobs(finished_at, workload, route)
    WHERE finished_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS job_attempts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id        TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    attempt       INTEGER NOT NULL,
    worker_id     TEXT NOT NULL,
    slot_id       TEXT NOT NULL,
    state         TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    duration_ms   INTEGER,
    error_code    TEXT,
    error_message TEXT,
    UNIQUE(job_id, attempt)
);
CREATE INDEX IF NOT EXISTS job_attempts_finished_state
    ON job_attempts(finished_at, state)
    WHERE finished_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS job_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id     TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    actor      TEXT,
    detail_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS job_events_job ON job_events(job_id, id);

CREATE TABLE IF NOT EXISTS auth_tokens (
    id          TEXT PRIMARY KEY,
    secret_hash TEXT NOT NULL,
    salt_hex    TEXT NOT NULL,
    role        TEXT NOT NULL,
    subject     TEXT NOT NULL,
    scopes_json TEXT NOT NULL,
    expires_at  TEXT,
    disabled    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    updated_at  TEXT,
    revoked_at  TEXT,
    last_used_at TEXT
);

CREATE INDEX IF NOT EXISTS auth_tokens_subject ON auth_tokens(role, subject, disabled);

CREATE TABLE IF NOT EXISTS deployments (
    id              TEXT PRIMARY KEY,
    pool_id         TEXT NOT NULL,
    provider        TEXT NOT NULL,
    resource_id     TEXT NOT NULL,
    resource_name   TEXT NOT NULL,
    worker_id       TEXT NOT NULL,
    token_id        TEXT NOT NULL,
    workload        TEXT NOT NULL,
    route           TEXT NOT NULL,
    capability      TEXT NOT NULL,
    image           TEXT NOT NULL,
    hourly_cost_usd REAL,
    state           TEXT NOT NULL,
    detail          TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    destroyed_at    TEXT
    ,offer_id TEXT, gpu_model TEXT, priority TEXT, region TEXT, price_source TEXT,
    availability_at_selection INTEGER, rule_id TEXT, action_id TEXT,
    provider_create_response TEXT, startup_deadline TEXT, first_heartbeat TEXT,
    first_completion TEXT, failure_classification TEXT
);

CREATE INDEX IF NOT EXISTS deployments_worker ON deployments(worker_id);
CREATE INDEX IF NOT EXISTS deployments_state ON deployments(state);

CREATE TABLE IF NOT EXISTS scaling_rules (
    id                   TEXT PRIMARY KEY,
    name                 TEXT NOT NULL,
    pool_id              TEXT NOT NULL,
    mode                 TEXT NOT NULL,
    enabled              INTEGER NOT NULL DEFAULT 0,
    version              INTEGER NOT NULL DEFAULT 1,
    trigger_json         TEXT NOT NULL,
    scale_up_json        TEXT NOT NULL,
    providers_json       TEXT NOT NULL,
    budgets_json         TEXT NOT NULL,
    scale_down_json      TEXT NOT NULL,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL,
    created_by           TEXT,
    updated_by           TEXT,
    last_evaluation_at   TEXT,
    last_action_at       TEXT,
    kill_switch_generation INTEGER NOT NULL DEFAULT 0
    ,threshold_state TEXT, threshold_since TEXT, deleted_at TEXT
);

CREATE INDEX IF NOT EXISTS scaling_rules_pool ON scaling_rules(pool_id);

CREATE TABLE IF NOT EXISTS scaling_evaluations (
    id                   TEXT PRIMARY KEY,
    rule_id              TEXT REFERENCES scaling_rules(id) ON DELETE SET NULL,
    rule_version         INTEGER NOT NULL,
    snapshot_json        TEXT NOT NULL,
    threshold_state      TEXT,
    current_slots        INTEGER NOT NULL DEFAULT 0,
    in_flight_slots      INTEGER NOT NULL DEFAULT 0,
    desired_slots        INTEGER NOT NULL DEFAULT 0,
    decision             TEXT NOT NULL,
    blocked_reason       TEXT,
    selected_candidate_json TEXT,
    alternatives_json    TEXT NOT NULL DEFAULT '[]',
    projected_eta_hours  REAL,
    projected_cost_usd   REAL,
    created_at           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS scaling_evaluations_rule ON scaling_evaluations(rule_id, created_at);

CREATE TABLE IF NOT EXISTS scaling_actions (
    id                   TEXT PRIMARY KEY,
    rule_id              TEXT REFERENCES scaling_rules(id) ON DELETE SET NULL,
    evaluation_id        TEXT REFERENCES scaling_evaluations(id),
    idempotency_key      TEXT NOT NULL UNIQUE,
    action               TEXT NOT NULL,
    state                TEXT NOT NULL,
    origin               TEXT,
    deployment_id        TEXT,
    token_id             TEXT,
    reservation_json     TEXT DEFAULT '{}',
    provider_request_json TEXT,
    provider_result_json TEXT,
    trace_id             TEXT,
    fallback_lineage_json TEXT DEFAULT '[]',
    created_at           TEXT NOT NULL,
    started_at           TEXT,
    finished_at          TEXT
);

CREATE INDEX IF NOT EXISTS scaling_actions_rule ON scaling_actions(rule_id, created_at);

CREATE TABLE IF NOT EXISTS scaling_observations (
    id                   TEXT PRIMARY KEY,
    kind                 TEXT NOT NULL,
    provider             TEXT,
    offer_id             TEXT,
    gpu_model            TEXT,
    route                TEXT,
    value                REAL,
    count                INTEGER DEFAULT 1,
    success_count        INTEGER DEFAULT 0,
    observed_at          TEXT NOT NULL,
    expires_at           TEXT
);

CREATE INDEX IF NOT EXISTS scaling_observations_lookup ON scaling_observations(kind, provider, offer_id, route, observed_at);

CREATE TABLE IF NOT EXISTS budget_reservations (
    id                   TEXT PRIMARY KEY,
    rule_id              TEXT,
    provider             TEXT,
    amount_usd           REAL NOT NULL,
    reserved_at          TEXT NOT NULL,
    expires_at           TEXT NOT NULL,
    reconciled_at        TEXT
);

CREATE INDEX IF NOT EXISTS budget_reservations_rule ON budget_reservations(rule_id);
CREATE INDEX IF NOT EXISTS budget_reservations_provider ON budget_reservations(provider);

CREATE TABLE IF NOT EXISTS automation_state (
    id INTEGER PRIMARY KEY CHECK (id = 1), paused INTEGER NOT NULL DEFAULT 0,
    generation INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL, updated_by TEXT
);

CREATE TABLE IF NOT EXISTS scaling_leases (
    rule_id              TEXT PRIMARY KEY REFERENCES scaling_rules(id) ON DELETE CASCADE,
    owner                TEXT NOT NULL,
    acquired_at          TEXT NOT NULL,
    expires_at           TEXT NOT NULL
);

-- Durable grant records for client-uploaded job input blobs. The blob bytes
-- live in the content-addressed artifact store; this table is the
-- authorization record that binds a blob digest to one project and records
-- the workload contract the upload was validated against. There is no delete:
-- blobs are immutable, so the grant stays valid for replayed submissions.
CREATE TABLE IF NOT EXISTS inputs (
    project_id TEXT NOT NULL,
    sha256     TEXT NOT NULL,
    name       TEXT NOT NULL,
    media_type TEXT NOT NULL,
    size       INTEGER NOT NULL,
    workload   TEXT NOT NULL,
    uploader   TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (project_id, sha256)
);

CREATE INDEX IF NOT EXISTS inputs_digest ON inputs(sha256);
"""


INPUT_URI_RE = re.compile(
    r"^/v1/projects/(?P<project>[a-z][a-z0-9_-]{1,62})/inputs/(?P<sha256>[0-9a-f]{64})$"
)
INPUT_ENTRY_KEYS = {"name", "uri", "sha256", "size", "media_type"}


JSON_COLUMNS = {
    "payload_json": "payload",
    "inputs_json": "inputs",
    "requirements_json": "requirements",
    "executor_json": "executor",
    "input_contract_json": "input_contract",
    "output_contract_json": "output_contract",
    "retryable_error_codes_json": "retryable_error_codes",
    "outputs_json": "outputs",
}


def _json(value: Any) -> str:
    def safe(item: Any) -> Any:
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {str(key): safe(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(value) for value in item]
        return item
    return json.dumps(safe(value), separators=(",", ":"), sort_keys=True, allow_nan=False)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _new_id(prefix: str) -> str:
    # Time prefix keeps ordinary SQLite listings approximately chronological.
    return f"{prefix}_{time.time_ns():x}_{secrets.token_hex(6)}"


def _lease_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _version_tuple(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    try:
        return tuple(int(part) for part in value.split("."))
    except ValueError:
        return ()


class Store:
    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], datetime] = _utcnow,
    ):
        self.path = str(path)
        self.clock = clock
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._memory_connection: sqlite3.Connection | None = None
        if self.path == ":memory:":
            # A keeper connection is required because each operation uses its
            # own connection and a normal :memory: DB is connection-local.
            self.path = f"file:sparehour-{secrets.token_hex(8)}?mode=memory&cache=shared"
            self._memory_connection = self._connect()
        with self._connection() as conn:
            conn.executescript(SCHEMA)
            # Additive migration for databases created before token lifecycle
            # timestamps were recorded. Revoked rows remain durable audit data.
            token_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(auth_tokens)")
            }
            if "updated_at" not in token_columns:
                conn.execute("ALTER TABLE auth_tokens ADD COLUMN updated_at TEXT")
            if "revoked_at" not in token_columns:
                conn.execute("ALTER TABLE auth_tokens ADD COLUMN revoked_at TEXT")
            if "last_used_at" not in token_columns:
                conn.execute("ALTER TABLE auth_tokens ADD COLUMN last_used_at TEXT")
            additions = {
                "deployments": {"offer_id": "TEXT", "gpu_model": "TEXT", "priority": "TEXT", "region": "TEXT", "price_source": "TEXT", "availability_at_selection": "INTEGER", "rule_id": "TEXT", "action_id": "TEXT", "provider_create_response": "TEXT", "startup_deadline": "TEXT", "first_heartbeat": "TEXT", "first_completion": "TEXT", "failure_classification": "TEXT"},
                "scaling_rules": {"threshold_state": "TEXT", "threshold_since": "TEXT", "deleted_at": "TEXT"},
            }
            for table, columns in additions.items():
                existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
                for name, definition in columns.items():
                    if name not in existing:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
            budget_columns = {row["name"] for row in conn.execute("PRAGMA table_info(budget_reservations)")}
            if "provider" not in budget_columns:
                conn.execute("ALTER TABLE budget_reservations ADD COLUMN provider TEXT")
            auto_columns = {row["name"] for row in conn.execute("PRAGMA table_info(automation_state)")}
            if "generation" not in auto_columns:
                conn.execute("ALTER TABLE automation_state ADD COLUMN generation INTEGER NOT NULL DEFAULT 0")
            conn.execute("INSERT OR IGNORE INTO automation_state(id, paused, generation, updated_at, updated_by) VALUES (1, 0, 0, ?, 'system')", (_iso(self.clock()),))

    def close(self) -> None:
        if self._memory_connection is not None:
            self._memory_connection.close()
            self._memory_connection = None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path,
            timeout=5.0,
            isolation_level=None,
            uri=self.path.startswith("file:"),
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except Exception:
                conn.rollback()
                raise
            else:
                conn.commit()

    def submit_job(
        self,
        admitted: AdmittedJob,
        *,
        project_id: str,
        priority: int = 0,
        idempotency_key: str | None = None,
        inputs: Sequence[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        now = _iso(self.clock())
        resolved_inputs = self._resolve_inputs(project_id, inputs or ())
        with self._transaction() as conn:
            if idempotency_key:
                existing = conn.execute(
                    "SELECT * FROM jobs WHERE project_id = ? AND idempotency_key = ?",
                    (project_id, idempotency_key),
                ).fetchone()
                if existing is not None:
                    same_request = (
                        existing["workload"] == admitted.workload
                        and existing["route"] == admitted.route
                        and existing["payload_json"] == _json(admitted.payload)
                        and existing["inputs_json"] == _json(resolved_inputs)
                        and int(existing["priority"]) == int(priority)
                    )
                    if not same_request:
                        raise ConflictError(
                            "idempotency key was already used for a different request"
                        )
                    return self._job_doc(existing)

            job_id = _new_id("job")
            conn.execute(
                """
                INSERT INTO jobs (
                    id, project_id, workload, workload_version, route,
                    payload_json, inputs_json, requirements_json, executor_json,
                    input_contract_json, output_contract_json,
                    retryable_error_codes_json, cost_unit, estimated_units,
                    priority, state, max_attempts, idempotency_key, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    project_id,
                    admitted.workload,
                    admitted.workload_version,
                    admitted.route,
                    _json(admitted.payload),
                    _json(resolved_inputs),
                    _json(admitted.requirements),
                    _json(admitted.executor),
                    _json(admitted.input_contract),
                    _json(admitted.output_contract),
                    _json(sorted(admitted.retryable_error_codes)),
                    admitted.cost_unit,
                    admitted.estimated_units,
                    int(priority),
                    JobState.PENDING.value,
                    admitted.max_attempts,
                    idempotency_key,
                    now,
                ),
            )
            self._event(conn, job_id, "submitted", project_id, {}, now)
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            assert row is not None
            return self._job_doc(row)

    # ---- project-scoped input blobs ----

    def put_input(
        self,
        *,
        project_id: str,
        name: str,
        sha256: str,
        size: int,
        media_type: str,
        workload: str,
        uploader: str,
    ) -> dict[str, Any]:
        """Record a durable grant for an uploaded input blob.

        Grants are immutable and content-keyed: re-uploading identical bytes
        under a different name keeps the first record, which is safe because
        only ``(project_id, sha256)`` authorizes later use.
        """
        now = _iso(self.clock())
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO inputs (
                    project_id, sha256, name, media_type, size, workload,
                    uploader, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    sha256,
                    name,
                    media_type,
                    int(size),
                    workload,
                    uploader,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM inputs WHERE project_id = ? AND sha256 = ?",
                (project_id, sha256),
            ).fetchone()
            assert row is not None
            return dict(row)

    def get_input(self, project_id: str, sha256: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM inputs WHERE project_id = ? AND sha256 = ?",
                (project_id, sha256),
            ).fetchone()
        return dict(row) if row is not None else None

    def _resolve_inputs(
        self, project_id: str, inputs: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Bind declared job inputs to project-owned blobs, or reject them.

        A job may only reference blobs previously uploaded through
        ``/v1/projects/{project}/inputs``; external ``http``/``file`` URIs or
        digests outside the project's grants are forged and fail submission.
        The stored document is canonicalized to the controller's URI and the
        grant's verified size/digest/media type.
        """
        resolved: list[dict[str, Any]] = []
        for index, item in enumerate(inputs):
            if not isinstance(item, dict):
                raise ValueError(f"inputs[{index}] must be an object")
            unknown = set(item) - INPUT_ENTRY_KEYS
            if unknown:
                raise ValueError(
                    f"inputs[{index}] has unknown fields: {', '.join(sorted(unknown))}"
                )
            name = item.get("name")
            uri = item.get("uri")
            sha256 = item.get("sha256")
            if not isinstance(name, str) or not name:
                raise ValueError(f"inputs[{index}].name must be a non-empty string")
            if not isinstance(uri, str):
                raise ValueError(f"inputs[{index}].uri must be a string")
            match = INPUT_URI_RE.fullmatch(uri)
            if match is None:
                raise ValueError(
                    f"inputs[{index}].uri must be an uploaded controller input "
                    "(/v1/projects/<project>/inputs/<sha256>); external or file "
                    "URIs are not allowed"
                )
            if match.group("project") != project_id:
                raise ValueError(
                    f"inputs[{index}] references a blob owned by another project"
                )
            # The URI carries the digest; a supplied sha256 must agree with it
            # (forged mismatches fail), and a missing one is derived.
            if sha256 is not None and (
                not isinstance(sha256, str) or sha256 != match.group("sha256")
            ):
                raise ValueError(
                    f"inputs[{index}].sha256 must match the digest in uri"
                )
            sha256 = match.group("sha256")
            declared_size = item.get("size")
            if declared_size is not None and (
                isinstance(declared_size, bool)
                or not isinstance(declared_size, int)
                or declared_size < 0
            ):
                raise ValueError(f"inputs[{index}].size must be a non-negative integer")
            grant = self.get_input(project_id, sha256)
            if grant is None:
                raise ValueError(
                    f"inputs[{index}] references an unknown input blob {sha256[:16]}…"
                )
            if declared_size is not None and declared_size != int(grant["size"]):
                raise ValueError(
                    f"inputs[{index}].size does not match the uploaded blob"
                )
            media_type = item.get("media_type")
            if media_type is not None and media_type != grant["media_type"]:
                raise ValueError(
                    f"inputs[{index}].media_type does not match the uploaded blob"
                )
            resolved.append(
                {
                    "name": name,
                    "uri": uri,
                    "sha256": sha256,
                    "size": int(grant["size"]),
                    "media_type": grant["media_type"],
                }
            )
        return resolved

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"job {job_id!r} not found")
        return self._job_doc(row)

    def list_jobs(
        self,
        *,
        project_id: str | None = None,
        workload: str | None = None,
        route: str | None = None,
        state: str | None = None,
        worker_id: str | None = None,
        slot_id: str | None = None,
        query: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        args: list[Any] = []
        for column, value in (
            ("project_id", project_id),
            ("workload", workload),
            ("route", route),
            ("state", state),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                args.append(value)
        if worker_id is not None:
            clauses.append(
                "(jobs.worker_id = ? OR EXISTS (SELECT 1 FROM job_attempts ja "
                "WHERE ja.job_id = jobs.id AND ja.worker_id = ?))"
            )
            args.extend((worker_id, worker_id))
        if slot_id is not None:
            clauses.append(
                "(jobs.slot_id = ? OR EXISTS (SELECT 1 FROM job_attempts ja "
                "WHERE ja.job_id = jobs.id AND ja.slot_id = ?))"
            )
            args.extend((slot_id, slot_id))
        if query:
            pattern = f"%{query}%"
            clauses.append(
                "(jobs.id LIKE ? OR jobs.project_id LIKE ? OR jobs.payload_json LIKE ?)"
            )
            args.extend((pattern, pattern, pattern))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        args.extend((max(1, min(int(limit), 1000)), max(0, int(offset))))
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM jobs {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                args,
            ).fetchall()
        return [self._job_doc(row) for row in rows]

    def job_attempts(self, job_id: str) -> list[dict[str, Any]]:
        # Verify the job separately so an empty list is not confused with 404.
        self.get_job(job_id)
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM job_attempts WHERE job_id = ? ORDER BY attempt", (job_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def job_events(self, job_id: str) -> list[dict[str, Any]]:
        self.get_job(job_id)
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM job_events WHERE job_id = ? ORDER BY id", (job_id,)
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["detail"] = json.loads(item.pop("detail_json"))
            result.append(item)
        return result

    def register_agent(self, registration: AgentRegistration) -> dict[str, Any]:
        now = _iso(self.clock())
        with self._transaction() as conn:
            incoming_ids = {slot.id for slot in registration.slots}
            if incoming_ids:
                placeholders = ",".join("?" for _ in incoming_ids)
                owners = conn.execute(
                    f"SELECT id, agent_id FROM slots WHERE id IN ({placeholders})",
                    tuple(incoming_ids),
                ).fetchall()
                stolen = [
                    row["id"]
                    for row in owners
                    if row["agent_id"] != registration.agent_id
                ]
                if stolen:
                    raise ConflictError(
                        f"slot IDs are already owned by another agent: {', '.join(sorted(stolen))}"
                    )
            existing = conn.execute(
                "SELECT registered_at FROM agents WHERE id = ?", (registration.agent_id,)
            ).fetchone()
            registered_at = existing["registered_at"] if existing else now
            conn.execute(
                """
                INSERT INTO agents (
                    id, provider, provider_resource_id, hostname, cpu_cores,
                    ram_mb, disk_free_mb, labels_json, state, registered_at, heartbeat_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    provider = excluded.provider,
                    provider_resource_id = excluded.provider_resource_id,
                    hostname = excluded.hostname,
                    cpu_cores = excluded.cpu_cores,
                    ram_mb = excluded.ram_mb,
                    disk_free_mb = excluded.disk_free_mb,
                    labels_json = excluded.labels_json,
                    state = CASE WHEN agents.state = 'draining' THEN agents.state ELSE 'ready' END,
                    heartbeat_at = excluded.heartbeat_at
                """,
                (
                    registration.agent_id,
                    registration.provider,
                    registration.provider_resource_id,
                    registration.hostname,
                    registration.cpu_cores,
                    registration.ram_mb,
                    registration.disk_free_mb,
                    _json(registration.labels),
                    registered_at,
                    now,
                ),
            )

            existing_slots = conn.execute(
                "SELECT id, active_job_id FROM slots WHERE agent_id = ?", (registration.agent_id,)
            ).fetchall()
            for row in existing_slots:
                if row["id"] not in incoming_ids and row["active_job_id"] is None:
                    conn.execute("UPDATE slots SET state = 'offline' WHERE id = ?", (row["id"],))

            for slot in registration.slots:
                conn.execute(
                    """
                    INSERT INTO slots (
                        id, agent_id, gpu_index, gpu_uuid, vendor, model, vram_mb,
                        compute_capability, capabilities_json, model_packs_json,
                        labels_json, state, heartbeat_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?)
                    ON CONFLICT(id) DO UPDATE SET
                        agent_id = excluded.agent_id,
                        gpu_index = excluded.gpu_index,
                        gpu_uuid = excluded.gpu_uuid,
                        vendor = excluded.vendor,
                        model = excluded.model,
                        vram_mb = excluded.vram_mb,
                        compute_capability = excluded.compute_capability,
                        capabilities_json = excluded.capabilities_json,
                        model_packs_json = excluded.model_packs_json,
                        labels_json = excluded.labels_json,
                        state = CASE
                            WHEN slots.active_job_id IS NOT NULL THEN 'busy'
                            ELSE 'ready'
                        END,
                        heartbeat_at = excluded.heartbeat_at
                    """,
                    (
                        slot.id,
                        registration.agent_id,
                        slot.gpu_index,
                        slot.gpu_uuid,
                        slot.vendor,
                        slot.model,
                        slot.vram_mb,
                        slot.compute_capability,
                        _json(slot.capabilities),
                        _json(slot.model_packs),
                        _json(slot.labels),
                        now,
                    ),
                )
        return self.get_agent(registration.agent_id)

    def heartbeat_agent(
        self,
        agent_id: str,
        *,
        disk_free_mb: int | None = None,
        draining: bool | None = None,
    ) -> dict[str, Any]:
        now = _iso(self.clock())
        assignments = ["heartbeat_at = ?"]
        args: list[Any] = [now]
        if disk_free_mb is not None:
            assignments.append("disk_free_mb = ?")
            args.append(max(0, int(disk_free_mb)))
        if draining is not None:
            assignments.append("state = ?")
            args.append("draining" if draining else "ready")
        args.append(agent_id)
        with self._transaction() as conn:
            cursor = conn.execute(
                f"UPDATE agents SET {', '.join(assignments)} WHERE id = ?", args
            )
            if cursor.rowcount == 0:
                raise NotFoundError(f"agent {agent_id!r} not found")
            conn.execute(
                "UPDATE slots SET heartbeat_at = ? WHERE agent_id = ?", (now, agent_id)
            )
        return self.get_agent(agent_id)

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"agent {agent_id!r} not found")
            slots = conn.execute(
                "SELECT * FROM slots WHERE agent_id = ? ORDER BY gpu_index", (agent_id,)
            ).fetchall()
        return self._agent_doc(row, slots)

    def get_slot(self, slot_id: str) -> dict[str, Any]:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM slots WHERE id = ?", (slot_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"slot {slot_id!r} not found")
        document = dict(row)
        document["capabilities"] = json.loads(document.pop("capabilities_json"))
        document["model_packs"] = json.loads(document.pop("model_packs_json"))
        document["labels"] = json.loads(document.pop("labels_json"))
        return document

    def list_agents(self, *, provider: str | None = None) -> list[dict[str, Any]]:
        with self._connection() as conn:
            if provider:
                rows = conn.execute(
                    "SELECT * FROM agents WHERE provider = ? ORDER BY id", (provider,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM agents ORDER BY id").fetchall()
            result = []
            for row in rows:
                slots = conn.execute(
                    "SELECT * FROM slots WHERE agent_id = ? ORDER BY gpu_index", (row["id"],)
                ).fetchall()
                result.append(self._agent_doc(row, slots))
        return result

    def claim_next(
        self,
        slot_id: str,
        *,
        lease_seconds: int = 90,
        scan_limit: int = 500,
    ) -> dict[str, Any] | None:
        if lease_seconds < 10:
            raise ValueError("lease_seconds must be at least 10")
        now_dt = self.clock()
        now = _iso(now_dt)
        expires = _iso(now_dt + timedelta(seconds=lease_seconds))
        with self._transaction() as conn:
            slot = conn.execute(
                """
                SELECT s.*, a.provider, a.cpu_cores, a.ram_mb, a.disk_free_mb,
                       a.labels_json AS agent_labels_json, a.state AS agent_state,
                       a.id AS worker_id
                  FROM slots s JOIN agents a ON a.id = s.agent_id
                 WHERE s.id = ?
                """,
                (slot_id,),
            ).fetchone()
            if slot is None:
                raise NotFoundError(f"slot {slot_id!r} not found")
            if slot["agent_state"] != "ready":
                return None
            if slot["state"] != "ready" or slot["active_job_id"] is not None:
                raise ConflictError(f"slot {slot_id!r} is not available")
            heartbeat_at = slot["heartbeat_at"]
            if heartbeat_at:
                try:
                    hb = datetime.fromisoformat(str(heartbeat_at))
                    if hb.tzinfo is None:
                        hb = hb.replace(tzinfo=UTC)
                    if (now_dt - hb).total_seconds() > AGENT_STALE_SECONDS:
                        return None
                except (TypeError, ValueError):
                    pass

            candidates = conn.execute(
                """
                SELECT * FROM jobs
                 WHERE state = 'pending' AND cancel_requested = 0
                 ORDER BY priority DESC, created_at ASC
                 LIMIT ?
                """,
                (max(1, min(scan_limit, 5000)),),
            ).fetchall()
            selected = next(
                (row for row in candidates if self._slot_matches(slot, json.loads(row["requirements_json"]))),
                None,
            )
            if selected is None:
                return None

            token = secrets.token_urlsafe(32)
            conn.execute(
                """
                UPDATE jobs
                   SET state = 'leased', worker_id = ?, slot_id = ?,
                       lease_token_hash = ?, lease_expires_at = ?,
                       heartbeat_at = ?, leased_at = ?, error_code = NULL,
                       error_message = NULL
                 WHERE id = ? AND state = 'pending'
                """,
                (
                    slot["worker_id"],
                    slot_id,
                    _lease_hash(token),
                    expires,
                    now,
                    now,
                    selected["id"],
                ),
            )
            conn.execute(
                "UPDATE slots SET state = 'busy', active_job_id = ?, heartbeat_at = ? WHERE id = ?",
                (selected["id"], now, slot_id),
            )
            self._event(
                conn,
                selected["id"],
                "leased",
                slot["worker_id"],
                {"slot_id": slot_id, "lease_expires_at": expires},
                now,
            )
            claimed = conn.execute("SELECT * FROM jobs WHERE id = ?", (selected["id"],)).fetchone()
            assert claimed is not None
            document = self._job_doc(claimed)
            document["lease_token"] = token
            return document

    def confirm(self, job_id: str, slot_id: str, lease_token: str) -> dict[str, Any]:
        now = _iso(self.clock())
        with self._transaction() as conn:
            row = self._held_job(conn, job_id, slot_id, lease_token, {JobState.LEASED.value})
            attempt = int(row["attempts"]) + 1
            conn.execute(
                """
                UPDATE jobs SET state = 'running', attempts = ?, started_at = ?, heartbeat_at = ?
                 WHERE id = ?
                """,
                (attempt, now, now, job_id),
            )
            conn.execute(
                """
                INSERT INTO job_attempts (
                    job_id, attempt, worker_id, slot_id, state, started_at
                ) VALUES (?, ?, ?, ?, 'running', ?)
                """,
                (job_id, attempt, row["worker_id"], slot_id, now),
            )
            self._event(conn, job_id, "running", row["worker_id"], {"attempt": attempt}, now)
            updated = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            assert updated is not None
            return self._job_doc(updated)

    def heartbeat(
        self,
        job_id: str,
        slot_id: str,
        lease_token: str,
        *,
        lease_seconds: int = 90,
    ) -> bool:
        now_dt = self.clock()
        now = _iso(now_dt)
        expires = _iso(now_dt + timedelta(seconds=lease_seconds))
        with self._transaction() as conn:
            self._held_job(
                conn,
                job_id,
                slot_id,
                lease_token,
                {JobState.LEASED.value, JobState.RUNNING.value},
            )
            conn.execute(
                "UPDATE jobs SET heartbeat_at = ?, lease_expires_at = ? WHERE id = ?",
                (now, expires, job_id),
            )
            conn.execute("UPDATE slots SET heartbeat_at = ? WHERE id = ?", (now, slot_id))
            conn.execute(
                """
                UPDATE agents SET heartbeat_at = ?
                 WHERE id = (SELECT agent_id FROM slots WHERE id = ?)
                """,
                (now, slot_id),
            )
            cancel = conn.execute(
                "SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            assert cancel is not None
            return bool(cancel["cancel_requested"])

    def verify_lease(
        self,
        job_id: str,
        slot_id: str,
        lease_token: str,
        *,
        allowed_states: set[str] | None = None,
    ) -> dict[str, Any]:
        with self._connection() as conn:
            row = self._held_job(
                conn,
                job_id,
                slot_id,
                lease_token,
                allowed_states or {JobState.LEASED.value, JobState.RUNNING.value},
            )
            return self._job_doc(row)

    def complete(
        self,
        job_id: str,
        slot_id: str,
        lease_token: str,
        outputs: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        now_dt = self.clock()
        now = _iso(now_dt)
        with self._transaction() as conn:
            row = self._held_job(conn, job_id, slot_id, lease_token, {JobState.RUNNING.value})
            if row["cancel_requested"]:
                raise ConflictError("job cancellation has been requested")
            conn.execute(
                """
                UPDATE jobs
                   SET state = 'succeeded', outputs_json = ?, finished_at = ?,
                       lease_token_hash = NULL, lease_expires_at = NULL,
                       cancel_requested = 0, error_code = NULL, error_message = NULL
                 WHERE id = ?
                """,
                (_json(list(outputs)), now, job_id),
            )
            self._finish_attempt(conn, row, "succeeded", now, None, None, now_dt)
            self._free_slot(conn, slot_id, job_id, now)
            self._event(conn, job_id, "succeeded", row["worker_id"], {}, now)
            updated = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            assert updated is not None
            return self._job_doc(updated)

    def fail(
        self,
        job_id: str,
        slot_id: str,
        lease_token: str,
        *,
        error_code: str,
        error_message: str,
        retryable: bool,
    ) -> dict[str, Any]:
        now_dt = self.clock()
        now = _iso(now_dt)
        with self._transaction() as conn:
            row = self._held_job(
                conn,
                job_id,
                slot_id,
                lease_token,
                {JobState.LEASED.value, JobState.RUNNING.value},
            )
            attempts = int(row["attempts"])
            if row["state"] == JobState.LEASED.value:
                attempts += 1
                conn.execute("UPDATE jobs SET attempts = ?, started_at = ? WHERE id = ?", (attempts, now, job_id))
                conn.execute(
                    """
                    INSERT INTO job_attempts (
                        job_id, attempt, worker_id, slot_id, state, started_at
                    ) VALUES (?, ?, ?, ?, 'running', ?)
                    """,
                    (job_id, attempts, row["worker_id"], slot_id, now),
                )
                row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                assert row is not None

            if row["cancel_requested"]:
                next_state = JobState.CANCELLED.value
            elif retryable and attempts < int(row["max_attempts"]):
                next_state = JobState.PENDING.value
            else:
                next_state = JobState.DEAD_LETTER.value
            finished_at = now if next_state != JobState.PENDING.value else None
            conn.execute(
                """
                UPDATE jobs
                   SET state = ?, worker_id = NULL, slot_id = NULL,
                       lease_token_hash = NULL, lease_expires_at = NULL,
                       heartbeat_at = NULL, leased_at = NULL,
                       cancel_requested = 0, error_code = ?, error_message = ?,
                       finished_at = ?
                 WHERE id = ?
                """,
                (next_state, error_code, error_message[:4000], finished_at, job_id),
            )
            self._finish_attempt(
                conn, row, next_state if next_state != JobState.PENDING.value else "failed", now,
                error_code, error_message[:4000], now_dt,
            )
            self._free_slot(conn, slot_id, job_id, now)
            self._event(
                conn,
                job_id,
                "failed" if next_state == JobState.PENDING.value else next_state,
                row["worker_id"],
                {"error_code": error_code, "retryable": retryable, "next_state": next_state},
                now,
            )
            updated = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            assert updated is not None
            return self._job_doc(updated)

    def release(
        self,
        job_id: str,
        slot_id: str,
        lease_token: str,
        *,
        reason: str,
    ) -> dict[str, Any]:
        now = _iso(self.clock())
        with self._transaction() as conn:
            row = self._held_job(conn, job_id, slot_id, lease_token, {JobState.LEASED.value})
            conn.execute(
                """
                UPDATE jobs
                   SET state = 'pending', worker_id = NULL, slot_id = NULL,
                       lease_token_hash = NULL, lease_expires_at = NULL,
                       heartbeat_at = NULL, leased_at = NULL
                 WHERE id = ?
                """,
                (job_id,),
            )
            self._free_slot(conn, slot_id, job_id, now)
            self._event(conn, job_id, "released", row["worker_id"], {"reason": reason}, now)
            updated = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            assert updated is not None
            return self._job_doc(updated)

    def cancel(self, job_id: str, *, actor: str = "client") -> dict[str, Any]:
        now = _iso(self.clock())
        with self._transaction() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"job {job_id!r} not found")
            if row["state"] == JobState.PENDING.value:
                conn.execute(
                    "UPDATE jobs SET state = 'cancelled', finished_at = ? WHERE id = ?",
                    (now, job_id),
                )
                event = "cancelled"
            elif row["state"] in {JobState.LEASED.value, JobState.RUNNING.value}:
                conn.execute("UPDATE jobs SET cancel_requested = 1 WHERE id = ?", (job_id,))
                event = "cancel-requested"
            else:
                raise ConflictError(f"job is already terminal: {row['state']}")
            self._event(conn, job_id, event, actor, {}, now)
            updated = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            assert updated is not None
            return self._job_doc(updated)

    def reap_expired_leases(self) -> dict[str, int]:
        now_dt = self.clock()
        now = _iso(now_dt)
        counts = {"released": 0, "retried": 0, "dead_lettered": 0}
        with self._transaction() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs
                 WHERE state IN ('leased', 'running')
                   AND lease_expires_at IS NOT NULL
                   AND lease_expires_at < ?
                 ORDER BY lease_expires_at
                """,
                (now,),
            ).fetchall()
            for row in rows:
                if row["state"] == JobState.LEASED.value:
                    next_state = JobState.PENDING.value
                    counts["released"] += 1
                elif int(row["attempts"]) < int(row["max_attempts"]):
                    next_state = JobState.PENDING.value
                    counts["retried"] += 1
                    self._finish_attempt(
                        conn, row, "stale-requeued", now, "worker_lost",
                        "worker lease expired", now_dt,
                    )
                else:
                    next_state = JobState.DEAD_LETTER.value
                    counts["dead_lettered"] += 1
                    self._finish_attempt(
                        conn, row, JobState.DEAD_LETTER.value, now, "worker_lost",
                        "worker lease expired", now_dt,
                    )
                finished_at = now if next_state == JobState.DEAD_LETTER.value else None
                conn.execute(
                    """
                    UPDATE jobs
                       SET state = ?, worker_id = NULL, slot_id = NULL,
                           lease_token_hash = NULL, lease_expires_at = NULL,
                           heartbeat_at = NULL, leased_at = NULL,
                           error_code = 'worker_lost', error_message = 'worker lease expired',
                           finished_at = ?
                     WHERE id = ?
                    """,
                    (next_state, finished_at, row["id"]),
                )
                if row["slot_id"]:
                    self._free_slot(conn, row["slot_id"], row["id"], now)
                self._event(
                    conn,
                    row["id"],
                    "lease-expired",
                    "scheduler",
                    {"previous_state": row["state"], "next_state": next_state},
                    now,
                )
        return counts

    def backlog(self) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT workload, route, cost_unit, state,
                       COUNT(*) AS jobs, COALESCE(SUM(estimated_units), 0) AS units
                  FROM jobs
                 WHERE state IN ('pending', 'leased', 'running')
                 GROUP BY workload, route, cost_unit, state
                 ORDER BY workload, route, state
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def pending_counts(self) -> dict[tuple[str, str], int]:
        """Pending job count per (workload, route) — the queue-depth signal."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT workload, route, COUNT(*) AS pending
                  FROM jobs
                 WHERE state = 'pending' AND cancel_requested = 0
                 GROUP BY workload, route
                """
            ).fetchall()
        return {(row["workload"], row["route"]): int(row["pending"]) for row in rows}

    def queue_stats(
        self, *, project_id: str | None = None, window_hours: int = 24
    ) -> list[dict[str, Any]]:
        """Aggregate route queues, observed service rates, and worker activity."""
        cutoff = _iso(self.clock() - timedelta(hours=max(1, min(window_hours, 168))))
        scope = "WHERE project_id = ?" if project_id is not None else ""
        scoped_args: list[Any] = [project_id] if project_id is not None else []
        recent_scope = "AND j.project_id = ?" if project_id is not None else ""
        with self._connection() as conn:
            totals = conn.execute(
                f"""
                SELECT workload, route, cost_unit,
                       SUM(CASE WHEN state = 'pending' THEN 1 ELSE 0 END) pending_jobs,
                       SUM(CASE WHEN state = 'leased' THEN 1 ELSE 0 END) leased_jobs,
                       SUM(CASE WHEN state = 'running' THEN 1 ELSE 0 END) running_jobs,
                       SUM(CASE WHEN state = 'succeeded' THEN 1 ELSE 0 END) succeeded_jobs,
                       SUM(CASE WHEN state = 'dead-letter' THEN 1 ELSE 0 END) dead_letter_jobs,
                       COALESCE(SUM(CASE WHEN state IN ('pending','leased','running')
                            THEN estimated_units ELSE 0 END), 0) backlog_units
                  FROM jobs {scope}
                 GROUP BY workload, route, cost_unit
                """,
                scoped_args,
            ).fetchall()
            recent = conn.execute(
                f"""
                SELECT workload, route,
                       SUM(CASE WHEN state = 'succeeded' THEN 1 ELSE 0 END) completed_jobs,
                       SUM(CASE WHEN state = 'dead-letter' THEN 1 ELSE 0 END) failed_jobs,
                       COALESCE(SUM(CASE WHEN state = 'succeeded' THEN estimated_units ELSE 0 END),0) completed_units,
                       COALESCE(SUM(CASE WHEN state = 'succeeded' AND started_at IS NOT NULL
                            THEN (julianday(finished_at)-julianday(started_at))*86400000 ELSE 0 END),0) busy_ms
                  FROM jobs j
                 WHERE finished_at >= ? {recent_scope}
                 GROUP BY workload, route
                """,
                [cutoff, *scoped_args],
            ).fetchall()
            workers = conn.execute(
                f"""
                SELECT j.workload, j.route, a.worker_id, a.slot_id,
                       COALESCE(s.model, a.slot_id) model,
                       COALESCE(g.provider, 'unknown') provider,
                       COUNT(*) completed_jobs,
                       COALESCE(AVG(a.duration_ms),0) avg_duration_ms
                  FROM job_attempts a
                  JOIN jobs j ON j.id = a.job_id
                  LEFT JOIN slots s ON s.id = a.slot_id
                  LEFT JOIN agents g ON g.id = a.worker_id
                 WHERE a.state = 'succeeded' AND a.finished_at >= ? {recent_scope}
                 GROUP BY j.workload, j.route, a.worker_id, a.slot_id, s.model, g.provider
                """,
                [cutoff, *scoped_args],
            ).fetchall()
            active = conn.execute(
                f"""
                SELECT j.workload, j.route, j.worker_id, j.slot_id,
                       COALESCE(s.model, j.slot_id) model,
                       COALESCE(g.provider, 'unknown') provider, j.id job_id, j.state
                  FROM jobs j
                  LEFT JOIN slots s ON s.id = j.slot_id
                  LEFT JOIN agents g ON g.id = j.worker_id
                 WHERE j.state IN ('leased','running') {recent_scope}
                """,
                scoped_args,
            ).fetchall()

        documents: dict[tuple[str, str], dict[str, Any]] = {}
        for row in totals:
            item = dict(row)
            item["recent_completed_jobs"] = 0
            item["recent_failed_jobs"] = 0
            item["recent_completed_units"] = 0.0
            item["recent_busy_ms"] = 0.0
            item["workers"] = []
            item["active_workers"] = []
            documents[(row["workload"], row["route"])] = item
        for row in recent:
            item = documents.get((row["workload"], row["route"]))
            if item:
                item.update(
                    recent_completed_jobs=int(row["completed_jobs"] or 0),
                    recent_failed_jobs=int(row["failed_jobs"] or 0),
                    recent_completed_units=float(row["completed_units"] or 0),
                    recent_busy_ms=float(row["busy_ms"] or 0),
                )
        for row in workers:
            item = documents.get((row["workload"], row["route"]))
            if item:
                item["workers"].append(dict(row))
        for row in active:
            item = documents.get((row["workload"], row["route"]))
            if item:
                item["active_workers"].append(dict(row))
        return list(documents.values())

    def job_stats(self, *, project_id: str | None = None) -> dict[str, int]:
        """Return a per-state job count, optionally scoped to one project."""
        clauses: list[str] = []
        args: list[Any] = []
        if project_id is not None:
            clauses.append("project_id = ?")
            args.append(project_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT state, COUNT(*) AS n FROM jobs {where} GROUP BY state", args
            ).fetchall()
        return {row["state"]: int(row["n"]) for row in rows}

    def recent_events(
        self, *, limit: int = 50, project_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Return the most recent fleet events, newest first.

        ``project_id`` scopes the feed to a single tenant; without it the feed
        spans the whole control plane and must be gated to operators.
        """
        capped = max(1, min(int(limit), 500))
        with self._connection() as conn:
            if project_id is not None:
                rows = conn.execute(
                    """
                    SELECT e.id, e.job_id, e.event_type, e.actor, e.detail_json,
                           e.created_at
                      FROM job_events e JOIN jobs j ON j.id = e.job_id
                     WHERE j.project_id = ?
                     ORDER BY e.id DESC LIMIT ?
                    """,
                    (project_id, capped),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, job_id, event_type, actor, detail_json, created_at
                      FROM job_events
                     ORDER BY id DESC LIMIT ?
                    """,
                    (capped,),
                ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["detail"] = json.loads(item.pop("detail_json"))
            result.append(item)
        return result

    def issue_token(
        self,
        *,
        role: str,
        subject: str,
        scopes: dict[str, Any] | None = None,
        expires_at: datetime | None = None,
        token_id: str | None = None,
    ) -> str:
        if role not in {"admin", "client", "agent", "controller"}:
            raise ValueError("role must be admin, client, agent, or controller")
        if not subject:
            raise ValueError("token subject is required")
        identifier = token_id or secrets.token_hex(8)
        if "." in identifier:
            raise ValueError("token_id must not contain dots")
        secret = secrets.token_urlsafe(32)
        salt = secrets.token_bytes(16)
        derived = hashlib.scrypt(
            secret.encode("utf-8"), salt=salt, n=1 << 14, r=8, p=1, dklen=32
        ).hex()
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO auth_tokens (
                    id, secret_hash, salt_hex, role, subject, scopes_json,
                    expires_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    derived,
                    salt.hex(),
                    role,
                    subject,
                    _json(scopes or {}),
                    _iso(expires_at) if expires_at else None,
                    _iso(self.clock()),
                    _iso(self.clock()),
                ),
            )
        return f"sph_{identifier}.{secret}"

    def authenticate(self, raw_token: str) -> dict[str, Any]:
        parts = raw_token.removeprefix("sph_").split(".", 1)
        if not raw_token.startswith("sph_") or len(parts) != 2 or not parts[0] or not parts[1]:
            raise PermissionError("invalid token format")
        token_id, secret = parts
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM auth_tokens WHERE id = ?", (token_id,)
            ).fetchone()
        if row is None or row["disabled"]:
            raise PermissionError("invalid token")
        derived = hashlib.scrypt(
            secret.encode("utf-8"),
            salt=bytes.fromhex(row["salt_hex"]),
            n=1 << 14,
            r=8,
            p=1,
            dklen=32,
        ).hex()
        if not hmac.compare_digest(derived, row["secret_hash"]):
            raise PermissionError("invalid token")
        if row["expires_at"] and row["expires_at"] <= _iso(self.clock()):
            raise PermissionError("token expired")
        with self._transaction() as conn:
            conn.execute(
                "UPDATE auth_tokens SET last_used_at = ? WHERE id = ?",
                (_iso(self.clock()), token_id),
            )
        return {
            "id": row["id"],
            "role": row["role"],
            "subject": row["subject"],
            "scopes": json.loads(row["scopes_json"]),
            "expires_at": row["expires_at"],
        }

    def list_tokens(self) -> list[dict[str, Any]]:
        """Return safe token metadata; hashes, salts, and secrets never leave the store."""

        now = _iso(self.clock())
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, role, subject, scopes_json, expires_at, disabled,
                       created_at, updated_at, revoked_at, last_used_at
                  FROM auth_tokens
                 ORDER BY created_at DESC, id
                """
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["scopes"] = json.loads(item.pop("scopes_json"))
            item["disabled"] = bool(item["disabled"])
            item["updated_at"] = item["updated_at"] or item["created_at"]
            item["status"] = (
                "revoked" if item["disabled"]
                else "expired" if item["expires_at"] and item["expires_at"] <= now
                else "active"
            )
            result.append(item)
        return result

    def update_token(
        self,
        token_id: str,
        *,
        role: str,
        subject: str,
        scopes: dict[str, Any],
        expires_at: datetime | None,
    ) -> dict[str, Any]:
        """Update live credential claims without rotating its immutable secret."""

        if role not in {"admin", "client", "agent", "controller"}:
            raise ValueError("role must be admin, client, agent, or controller")
        if not subject:
            raise ValueError("token subject is required")
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT disabled FROM auth_tokens WHERE id = ?", (token_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"token {token_id!r} not found")
            if row["disabled"]:
                raise ConflictError("revoked tokens are immutable")
            conn.execute(
                """
                UPDATE auth_tokens
                   SET role = ?, subject = ?, scopes_json = ?, expires_at = ?, updated_at = ?
                 WHERE id = ?
                """,
                (
                    role, subject, _json(scopes),
                    _iso(expires_at) if expires_at else None,
                    _iso(self.clock()), token_id,
                ),
            )
        return next(row for row in self.list_tokens() if row["id"] == token_id)

    def disable_token(self, token_id: str) -> bool:
        now = _iso(self.clock())
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE auth_tokens
                   SET disabled = 1, revoked_at = ?, updated_at = ?
                 WHERE id = ? AND disabled = 0
                """,
                (now, now, token_id),
            )
            return cursor.rowcount > 0

    # ---- deployment ledger (operator-initiated provider workers) ----

    def record_deployment(
        self,
        *,
        pool_id: str,
        provider: str,
        resource_id: str,
        resource_name: str,
        worker_id: str,
        token_id: str,
        workload: str,
        route: str,
        capability: str,
        image: str,
        hourly_cost_usd: float | None,
        state: str,
        detail: str | None = None,
        **audit: Any,
    ) -> dict[str, Any]:
        now = _iso(self.clock())
        deployment_id = _new_id("dep")
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO deployments (
                    id, pool_id, provider, resource_id, resource_name,
                    worker_id, token_id, workload, route, capability, image,
                    hourly_cost_usd, state, detail, created_at, updated_at,
                    offer_id, gpu_model, priority, region, price_source,
                    availability_at_selection, rule_id, action_id,
                    provider_create_response, startup_deadline, failure_classification
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    deployment_id, pool_id, provider, resource_id, resource_name,
                    worker_id, token_id, workload, route, capability, image,
                    hourly_cost_usd, state, detail, now, now,
                    audit.get("offer_id"), audit.get("gpu_model"), audit.get("priority"),
                    audit.get("region"), audit.get("price_source"), audit.get("availability_at_selection"),
                    audit.get("rule_id"), audit.get("action_id"),
                    _json(audit["provider_create_response"]) if audit.get("provider_create_response") is not None else None,
                    audit.get("startup_deadline"), audit.get("failure_classification"),
                ),
            )
            row = conn.execute(
                "SELECT * FROM deployments WHERE id = ?", (deployment_id,)
            ).fetchone()
        assert row is not None
        return dict(row)

    def get_deployment(self, deployment_id: str) -> dict[str, Any]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM deployments WHERE id = ?", (deployment_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"deployment {deployment_id!r} not found")
        return dict(row)

    def list_deployments(
        self, *, rule_id: str | None = None, action_id: str | None = None, state: str | None = None, limit: int = 500
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        args: list[Any] = []
        if rule_id is not None:
            clauses.append("rule_id = ?")
            args.append(rule_id)
        if action_id is not None:
            clauses.append("action_id = ?")
            args.append(action_id)
        if state is not None:
            clauses.append("state = ?")
            args.append(state)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM deployments {where} ORDER BY created_at DESC LIMIT ?",
                (*args, max(1, min(int(limit), 1000))),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_deployment(
        self,
        deployment_id: str,
        *,
        state: str | None = None,
        detail: str | None = None,
        destroyed_at: str | None = None,
        hourly_cost_usd: float | None = None,
        resource_id: str | None = None,
        resource_name: str | None = None,
        **audit: Any,
    ) -> dict[str, Any]:
        assignments = ["updated_at = ?"]
        args: list[Any] = [_iso(self.clock())]
        if state is not None:
            assignments.append("state = ?")
            args.append(state)
        if detail is not None:
            assignments.append("detail = ?")
            args.append(detail)
        if destroyed_at is not None:
            assignments.append("destroyed_at = ?")
            args.append(destroyed_at)
        if hourly_cost_usd is not None:
            assignments.append("hourly_cost_usd = ?")
            args.append(hourly_cost_usd)
        if resource_id is not None:
            assignments.append("resource_id = ?")
            args.append(resource_id)
        if resource_name is not None:
            assignments.append("resource_name = ?")
            args.append(resource_name)
        for field in ("offer_id", "gpu_model", "priority", "region", "price_source", "availability_at_selection", "rule_id", "action_id", "startup_deadline", "first_heartbeat", "first_completion", "failure_classification"):
            if field in audit:
                assignments.append(f"{field} = ?")
                args.append(audit[field])
        if "provider_create_response" in audit:
            assignments.append("provider_create_response = ?")
            args.append(_json(audit["provider_create_response"]) if audit["provider_create_response"] is not None else None)
        args.append(deployment_id)
        with self._transaction() as conn:
            cursor = conn.execute(
                f"UPDATE deployments SET {', '.join(assignments)} WHERE id = ?", args
            )
            if cursor.rowcount == 0:
                raise NotFoundError(f"deployment {deployment_id!r} not found")
        return self.get_deployment(deployment_id)

    def active_jobs_for_worker(self, worker_id: str) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, state FROM jobs
                 WHERE worker_id = ? AND state IN ('leased', 'running')
                """,
                (worker_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def last_job_finished_at(self, worker_id: str) -> str | None:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT MAX(finished_at) AS finished_at FROM jobs
                 WHERE worker_id = ? AND state IN ('succeeded', 'failed')
                """,
                (worker_id,),
            ).fetchone()
        return row["finished_at"] if row and row["finished_at"] else None

    def _held_job(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        slot_id: str,
        lease_token: str,
        allowed_states: set[str],
    ) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"job {job_id!r} not found")
        if row["state"] not in allowed_states or row["slot_id"] != slot_id:
            raise LeaseError("job is not held by this slot in the required state")
        expected = row["lease_token_hash"] or ""
        if not expected or not hmac.compare_digest(expected, _lease_hash(lease_token)):
            raise LeaseError("invalid or stale lease token")
        if row["lease_expires_at"] and row["lease_expires_at"] < _iso(self.clock()):
            raise LeaseError("lease has expired")
        return row

    @staticmethod
    def _slot_matches(slot: sqlite3.Row, requirements: dict[str, Any]) -> bool:
        if int(requirements.get("gpu_count", 1)) != 1:
            return False
        if str(requirements.get("vendor") or "").casefold() != str(slot["vendor"]).casefold():
            return False
        if int(slot["vram_mb"]) < int(requirements.get("min_vram_mb") or 0):
            return False
        minimum_cc = _version_tuple(requirements.get("min_compute_capability"))
        if minimum_cc and _version_tuple(slot["compute_capability"]) < minimum_cc:
            return False
        if float(slot["cpu_cores"]) < float(requirements.get("min_cpu_cores") or 0):
            return False
        if int(slot["ram_mb"]) < int(requirements.get("min_ram_mb") or 0):
            return False
        if int(slot["disk_free_mb"]) < int(requirements.get("min_disk_mb") or 0):
            return False

        providers = {str(value).casefold() for value in requirements.get("allowed_providers") or []}
        if providers and str(slot["provider"]).casefold() not in providers:
            return False
        capabilities = set(json.loads(slot["capabilities_json"]))
        if requirements.get("capability") not in capabilities:
            return False
        available_packs = set(json.loads(slot["model_packs_json"]))
        if not set(requirements.get("model_packs") or []).issubset(available_packs):
            return False

        model = str(slot["model"]).casefold()
        allowed_models = [str(value).casefold() for value in requirements.get("gpu_models") or []]
        excluded_models = [
            str(value).casefold() for value in requirements.get("excluded_gpu_models") or []
        ]
        if allowed_models and not any(fnmatch.fnmatch(model, pattern) for pattern in allowed_models):
            return False
        if any(fnmatch.fnmatch(model, pattern) for pattern in excluded_models):
            return False

        labels = json.loads(slot["agent_labels_json"])
        labels.update(json.loads(slot["labels_json"]))
        for key, value in (requirements.get("required_labels") or {}).items():
            if labels.get(key) != value:
                return False
        return True

    @staticmethod
    def _free_slot(
        conn: sqlite3.Connection, slot_id: str, job_id: str, heartbeat_at: str
    ) -> None:
        conn.execute(
            """
            UPDATE slots SET state = 'ready', active_job_id = NULL, heartbeat_at = ?
             WHERE id = ? AND active_job_id = ?
            """,
            (heartbeat_at, slot_id, job_id),
        )

    @staticmethod
    def _finish_attempt(
        conn: sqlite3.Connection,
        job_row: sqlite3.Row,
        state: str,
        finished_at: str,
        error_code: str | None,
        error_message: str | None,
        now_dt: datetime,
    ) -> None:
        if int(job_row["attempts"]) <= 0:
            return
        started = datetime.fromisoformat(job_row["started_at"]) if job_row["started_at"] else now_dt
        duration_ms = max(0, int((now_dt - started).total_seconds() * 1000))
        conn.execute(
            """
            UPDATE job_attempts
               SET state = ?, finished_at = ?, duration_ms = ?,
                   error_code = ?, error_message = ?
             WHERE job_id = ? AND attempt = ?
            """,
            (
                state,
                finished_at,
                duration_ms,
                error_code,
                error_message,
                job_row["id"],
                job_row["attempts"],
            ),
        )

    @staticmethod
    def _event(
        conn: sqlite3.Connection,
        job_id: str,
        event_type: str,
        actor: str | None,
        detail: dict[str, Any],
        created_at: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO job_events(job_id, event_type, actor, detail_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (job_id, event_type, actor, _json(detail), created_at),
        )

    @staticmethod
    def _job_doc(row: sqlite3.Row) -> dict[str, Any]:
        document = dict(row)
        for source, target in JSON_COLUMNS.items():
            document[target] = json.loads(document.pop(source))
        document.pop("lease_token_hash", None)
        document["cancel_requested"] = bool(document["cancel_requested"])
        return document

    @staticmethod
    def _agent_doc(row: sqlite3.Row, slots: Sequence[sqlite3.Row]) -> dict[str, Any]:
        document = dict(row)
        document["labels"] = json.loads(document.pop("labels_json"))
        slot_docs = []
        for slot in slots:
            item = dict(slot)
            item["capabilities"] = json.loads(item.pop("capabilities_json"))
            item["model_packs"] = json.loads(item.pop("model_packs_json"))
            item["labels"] = json.loads(item.pop("labels_json"))
            slot_docs.append(item)
        document["slots"] = slot_docs
        return document

    # ---- scaling rules, evaluations, and actions ----

    def _rule_doc(self, row: sqlite3.Row) -> dict[str, Any]:
        document = dict(row)
        for key in (
            "trigger_json",
            "scale_up_json",
            "providers_json",
            "budgets_json",
            "scale_down_json",
        ):
            document[key.replace("_json", "")] = json.loads(document.pop(key))
        document["enabled"] = bool(document["enabled"])
        document["actor"] = document.pop("created_by") or document.get("updated_by") or "system"
        document.pop("updated_by", None)
        document.pop("kill_switch_generation", None)
        return document

    def _ensure_one_apply_rule_per_pool(
        self, pool_id: str, exclude_rule_id: str | None = None
    ) -> None:
        query = "SELECT id FROM scaling_rules WHERE pool_id = ? AND mode = 'apply' AND enabled = 1 AND deleted_at IS NULL"
        args: list[Any] = [pool_id]
        if exclude_rule_id is not None:
            query += " AND id != ?"
            args.append(exclude_rule_id)
        query += " LIMIT 1"
        with self._connection() as conn:
            row = conn.execute(query, args).fetchone()
        if row is not None:
            raise ConflictError(
                f"pool {pool_id!r} already has an enabled apply rule {row['id']!r}"
            )

    def create_scaling_rule(
        self,
        *,
        name: str,
        pool_id: str,
        mode: str,
        enabled: bool,
        trigger: dict[str, Any],
        scale_up: dict[str, Any],
        providers: dict[str, Any],
        budgets: dict[str, Any],
        scale_down: dict[str, Any],
        actor: str = "system",
    ) -> dict[str, Any]:
        if mode not in {"off", "report-only", "apply"}:
            raise ValueError(f"invalid scaling mode {mode!r}")
        if enabled and mode == "apply":
            self._ensure_one_apply_rule_per_pool(pool_id)
        rule_id = _new_id("rule")
        now = _iso(self.clock())
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO scaling_rules (
                    id, name, pool_id, mode, enabled, version,
                    trigger_json, scale_up_json, providers_json,
                    budgets_json, scale_down_json,
                    created_at, updated_at, created_by, updated_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rule_id,
                    name,
                    pool_id,
                    mode,
                    int(bool(enabled)),
                    1,
                    _json(trigger),
                    _json(scale_up),
                    _json(providers),
                    _json(budgets),
                    _json(scale_down),
                    now,
                    now,
                    actor,
                    actor,
                ),
            )
        return self.get_scaling_rule(rule_id)

    def get_scaling_rule(self, rule_id: str) -> dict[str, Any]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM scaling_rules WHERE id = ?", (rule_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"scaling rule {rule_id!r} not found")
        return self._rule_doc(row)

    def list_scaling_rules(
        self, *, pool_id: str | None = None, enabled: bool | None = None, deleted: bool = False
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        args: list[Any] = []
        if not deleted:
            clauses.append("deleted_at IS NULL")
        if pool_id is not None:
            clauses.append("pool_id = ?")
            args.append(pool_id)
        if enabled is not None:
            clauses.append("enabled = ?")
            args.append(int(bool(enabled)))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM scaling_rules {where} ORDER BY created_at DESC", args
            ).fetchall()
        return [self._rule_doc(row) for row in rows]

    def update_scaling_rule(
        self,
        rule_id: str,
        *,
        name: str | None = None,
        mode: str | None = None,
        enabled: bool | None = None,
        trigger: dict[str, Any] | None = None,
        scale_up: dict[str, Any] | None = None,
        providers: dict[str, Any] | None = None,
        budgets: dict[str, Any] | None = None,
        scale_down: dict[str, Any] | None = None,
        expected_version: int | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        if mode is not None and mode not in {"off", "report-only", "apply"}:
            raise ValueError(f"invalid scaling mode {mode!r}")
        existing = self.get_scaling_rule(rule_id)
        new_mode = mode if mode is not None else existing["mode"]
        new_enabled = enabled if enabled is not None else existing["enabled"]
        if new_enabled and new_mode == "apply":
            self._ensure_one_apply_rule_per_pool(existing["pool_id"], exclude_rule_id=rule_id)
        fields: dict[str, Any] = {}
        if name is not None:
            fields["name"] = name
        if mode is not None:
            fields["mode"] = mode
        if enabled is not None:
            fields["enabled"] = int(bool(enabled))
        if trigger is not None:
            fields["trigger_json"] = _json(trigger)
        if scale_up is not None:
            fields["scale_up_json"] = _json(scale_up)
        if providers is not None:
            fields["providers_json"] = _json(providers)
        if budgets is not None:
            fields["budgets_json"] = _json(budgets)
        if scale_down is not None:
            fields["scale_down_json"] = _json(scale_down)
        if not fields:
            return self.get_scaling_rule(rule_id)
        fields["updated_at"] = _iso(self.clock())
        fields["updated_by"] = actor
        fields["version"] = "version + 1"
        # version bump is a SQL expression; handle separately
        assignments = [f"{col} = ?" for col in fields if col != "version"]
        assignments.append("version = version + 1")
        args = [fields[col] for col in fields if col != "version"]
        where = "id = ?"
        args.append(rule_id)
        if expected_version is not None:
            where += " AND version = ?"
            args.append(int(expected_version))
        with self._transaction() as conn:
            cursor = conn.execute(
                f"UPDATE scaling_rules SET {', '.join(assignments)} WHERE {where}",
                args,
            )
            if cursor.rowcount == 0:
                if expected_version is not None:
                    raise ConflictError("scaling rule version mismatch")
                raise NotFoundError(f"scaling rule {rule_id!r} not found")
        return self.get_scaling_rule(rule_id)

    def delete_scaling_rule(self, rule_id: str) -> None:
        now = _iso(self.clock())
        with self._transaction() as conn:
            cursor = conn.execute("UPDATE scaling_rules SET enabled = 0, deleted_at = ?, updated_at = ? WHERE id = ? AND deleted_at IS NULL", (now, now, rule_id))
            if cursor.rowcount == 0:
                raise NotFoundError(f"scaling rule {rule_id!r} not found")

    def update_threshold_state(self, rule_id: str, state: str | None) -> dict[str, Any]:
        now = _iso(self.clock())
        with self._transaction() as conn:
            row = conn.execute("SELECT threshold_state, threshold_since FROM scaling_rules WHERE id = ?", (rule_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"scaling rule {rule_id!r} not found")
            since = row["threshold_since"] if row["threshold_state"] == state else now
            conn.execute("UPDATE scaling_rules SET threshold_state = ?, threshold_since = ? WHERE id = ?", (state, since, rule_id))
        return {"state": state, "since": since}

    def automation_state(self) -> dict[str, Any]:
        with self._connection() as conn:
            row = conn.execute("SELECT paused, generation, updated_at, updated_by FROM automation_state WHERE id = 1").fetchone()
        return {"paused": bool(row["paused"]), "generation": int(row["generation"]), "updated_at": row["updated_at"], "updated_by": row["updated_by"]}

    def set_automation_paused(self, paused: bool, actor: str) -> dict[str, Any]:
        with self._transaction() as conn:
            conn.execute("UPDATE automation_state SET paused = ?, generation = generation + 1, updated_at = ?, updated_by = ? WHERE id = 1", (int(paused), _iso(self.clock()), actor))
        return self.automation_state()

    def reserve_budget(
        self,
        *,
        rule_id: str,
        amount_usd: float,
        ttl_seconds: int = 3600,
        provider: str | None = None,
        automation_generation: int | None = None,
    ) -> dict[str, Any]:
        now = self.clock(); reservation_id = _new_id("reserve")
        with self._transaction() as conn:
            if automation_generation is not None:
                state = conn.execute("SELECT paused, generation FROM automation_state WHERE id = 1").fetchone()
                if state is None or int(state["paused"]) or int(state["generation"]) != automation_generation:
                    raise ConflictError("automation state changed during evaluation")
            conn.execute(
                "INSERT INTO budget_reservations(id, rule_id, provider, amount_usd, reserved_at, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
                (reservation_id, rule_id, provider, amount_usd, _iso(now), _iso(now + timedelta(seconds=ttl_seconds))),
            )
        return {"id": reservation_id, "amount_usd": amount_usd, "provider": provider}

    def sum_unexpired_reservations(
        self,
        *,
        rule_id: str | None = None,
        provider: str | None = None,
    ) -> float:
        query = "SELECT COALESCE(SUM(amount_usd), 0) AS total FROM budget_reservations WHERE reconciled_at IS NULL AND expires_at > ?"
        now = _iso(self.clock())
        args: list[Any] = [now]
        if rule_id is not None:
            query += " AND rule_id = ?"
            args.append(rule_id)
        if provider is not None:
            query += " AND provider = ?"
            args.append(provider)
        with self._connection() as conn:
            row = conn.execute(query, args).fetchone()
        return float(row["total"]) if row else 0.0

    def release_budget_reservation(self, reservation_id: str) -> None:
        with self._transaction() as conn:
            conn.execute("UPDATE budget_reservations SET reconciled_at = ? WHERE id = ? AND reconciled_at IS NULL", (_iso(self.clock()), reservation_id))

    def record_observation(self, *, kind: str, provider: str | None = None, offer_id: str | None = None, gpu_model: str | None = None, route: str | None = None, value: float | None = None, success: bool = False, expires_at: str | None = None) -> dict[str, Any]:
        observation_id = _new_id("obs")
        with self._transaction() as conn:
            conn.execute("INSERT INTO scaling_observations(id, kind, provider, offer_id, gpu_model, route, value, success_count, observed_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (observation_id, kind, provider, offer_id, gpu_model, route, value if value is None or math.isfinite(value) else None, int(success), _iso(self.clock()), expires_at))
        return {"id": observation_id}

    def recent_observations(self, *, kind: str, provider: str | None = None, offer_id: str | None = None, seconds: int = 3600) -> list[dict[str, Any]]:
        cutoff = _iso(self.clock() - timedelta(seconds=seconds)); clauses = ["kind = ?", "observed_at >= ?"]; args: list[Any] = [kind, cutoff]
        if provider is not None: clauses.append("provider = ?"); args.append(provider)
        if offer_id is not None: clauses.append("offer_id = ?"); args.append(offer_id)
        with self._connection() as conn: rows = conn.execute(f"SELECT * FROM scaling_observations WHERE {' AND '.join(clauses)} ORDER BY observed_at DESC", args).fetchall()
        return [dict(row) for row in rows]

    def count_recent_actions(self, rule_id: str, *, action: str = "deploy", hours: int = 24) -> int:
        cutoff = _iso(self.clock() - timedelta(hours=hours))
        with self._connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM scaling_actions WHERE rule_id = ? AND action = ? AND created_at >= ?", (rule_id, action, cutoff)).fetchone()[0])

    def record_evaluation(
        self,
        *,
        rule_id: str,
        rule_version: int,
        snapshot: dict[str, Any],
        threshold_state: str | None,
        current_slots: int,
        in_flight_slots: int,
        desired_slots: int,
        decision: str,
        blocked_reason: str | None,
        selected_candidate: dict[str, Any] | None,
        alternatives: list[dict[str, Any]],
        projected_eta_hours: float | None,
        projected_cost_usd: float | None,
    ) -> dict[str, Any]:
        now = _iso(self.clock())
        evaluation_id = _new_id("eval")
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO scaling_evaluations (
                    id, rule_id, rule_version, snapshot_json, threshold_state,
                    current_slots, in_flight_slots, desired_slots, decision,
                    blocked_reason, selected_candidate_json, alternatives_json,
                    projected_eta_hours, projected_cost_usd, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_id,
                    rule_id,
                    rule_version,
                    _json(snapshot),
                    threshold_state,
                    current_slots,
                    in_flight_slots,
                    desired_slots,
                    decision,
                    blocked_reason,
                    _json(selected_candidate) if selected_candidate is not None else None,
                    _json(alternatives),
                    projected_eta_hours,
                    projected_cost_usd,
                    now,
                ),
            )
            conn.execute(
                "UPDATE scaling_rules SET last_evaluation_at = ? WHERE id = ?",
                (now, rule_id),
            )
        return self.get_evaluation(evaluation_id)

    def get_evaluation(self, evaluation_id: str) -> dict[str, Any]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM scaling_evaluations WHERE id = ?", (evaluation_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"evaluation {evaluation_id!r} not found")
        document = dict(row)
        document["snapshot"] = json.loads(document.pop("snapshot_json"))
        document["selected_candidate"] = (
            json.loads(document["selected_candidate_json"])
            if document["selected_candidate_json"]
            else None
        )
        document["alternatives"] = json.loads(document.pop("alternatives_json"))
        return document

    def list_evaluations(
        self, *, rule_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        capped = max(1, min(int(limit), 1000))
        with self._connection() as conn:
            if rule_id:
                rows = conn.execute(
                    "SELECT * FROM scaling_evaluations WHERE rule_id = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (rule_id, capped),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM scaling_evaluations ORDER BY created_at DESC LIMIT ?",
                    (capped,),
                ).fetchall()
        return [self.get_evaluation(row["id"]) for row in rows]

    def record_action(
        self,
        *,
        rule_id: str,
        evaluation_id: str | None,
        idempotency_key: str,
        action: str,
        state: str,
        origin: str | None = None,
        deployment_id: str | None = None,
        token_id: str | None = None,
        reservation: dict[str, Any] | None = None,
        provider_request: dict[str, Any] | None = None,
        provider_result: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        now = _iso(self.clock())
        action_id = _new_id("act")
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO scaling_actions (
                    id, rule_id, evaluation_id, idempotency_key, action, state,
                    origin, deployment_id, token_id, reservation_json,
                    provider_request_json, provider_result_json, trace_id,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action_id,
                    rule_id,
                    evaluation_id,
                    idempotency_key,
                    action,
                    state,
                    origin,
                    deployment_id,
                    token_id,
                    _json(reservation) if reservation is not None else None,
                    _json(provider_request) if provider_request is not None else None,
                    _json(provider_result) if provider_result is not None else None,
                    trace_id,
                    now,
                ),
            )
            conn.execute(
                "UPDATE scaling_rules SET last_action_at = ? WHERE id = ?",
                (now, rule_id),
            )
        return self.get_action(action_id)

    def get_action(self, action_id: str) -> dict[str, Any]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM scaling_actions WHERE id = ?", (action_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"action {action_id!r} not found")
        document = dict(row)
        document["reservation"] = (
            json.loads(document["reservation_json"]) if document["reservation_json"] else None
        )
        document["provider_request"] = (
            json.loads(document["provider_request_json"]) if document["provider_request_json"] else None
        )
        document["provider_result"] = (
            json.loads(document["provider_result_json"]) if document["provider_result_json"] else None
        )
        document["fallback_lineage"] = json.loads(document.pop("fallback_lineage_json"))
        return document

    def list_actions(
        self, *, rule_id: str | None = None, limit: int = 100, state: str | None = None
    ) -> list[dict[str, Any]]:
        capped = max(1, min(int(limit), 1000))
        clauses: list[str] = []
        args: list[Any] = []
        if rule_id:
            clauses.append("rule_id = ?")
            args.append(rule_id)
        if state:
            clauses.append("state = ?")
            args.append(state)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM scaling_actions {where} ORDER BY created_at DESC LIMIT ?",
                (*args, capped),
            ).fetchall()
        return [self.get_action(row["id"]) for row in rows]

    def update_action_state(
        self,
        action_id: str,
        *,
        state: str,
        deployment_id: str | None = None,
        token_id: str | None = None,
        provider_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        fields = ["state = ?", "finished_at = ?"]
        now = _iso(self.clock())
        args: list[Any] = [state, now]
        if deployment_id is not None:
            fields.append("deployment_id = ?")
            args.append(deployment_id)
        if token_id is not None:
            fields.append("token_id = ?")
            args.append(token_id)
        if provider_result is not None:
            fields.append("provider_result_json = ?")
            args.append(_json(provider_result))
        args.append(action_id)
        with self._transaction() as conn:
            cursor = conn.execute(
                f"UPDATE scaling_actions SET {', '.join(fields)} WHERE id = ?",
                args,
            )
            if cursor.rowcount == 0:
                raise NotFoundError(f"action {action_id!r} not found")
        return self.get_action(action_id)

    def acquire_scaling_lease(
        self, rule_id: str, owner: str, *, ttl_seconds: int = 60
    ) -> bool:
        now_dt = self.clock()
        now = _iso(now_dt)
        expires = _iso(now_dt + timedelta(seconds=max(10, int(ttl_seconds))))
        with self._transaction() as conn:
            existing = conn.execute(
                "SELECT owner, expires_at FROM scaling_leases WHERE rule_id = ?", (rule_id,)
            ).fetchone()
            if existing is not None and existing["expires_at"] > now and existing["owner"] != owner:
                return False
            conn.execute(
                """
                INSERT INTO scaling_leases (rule_id, owner, acquired_at, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(rule_id) DO UPDATE SET
                    owner = excluded.owner,
                    acquired_at = excluded.acquired_at,
                    expires_at = excluded.expires_at
                """,
                (rule_id, owner, now, expires),
            )
        return True

    def release_scaling_lease(self, rule_id: str, owner: str) -> bool:
        with self._transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM scaling_leases WHERE rule_id = ? AND owner = ?",
                (rule_id, owner),
            )
            return cursor.rowcount > 0
