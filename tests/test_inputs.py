"""Scoped input upload, materialization, and plugin-lifecycle tests.

Inputs are private project blobs uploaded through
``POST /v1/projects/{project}/inputs`` and fetched by the leased agent through
the returned URI. These tests cover the upload boundary, the job-references
authorization path, agent-side materialization safety, and per-slot plugin
instance reuse/release.
"""

from __future__ import annotations

import hashlib
import sys
import textwrap

import httpx
import pytest

from fastapi.testclient import TestClient

from sparehour.agent import (
    AgentConfig,
    AgentError,
    AgentRunner,
    ControlPlaneClient,
    ExecutionError,
    _process_request,
)
from sparehour.api import create_app
from sparehour.artifacts import ArtifactStore
from sparehour.models import AgentRegistration, SlotInventory, WorkloadSpec
from sparehour.service import Orchestrator
from sparehour.store import NotFoundError

from conftest import register_agent


AUDIO_WORKLOAD = {
    "name": "audio-copy",
    "version": "1",
    "default_route": "command",
    "payload_schema": {
        "type": "object",
        "required": ["message"],
        "properties": {"message": {"type": "string"}},
    },
    "inputs": [
        {
            "name": "audio",
            "media_types": ["audio/wav"],
            "max_bytes": 4096,
            "validator": "sha256",
        },
        {"name": "notes", "required": False, "validator": "sha256"},
    ],
    "outputs": [
        {
            "name": "result",
            "media_types": ["application/json"],
            "max_bytes": 1048576,
            "validator": "json",
        }
    ],
    "routes": {
        "command": {
            "compatibility": {"capability": "audio-copy", "model_packs": []},
            "resources": {
                "gpu_count": 1,
                "vendor": "nvidia",
                "min_vram_mb": 1,
                "min_cpu_cores": 0,
                "min_ram_mb": 0,
                "min_disk_mb": 0,
            },
            "executor": {
                "type": "command",
                "command": ["{agent_python}", "-m", "sparehour.demo_executor"],
                "timeout_seconds": 60,
            },
            "cost": {
                "unit": "item",
                "default_units_per_job": 1,
                "default_units_per_gpu_hour": 3600,
            },
            "scaling": {"enabled": False, "min_slots": 0, "max_slots": 1},
        }
    },
}

AUDIO_BYTES = b"RIFF....WAVEfmt fake-audio-payload"
AUDIO_SHA = hashlib.sha256(AUDIO_BYTES).hexdigest()


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def audio_orchestrator(registry, store) -> Orchestrator:
    registry.add(WorkloadSpec.model_validate(AUDIO_WORKLOAD))
    return Orchestrator(registry, store)


@pytest.fixture
def api(audio_orchestrator, tmp_path):
    return TestClient(
        create_app(audio_orchestrator, ArtifactStore(tmp_path / "artifacts"))
    )


def _upload(api, token, *, project="project-a", name="audio", data=AUDIO_BYTES,
            content_type="audio/wav", workload="audio-copy"):
    return api.post(
        f"/v1/projects/{project}/inputs",
        params={"workload": workload, "name": name},
        headers={**bearer(token), "Content-Type": content_type},
        content=data,
    )


def _submit(api, token, *, inputs=None, project="project-a"):
    return api.post(
        "/v1/jobs",
        headers=bearer(token),
        json={
            "project_id": project,
            "workload": "audio-copy",
            "payload": {"message": "hello"},
            "inputs": inputs or [],
        },
    )


def test_input_upload_download_and_job_reference(api, store):
    client_token = store.issue_token(role="client", subject="project-a")
    upload = _upload(api, client_token)
    assert upload.status_code == 201, upload.text
    artifact = upload.json()
    assert artifact["sha256"] == AUDIO_SHA
    assert artifact["size"] == len(AUDIO_BYTES)
    assert artifact["uri"] == f"/v1/projects/project-a/inputs/{AUDIO_SHA}"

    downloaded = api.get(artifact["uri"], headers=bearer(client_token))
    assert downloaded.status_code == 200
    assert downloaded.content == AUDIO_BYTES

    submitted = _submit(
        api, client_token, inputs=[{"name": "audio", "uri": artifact["uri"]}]
    )
    assert submitted.status_code == 201, submitted.text
    stored = submitted.json()["inputs"][0]
    assert stored["sha256"] == AUDIO_SHA
    assert stored["size"] == len(AUDIO_BYTES)
    assert stored["media_type"] == "audio/wav"


def test_upload_requires_project_and_workload_scope(api, store):
    client_token = store.issue_token(role="client", subject="project-a")
    other_project = store.issue_token(role="client", subject="project-b")
    scoped = store.issue_token(
        role="client",
        subject="project-a",
        scopes={"workloads": ["demo-gpu"]},
    )
    agent_token = store.issue_token(role="agent", subject="agent-a")

    assert _upload(api, other_project).status_code == 403
    assert _upload(api, scoped).status_code == 403
    assert _upload(api, agent_token).status_code == 403
    assert api.post(
        "/v1/projects/project-a/inputs",
        params={"workload": "audio-copy", "name": "audio"},
        content=AUDIO_BYTES,
    ).status_code == 401


def test_upload_rejects_undeclared_name_type_and_oversize(api, store):
    client_token = store.issue_token(role="client", subject="project-a")
    assert _upload(api, client_token, name="archive").status_code == 422
    assert (
        _upload(api, client_token, content_type="text/plain").status_code == 422
    )
    oversized = _upload(api, client_token, data=b"x" * 5000)
    assert oversized.status_code == 422
    assert "max_bytes" in oversized.text
    empty = _upload(api, client_token, data=b"")
    assert empty.status_code == 422


def test_submit_rejects_forged_or_foreign_inputs(api, store, audio_orchestrator):
    client_token = store.issue_token(role="client", subject="project-a")
    other_token = store.issue_token(role="client", subject="project-b")
    artifact = _upload(api, client_token).json()
    uri = artifact["uri"]

    forged = [
        {"name": "audio", "uri": "file:///etc/passwd", "sha256": AUDIO_SHA},
        {"name": "audio", "uri": "https://evil.example/x", "sha256": AUDIO_SHA},
        {"name": "audio", "uri": "/v1/projects/project-b/inputs/" + AUDIO_SHA,
         "sha256": AUDIO_SHA},
        {"name": "audio", "uri": "/v1/projects/project-a/inputs/" + "0" * 64,
         "sha256": "0" * 64},
        {"name": "audio", "uri": uri, "sha256": "f" * 64},
        {"name": "audio", "uri": uri, "sha256": AUDIO_SHA, "size": 999999},
        {"name": "audio", "uri": uri, "sha256": AUDIO_SHA,
         "media_type": "application/octet-stream"},
        {"name": "audio", "uri": uri, "sha256": AUDIO_SHA, "path": "/tmp/x"},
        {"name": "missing", "uri": uri, "sha256": AUDIO_SHA},
        "not-an-object",
        {"name": "audio"},
    ]
    for entry in forged:
        response = _submit(api, client_token, inputs=[entry])
        assert response.status_code == 422, (entry, response.text)
    # project-b's token may submit to project-b, but a blob granted to
    # project-a is rejected as a foreign/forged input reference (422).
    assert (
        _submit(api, other_token, project="project-b",
                inputs=[{"name": "audio", "uri": uri, "sha256": AUDIO_SHA}])
        .status_code
        == 422
    )


def test_agent_download_requires_lease_and_membership(api, store, audio_orchestrator):
    register_agent(store, "agent-a", capabilities=["audio-copy", "demo-gpu"])
    client_token = store.issue_token(role="client", subject="project-a")
    agent_token = store.issue_token(role="agent", subject="agent-a")
    artifact = _upload(api, client_token).json()

    # Bearer-only agent access must not reach project blobs.
    assert api.get(artifact["uri"], headers=bearer(agent_token)).status_code == 403

    job = _submit(
        api,
        client_token,
        inputs=[{"name": "audio", "uri": artifact["uri"]}],
    ).json()
    claim = api.post(
        "/v1/slots/agent-a-slot-0/claim",
        headers=bearer(agent_token),
        json={"lease_seconds": 90},
    ).json()
    assert claim["id"] == job["id"]
    lease_headers = {
        **bearer(agent_token),
        "X-SPAREHOUR-Lease-Token": claim["lease_token"],
    }
    params = {"job_id": job["id"], "slot_id": "agent-a-slot-0"}

    fetched = api.get(artifact["uri"], headers=lease_headers, params=params)
    assert fetched.status_code == 200
    assert fetched.content == AUDIO_BYTES

    # Incomplete lease credentials are refused.
    assert api.get(
        artifact["uri"], headers=bearer(agent_token), params=params
    ).status_code == 403
    assert api.get(
        artifact["uri"], headers=lease_headers, params={"slot_id": "agent-a-slot-0"}
    ).status_code == 403

    # A lease for a different job does not reference this blob.
    other_job = api.post(
        "/v1/jobs",
        headers=bearer(client_token),
        json={
            "project_id": "project-a",
            "workload": "demo-gpu",
            "payload": {"message": "x"},
        },
    ).json()
    # Claim on the same slot after the first job completes.
    api.post(
        f"/v1/jobs/{job['id']}/confirm?slot_id=agent-a-slot-0",
        headers=lease_headers,
    )
    api.post(
        f"/v1/jobs/{job['id']}/fail?slot_id=agent-a-slot-0",
        headers=lease_headers,
        json={"error_code": "executor_failed", "error_message": "stop"},
    )
    # Dead-lettered? demo retry policy may retry; claim whichever job returns.
    second = api.post(
        "/v1/slots/agent-a-slot-0/claim",
        headers=bearer(agent_token),
        json={"lease_seconds": 90},
    ).json()
    assert second is not None
    second_headers = {
        **bearer(agent_token),
        "X-SPAREHOUR-Lease-Token": second["lease_token"],
    }
    if second["id"] == other_job["id"]:
        # A job that never declared the blob cannot authorize its download.
        assert api.get(
            artifact["uri"], headers=second_headers,
            params={"job_id": other_job["id"], "slot_id": "agent-a-slot-0"},
        ).status_code == 403
    # Wrong lease token is rejected even for a valid job.
    assert api.get(
        artifact["uri"],
        headers={**bearer(agent_token), "X-SPAREHOUR-Lease-Token": "forged"},
        params={"job_id": second["id"], "slot_id": "agent-a-slot-0"},
    ).status_code == 409

    # A cancelled job's lease no longer authorizes anything.
    api.delete(f"/v1/jobs/{second['id']}", headers=bearer(client_token))
    api.post(
        f"/v1/jobs/{second['id']}/release?slot_id=agent-a-slot-0",
        headers=second_headers,
        json={"reason": "done"},
    )
    assert api.get(
        artifact["uri"], headers=second_headers,
        params={"job_id": second["id"], "slot_id": "agent-a-slot-0"},
    ).status_code in {403, 409}

    # Other projects' clients still cannot read it.
    other_token = store.issue_token(role="client", subject="project-b")
    assert api.get(artifact["uri"], headers=bearer(other_token)).status_code == 403
    # Unknown digests are 404, malformed ones too.
    assert (
        api.get("/v1/projects/project-a/inputs/" + "0" * 64,
                headers=bearer(client_token)).status_code == 404
    )
    assert (
        api.get("/v1/projects/project-a/inputs/notasha",
                headers=bearer(client_token)).status_code == 404
    )


def test_input_grant_is_durable_and_content_keyed(store):
    first = store.put_input(
        project_id="project-a", name="audio", sha256=AUDIO_SHA,
        size=len(AUDIO_BYTES), media_type="audio/wav",
        workload="audio-copy", uploader="project-a",
    )
    second = store.put_input(
        project_id="project-a", name="renamed", sha256=AUDIO_SHA,
        size=len(AUDIO_BYTES), media_type="audio/wav",
        workload="audio-copy", uploader="project-a",
    )
    assert second["name"] == first["name"] == "audio"
    assert store.get_input("project-a", AUDIO_SHA)["size"] == len(AUDIO_BYTES)
    assert store.get_input("project-b", AUDIO_SHA) is None
    assert store.get_input("project-a", "0" * 64) is None
    with pytest.raises(NotFoundError):
        store.get_job("job_missing")


# ---- agent-side materialization ----


class _FakeClient:
    """Minimal transport fake: returns fixed bytes for any input URI."""

    def __init__(self, payload: bytes = AUDIO_BYTES, **overrides):
        self.payload = payload
        self.calls = []
        self.resolve_error = overrides.get("resolve_error")

    def resolve_input_uri(self, uri: str) -> str:
        if self.resolve_error is not None:
            raise AgentError(self.resolve_error)
        if not (isinstance(uri, str) and uri.startswith("/v1/projects/")):
            raise AgentError("input uri must stay on the controller origin")
        return uri

    def download_input(self, uri, *, job_id, slot_id, lease_token, destination,
                       max_bytes, check=None):
        self.calls.append(uri)
        if check is not None:
            check()
        destination.write_bytes(self.payload)
        return len(self.payload), hashlib.sha256(self.payload).hexdigest()


class _Lease:
    job_id = "job-a"
    slot_id = "slot-a"
    lease_token = "lease"

    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after

    def check(self):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise ExecutionError("cancelled", "job cancellation was requested")


def _runner(tmp_path, client=None):
    slot = SlotInventory(
        id="slot-a", gpu_index=0, gpu_uuid="GPU-a", model="fake", vram_mb=8192
    )
    registration = AgentRegistration(
        agent_id="agent-a", provider="local", hostname="h", slots=[slot]
    )
    runner = AgentRunner(
        client or _FakeClient(),
        AgentConfig(registration=registration, workdir_root=tmp_path / "root"),
    )
    return runner, slot


def _job(inputs, input_contract=None):
    return {
        "id": "job-a",
        "lease_token": "lease",
        "payload": {},
        "inputs": inputs,
        "input_contract": input_contract
        or [
            {
                "name": "audio",
                "media_types": ["audio/wav"],
                "max_bytes": 4096,
                "validator": "sha256",
                "required": True,
                "multiple": False,
            }
        ],
        "output_contract": [],
        "executor": {"type": "plugin", "handler": "fake:Fake"},
    }


def _input_entry(**overrides):
    entry = {
        "name": "audio",
        "uri": f"/v1/projects/project-a/inputs/{AUDIO_SHA}",
        "sha256": AUDIO_SHA,
        "size": len(AUDIO_BYTES),
        "media_type": "audio/wav",
    }
    entry.update(overrides)
    return entry


def test_materialize_downloads_verifies_and_rewrites(tmp_path):
    runner, slot = _runner(tmp_path)
    workdir = tmp_path / "job"
    workdir.mkdir()
    job = _job([_input_entry()])
    runner._materialize_inputs(job, workdir, _Lease())

    item = job["inputs"][0]
    target = workdir / "inputs" / "audio"
    assert item["path"] == str(target)
    assert item["uri"] == target.resolve().as_uri()
    assert target.read_bytes() == AUDIO_BYTES


def test_materialize_rejects_forged_uris_and_metadata(tmp_path):
    runner, _ = _runner(tmp_path)
    workdir = tmp_path / "job"
    workdir.mkdir()
    lease = _Lease()

    cases = [
        _input_entry(uri="file:///etc/passwd"),
        _input_entry(uri="https://evil.example/x"),
        _input_entry(uri="/v1/jobs/job-a/artifacts/" + AUDIO_SHA),
        _input_entry(name="../escape"),
        _input_entry(name="UPPER"),
        _input_entry(size=-1),
        _input_entry(size=True),
        _input_entry(size=99999),
        _input_entry(media_type="application/octet-stream"),
    ]
    for entry in cases:
        with pytest.raises(ExecutionError) as excinfo:
            runner._materialize_inputs(_job([entry]), workdir, lease)
        assert excinfo.value.code == "input_validation_failed", entry
    assert not (workdir / "inputs" / "escape").exists()


def test_materialize_fails_on_size_sha_mismatch_and_cancel(tmp_path):
    workdir = tmp_path / "job"
    workdir.mkdir()

    runner, _ = _runner(tmp_path, client=_FakeClient(payload=b"shorter"))
    with pytest.raises(ExecutionError) as excinfo:
        runner._materialize_inputs(_job([_input_entry()]), workdir, _Lease())
    assert excinfo.value.code == "artifact_transport_failed"

    corrupt = AUDIO_SHA[:-1] + ("0" if AUDIO_SHA[-1] != "0" else "1")
    runner, _ = _runner(tmp_path)
    with pytest.raises(ExecutionError) as excinfo:
        runner._materialize_inputs(
            _job([_input_entry(sha256=corrupt)]), workdir, _Lease()
        )
    assert excinfo.value.code == "artifact_transport_failed"

    class CancellingLease(_Lease):
        def check(self):
            raise ExecutionError("cancelled", "job cancellation was requested")

    runner, _ = _runner(tmp_path)
    with pytest.raises(ExecutionError) as excinfo:
        runner._materialize_inputs(
            _job([_input_entry()]), workdir, CancellingLease()
        )
    assert excinfo.value.code == "cancelled"


def test_download_input_streams_bounded_and_cleans_partial(tmp_path):
    destination = tmp_path / "inputs" / "audio"
    destination.parent.mkdir()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=AUDIO_BYTES)

    client = ControlPlaneClient("http://controller.test", "token")
    client.client = httpx.Client(
        base_url="http://controller.test",
        transport=httpx.MockTransport(handler),
    )
    size, digest = client.download_input(
        "/v1/projects/project-a/inputs/" + AUDIO_SHA,
        job_id="job-a",
        slot_id="slot-a",
        lease_token="lease",
        destination=destination,
        max_bytes=len(AUDIO_BYTES),
    )
    assert (size, digest) == (len(AUDIO_BYTES), AUDIO_SHA)
    assert destination.read_bytes() == AUDIO_BYTES

    # A download bigger than the declared bound aborts without a final file.
    big = destination.with_name("big")
    with pytest.raises(AgentError):
        client.download_input(
            "/v1/projects/project-a/inputs/" + AUDIO_SHA,
            job_id="job-a",
            slot_id="slot-a",
            lease_token="lease",
            destination=big,
            max_bytes=4,
        )
    assert not big.exists()
    assert not list(destination.parent.glob("*.part"))
    client.close()


def test_resolve_input_uri_refuses_foreign_schemes():
    client = ControlPlaneClient("http://127.0.0.1:8088", "token")
    try:
        good = "/v1/projects/project-a/inputs/" + AUDIO_SHA
        assert client.resolve_input_uri(good) == good
        assert client.resolve_input_uri(
            "http://127.0.0.1:8088" + good
        ) == good
        for bad in (
            "file:///etc/passwd",
            "https://evil.example/v1/projects/p/inputs/" + AUDIO_SHA,
            "http://127.0.0.1:9999" + good,
            "relative/path",
            "/v1/jobs/job-a/artifacts/" + AUDIO_SHA,
            "",
        ):
            with pytest.raises(AgentError):
                client.resolve_input_uri(bad)
    finally:
        client.close()


def test_process_request_maps_container_paths(tmp_path):
    workdir = tmp_path / "job"
    inside = workdir / "inputs" / "audio"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(AUDIO_BYTES)
    job = _job(
        [
            _input_entry(
                path=str(inside), uri=inside.resolve().as_uri()
            )
        ]
    )

    command_doc = _process_request(job, workdir, container=False)
    assert command_doc["inputs"][0]["path"] == str(inside)
    # The original job document is not mutated.
    assert job["inputs"][0]["path"] == str(inside)

    container_doc = _process_request(job, workdir, container=True)
    assert container_doc["inputs"][0]["path"] == "/job/inputs/audio"
    assert container_doc["inputs"][0]["uri"] == "file:///job/inputs/audio"
    assert job["inputs"][0]["path"] == str(inside)

    escaped = _job([_input_entry(path="/etc/passwd")])
    with pytest.raises(ExecutionError):
        _process_request(escaped, workdir, container=True)


# ---- per-slot plugin instance lifetime ----


@pytest.fixture
def fake_plugins(tmp_path, monkeypatch):
    module = tmp_path / "fakeplug.py"
    module.write_text(
        textwrap.dedent(
            """
            INSTANCES = []

            class Echo:
                def __init__(self):
                    self.released = []
                    INSTANCES.append(self)

                def execute(self, job, workdir, gpu_index):
                    return []

                def release_gpu(self, gpu_index):
                    self.released.append(gpu_index)

            class Other(Echo):
                pass

            class Sticky(Echo):
                def release_gpu(self, gpu_index):
                    self.released.append(gpu_index)
                    raise RuntimeError("VRAM still allocated")
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("fakeplug", None)
    import fakeplug  # noqa: E402

    yield fakeplug
    sys.modules.pop("fakeplug", None)


def test_plugin_instance_reused_and_released_on_switch(tmp_path, fake_plugins):
    runner, slot = _runner(tmp_path)
    workdir = tmp_path / "job"
    workdir.mkdir()

    job_a = _job([])
    job_a["executor"] = {"type": "plugin", "handler": "fakeplug:Echo"}
    runner._execute_plugin(slot, job_a, workdir, _Lease())
    runner._execute_plugin(slot, job_a, workdir, _Lease())
    assert len(fake_plugins.INSTANCES) == 1

    job_b = _job([])
    job_b["executor"] = {"type": "plugin", "handler": "fakeplug:Other"}
    runner._execute_plugin(slot, job_b, workdir, _Lease())
    assert len(fake_plugins.INSTANCES) == 2
    first, second = fake_plugins.INSTANCES
    assert first.released == [0]
    assert second.released == []
    assert runner._plugins[slot.id][1] is second

    # A non-plugin job releases the live plugin before dispatch.
    command_job = _job([])
    command_job["executor"] = {
        "type": "command",
        "command": ["true"],
        "timeout_seconds": 5,
    }
    seen = {}
    def record(*args, **kwargs):
        seen["plugins"] = dict(runner._plugins)
        return []

    runner._execute_process = record
    runner._execute(slot, command_job, workdir, _Lease())
    assert seen["plugins"] == {}
    assert second.released == [0]


def test_shutdown_releases_active_plugin(tmp_path, fake_plugins, monkeypatch):
    runner, slot = _runner(tmp_path)
    workdir = tmp_path / "job"
    workdir.mkdir()
    job = _job([])
    job["executor"] = {"type": "plugin", "handler": "fakeplug:Echo"}
    runner._execute_plugin(slot, job, workdir, _Lease())
    assert runner._plugins

    class Client:
        def register(self, _registration):
            return None

        def heartbeat_agent(self, *_args, **kwargs):
            return None

        def claim(self, *_args):
            return None

    monkeypatch.setattr(runner, "client", Client())
    runner.run(once=True)
    assert fake_plugins.INSTANCES[0].released == [0]
    assert runner._plugins == {}


def test_failed_release_retains_instance_and_is_fatal(tmp_path, fake_plugins):
    """A failed release_gpu must not hand off the GPU or drop the instance."""
    from sparehour.agent import DRAINING_EXECUTION_ERRORS

    runner, slot = _runner(tmp_path)
    workdir = tmp_path / "job"
    workdir.mkdir()

    job_a = _job([])
    job_a["executor"] = {"type": "plugin", "handler": "fakeplug:Sticky"}
    runner._execute_plugin(slot, job_a, workdir, _Lease())
    sticky = fake_plugins.INSTANCES[0]

    # Handler switch triggers release; failure must abort the handoff and
    # keep the instance registered (VRAM still owned by Sticky).
    job_b = _job([])
    job_b["executor"] = {"type": "plugin", "handler": "fakeplug:Other"}
    with pytest.raises(ExecutionError) as excinfo:
        runner._execute_plugin(slot, job_b, workdir, _Lease())
    assert excinfo.value.code == "plugin_release_failed"
    assert "plugin_release_failed" in DRAINING_EXECUTION_ERRORS
    assert "cuda_context_failure" in DRAINING_EXECUTION_ERRORS
    assert runner._plugins[slot.id][1] is sticky
    assert len(fake_plugins.INSTANCES) == 1

    # Same invariant on the non-plugin path: no pop-then-forget.
    command_job = _job([])
    command_job["executor"] = {
        "type": "command",
        "command": ["true"],
        "timeout_seconds": 5,
    }
    runner._execute_process = lambda *args, **kwargs: []
    with pytest.raises(ExecutionError) as excinfo:
        runner._execute(slot, command_job, workdir, _Lease())
    assert excinfo.value.code == "plugin_release_failed"
    assert runner._plugins[slot.id][1] is sticky
    assert sticky.released == [0, 0]


@pytest.mark.parametrize("role", ["agent", "controller"])
def test_non_client_subject_collision_cannot_read_project_inputs(api, store, role):
    owner = store.issue_token(role="client", subject="project-a")
    artifact = _upload(api, owner).json()
    unrelated_role = store.issue_token(role=role, subject="project-a")

    response = api.get(artifact["uri"], headers=bearer(unrelated_role))

    assert response.status_code == 403
    assert AUDIO_BYTES not in response.content


def test_foreign_input_membership_is_not_disclosed(api, store):
    owner = store.issue_token(role="client", subject="project-a")
    artifact = _upload(api, owner).json()
    outsider = store.issue_token(role="client", subject="project-b")
    existing = api.get(artifact["uri"], headers=bearer(outsider))
    absent = api.get(
        "/v1/projects/project-a/inputs/" + "f" * 64, headers=bearer(outsider)
    )

    assert existing.status_code == absent.status_code == 403
    assert existing.json() == absent.json()
