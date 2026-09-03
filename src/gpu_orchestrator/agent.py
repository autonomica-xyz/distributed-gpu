"""GPU host agent and trusted executor boundary.

One agent process registers one slot per physical GPU. Each slot claims and
executes jobs independently, so a four-GPU host naturally runs up to four jobs
without requiring the jobs themselves to know about distributed execution.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import httpx

from .models import AgentRegistration, SlotInventory


class AgentError(RuntimeError):
    pass


class ExecutionError(AgentError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def discover_nvidia_slots(
    agent_id: str,
    *,
    capabilities: Sequence[str],
    model_packs: Sequence[str],
    labels: dict[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[SlotInventory]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = runner(command, text=True, capture_output=True, check=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        raise AgentError(f"could not discover NVIDIA GPUs with nvidia-smi: {exc}") from exc
    slots = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise AgentError(f"unexpected nvidia-smi row: {line!r}")
        index, gpu_uuid, model, vram_mb, compute_capability = parts
        slots.append(
            SlotInventory(
                id=f"{agent_id}-gpu-{int(index)}",
                gpu_index=int(index),
                gpu_uuid=gpu_uuid,
                model=model,
                vram_mb=int(float(vram_mb)),
                compute_capability=compute_capability,
                capabilities=list(capabilities),
                model_packs=list(model_packs),
                labels=labels or {},
            )
        )
    if not slots:
        raise AgentError("nvidia-smi returned no GPUs")
    return slots


class ControlPlaneClient:
    def __init__(self, endpoint: str, token: str, *, timeout: float = 60.0):
        self.client = httpx.Client(
            base_url=endpoint.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {token}"},
        )

    def close(self) -> None:
        self.client.close()

    def _json(self, response: httpx.Response, operation: str) -> Any:
        if response.status_code < 200 or response.status_code >= 300:
            raise AgentError(f"{operation} failed: HTTP {response.status_code}: {response.text[:1000]}")
        return response.json() if response.content else None

    def register(self, registration: AgentRegistration) -> dict[str, Any]:
        return self._json(
            self.client.post("/v1/agents/register", json=registration.model_dump(mode="json")),
            "agent registration",
        )

    def heartbeat_agent(self, agent_id: str, *, draining: bool | None = None) -> None:
        body = {} if draining is None else {"draining": draining}
        self._json(
            self.client.post(f"/v1/agents/{agent_id}/heartbeat", json=body),
            "agent heartbeat",
        )

    def claim(self, slot_id: str, lease_seconds: int) -> dict[str, Any] | None:
        return self._json(
            self.client.post(
                f"/v1/slots/{slot_id}/claim", json={"lease_seconds": lease_seconds}
            ),
            "claim",
        )

    def confirm(self, job_id: str, slot_id: str, lease_token: str) -> None:
        self._lease_post(job_id, "confirm", slot_id, lease_token)

    def heartbeat_job(
        self, job_id: str, slot_id: str, lease_token: str, lease_seconds: int
    ) -> bool:
        result = self._lease_post(
            job_id,
            "heartbeat",
            slot_id,
            lease_token,
            params={"lease_seconds": lease_seconds},
        )
        return bool(result.get("cancel_requested"))

    def complete(
        self, job_id: str, slot_id: str, lease_token: str, outputs: list[dict[str, Any]]
    ) -> None:
        self._lease_post(
            job_id, "complete", slot_id, lease_token, json_body={"outputs": outputs}
        )

    def upload_output(
        self,
        job_id: str,
        slot_id: str,
        lease_token: str,
        *,
        name: str,
        path: Path,
        media_type: str,
    ) -> dict[str, Any]:
        with path.open("rb") as handle:
            response = self.client.put(
                f"/v1/jobs/{job_id}/outputs/{name}",
                params={"slot_id": slot_id},
                headers={
                    "X-DGPU-Lease-Token": lease_token,
                    "Content-Type": media_type,
                },
                content=handle,
            )
        return self._json(response, "output upload")

    def fail(
        self,
        job_id: str,
        slot_id: str,
        lease_token: str,
        code: str,
        message: str,
    ) -> None:
        self._lease_post(
            job_id,
            "fail",
            slot_id,
            lease_token,
            json_body={"error_code": code, "error_message": message[:4000]},
        )

    def release(
        self, job_id: str, slot_id: str, lease_token: str, reason: str
    ) -> None:
        self._lease_post(
            job_id, "release", slot_id, lease_token, json_body={"reason": reason[:500]}
        )

    def _lease_post(
        self,
        job_id: str,
        action: str,
        slot_id: str,
        lease_token: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        response = self.client.post(
            f"/v1/jobs/{job_id}/{action}",
            params={"slot_id": slot_id, **(params or {})},
            headers={"X-DGPU-Lease-Token": lease_token},
            json=json_body,
        )
        return self._json(response, action)


@dataclass(frozen=True)
class AgentConfig:
    registration: AgentRegistration
    workdir_root: Path
    heartbeat_seconds: int = 15
    lease_seconds: int = 90
    poll_seconds: float = 1.0


DRAINING_EXECUTION_ERRORS = {"model_server_unavailable", "model_load_failed"}


class AgentRunner:
    def __init__(self, client: ControlPlaneClient, config: AgentConfig):
        if config.heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        if config.heartbeat_seconds >= config.lease_seconds:
            raise ValueError("heartbeat_seconds must be shorter than lease_seconds")
        self.client = client
        self.config = config
        self.stop = threading.Event()

    def request_drain(self) -> None:
        self.stop.set()
        try:
            self.client.heartbeat_agent(self.config.registration.agent_id, draining=True)
        except AgentError:
            pass

    def run(self, *, once: bool = False) -> None:
        self.config.workdir_root.mkdir(parents=True, exist_ok=True)
        self.client.register(self.config.registration)
        # A prior process marks the durable agent row draining on SIGTERM.
        # Successful registration by a fresh runner explicitly restores
        # readiness; ordinary heartbeats intentionally preserve current state.
        self.client.heartbeat_agent(self.config.registration.agent_id, draining=False)
        previous_handlers: dict[int, Any] = {}
        heartbeat_stop = threading.Event()
        thread_errors: list[BaseException] = []

        def drain_handler(_signum, _frame):
            self.request_drain()

        def heartbeat_loop() -> None:
            while not heartbeat_stop.wait(self.config.heartbeat_seconds):
                try:
                    self.client.heartbeat_agent(self.config.registration.agent_id)
                except AgentError as exc:
                    print(
                        f"agent {self.config.registration.agent_id}: heartbeat failed: {exc}",
                        flush=True,
                    )

        def slot_main(slot: SlotInventory) -> None:
            try:
                self._slot_loop(slot, once)
            except BaseException as exc:
                thread_errors.append(exc)
                self.stop.set()

        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
                previous_handlers[sig] = signal.signal(sig, drain_handler)
        heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            name=f"heartbeat-{self.config.registration.agent_id}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            threads = [
                threading.Thread(
                    target=slot_main,
                    args=(slot,),
                    name=f"slot-{slot.id}",
                )
                for slot in self.config.registration.slots
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(
                timeout=max(1.0, float(self.config.heartbeat_seconds) + 1.0)
            )
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        if thread_errors:
            raise thread_errors[0]

    def _slot_loop(self, slot: SlotInventory, once: bool) -> None:
        while not self.stop.is_set():
            try:
                job = self.client.claim(slot.id, self.config.lease_seconds)
                if job is None:
                    if once:
                        return
                    self.stop.wait(self.config.poll_seconds)
                    continue
                self._run_job(slot, job)
                if once:
                    return
            except AgentError as exc:
                if once:
                    raise
                print(f"slot {slot.id}: {exc}", flush=True)
                self.stop.wait(self.config.poll_seconds)

    def _run_job(self, slot: SlotInventory, job: dict[str, Any]) -> None:
        job_id = job["id"]
        token = job["lease_token"]
        self.client.confirm(job_id, slot.id, token)
        workdir = Path(
            tempfile.mkdtemp(prefix=f"dgpu-{job_id}-", dir=self.config.workdir_root)
        )
        lease = _LeaseKeeper(
            self.client,
            job_id=job_id,
            slot_id=slot.id,
            lease_token=token,
            heartbeat_seconds=self.config.heartbeat_seconds,
            lease_seconds=self.config.lease_seconds,
        )
        lease.start()
        try:
            local_outputs = self._execute(slot, job, workdir, lease)
            lease.check()
            outputs = []
            for output in local_outputs:
                lease.check()
                uri = str(output["uri"])
                if not uri.startswith("file://"):
                    raise ExecutionError(
                        "artifact_transport_failed", "executor output is not a local file"
                    )
                outputs.append(
                    self.client.upload_output(
                        job_id,
                        slot.id,
                        token,
                        name=output["name"],
                        path=Path(urllib.parse.unquote(urllib.parse.urlparse(uri).path)),
                        media_type=output["media_type"],
                    )
                )
            # Keep the lease alive throughout potentially large uploads, then
            # fence completion against any cancellation or renewal failure.
            lease.stop()
            lease.check()
            self.client.complete(job_id, slot.id, token, outputs)
        except ExecutionError as exc:
            lease.stop()
            self.client.fail(job_id, slot.id, token, exc.code, str(exc))
            if exc.code in DRAINING_EXECUTION_ERRORS:
                print(
                    f"slot {slot.id}: draining after fatal worker error {exc.code}",
                    flush=True,
                )
                self.request_drain()
        except AgentError as exc:
            lease.stop()
            self.client.fail(
                job_id,
                slot.id,
                token,
                "artifact_transport_failed",
                str(exc),
            )
        except Exception as exc:
            lease.stop()
            self.client.fail(job_id, slot.id, token, "executor_failed", str(exc))
        finally:
            lease.stop()
            shutil.rmtree(workdir, ignore_errors=True)

    def _execute(
        self,
        slot: SlotInventory,
        job: dict[str, Any],
        workdir: Path,
        lease: "_LeaseKeeper",
    ) -> list[dict[str, Any]]:
        executor = job["executor"]
        executor_type = executor["type"]
        if executor_type in {"command", "container"}:
            return self._execute_process(
                slot,
                job,
                workdir,
                lease,
                container=executor_type == "container",
            )
        if executor_type == "http":
            return self._execute_http(slot, job, workdir, lease)
        if executor_type == "plugin":
            return self._execute_plugin(slot, job, workdir)
        raise ExecutionError("executor_unsupported", f"unsupported executor {executor_type!r}")

    def _execute_process(
        self,
        slot: SlotInventory,
        job: dict[str, Any],
        workdir: Path,
        lease: "_LeaseKeeper",
        *,
        container: bool,
    ) -> list[dict[str, Any]]:
        executor = job["executor"]
        request_path = workdir / "request.json"
        request_path.write_text(json.dumps(job, sort_keys=True), encoding="utf-8")
        output_manifest = workdir / "outputs.json"
        environment = {
            **os.environ,
            **executor.get("environment", {}),
            "CUDA_VISIBLE_DEVICES": str(slot.gpu_index),
            "DGPU_JOB_ID": job["id"],
            "DGPU_REQUEST_JSON": str(request_path),
            "DGPU_OUTPUT_DIR": str(workdir),
            "DGPU_OUTPUT_MANIFEST": str(output_manifest),
        }
        command = [
            sys.executable if argument == "{agent_python}" else argument
            for argument in executor["command"]
        ]
        if container:
            container_environment = []
            for name in sorted(executor.get("environment", {})):
                container_environment.extend(["-e", name])
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                f"dgpu-{job['id'][:48]}",
                "--gpus",
                f"device={slot.gpu_uuid}",
                "--mount",
                f"type=bind,src={workdir},dst=/job",
                "-e",
                "DGPU_JOB_ID",
                "-e",
                "DGPU_REQUEST_JSON=/job/request.json",
                "-e",
                "DGPU_OUTPUT_DIR=/job",
                "-e",
                "DGPU_OUTPUT_MANIFEST=/job/outputs.json",
                *container_environment,
                executor["image"],
                *executor["command"],
            ]
        stdout_path = workdir / "executor.stdout.log"
        stderr_path = workdir / "executor.stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=workdir,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                start_new_session=True,
            )
            timeout = float(executor.get("timeout_seconds") or 3600)
            deadline = time.monotonic() + timeout
            try:
                while process.poll() is None:
                    lease.check()
                    wait = min(1.0, max(0.0, deadline - time.monotonic()))
                    if wait <= 0:
                        raise ExecutionError(
                            "execution_timeout", f"executor exceeded {timeout:g}s"
                        )
                    try:
                        process.wait(timeout=wait)
                    except subprocess.TimeoutExpired:
                        continue
            except BaseException:
                _terminate_process(process)
                raise
        if process.returncode != 0:
            stdout = _tail_text(stdout_path)
            stderr = _tail_text(stderr_path)
            message = (stderr or stdout or f"exit code {process.returncode}")[-3000:]
            code = "gpu_oom" if "out of memory" in message.casefold() else "executor_failed"
            raise ExecutionError(code, message)
        return _load_output_manifest(output_manifest, workdir)

    def _execute_http(
        self,
        slot: SlotInventory,
        job: dict[str, Any],
        workdir: Path,
        lease: "_LeaseKeeper",
    ) -> list[dict[str, Any]]:
        executor = job["executor"]
        contract = job["output_contract"]
        if len(contract) != 1:
            raise ExecutionError(
                "executor_contract_invalid", "generic HTTP executor requires exactly one output"
            )
        spec = contract[0]
        path = workdir / "response.bin"
        size = 0
        try:
            with httpx.stream(
                "POST",
                executor["endpoint"],
                json=job["payload"],
                timeout=float(executor.get("timeout_seconds") or 3600),
            ) as response:
                response.raise_for_status()
                media_type = response.headers.get(
                    "content-type", "application/json"
                ).split(";", 1)[0]
                with path.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        lease.check()
                        size += len(chunk)
                        if size > int(spec["max_bytes"]):
                            raise ExecutionError(
                                "output_validation_failed",
                                f"HTTP response exceeds max_bytes={spec['max_bytes']}",
                            )
                        handle.write(chunk)
        except httpx.HTTPError as exc:
            raise ExecutionError("model_server_unavailable", str(exc)) from exc
        return [_artifact_from_path(spec["name"], path, media_type)]

    def _execute_plugin(
        self, slot: SlotInventory, job: dict[str, Any], workdir: Path
    ) -> list[dict[str, Any]]:
        module_name, separator, attribute = job["executor"]["handler"].partition(":")
        if not separator:
            raise ExecutionError("executor_contract_invalid", "plugin handler must be module:attribute")
        try:
            handler = getattr(importlib.import_module(module_name), attribute)
            result = handler().execute(job=job, workdir=workdir, gpu_index=slot.gpu_index)
        except (ImportError, AttributeError) as exc:
            raise ExecutionError("executor_unavailable", str(exc)) from exc
        return [
            _artifact_from_path(item["name"], Path(item["path"]), item["media_type"])
            for item in result
        ]


class _LeaseKeeper:
    """Renew a job lease while any trusted executor or upload is active."""

    def __init__(
        self,
        client: ControlPlaneClient,
        *,
        job_id: str,
        slot_id: str,
        lease_token: str,
        heartbeat_seconds: int,
        lease_seconds: int,
    ):
        self.client = client
        self.job_id = job_id
        self.slot_id = slot_id
        self.lease_token = lease_token
        self.heartbeat_seconds = heartbeat_seconds
        self.lease_seconds = lease_seconds
        self._stop = threading.Event()
        self._cancelled = threading.Event()
        self._failure: AgentError | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"lease-{job_id}",
            daemon=True,
        )
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._started and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(1.0, float(self.heartbeat_seconds) + 1.0))

    def check(self) -> None:
        if self._cancelled.is_set():
            raise ExecutionError("cancelled", "job cancellation was requested")
        if self._failure is not None:
            raise ExecutionError("worker_lost", f"job lease renewal failed: {self._failure}")

    def _run(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            try:
                cancelled = self.client.heartbeat_job(
                    self.job_id,
                    self.slot_id,
                    self.lease_token,
                    self.lease_seconds,
                )
            except AgentError as exc:
                self._failure = exc
                self._stop.set()
                return
            if cancelled:
                self._cancelled.set()
                self._stop.set()
                return


def _load_output_manifest(path: Path, workdir: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ExecutionError("output_validation_failed", "executor did not create outputs.json")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        files = document["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ExecutionError("output_validation_failed", f"invalid outputs.json: {exc}") from exc
    outputs = []
    for item in files:
        candidate = Path(item["path"])
        candidate = candidate if candidate.is_absolute() else workdir / candidate
        try:
            candidate.resolve().relative_to(workdir.resolve())
        except ValueError as exc:
            raise ExecutionError("output_validation_failed", "output path escapes workdir") from exc
        outputs.append(_artifact_from_path(item["name"], candidate, item["media_type"]))
    return outputs


def _artifact_from_path(name: str, path: Path, media_type: str) -> dict[str, Any]:
    if not path.is_file():
        raise ExecutionError("output_validation_failed", f"output file is missing: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return {
        "name": name,
        "uri": path.resolve().as_uri(),
        "media_type": media_type,
        "size": size,
        "sha256": digest.hexdigest(),
    }


def _tail_text(path: Path, max_bytes: int = 16 * 1024) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def _terminate_process(process: subprocess.Popen[str], grace_seconds: float = 10.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace_seconds)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        else:
            process.wait()
