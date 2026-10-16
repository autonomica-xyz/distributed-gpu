#!/usr/bin/env python3
"""End-to-end harness for real workload families on a real worker.

Drives an isolated TLS control plane (fresh DB, self-signed CA), scoped
tokens, and the real ``sparehour-worker`` CLI (``sparehour-worker run --no-dashboard`` against a
harness-owned config). Uploads declared job inputs, downloads output
artifacts, verifies digests and case expectations, and emits a JSON evidence
report. No mocks: every job runs the registered executor.

Subcommands:
  run     bring up controller + worker, execute cases, write report
  verify  re-verify downloaded output files against expectations

Case file schema (--cases):
  {
    "name": "jfk-parakeet",
    "workload": "transcribe",
    "route": "parakeet",                       # optional; default route
    "payload": {"audio_hours": 0.003, "language": "en"},
    "inputs": {"audio": "fixtures/jfk.wav"},   # optional local files
    "expected_strings": ["ask not what your country can do for you"],
    "expected_phrases": ["..."],               # substring match on decoded text
    "expected_image_dims": [[1024, 1024]]      # for image outputs
  }

The parent supplies runtime model paths/endpoints through --worker-env and
the workload registry through --workloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import time
from typing import Any

import httpx

# The harness exercises the checkout under test, so prefer repo src over any
# installed copy when the module is not already importable from cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sparehour.registry import WorkloadRegistry  # noqa: E402
from sparehour.store import Store  # noqa: E402

TERMINAL = {"succeeded", "dead-letter", "cancelled"}

_CONTROLLER_MAIN = """
import os, uvicorn
from sparehour.api import create_app
from sparehour.artifacts import ArtifactStore
from sparehour.registry import WorkloadRegistry
from sparehour.service import Orchestrator
from sparehour.store import Store
store = Store(os.environ["SPAREHOUR_E2E_DB"])
orchestrator = Orchestrator(
    WorkloadRegistry.from_directory(os.environ["SPAREHOUR_E2E_WORKLOADS"]), store)
app = create_app(orchestrator, ArtifactStore(os.environ["SPAREHOUR_E2E_ARTIFACTS"]),
                 enable_dashboard=False)
uvicorn.run(app, host="127.0.0.1", port=int(os.environ["SPAREHOUR_E2E_PORT"]),
            ssl_certfile=os.environ["SPAREHOUR_E2E_CERT"],
            ssl_keyfile=os.environ["SPAREHOUR_E2E_KEY"], log_level="warning")
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_url(client: httpx.Client, path: str, timeout: float, proc=None) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"process exited early with {proc.returncode}")
        try:
            response = client.get(path)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"timed out waiting for {path}")


def _make_ca(tls_dir: Path) -> tuple[Path, Path]:
    """Self-signed 127.0.0.1 certificate for the isolated control plane."""
    cert, key = tls_dir / "server.crt", tls_dir / "server.key"
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key), "-out", str(cert), "-days", "2",
            "-subj", "/CN=127.0.0.1",
            "-addext", "subjectAltName=IP:127.0.0.1,DNS:localhost",
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _decode_output_text(path: Path, media_type: str) -> str:
    """Pull searchable text out of a downloaded output artifact."""
    raw = path.read_bytes()
    if media_type == "application/json" or path.suffix == ".json":
        try:
            document = json.loads(raw)
        except json.JSONDecodeError:
            return raw.decode("utf-8", errors="replace")
        return json.dumps(document, ensure_ascii=False)
    return raw.decode("utf-8", errors="replace")


def _verify_outputs(
    files: list[dict[str, Any]], case: dict[str, Any]
) -> list[dict[str, Any]]:
    """Check downloaded outputs against declared digests and expectations.

    Expectations are opt-in per case: ``expected_strings``/``expected_phrases``
    are substring matches across decoded output text (phrase entries match
    case-insensitively and tolerate punctuation collapse), and
    ``expected_image_dims`` requires a real image decode of the declared
    dimensions.
    """
    results = []
    corpus = ""
    for item in files:
        corpus += "\n" + _decode_output_text(
            Path(item["downloaded_to"]), item.get("media_type", "")
        )
    folded = " ".join(
        "".join(ch for ch in corpus.casefold() if ch.isalnum() or ch.isspace()).split()
    )
    for phrase in case.get("expected_strings") or []:
        results.append(
            {"check": "string", "value": phrase, "ok": phrase in corpus}
        )
    for phrase in case.get("expected_phrases") or []:
        needle = " ".join(
            "".join(c for c in phrase.casefold() if c.isalnum() or c.isspace()).split()
        )
        results.append({"check": "phrase", "value": phrase, "ok": needle in folded})
    for dims in case.get("expected_image_dims") or []:
        matched = False
        for item in files:
            try:
                from PIL import Image

                with Image.open(item["downloaded_to"]) as image:
                    matched = list(image.size) == list(dims)
            except Exception:
                continue
            if matched:
                break
        results.append({"check": "image_dims", "value": dims, "ok": matched})
    for item in files:
        digest, size = _sha256(Path(item["downloaded_to"]))
        results.append(
            {
                "check": "sha256",
                "value": item["name"],
                "ok": digest == item["sha256"] and size == item["size"],
            }
        )
    return results


def _load_cases(path: Path) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    cases = document if isinstance(document, list) else document["cases"]
    for index, case in enumerate(cases):
        for key in ("name", "workload", "payload"):
            if key not in case:
                raise ValueError(f"cases[{index}] is missing {key!r}")
    return cases


def _agent_requirements(
    registry: WorkloadRegistry, cases: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    """Capability/model-pack sets the worker must advertise for the cases."""
    capabilities: list[str] = []
    packs: list[str] = []
    for case in cases:
        workload = registry.get(case["workload"])
        route = workload.routes[case.get("route") or workload.default_route]
        capability = route.compatibility.capability
        if capability not in capabilities:
            capabilities.append(capability)
        for pack in route.compatibility.model_packs:
            if pack not in packs:
                packs.append(pack)
    return capabilities, packs


def _submit_case(
    client: httpx.Client, project: str, case: dict[str, Any]
) -> dict[str, Any]:
    inputs = []
    for name, raw_path in (case.get("inputs") or {}).items():
        source = Path(raw_path)
        if not source.is_file():
            raise RuntimeError(f"input file missing for {case['name']}: {source}")
        media_type = "application/octet-stream"
        suffixes = {".wav": "audio/wav", ".flac": "audio/flac",
                    ".mp3": "audio/mpeg", ".mp4": "audio/mp4",
                    ".json": "application/json"}
        media_type = suffixes.get(source.suffix.lower(), media_type)
        response = client.post(
            f"/v1/projects/{project}/inputs",
            params={"workload": case["workload"], "name": name},
            content=source.read_bytes(),
            headers={"Content-Type": media_type},
        )
        if response.status_code >= 300:
            raise RuntimeError(
                f"input upload for {case['name']} failed: {response.status_code} "
                f"{response.text[:300]}"
            )
        inputs.append(response.json())
    response = client.post(
        "/v1/jobs",
        json={
            "project_id": project,
            "workload": case["workload"],
            "route": case.get("route"),
            "payload": case["payload"],
            "inputs": [
                {"name": item["name"], "uri": item["uri"], "sha256": item["sha256"],
                 "size": item["size"], "media_type": item["media_type"]}
                for item in inputs
            ],
        },
    )
    if response.status_code >= 300:
        raise RuntimeError(
            f"submit for {case['name']} failed: {response.status_code} "
            f"{response.text[:300]}"
        )
    return response.json()


def _wait_job(client: httpx.Client, job_id: str, timeout: float,
              cancel_after_running: bool = False) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    cancel_sent = False
    while time.monotonic() < deadline:
        response = client.get(f"/v1/jobs/{job_id}")
        job = response.json()
        if cancel_after_running and job["state"] == "running" and not cancel_sent:
            # Let the executor start; cancelling only the scheduling transition
            # would not exercise interruption of a running process/container.
            time.sleep(1.0)
            client.delete(f"/v1/jobs/{job_id}").raise_for_status()
            cancel_sent = True
        if job["state"] in TERMINAL:
            return job
        time.sleep(0.5)
    raise RuntimeError(f"job {job_id} did not reach a terminal state in {timeout}s")


def _download_outputs(
    client: httpx.Client, job: dict[str, Any], destination_dir: Path
) -> list[dict[str, Any]]:
    files = []
    for output in job.get("outputs") or []:
        uri = output["uri"]
        response = client.get(uri)
        if response.status_code >= 300:
            raise RuntimeError(
                f"output download {uri} failed: HTTP {response.status_code}"
            )
        target = destination_dir / f"{job['id']}-{output['name']}"
        target.write_bytes(response.content)
        files.append({**output, "downloaded_to": str(target)})
    return files


def _job_events(client: httpx.Client, job_id: str) -> list[dict[str, Any]]:
    response = client.get(f"/v1/jobs/{job_id}/events")
    if response.status_code >= 300:
        return []
    return [
        {"event_type": row["event_type"], "actor": row.get("actor")}
        for row in response.json()
    ]


def _worker_config(
    args: argparse.Namespace,
    endpoint: str,
    token: str,
    capabilities: list[str],
    packs: list[str],
    config_path: Path,
) -> Path:
    """Write the harness-owned sparehour-worker config (mode 0600, private token)."""
    dynamic = bool(args.dynamic)
    config = {
        "endpoint": endpoint,
        "token": token,
        "agent_id": args.agent_id,
        "provider": "e2e-local",
        # Dynamic mode starts unadvertised; the manager provisions
        # capabilities/packs from the lock catalog as queue depth appears.
        "capabilities": [] if dynamic else list(capabilities),
        "model_packs": [] if dynamic else list(packs),
        "labels": {},
        "cpu_slots": int(args.cpu_slots),
        "idle_only": False,
        "dynamic": dynamic,
        "dynamic_poll_seconds": int(args.dynamic_poll_seconds),
        "model_catalog": args.model_catalog or "",
        "workdir": str(Path(args.root).resolve() / "workdirs" / "sparehour-worker"),
        "dashboard": False,
        "heartbeat_seconds": int(args.heartbeat_seconds),
        "lease_seconds": int(args.lease_seconds),
        "poll_seconds": float(args.poll_seconds),
    }
    descriptor = os.open(
        config_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    return config_path


def _spawn_worker(
    args: argparse.Namespace, config_path: Path, log: Any
) -> subprocess.Popen:
    """Run the real sparehour-worker CLI: ``sparehour-worker --config <cfg> run --no-dashboard``.

    sparehour-worker ships no ``__main__``, so the module entrypoint is invoked
    explicitly with the provided ``--worker-python`` interpreter.
    """
    command = [
        str(args.worker_python),
        "-c",
        "from sparehour_worker.cli import main; raise SystemExit(main())",
        "--config",
        str(config_path),
        "run",
        "--no-dashboard",
    ]
    env = dict(os.environ)
    env.update(args.worker_env or {})
    env["SSL_CERT_FILE"] = str(args.worker_ca)
    srcs = [str(REPO_ROOT / "src")]
    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        os.pathsep.join(srcs)
        if not python_path
        else f"{os.pathsep.join(srcs)}{os.pathsep}{python_path}"
    )
    return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)


def cmd_run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    tls_dir = root / "tls"
    tls_dir.mkdir(exist_ok=True)
    args.cert, _key = _make_ca(tls_dir)
    # The worker must trust the test control plane AND real public endpoints
    # it may download models from (HF hub etc.), so it gets certifi roots +
    # the test cert — never the test cert alone. The harness's own clients
    # verify only the test cert, keeping the controller isolated.
    import certifi

    args.worker_ca = tls_dir / "combined.pem"
    args.worker_ca.write_bytes(
        Path(certifi.where()).read_bytes() + args.cert.read_bytes()
    )
    db_path = root / "control.sqlite3"
    artifacts = root / "artifacts"
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    report_path = Path(args.report or root / "report.json")

    cases = _load_cases(Path(args.cases))
    registry = WorkloadRegistry.from_directory(args.workloads)
    capabilities, packs = _agent_requirements(registry, cases)

    port = _free_port()
    endpoint = f"https://127.0.0.1:{port}"
    evidence: dict[str, Any] = {
        "endpoint": endpoint,
        "root": str(root),
        "cases": [],
        "scenarios": [],
        "verifier_results": [],
        "ok": True,
    }

    ssl_context = ssl.create_default_context(cafile=str(args.cert))
    client = httpx.Client(
        base_url=endpoint, verify=ssl_context, timeout=60.0, trust_env=False
    )

    store = Store(db_path)
    admin = store.issue_token(role="admin", subject="e2e-operator")
    client_token = store.issue_token(
        role="client",
        subject=args.project,
        scopes={"projects": [args.project], "workloads": sorted(
            {case["workload"] for case in cases}
        )},
    )
    agent_scopes: dict[str, Any] = {"routes": capabilities}
    if args.dynamic:
        # Queue-depth visibility does not widen the worker's route grants.
        agent_scopes["perms"] = ["dynamic"]
    agent_token = store.issue_token(
        role="agent", subject=args.agent_id, scopes=agent_scopes
    )
    token_dir = root / "tokens"
    token_dir.mkdir(exist_ok=True)
    for label, raw in (("admin", admin), ("client", client_token), ("agent", agent_token)):
        path = token_dir / f"{label}.token"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(raw)
    store.close()

    client.headers["Authorization"] = f"Bearer {client_token}"
    admin_client = httpx.Client(
        base_url=endpoint,
        verify=ssl_context,
        timeout=60.0,
        headers={"Authorization": f"Bearer {admin}"},
        trust_env=False,
    )

    server_env = dict(os.environ)
    server_env.update(
        {
            "SPAREHOUR_E2E_DB": str(db_path),
            "SPAREHOUR_E2E_WORKLOADS": str(Path(args.workloads).resolve()),
            "SPAREHOUR_E2E_ARTIFACTS": str(artifacts),
            "SPAREHOUR_E2E_PORT": str(port),
            "SPAREHOUR_E2E_CERT": str(args.cert),
            "SPAREHOUR_E2E_KEY": str(tls_dir / "server.key"),
            "PYTHONPATH": str(REPO_ROOT / "src"),
        }
    )
    server_log = (root / "controller.log").open("w", encoding="utf-8")
    server = subprocess.Popen(
        [sys.executable, "-c", _CONTROLLER_MAIN],
        stdout=server_log,
        stderr=subprocess.STDOUT,
        env=server_env,
    )
    agent: subprocess.Popen | None = None
    agent_log = None
    try:
        _wait_url(admin_client, "/healthz", timeout=30.0, proc=server)
        evidence["controller_tls"] = True

        config_path = _worker_config(
            args,
            endpoint,
            agent_token,
            capabilities,
            packs,
            root / "sparehour-worker-config.json",
        )
        agent_log = (root / "agent.log").open("w", encoding="utf-8")
        agent = _spawn_worker(args, config_path, agent_log)
        # Wait until the slot registers before submitting.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if agent.poll() is not None:
                raise RuntimeError(
                    f"worker exited early with {agent.returncode}; see agent.log"
                )
            rows = admin_client.get("/v1/control/agents").json()
            if any(
                slot["state"] == "ready"
                for row in rows
                for slot in row.get("slots", [])
            ):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("worker never registered a ready slot")

        iterations = max(1, int(args.repeat))
        for _iteration in range(iterations):
            if args.mixed_queue:
                submitted = [
                    (case, _submit_case(client, args.project, case))
                    for case in cases
                ]
                for case, job in submitted:
                    evidence["cases"].append(
                        _finish_case(client, case, job, outputs_dir, args)
                    )
            else:
                for case in cases:
                    job = _submit_case(client, args.project, case)
                    evidence["cases"].append(
                        _finish_case(client, case, job, outputs_dir, args)
                    )
            # Every repeat reuses the same warm worker process.

        if args.cancel_running:
            case = cases[0]
            job = _submit_case(client, args.project, case)
            final = _wait_job(
                client, job["id"], args.job_timeout, cancel_after_running=True
            )
            evidence["scenarios"].append(
                {
                    "scenario": "cancel_running",
                    "job_id": job["id"],
                    "final_state": final["state"],
                    "events": _job_events(client, job["id"]),
                    "ok": final["state"] == "cancelled",
                }
            )

        if args.recovery:
            case = cases[0]
            job = _submit_case(client, args.project, case)
            # Wait for the job to go running, then kill the worker mid-flight.
            deadline = time.monotonic() + args.job_timeout
            while time.monotonic() < deadline:
                state = client.get(f"/v1/jobs/{job['id']}").json()["state"]
                if state == "running":
                    break
                if state in TERMINAL:
                    raise RuntimeError("recovery job finished before kill")
                time.sleep(0.3)
            agent.kill()
            agent.wait(timeout=10)
            # The lease expires; the scheduler requeues the job.
            expiry = time.monotonic() + args.lease_seconds + 15
            while time.monotonic() < expiry:
                reap = admin_client.post("/v1/control/reap-expired")
                if reap.json().get("retried") or reap.json().get("released"):
                    break
                time.sleep(1.0)
            if agent_log is not None:
                agent_log.close()
            agent_log = (root / "agent-recovery.log").open("w", encoding="utf-8")
            agent = _spawn_worker(args, config_path, agent_log)
            final = _wait_job(client, job["id"], args.job_timeout)
            recovered_outputs = _download_outputs(client, final, outputs_dir)
            recovered_checks = _verify_outputs(recovered_outputs, case)
            evidence["scenarios"].append(
                {
                    "scenario": "lease_recovery",
                    "job_id": job["id"],
                    "final_state": final["state"],
                    "attempts": admin_client.get(
                        f"/v1/jobs/{job['id']}/attempts"
                    ).json(),
                    "events": _job_events(client, job["id"]),
                    "outputs": recovered_outputs,
                    "checks": recovered_checks,
                    "ok": final["state"] == "succeeded"
                    and final["attempts"] >= 2
                    and all(check["ok"] for check in recovered_checks),
                }
            )
    except Exception as exc:
        evidence["ok"] = False
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for item in evidence["cases"]:
            checks = item.pop("checks", [])
            evidence["verifier_results"].extend(checks)
            if item.get("state") != "succeeded" or not all(
                check["ok"] for check in checks
            ):
                item["ok"] = False
                evidence["ok"] = False
            else:
                item["ok"] = True
        for scenario in evidence["scenarios"]:
            if not scenario.get("ok"):
                evidence["ok"] = False
        if agent is not None:
            agent.terminate()
            try:
                agent.wait(timeout=10)
            except subprocess.TimeoutExpired:
                agent.kill()
                agent.wait(timeout=10)
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
        if agent_log is not None:
            agent_log.close()
        server_log.close()
        report_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(json.dumps(evidence, indent=2))
    return 0 if evidence["ok"] else 1


def _finish_case(
    client: httpx.Client,
    case: dict[str, Any],
    job: dict[str, Any],
    outputs_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    final = _wait_job(client, job["id"], args.job_timeout)
    record: dict[str, Any] = {
        "case": case["name"],
        "workload": case["workload"],
        "route": case.get("route"),
        "job_id": job["id"],
        "state": final["state"],
        "attempts": final["attempts"],
        "error_code": final.get("error_code"),
        "error_message": final.get("error_message"),
        "events": _job_events(client, job["id"]),
        "checks": [],
    }
    if final["state"] == "succeeded":
        files = _download_outputs(client, final, outputs_dir)
        # downloaded_to stays in evidence so `workload_e2e verify` can
        # re-check the artifacts offline.
        record["outputs"] = files
        record["checks"] = _verify_outputs(files, case)
    return record


def cmd_verify(args: argparse.Namespace) -> int:
    """Re-check downloaded output files for one case offline."""
    case_files = [
        {
            "name": item["name"],
            "sha256": item["sha256"],
            "size": item["size"],
            "media_type": item["media_type"],
            "downloaded_to": item["downloaded_to"],
        }
        for item in json.loads(Path(args.outputs).read_text(encoding="utf-8"))
    ]
    case = json.loads(Path(args.case).read_text(encoding="utf-8"))
    results = _verify_outputs(case_files, case)
    print(json.dumps(results, indent=2))
    return 0 if all(item["ok"] for item in results) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="workload_e2e", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the full isolated lifecycle")
    run.add_argument("--root", required=True, help="owned scratch directory")
    run.add_argument("--workloads", required=True,
                     help="workload registry directory under test")
    run.add_argument("--cases", required=True, help="JSON case file")
    run.add_argument("--project", default="e2e-project")
    run.add_argument("--agent-id", default="e2e-agent")
    run.add_argument("--worker-python", default=sys.executable)
    run.add_argument("--worker-env", type=json.loads, default={},
                     help="JSON object merged into the worker environment "
                          "(route model dirs/endpoints)")
    run.add_argument("--cpu-slots", type=int, default=0,
                     help="vendor=cpu slots for CPU routes (sparehour-worker cpu_slots)")
    run.add_argument("--dynamic", action="store_true",
                     help="enable sparehour-worker dynamic provisioning: worker starts "
                          "with no advertised capabilities and selects from "
                          "--model-catalog as queue depth appears")
    run.add_argument("--dynamic-poll-seconds", type=int, default=5)
    run.add_argument("--model-catalog", default=None,
                     help="model lock catalog path for dynamic provisioning")
    run.add_argument("--repeat", type=int, default=1,
                     help="run all cases N times against the same warm worker")
    run.add_argument("--mixed-queue", action="store_true",
                     help="submit all cases before waiting (queue mixing)")
    run.add_argument("--cancel-running", action="store_true",
                     help="cancel a running job and require cancelled state")
    run.add_argument("--recovery", action="store_true",
                     help="kill the worker mid-job and require lease recovery")
    run.add_argument("--job-timeout", type=float, default=1800.0)
    run.add_argument("--heartbeat-seconds", type=int, default=5)
    run.add_argument("--lease-seconds", type=int, default=30)
    run.add_argument("--poll-seconds", type=float, default=0.5)
    run.add_argument("--report", default=None, help="report output path")
    run.set_defaults(func=cmd_run)

    verify = sub.add_parser("verify", help="verify downloaded outputs offline")
    verify.add_argument("--outputs", required=True,
                        help="JSON list of {name,sha256,size,media_type,downloaded_to}")
    verify.add_argument("--case", required=True, help="single case JSON file")
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
