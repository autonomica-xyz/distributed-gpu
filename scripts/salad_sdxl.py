#!/usr/bin/env python3
"""Create, inspect, and safely destroy one Salad SDXL worker group."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse


from gpu_orchestrator.models import ResourceRequirements
from gpu_orchestrator.providers import (
    SaladProvider,
    WorkerLaunchSpec,
    destroy_and_verify_absent,
)


def _provider(args: argparse.Namespace) -> SaladProvider:
    key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError("Salad API key file is empty")
    return SaladProvider(key, args.organization, args.project)


def _wheel_source(args: argparse.Namespace) -> tuple[str, str]:
    """Return ``(url, sha256)`` for the bootstrap wheel.

    The URL must be http(s) with no userinfo or fragment — the worker
    command appends ``#sha256=<digest>`` itself so pip verifies the wheel
    it downloads. The digest comes from ``--wheel-sha256`` or by hashing
    ``--wheel-file``.
    """
    url = str(args.wheel_url).strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("--wheel-url must be an http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError("--wheel-url must not embed credentials")
    if parsed.fragment:
        raise ValueError("--wheel-url must not carry a fragment")
    if args.wheel_sha256:
        digest = args.wheel_sha256.strip().lower()
    elif args.wheel_file:
        digest = hashlib.sha256(Path(args.wheel_file).read_bytes()).hexdigest()
    else:
        raise ValueError("create requires --wheel-sha256 or --wheel-file")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("wheel sha256 must be 64 lowercase hex characters")
    return url, digest



def create(args: argparse.Namespace) -> None:
    provider = _provider(args)
    template = json.loads(Path(args.template).read_text(encoding="utf-8"))
    image = template["container"]["image"]
    command = list(template["container"]["command"])
    token = Path(args.agent_token_file).read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("agent token file is empty")
    wheel_url, wheel_sha256 = _wheel_source(args)
    environment = {
        "DGPU_ENDPOINT": args.endpoint.rstrip("/"),
        "DGPU_WHEEL_URL": wheel_url,
        "DGPU_WHEEL_SHA256": wheel_sha256,
        "DGPU_AGENT_ID": args.name,
        "DGPU_PROVIDER_RESOURCE_ID": args.name,
        "DGPU_TOKEN": token,
        "HF_HOME": "/models/huggingface",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONUNBUFFERED": "1",
    }
    resource = provider.launch(
        None,
        WorkerLaunchSpec(
            name=args.name,
            worker_id=args.name,
            image=image,
            command=command,
            environment=environment,
            disk_gb=30,
            resources=ResourceRequirements(
                min_vram_mb=16384,
                min_cpu_cores=4,
                min_ram_mb=12288,
                min_disk_mb=20480,
                allowed_providers=["salad"],
            ),
            provider_template=template,
        ),
    )
    print(
        json.dumps(
            {
                "provider": resource.provider,
                "resource_id": resource.resource_id,
                "name": resource.name,
                "state": resource.state,
                "worker_id": resource.worker_id,
            },
            indent=2,
            sort_keys=True,
        )
    )


def status(args: argparse.Namespace) -> None:
    resources = [
        item
        for item in _provider(args).inventory()
        if item.name == args.name or item.resource_id == args.name
    ]
    print(
        json.dumps(
            [
                {
                    "provider": item.provider,
                    "resource_id": item.resource_id,
                    "name": item.name,
                    "state": item.state,
                    "worker_id": item.worker_id,
                }
                for item in resources
            ],
            indent=2,
            sort_keys=True,
        )
    )


def destroy(args: argparse.Namespace) -> None:
    provider = _provider(args)
    destroy_and_verify_absent(
        provider,
        args.name,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=2,
    )
    print(json.dumps({"name": args.name, "verified_absent": True}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--api-key-file", default="/run/secrets/salad-api-key")
    result.add_argument("--organization", required=True)
    result.add_argument("--project", default="default")
    result.add_argument("--name", required=True)
    sub = result.add_subparsers(dest="command", required=True)

    create_parser = sub.add_parser("create")
    create_parser.add_argument("--endpoint", required=True)
    create_parser.add_argument("--wheel-url", required=True)
    create_parser.add_argument(
        "--wheel-sha256",
        help="SHA-256 of the bootstrap wheel; the worker pip-installs "
        "--wheel-url with a #sha256= fragment so the digest is verified.",
    )
    create_parser.add_argument(
        "--wheel-file",
        help="local wheel to hash instead of passing --wheel-sha256",
    )
    create_parser.add_argument("--agent-token-file", required=True)
    create_parser.add_argument(
        "--template", default="deploy/salad-sdxl-worker.json"
    )
    create_parser.set_defaults(func=create)

    status_parser = sub.add_parser("status")
    status_parser.set_defaults(func=status)

    destroy_parser = sub.add_parser("destroy")
    destroy_parser.add_argument("--timeout-seconds", type=float, default=180)
    destroy_parser.set_defaults(func=destroy)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
