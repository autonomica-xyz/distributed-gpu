#!/usr/bin/env python3
"""Build every verified model directory in one locked translation shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpu_orchestrator_workloads.argos_pack import build_model_pack_from_lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-cache", type=Path)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    try:
        models = catalog["shards"][args.shard]
    except KeyError as exc:
        raise ValueError(f"unknown locked shard {args.shard!r}") from exc
    if args.output.exists():
        raise FileExistsError(f"destination already exists: {args.output}")
    args.output.mkdir(parents=True)
    built = []
    try:
        for lock in models:
            pair = f"{lock['from_code']}_{lock['to_code']}"
            archive = (
                args.archive_cache / lock["archive_filename"]
                if args.archive_cache is not None
                else None
            )
            manifest = build_model_pack_from_lock(
                lock, args.output / pair, archive_path=archive
            )
            built.append(
                {
                    "pair": pair,
                    "model_pack": manifest["model_pack"],
                    "installed_bytes": sum(item["size"] for item in manifest["files"]),
                }
            )
    except Exception:
        # Keep successfully built directories for forensic inspection. The
        # all-or-nothing image build only starts after this command succeeds.
        raise
    print(json.dumps({"shard": args.shard, "models": built}, indent=2))


if __name__ == "__main__":
    main()
