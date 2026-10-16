#!/usr/bin/env python3
"""Materialize one verified Argos model pack from a checked-in lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparehour_workloads.argos_pack import build_model_pack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_model_pack(args.lock, args.output)
    print(
        json.dumps(
            {
                "model_pack": manifest["model_pack"],
                "archive_sha256": manifest["archive_sha256"],
                "file_count": len(manifest["files"]),
                "installed_bytes": sum(item["size"] for item in manifest["files"]),
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
