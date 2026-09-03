"""Minimal executor used to smoke-test a local agent without model downloads."""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    request = json.loads(Path(os.environ["DGPU_REQUEST_JSON"]).read_text(encoding="utf-8"))
    output_dir = Path(os.environ["DGPU_OUTPUT_DIR"])
    result = output_dir / "result.json"
    result.write_text(
        json.dumps(
            {
                "job_id": request["id"],
                "message": request["payload"]["message"],
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    Path(os.environ["DGPU_OUTPUT_MANIFEST"]).write_text(
        json.dumps(
            {
                "files": [
                    {
                        "name": "result",
                        "path": result.name,
                        "media_type": "application/json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

