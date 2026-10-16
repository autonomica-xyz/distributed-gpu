"""Content-addressed artifact storage with workload-defined validation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, AsyncIterable

from jsonschema import Draft202012Validator
from PIL import Image


class ArtifactError(ValueError):
    pass


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.blobs = self.root / "sha256"
        self.temporary = self.root / ".tmp"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.temporary.mkdir(parents=True, exist_ok=True)

    def path_for(self, sha256: str) -> Path:
        if len(sha256) != 64 or any(ch not in "0123456789abcdef" for ch in sha256):
            raise ArtifactError("invalid sha256")
        return self.blobs / sha256[:2] / sha256[2:4] / sha256

    async def ingest(
        self,
        chunks: AsyncIterable[bytes],
        *,
        max_bytes: int,
        validator: str,
        json_schema: dict[str, Any] | None = None,
    ) -> tuple[str, int, Path]:
        descriptor, temporary_name = tempfile.mkstemp(prefix="upload-", dir=self.temporary)
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(descriptor, "wb") as handle:
                async for chunk in chunks:
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > max_bytes:
                        raise ArtifactError(f"artifact exceeds max_bytes={max_bytes}")
                    digest.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if size <= 0:
                raise ArtifactError("artifact is empty")
            self.validate(temporary, validator=validator, json_schema=json_schema)
            sha256 = digest.hexdigest()
            destination = self.path_for(sha256)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(temporary, destination)
            except FileExistsError:
                pass
            except OSError:
                # Cross-filesystem configurations still get an atomic final
                # rename because the fallback temp is placed beside the blob.
                local_tmp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
                shutil.copyfile(temporary, local_tmp)
                os.replace(local_tmp, destination)
            return sha256, size, destination
        finally:
            temporary.unlink(missing_ok=True)

    def validate(
        self,
        path: Path,
        *,
        validator: str,
        json_schema: dict[str, Any] | None = None,
    ) -> None:
        if validator in {"none", "sha256"}:
            return
        if validator in {"json", "json_schema"}:
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ArtifactError(f"artifact is not valid JSON: {exc}") from exc
            if validator == "json_schema":
                if json_schema is None:
                    raise ArtifactError("json_schema validator has no schema")
                errors = sorted(
                    Draft202012Validator(json_schema).iter_errors(document),
                    key=lambda error: list(error.absolute_path),
                )
                if errors:
                    raise ArtifactError(f"artifact JSON schema failed: {errors[0].message}")
            return
        if validator == "image":
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                raise ArtifactError(f"artifact is not a decodable image: {exc}") from exc
            return
        if validator == "media_probe":
            try:
                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration,format_name",
                        "-of",
                        "json",
                        str(path),
                    ],
                    text=True,
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
            except OSError as exc:
                raise ArtifactError(f"ffprobe is unavailable: {exc}") from exc
            if result.returncode != 0:
                raise ArtifactError(f"ffprobe rejected artifact: {result.stderr[-1000:]}")
            return
        raise ArtifactError(f"unknown artifact validator {validator!r}")

