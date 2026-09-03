"""Build a verified runtime model directory from a locked Argos archive."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_EXTRACTED_BYTES = 1024 * 1024 * 1024
MAX_FILES = 10000


def build_model_pack(
    lock_path: Path, destination: Path, *, archive_path: Path | None = None
) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    return build_model_pack_from_lock(lock, destination, archive_path=archive_path)


def build_model_pack_from_lock(
    lock: dict[str, Any], destination: Path, *, archive_path: Path | None = None
) -> dict[str, Any]:
    expected = lock["sha256"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")

    temporary_archive = archive_path is None
    if temporary_archive:
        descriptor, archive_name = tempfile.mkstemp(
            prefix="argos-model-", suffix=".argosmodel", dir=destination.parent
        )
        os.close(descriptor)
        archive = Path(archive_name)
    else:
        archive = Path(archive_path)
        if not archive.is_file():
            raise FileNotFoundError(f"locked model archive is unavailable: {archive}")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        if temporary_archive:
            _download(lock["url"], archive)
        actual = _sha256_file(archive)
        if actual != expected:
            raise ValueError(f"archive SHA-256 mismatch: expected {expected}, got {actual}")
        files = _safe_extract(archive, staging, lock["archive_root"])
        metadata_path = staging / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("from_code") != lock["from_code"] or metadata.get("to_code") != lock["to_code"]:
            raise ValueError("archive metadata language pair does not match lock")
        manifest = {
            "schema_version": 1,
            "model_pack": lock["model_pack"],
            "archive_url": lock["url"],
            "archive_sha256": expected,
            "metadata": metadata,
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(destination)
        return manifest
    finally:
        if temporary_archive:
            archive.unlink(missing_ok=True)
        if staging.exists():
            shutil.rmtree(staging)


def _download(url: str, destination: Path) -> None:
    if not url.startswith("https://"):
        raise ValueError("model URL must use HTTPS")
    request = urllib.request.Request(url, headers={"User-Agent": "distributed-gpu/0.1"})
    total = 0
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError("model archive exceeds maximum size")
            output.write(chunk)


def _safe_extract(archive: Path, destination: Path, archive_root: str) -> list[dict[str, Any]]:
    root = PurePosixPath(archive_root)
    if root.is_absolute() or len(root.parts) != 1 or root.parts[0] in {"", ".", ".."}:
        raise ValueError("archive_root must be one safe directory name")
    extracted: list[dict[str, Any]] = []
    total = 0
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) > MAX_FILES:
            raise ValueError("model archive has too many entries")
        for member in members:
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != root.name:
                raise ValueError(f"unsafe archive member: {member.filename!r}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"archive symlink is forbidden: {member.filename!r}")
            relative = PurePosixPath(*path.parts[1:])
            if not relative.parts:
                continue
            target = destination.joinpath(*relative.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            total += member.file_size
            if total > MAX_EXTRACTED_BYTES:
                raise ValueError("extracted model exceeds maximum size")
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with bundle.open(member) as source, target.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    digest.update(chunk)
                    output.write(chunk)
            if size != member.file_size:
                raise ValueError(f"archive member size mismatch: {member.filename!r}")
            extracted.append(
                {"path": relative.as_posix(), "size": size, "sha256": digest.hexdigest()}
            )
    extracted.sort(key=lambda item: item["path"])
    return extracted


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
