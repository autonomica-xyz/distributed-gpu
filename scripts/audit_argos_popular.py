#!/usr/bin/env python3
"""Download and audit the proposed popular Argos model shards.

This emits a lock catalog to stdout. Redirecting it into the reviewed catalog
is an explicit operator action; running the audit never changes trusted locks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any


INDEX_URL = "https://raw.githubusercontent.com/argosopentech/argospm-index/main/index.json"
SHARDS = {
    "core-western": ["es", "fr", "de", "pt", "it"],
    "global": ["zh", "ja", "ko", "ar", "hi", "ru"],
}
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024


def _download_and_audit(row: dict[str, Any], cache: Path) -> dict[str, Any]:
    url = row["links"][0]
    filename = Path(urllib.parse.urlparse(url).path).name
    if not filename.endswith(".argosmodel") or filename != Path(filename).name:
        raise ValueError(f"unsafe model archive name: {filename!r}")
    archive = cache / filename
    digest = hashlib.sha256()
    total = 0
    if not archive.is_file() or archive.stat().st_size == 0:
        request = urllib.request.Request(url, headers={"User-Agent": "distributed-gpu/0.1"})
        with urllib.request.urlopen(request, timeout=180) as response, archive.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_ARCHIVE_BYTES:
                    raise ValueError(f"archive exceeds limit: {filename}")
                digest.update(chunk)
                output.write(chunk)
    else:
        with archive.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_ARCHIVE_BYTES:
                    raise ValueError(f"archive exceeds limit: {filename}")
                digest.update(chunk)

    with zipfile.ZipFile(archive) as bundle:
        roots: set[str] = set()
        for member in bundle.infolist():
            path = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or stat.S_ISLNK(mode)
            ):
                raise ValueError(f"unsafe archive member in {filename}: {member.filename!r}")
            roots.add(path.parts[0])
        if len(roots) != 1:
            raise ValueError(f"archive must have one root in {filename}: {sorted(roots)}")
        archive_root = roots.pop()
        metadata_name = f"{archive_root}/metadata.json"
        metadata = json.loads(bundle.read(metadata_name))
    if (
        metadata.get("from_code") != row["from_code"]
        or metadata.get("to_code") != row["to_code"]
    ):
        raise ValueError(f"metadata pair mismatch in {filename}")
    sha256 = digest.hexdigest()
    version = str(row["package_version"])
    return {
        "archive_filename": filename,
        "archive_root": archive_root,
        "argos_version": str(row["argos_version"]),
        "compressed_bytes": total,
        "evaluation_only": True,
        "from_code": row["from_code"],
        "model_pack": f"argos-{row['from_code']}-{row['to_code']}-{version}-{sha256[:8]}",
        "package_version": version,
        "provenance_review_status": "pending",
        "sha256": sha256,
        "to_code": row["to_code"],
        "url": url,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.archive_cache.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(INDEX_URL, timeout=60) as response:
        index = json.load(response)
    rows = {
        (row["from_code"], row["to_code"]): row
        for row in index
        if isinstance(row, dict)
    }
    selected: list[tuple[str, dict[str, Any]]] = []
    for shard, languages in SHARDS.items():
        for language in languages:
            for pair in (("en", language), (language, "en")):
                if pair not in rows:
                    raise ValueError(f"official index has no direct pair {pair[0]}->{pair[1]}")
                selected.append((shard, rows[pair]))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        audited = list(executor.map(lambda item: _download_and_audit(item[1], args.archive_cache), selected))
    shards: dict[str, list[dict[str, Any]]] = {name: [] for name in SHARDS}
    for (shard, _), model in zip(selected, audited, strict=True):
        shards[shard].append(model)
    for models in shards.values():
        models.sort(key=lambda item: (item["from_code"], item["to_code"]))
    print(
        json.dumps(
            {
                "schema_version": 1,
                "audited_at": datetime.now(UTC).isoformat(),
                "index_url": INDEX_URL,
                "shards": shards,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
