"""Verify and install Hugging Face model packs from immutable locks.

A Hugging Face pack lock pins ``model_id`` to an immutable 40-hex commit and
bounds the download with ``allow_patterns``::

    {"format": "huggingface",
     "model_pack": "sdxl-base-1.0-fp16-76d28af",
     "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
     "revision": "76d28af79639c28a79fa5c6c6468febd3490a37e",
     "allow_patterns": ["*.json", "*.txt", "*.safetensors", ...]}

``install_hf_pack`` runs the actual Hub I/O in an owned child process
(``python -m sparehour_workloads.model_packs``): the child lists the
repo tree at the pinned revision, downloads with ``huggingface_hub`` (lazy
dependency — the module imports fine without it), verifies every file
against the repo's own metadata (LFS sha256 / git blob oid), and reports a
bounded classified result. Process termination is the cancellation
mechanism: an in-flight download parked inside OS DNS/connect/TLS/read is
reaped, not waited out. The parent re-verifies every staged file it was
told about, writes ``manifest.json``, runs the full ``verify_hf_pack``
check, and only then swaps the directory into place.

``verify_hf_pack`` is the consumer-facing check: it is pure local I/O (no
network, no Hub dependency) so a preinstalled pack — e.g. one built by an
operator from a verified cache — verifies without any gated fetch or token.

Ownership discipline matches the Argos pack path in ``sparehour_worker.dynamic``:
the installer never deletes or overwrites content it cannot prove is one of
its own pack dirs — manifest identity plus fully manifest-accounted
contents — and a symlink anywhere in the path traversal refuses the
operation entirely.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

MAX_PACK_FILES = 100_000
MAX_PACK_BYTES = 512 * 1024 * 1024 * 1024  # 512 GiB ceiling on lock-tunable bound
_DOWNLOAD_DEADLINE_SECONDS = 4 * 3600.0
_CHILD_RESULT_MAX_BYTES = 4096
_POLL_SECONDS = 0.25
_TERMINATE_GRACE_SECONDS = 2.0
_CHUNK_BYTES = 1024 * 1024
_MANIFEST_NAME = "manifest.json"

_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_MODEL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")
_MAX_PATTERN_LENGTH = 256


class ModelPackError(RuntimeError):
    """Provisioning failed (invalid lock, unsafe target, download error)."""


class PackVerificationError(ModelPackError):
    """An on-disk pack directory failed manifest/digest verification."""


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    host = host.casefold().strip("[]")
    if host in {"localhost", "localhost."}:
        return True
    try:
        import ipaddress

        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def normalize_hf_lock(lock: dict[str, Any]) -> dict[str, Any]:
    """Validate a ``format: huggingface`` lock entry; return the spec dict.

    Whitelists the fields the installer honours, enforces the immutable
    revision, and bounds the pattern/size limits a catalog may declare.
    Raises ModelPackError on any defect — callers surface the message as a
    status note, never traceback text.
    """
    if not isinstance(lock, dict):
        raise ModelPackError("hf lock entry is not a JSON object")
    if lock.get("format") != "huggingface":
        raise ModelPackError(
            f"hf lock format must be 'huggingface', got {lock.get('format')!r}"
        )
    pack = lock.get("model_pack")
    if not isinstance(pack, str) or not pack:
        raise ModelPackError("hf lock requires a non-empty model_pack")
    model_id = lock.get("model_id")
    if not isinstance(model_id, str) or not _MODEL_ID_RE.fullmatch(model_id):
        raise ModelPackError(f"hf lock has an unsafe model_id {model_id!r}")
    revision = lock.get("revision")
    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
        raise ModelPackError(
            f"hf pack {pack!r} revision must be an immutable 40-hex commit, "
            f"got {revision!r}"
        )
    patterns = lock.get("allow_patterns")
    if (
        not isinstance(patterns, list)
        or not patterns
        or not all(isinstance(pattern, str) for pattern in patterns)
    ):
        raise ModelPackError(
            f"hf pack {pack!r} requires a non-empty allow_patterns list"
        )
    for pattern in patterns:
        path = PurePosixPath(pattern)
        if (
            len(pattern) > _MAX_PATTERN_LENGTH
            or path.is_absolute()
            or ".." in path.parts
        ):
            raise ModelPackError(
                f"hf pack {pack!r} has an unsafe allow_pattern {pattern!r}"
            )
    endpoint = lock.get("endpoint")
    if endpoint is not None:
        endpoint = str(endpoint)
        parsed = urlparse(endpoint)
        if parsed.scheme == "https":
            pass
        elif parsed.scheme == "http" and _is_loopback_host(parsed.hostname):
            pass
        else:
            raise ModelPackError(
                f"hf pack {pack!r} endpoint must be HTTPS (or loopback HTTP "
                f"for a trusted mirror)"
            )
        if parsed.username or parsed.password:
            raise ModelPackError(
                f"hf pack {pack!r} endpoint must not embed credentials"
            )
    spec: dict[str, Any] = {
        "format": "huggingface",
        "model_pack": pack,
        "model_id": model_id,
        "revision": revision,
        "allow_patterns": list(patterns),
    }
    if endpoint is not None:
        spec["endpoint"] = endpoint
    for key, ceiling in (("max_files", MAX_PACK_FILES), ("max_bytes", MAX_PACK_BYTES)):
        value = lock.get(key)
        if value is None:
            spec[key] = ceiling
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ModelPackError(f"hf pack {pack!r} {key} must be a positive int")
        spec[key] = min(int(value), ceiling)
    return spec


# -- pack directory verification ----------------------------------------------


def _first_symlink_component(path: Path) -> Path | None:
    """First symlink anywhere in ``path``'s traversal, else ``None``.

    Walks every component from the filesystem root down to the path itself
    with ``lstat`` so a symlinked ancestor — not just a symlinked leaf —
    is caught. Missing components stop the walk: later components can only
    be created as real directories by our own install calls.
    """
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = os.lstat(current)
        except OSError:
            return None  # absent or unreachable: nothing symlinked to report
        if stat.S_ISLNK(info.st_mode):
            return current
        if not stat.S_ISDIR(info.st_mode):
            return None  # a non-directory ends traversal here
    return None


def _pack_manifest(target: Path) -> dict[str, Any] | None:
    """The manifest inside ``target``, or ``None`` if unreadable.

    A symlinked manifest is unreadable: following it would let a swapped-in
    file claim ownership of a directory it does not describe.
    """
    manifest_path = target / _MANIFEST_NAME
    if manifest_path.is_symlink():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return manifest if isinstance(manifest, dict) else None


def _manifest_identifies(manifest: dict[str, Any], lock: dict[str, Any]) -> bool:
    """True when the manifest is one of our HF packs for ``lock``'s pack id."""
    return (
        manifest.get("format") == "huggingface"
        and manifest.get("model_pack") == lock.get("model_pack")
    )


def _managed_paths(
    manifest: dict[str, Any],
) -> tuple[set[PurePosixPath], set[PurePosixPath]] | None:
    """(files, dirs) a real manifest claims to own, or ``None``.

    A manifest is only coherent proof of ownership when its file list is a
    nonempty set of safe, unique relative paths — the same invariants
    ``verify_hf_pack`` enforces. Anything else cannot account for the
    directory's contents, so nothing may be deleted on its say-so.
    """
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        return None
    file_paths: set[PurePosixPath] = {PurePosixPath(_MANIFEST_NAME)}
    dir_paths: set[PurePosixPath] = set()
    for item in files:
        if not isinstance(item, dict):
            return None
        relative = PurePosixPath(str(item.get("path", "")))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            return None
        if relative in file_paths:
            return None
        file_paths.add(relative)
        for parent in relative.parents:
            if not parent.parts or parent == PurePosixPath("."):
                break
            dir_paths.add(parent)
    if file_paths & dir_paths:
        return None  # one path claimed as both file and directory: incoherent
    return file_paths, dir_paths


def _first_unmanaged_entry(target: Path, manifest: dict[str, Any]) -> Path | None:
    """First on-disk entry the manifest does not account for, else ``None``.

    Ownership of a corrupt pack dir is only proven when EVERYTHING in the
    tree is managed: the manifest itself, every listed member file, and the
    directories that contain them. An unlisted file, an unlisted directory,
    a listed path of the wrong kind, or a symlink anywhere under ``target``
    is unmanaged content — evidence the directory holds data that is not
    ours to delete, so it is reported and never removed.
    """
    managed = _managed_paths(manifest)
    if managed is None:
        return target  # incoherent manifest accounts for nothing on disk
    file_paths, dir_paths = managed
    pending = [target]
    while pending:
        directory = pending.pop()
        try:
            children = sorted(directory.iterdir())
        except OSError:
            continue  # unreadable mid-scan: verification fails on its own
        for child in children:
            relative = child.relative_to(target)
            try:
                info = os.lstat(child)
            except OSError:
                continue  # vanished mid-scan: verify/blocker rerun decides
            if stat.S_ISLNK(info.st_mode):
                return child  # a symlink is never managed pack content
            if relative in file_paths:
                if not stat.S_ISREG(info.st_mode):
                    return child  # listed file is not a regular file
                continue
            if relative in dir_paths:
                if not stat.S_ISDIR(info.st_mode):
                    return child  # listed directory is not a directory
                pending.append(child)
                continue
            return child  # unlisted entry: not ours to delete
    return None


def hf_pack_provision_blocker(
    target: Path, lock: dict[str, Any]
) -> str | None:
    """Why ``install_hf_pack`` must refuse ``target``, or ``None`` if it may build.

    Fail-closed order: a symlink anywhere in the path traversal or anything
    occupying ``target`` that is not provably one of our own pack dirs —
    manifest identity AND fully manifest-accounted contents — blocks
    provisioning. Deleting unowned or unmanaged content is never an option:
    a route-controlled path must not become remote code for unlinking user
    data; the refusal names the offending entry so it is actionable.
    """
    alias = _first_symlink_component(target)
    if alias is not None:
        return (
            f"{alias} is a symlink; refusing to provision {target} through it — "
            "remove or repoint the alias and retry"
        )
    try:
        info = os.lstat(target)
    except OSError:
        return None  # absent: nothing to protect, build is allowed
    refusal = (
        f"{target} already exists and is not a verified hf pack for "
        f"{lock.get('model_pack')!r}; refusing to remove or overwrite it — "
        "clear the path manually or pin a different model dir"
    )
    if not stat.S_ISDIR(info.st_mode):  # files and symlinks are foreign
        return refusal
    manifest = _pack_manifest(target)
    if manifest is None or not _manifest_identifies(manifest, lock):
        return refusal
    unmanaged = _first_unmanaged_entry(target, manifest)
    if unmanaged is not None:
        return (
            f"{target} is an hf pack for {lock.get('model_pack')!r} but holds "
            f"unmanaged content at {unmanaged}; refusing to replace it — "
            "clear the entry or pin a different model dir"
        )
    return None


def verify_hf_pack(
    path: Path, *, model_pack: str, model_id: str, revision: str
) -> None:
    """Full integrity check of an installed HF pack against its identity.

    Pure local I/O — no network and no ``huggingface_hub`` dependency, so a
    preinstalled pack verifies without a token or a gated fetch. Verifies
    the manifest's pack id, model id, and immutable revision, then every
    listed member's path safety, size, and SHA-256. ``path``, every ancestor
    component, the manifest, and every member path component must be real
    directories/files — a symlink anywhere in the traversal means the check
    was never on the directory it claims to describe. Extra on-disk files
    the manifest does not list fail verification: the directory must be
    exactly the pack contents. Raises PackVerificationError on the first
    defect.
    """
    if not _REVISION_RE.fullmatch(str(revision)):
        raise PackVerificationError(
            f"hf pack revision must be an immutable 40-hex commit, got {revision!r}"
        )
    target = Path(path)
    alias = _first_symlink_component(target)
    if alias is not None:
        raise PackVerificationError(f"{target}: path component {alias} is a symlink")
    if not target.is_dir():
        raise PackVerificationError(f"{target}: not a directory")
    manifest_path = target / _MANIFEST_NAME
    if manifest_path.is_symlink():
        raise PackVerificationError(f"{target}: manifest is a symlink")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackVerificationError(f"{target}: unreadable manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise PackVerificationError(f"{target}: manifest is not a JSON object")
    if manifest.get("format") != "huggingface":
        raise PackVerificationError(f"{target}: manifest is not an hf pack")
    if manifest.get("model_pack") != model_pack:
        raise PackVerificationError(f"{target}: manifest pack id does not match")
    if manifest.get("model_id") != model_id:
        raise PackVerificationError(f"{target}: manifest model id does not match")
    if manifest.get("revision") != revision:
        raise PackVerificationError(f"{target}: manifest revision does not match")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise PackVerificationError(f"{target}: manifest lists no files")
    expected: set[PurePosixPath] = set()
    for item in files:
        if not isinstance(item, dict):
            raise PackVerificationError(f"{target}: manifest entry is not a JSON object")
        relative = PurePosixPath(str(item.get("path", "")))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise PackVerificationError(f"{target}: unsafe manifest path {relative}")
        if relative in expected:
            raise PackVerificationError(f"{target}: duplicate manifest path {relative}")
        expected.add(relative)
        member = target.joinpath(*relative.parts)
        member_alias = _first_symlink_component(member)
        if member_alias is not None:
            raise PackVerificationError(
                f"{target}: member path component is a symlink: {member_alias}"
            )
        try:
            info = os.lstat(member)
        except OSError as exc:
            raise PackVerificationError(
                f"{target}: missing member {relative}"
            ) from exc
        if not stat.S_ISREG(info.st_mode):
            raise PackVerificationError(
                f"{target}: member is not a regular file: {relative}"
            )
        digest = hashlib.sha256()
        size = 0
        try:
            with member.open("rb") as handle:
                while chunk := handle.read(_CHUNK_BYTES):
                    digest.update(chunk)
                    size += len(chunk)
        except OSError as exc:
            raise PackVerificationError(
                f"{target}: unreadable member {relative}: {exc}"
            ) from exc
        if size != item.get("size") or digest.hexdigest() != item.get("sha256"):
            raise PackVerificationError(
                f"{target}: digest mismatch for member {relative}"
            )
    # The directory must be exactly the pack: any file the manifest does
    # not list fails verification outright.
    unmanaged = _first_unmanaged_entry(target, manifest)
    if unmanaged is not None:
        raise PackVerificationError(
            f"{target}: unmanaged content {unmanaged.name!r} is not part of the pack"
        )


def is_hf_pack_installed(target: Path, lock: dict[str, Any]) -> bool:
    """True only when ``target`` verifies completely against ``lock``."""
    try:
        spec = normalize_hf_lock(lock)
        verify_hf_pack(
            target,
            model_pack=spec["model_pack"],
            model_id=spec["model_id"],
            revision=spec["revision"],
        )
    except PackVerificationError:
        return False
    return True


# -- install -------------------------------------------------------------------


def install_hf_pack(
    lock: dict[str, Any], target: Path, *, stop: Any | None = None
) -> dict[str, Any]:
    """Verified download of a pinned HF snapshot. Idempotent: a verified
    directory is a cache hit and returns its manifest without any I/O.

    This runs under route-controlled paths, so it NEVER deletes content it
    did not create. A symlink anywhere in ``target``'s traversal, or any
    existing file/symlink/directory that is not provably one of our own pack
    dirs — manifest identity plus fully manifest-accounted contents —
    raises ModelPackError and leaves the path untouched. A corrupt-but-owned
    pack dir is rebuilt only through a verified staging directory swapped
    in atomically; if the swap or rebuild fails, the previous contents are
    preserved.
    """
    spec = normalize_hf_lock(lock)
    try:
        verify_hf_pack(
            target,
            model_pack=spec["model_pack"],
            model_id=spec["model_id"],
            revision=spec["revision"],
        )
        return json.loads((target / _MANIFEST_NAME).read_text(encoding="utf-8"))
    except PackVerificationError:
        pass
    blocker = hf_pack_provision_blocker(target, lock)
    if blocker is not None:
        raise ModelPackError(
            f"cannot install model pack {lock.get('model_pack')!r}: {blocker}"
        )
    if importlib.util.find_spec("huggingface_hub") is None:
        raise ModelPackError(
            "installing hf packs requires huggingface_hub "
            "(install the sparehour 'models' extra)"
        )
    target = Path(target)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ModelPackError(f"{target}: cannot create parent directories: {exc}") from exc
    descriptor, report_name = tempfile.mkstemp(
        prefix=f".{target.name}-report-", suffix=".json", dir=target.parent
    )
    os.close(descriptor)
    report_path = Path(report_name)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}-new-", dir=target.parent)
    )
    try:
        child_spec = dict(spec)
        child_spec["staging"] = str(staging)
        child_spec["report"] = str(report_path)
        _run_download_child(
            child_spec,
            stop=stop,
            deadline=time.monotonic() + _DOWNLOAD_DEADLINE_SECONDS,
            cancelled="model download cancelled",
            timed_out="model download timed out",
        )
        if stop is not None and stop.is_set():
            raise ModelPackError("model install cancelled")
        # Re-verify every staged file the child reported: the report claims
        # the set, this pass proves the bytes. Only then does the manifest
        # land and the full check run before the swap.
        files = _verify_staged_files(staging, report_path, spec)
        manifest = {
            "schema_version": 1,
            "format": "huggingface",
            "model_pack": spec["model_pack"],
            "model_id": spec["model_id"],
            "revision": spec["revision"],
            "files": files,
        }
        (staging / _MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        verify_hf_pack(
            staging,
            model_pack=spec["model_pack"],
            model_id=spec["model_id"],
            revision=spec["revision"],
        )
        if os.path.lexists(target):
            _swap_verified_pack(staging, target, lock)
        else:
            staging.rename(target)
        return manifest
    finally:
        report_path.unlink(missing_ok=True)
        if staging.exists():
            shutil.rmtree(staging)


def _verify_staged_files(
    staging: Path, report_path: Path, spec: dict[str, Any]
) -> list[dict[str, Any]]:
    """Re-hash every file the child reported; return manifest file rows.

    The child already verified staged bytes against trusted repo metadata;
    this pass proves the report's claims match what is actually on disk —
    a wrong size or hash, a missing member, or a member the report never
    claimed all fail before the pack is ever swapped in.
    """
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelPackError(f"download child left an unreadable report: {exc}") from exc
    entries = report.get("files") if isinstance(report, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ModelPackError("download child reported no files")
    if len(entries) > spec["max_files"]:
        raise ModelPackError("download child reported too many files")
    seen: set[PurePosixPath] = set()
    files: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ModelPackError("download child report entry is not an object")
        relative = PurePosixPath(str(entry.get("path", "")))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ModelPackError(f"download child reported unsafe path {relative}")
        if relative in seen:
            raise ModelPackError(f"download child reported duplicate path {relative}")
        seen.add(relative)
        member = staging.joinpath(*relative.parts)
        member_alias = _first_symlink_component(member)
        if member_alias is not None:
            raise ModelPackError(
                f"staged member path component is a symlink: {member_alias}"
            )
        digest = hashlib.sha256()
        size = 0
        try:
            with member.open("rb") as handle:
                while chunk := handle.read(_CHUNK_BYTES):
                    digest.update(chunk)
                    size += len(chunk)
        except OSError as exc:
            raise ModelPackError(f"staged member {relative} unreadable: {exc}") from exc
        if size != entry.get("size") or digest.hexdigest() != entry.get("sha256"):
            raise ModelPackError(
                f"staged member {relative} does not match the download report"
            )
        files.append(
            {"path": relative.as_posix(), "size": size, "sha256": digest.hexdigest()}
        )
    files.sort(key=lambda item: item["path"])
    return files


def _swap_verified_pack(staging: Path, target: Path, lock: dict[str, Any]) -> None:
    """Atomically replace owned-but-corrupt ``target`` with verified ``staging``.

    The old directory moves aside first, the new one lands in a single
    rename, and the backup is kept if any step fails — the previous content
    is never destroyed by a rebuild that could not complete. The manifest
    identity/ownership proof is re-checked on the moved-aside backup before
    it is deleted: content that changed between the blocker check and the
    swap is preserved, never removed.
    """
    backup = Path(tempfile.mkdtemp(prefix=f".{target.name}-old-", dir=target.parent))
    backup.rmdir()  # reserve the name; the old dir renames onto it
    os.rename(target, backup)
    try:
        staging.rename(target)
    except OSError:
        try:
            os.rename(backup, target)
        except OSError as restore_exc:
            raise ModelPackError(
                f"{target}: install failed and the previous pack could not be "
                f"restored; preserved at {backup}: {restore_exc}"
            ) from restore_exc
        raise ModelPackError(
            f"{target}: could not move verified pack into place; previous "
            "contents restored"
        )
    # The verified pack is in place; delete the backup only while it is
    # still provably wholly managed — content swapped in or grown between
    # the blocker check and now is preserved, not removed.
    manifest = _pack_manifest(backup)
    if (
        manifest is None
        or not _manifest_identifies(manifest, lock)
        or _first_unmanaged_entry(backup, manifest) is not None
    ):
        raise ModelPackError(
            f"{target}: installed the verified pack, but the replaced directory "
            f"held unmanaged content; preserved at {backup} instead of deleting"
        )
    try:
        shutil.rmtree(backup)
    except OSError:
        pass  # leftover backup is inert; the verified pack is in place


# -- owned download child -------------------------------------------------------


class _Refused(Exception):
    """A policy refusal surfaced verbatim (already sanitized)."""


def _error_kind(exc: BaseException) -> str:
    """``TypeName`` or ``HTTP status N`` — never remote-controlled text."""
    import http.client
    import urllib.error

    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP status {exc.code}"
    if isinstance(exc, http.client.HTTPException) and getattr(exc, "args", None):
        return type(exc).__name__
    return type(exc).__name__


def _run_download_child(
    spec: dict[str, Any], *, stop: Any | None, deadline: float,
    cancelled: str, timed_out: str,
) -> None:
    """Spawn and own one HF download child; raise ModelPackError on failure.

    Polls the child against ``stop`` and the monotonic ``deadline``; on
    either it terminates, escalates to kill if needed, and always reaps —
    which cancels OS DNS, connect, TLS, header, and body phases alike. The
    child's stdout result is the only cross-process payload: bounded,
    classified JSON, never raw remote text. The model spec rides stdin, so
    nothing repo- or endpoint-specific ever reaches argv or the process
    table.
    """
    proc = subprocess.Popen(
        [sys.executable, "-m", "sparehour_workloads.model_packs"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        try:
            proc.stdin.write(json.dumps(spec).encode())
        except (BrokenPipeError, OSError):
            pass  # child already exited; its stdout says why
        proc.stdin.close()
        while True:
            if stop is not None and stop.is_set():
                raise ModelPackError(cancelled)
            if time.monotonic() > deadline:
                raise ModelPackError(timed_out)
            try:
                proc.wait(_POLL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                continue
        output = proc.stdout.read(_CHILD_RESULT_MAX_BYTES + 1)
        if proc.returncode != 0 or len(output) > _CHILD_RESULT_MAX_BYTES:
            raise ModelPackError("download child exited without a result")
        try:
            result = json.loads(output)
        except ValueError:
            raise ModelPackError("download child returned an unreadable result")
        if not isinstance(result, dict) or "ok" not in result:
            raise ModelPackError("download child returned an unreadable result")
        if result.get("ok"):
            return
        if result.get("policy"):
            raise ModelPackError(str(result["policy"]))
        raise ModelPackError(
            f"model download failed: {result.get('kind', 'unknown error')}"
        )
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(_TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()


def _child_download(spec: dict[str, Any]) -> None:
    """Child half: list repo metadata, download, verify, write the report.

    ``huggingface_hub`` is imported lazily here — the parent module stays
    importable without it. Both the 0.36 and 1.x APIs are honoured: no
    ``local_dir_use_symlinks`` (removed in 1.x) and no token plumbing beyond
    the ambient HF_TOKEN/HF_ENDPOINT environment the operator provides.
    """
    from huggingface_hub import HfApi, snapshot_download  # noqa: PLC0415

    staging = Path(spec["staging"])
    report_path = Path(spec["report"])
    model_id = spec["model_id"]
    revision = spec["revision"]
    allow_patterns = spec["allow_patterns"]
    endpoint = spec.get("endpoint")

    api_kwargs = {"endpoint": endpoint} if endpoint else {}
    api = HfApi(**api_kwargs)
    entries: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for item in api.list_repo_tree(
        repo_id=model_id, revision=revision, recursive=True, expand=True
    ):
        blob_id = getattr(item, "blob_id", None)
        if blob_id is None or getattr(item, "type", "file") != "file":
            continue  # RepoFolder: only files carry blob ids
        path = str(item.path)
        if not any(fnmatch(path, pattern) for pattern in allow_patterns):
            continue
        lfs = getattr(item, "lfs", None)
        size = item.size
        if size is None and isinstance(lfs, dict):
            size = lfs.get("size")  # some hub versions only carry it under lfs
        lfs_sha256 = (
            getattr(lfs, "sha256", None)
            or (lfs.get("sha256") or lfs.get("oid") if isinstance(lfs, dict) else "")
        )
        entries[path] = {
            "size": int(size or 0),
            "oid": str(lfs_sha256 or ""),
            "git_oid": str(blob_id),
            "lfs": bool(lfs),
        }
        total_bytes += entries[path]["size"]
    if not entries:
        raise _Refused(
            f"allow_patterns selected no files in {model_id!r}@{revision}"
        )
    if len(entries) > int(spec["max_files"]):
        raise _Refused("model snapshot exceeds the file count bound")
    if total_bytes > int(spec["max_bytes"]):
        raise _Refused("model snapshot exceeds the byte bound")

    download_kwargs = {"endpoint": endpoint} if endpoint else {}
    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=allow_patterns,
            **download_kwargs,
        )
    )

    # Copy the pinned snapshot into staging as real files (hub snapshot
    # members may be symlinks into the shared cache on older hub versions)
    # while verifying each file against trusted repo metadata: the LFS
    # sha256 for pointer-backed files, the git blob oid otherwise.
    files: list[dict[str, Any]] = []
    for root, dirnames, filenames in os.walk(snapshot):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        base = Path(root)
        for name in filenames:
            source = base / name
            relative = source.relative_to(snapshot).as_posix()
            expected = entries.get(relative)
            if expected is None:
                continue  # hub bookkeeping outside our selection
            if not source.is_file():
                raise _Refused("hub snapshot member is not a file")
            # Snapshot members may symlink into the shared hub cache: copying
            # the resolved bytes into staging is safe because every file is
            # verified against repo metadata before the pack is trusted, and
            # the pack dir itself must contain real files, never links.
            sha = hashlib.sha256()
            git = hashlib.sha1()
            git.update(b"blob " + str(expected["size"]).encode() + b"\0")
            size = 0
            destination = staging.joinpath(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as handle, destination.open("wb") as output:
                while chunk := handle.read(_CHUNK_BYTES):
                    sha.update(chunk)
                    git.update(chunk)
                    output.write(chunk)
                    size += len(chunk)
            if size != expected["size"]:
                raise _Refused("downloaded file size does not match repo metadata")
            if expected["lfs"]:
                if sha.hexdigest() != expected["oid"]:
                    raise _Refused("downloaded LFS file fails repo sha256")
            elif git.hexdigest() != expected["git_oid"]:
                raise _Refused("downloaded file fails repo git oid")
            files.append(
                {"path": relative, "size": size, "sha256": sha.hexdigest()}
            )
    if {item["path"] for item in files} != set(entries):
        raise _Refused("downloaded snapshot is missing pinned files")
    files.sort(key=lambda item: item["path"])
    report_path.write_text(json.dumps({"files": files}), encoding="utf-8")


def _child_entry() -> int:
    try:
        spec = json.loads(sys.stdin.read())
        _child_download(spec)
    except _Refused as exc:
        payload = {"ok": False, "policy": str(exc)}
    except BaseException as exc:  # classified; remote text never crosses
        payload = {"ok": False, "kind": _error_kind(exc)}
    else:
        payload = {"ok": True}
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(_child_entry())
