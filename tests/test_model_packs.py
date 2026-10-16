"""HF model-pack verification/install tests: real on-disk manifests, digest
checks, ownership refusal, and a faked download child for the rebuild path.

No test exercises the network child for real — ``_run_download_child`` is
either uninvoked (verify/refuse paths, which must never spawn it) or faked
to emit a report over real staged files. huggingface_hub itself is a lazy
dependency and is not required for any test here.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

model_packs = pytest.importorskip("sparehour_workloads.model_packs")
mp = model_packs

MODEL_ID = "some-org/some-model"
REVISION = "a" * 40


def _lock(pack="test-pack", model_id=MODEL_ID, revision=REVISION, **over):
    lock = {
        "format": "huggingface",
        "model_pack": pack,
        "model_id": model_id,
        "revision": revision,
        "allow_patterns": ["*.json", "*.bin"],
    }
    lock.update(over)
    return lock


def _make_pack_dir(target: Path, *, pack="test-pack", model_id=MODEL_ID,
                   revision=REVISION,
                   files={"config.json": b"{}", "weights/model.bin": b"w"},
                   write_manifest=True) -> Path:
    """Build a real on-disk pack dir with a real manifest."""
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, data in files.items():
        member = target.joinpath(*name.split("/"))
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_bytes(data)
        rows.append(
            {
                "path": name,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    rows.sort(key=lambda item: item["path"])
    if write_manifest:
        manifest = {
            "schema_version": 1,
            "format": "huggingface",
            "model_pack": pack,
            "model_id": model_id,
            "revision": revision,
            "files": rows,
        }
        (target / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    return target


# -- lock normalization --------------------------------------------------------


def test_normalize_lock_requires_immutable_revision():
    for bad in ("main", "v1", "a" * 39, "g" * 40, "A" * 40, 123, None):
        with pytest.raises(mp.ModelPackError, match="revision"):
            mp.normalize_hf_lock(_lock(revision=bad))


def test_normalize_lock_rejects_bad_id_and_patterns():
    with pytest.raises(mp.ModelPackError, match="model_id"):
        mp.normalize_hf_lock(_lock(model_id="no slash"))
    with pytest.raises(mp.ModelPackError, match="model_id"):
        mp.normalize_hf_lock(_lock(model_id="a/b/extra"))
    for patterns in (None, [], "not-a-list", ["../x"], ["/abs/*"], ["ok", ".."]):
        with pytest.raises(mp.ModelPackError, match="allow_pattern"):
            mp.normalize_hf_lock(_lock(allow_patterns=patterns))
    with pytest.raises(mp.ModelPackError, match="format"):
        mp.normalize_hf_lock(_lock(format="argos"))
    with pytest.raises(mp.ModelPackError, match="credentials"):
        mp.normalize_hf_lock(_lock(endpoint="https://user:tok@hf.invalid/"))
    with pytest.raises(mp.ModelPackError, match="HTTPS"):
        mp.normalize_hf_lock(_lock(endpoint="http://hf.invalid/"))


def test_normalize_lock_whitelists_and_bounds():
    spec = mp.normalize_hf_lock(
        _lock(max_bytes=10**20, max_files=10**9, ignored_extra="x")
    )
    assert spec["max_bytes"] == mp.MAX_PACK_BYTES
    assert spec["max_files"] == mp.MAX_PACK_FILES
    assert "ignored_extra" not in spec
    spec = mp.normalize_hf_lock(_lock(endpoint="http://127.0.0.1:9999/hf"))
    assert spec["endpoint"] == "http://127.0.0.1:9999/hf"


# -- directory verification ------------------------------------------------------


def test_verify_hf_pack_round_trip(tmp_path):
    lock = _lock()
    target = _make_pack_dir(tmp_path / "pack")
    mp.verify_hf_pack(
        target, model_pack="test-pack", model_id=MODEL_ID, revision=REVISION
    )
    assert mp.is_hf_pack_installed(target, lock)


def test_verify_hf_pack_identity_mismatches(tmp_path):
    target = _make_pack_dir(tmp_path / "pack")
    for kwargs in (
        {"model_pack": "other"},
        {"model_id": "other/model"},
        {"revision": "b" * 40},
        {"revision": "main"},
    ):
        merged = dict(model_pack="test-pack", model_id=MODEL_ID, revision=REVISION)
        merged.update(kwargs)
        with pytest.raises(mp.PackVerificationError):
            mp.verify_hf_pack(target, **merged)


def test_verify_hf_pack_catches_bitrot_and_extras(tmp_path):
    target = _make_pack_dir(tmp_path / "pack")
    member = target / "weights" / "model.bin"
    member.write_bytes(b"corrupted")
    with pytest.raises(mp.PackVerificationError, match="digest mismatch"):
        mp.verify_hf_pack(
            target, model_pack="test-pack", model_id=MODEL_ID, revision=REVISION
        )
    assert not mp.is_hf_pack_installed(target, _lock())

    member.write_bytes(b"w")  # restore content
    stray = target / "stray.bin"
    stray.write_bytes(b"x")
    with pytest.raises(mp.PackVerificationError, match="unmanaged"):
        mp.verify_hf_pack(
            target, model_pack="test-pack", model_id=MODEL_ID, revision=REVISION
        )
    stray.unlink()
    mp.verify_hf_pack(
        target, model_pack="test-pack", model_id=MODEL_ID, revision=REVISION
    )


def test_verify_hf_pack_rejects_symlink_traversal(tmp_path):
    target = _make_pack_dir(tmp_path / "pack")
    link = tmp_path / "as-link"
    link.symlink_to(target)
    with pytest.raises(mp.PackVerificationError, match="symlink"):
        mp.verify_hf_pack(
            link, model_pack="test-pack", model_id=MODEL_ID, revision=REVISION
        )
    (target / "config.json").unlink()
    (target / "config.json").symlink_to(tmp_path / "elsewhere")
    assert not mp.is_hf_pack_installed(target, _lock())


def test_verify_hf_pack_needs_no_network_or_hub(tmp_path, monkeypatch):
    """Verify is pure local I/O: it must not import huggingface_hub."""
    target = _make_pack_dir(tmp_path / "pack")
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", None)
    mp.verify_hf_pack(
        target, model_pack="test-pack", model_id=MODEL_ID, revision=REVISION
    )


# -- install ----------------------------------------------------------------------


def test_install_hf_pack_cache_hit_never_spawns(tmp_path, monkeypatch):
    lock = _lock()
    target = _make_pack_dir(tmp_path / "pack")
    monkeypatch.setattr(
        mp, "_run_download_child",
        lambda *a, **k: pytest.fail("download child spawned on a cache hit"),
    )
    manifest = mp.install_hf_pack(lock, target)
    assert manifest["model_pack"] == "test-pack"


def test_install_hf_pack_refuses_and_preserves_foreign(tmp_path):
    lock = _lock()
    file_target = tmp_path / "a-file"
    file_target.write_text("precious")
    dir_target = tmp_path / "a-dir"
    dir_target.mkdir()
    (dir_target / "keep.txt").write_text("precious")
    foreign = _make_pack_dir(tmp_path / "foreign", pack="other-pack")
    link = tmp_path / "a-link"
    link.symlink_to(foreign)
    for path in (file_target, dir_target, foreign, link):
        with pytest.raises(mp.ModelPackError, match="refus"):
            mp.install_hf_pack(lock, path)
    assert file_target.read_text() == "precious"
    assert (dir_target / "keep.txt").read_text() == "precious"
    assert link.is_symlink()


def test_install_hf_pack_refuses_symlinked_ancestor(tmp_path):
    lock = _lock()
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "via").symlink_to(real)
    with pytest.raises(mp.ModelPackError, match="symlink"):
        mp.install_hf_pack(lock, tmp_path / "via" / "pack")
    assert list(real.iterdir()) == []


def test_install_hf_pack_needs_hub_only_when_downloading(tmp_path, monkeypatch):
    """Cache hits and refusals never require huggingface_hub; a real download
    does, and fails closed before any child is spawned."""
    lock = _lock()
    installed = _make_pack_dir(tmp_path / "installed")
    monkeypatch.setattr(
        mp.importlib.util, "find_spec", lambda name: None
    )
    mp.install_hf_pack(lock, installed)  # cache hit: no hub needed

    def fail_spawn(*a, **k):
        pytest.fail("download child spawned without huggingface_hub")

    monkeypatch.setattr(mp.subprocess, "Popen", fail_spawn)
    with pytest.raises(mp.ModelPackError, match="huggingface_hub"):
        mp.install_hf_pack(lock, tmp_path / "fresh")
    assert not (tmp_path / "fresh").exists()


def _fake_child(tmp_path, monkeypatch, files, tamper=None):
    """Stub _run_download_child: stage real files and write a real report."""
    def fake(spec, *, stop=None, deadline=None, cancelled="", timed_out=""):
        staging = Path(spec["staging"])
        report = Path(spec["report"])
        rows = []
        for name, data in files.items():
            member = staging.joinpath(*name.split("/"))
            member.parent.mkdir(parents=True, exist_ok=True)
            member.write_bytes(data)
            rows.append(
                {
                    "path": name,
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        if tamper:
            tamper(rows, staging)
        report.write_text(json.dumps({"files": rows}))

    monkeypatch.setattr(mp, "_run_download_child", fake)


def test_install_hf_pack_builds_manifest_and_verifies(tmp_path, monkeypatch):
    lock = _lock()
    target = tmp_path / "packs" / "test-pack"
    _fake_child(
        tmp_path, monkeypatch,
        {"config.json": b"{}", "weights/model.bin": b"real bytes"},
    )
    manifest = mp.install_hf_pack(lock, target)
    assert manifest["format"] == "huggingface"
    assert manifest["model_id"] == MODEL_ID
    assert manifest["revision"] == REVISION
    assert (target / "weights" / "model.bin").read_bytes() == b"real bytes"
    mp.verify_hf_pack(
        target, model_pack="test-pack", model_id=MODEL_ID, revision=REVISION
    )
    leftovers = list(target.parent.iterdir())
    assert sorted(p.name for p in leftovers) == ["test-pack"]


def test_install_hf_pack_rebuilds_owned_corrupt_dir(tmp_path, monkeypatch):
    lock = _lock()
    target = _make_pack_dir(tmp_path / "pack")
    (target / "config.json").write_bytes(b"corrupted")
    _fake_child(tmp_path, monkeypatch, {"config.json": b"{}", "weights/model.bin": b"w"})
    mp.install_hf_pack(lock, target)
    assert (target / "config.json").read_bytes() == b"{}"
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(f".{target.name}-")]


def test_install_hf_pack_refuses_owned_dir_with_unmanaged(tmp_path, monkeypatch):
    """Manifest-matching dir + unlisted file: refuse, preserve, never spawn."""
    lock = _lock()
    target = _make_pack_dir(tmp_path / "pack")
    (target / "config.json").write_bytes(b"corrupted")
    (target / "user-notes.txt").write_text("precious")
    monkeypatch.setattr(
        mp, "_run_download_child",
        lambda *a, **k: pytest.fail("download child spawned for unsafe target"),
    )
    with pytest.raises(mp.ModelPackError, match="refus"):
        mp.install_hf_pack(lock, target)
    assert (target / "user-notes.txt").read_text() == "precious"
    assert (target / "config.json").read_bytes() == b"corrupted"


def test_install_hf_pack_rejects_lying_report(tmp_path, monkeypatch):
    """A report whose hashes do not match staged bytes fails before swap."""
    lock = _lock()

    def corrupt(rows, staging):
        rows[0]["sha256"] = "0" * 64

    _fake_child(
        tmp_path, monkeypatch, {"config.json": b"{}"}, tamper=corrupt
    )
    target = tmp_path / "pack"
    with pytest.raises(mp.ModelPackError, match="report"):
        mp.install_hf_pack(lock, target)
    assert not target.exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".pack-")]


def test_install_hf_pack_cancelled_mid_download(tmp_path, monkeypatch):
    lock = _lock()
    monkeypatch.setattr(mp.importlib.util, "find_spec", lambda name: True)

    def stall(spec, *, stop=None, deadline=None, cancelled="", timed_out=""):
        raise mp.ModelPackError(cancelled or "model download cancelled")

    monkeypatch.setattr(mp, "_run_download_child", stall)
    stop = threading.Event()
    stop.set()
    target = tmp_path / "pack"
    with pytest.raises(mp.ModelPackError, match="cancelled"):
        mp.install_hf_pack(lock, target, stop=stop)
    assert not target.exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".pack-")]


def test_child_result_protocol_is_bounded_and_classified(tmp_path, monkeypatch):
    """The parent honours only bounded {"ok": ...} JSON on child stdout."""
    import subprocess as sp

    class FakeProc:
        def __init__(self, output: bytes, code: int = 0):
            import io

            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO(output)
            self.returncode = code
            self._waited = True

        def wait(self, timeout=None):
            return self.returncode

        def poll(self):
            return self.returncode

        def terminate(self):
            pass

        def kill(self):
            pass

    monkeypatch.setattr(
        mp.subprocess, "Popen",
        lambda *a, **k: FakeProc(b'{"ok": false, "kind": "HTTP status 401"}'),
    )
    with pytest.raises(mp.ModelPackError, match="401"):
        mp._run_download_child(
            {"model_id": MODEL_ID}, stop=None,
            deadline=time.monotonic() + 10, cancelled="x", timed_out="x",
        )

    monkeypatch.setattr(
        mp.subprocess, "Popen",
        lambda *a, **k: FakeProc(b"remote hostile text s3cr3t", code=0),
    )
    with pytest.raises(mp.ModelPackError) as caught:
        mp._run_download_child(
            {"model_id": MODEL_ID}, stop=None,
            deadline=time.monotonic() + 10, cancelled="x", timed_out="x",
        )
    assert "s3cr3t" not in str(caught.value)
    assert "hostile" not in str(caught.value)


def test_download_child_cancelled_while_running(tmp_path, monkeypatch):
    """A spawned-but-silent child is terminated when stop fires."""
    stop = threading.Event()
    stop.set()

    class HangingProc:
        def __init__(self):
            import io

            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO()
            self.returncode = None
            self.killed = False

        def wait(self, timeout=None):
            if self.returncode is None:
                raise mp.subprocess.TimeoutExpired("x", timeout)
            return self.returncode

        def poll(self):
            return self.returncode if self.returncode is not None else None

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    proc = HangingProc()
    monkeypatch.setattr(mp.subprocess, "Popen", lambda *a, **k: proc)
    with pytest.raises(mp.ModelPackError, match="cancelled"):
        mp._run_download_child(
            {"model_id": MODEL_ID}, stop=stop,
            deadline=time.monotonic() + 30,
            cancelled="model download cancelled",
            timed_out="model download timed out",
        )
    assert proc.returncode == -15  # terminated, then reaped


@pytest.mark.parametrize("corrupt", [False, True])
def test_hub_lfs_sha256_authenticates_downloaded_weights(tmp_path, monkeypatch, corrupt):
    from types import SimpleNamespace
    import huggingface_hub

    reviewed = b"verified"
    expected_sha = hashlib.sha256(reviewed).hexdigest()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "weights.bin").write_bytes(b"tampered" if corrupt else reviewed)
    metadata = SimpleNamespace(
        path="weights.bin", size=len(reviewed), blob_id="a" * 40,
        lfs={"sha256": expected_sha, "size": len(reviewed)}, type="file",
    )
    api = SimpleNamespace(list_repo_tree=lambda **kwargs: iter([metadata]))
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kwargs: api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(snapshot))
    staging = tmp_path / "staging"
    staging.mkdir()
    report = tmp_path / "report.json"
    spec = {
        **mp.normalize_hf_lock(_lock()),
        "staging": str(staging), "report": str(report),
    }

    if corrupt:
        with pytest.raises(mp._Refused):
            mp._child_download(spec)
        assert not report.exists()
    else:
        mp._child_download(spec)
        document = json.loads(report.read_text())
        assert document["files"] == [
            {"path": "weights.bin", "size": len(reviewed), "sha256": expected_sha}
        ]
