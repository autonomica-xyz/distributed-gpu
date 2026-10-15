from __future__ import annotations

import hashlib
import json
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from gpu_orchestrator.agent import ExecutionError
from gpu_orchestrator_workloads import sdxl

MODEL_ID = sdxl.DEFAULT_MODEL_ID
REVISION = sdxl.DEFAULT_MODEL_REVISION
PACK = "sdxl-base-1.0-fp16-76d28af"


@pytest.fixture(autouse=True)
def clear_pipelines():
    sdxl._PIPELINES.clear()
    yield
    sdxl._PIPELINES.clear()


def _snapshot(root: Path, *, revision=REVISION, pipeline="StableDiffusionXLPipeline"):
    root.mkdir()
    (root / "model_index.json").write_text(json.dumps({"_class_name": pipeline}))
    (root / "weights.safetensors").write_bytes(b"reviewed test model bytes")
    files = []
    for name in ("model_index.json", "weights.safetensors"):
        data = (root / name).read_bytes()
        files.append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    (root / "manifest.json").write_text(json.dumps({
        "format": "huggingface", "model_pack": PACK, "model_id": MODEL_ID,
        "revision": revision, "files": files,
    }))
    return root


@pytest.fixture
def snapshot(tmp_path):
    return _snapshot(tmp_path / "model")


def _job(snapshot, **payload):
    return {
        "id": "job-image-1",
        "payload": {"prompt": "a blue lighthouse", "width": 512, "height": 512,
                    "steps": 10, "guidance_scale": 7.0, "seed": 42, **payload},
        "executor": {"environment": {
            "SDXL_MODEL_ID": MODEL_ID, "SDXL_MODEL_REVISION": REVISION,
            "SDXL_MODEL_PACK": PACK, "SDXL_MODEL_DIR": str(snapshot),
            "SDXL_MODEL_VARIANT": "fp16",
        }},
    }


class _Pipeline:
    def __call__(self, *, width, height, **kwargs):
        return SimpleNamespace(images=[Image.new("RGB", (width, height), "navy")])


@pytest.fixture
def loader(monkeypatch):
    loaded = []

    def load(**kwargs):
        pipeline = _Pipeline()
        loaded.append(weakref.ref(pipeline))
        return pipeline

    monkeypatch.setattr(sdxl, "_load_pipeline", load)
    monkeypatch.setattr(sdxl, "_seeded_generator", lambda device, seed: seed)
    return loaded


def test_png_dimensions_and_model_provenance(snapshot, loader, tmp_path):
    result = sdxl.SdxlExecutor().execute(job=_job(snapshot), workdir=tmp_path, gpu_index=0)
    assert result[0]["media_type"] == "image/png"
    with Image.open(result[0]["path"]) as image:
        image.load()
        assert image.format == "PNG" and image.mode == "RGB"
        assert image.size == (512, 512)
        provenance = json.loads(image.text["dgpu"])
    assert provenance == {
        "job_id": "job-image-1", "model": MODEL_ID, "revision": REVISION,
        "variant": "fp16", "seed": 42, "prompt": "a blue lighthouse",
        "width": 512, "height": 512,
    }


def test_warm_model_is_reused_and_reloaded_after_release(snapshot, loader, tmp_path):
    executor = sdxl.SdxlExecutor()
    executor.execute(job=_job(snapshot), workdir=tmp_path, gpu_index=0)
    executor.execute(job=_job(snapshot, seed=7), workdir=tmp_path, gpu_index=0)
    assert len(loader) == 1 and loader[0]() is not None
    executor.release_gpu(0)
    assert loader[0]() is None
    executor.execute(job=_job(snapshot), workdir=tmp_path, gpu_index=0)
    assert len(loader) == 2 and loader[1]() is not None


def test_model_switch_releases_previous_weights(snapshot, loader, tmp_path):
    executor = sdxl.SdxlExecutor()
    executor.execute(job=_job(snapshot), workdir=tmp_path, gpu_index=0)
    other = _snapshot(tmp_path / "other", revision="a" * 40)
    job = _job(other)
    job["executor"]["environment"]["SDXL_MODEL_REVISION"] = "a" * 40
    executor.execute(job=job, workdir=tmp_path, gpu_index=0)
    assert loader[0]() is None and loader[1]() is not None


def test_releasing_one_gpu_preserves_other_gpu_model(snapshot, loader, tmp_path):
    executor = sdxl.SdxlExecutor()
    executor.execute(job=_job(snapshot), workdir=tmp_path, gpu_index=0)
    executor.execute(job=_job(snapshot), workdir=tmp_path, gpu_index=1)
    executor.release_gpu(0)
    assert loader[0]() is None and loader[1]() is not None


@pytest.mark.parametrize("field,value", [
    ("model_pack", "wrong-pack"), ("model_id", "wrong/model"),
    ("revision", "b" * 40),
])
def test_wrong_snapshot_identity_blocks_loading(snapshot, monkeypatch, tmp_path, field, value):
    manifest = snapshot / "manifest.json"
    document = json.loads(manifest.read_text())
    document[field] = value
    manifest.write_text(json.dumps(document))
    monkeypatch.setattr(sdxl, "_load_pipeline", lambda **kwargs: pytest.fail("unverified model loaded"))
    with pytest.raises(ExecutionError) as caught:
        sdxl.SdxlExecutor().execute(job=_job(snapshot), workdir=tmp_path, gpu_index=0)
    assert caught.value.code == "model_load_failed"


def test_corrupt_model_bytes_block_loading(snapshot, monkeypatch, tmp_path):
    (snapshot / "weights.safetensors").write_bytes(b"unreviewed replacement")
    monkeypatch.setattr(sdxl, "_load_pipeline", lambda **kwargs: pytest.fail("corrupt model loaded"))
    with pytest.raises(ExecutionError) as caught:
        sdxl.SdxlExecutor().execute(job=_job(snapshot), workdir=tmp_path, gpu_index=0)
    assert caught.value.code == "model_load_failed"


def test_symlinked_snapshot_is_not_accepted(snapshot, tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(snapshot, target_is_directory=True)
    with pytest.raises(ExecutionError) as caught:
        sdxl.SdxlExecutor().execute(job=_job(alias), workdir=tmp_path, gpu_index=0)
    assert caught.value.code == "model_load_failed"


def test_wrong_pipeline_class_is_rejected(monkeypatch, tmp_path):
    root = _snapshot(tmp_path / "wrong", pipeline="AutoModelForCausalLM")
    monkeypatch.setattr(sdxl, "_load_pipeline", lambda **kwargs: pytest.fail("wrong pipeline loaded"))
    with pytest.raises(ExecutionError) as caught:
        sdxl.SdxlExecutor().execute(job=_job(root), workdir=tmp_path, gpu_index=0)
    assert caught.value.code == "model_load_failed"


@pytest.mark.parametrize("revision", ["main", "76d28af", "z" * 40, ""])
def test_mutable_or_malformed_revision_is_rejected(snapshot, tmp_path, revision):
    job = _job(snapshot)
    job["executor"]["environment"]["SDXL_MODEL_REVISION"] = revision
    with pytest.raises(ExecutionError) as caught:
        sdxl.SdxlExecutor().execute(job=job, workdir=tmp_path, gpu_index=0)
    assert caught.value.code == "executor_contract_invalid"


def test_local_snapshot_requires_pack_identity(snapshot, tmp_path):
    job = _job(snapshot)
    del job["executor"]["environment"]["SDXL_MODEL_PACK"]
    with pytest.raises(ExecutionError) as caught:
        sdxl.SdxlExecutor().execute(job=job, workdir=tmp_path, gpu_index=0)
    assert caught.value.code == "executor_contract_invalid"


@pytest.mark.parametrize("images", [[], [Image.new("RGB", (64, 64))]])
def test_incomplete_or_wrong_size_generation_is_rejected(snapshot, monkeypatch, tmp_path, images):
    monkeypatch.setattr(sdxl, "_load_pipeline", lambda **kwargs: lambda **options: SimpleNamespace(images=images))
    monkeypatch.setattr(sdxl, "_seeded_generator", lambda device, seed: seed)
    with pytest.raises(ExecutionError) as caught:
        sdxl.SdxlExecutor().execute(job=_job(snapshot), workdir=tmp_path, gpu_index=0)
    assert caught.value.code == "output_validation_failed"


def test_gpu_oom_keeps_retryable_error_class(snapshot, monkeypatch, tmp_path):
    def generate(**kwargs):
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(sdxl, "_load_pipeline", lambda **kwargs: generate)
    monkeypatch.setattr(sdxl, "_seeded_generator", lambda device, seed: seed)
    with pytest.raises(ExecutionError) as caught:
        sdxl.SdxlExecutor().execute(job=_job(snapshot), workdir=tmp_path, gpu_index=0)
    assert caught.value.code == "gpu_oom"


def test_gpu_model_dies_before_allocator_cache_is_flushed(monkeypatch):
    pipeline = _Pipeline()
    reference = weakref.ref(pipeline)
    sdxl._PIPELINES[(0, MODEL_ID, REVISION, "fp16")] = pipeline
    del pipeline
    observations = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, empty_cache=lambda: observations.append(reference() is None),
    )))
    sdxl.SdxlExecutor().release_gpu(0)
    assert observations == [True] and reference() is None
