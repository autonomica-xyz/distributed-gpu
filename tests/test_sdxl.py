from __future__ import annotations

import json
from types import SimpleNamespace

from PIL import Image

from gpu_orchestrator_workloads import sdxl


def test_sdxl_executor_generates_png_and_reuses_pipeline(monkeypatch, tmp_path):
    calls = []

    class FakePipeline:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (64, 64), "navy")])

    loads = []

    def load_pipeline(**kwargs):
        loads.append(kwargs)
        return FakePipeline()

    monkeypatch.setattr(sdxl, "_load_pipeline", load_pipeline)
    monkeypatch.setattr(sdxl, "_seeded_generator", lambda device, seed: (device, seed))
    sdxl._PIPELINES.clear()
    job = {
        "id": "job-image-1",
        "payload": {
            "prompt": "a blue lighthouse",
            "negative_prompt": "text",
            "width": 1024,
            "height": 1024,
            "steps": 25,
            "guidance_scale": 6.5,
            "seed": 42,
        },
        "executor": {
            "environment": {
                "SDXL_MODEL_ID": "example/model",
                "SDXL_MODEL_REVISION": "abc123",
                "SDXL_MODEL_VARIANT": "fp16",
                "HF_HOME": "/models",
            }
        },
    }

    first = sdxl.SdxlExecutor().execute(job=job, workdir=tmp_path, gpu_index=0)
    second = sdxl.SdxlExecutor().execute(job=job, workdir=tmp_path, gpu_index=0)

    assert len(loads) == 1
    assert len(calls) == 2
    assert calls[0]["generator"] == ("cuda:0", 42)
    assert first[0]["media_type"] == "image/png"
    assert second[0]["path"] == tmp_path / "image.png"
    with Image.open(tmp_path / "image.png") as image:
        assert image.size == (64, 64)
        metadata = json.loads(image.text["dgpu"])
    assert metadata["model"] == "example/model"
    assert metadata["seed"] == 42
