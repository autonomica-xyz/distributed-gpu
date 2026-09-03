"""Fixed SDXL Diffusers executor.

The workload route controls the model and revision. A job can vary bounded
inference parameters, but it cannot name code, a checkpoint, a LoRA, or a
workflow graph. One module-level pipeline is retained per GPU so subsequent
jobs do not reload seven gigabytes of weights.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from PIL.PngImagePlugin import PngInfo

from gpu_orchestrator.agent import ExecutionError


DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_MODEL_REVISION = "76d28af79639c28a79fa5c6c6468febd3490a37e"

_PIPELINES: dict[tuple[int, str, str, str], Any] = {}
_PIPELINE_LOCK = threading.Lock()


class SdxlExecutor:
    def execute(
        self,
        *,
        job: dict[str, Any],
        workdir: Path,
        gpu_index: int,
    ) -> list[dict[str, Any]]:
        payload = job["payload"]
        environment = job["executor"].get("environment") or {}
        model_id = environment.get("SDXL_MODEL_ID", DEFAULT_MODEL_ID)
        revision = environment.get("SDXL_MODEL_REVISION", DEFAULT_MODEL_REVISION)
        variant = environment.get("SDXL_MODEL_VARIANT", "fp16")
        cache_dir = environment.get("HF_HOME", "/models/huggingface")
        device = f"cuda:{gpu_index}"

        try:
            pipeline = _pipeline_for_gpu(
                gpu_index,
                model_id=model_id,
                revision=revision,
                variant=variant,
                cache_dir=cache_dir,
            )
            generator = _seeded_generator(device, int(payload["seed"]))
            result = pipeline(
                prompt=payload["prompt"],
                negative_prompt=payload.get("negative_prompt", ""),
                width=int(payload.get("width", 1024)),
                height=int(payload.get("height", 1024)),
                num_inference_steps=int(payload.get("steps", 30)),
                guidance_scale=float(payload.get("guidance_scale", 7.0)),
                generator=generator,
            )
        except ExecutionError:
            raise
        except Exception as exc:
            message = str(exc)
            lowered = message.casefold()
            if "out of memory" in lowered or "cuda oom" in lowered:
                raise ExecutionError("gpu_oom", message) from exc
            if any(marker in lowered for marker in ("download", "connection", "huggingface")):
                raise ExecutionError("model_download_failed", message) from exc
            raise ExecutionError("executor_failed", message) from exc

        if not getattr(result, "images", None):
            raise ExecutionError("output_validation_failed", "SDXL returned no image")
        output = workdir / "image.png"
        metadata = PngInfo()
        metadata.add_text(
            "dgpu",
            json.dumps(
                {
                    "job_id": job["id"],
                    "model": model_id,
                    "revision": revision,
                    "seed": int(payload["seed"]),
                    "prompt": payload["prompt"],
                },
                sort_keys=True,
            ),
        )
        result.images[0].save(output, format="PNG", pnginfo=metadata)
        return [{"name": "image", "path": output, "media_type": "image/png"}]


def _pipeline_for_gpu(
    gpu_index: int,
    *,
    model_id: str,
    revision: str,
    variant: str,
    cache_dir: str,
) -> Any:
    key = (gpu_index, model_id, revision, variant)
    with _PIPELINE_LOCK:
        pipeline = _PIPELINES.get(key)
        if pipeline is None:
            pipeline = _load_pipeline(
                model_id=model_id,
                revision=revision,
                variant=variant,
                cache_dir=cache_dir,
                device=f"cuda:{gpu_index}",
            )
            _PIPELINES[key] = pipeline
        return pipeline


def _load_pipeline(
    *,
    model_id: str,
    revision: str,
    variant: str,
    cache_dir: str,
    device: str,
) -> Any:
    try:
        import torch
        from diffusers import DiffusionPipeline
    except ImportError as exc:
        raise ExecutionError(
            "executor_unavailable",
            "SDXL worker requires torch and diffusers",
        ) from exc

    torch.backends.cuda.matmul.allow_tf32 = True
    pipeline = DiffusionPipeline.from_pretrained(
        model_id,
        revision=revision,
        variant=variant,
        torch_dtype=torch.float16,
        use_safetensors=True,
        cache_dir=cache_dir,
    )
    pipeline.enable_vae_slicing()
    pipeline.set_progress_bar_config(disable=True)
    return pipeline.to(device)


def _seeded_generator(device: str, seed: int) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ExecutionError("executor_unavailable", "SDXL worker requires torch") from exc
    return torch.Generator(device=device).manual_seed(seed)
