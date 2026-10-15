"""Fixed SDXL Diffusers executor.

The workload route controls the model: either a hub ``model_id`` plus an
immutable 40-hex revision, or a controller-provisioned local snapshot via
``SDXL_MODEL_DIR`` whose ``manifest.json`` must match the pinned route
identity. A job can vary bounded inference parameters, but it cannot name
code, a checkpoint, a LoRA, or a workflow graph.

At most one pipeline is retained per GPU: a route change on the same slot
releases the previous model before loading, so sequential mixed workloads on
a 16 GiB card do not accumulate resident weights. ``release_gpu`` lets the
agent drop everything cached on a slot when the handler changes or the slot
shuts down.
"""

from __future__ import annotations

import gc
import json
import re
import threading
from pathlib import Path
from typing import Any

from PIL.PngImagePlugin import PngInfo

from gpu_orchestrator.agent import ExecutionError


DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_MODEL_REVISION = "76d28af79639c28a79fa5c6c6468febd3490a37e"
DEFAULT_VARIANT = "fp16"

_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_PIPELINE_INDEX_NAME = "model_index.json"
_PIPELINE_INDEX_CLASSES = {"StableDiffusionXLPipeline"}

# (gpu_index, source, revision, variant) -> pipeline. source is the local
# snapshot path or the hub model id. Bounded to one pipeline per gpu_index.
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
        variant = environment.get("SDXL_MODEL_VARIANT", DEFAULT_VARIANT)
        model_dir = environment.get("SDXL_MODEL_DIR") or None
        model_pack = environment.get("SDXL_MODEL_PACK") or None
        cache_dir = environment.get("HF_HOME", "/models/huggingface")
        width = int(payload.get("width", 1024))
        height = int(payload.get("height", 1024))
        device = f"cuda:{gpu_index}"

        if not _REVISION_PATTERN.match(revision):
            raise ExecutionError(
                "executor_contract_invalid",
                f"SDXL revision must be an immutable 40-hex commit, got {revision!r}",
            )

        source = model_id
        if model_dir is not None:
            source = _verified_model_dir(
                Path(model_dir),
                model_id=model_id,
                revision=revision,
                model_pack=model_pack,
            )

        try:
            pipeline = _pipeline_for_gpu(
                gpu_index,
                source=source,
                revision=revision,
                variant=variant,
                model_dir=model_dir is not None,
                cache_dir=cache_dir,
            )
            generator = _seeded_generator(device, int(payload["seed"]))
            result = pipeline(
                prompt=payload["prompt"],
                negative_prompt=payload.get("negative_prompt", ""),
                width=width,
                height=height,
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

        images = getattr(result, "images", None)
        if not images:
            raise ExecutionError("output_validation_failed", "SDXL returned no image")
        image = images[0]
        if getattr(image, "size", None) != (width, height):
            raise ExecutionError(
                "output_validation_failed",
                f"SDXL returned {getattr(image, 'size', None)} instead of "
                f"{width}x{height}",
            )
        output = workdir / "image.png"
        metadata = PngInfo()
        metadata.add_text(
            "dgpu",
            json.dumps(
                {
                    "job_id": job["id"],
                    "model": model_id,
                    "revision": revision,
                    "variant": variant,
                    "seed": int(payload["seed"]),
                    "prompt": payload["prompt"],
                    "width": width,
                    "height": height,
                },
                sort_keys=True,
            ),
        )
        image.save(output, format="PNG", pnginfo=metadata)
        return [{"name": "image", "path": output, "media_type": "image/png"}]

    def release_gpu(self, gpu_index: int) -> None:
        """Drop every pipeline cached on ``gpu_index`` and free device memory.

        Called by the agent when the slot's handler changes or the slot is
        released; also usable directly. Pipelines for other GPUs are retained.
        """
        _release_gpu_pipelines(gpu_index)


def _verified_model_dir(
    model_dir: Path, *, model_id: str, revision: str, model_pack: str | None
) -> str:
    """Hash-verify a provisioned snapshot against the pinned route identity.

    Delegates to ``model_packs.verify_hf_pack``, which re-checks every file's
    recorded SHA-256/size in ``manifest.json`` before the weights are touched.
    The executor additionally requires the snapshot to look like a Diffusers
    SDXL pipeline before handing it to ``from_pretrained``.
    """
    if not model_pack:
        raise ExecutionError(
            "executor_contract_invalid",
            "SDXL_MODEL_DIR requires SDXL_MODEL_PACK so the pack identity is "
            "verifiable",
        )
    root = model_dir.absolute()

    from gpu_orchestrator_workloads import model_packs

    try:
        model_packs.verify_hf_pack(
            root, model_pack=model_pack, model_id=model_id, revision=revision
        )
    except ExecutionError:
        raise
    except Exception as exc:
        raise ExecutionError(
            "model_load_failed", f"SDXL pack verification failed: {exc}"
        ) from exc

    index_path = root / _PIPELINE_INDEX_NAME
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionError(
            "model_load_failed", f"SDXL snapshot lacks model_index.json: {exc}"
        ) from exc
    if index.get("_class_name") not in _PIPELINE_INDEX_CLASSES:
        raise ExecutionError(
            "model_load_failed",
            f"SDXL snapshot declares pipeline {index.get('_class_name')!r}",
        )
    return str(root)


def _pipeline_for_gpu(
    gpu_index: int,
    *,
    source: str,
    revision: str,
    variant: str,
    model_dir: bool,
    cache_dir: str,
) -> Any:
    key = (gpu_index, source, revision, variant)
    with _PIPELINE_LOCK:
        pipeline = _PIPELINES.get(key)
        if pipeline is None:
            # One warm model per GPU: release other pipelines resident on this
            # slot before loading so a route switch frees VRAM first.
            evict = [existing for existing in _PIPELINES if existing[0] == gpu_index]
            for existing in evict:
                del _PIPELINES[existing]
            if evict:
                _clear_cuda_cache()
            pipeline = _load_pipeline(
                source=source,
                revision=revision,
                variant=variant,
                model_dir=model_dir,
                cache_dir=cache_dir,
                device=f"cuda:{gpu_index}",
            )
            _PIPELINES[key] = pipeline
        return pipeline


def _release_gpu_pipelines(gpu_index: int) -> None:
    with _PIPELINE_LOCK:
        for key in list(_PIPELINES):
            if key[0] == gpu_index:
                del _PIPELINES[key]
    _clear_cuda_cache()


def _clear_cuda_cache() -> None:
    """Flush allocator memory only after the last cached model reference dies."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_pipeline(
    *,
    source: str,
    revision: str,
    variant: str,
    model_dir: bool,
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
    if model_dir:
        # Verified local snapshot: no network, no revision resolution.
        pipeline = DiffusionPipeline.from_pretrained(
            source,
            variant=variant,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,
        )
    else:
        pipeline = DiffusionPipeline.from_pretrained(
            source,
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
