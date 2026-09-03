"""Fixed, offline Argos-package translation through CTranslate2.

Jobs provide already segmented text. The trusted route fixes the language pair,
model pack, compute profile, and model location. This intentionally avoids the
Argos package manager and sentence-boundary stack at runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from typing import Any

from gpu_orchestrator.agent import ExecutionError


_MAX_TOTAL_CHARACTERS = 8 * 1024 * 1024
_RUNTIMES: OrderedDict[tuple[str, str, int, str], tuple[Any, Any]] = OrderedDict()
_RUNTIME_LOCK = threading.Lock()


class ArgosTranslateExecutor:
    def execute(
        self,
        *,
        job: dict[str, Any],
        workdir: Path,
        gpu_index: int,
    ) -> list[dict[str, Any]]:
        payload = job["payload"]
        segments = payload["segments"]
        total_characters = sum(len(segment) for segment in segments)
        if total_characters > _MAX_TOTAL_CHARACTERS:
            raise ExecutionError(
                "executor_contract_invalid",
                f"segment batch exceeds {_MAX_TOTAL_CHARACTERS} characters",
            )

        environment = job["executor"].get("environment") or {}
        model_dir = Path(environment["ARGOS_MODEL_DIR"])
        model_pack = environment["ARGOS_MODEL_PACK"]
        expected_archive_sha256 = environment["ARGOS_MODEL_ARCHIVE_SHA256"]
        source_language = environment["ARGOS_SOURCE_LANGUAGE"]
        target_language = environment["ARGOS_TARGET_LANGUAGE"]
        configured_device = environment.get("ARGOS_DEVICE", "cuda")
        device = os.getenv("DGPU_TRANSLATE_DEVICE_OVERRIDE", configured_device)
        if device not in {"cpu", "cuda"}:
            raise ExecutionError("executor_contract_invalid", f"invalid device {device!r}")
        compute_type = environment[
            "ARGOS_GPU_COMPUTE_TYPE" if device == "cuda" else "ARGOS_CPU_COMPUTE_TYPE"
        ]
        beam_size = int(environment.get("ARGOS_BEAM_SIZE", "1"))
        max_batch_size = int(environment.get("ARGOS_MAX_BATCH_SIZE", "1024"))
        max_cached_models = int(environment.get("ARGOS_MAX_CACHED_MODELS_PER_GPU", "1"))
        if not 1 <= max_cached_models <= 4:
            raise ExecutionError(
                "executor_contract_invalid",
                "ARGOS_MAX_CACHED_MODELS_PER_GPU must be between 1 and 4",
            )

        started = time.monotonic()
        try:
            translator, tokenizer = _runtime_for_device(
                model_dir=model_dir,
                model_pack=model_pack,
                expected_archive_sha256=expected_archive_sha256,
                source_language=source_language,
                target_language=target_language,
                device=device,
                gpu_index=gpu_index,
                compute_type=compute_type,
                max_cached_models=max_cached_models,
            )
            tokenized = [tokenizer.encode(segment, out_type=str) for segment in segments]
            results = translator.translate_batch(
                tokenized,
                max_batch_size=max_batch_size,
                batch_type="tokens",
                beam_size=beam_size,
                num_hypotheses=1,
                replace_unknowns=True,
                return_scores=False,
            )
            translations = [tokenizer.decode(result.hypotheses[0]) for result in results]
        except ExecutionError:
            raise
        except Exception as exc:
            message = str(exc)
            lowered = message.casefold()
            if "out of memory" in lowered or "cuda oom" in lowered:
                raise ExecutionError("gpu_oom", message) from exc
            if any(marker in lowered for marker in ("model", "cuda", "cublas", "compute type")):
                raise ExecutionError("model_load_failed", message) from exc
            raise ExecutionError("executor_failed", message) from exc

        if len(translations) != len(segments):
            raise ExecutionError(
                "output_validation_failed",
                f"translator returned {len(translations)} results for {len(segments)} segments",
            )
        target_tokens = [tokenizer.encode(text, out_type=str) for text in translations]
        document = {
            "job_id": job["id"],
            "source_language": source_language,
            "target_language": target_language,
            "model_pack": model_pack,
            "device": device,
            "compute_type": compute_type,
            "beam_size": beam_size,
            "segment_count": len(segments),
            "source_character_count": total_characters,
            "source_token_count": sum(len(tokens) for tokens in tokenized),
            "target_token_count": sum(len(tokens) for tokens in target_tokens),
            "elapsed_seconds": time.monotonic() - started,
            "translations": translations,
        }
        output = workdir / "translations.json"
        output.write_text(
            json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return [{"name": "translations", "path": output, "media_type": "application/json"}]


def _runtime_for_device(
    *,
    model_dir: Path,
    model_pack: str,
    expected_archive_sha256: str,
    source_language: str,
    target_language: str,
    device: str,
    gpu_index: int,
    compute_type: str,
    max_cached_models: int,
) -> tuple[Any, Any]:
    key = (str(model_dir), device, gpu_index if device == "cuda" else -1, compute_type)
    cache_slot = (device, gpu_index if device == "cuda" else -1)
    with _RUNTIME_LOCK:
        runtime = _RUNTIMES.get(key)
        if runtime is None:
            _verify_model_pack(
                model_dir,
                model_pack,
                expected_archive_sha256,
                source_language,
                target_language,
            )
            runtime = _load_runtime(
                model_dir=model_dir,
                source_language=source_language,
                target_language=target_language,
                device=device,
                gpu_index=gpu_index,
                compute_type=compute_type,
            )
            _RUNTIMES[key] = runtime
            while (
                sum(
                    1
                    for cached_key in _RUNTIMES
                    if (cached_key[1], cached_key[2]) == cache_slot
                )
                > max_cached_models
            ):
                oldest = next(
                    cached_key
                    for cached_key in _RUNTIMES
                    if (cached_key[1], cached_key[2]) == cache_slot
                )
                del _RUNTIMES[oldest]
        else:
            _RUNTIMES.move_to_end(key)
        return runtime


def _load_runtime(
    *,
    model_dir: Path,
    source_language: str,
    target_language: str,
    device: str,
    gpu_index: int,
    compute_type: str,
) -> tuple[Any, Any]:
    try:
        import ctranslate2
        import sentencepiece
    except ImportError as exc:
        raise ExecutionError(
            "executor_unavailable",
            "translation worker requires ctranslate2 and sentencepiece",
        ) from exc
    translator_args: dict[str, Any] = {
        "model_path": str(model_dir / "model"),
        "device": device,
        "compute_type": compute_type,
        "inter_threads": 1,
        "intra_threads": 2,
    }
    if device == "cuda":
        translator_args["device_index"] = gpu_index
    translator = ctranslate2.Translator(**translator_args)
    sentencepiece_model = model_dir / "sentencepiece.model"
    bpe_model = model_dir / "bpe.model"
    if sentencepiece_model.is_file():
        tokenizer = _SentencePieceTokenizer(
            sentencepiece.SentencePieceProcessor(model_file=str(sentencepiece_model))
        )
    elif bpe_model.is_file():
        tokenizer = _BpeTokenizer(bpe_model, source_language, target_language)
    else:
        raise ExecutionError(
            "model_load_failed", "model pack has no supported sentencepiece.model or bpe.model"
        )
    return translator, tokenizer


class _BpeTokenizer:
    """Thin equivalent of Argos' legacy BPE tokenizer without its runtime."""

    def __init__(self, model_file: Path, source_language: str, target_language: str):
        try:
            from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer
            from subword_nmt.apply_bpe import BPE
        except ImportError as exc:
            raise ExecutionError(
                "executor_unavailable",
                "BPE translation packs require sacremoses and subword-nmt",
            ) from exc
        self.normalizer = MosesPunctNormalizer(lang=source_language)
        self.tokenizer = MosesTokenizer(lang=source_language)
        self.detokenizer = MosesDetokenizer(lang=target_language)
        with model_file.open("r", encoding="utf-8") as handle:
            self.bpe = BPE(handle)

    def encode(self, sentence: str, out_type: type[str] = str) -> list[str]:
        if out_type is not str:
            raise TypeError("BPE tokenizer supports string tokens only")
        normalized = self.normalizer.normalize(sentence)
        tokens = self.tokenizer.tokenize(normalized, return_str=False)
        return self.bpe.segment_tokens(tokens)

    def decode(self, tokens: list[str]) -> str:
        merged = " ".join(tokens).replace("@@ ", "").split(" ")
        return self.detokenizer.detokenize(merged)


class _SentencePieceTokenizer:
    def __init__(self, processor: Any):
        self.processor = processor

    def encode(self, sentence: str, out_type: type[str] = str) -> list[str]:
        return self.processor.encode(sentence, out_type=out_type)

    def decode(self, tokens: list[str]) -> str:
        return (
            self.processor.decode_pieces(tokens)
            .replace("▁", " ")
            .replace("_", " ")
            .strip()
        )


def _verify_model_pack(
    model_dir: Path,
    model_pack: str,
    expected_archive_sha256: str,
    source_language: str,
    target_language: str,
) -> None:
    manifest_path = model_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExecutionError("model_load_failed", f"invalid model manifest: {exc}") from exc
    if manifest.get("model_pack") != model_pack:
        raise ExecutionError("model_load_failed", "model-pack ID does not match route")
    if manifest.get("archive_sha256") != expected_archive_sha256:
        raise ExecutionError("model_load_failed", "model archive digest does not match route")
    metadata = manifest.get("metadata") or {}
    if (
        metadata.get("from_code") != source_language
        or metadata.get("to_code") != target_language
    ):
        raise ExecutionError(
            "model_load_failed",
            f"model language pair is not {source_language}->{target_language}",
        )
    files = manifest.get("files") or []
    if not files:
        raise ExecutionError("model_load_failed", "model manifest has no files")
    seen_paths: set[PurePosixPath] = set()
    for item in files:
        relative = PurePosixPath(str(item.get("path", "")))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ExecutionError("model_load_failed", "model manifest contains unsafe path")
        if relative in seen_paths:
            raise ExecutionError("model_load_failed", "model manifest contains duplicate path")
        seen_paths.add(relative)
        path = model_dir.joinpath(*relative.parts)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ExecutionError("model_load_failed", f"model file unavailable: {relative}") from exc
        if len(data) != item.get("size") or hashlib.sha256(data).hexdigest() != item.get("sha256"):
            raise ExecutionError("model_load_failed", f"model file digest failed: {relative}")
