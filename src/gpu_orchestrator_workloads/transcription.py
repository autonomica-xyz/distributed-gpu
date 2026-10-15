"""Fixed GPU transcription executors for the ``transcribe`` workload.

Two routes share this module:

* ``ParakeetExecutor`` — NVIDIA NeMo ``parakeet-tdt-0.6b-v3`` restored from a
  provisioned ``.nemo`` pack. It emits word and segment timestamps but has no
  hotword or diarization knobs, so asking for either is a contract error,
  never a silently ignored option.
* ``WhisperXExecutor`` — faster-whisper ``large-v3`` with forced wav2vec2
  word alignment and optional real pyannote speaker diarization, all loaded
  from provisioned local packs. No model name, checkpoint, token, or network
  access comes from a job.

Every model directory the route names is hash-verified against its
``manifest.json`` through ``model_packs.verify_hf_pack`` before weights load,
and the manifest identity must equal the route's pinned ``*_MODEL_PACK`` /
``*_MODEL_ID`` / ``*_MODEL_REVISION`` triple. At most one model per family is
kept warm per GPU; ``release_gpu`` drops every cache for one slot so the
agent can switch handlers on a 16 GiB card without leaking VRAM.
"""

from __future__ import annotations

import dataclasses
import gc
import importlib
import json
import math
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from gpu_orchestrator.agent import ExecutionError


DEFAULT_PARAKEET_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_PARAKEET_MODEL_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
DEFAULT_WHISPERX_MODEL_ID = "Systran/faster-whisper-large-v3"
DEFAULT_WHISPERX_MODEL_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
DEFAULT_ALIGN_MODEL_ID = "facebook/wav2vec2-base-960h"
DEFAULT_ALIGN_MODEL_REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
DEFAULT_DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-community-1"
DEFAULT_DIARIZATION_MODEL_REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"

AUTO_LANGUAGE = "auto"
AUDIO_INPUT_NAME = "audio"
SAMPLE_RATE = 16000

_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_LANGUAGE_PATTERN = re.compile(r"^[a-z]{2,3}$")

_MAX_AUDIO_SECONDS_DEFAULT = 14_400.0
_MAX_AUDIO_SECONDS_LIMIT = 86_400.0
_DECLARED_AUDIO_SLACK_SECONDS = 5.0
_MAX_HOTWORDS = 64
_MAX_HOTWORD_CHARACTERS = 2_000
_MAX_HOTWORD_LENGTH = 200
_MAX_SPEAKERS = 32
_MAX_PROBE_DRIFT_SECONDS = 1.0
_MAX_SEGMENT_OVERLAP_SECONDS = 1.0
_TRUE_VALUES = {"1", "true", "yes", "on"}

_PAYLOAD_FIELDS = {
    "audio_hours",
    "language",
    "hotwords",
    "diarize",
    "num_speakers",
    "min_speakers",
    "max_speakers",
}

# (gpu_index, model_dir, ...) -> loaded runtime. Bounded: entries for a GPU
# are dropped before a different model loads on that GPU and wholesale by
# release_gpu. Deleting dict entries drops the only strong reference so
# empty_cache() runs after the weights are actually unreachable.
_PARAKEET_MODELS: dict[tuple[int, str], Any] = {}
_WHISPERX_MODELS: dict[tuple[int, str, str], Any] = {}
_ALIGN_MODELS: dict[tuple[int, str, str], tuple[Any, dict[str, Any]]] = {}
_DIARIZE_PIPELINES: dict[tuple[int, str], Any] = {}
_ALL_CACHES = (
    _PARAKEET_MODELS,
    _WHISPERX_MODELS,
    _ALIGN_MODELS,
    _DIARIZE_PIPELINES,
)
_CACHE_LOCK = threading.Lock()


@dataclasses.dataclass(frozen=True)
class _JobSpec:
    language: str
    hotwords: tuple[str, ...]
    diarize: bool
    num_speakers: int | None
    min_speakers: int | None
    max_speakers: int | None
    audio_hours: float


class ParakeetExecutor:
    """NeMo parakeet-tdt-0.6b-v3 transcription with word timestamps."""

    def __init__(self) -> None:
        _preload_runtime("nemo.collections.asr")

    def execute(
        self,
        *,
        job: dict[str, Any],
        workdir: Path,
        gpu_index: int,
    ) -> list[dict[str, Any]]:
        spec = _normalize_payload(job.get("payload"))
        if spec.diarize:
            raise ExecutionError(
                "unsupported_option",
                "the parakeet route does not support speaker diarization; "
                "submit with diarize=false or use the whisperx route",
            )
        if spec.hotwords:
            raise ExecutionError(
                "unsupported_option",
                "the parakeet route does not support hotword biasing; "
                "omit hotwords or use the whisperx route",
            )

        environment = job["executor"].get("environment") or {}
        model_dir = _required_env(environment, "PARAKEET_MODEL_DIR")
        model_pack = _required_env(environment, "PARAKEET_MODEL_PACK")
        model_id = environment.get("PARAKEET_MODEL_ID") or DEFAULT_PARAKEET_MODEL_ID
        revision = _route_revision(
            environment, "PARAKEET_MODEL_REVISION", DEFAULT_PARAKEET_MODEL_REVISION
        )
        batch_size = _env_int(environment, "PARAKEET_BATCH_SIZE", 8, 1, 64)
        max_seconds = _env_seconds(
            environment, "PARAKEET_MAX_AUDIO_SECONDS", _MAX_AUDIO_SECONDS_DEFAULT
        )

        source = _materialized_input(job, workdir)
        duration = _probe_duration_seconds(source)
        _check_duration(duration, max_seconds, spec.audio_hours)
        decoded = workdir / "audio_16k.wav"
        _decode_audio_wav(source, decoded)

        started = time.monotonic()
        model = _parakeet_for_gpu(
            gpu_index,
            model_dir=model_dir,
            model_pack=model_pack,
            model_id=model_id,
            revision=revision,
        )
        try:
            results = model.transcribe(
                [str(decoded)], batch_size=batch_size, timestamps=True
            )
        except ExecutionError:
            raise
        except Exception as exc:
            _raise_classified("parakeet transcribe", exc)

        hypothesis = _single_hypothesis(results)
        text, detected_language, segments = _parakeet_result(hypothesis, duration)
        if (
            spec.language != AUTO_LANGUAGE
            and detected_language
            and detected_language != spec.language
        ):
            raise ExecutionError(
                "output_validation_failed",
                f"parakeet detected language {detected_language!r} but the job "
                f"declared {spec.language!r}",
            )
        language = detected_language or (
            spec.language if spec.language != AUTO_LANGUAGE else "und"
        )
        _validate_segments(segments, duration)

        document = _transcript_document(
            job=job,
            engine="parakeet",
            spec=spec,
            source=source,
            duration=duration,
            language=language,
            speaker_count=None,
            models=[
                {
                    "role": "asr",
                    "model_pack": model_pack,
                    "model_id": model_id,
                    "revision": revision,
                }
            ],
            segments=segments,
            text=text,
            gpu_index=gpu_index,
            elapsed_seconds=time.monotonic() - started,
        )
        return [_write_transcript(workdir, document)]

    def release_gpu(self, gpu_index: int) -> None:
        """Drop all transcription weights cached on ``gpu_index``."""
        _release_gpu_caches(gpu_index)


class WhisperXExecutor:
    """faster-whisper large-v3 + wav2vec2 alignment + optional diarization."""

    def __init__(self) -> None:
        _preload_runtime("whisperx.asr", "whisperx.alignment", "whisperx.diarize")

    def execute(
        self,
        *,
        job: dict[str, Any],
        workdir: Path,
        gpu_index: int,
    ) -> list[dict[str, Any]]:
        spec = _normalize_payload(job.get("payload"))
        environment = job["executor"].get("environment") or {}
        model_dir = _required_env(environment, "WHISPERX_MODEL_DIR")
        model_pack = _required_env(environment, "WHISPERX_MODEL_PACK")
        model_id = environment.get("WHISPERX_MODEL_ID") or DEFAULT_WHISPERX_MODEL_ID
        revision = _route_revision(
            environment, "WHISPERX_MODEL_REVISION", DEFAULT_WHISPERX_MODEL_REVISION
        )
        align_dir = _required_env(environment, "WHISPERX_ALIGN_MODEL_DIR")
        align_pack = _required_env(environment, "WHISPERX_ALIGN_MODEL_PACK")
        align_id = (
            environment.get("WHISPERX_ALIGN_MODEL_ID") or DEFAULT_ALIGN_MODEL_ID
        )
        align_revision = _route_revision(
            environment,
            "WHISPERX_ALIGN_MODEL_REVISION",
            DEFAULT_ALIGN_MODEL_REVISION,
        )
        align_languages = {
            code.strip()
            for code in _required_env(
                environment, "WHISPERX_ALIGN_LANGUAGES"
            ).split(",")
            if code.strip()
        }
        if not align_languages:
            raise ExecutionError(
                "executor_contract_invalid",
                "WHISPERX_ALIGN_LANGUAGES must list at least one language",
            )
        for code in align_languages:
            if not _LANGUAGE_PATTERN.match(code):
                raise ExecutionError(
                    "executor_contract_invalid",
                    f"WHISPERX_ALIGN_LANGUAGES entry {code!r} is not a language code",
                )
        if spec.language != AUTO_LANGUAGE and spec.language not in align_languages:
            raise ExecutionError(
                "unsupported_language",
                f"language {spec.language!r} has no provisioned alignment model "
                f"(supported: {sorted(align_languages)})",
            )

        diarization_dir = environment.get("DIARIZATION_MODEL_DIR") or ""
        if spec.diarize:
            if environment.get("WHISPERX_DIARIZATION", "").casefold() not in (
                _TRUE_VALUES
            ):
                raise ExecutionError(
                    "unsupported_option",
                    "this whisperx route does not enable speaker diarization",
                )
            if not diarization_dir:
                raise ExecutionError(
                    "executor_contract_invalid",
                    "route enables diarization but lacks DIARIZATION_MODEL_DIR",
                )
            diar_pack = _required_env(environment, "DIARIZATION_MODEL_PACK")
            diar_id = (
                environment.get("DIARIZATION_MODEL_ID")
                or DEFAULT_DIARIZATION_MODEL_ID
            )
            diar_revision = _route_revision(
                environment,
                "DIARIZATION_MODEL_REVISION",
                DEFAULT_DIARIZATION_MODEL_REVISION,
            )
        else:
            diar_pack = diar_id = diar_revision = ""

        compute_type = environment.get("WHISPERX_COMPUTE_TYPE") or "float16"
        batch_size = _env_int(environment, "WHISPERX_BATCH_SIZE", 16, 1, 64)
        max_seconds = _env_seconds(
            environment, "WHISPERX_MAX_AUDIO_SECONDS", _MAX_AUDIO_SECONDS_DEFAULT
        )

        source = _materialized_input(job, workdir)
        duration = _probe_duration_seconds(source)
        _check_duration(duration, max_seconds, spec.audio_hours)
        audio = _decode_audio_f32(source)

        started = time.monotonic()
        asr = _whisperx_for_gpu(
            gpu_index,
            model_dir=Path(model_dir),
            model_pack=model_pack,
            model_id=model_id,
            revision=revision,
            compute_type=compute_type,
        )
        whisperx_align = _alignment_fn()

        hotwords = ", ".join(spec.hotwords) if spec.hotwords else None
        restore_options = _apply_hotwords(asr, hotwords)
        try:
            result = asr.transcribe(
                audio,
                batch_size=batch_size,
                language=None if spec.language == AUTO_LANGUAGE else spec.language,
            )
        except ExecutionError:
            raise
        except Exception as exc:
            _raise_classified("whisperx transcribe", exc)
        finally:
            restore_options()

        language = str(result.get("language") or "").strip() or (
            spec.language if spec.language != AUTO_LANGUAGE else "und"
        )
        if language not in align_languages:
            raise ExecutionError(
                "unsupported_language",
                f"detected language {language!r} has no provisioned alignment "
                f"model (supported: {sorted(align_languages)})",
            )
        raw_segments = result.get("segments") or []

        if raw_segments:
            align_model, align_metadata = _align_for_gpu(
                gpu_index,
                model_dir=Path(align_dir),
                model_pack=align_pack,
                model_id=align_id,
                revision=align_revision,
                language=language,
            )
            try:
                aligned = whisperx_align(
                    raw_segments,
                    align_model,
                    align_metadata,
                    audio,
                    f"cuda:{gpu_index}",
                )
            except ExecutionError:
                raise
            except Exception as exc:
                _raise_classified("whisperx alignment", exc)
            aligned_segments = aligned.get("segments") or []
        else:
            aligned_segments = []

        speaker_count: int | None = None
        if spec.diarize:
            pipeline = _diarize_for_gpu(
                gpu_index,
                model_dir=Path(diarization_dir),
                model_pack=diar_pack,
                model_id=diar_id,
                revision=diar_revision,
            )
            try:
                diarize_df = pipeline(
                    audio,
                    num_speakers=spec.num_speakers,
                    min_speakers=spec.min_speakers,
                    max_speakers=spec.max_speakers,
                )
            except ExecutionError:
                raise
            except Exception as exc:
                _raise_classified("speaker diarization", exc)
            intervals = _diarize_intervals(diarize_df)
            if not intervals:
                raise ExecutionError(
                    "output_validation_failed",
                    "diarization produced no speaker segments",
                )
            speaker_count = _assign_speakers(aligned_segments, intervals)

        segments = _normalize_whisperx_segments(
            aligned_segments, diarize=spec.diarize
        )
        _validate_segments(segments, duration)

        models = [
            {
                "role": "asr",
                "model_pack": model_pack,
                "model_id": model_id,
                "revision": revision,
            },
            {
                "role": "alignment",
                "model_pack": align_pack,
                "model_id": align_id,
                "revision": align_revision,
            },
        ]
        if spec.diarize:
            models.append(
                {
                    "role": "diarization",
                    "model_pack": diar_pack,
                    "model_id": diar_id,
                    "revision": diar_revision,
                }
            )
        document = _transcript_document(
            job=job,
            engine="whisperx",
            spec=spec,
            source=source,
            duration=duration,
            language=language,
            speaker_count=speaker_count,
            models=models,
            segments=segments,
            text=" ".join(segment["text"] for segment in segments).strip(),
            gpu_index=gpu_index,
            elapsed_seconds=time.monotonic() - started,
        )
        return [_write_transcript(workdir, document)]

    def release_gpu(self, gpu_index: int) -> None:
        """Drop all transcription weights cached on ``gpu_index``."""
        _release_gpu_caches(gpu_index)


# ---------------------------------------------------------------------------
# Payload and environment validation


def _normalize_payload(payload: Any) -> _JobSpec:
    if not isinstance(payload, dict):
        raise ExecutionError("executor_contract_invalid", "job payload is not an object")
    unknown = sorted(set(payload) - _PAYLOAD_FIELDS)
    if unknown:
        raise ExecutionError(
            "executor_contract_invalid",
            f"unsupported payload fields {unknown}; the route accepts only "
            f"{sorted(_PAYLOAD_FIELDS)}",
        )

    language = payload.get("language", AUTO_LANGUAGE)
    if not isinstance(language, str) or (
        language != AUTO_LANGUAGE and not _LANGUAGE_PATTERN.match(language)
    ):
        raise ExecutionError(
            "executor_contract_invalid",
            f"language must be 'auto' or a 2-3 letter code, got {language!r}",
        )

    hotwords = _normalize_hotwords(payload.get("hotwords"))

    diarize = payload.get("diarize", False)
    if not isinstance(diarize, bool):
        raise ExecutionError(
            "executor_contract_invalid", f"diarize must be a boolean, got {diarize!r}"
        )
    num_speakers = _speaker_bound(payload, "num_speakers")
    min_speakers = _speaker_bound(payload, "min_speakers")
    max_speakers = _speaker_bound(payload, "max_speakers")
    if not diarize and any(
        value is not None for value in (num_speakers, min_speakers, max_speakers)
    ):
        raise ExecutionError(
            "executor_contract_invalid",
            "speaker bounds require diarize=true",
        )
    if (
        min_speakers is not None
        and max_speakers is not None
        and min_speakers > max_speakers
    ):
        raise ExecutionError(
            "executor_contract_invalid",
            f"min_speakers {min_speakers} exceeds max_speakers {max_speakers}",
        )
    if num_speakers is not None:
        if min_speakers is not None and num_speakers < min_speakers:
            raise ExecutionError(
                "executor_contract_invalid",
                f"num_speakers {num_speakers} is below min_speakers {min_speakers}",
            )
        if max_speakers is not None and num_speakers > max_speakers:
            raise ExecutionError(
                "executor_contract_invalid",
                f"num_speakers {num_speakers} exceeds max_speakers {max_speakers}",
            )

    audio_hours = payload.get("audio_hours")
    if isinstance(audio_hours, bool) or not isinstance(audio_hours, (int, float)):
        raise ExecutionError(
            "executor_contract_invalid", "audio_hours is required and must be a number"
        )
    audio_hours = float(audio_hours)
    if not math.isfinite(audio_hours) or audio_hours <= 0 or audio_hours > 24:
        raise ExecutionError(
            "executor_contract_invalid",
            f"audio_hours must be in (0, 24], got {audio_hours}",
        )

    return _JobSpec(
        language=language,
        hotwords=tuple(hotwords),
        diarize=diarize,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        audio_hours=audio_hours,
    )


def _normalize_hotwords(raw: Any) -> list[str]:
    if raw is None or raw is False:
        return []
    if isinstance(raw, str):
        items = [word.strip() for word in raw.split(",")]
    elif isinstance(raw, list):
        items = []
        for word in raw:
            if not isinstance(word, str):
                raise ExecutionError(
                    "executor_contract_invalid",
                    f"hotwords entries must be strings, got {word!r}",
                )
            items.append(word.strip())
    else:
        raise ExecutionError(
            "executor_contract_invalid",
            f"hotwords must be a comma-separated string or a string array, "
            f"got {type(raw).__name__}",
        )
    words = [word for word in items if word]
    if len(words) > _MAX_HOTWORDS:
        raise ExecutionError(
            "executor_contract_invalid",
            f"at most {_MAX_HOTWORDS} hotwords are allowed",
        )
    if sum(len(word) for word in words) > _MAX_HOTWORD_CHARACTERS:
        raise ExecutionError(
            "executor_contract_invalid",
            f"hotwords exceed {_MAX_HOTWORD_CHARACTERS} total characters",
        )
    for word in words:
        if len(word) > _MAX_HOTWORD_LENGTH:
            raise ExecutionError(
                "executor_contract_invalid",
                f"hotword exceeds {_MAX_HOTWORD_LENGTH} characters",
            )
    return words


def _speaker_bound(payload: dict[str, Any], name: str) -> int | None:
    value = payload.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExecutionError(
            "executor_contract_invalid", f"{name} must be an integer, got {value!r}"
        )
    if not 1 <= value <= _MAX_SPEAKERS:
        raise ExecutionError(
            "executor_contract_invalid",
            f"{name} must be in [1, {_MAX_SPEAKERS}], got {value}",
        )
    return value


def _required_env(environment: dict[str, Any], name: str) -> str:
    value = environment.get(name)
    if not value or not str(value).strip():
        raise ExecutionError(
            "executor_contract_invalid",
            f"route executor environment must set {name}",
        )
    return str(value)


def _route_revision(environment: dict[str, Any], name: str, default: str) -> str:
    revision = str(environment.get(name) or default)
    if not _REVISION_PATTERN.match(revision):
        raise ExecutionError(
            "executor_contract_invalid",
            f"{name} must be an immutable 40-hex commit, got {revision!r}",
        )
    return revision


def _env_int(
    environment: dict[str, Any], name: str, default: int, minimum: int, maximum: int
) -> int:
    raw = environment.get(name)
    if raw is None:
        return default
    try:
        value = int(str(raw))
    except ValueError:
        raise ExecutionError(
            "executor_contract_invalid", f"{name} must be an integer, got {raw!r}"
        ) from None
    if not minimum <= value <= maximum:
        raise ExecutionError(
            "executor_contract_invalid",
            f"{name} must be in [{minimum}, {maximum}], got {value}",
        )
    return value


def _env_seconds(environment: dict[str, Any], name: str, default: float) -> float:
    raw = environment.get(name)
    if raw is None:
        return default
    try:
        value = float(str(raw))
    except ValueError:
        raise ExecutionError(
            "executor_contract_invalid", f"{name} must be a number, got {raw!r}"
        ) from None
    if not math.isfinite(value) or not 0 < value <= _MAX_AUDIO_SECONDS_LIMIT:
        raise ExecutionError(
            "executor_contract_invalid",
            f"{name} must be in (0, {_MAX_AUDIO_SECONDS_LIMIT}], got {value}",
        )
    return value


# ---------------------------------------------------------------------------
# Materialized audio input


def _materialized_input(job: dict[str, Any], workdir: Path) -> Path:
    """Return the verified path of the single ``audio`` input.

    The agent materializes inputs under ``workdir/inputs`` and verifies the
    declared SHA-256 before executors run; the executor still enforces that
    the path stays inside that directory and matches the declared size so a
    crafted job cannot point the route at arbitrary files.
    """
    inputs = job.get("inputs") or []
    entries = [
        item for item in inputs if isinstance(item, dict) and item.get("name") == AUDIO_INPUT_NAME
    ]
    if len(entries) != 1:
        raise ExecutionError(
            "input_validation_failed",
            f"expected exactly one materialized {AUDIO_INPUT_NAME!r} input, "
            f"got {len(entries)}",
        )
    entry = entries[0]
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ExecutionError(
            "input_validation_failed", "audio input lacks a materialized path"
        )
    root = (workdir / "inputs").resolve()
    try:
        resolved = Path(raw_path).resolve()
    except OSError as exc:
        raise ExecutionError(
            "input_validation_failed", f"audio input path is not resolvable: {exc}"
        ) from exc
    if resolved != root and root not in resolved.parents:
        raise ExecutionError(
            "input_validation_failed",
            "audio input path is outside the materialized inputs directory",
        )
    try:
        stat_result = resolved.stat()
    except OSError as exc:
        raise ExecutionError(
            "input_validation_failed", f"materialized audio input is missing: {exc}"
        ) from exc
    if not resolved.is_file():
        raise ExecutionError(
            "input_validation_failed", "materialized audio input is not a file"
        )
    declared_size = entry.get("size")
    if declared_size is not None:
        if not isinstance(declared_size, int) or declared_size <= 0:
            raise ExecutionError(
                "input_validation_failed",
                f"audio input declares invalid size {declared_size!r}",
            )
        if stat_result.st_size != declared_size:
            raise ExecutionError(
                "input_validation_failed",
                f"audio input size {stat_result.st_size} does not match the "
                f"declared {declared_size} bytes",
            )
    return resolved


def _ffmpeg_binary(name: str) -> str:
    binary = shutil.which(name)
    if binary is None:
        raise ExecutionError(
            "executor_unavailable",
            f"worker image lacks {name}; audio decoding requires ffmpeg",
        )
    return binary


def _probe_duration_seconds(path: Path) -> float:
    command = [
        _ffmpeg_binary("ffprobe"),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=120
        )
    except subprocess.TimeoutExpired as exc:
        raise ExecutionError(
            "ffmpeg_decode_error", "ffprobe timed out on the audio input"
        ) from exc
    except OSError as exc:
        raise ExecutionError("ffmpeg_decode_error", f"ffprobe failed: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-500:]
        raise ExecutionError(
            "ffmpeg_decode_error", f"ffprobe rejected the audio input: {detail}"
        )
    try:
        duration = float(json.loads(completed.stdout)["format"]["duration"])
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ExecutionError(
            "ffmpeg_decode_error", "ffprobe did not report an audio duration"
        ) from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ExecutionError(
            "ffmpeg_decode_error", f"audio duration is not a positive number ({duration})"
        )
    return duration


def _check_duration(duration: float, max_seconds: float, audio_hours: float) -> None:
    if duration > max_seconds + _MAX_PROBE_DRIFT_SECONDS:
        raise ExecutionError(
            "input_validation_failed",
            f"audio duration {duration:.3f}s exceeds the route maximum "
            f"{max_seconds:.0f}s",
        )
    declared_seconds = audio_hours * 3600.0
    if duration > declared_seconds + _DECLARED_AUDIO_SLACK_SECONDS:
        raise ExecutionError(
            "input_validation_failed",
            f"audio duration {duration:.3f}s exceeds the declared "
            f"audio_hours={audio_hours:g}",
        )


def _decode_audio_wav(source: Path, destination: Path) -> None:
    command = [
        _ffmpeg_binary("ffmpeg"),
        "-y",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(source),
        "-vn",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    _run_decode(command, source)
    try:
        if destination.stat().st_size <= 44:
            raise ExecutionError(
                "ffmpeg_decode_error", "decoded audio produced an empty stream"
            )
    except OSError as exc:
        raise ExecutionError(
            "ffmpeg_decode_error", f"decoded audio is missing: {exc}"
        ) from exc


def _decode_audio_f32(source: Path) -> Any:
    """Decode to 16 kHz mono float32 samples for whisperx/pyannote."""
    command = [
        _ffmpeg_binary("ffmpeg"),
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(source),
        "-vn",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=3600)
    except subprocess.TimeoutExpired as exc:
        raise ExecutionError(
            "ffmpeg_decode_error", "ffmpeg decode timed out on the audio input"
        ) from exc
    except OSError as exc:
        raise ExecutionError("ffmpeg_decode_error", f"ffmpeg failed: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or b"").decode("utf-8", "replace").strip()[-500:]
        raise ExecutionError(
            "ffmpeg_decode_error", f"ffmpeg could not decode the audio input: {detail}"
        )
    if len(completed.stdout) % 4 or not completed.stdout:
        raise ExecutionError(
            "ffmpeg_decode_error", "ffmpeg produced a malformed PCM stream"
        )
    try:
        import numpy as np
    except ImportError as exc:
        raise ExecutionError(
            "executor_unavailable", "whisperx worker requires numpy"
        ) from exc
    return np.frombuffer(completed.stdout, dtype=np.float32).copy()


def _run_decode(command: list[str], source: Path) -> None:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=3600
        )
    except subprocess.TimeoutExpired as exc:
        raise ExecutionError(
            "ffmpeg_decode_error", "ffmpeg decode timed out on the audio input"
        ) from exc
    except OSError as exc:
        raise ExecutionError("ffmpeg_decode_error", f"ffmpeg failed: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-500:]
        raise ExecutionError(
            "ffmpeg_decode_error", f"ffmpeg could not decode the audio input: {detail}"
        )


# ---------------------------------------------------------------------------
# Model pack verification and warm caches


def _verify_pack(
    directory: Path, *, model_pack: str, model_id: str, revision: str
) -> Path:
    """Hash-verify a provisioned pack before any weights load.

    ``verify_hf_pack`` re-hashes every manifest file and rejects missing,
    extra, or unsafe files; any defect is a worker-side ``model_load_failed``
    so the agent drains instead of serving a tampered pack.
    """
    try:
        root = Path(directory).resolve(strict=True)
    except OSError as exc:
        raise ExecutionError(
            "model_load_failed", f"model pack directory {directory} is missing: {exc}"
        ) from exc
    try:
        from gpu_orchestrator_workloads.model_packs import verify_hf_pack
    except ImportError as exc:
        raise ExecutionError(
            "executor_unavailable", "model pack verification module is unavailable"
        ) from exc
    try:
        verify_hf_pack(
            root, model_pack=model_pack, model_id=model_id, revision=revision
        )
    except ExecutionError:
        raise
    except Exception as exc:
        raise ExecutionError(
            "model_load_failed",
            f"pack verification failed for {model_id}@{revision}: {exc}",
        ) from exc
    return root


def _parakeet_for_gpu(
    gpu_index: int,
    *,
    model_dir: Path,
    model_pack: str,
    model_id: str,
    revision: str,
) -> Any:
    key = (gpu_index, str(model_dir))
    with _CACHE_LOCK:
        model = _PARAKEET_MODELS.get(key)
        if model is not None:
            return model
        _evict_gpu_locked(gpu_index)
        root = _verify_pack(
            model_dir, model_pack=model_pack, model_id=model_id, revision=revision
        )
        model = _load_parakeet(model_dir=root, gpu_index=gpu_index)
        _PARAKEET_MODELS[key] = model
        return model


def _whisperx_for_gpu(
    gpu_index: int,
    *,
    model_dir: Path,
    model_pack: str,
    model_id: str,
    revision: str,
    compute_type: str,
) -> Any:
    key = (gpu_index, str(model_dir), compute_type)
    with _CACHE_LOCK:
        model = _WHISPERX_MODELS.get(key)
        if model is not None:
            return model
        # A parakeet model cannot share the card with a whisperx stack.
        _evict_cache_locked(_PARAKEET_MODELS, gpu_index)
        _evict_cache_locked(_WHISPERX_MODELS, gpu_index)
        root = _verify_pack(
            model_dir, model_pack=model_pack, model_id=model_id, revision=revision
        )
        model = _load_whisperx(
            model_dir=root, gpu_index=gpu_index, compute_type=compute_type
        )
        _WHISPERX_MODELS[key] = model
        return model


def _align_for_gpu(
    gpu_index: int,
    *,
    model_dir: Path,
    model_pack: str,
    model_id: str,
    revision: str,
    language: str,
) -> tuple[Any, dict[str, Any]]:
    key = (gpu_index, str(model_dir), language)
    with _CACHE_LOCK:
        cached = _ALIGN_MODELS.get(key)
        if cached is not None:
            return cached
        _evict_cache_locked(_PARAKEET_MODELS, gpu_index)
        _evict_cache_locked(_ALIGN_MODELS, gpu_index)
        root = _verify_pack(
            model_dir, model_pack=model_pack, model_id=model_id, revision=revision
        )
        loaded = _load_align_model(
            model_dir=root, gpu_index=gpu_index, language=language
        )
        _ALIGN_MODELS[key] = loaded
        return loaded


def _diarize_for_gpu(
    gpu_index: int,
    *,
    model_dir: Path,
    model_pack: str,
    model_id: str,
    revision: str,
) -> Any:
    key = (gpu_index, str(model_dir))
    with _CACHE_LOCK:
        pipeline = _DIARIZE_PIPELINES.get(key)
        if pipeline is not None:
            return pipeline
        _evict_cache_locked(_PARAKEET_MODELS, gpu_index)
        _evict_cache_locked(_DIARIZE_PIPELINES, gpu_index)
        root = _verify_pack(
            model_dir, model_pack=model_pack, model_id=model_id, revision=revision
        )
        pipeline = _load_diarization(model_dir=root, gpu_index=gpu_index)
        _DIARIZE_PIPELINES[key] = pipeline
        return pipeline


def _evict_cache_locked(cache: dict, gpu_index: int) -> None:
    """Delete this GPU's entries; values are dropped, not retained."""
    for key in [existing for existing in cache if existing[0] == gpu_index]:
        del cache[key]


def _evict_gpu_locked(gpu_index: int) -> None:
    for cache in _ALL_CACHES:
        _evict_cache_locked(cache, gpu_index)


def _release_gpu_caches(gpu_index: int) -> None:
    with _CACHE_LOCK:
        _evict_gpu_locked(gpu_index)
    # Every cache reference is gone at this point; only then ask the
    # allocator to return device memory.
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _preload_runtime(*modules: str) -> None:
    # Optional import managers (notably Optuna) retain deferred exception
    # tracebacks. Resolve them in model-free initialization frames, not from
    # execute/load frames that later acquire CUDA model references.
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError as exc:
            raise ExecutionError(
                "executor_unavailable", f"transcription runtime unavailable: {module}"
            ) from exc


def _load_parakeet(*, model_dir: Path, gpu_index: int) -> Any:
    try:
        import torch
        import nemo.collections.asr as nemo_asr
    except ImportError as exc:
        raise ExecutionError(
            "executor_unavailable",
            "parakeet worker requires torch and nemo_toolkit[asr]",
        ) from exc
    archives = sorted(model_dir.glob("*.nemo"))
    if len(archives) != 1:
        raise ExecutionError(
            "model_load_failed",
            f"parakeet pack must contain exactly one .nemo archive, "
            f"found {len(archives)}",
        )
    try:
        model = nemo_asr.models.ASRModel.restore_from(
            str(archives[0]), map_location="cpu"
        )
        model = model.to(torch.device(f"cuda:{gpu_index}"))
        model.eval()
    except ExecutionError:
        raise
    except Exception as exc:
        _raise_classified("parakeet model load", exc)
    return model


def _load_whisperx(*, model_dir: Path, gpu_index: int, compute_type: str) -> Any:
    try:
        import whisperx
    except ImportError as exc:
        raise ExecutionError(
            "executor_unavailable",
            "whisperx worker requires the pinned whisperx runtime",
        ) from exc
    try:
        return whisperx.load_model(
            str(model_dir),
            device="cuda",
            device_index=gpu_index,
            compute_type=compute_type,
            vad_method="pyannote",
            local_files_only=True,
        )
    except ExecutionError:
        raise
    except Exception as exc:
        _raise_classified("whisperx model load", exc)


def _load_align_model(
    *, model_dir: Path, gpu_index: int, language: str
) -> tuple[Any, dict[str, Any]]:
    try:
        import nltk
        import whisperx
    except ImportError as exc:
        raise ExecutionError(
            "executor_unavailable",
            "whisperx alignment requires nltk and whisperx",
        ) from exc
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError as exc:
        raise ExecutionError(
            "executor_unavailable",
            "worker image lacks NLTK punkt_tab tokenizer data required by "
            "whisperx alignment; bake it into the image (offline workers must "
            "never call nltk.download)",
        ) from exc
    try:
        return whisperx.load_align_model(
            language_code=language,
            device=f"cuda:{gpu_index}",
            model_name=str(model_dir),
            model_cache_only=True,
        )
    except ExecutionError:
        raise
    except Exception as exc:
        _raise_classified("alignment model load", exc)


def _load_diarization(*, model_dir: Path, gpu_index: int) -> Any:
    try:
        from whisperx.diarize import DiarizationPipeline
    except ImportError as exc:
        raise ExecutionError(
            "executor_unavailable",
            "speaker diarization requires whisperx and pyannote-audio",
        ) from exc
    try:
        # Local provisioned pack: config.yaml resolves $model/... subfolder
        # weights relative to the pack directory; nothing authenticates or
        # downloads. A fresh gated fetch still needs an approved HF token at
        # provisioning time, never inside a job.
        return DiarizationPipeline(
            model_name=str(model_dir), token=None, device=f"cuda:{gpu_index}"
        )
    except ExecutionError:
        raise
    except Exception as exc:
        _raise_classified("diarization model load", exc)

def _alignment_fn() -> Any:
    try:
        from whisperx.alignment import align
    except ImportError as exc:
        raise ExecutionError(
            "executor_unavailable", "whisperx worker image lacks whisperx"
        ) from exc
    return align


def _apply_hotwords(asr: Any, hotwords: str | None):
    """Install per-job hotwords on the warm pipeline.

    faster-whisper only reads ``hotwords`` from the pipeline's
    ``TranscriptionOptions``, so the caller swaps them in for one
    ``transcribe`` call. Returns a zero-argument callable that restores the
    previous options (a no-op when hotwords were not applied).
    """

    def noop() -> None:
        return None

    options = getattr(asr, "options", None)
    if options is None or hotwords is None:
        return noop
    if dataclasses.is_dataclass(options) and not isinstance(options, type):
        asr.options = dataclasses.replace(options, hotwords=hotwords)

        def restore() -> None:
            asr.options = options

        return restore
    if hasattr(options, "hotwords"):
        previous = options.hotwords
        options.hotwords = hotwords

        def restore() -> None:
            options.hotwords = previous

        return restore
    return noop


# ---------------------------------------------------------------------------
# Result normalization and validation


def _single_hypothesis(results: Any) -> Any:
    if isinstance(results, tuple) and len(results) == 2:
        results = results[0]
    if not isinstance(results, (list, tuple)) or len(results) != 1:
        raise ExecutionError(
            "output_validation_failed",
            "parakeet did not return exactly one transcript hypothesis",
        )
    return results[0]


def _parakeet_result(
    hypothesis: Any, duration: float
) -> tuple[str, str | None, list[dict[str, Any]]]:
    text = getattr(hypothesis, "text", None)
    if text is None:
        text = str(hypothesis)
    text = str(text).strip()
    detected = getattr(hypothesis, "language", None)
    if isinstance(detected, str):
        detected = detected.strip() or None
    else:
        detected = None

    timestamps = getattr(hypothesis, "timestamp_dict", None)
    if not isinstance(timestamps, dict):
        timestamps = getattr(hypothesis, "timestamp", None)
    if not isinstance(timestamps, dict):
        timestamps = {}

    word_entries = timestamps.get("word") or []
    words = [
        {
            "word": str(entry.get("word", entry.get("segment", ""))).strip(),
            "start": float(entry.get("start", 0.0)),
            "end": float(entry.get("end", 0.0)),
            "speaker": None,
        }
        for entry in word_entries
        if isinstance(entry, dict)
    ]
    if text and not words:
        raise ExecutionError(
            "output_validation_failed",
            "parakeet returned a transcript without word timestamps",
        )

    segment_entries = timestamps.get("segment") or []
    segments: list[dict[str, Any]] = []
    for entry in segment_entries:
        if not isinstance(entry, dict):
            continue
        segments.append(
            {
                "start": float(entry.get("start", 0.0)),
                "end": float(entry.get("end", 0.0)),
                "text": str(
                    entry.get("segment", entry.get("text", ""))
                ).strip(),
                "speaker": None,
                "words": [],
            }
        )
    if text and not segments:
        segments = [
            {
                "start": 0.0,
                "end": duration,
                "text": text,
                "speaker": None,
                "words": [],
            }
        ]
    _place_words(segments, words)
    return text, detected, segments


def _normalize_whisperx_segments(
    raw_segments: Any, *, diarize: bool
) -> list[dict[str, Any]]:
    if not isinstance(raw_segments, list):
        raise ExecutionError(
            "output_validation_failed", "whisperx segments are not a list"
        )
    segments: list[dict[str, Any]] = []
    for raw in raw_segments:
        if not isinstance(raw, dict):
            raise ExecutionError(
                "output_validation_failed", "whisperx segment is not an object"
            )
        text = str(raw.get("text") or "").strip()
        try:
            start = float(raw["start"])
            end = float(raw["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExecutionError(
                "output_validation_failed",
                f"whisperx segment lacks numeric bounds: {raw!r}",
            ) from exc
        words = []
        for word in raw.get("words") or []:
            if not isinstance(word, dict) or "start" not in word:
                continue
            try:
                words.append(
                    {
                        "word": str(word.get("word", "")).strip(),
                        "start": float(word["start"]),
                        "end": float(word.get("end", word["start"])),
                        "speaker": str(word["speaker"])
                        if diarize and word.get("speaker") is not None
                        else None,
                    }
                )
            except (TypeError, ValueError):
                continue
        if not text:
            continue
        segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "speaker": str(raw["speaker"])
                if diarize and raw.get("speaker") is not None
                else None,
                "words": words,
            }
        )
    return segments


def _place_words(segments: list[dict[str, Any]], words: list[dict[str, Any]]) -> None:
    """Attach words to the segment that contains their midpoint."""
    for word in words:
        midpoint = (word["start"] + word["end"]) / 2.0
        for segment in segments:
            if segment["start"] - 1.0 <= midpoint <= segment["end"] + 1.0:
                segment["words"].append(word)
                break
        else:
            if segments:
                segments[-1]["words"].append(word)


def _validate_segments(segments: list[dict[str, Any]], duration: float) -> None:
    limit = duration + _MAX_PROBE_DRIFT_SECONDS
    previous_end = 0.0
    for segment in segments:
        start, end = segment["start"], segment["end"]
        if not (
            math.isfinite(start)
            and math.isfinite(end)
            and 0.0 <= start <= end <= limit
        ):
            raise ExecutionError(
                "output_validation_failed",
                f"segment bounds {start}..{end} exceed the audio duration "
                f"{duration:.3f}s",
            )
        if start < previous_end - _MAX_SEGMENT_OVERLAP_SECONDS:
            raise ExecutionError(
                "output_validation_failed",
                "segments are not monotonically ordered",
            )
        previous_end = max(previous_end, end)
        if not segment["text"]:
            raise ExecutionError(
                "output_validation_failed", "segment has empty text"
            )
        for word in segment["words"]:
            word_start, word_end = word["start"], word["end"]
            if not (
                math.isfinite(word_start)
                and math.isfinite(word_end)
                and word_start <= word_end
                and word_start >= start - _MAX_SEGMENT_OVERLAP_SECONDS
                and word_end <= end + _MAX_SEGMENT_OVERLAP_SECONDS
            ):
                raise ExecutionError(
                    "output_validation_failed",
                    f"word timing {word_start}..{word_end} is outside its "
                    f"segment {start}..{end}",
                )
            if not word["word"]:
                raise ExecutionError(
                    "output_validation_failed", "word entry has empty text"
                )


def _diarize_intervals(diarize_df: Any) -> list[tuple[float, float, str]]:
    intervals: list[tuple[float, float, str]] = []
    if diarize_df is None or len(diarize_df) == 0:
        return intervals
    for _, row in diarize_df.iterrows():
        try:
            intervals.append(
                (float(row["start"]), float(row["end"]), str(row["speaker"]))
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ExecutionError(
                "output_validation_failed",
                "diarization produced a malformed segment row",
            ) from exc
    return intervals


def _assign_speakers(
    segments: list[dict[str, Any]], intervals: list[tuple[float, float, str]]
) -> int:
    """Assign the dominant-overlap speaker to each segment and timed word.

    Segments or words with no direct overlap fall back to the nearest
    diarization midpoint so every emitted entry carries a real speaker label.
    Returns the number of distinct speakers observed.
    """
    speakers = {speaker for _, _, speaker in intervals}

    def dominant(start: float, end: float) -> str | None:
        totals: dict[str, float] = {}
        for segment_start, segment_end, speaker in intervals:
            overlap = min(end, segment_end) - max(start, segment_start)
            if overlap > 0:
                totals[speaker] = totals.get(speaker, 0.0) + overlap
        if totals:
            return max(totals.items(), key=lambda item: item[1])[0]
        midpoint = (start + end) / 2.0
        nearest = min(
            intervals,
            key=lambda item: abs(((item[0] + item[1]) / 2.0) - midpoint),
        )
        return nearest[2]

    for segment in segments:
        segment["speaker"] = dominant(
            float(segment["start"]), float(segment["end"])
        )
        for word in segment.get("words") or []:
            if "start" not in word:
                continue
            word["speaker"] = dominant(
                float(word["start"]), float(word.get("end", word["start"]))
            )
    return len(speakers)


def _transcript_document(
    *,
    job: dict[str, Any],
    engine: str,
    spec: _JobSpec,
    source: Path,
    duration: float,
    language: str,
    speaker_count: int | None,
    models: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    text: str,
    gpu_index: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    entry = next(
        item
        for item in job.get("inputs") or []
        if item.get("name") == AUDIO_INPUT_NAME
    )
    return {
        "job_id": job.get("id", ""),
        "engine": engine,
        "language": language,
        "language_source": "declared"
        if spec.language != AUTO_LANGUAGE
        else "detected",
        "audio": {
            "name": entry.get("name"),
            "sha256": entry.get("sha256"),
            "size": entry.get("size"),
            "media_type": entry.get("media_type"),
            "duration_seconds": round(duration, 3),
        },
        "audio_hours_declared": spec.audio_hours,
        "hotwords": list(spec.hotwords),
        "diarized": spec.diarize,
        "speaker_count": speaker_count,
        "models": models,
        "device": f"cuda:{gpu_index}",
        "elapsed_seconds": round(elapsed_seconds, 3),
        "segment_count": len(segments),
        "word_count": sum(len(segment["words"]) for segment in segments),
        "text": text,
        "segments": segments,
    }


def _write_transcript(workdir: Path, document: dict[str, Any]) -> dict[str, Any]:
    output = workdir / "transcript.json"
    output.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"name": "transcript", "path": output, "media_type": "application/json"}


def _raise_classified(stage: str, exc: Exception) -> None:
    if isinstance(exc, ExecutionError):
        raise exc
    raise ExecutionError(_classify(str(exc)), f"{stage}: {exc}") from exc


def _classify(message: str) -> str:
    lowered = message.casefold()
    if "out of memory" in lowered or "cuda oom" in lowered:
        return "gpu_oom"
    if any(
        marker in lowered
        for marker in (
            "cuda error",
            "cuda_error",
            "cuda driver",
            "cudnn",
            "cublas",
            "nccl",
            "illegal memory access",
            "device-side assert",
            "no kernel image",
            "device or resource busy",
        )
    ):
        return "cuda_context_failure"
    if any(
        marker in lowered
        for marker in (
            "gated repo",
            "restricted",
            "401",
            "403",
            "unauthorized",
            "not found",
            "no such file",
            "download",
            "connection",
            "timeout",
            "certificate",
            "ssl",
        )
    ):
        return "model_load_failed"
    return "executor_failed"
