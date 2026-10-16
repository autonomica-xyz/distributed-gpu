"""Fixed adapter for a local CUDA LibreTranslate service.

Jobs provide one bounded text segment plus a schema-bounded source and target
language from the set the pinned service loads. The trusted route fixes the
local model-service endpoint, the wire format, and the response size bound. A
job never selects the model server, an API key, or any execution code.

Before the first translation on an endpoint the adapter probes the service's
``/languages`` listing and requires the requested pair to be loaded, so the
slot only serves jobs while the correct pinned models are actually resident.
All response reads are bounded: error bodies are capped and sanitized before
they are embedded in an ``ExecutionError``.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from sparehour.agent import ExecutionError


DEFAULT_ENDPOINT = "http://127.0.0.1:5001/translate"
DEFAULT_FORMAT = "text"
LANGUAGES_PATH = "/languages"

# Bounded streaming keeps a misbehaving service from exhausting worker disk.
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_ERROR_BYTES = 64 * 1024
MAX_ERROR_CHARS = 500
MAX_BATCH_SEGMENTS = 64
MAX_BATCH_CHARACTERS = 256_000
MAX_READY_ENDPOINTS = 8
PROBE_TIMEOUT_SECONDS = 15.0

_CONTROL_CHARS = re.compile(r"[^\x20-\x7e\r\n\t]")

# (languages_url, source, target) pairs confirmed loaded by /languages.
# Bounded: routes pin one endpoint, so this never grows under normal use.
_READY_PAIRS: set[tuple[str, str, str]] = set()
_READY_LOCK = threading.Lock()

def _languages_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    path, sep, _ = base.rpartition("/translate")
    if sep:
        return f"{path}{LANGUAGES_PATH}"
    return f"{base}{LANGUAGES_PATH}"



def _read_bounded_body(response: httpx.Response, limit: int) -> bytes:
    """Drain a streamed response body, rejecting anything over ``limit``."""
    size = 0
    chunks: list[bytes] = []
    for chunk in response.iter_bytes():
        size += len(chunk)
        if size > limit:
            raise ExecutionError(
                "output_validation_failed",
                f"LibreTranslate response exceeds {limit} bytes",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _error_detail(response: httpx.Response) -> str:
    """Bounded, sanitized excerpt of a failed response body."""
    body = _read_bounded_body(response, MAX_ERROR_BYTES)
    text = body.decode("utf-8", errors="replace")
    text = _CONTROL_CHARS.sub(" ", text)[:MAX_ERROR_CHARS]
    return text.strip()


def _ensure_ready(
    *,
    languages_url: str,
    source: str,
    target: str,
) -> None:
    """Confirm the pinned service has the pair's models actually loaded."""
    key = (languages_url, source, target)
    with _READY_LOCK:
        if key in _READY_PAIRS:
            return
    try:
        with httpx.stream(
            "GET", languages_url, timeout=PROBE_TIMEOUT_SECONDS, trust_env=False
        ) as response:
            if response.status_code != 200:
                _read_bounded_body(response, MAX_ERROR_BYTES)
                raise ExecutionError(
                    "model_server_unavailable",
                    f"LibreTranslate /languages returned HTTP "
                    f"{response.status_code}",
                )
            body = _read_bounded_body(response, MAX_RESPONSE_BYTES)
    except httpx.HTTPError as exc:
        raise ExecutionError("model_server_unavailable", str(exc)) from exc

    try:
        languages = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ExecutionError(
            "model_server_error", f"unexpected /languages response: {exc}"
        ) from exc
    if not isinstance(languages, list) or not all(
        isinstance(entry, dict)
        and isinstance(entry.get("code"), str)
        and isinstance(entry.get("targets"), list)
        for entry in languages
    ):
        raise ExecutionError(
            "model_server_error", "unexpected /languages response shape"
        )

    loaded_targets: set[str] = set()
    for entry in languages:
        if source != "auto" and entry["code"] != source:
            continue
        loaded_targets.update(str(item) for item in entry["targets"])
    if target not in loaded_targets:
        raise ExecutionError(
            "model_load_failed",
            f"LibreTranslate has no loaded model for {source!r} -> {target!r}",
        )
    with _READY_LOCK:
        if len(_READY_PAIRS) < MAX_READY_ENDPOINTS:
            _READY_PAIRS.add(key)




def _translate(
    *,
    endpoint: str,
    request_format: str,
    source: str,
    target: str,
    texts: str | list[str],
    timeout: float,
) -> str | list[str]:
    _ensure_ready(
        languages_url=_languages_url(endpoint), source=source, target=target
    )
    request_body = {
        "q": texts,
        "source": source,
        "target": target,
        "format": request_format,
    }
    try:
        with httpx.stream(
            "POST", endpoint, json=request_body, timeout=timeout, trust_env=False
        ) as response:
            if response.status_code >= 400:
                detail = _error_detail(response)
                if response.status_code >= 500:
                    raise ExecutionError(
                        "model_server_error",
                        f"LibreTranslate returned HTTP {response.status_code}: {detail}",
                    )
                raise ExecutionError(
                    "translation_rejected",
                    f"LibreTranslate rejected the input (HTTP {response.status_code}): {detail}",
                )
            body = _read_bounded_body(response, MAX_RESPONSE_BYTES)
    except httpx.HTTPError as exc:
        raise ExecutionError("model_server_unavailable", str(exc)) from exc

    try:
        document = json.loads(body)
        return document["translatedText"]
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as exc:
        raise ExecutionError(
            "model_server_error", f"unexpected LibreTranslate response: {exc}"
        ) from exc


class LibreTranslateExecutor:
    def execute(
        self,
        *,
        job: dict[str, Any],
        workdir: Path,
        gpu_index: int,
    ) -> list[dict[str, Any]]:
        # gpu_index is intentionally unused: the pinned model service owns the
        # GPU for this route; the slot merely gates concurrent claim capacity.
        payload = job["payload"]
        environment = job["executor"].get("environment") or {}
        endpoint = environment.get("LIBRETRANSLATE_ENDPOINT", DEFAULT_ENDPOINT)
        request_format = environment.get("LIBRETRANSLATE_FORMAT", DEFAULT_FORMAT)
        translated = _translate(
            endpoint=endpoint,
            request_format=request_format,
            source=str(payload["source"]),
            target=str(payload["target"]),
            texts=str(payload["text"]),
            timeout=float(job["executor"].get("timeout_seconds") or 180),
        )
        if not isinstance(translated, str):
            raise ExecutionError(
                "model_server_error", "translatedText is not a single text segment"
            )

        output = workdir / "translation.txt"
        output.write_text(translated, encoding="utf-8")
        return [{"name": "translation", "path": output, "media_type": "text/plain"}]


class LibreTranslateBatchExecutor:
    """Translate one ordered, same-language batch through a warm local service.

    The explicit batch is generic: producers retain ownership of item identity,
    while SPAREHOUR validates only ordered text segments and an allowlisted language
    pair. This amortizes leases, HTTP setup, artifacts, and queue transitions.
    """

    def execute(
        self,
        *,
        job: dict[str, Any],
        workdir: Path,
        gpu_index: int,
    ) -> list[dict[str, Any]]:
        del gpu_index
        payload = job["payload"]
        segments = [str(value) for value in payload["segments"]]
        if not 1 <= len(segments) <= MAX_BATCH_SEGMENTS:
            raise ExecutionError(
                "executor_contract_invalid",
                f"batch must contain 1-{MAX_BATCH_SEGMENTS} segments",
            )
        total_characters = sum(len(value) for value in segments)
        if total_characters > MAX_BATCH_CHARACTERS:
            raise ExecutionError(
                "executor_contract_invalid",
                f"batch exceeds {MAX_BATCH_CHARACTERS} characters",
            )

        environment = job["executor"].get("environment") or {}
        started = time.monotonic()
        translated = _translate(
            endpoint=environment.get("LIBRETRANSLATE_ENDPOINT", DEFAULT_ENDPOINT),
            request_format=environment.get("LIBRETRANSLATE_FORMAT", DEFAULT_FORMAT),
            source=str(payload["source"]),
            target=str(payload["target"]),
            texts=segments,
            timeout=float(job["executor"].get("timeout_seconds") or 600),
        )
        if not isinstance(translated, list) or len(translated) != len(segments):
            raise ExecutionError(
                "output_validation_failed",
                "LibreTranslate returned a batch with the wrong length",
            )
        if not all(isinstance(value, str) for value in translated):
            raise ExecutionError(
                "output_validation_failed",
                "LibreTranslate batch contains a non-string translation",
            )

        document = {
            "job_id": job["id"],
            "source_language": payload["source"],
            "target_language": payload["target"],
            "segment_count": len(segments),
            "source_character_count": total_characters,
            "elapsed_seconds": time.monotonic() - started,
            "translations": translated,
        }
        output = workdir / "translations.json"
        output.write_text(
            json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return [{"name": "translations", "path": output, "media_type": "application/json"}]
