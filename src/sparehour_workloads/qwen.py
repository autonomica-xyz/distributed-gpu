"""Fixed adapter for a warm local vLLM OpenAI-compatible server.

The trusted route pins the loopback endpoint, the served-model alias, the
reviewed model id and its immutable 40-hex revision. A job supplies only
schema-bounded prompts and sampling knobs; it can never select the server,
the model, a revision, or a URL. Weights are delivered by the reviewed
model pack (see ``models/qwen.lock.json``) and file-hash verified by model
provisioning before the vLLM server is launched against that directory.

The executor talks to exactly one endpoint class:

- ``GET /v1/models``     -- served identity must equal the trusted alias.
- ``GET /is_sleeping``,
  ``POST /wake_up``      -- release/wake handoff for shared-GPU workers.
- ``POST /sleep``        -- ``release_gpu()`` frees VRAM (sleep level 1).
- ``POST /tokenize``     -- trusted chat template renders each prompt.
- ``POST /v1/completions`` -- one true batch request carrying every prompt
  as token ids, so generation order and per-item ``index`` come from the
  server, not from serialized client calls.

Sleep endpoints are registered only when the server runs with
``--enable-sleep-mode`` and ``VLLM_SERVER_DEV_MODE=1`` (vLLM 0.20.1). The
companion launch requirements live in ``deploy/llm/requirements.txt`` and
the route environment in ``workloads/llm-batch.yaml``.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from sparehour.agent import ExecutionError


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL_ID = "cyankiwi/Qwen3.5-9B-AWQ-4bit"
DEFAULT_MODEL_REVISION = "156edc4bbeb8d1910ee7be9196bafaf1bc052156"
DEFAULT_MODEL_PACK = "qwen3.5-9b-awq-156edc4"
DEFAULT_MAX_CONTEXT_TOKENS = 8192
DEFAULT_MAX_OUTPUT_TOKENS = 4096

# Mirrors the payload schema bounds in workloads/llm-batch.yaml; the
# executor enforces them again so direct plugin invocations are bounded.
MAX_PROMPTS = 64
MAX_PROMPT_CHARS = 131_072
MAX_BATCH_CHARS = 1_048_576
MAX_SAMPLING_TOKENS = 8192
MAX_STOP_SEQUENCES = 8
MAX_STOP_CHARS = 256

# Bounded streaming keeps a misbehaving server from exhausting worker disk
# or hanging a lease. Applies to every response body.
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
WAKE_POLL_SECONDS = 1.0
RELEASE_TIMEOUT_SECONDS = 60.0

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_FINISH_REASONS = {"stop", "length", "content_filter"}
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


@dataclass(frozen=True)
class _VllmConfig:
    base_url: str
    served_model: str
    model_id: str
    model_revision: str
    model_pack: str
    max_context_tokens: int
    max_output_tokens: int


def _resolve_config(environment: dict[str, Any]) -> _VllmConfig:
    """Bind the trusted route environment; raises executor_config_invalid."""
    base_url = str(environment.get("VLLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    try:
        parsed = httpx.URL(base_url)
    except httpx.InvalidURL as exc:
        raise ExecutionError(
            "executor_config_invalid", "VLLM_BASE_URL is not a valid URL"
        ) from exc
    if (
        parsed.scheme != "http"
        or parsed.host not in _LOOPBACK_HOSTS
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ExecutionError(
            "executor_config_invalid",
            "VLLM_BASE_URL must be a credential-free http loopback origin",
        )
    model_id = str(environment.get("VLLM_MODEL_ID") or DEFAULT_MODEL_ID)
    revision = str(environment.get("VLLM_MODEL_REVISION") or DEFAULT_MODEL_REVISION)
    if not _REVISION_RE.match(revision):
        raise ExecutionError(
            "executor_config_invalid",
            "VLLM_MODEL_REVISION must be an immutable 40-hex commit",
        )
    served_model = str(
        environment.get("VLLM_SERVED_MODEL")
        or environment.get("VLLM_MODEL_PACK")
        or DEFAULT_MODEL_PACK
    )
    if not served_model:
        raise ExecutionError(
            "executor_config_invalid", "VLLM_SERVED_MODEL must be non-empty"
        )
    try:
        max_context = int(
            environment.get("VLLM_MAX_CONTEXT_TOKENS") or DEFAULT_MAX_CONTEXT_TOKENS
        )
        max_output = int(
            environment.get("VLLM_MAX_OUTPUT_TOKENS") or DEFAULT_MAX_OUTPUT_TOKENS
        )
    except (TypeError, ValueError) as exc:
        raise ExecutionError(
            "executor_config_invalid", f"invalid vLLM token bounds: {exc}"
        ) from exc
    if not (1 <= max_output < max_context):
        raise ExecutionError(
            "executor_config_invalid",
            "VLLM_MAX_OUTPUT_TOKENS must be smaller than VLLM_MAX_CONTEXT_TOKENS",
        )
    return _VllmConfig(
        base_url=base_url,
        served_model=served_model,
        model_id=model_id,
        model_revision=revision,
        model_pack=str(environment.get("VLLM_MODEL_PACK") or DEFAULT_MODEL_PACK),
        max_context_tokens=max_context,
        max_output_tokens=max_output,
    )


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ExecutionError(
            "job_deadline_exceeded", "job exceeded its executor deadline"
        )
    return remaining


def _request(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    deadline: float,
    params: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
) -> tuple[int, bytes]:
    """Issue one bounded request; classify transport failures only.

    Error classification never copies the server body into the message: a
    misbehaving endpoint could echo prompts, headers, or internals.
    """
    try:
        with client.stream(
            method,
            path,
            params=params,
            json=json_body,
            timeout=_remaining(deadline),
        ) as response:
            status = response.status_code
            size = 0
            chunks: list[bytes] = []
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES:
                    raise ExecutionError(
                        "output_validation_failed",
                        f"vLLM response for {path} exceeds {MAX_RESPONSE_BYTES} bytes",
                    )
                chunks.append(chunk)
            _remaining(deadline)
    except httpx.HTTPError as exc:
        raise ExecutionError("model_server_unavailable", type(exc).__name__) from exc
    return status, b"".join(chunks)


def _parse_json(status: int, body: bytes, *, path: str) -> Any:
    if status == 404:
        raise ExecutionError(
            "model_server_error",
            f"vLLM endpoint {path} is missing (sleep/tokenize support required)",
        )
    if status >= 500:
        raise ExecutionError(
            "model_server_error", f"vLLM {path} returned HTTP {status}"
        )
    if status >= 400:
        raise ExecutionError(
            "prompt_rejected", f"vLLM {path} rejected the request (HTTP {status})"
        )
    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ExecutionError(
            "model_server_error", f"vLLM {path} returned malformed JSON"
        ) from exc


def _served_model_ids(client: httpx.Client, *, deadline: float) -> set[str]:
    """Readiness probe that never touches the GPU: model listing only."""
    status, body = _request(client, "GET", "/v1/models", deadline=deadline)
    document = _parse_json(status, body, path="/v1/models")
    data = document.get("data")
    if not isinstance(data, list):
        raise ExecutionError(
            "model_server_error", "vLLM /v1/models response has no data list"
        )
    ids: set[str] = set()
    for entry in data:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ExecutionError(
                "model_server_error", "vLLM /v1/models entry is malformed"
            )
        ids.add(entry["id"])
    return ids


def _is_sleeping(client: httpx.Client, *, deadline: float) -> bool:
    status, body = _request(client, "GET", "/is_sleeping", deadline=deadline)
    document = _parse_json(status, body, path="/is_sleeping")
    sleeping = document.get("is_sleeping") if isinstance(document, dict) else None
    if not isinstance(sleeping, bool):
        raise ExecutionError(
            "model_server_error", "vLLM /is_sleeping response is malformed"
        )
    return sleeping


def _wake(client: httpx.Client, *, deadline: float) -> None:
    status, _ = _request(client, "POST", "/wake_up", deadline=deadline)
    if status >= 400:
        raise ExecutionError(
            "model_server_error", f"vLLM /wake_up returned HTTP {status}"
        )
    while _is_sleeping(client, deadline=deadline):
        _remaining(deadline)
        time.sleep(WAKE_POLL_SECONDS)


def _tokenize_prompt(
    client: httpx.Client,
    config: _VllmConfig,
    prompt: str,
    *,
    deadline: float,
) -> tuple[list[int], int]:
    """Render the prompt through the model's own chat template.

    ``enable_thinking`` is disabled so bounded outputs carry the answer,
    not a truncated reasoning stream. ``max_model_len`` reported by the
    server bounds the per-prompt context budget.
    """
    status, body = _request(
        client,
        "POST",
        "/tokenize",
        deadline=deadline,
        json_body={
            "model": config.served_model,
            "messages": [{"role": "user", "content": prompt}],
            "add_generation_prompt": True,
            "add_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    document = _parse_json(status, body, path="/tokenize")
    if not isinstance(document, dict):
        raise ExecutionError(
            "model_server_error", "vLLM /tokenize response is malformed"
        )
    tokens = document.get("tokens")
    count = document.get("count")
    max_model_len = document.get("max_model_len")
    if (
        not isinstance(tokens, list)
        or not tokens
        or not all(isinstance(t, int) and not isinstance(t, bool) for t in tokens)
        or not isinstance(max_model_len, int)
        or max_model_len <= 0
    ):
        raise ExecutionError(
            "model_server_error", "vLLM /tokenize returned inconsistent token data"
        )
    return tokens, max_model_len


class VllmBatchExecutor:
    """Run one ordered prompt batch against the pinned warm vLLM server."""

    def __init__(self) -> None:
        # Retained across execute() calls by the agent so release_gpu() can
        # reach the same trusted endpoint without re-reading a job payload.
        self._config: _VllmConfig | None = None

    def release_gpu(self, gpu_index: int) -> None:
        """Ask the warm server to offload weights (vLLM sleep level 1).

        gpu_index is unused: the pinned local server owns the slot's GPU.
        A failed request cannot prove VRAM is free: fail closed rather than
        starting another model beside a possibly still-resident server.
        """
        config = self._config
        if config is None:
            return
        deadline = time.monotonic() + RELEASE_TIMEOUT_SECONDS
        with httpx.Client(base_url=config.base_url, trust_env=False) as client:
            status, _ = _request(
                client,
                "POST",
                "/sleep",
                deadline=deadline,
                params={"level": "1"},
            )
        if not 200 <= status < 300:
            raise ExecutionError(
                "model_server_error", f"vLLM /sleep returned HTTP {status}"
            )

    def execute(
        self,
        *,
        job: dict[str, Any],
        workdir: Path,
        gpu_index: int,
    ) -> list[dict[str, Any]]:
        # gpu_index is intentionally unused: the pinned model service owns
        # the GPU for this route; the slot only gates concurrent claims.
        payload = job["payload"]
        config = _resolve_config(job["executor"].get("environment") or {})
        if self._config is not None and self._config != config:
            self.release_gpu(gpu_index)
        self._config = config

        prompts = payload["prompts"]
        if (
            not isinstance(prompts, list)
            or not prompts
            or len(prompts) > MAX_PROMPTS
            or not all(isinstance(item, str) for item in prompts)
        ):
            raise ExecutionError(
                "payload_invalid",
                f"prompts must be an array of 1-{MAX_PROMPTS} strings",
            )
        total_chars = 0
        for prompt in prompts:
            if not prompt or len(prompt) > MAX_PROMPT_CHARS:
                raise ExecutionError(
                    "payload_invalid",
                    f"each prompt must be 1-{MAX_PROMPT_CHARS} characters",
                )
            total_chars += len(prompt)
        if total_chars > MAX_BATCH_CHARS:
            raise ExecutionError(
                "payload_invalid",
                f"prompts exceed {MAX_BATCH_CHARS} total characters",
            )

        try:
            max_output_tokens = int(payload["max_output_tokens"])
        except (TypeError, ValueError) as exc:
            raise ExecutionError(
                "payload_invalid", "max_output_tokens must be an integer"
            ) from exc
        if not (1 <= max_output_tokens <= MAX_SAMPLING_TOKENS):
            raise ExecutionError(
                "payload_invalid",
                f"max_output_tokens must be 1-{MAX_SAMPLING_TOKENS}",
            )
        if max_output_tokens > config.max_output_tokens:
            raise ExecutionError(
                "prompt_rejected",
                f"max_output_tokens exceeds route bound {config.max_output_tokens}",
            )

        try:
            temperature = float(payload.get("temperature", 0.7))
            top_p = float(payload.get("top_p", 0.8))
        except (TypeError, ValueError) as exc:
            raise ExecutionError(
                "payload_invalid", "temperature and top_p must be numbers"
            ) from exc
        if not (0.0 <= temperature <= 2.0):
            raise ExecutionError("payload_invalid", "temperature must be in [0, 2]")
        if not (0.0 < top_p <= 1.0):
            raise ExecutionError("payload_invalid", "top_p must be in (0, 1]")
        seed = payload.get("seed")
        if seed is not None and (
            not isinstance(seed, int) or isinstance(seed, bool)
        ):
            raise ExecutionError("payload_invalid", "seed must be an integer")
        stop = payload.get("stop")
        if stop is not None and (
            not isinstance(stop, list)
            or len(stop) > MAX_STOP_SEQUENCES
            or not all(isinstance(s, str) and 0 < len(s) <= MAX_STOP_CHARS for s in stop)
        ):
            raise ExecutionError(
                "payload_invalid",
                f"stop must be up to {MAX_STOP_SEQUENCES} short strings",
            )

        deadline = time.monotonic() + float(
            job["executor"].get("timeout_seconds") or 1800
        )

        with httpx.Client(
            base_url=config.base_url, timeout=60.0, trust_env=False
        ) as client:
            served = _served_model_ids(client, deadline=deadline)
            if served != {config.served_model}:
                raise ExecutionError(
                    "model_identity_mismatch",
                    f"vLLM serves {sorted(served)!r}, expected only "
                    f"{config.served_model!r}",
                )

            if _is_sleeping(client, deadline=deadline):
                _wake(client, deadline=deadline)

            prompt_tokens: list[list[int]] = []
            context_budget = config.max_context_tokens
            for position, prompt in enumerate(prompts):
                tokens, server_max_len = _tokenize_prompt(
                    client, config, prompt, deadline=deadline
                )
                context_budget = min(context_budget, server_max_len)
                if len(tokens) + max_output_tokens > context_budget:
                    raise ExecutionError(
                        "context_length_exceeded",
                        f"prompt {position} needs {len(tokens)} tokens; "
                        f"with {max_output_tokens} output tokens it exceeds "
                        f"the {context_budget} token context budget",
                    )
                prompt_tokens.append(tokens)

            request_body: dict[str, Any] = {
                "model": config.served_model,
                "prompt": prompt_tokens,
                "n": 1,
                "stream": False,
                "max_tokens": max_output_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            if seed is not None:
                request_body["seed"] = seed
            if stop:
                request_body["stop"] = stop
            status, body = _request(
                client,
                "POST",
                "/v1/completions",
                deadline=deadline,
                json_body=request_body,
            )
            document = _parse_json(status, body, path="/v1/completions")

        if not isinstance(document, dict):
            raise ExecutionError(
                "model_server_error", "vLLM /v1/completions response is malformed"
            )
        if document.get("model") != config.served_model:
            raise ExecutionError(
                "model_identity_mismatch",
                "vLLM completion reported a different served model",
            )
        choices = document.get("choices")
        if not isinstance(choices, list) or len(choices) != len(prompts):
            raise ExecutionError(
                "model_server_error",
                "vLLM returned an incomplete choice list",
            )
        ordered: list[dict[str, Any] | None] = [None] * len(prompts)
        for choice in choices:
            if not isinstance(choice, dict):
                raise ExecutionError(
                    "model_server_error", "vLLM choice entry is malformed"
                )
            index = choice.get("index")
            text = choice.get("text")
            finish_reason = choice.get("finish_reason")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < len(prompts)
                or ordered[index] is not None
                or not isinstance(text, str)
                or finish_reason not in _FINISH_REASONS
            ):
                raise ExecutionError(
                    "model_server_error",
                    "vLLM choices are missing, duplicated, or malformed",
                )
            ordered[index] = {
                "index": index,
                "prompt_sha256": hashlib.sha256(
                    prompts[index].encode("utf-8")
                ).hexdigest(),
                "prompt_chars": len(prompts[index]),
                "prompt_tokens": len(prompt_tokens[index]),
                "text": text,
                "finish_reason": finish_reason,
            }
        if any(item is None for item in ordered):
            raise ExecutionError(
                "model_server_error", "vLLM omitted at least one batch choice"
            )

        usage = document.get("usage")
        if not isinstance(usage, dict):
            raise ExecutionError(
                "model_server_error", "vLLM completion has no usage block"
            )
        clean_usage: dict[str, int] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ExecutionError(
                    "model_server_error", f"vLLM usage field {key!r} is malformed"
                )
            clean_usage[key] = value

        artifact = {
            "model_pack": config.model_pack,
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "served_model": config.served_model,
            "context_budget_tokens": context_budget,
            "prompt_count": len(prompts),
            "sampling": {
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
                "max_output_tokens": max_output_tokens,
                "stop": stop or [],
            },
            "usage": clean_usage,
            "responses": ordered,
        }
        output = workdir / "responses.json"
        encoded = json.dumps(artifact, ensure_ascii=False).encode("utf-8")
        if len(encoded) > 64 * 1024 * 1024:
            raise ExecutionError(
                "output_validation_failed", "responses artifact exceeds 64 MiB"
            )
        output.write_bytes(encoded)
        return [
            {"name": "responses", "path": output, "media_type": "application/json"}
        ]
