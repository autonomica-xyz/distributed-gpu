from __future__ import annotations

import hashlib
import http.server
import json
import socket
import threading
from urllib.parse import parse_qs, urlparse

import pytest

from sparehour.agent import ExecutionError
from sparehour.registry import AdmissionError
from sparehour_workloads import qwen

SERVED_MODEL = "qwen3.5-9b-awq-156edc4"
MODEL_ID = "cyankiwi/Qwen3.5-9B-AWQ-4bit"
MODEL_REVISION = "156edc4bbeb8d1910ee7be9196bafaf1bc052156"


def _tokens_for(content: str) -> list[int]:
    # Deterministic stand-in for the model's own chat template rendering.
    return [ord(char) % 50_000 for char in content] or [0]


def _completion_document(prompts, served_model):
    choices = [
        {
            "index": index,
            "text": f"answer-{index}",
            "finish_reason": "stop",
            "logprobs": None,
        }
        for index in range(len(prompts))
    ]
    prompt_tokens = sum(len(item) for item in prompts)
    completion_tokens = len(prompts) * 3
    return {
        "id": "cmpl-stub",
        "object": "text_completion",
        "model": served_model,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class _StubVllm:
    """Loopback stand-in for the pinned warm vLLM server."""

    def __init__(
        self,
        *,
        served_model=SERVED_MODEL,
        sleeping=False,
        completion_status=200,
        completion_body=None,
        completion_choices=None,
        completion_model=None,
        max_model_len=8192,
        models_body=None,
        sleep_status=200,
        tokenize_status=200,
        tokenize_body=None,
    ):
        self.sleeping = sleeping
        self.requests: list[dict] = []
        service = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _reply(self, status, document):
                body = (
                    document
                    if isinstance(document, bytes)
                    else json.dumps(document).encode()
                )
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _body(self):
                length = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(length) or b"null")

            def do_GET(self):
                path = urlparse(self.path).path
                service.requests.append({"method": "GET", "path": path})
                if path == "/v1/models":
                    document = models_body or {
                        "object": "list",
                        "data": [{"id": served_model, "object": "model"}],
                    }
                    self._reply(200, document)
                elif path == "/is_sleeping":
                    self._reply(200, {"is_sleeping": service.sleeping})
                else:
                    self._reply(404, {"error": "unknown"})

            def do_POST(self):
                parsed = urlparse(self.path)
                path = parsed.path
                body = self._body()
                service.requests.append(
                    {
                        "method": "POST",
                        "path": path,
                        "query": parse_qs(parsed.query),
                        "body": body,
                    }
                )
                if path == "/wake_up":
                    service.sleeping = False
                    self._reply(200, b"")
                elif path == "/sleep":
                    if sleep_status >= 400:
                        self._reply(sleep_status, {"error": "nope"})
                    else:
                        service.sleeping = True
                        self._reply(200, b"")
                elif path == "/tokenize":
                    if tokenize_status >= 400:
                        self._reply(tokenize_status, tokenize_body or b"rejected")
                        return
                    tokens = _tokens_for(body["messages"][0]["content"])
                    self._reply(
                        200,
                        {
                            "count": len(tokens),
                            "max_model_len": max_model_len,
                            "tokens": tokens,
                        },
                    )
                elif path == "/v1/completions":
                    if completion_status >= 400:
                        self._reply(
                            completion_status,
                            completion_body or b"server exploded",
                        )
                        return
                    if completion_body is not None:
                        self._reply(200, completion_body)
                        return
                    document = _completion_document(
                        body["prompt"], completion_model or served_model
                    )
                    if completion_choices is not None:
                        document["choices"] = completion_choices
                    self._reply(200, document)
                else:
                    self._reply(404, {"error": "unknown"})

            def log_message(self, *args):
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()

    def paths(self, path):
        return [r for r in self.requests if r["path"] == path]

    def close(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _job(base_url, payload, environment=None):
    env = {
        "VLLM_BASE_URL": base_url,
        "VLLM_MODEL_PACK": SERVED_MODEL,
        "VLLM_MODEL_ID": MODEL_ID,
        "VLLM_MODEL_REVISION": MODEL_REVISION,
        "VLLM_SERVED_MODEL": SERVED_MODEL,
        "VLLM_MODEL_DIR": "/models/qwen3.5-9b-awq-156edc4",
        "VLLM_MAX_CONTEXT_TOKENS": "8192",
        "VLLM_MAX_OUTPUT_TOKENS": "4096",
    }
    env.update(environment or {})
    return {
        "id": "job-llm-1",
        "payload": payload,
        "executor": {
            "type": "plugin",
            "timeout_seconds": 30,
            "environment": env,
        },
    }


def _payload(**overrides):
    payload = {
        "prompts": ["Capital of France?", "2 + 2?", "Say gracias"],
        "max_output_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    payload.update(overrides)
    return payload


def _execute(service, payload, tmp_path, environment=None):
    return qwen.VllmBatchExecutor().execute(
        job=_job(service.url, payload, environment), workdir=tmp_path, gpu_index=0
    )


def test_runs_ordered_batch_as_single_completion_request(tmp_path):
    service = _StubVllm()
    try:
        result = _execute(service, _payload(), tmp_path)
    finally:
        service.close()

    completions = service.paths("/v1/completions")
    assert len(completions) == 1
    body = completions[0]["body"]
    assert body["model"] == SERVED_MODEL
    assert body["n"] == 1 and body["stream"] is False
    assert body["max_tokens"] == 64
    assert len(body["prompt"]) == 3
    assert all(isinstance(tokens, list) for tokens in body["prompt"])
    assert body["prompt"][0] == _tokens_for("Capital of France?")

    tokenizes = service.paths("/tokenize")
    assert len(tokenizes) == 3
    for request in tokenizes:
        message = request["body"]["messages"][0]
        assert message["role"] == "user"
        assert request["body"]["chat_template_kwargs"] == {"enable_thinking": False}

    assert result == [
        {
            "name": "responses",
            "path": tmp_path / "responses.json",
            "media_type": "application/json",
        }
    ]
    document = json.loads((tmp_path / "responses.json").read_bytes())
    assert document["model_id"] == MODEL_ID
    assert document["model_revision"] == MODEL_REVISION
    assert document["served_model"] == SERVED_MODEL
    assert document["prompt_count"] == 3
    assert document["usage"]["total_tokens"] > 0
    responses = document["responses"]
    assert [item["index"] for item in responses] == [0, 1, 2]
    assert [item["text"] for item in responses] == [
        "answer-0",
        "answer-1",
        "answer-2",
    ]
    assert [item["finish_reason"] for item in responses] == ["stop"] * 3
    assert responses[0]["prompt_sha256"] == hashlib.sha256(
        b"Capital of France?"
    ).hexdigest()


def test_out_of_order_choices_are_reordered(tmp_path):
    choices = [
        {"index": 2, "text": "third", "finish_reason": "stop"},
        {"index": 0, "text": "first", "finish_reason": "length"},
        {"index": 1, "text": "second", "finish_reason": "stop"},
    ]
    service = _StubVllm(completion_choices=choices)
    try:
        _execute(service, _payload(), tmp_path)
    finally:
        service.close()
    document = json.loads((tmp_path / "responses.json").read_bytes())
    assert [item["text"] for item in document["responses"]] == [
        "first",
        "second",
        "third",
    ]
    assert document["responses"][0]["finish_reason"] == "length"


def test_served_model_identity_mismatch_blocks_generation(tmp_path):
    service = _StubVllm(served_model="unrelated-model")
    try:
        with pytest.raises(ExecutionError) as excinfo:
            _execute(service, _payload(), tmp_path)
    finally:
        service.close()
    assert excinfo.value.code == "model_identity_mismatch"
    assert service.paths("/v1/completions") == []
    assert service.paths("/tokenize") == []


def test_sleeping_server_is_woken_before_generation(tmp_path):
    service = _StubVllm(sleeping=True)
    try:
        _execute(service, _payload(), tmp_path)
        assert service.sleeping is False
    finally:
        service.close()
    assert len(service.paths("/wake_up")) == 1
    assert len(service.paths("/v1/completions")) == 1


def test_release_gpu_posts_sleep_level_one(tmp_path):
    service = _StubVllm()
    executor = qwen.VllmBatchExecutor()
    try:
        executor.execute(
            job=_job(service.url, _payload()), workdir=tmp_path, gpu_index=0
        )
        executor.release_gpu(0)
        assert service.sleeping is True
    finally:
        service.close()
    sleeps = service.paths("/sleep")
    assert len(sleeps) == 1
    assert sleeps[0]["query"]["level"] == ["1"]




def test_release_gpu_reports_answered_failure(tmp_path):
    service = _StubVllm(sleep_status=500)
    executor = qwen.VllmBatchExecutor()
    try:
        executor.execute(
            job=_job(service.url, _payload()), workdir=tmp_path, gpu_index=0
        )
        with pytest.raises(ExecutionError) as excinfo:
            executor.release_gpu(0)
    finally:
        service.close()
    assert excinfo.value.code == "model_server_error"


def test_unreachable_server_is_retryable_unavailability(tmp_path):
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    job = _job(f"http://127.0.0.1:{port}", _payload())
    with pytest.raises(ExecutionError) as excinfo:
        qwen.VllmBatchExecutor().execute(
            job=job, workdir=tmp_path, gpu_index=0
        )
    assert excinfo.value.code == "model_server_unavailable"


def test_server_error_never_leaks_response_body(tmp_path):
    service = _StubVllm(
        completion_status=500, completion_body=b"SECRET-INTERNAL-TRACE"
    )
    try:
        with pytest.raises(ExecutionError) as excinfo:
            _execute(service, _payload(), tmp_path)
    finally:
        service.close()
    assert excinfo.value.code == "model_server_error"
    assert "SECRET" not in str(excinfo.value)


def test_client_error_is_prompt_rejection(tmp_path):
    service = _StubVllm(
        completion_status=400, completion_body=b'{"error": {"message": "nope"}}'
    )
    try:
        with pytest.raises(ExecutionError) as excinfo:
            _execute(service, _payload(), tmp_path)
    finally:
        service.close()
    assert excinfo.value.code == "prompt_rejected"


def test_malformed_completion_is_server_error(tmp_path):
    service = _StubVllm(completion_body=b"<html>not json</html>")
    try:
        with pytest.raises(ExecutionError) as excinfo:
            _execute(service, _payload(), tmp_path)
    finally:
        service.close()
    assert excinfo.value.code == "model_server_error"


def test_oversized_response_is_rejected(tmp_path):
    service = _StubVllm(models_body=b" " * (qwen.MAX_RESPONSE_BYTES + 1))
    try:
        with pytest.raises(ExecutionError) as excinfo:
            _execute(service, _payload(), tmp_path)
    finally:
        service.close()
    assert excinfo.value.code == "output_validation_failed"


def test_context_budget_is_enforced_per_prompt(tmp_path):
    service = _StubVllm(max_model_len=20)
    try:
        with pytest.raises(ExecutionError) as excinfo:
            _execute(service, _payload(), tmp_path)
    finally:
        service.close()
    assert excinfo.value.code == "context_length_exceeded"
    assert service.paths("/v1/completions") == []


def test_incomplete_choices_are_server_error(tmp_path):
    choices = [
        {"index": 0, "text": "only", "finish_reason": "stop"},
        {"index": 1, "text": "two", "finish_reason": "stop"},
    ]
    service = _StubVllm(completion_choices=choices)
    try:
        with pytest.raises(ExecutionError) as excinfo:
            _execute(service, _payload(), tmp_path)
    finally:
        service.close()
    assert excinfo.value.code == "model_server_error"


def test_duplicate_choice_index_is_server_error(tmp_path):
    choices = [
        {"index": 0, "text": "a", "finish_reason": "stop"},
        {"index": 0, "text": "b", "finish_reason": "stop"},
        {"index": 1, "text": "c", "finish_reason": "stop"},
    ]
    service = _StubVllm(completion_choices=choices)
    try:
        with pytest.raises(ExecutionError) as excinfo:
            _execute(service, _payload(), tmp_path)
    finally:
        service.close()
    assert excinfo.value.code == "model_server_error"


def test_unknown_finish_reason_is_server_error(tmp_path):
    choices = [
        {"index": index, "text": "x", "finish_reason": reason}
        for index, reason in enumerate(["stop", "weird", "stop"])
    ]
    service = _StubVllm(completion_choices=choices)
    try:
        with pytest.raises(ExecutionError) as excinfo:
            _execute(service, _payload(), tmp_path)
    finally:
        service.close()
    assert excinfo.value.code == "model_server_error"


@pytest.mark.parametrize(
    "payload",
    [
        _payload(prompts=[]),
        _payload(prompts=["a", ""] ),
        _payload(prompts=["ok"] * 65),
        _payload(max_output_tokens=0),
        _payload(max_output_tokens=99999),
        _payload(temperature=9.0),
        _payload(top_p=0.0),
        _payload(seed="not-an-int"),
        _payload(stop=[""]),
    ],
)
def test_payload_bounds_are_enforced(tmp_path, payload):
    service = _StubVllm()
    try:
        with pytest.raises(ExecutionError) as excinfo:
            _execute(service, payload, tmp_path)
    finally:
        service.close()
    assert excinfo.value.code in {"payload_invalid", "prompt_rejected"}
    assert service.paths("/v1/completions") == []


def test_payload_cannot_override_model_or_endpoint(tmp_path):
    service = _StubVllm()
    payload = _payload(model="evil/other", base_url="http://169.254.169.254")
    try:
        result = _execute(service, payload, tmp_path)
    finally:
        service.close()
    body = service.paths("/v1/completions")[0]["body"]
    assert body["model"] == SERVED_MODEL
    document = json.loads(result[0]["path"].read_bytes())
    assert document["served_model"] == SERVED_MODEL


def test_non_loopback_base_url_is_rejected(tmp_path):
    service = _StubVllm()
    try:
        with pytest.raises(ExecutionError) as excinfo:
            _execute(
                service,
                _payload(),
                tmp_path,
                environment={"VLLM_BASE_URL": "http://169.254.169.254:8000"},
            )
    finally:
        service.close()
    assert excinfo.value.code == "executor_config_invalid"


@pytest.mark.parametrize("revision", ["main", "156edc4", "z" * 40])
def test_mutable_revision_is_rejected(tmp_path, revision):
    service = _StubVllm()
    try:
        with pytest.raises(ExecutionError) as excinfo:
            _execute(
                service,
                _payload(),
                tmp_path,
                environment={"VLLM_MODEL_REVISION": revision},
            )
    finally:
        service.close()
    assert excinfo.value.code == "executor_config_invalid"


def test_admission_binds_llm_route_contract(registry):
    admitted = registry.admit("llm-batch", _payload())
    assert admitted.requirements["capability"] == "llm-batch"
    assert admitted.requirements["model_packs"] == [SERVED_MODEL]
    env = admitted.executor["environment"]
    assert env["VLLM_MODEL_ID"] == MODEL_ID
    assert env["VLLM_MODEL_REVISION"] == MODEL_REVISION
    assert env["VLLM_SERVED_MODEL"] == SERVED_MODEL
    assert env["VLLM_BASE_URL"] == "http://127.0.0.1:8000"


@pytest.mark.parametrize(
    ("payload", "marker"),
    [
        (_payload(prompts=[]), "prompts"),
        (_payload(prompts=["ok"] * 65), "prompts"),
        (_payload(prompts=["a", ""]), "prompts.1"),
        (_payload(max_output_tokens=0), "max_output_tokens"),
        (_payload(temperature=2.5), "temperature"),
        (_payload(model="evil/other"), "Additional properties"),
        (_payload(base_url="http://evil"), "Additional properties"),
    ],
)
def test_admission_rejects_out_of_contract_payloads(registry, payload, marker):
    with pytest.raises(AdmissionError, match=marker):
        registry.admit("llm-batch", payload)


def test_local_model_prompts_never_reach_ambient_proxy(monkeypatch, tmp_path):
    service = _StubVllm()
    proxy = _StubVllm()
    try:
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                     "http_proxy", "https_proxy", "all_proxy"):
            monkeypatch.setenv(name, proxy.url)
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("no_proxy", "")
        _execute(service, _payload(prompts=["private model input"]), tmp_path)
        assert service.paths("/v1/completions")
        assert proxy.requests == []
    finally:
        proxy.close()
        service.close()


@pytest.mark.parametrize(
    "url",
    [
        "http://user:secret@127.0.0.1:8000",
        "http://127.0.0.1:8000?secret=value",
        "http://127.0.0.1:8000#secret",
        "http://127.0.0.1:8000/unexpected-prefix",
    ],
)
def test_model_origin_rejects_hidden_credentials_and_prefixes(url):
    with pytest.raises(ExecutionError) as excinfo:
        qwen._resolve_config({"VLLM_BASE_URL": url})
    assert excinfo.value.code == "executor_config_invalid"
    assert "secret" not in str(excinfo.value)
