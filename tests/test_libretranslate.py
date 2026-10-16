from __future__ import annotations

import fnmatch
import http.server
import json
import socket
import threading

import pytest

from sparehour.agent import ExecutionError
from sparehour.registry import AdmissionError
from sparehour_workloads import libretranslate


_DEFAULT_LANGUAGES = [
    {"code": "tr", "name": "Turkish", "targets": ["en", "de", "es"]},
    {"code": "en", "name": "English", "targets": ["tr", "de", "es"]},
    {"code": "de", "name": "German", "targets": ["en", "tr"]},
    {"code": "es", "name": "Spanish", "targets": ["en", "tr"]},
]


class _StubService:
    """Minimal loopback stand-in for a pinned local LibreTranslate service."""

    def __init__(self, *, status=200, body=b'{"translatedText": "Hello world"}',
                 content_type="application/json", languages=None,
                 languages_status=200):
        self._languages = languages if languages is not None else _DEFAULT_LANGUAGES
        self._languages_status = languages_status
        handler = _make_handler(self, status, body, content_type)
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/translate"
        self.requests = handler.requests
        self.language_probes = handler.language_probes
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _make_handler(service, status, body, content_type):
    class Handler(http.server.BaseHTTPRequestHandler):
        requests = []
        language_probes = []

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            Handler.requests.append(json.loads(self.rfile.read(length) or b"null"))
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            Handler.language_probes.append(self.path)
            payload = json.dumps(service._languages).encode()
            self.send_response(service._languages_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    return Handler

@pytest.fixture(autouse=True)
def _clear_ready_pairs():
    libretranslate._READY_PAIRS.clear()
    yield
    libretranslate._READY_PAIRS.clear()



def _job(endpoint, payload):
    return {
        "id": "job-translate-1",
        "payload": payload,
        "executor": {
            "type": "plugin",
            "timeout_seconds": 30,
            "environment": {"LIBRETRANSLATE_ENDPOINT": endpoint},
        },
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "Merhaba dünya", "source": "tr", "target": "en"},
        {"text": "Merhaba dünya", "source": "auto", "target": "de"},
        {"text": "Hello world", "source": "en", "target": "es"},
    ],
)
def test_translates_segment_for_any_bounded_pair(tmp_path, payload):
    service = _StubService()
    try:
        result = libretranslate.LibreTranslateExecutor().execute(
            job=_job(service.url, payload), workdir=tmp_path, gpu_index=0
        )
    finally:
        service.close()

    # The wire request is assembled by the trusted route; only the bounded
    # source/target pair and text come from the job.
    assert service.requests == [
        {
            "q": payload["text"],
            "source": payload["source"],
            "target": payload["target"],
            "format": "text",
        }
    ]
    assert result == [
        {
            "name": "translation",
            "path": tmp_path / "translation.txt",
            "media_type": "text/plain",
        }
    ]
    assert (tmp_path / "translation.txt").read_text(encoding="utf-8") == "Hello world"


def test_translates_ordered_batch_and_reports_metrics(tmp_path):
    body = json.dumps({"translatedText": ["Hello", "World"]}).encode()
    service = _StubService(body=body)
    payload = {"segments": ["Merhaba", "Dünya"], "source": "tr", "target": "en"}
    try:
        result = libretranslate.LibreTranslateBatchExecutor().execute(
            job=_job(service.url, payload), workdir=tmp_path, gpu_index=0
        )
    finally:
        service.close()

    assert service.requests[0]["q"] == payload["segments"]
    assert result[0]["name"] == "translations"
    document = json.loads((tmp_path / "translations.json").read_text())
    assert document["translations"] == ["Hello", "World"]
    assert document["segment_count"] == 2
    assert document["source_character_count"] == 12
    assert document["source_language"] == "tr"
    assert document["target_language"] == "en"


def test_batch_rejects_wrong_length_response(tmp_path):
    service = _StubService(body=b'{"translatedText": ["only one"]}')
    try:
        with pytest.raises(ExecutionError) as excinfo:
            libretranslate.LibreTranslateBatchExecutor().execute(
                job=_job(service.url, {"segments": ["a", "b"], "source": "tr", "target": "en"}),
                workdir=tmp_path,
                gpu_index=0,
            )
        assert excinfo.value.code == "output_validation_failed"
    finally:
        service.close()


def test_admission_binds_batch_units_to_segment_count(registry):
    admitted = registry.admit(
        "translate-text-batch",
        {"segments": ["bir", "iki", "üç"], "source": "tr", "target": "en"},
    )
    assert admitted.route == "libretranslate-batch-v1"
    assert admitted.cost_unit == "segment"
    assert admitted.estimated_units == 3
    assert admitted.requirements["capability"] == "libretranslate-batch"


def test_client_error_is_not_retryable(tmp_path):
    service = _StubService(status=400, body=b'{"error": "invalid language"}')
    try:
        with pytest.raises(ExecutionError) as excinfo:
            libretranslate.LibreTranslateExecutor().execute(
                job=_job(service.url, {"text": "hi", "source": "tr", "target": "en"}),
                workdir=tmp_path,
                gpu_index=0,
            )
        assert excinfo.value.code == "translation_rejected"
    finally:
        service.close()


def test_server_error_is_retryable(tmp_path):
    service = _StubService(status=503, body=b"overloaded")
    try:
        with pytest.raises(ExecutionError) as excinfo:
            libretranslate.LibreTranslateExecutor().execute(
                job=_job(service.url, {"text": "hi", "source": "tr", "target": "en"}),
                workdir=tmp_path,
                gpu_index=0,
            )
        assert excinfo.value.code == "model_server_error"
    finally:
        service.close()


def test_malformed_success_response_is_a_service_error(tmp_path):
    service = _StubService(body=b"<html>maintenance</html>", content_type="text/html")
    try:
        with pytest.raises(ExecutionError) as excinfo:
            libretranslate.LibreTranslateExecutor().execute(
                job=_job(service.url, {"text": "hi", "source": "tr", "target": "en"}),
                workdir=tmp_path,
                gpu_index=0,
            )
        assert excinfo.value.code == "model_server_error"
    finally:
        service.close()


def test_oversized_response_is_rejected(tmp_path):
    body = b'{"translatedText": "' + b"a" * (libretranslate.MAX_RESPONSE_BYTES + 1) + b'"}'
    service = _StubService(body=body)
    try:
        with pytest.raises(ExecutionError) as excinfo:
            libretranslate.LibreTranslateExecutor().execute(
                job=_job(service.url, {"text": "hi", "source": "tr", "target": "en"}),
                workdir=tmp_path,
                gpu_index=0,
            )
        assert excinfo.value.code == "output_validation_failed"
    finally:
        service.close()


def test_unreachable_service_is_retryable_unavailability(tmp_path):
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    with pytest.raises(ExecutionError) as excinfo:
        libretranslate.LibreTranslateExecutor().execute(
            job=_job(f"http://127.0.0.1:{port}/translate",
                     {"text": "hi", "source": "tr", "target": "en"}),
            workdir=tmp_path,
            gpu_index=0,
        )
    assert excinfo.value.code == "model_server_unavailable"


def test_admission_binds_the_segment_contract(registry):
    admitted = registry.admit(
        "translate-segment",
        {"text": "Merhaba dünya", "source": "tr", "target": "en"},
    )
    assert admitted.route == "libretranslate-v1"
    assert admitted.cost_unit == "segment"
    assert admitted.estimated_units == 1
    assert admitted.requirements["capability"] == "libretranslate"
    assert admitted.requirements["model_packs"] == ["libretranslate-1.9.6-bounded-sbd"]
    patterns = [value.casefold() for value in admitted.requirements["gpu_models"]]
    assert any(fnmatch.fnmatch("nvidia rtx a5000 laptop gpu", value) for value in patterns)
    assert any(fnmatch.fnmatch("nvidia geforce gtx 1080 ti", value) for value in patterns)
    assert admitted.requirements["min_compute_capability"] == "6.1"
    # The endpoint is fixed by the trusted route; only the bounded pair rides
    # in the payload.
    assert admitted.executor["environment"]["LIBRETRANSLATE_ENDPOINT"].startswith(
        "http://127.0.0.1:"
    )
    assert "LIBRETRANSLATE_TARGET" not in admitted.executor["environment"]


@pytest.mark.parametrize(
    "payload, marker",
    [
        ({"source": "tr", "target": "en"}, "text"),
        ({"text": "hi", "target": "en"}, "source"),
        ({"text": "hi", "source": "tr"}, "target"),
        ({"text": "", "source": "tr", "target": "en"}, "text"),
        ({"text": "hi", "source": "klingon", "target": "en"}, "source"),
        ({"text": "hi", "source": "tr", "target": "klingon"}, "target"),
        ({"text": "hi", "source": "tr", "target": "auto"}, "target"),
        ({"text": "hi", "source": "tr", "target": "en", "extra": 1}, "extra"),
    ],
)
def test_admission_rejects_out_of_contract_payloads(registry, payload, marker):
    with pytest.raises(AdmissionError, match=marker):
        registry.admit("translate-segment", payload)


def test_probes_languages_once_then_translates(tmp_path):
    service = _StubService()
    try:
        executor = libretranslate.LibreTranslateExecutor()
        job = _job(service.url, {"text": "hi", "source": "tr", "target": "en"})
        executor.execute(job=job, workdir=tmp_path, gpu_index=0)
        executor.execute(job=job, workdir=tmp_path, gpu_index=0)
    finally:
        service.close()
    assert service.language_probes == ["/languages"]
    assert len(service.requests) == 2


def test_unloaded_language_pair_fails_model_load(tmp_path):
    service = _StubService(
        languages=[{"code": "tr", "name": "Turkish", "targets": ["de"]}]
    )
    try:
        with pytest.raises(ExecutionError) as excinfo:
            libretranslate.LibreTranslateExecutor().execute(
                job=_job(service.url,
                         {"text": "hi", "source": "tr", "target": "en"}),
                workdir=tmp_path,
                gpu_index=0,
            )
        assert excinfo.value.code == "model_load_failed"
        assert service.requests == []
    finally:
        service.close()


def test_languages_probe_failure_is_unavailable(tmp_path):
    service = _StubService(languages_status=503)
    try:
        with pytest.raises(ExecutionError) as excinfo:
            libretranslate.LibreTranslateExecutor().execute(
                job=_job(service.url,
                         {"text": "hi", "source": "tr", "target": "en"}),
                workdir=tmp_path,
                gpu_index=0,
            )
        assert excinfo.value.code == "model_server_unavailable"
        assert service.requests == []
    finally:
        service.close()


def test_malformed_languages_response_is_service_error(tmp_path):
    service = _StubService(languages={"not": "a list"})
    try:
        with pytest.raises(ExecutionError) as excinfo:
            libretranslate.LibreTranslateExecutor().execute(
                job=_job(service.url,
                         {"text": "hi", "source": "tr", "target": "en"}),
                workdir=tmp_path,
                gpu_index=0,
            )
        assert excinfo.value.code == "model_server_error"
    finally:
        service.close()


def test_error_body_is_sanitized_and_bounded(tmp_path):
    detail = "x" * (libretranslate.MAX_ERROR_CHARS + 200) + "\x00SECRET\xff"
    service = _StubService(status=400, body=json.dumps({"error": detail}).encode())
    try:
        with pytest.raises(ExecutionError) as excinfo:
            libretranslate.LibreTranslateExecutor().execute(
                job=_job(service.url,
                         {"text": "hi", "source": "tr", "target": "en"}),
                workdir=tmp_path,
                gpu_index=0,
            )
        assert excinfo.value.code == "translation_rejected"
        assert "\x00" not in str(excinfo.value)
        assert len(str(excinfo.value)) < libretranslate.MAX_ERROR_CHARS + 200
    finally:
        service.close()


def test_oversized_error_body_is_bounded(tmp_path):
    body = b'{"error": "' + b"a" * (libretranslate.MAX_ERROR_BYTES + 1) + b'"}'
    service = _StubService(status=500, body=body)
    try:
        with pytest.raises(ExecutionError) as excinfo:
            libretranslate.LibreTranslateExecutor().execute(
                job=_job(service.url,
                         {"text": "hi", "source": "tr", "target": "en"}),
                workdir=tmp_path,
                gpu_index=0,
            )
        assert excinfo.value.code == "output_validation_failed"
    finally:
        service.close()
