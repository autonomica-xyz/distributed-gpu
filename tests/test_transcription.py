from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparehour.agent import ExecutionError
from sparehour_workloads import transcription


PARAKEET_ENV = {
    "PARAKEET_MODEL_PACK": "parakeet-tdt-0.6b-v3",
    "PARAKEET_MODEL_ID": "nvidia/parakeet-tdt-0.6b-v3",
    "PARAKEET_MODEL_REVISION": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
    "PARAKEET_MODEL_DIR": "/models/parakeet-tdt-0.6b-v3",
    "PARAKEET_BATCH_SIZE": "8",
    "PARAKEET_MAX_AUDIO_SECONDS": "14400",
}

WHISPERX_ENV = {
    "WHISPERX_MODEL_PACK": "whisperx-large-v3",
    "WHISPERX_MODEL_ID": "Systran/faster-whisper-large-v3",
    "WHISPERX_MODEL_REVISION": "edaa852ec7e145841d8ffdb056a99866b5f0a478",
    "WHISPERX_MODEL_DIR": "/models/whisperx-large-v3",
    "WHISPERX_COMPUTE_TYPE": "float16",
    "WHISPERX_ALIGN_MODEL_PACK": "whisperx-align-en-wav2vec2-960h",
    "WHISPERX_ALIGN_MODEL_ID": "facebook/wav2vec2-base-960h",
    "WHISPERX_ALIGN_MODEL_REVISION": "22aad52d435eb6dbaf354bdad9b0da84ce7d6156",
    "WHISPERX_ALIGN_MODEL_DIR": "/models/whisperx-align-en-wav2vec2-960h",
    "WHISPERX_ALIGN_LANGUAGES": "en",
    "WHISPERX_DIARIZATION": "true",
    "DIARIZATION_MODEL_PACK": "speaker-diarization-community-1",
    "DIARIZATION_MODEL_ID": "pyannote/speaker-diarization-community-1",
    "DIARIZATION_MODEL_REVISION": "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
    "DIARIZATION_MODEL_DIR": "/models/speaker-diarization-community-1",
}


@pytest.fixture(autouse=True)
def clear_caches(monkeypatch):
    # Unit fixtures replace model loading, so their optional runtime imports
    # must stay lightweight too.
    for name in (
        "nemo.collections.asr", "whisperx", "whisperx.asr",
        "whisperx.alignment", "whisperx.diarize",
    ):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace())
    yield
    for cache in (
        transcription._PARAKEET_MODELS,
        transcription._WHISPERX_MODELS,
        transcription._ALIGN_MODELS,
        transcription._DIARIZE_PIPELINES,
    ):
        cache.clear()


def _job(tmp_path: Path, *, payload=None, environment=None, audio_bytes=b"audio"):
    workdir = tmp_path / "work"
    inputs_dir = workdir / "inputs"
    inputs_dir.mkdir(parents=True)
    audio = inputs_dir / "audio"
    audio.write_bytes(audio_bytes)
    job = {
        "id": "job-transcribe-1",
        "payload": payload if payload is not None else {"audio_hours": 1.0},
        "executor": {"environment": environment if environment is not None else PARAKEET_ENV},
        "inputs": [
            {
                "name": "audio",
                "uri": f"file://{audio}",
                "path": str(audio),
                "sha256": "0" * 64,
                "size": len(audio_bytes),
                "media_type": "audio/wav",
            }
        ],
    }
    return job, workdir


def _code(excinfo) -> str:
    return excinfo.value.code


class TestNormalizePayload:
    def test_defaults(self):
        spec = transcription._normalize_payload({"audio_hours": 0.5})
        assert spec.language == "auto"
        assert spec.hotwords == ()
        assert spec.diarize is False
        assert spec.audio_hours == 0.5

    def test_unknown_fields_rejected(self):
        with pytest.raises(ExecutionError) as excinfo:
            transcription._normalize_payload(
                {"audio_hours": 1, "expected_language": "en"}
            )
        assert _code(excinfo) == "executor_contract_invalid"

    def test_language_codes(self):
        spec = transcription._normalize_payload(
            {"audio_hours": 1, "language": "eng"}
        )
        assert spec.language == "eng"
        with pytest.raises(ExecutionError) as excinfo:
            transcription._normalize_payload(
                {"audio_hours": 1, "language": "EN"}
            )
        assert _code(excinfo) == "executor_contract_invalid"

    def test_hotword_string_normalization(self):
        spec = transcription._normalize_payload(
            {"audio_hours": 1, "hotwords": " GPU , transformer ,"}
        )
        assert spec.hotwords == ("GPU", "transformer")

    def test_hotword_bounds(self):
        with pytest.raises(ExecutionError) as excinfo:
            transcription._normalize_payload(
                {"audio_hours": 1, "hotwords": [f"w{i}" for i in range(65)]}
            )
        assert _code(excinfo) == "executor_contract_invalid"
        with pytest.raises(ExecutionError) as excinfo:
            transcription._normalize_payload(
                {"audio_hours": 1, "hotwords": [42]}
            )
        assert _code(excinfo) == "executor_contract_invalid"

    def test_speaker_bounds_require_diarize(self):
        with pytest.raises(ExecutionError) as excinfo:
            transcription._normalize_payload(
                {"audio_hours": 1, "min_speakers": 2}
            )
        assert _code(excinfo) == "executor_contract_invalid"

    def test_speaker_bound_consistency(self):
        with pytest.raises(ExecutionError):
            transcription._normalize_payload(
                {"audio_hours": 1, "diarize": True, "min_speakers": 3, "max_speakers": 2}
            )
        with pytest.raises(ExecutionError):
            transcription._normalize_payload(
                {"audio_hours": 1, "diarize": True, "num_speakers": 5, "max_speakers": 4}
            )
        spec = transcription._normalize_payload(
            {"audio_hours": 1, "diarize": True, "num_speakers": 2, "min_speakers": 1, "max_speakers": 4}
        )
        assert spec.num_speakers == 2

    def test_audio_hours_bounds(self):
        for value in (0, -1, 25, "one", True, None):
            with pytest.raises(ExecutionError) as excinfo:
                transcription._normalize_payload({"audio_hours": value})
            assert _code(excinfo) == "executor_contract_invalid"


class TestMaterializedInput:
    def test_happy_path(self, tmp_path):
        job, workdir = _job(tmp_path)
        resolved = transcription._materialized_input(job, workdir)
        assert resolved == (workdir / "inputs" / "audio").resolve()

    def test_missing_input(self, tmp_path):
        job, workdir = _job(tmp_path)
        job["inputs"] = []
        with pytest.raises(ExecutionError) as excinfo:
            transcription._materialized_input(job, workdir)
        assert _code(excinfo) == "input_validation_failed"

    def test_path_outside_workdir(self, tmp_path):
        job, workdir = _job(tmp_path)
        outside = tmp_path / "elsewhere.wav"
        outside.write_bytes(b"x")
        job["inputs"][0]["path"] = str(outside)
        with pytest.raises(ExecutionError) as excinfo:
            transcription._materialized_input(job, workdir)
        assert _code(excinfo) == "input_validation_failed"

    def test_traversal_rejected(self, tmp_path):
        job, workdir = _job(tmp_path)
        job["inputs"][0]["path"] = str(workdir / "inputs" / ".." / "secret")
        with pytest.raises(ExecutionError) as excinfo:
            transcription._materialized_input(job, workdir)
        assert _code(excinfo) == "input_validation_failed"

    def test_size_mismatch(self, tmp_path):
        job, workdir = _job(tmp_path)
        job["inputs"][0]["size"] = 999999
        with pytest.raises(ExecutionError) as excinfo:
            transcription._materialized_input(job, workdir)
        assert _code(excinfo) == "input_validation_failed"

    def test_missing_file(self, tmp_path):
        job, workdir = _job(tmp_path)
        job["inputs"][0]["path"] = str(workdir / "inputs" / "gone.wav")
        with pytest.raises(ExecutionError) as excinfo:
            transcription._materialized_input(job, workdir)
        assert _code(excinfo) == "input_validation_failed"


class TestDecode:
    def test_ffprobe_failure_is_decode_error(self, monkeypatch, tmp_path):
        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(command, 1, "", "moov atom not found")

        monkeypatch.setattr(transcription.subprocess, "run", fake_run)
        monkeypatch.setattr(transcription.shutil, "which", lambda name: name)
        with pytest.raises(ExecutionError) as excinfo:
            transcription._probe_duration_seconds(tmp_path / "in.wav")
        assert _code(excinfo) == "ffmpeg_decode_error"

    def test_ffprobe_missing_binary(self, monkeypatch, tmp_path):
        monkeypatch.setattr(transcription.shutil, "which", lambda name: None)
        with pytest.raises(ExecutionError) as excinfo:
            transcription._probe_duration_seconds(tmp_path / "in.wav")
        assert _code(excinfo) == "executor_unavailable"

    def test_ffprobe_duration_parsing(self, monkeypatch, tmp_path):
        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(
                command, 0, json.dumps({"format": {"duration": "11.000"}}), ""
            )

        monkeypatch.setattr(transcription.subprocess, "run", fake_run)
        monkeypatch.setattr(transcription.shutil, "which", lambda name: name)
        assert transcription._probe_duration_seconds(tmp_path / "in.wav") == 11.0

    def test_duration_bounds(self):
        with pytest.raises(ExecutionError) as excinfo:
            transcription._check_duration(100.0, max_seconds=50.0, audio_hours=1.0)
        assert _code(excinfo) == "input_validation_failed"
        with pytest.raises(ExecutionError) as excinfo:
            transcription._check_duration(7000.0, max_seconds=14400.0, audio_hours=1.0)
        assert _code(excinfo) == "input_validation_failed"
        transcription._check_duration(3590.0, max_seconds=14400.0, audio_hours=1.0)

    def test_ffmpeg_failure_is_decode_error(self, monkeypatch, tmp_path):
        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(command, 1, b"", b"invalid data")

        monkeypatch.setattr(transcription.subprocess, "run", fake_run)
        monkeypatch.setattr(transcription.shutil, "which", lambda name: name)
        with pytest.raises(ExecutionError) as excinfo:
            transcription._decode_audio_f32(tmp_path / "in.wav")
        assert _code(excinfo) == "ffmpeg_decode_error"


class TestParakeet:
    @pytest.mark.parametrize("option", [{"diarize": True}, {"hotwords": ["gpu"]}])
    def test_unsupported_parakeet_options_rejected(self, tmp_path, option):
        job, workdir = _job(tmp_path, payload={"audio_hours": 1, **option})
        with pytest.raises(ExecutionError) as excinfo:
            transcription.ParakeetExecutor().execute(
                job=job, workdir=workdir, gpu_index=0
            )
        assert _code(excinfo) == "unsupported_option"


    def test_missing_env_rejected(self, tmp_path):
        env = dict(PARAKEET_ENV)
        del env["PARAKEET_MODEL_DIR"]
        job, workdir = _job(tmp_path, environment=env)
        with pytest.raises(ExecutionError) as excinfo:
            transcription.ParakeetExecutor().execute(
                job=job, workdir=workdir, gpu_index=0
            )
        assert _code(excinfo) == "executor_contract_invalid"

    def test_non_immutable_revision_rejected(self, tmp_path):
        env = dict(PARAKEET_ENV, PARAKEET_MODEL_REVISION="main")
        job, workdir = _job(tmp_path, environment=env)
        with pytest.raises(ExecutionError) as excinfo:
            transcription.ParakeetExecutor().execute(
                job=job, workdir=workdir, gpu_index=0
            )
        assert _code(excinfo) == "executor_contract_invalid"

    def test_end_to_end_normalization(self, monkeypatch, tmp_path):
        model_dir = tmp_path / "pack"
        model_dir.mkdir()
        verified = []

        def fake_verify(directory, *, model_pack, model_id, revision):
            verified.append((model_pack, model_id, revision))
            return Path(directory)

        class FakeModel:
            def transcribe(self, paths, batch_size, timestamps):
                assert timestamps is True
                return [
                    SimpleNamespace(
                        text="Hello world",
                        language="en",
                        timestamp_dict={
                            "segment": [
                                {"start": 0.0, "end": 2.0, "segment": "Hello world"}
                            ],
                            "word": [
                                {"start": 0.0, "end": 0.8, "word": "Hello"},
                                {"start": 0.9, "end": 1.8, "word": "world"},
                            ],
                        },
                    )
                ]

        monkeypatch.setattr(transcription, "_verify_pack", fake_verify)
        monkeypatch.setattr(
            transcription, "_load_parakeet", lambda *, model_dir, gpu_index: FakeModel()
        )
        monkeypatch.setattr(
            transcription, "_probe_duration_seconds", lambda path: 11.0
        )
        monkeypatch.setattr(
            transcription, "_decode_audio_wav", lambda src, dst: dst.write_bytes(b"wav")
        )

        env = dict(PARAKEET_ENV, PARAKEET_MODEL_DIR=str(model_dir))
        job, workdir = _job(tmp_path, environment=env)
        outputs = transcription.ParakeetExecutor().execute(
            job=job, workdir=workdir, gpu_index=0
        )
        assert len(outputs) == 1
        assert outputs[0]["name"] == "transcript"
        assert outputs[0]["media_type"] == "application/json"
        document = json.loads(Path(outputs[0]["path"]).read_text())
        assert document["engine"] == "parakeet"
        assert document["language"] == "en"
        assert document["language_source"] == "detected"
        assert document["text"] == "Hello world"
        assert document["segment_count"] == 1
        assert document["word_count"] == 2
        assert document["speaker_count"] is None
        segment = document["segments"][0]
        assert segment["words"][0]["word"] == "Hello"
        assert segment["speaker"] is None
        assert document["models"] == [
            {
                "role": "asr",
                "model_pack": "parakeet-tdt-0.6b-v3",
                "model_id": "nvidia/parakeet-tdt-0.6b-v3",
                "revision": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
            }
        ]
        assert verified == [
            (
                "parakeet-tdt-0.6b-v3",
                "nvidia/parakeet-tdt-0.6b-v3",
                "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
            )
        ]

    def test_declared_language_mismatch_fails(self, monkeypatch, tmp_path):
        model_dir = tmp_path / "pack"
        model_dir.mkdir()
        monkeypatch.setattr(
            transcription, "_verify_pack", lambda directory, **kw: Path(directory)
        )
        monkeypatch.setattr(
            transcription,
            "_load_parakeet",
            lambda *, model_dir, gpu_index: SimpleNamespace(
                transcribe=lambda *a, **k: [
                    SimpleNamespace(
                        text="Bonjour",
                        language="fr",
                        timestamp_dict={
                            "word": [{"start": 0.0, "end": 1.0, "word": "Bonjour"}]
                        },
                    )
                ]
            ),
        )
        monkeypatch.setattr(transcription, "_probe_duration_seconds", lambda p: 11.0)
        monkeypatch.setattr(
            transcription, "_decode_audio_wav", lambda s, d: d.write_bytes(b"wav")
        )
        env = dict(PARAKEET_ENV, PARAKEET_MODEL_DIR=str(model_dir))
        job, workdir = _job(
            tmp_path, environment=env, payload={"audio_hours": 1, "language": "en"}
        )
        with pytest.raises(ExecutionError) as excinfo:
            transcription.ParakeetExecutor().execute(
                job=job, workdir=workdir, gpu_index=0
            )
        assert _code(excinfo) == "output_validation_failed"

    def test_pack_verification_failure_propagates(self, monkeypatch, tmp_path):
        model_dir = tmp_path / "pack"
        model_dir.mkdir()

        def reject(directory, **kwargs):
            raise ExecutionError("model_load_failed", "digest mismatch")

        monkeypatch.setattr(transcription, "_verify_pack", reject)
        monkeypatch.setattr(transcription, "_probe_duration_seconds", lambda p: 11.0)
        monkeypatch.setattr(
            transcription, "_decode_audio_wav", lambda s, d: d.write_bytes(b"wav")
        )
        env = dict(PARAKEET_ENV, PARAKEET_MODEL_DIR=str(model_dir))
        job, workdir = _job(tmp_path, environment=env)
        with pytest.raises(ExecutionError) as excinfo:
            transcription.ParakeetExecutor().execute(
                job=job, workdir=workdir, gpu_index=0
            )
        assert _code(excinfo) == "model_load_failed"

    def test_cache_reuse_and_release(self, monkeypatch, tmp_path):
        model_dir = tmp_path / "pack"
        model_dir.mkdir()
        loads = []
        monkeypatch.setattr(
            transcription, "_verify_pack", lambda directory, **kw: Path(directory)
        )
        monkeypatch.setattr(
            transcription,
            "_load_parakeet",
            lambda *, model_dir, gpu_index: loads.append(gpu_index) or object(),
        )
        first = transcription._parakeet_for_gpu(
            0,
            model_dir=model_dir,
            model_pack="p",
            model_id="m",
            revision="r" * 40,
        )
        second = transcription._parakeet_for_gpu(
            0,
            model_dir=model_dir,
            model_pack="p",
            model_id="m",
            revision="r" * 40,
        )
        assert first is second and loads == [0]
        transcription.ParakeetExecutor().release_gpu(0)
        assert not transcription._PARAKEET_MODELS
        third = transcription._parakeet_for_gpu(
            0,
            model_dir=model_dir,
            model_pack="p",
            model_id="m",
            revision="r" * 40,
        )
        assert third is not first and loads == [0, 0]


@dataclasses.dataclass(frozen=True)
class _Options:
    hotwords: str | None = None


class _FakeWhisperxModel:
    def __init__(self, result, calls):
        self.options = _Options()
        self.result = result
        self.calls = calls

    def transcribe(self, audio, batch_size=None, language=None):
        self.calls.append({"language": language, "hotwords": self.options.hotwords})
        return self.result


def _whisperx_setup(monkeypatch, tmp_path, *, result, diarize_df=None, env_overrides=None):
    for name in ("asr", "align", "diar"):
        (tmp_path / name).mkdir(exist_ok=True)
    calls = []
    model = _FakeWhisperxModel(result, calls)
    aligned_segments = []

    def fake_align(segments, align_model, metadata, audio, device):
        for segment in segments:
            aligned = dict(segment)
            aligned["words"] = [
                {"word": word, "start": segment["start"], "end": segment["end"]}
                for word in str(segment["text"]).split()
            ]
            aligned_segments.append(aligned)
        return {"segments": aligned_segments, "language": result["language"]}

    class FakeDiarization:
        def __call__(self, audio, num_speakers=None, min_speakers=None, max_speakers=None):
            calls[-1]["num_speakers"] = num_speakers
            return diarize_df

    monkeypatch.setattr(
        transcription, "_verify_pack", lambda directory, **kw: Path(directory)
    )
    monkeypatch.setattr(
        transcription,
        "_load_whisperx",
        lambda *, model_dir, gpu_index, compute_type: model,
    )
    monkeypatch.setattr(
        transcription,
        "_load_align_model",
        lambda *, model_dir, gpu_index, language: (object(), {}),
    )
    monkeypatch.setattr(
        transcription, "_load_diarization", lambda *, model_dir, gpu_index: FakeDiarization()
    )
    monkeypatch.setattr(transcription, "_alignment_fn", lambda: fake_align)
    monkeypatch.setattr(transcription, "_probe_duration_seconds", lambda p: 11.0)
    monkeypatch.setattr(
        transcription, "_decode_audio_f32", lambda p: [0.0] * 16000
    )

    env = dict(WHISPERX_ENV)
    env.update(
        {
            "WHISPERX_MODEL_DIR": str(tmp_path / "asr"),
            "WHISPERX_ALIGN_MODEL_DIR": str(tmp_path / "align"),
            "DIARIZATION_MODEL_DIR": str(tmp_path / "diar"),
        }
    )
    if env_overrides:
        env.update(env_overrides)
    return model, calls, env


class TestWhisperX:
    def test_transcribe_align_and_document(self, monkeypatch, tmp_path):
        result = {
            "language": "en",
            "segments": [{"text": "Hello world", "start": 0.0, "end": 2.0}],
        }
        _, calls, env = _whisperx_setup(monkeypatch, tmp_path, result=result)
        job, workdir = _job(
            tmp_path,
            environment=env,
            payload={"audio_hours": 1, "language": "en", "hotwords": ["gpu"]},
        )
        outputs = transcription.WhisperXExecutor().execute(
            job=job, workdir=workdir, gpu_index=0
        )
        document = json.loads(Path(outputs[0]["path"]).read_text())
        assert calls[0]["language"] == "en"
        assert calls[0]["hotwords"] == "gpu"
        assert document["engine"] == "whisperx"
        assert document["language_source"] == "declared"
        assert document["hotwords"] == ["gpu"]
        segment = document["segments"][0]
        assert segment["words"][0]["word"] == "Hello"
        assert segment["speaker"] is None
        roles = {model["role"] for model in document["models"]}
        assert roles == {"asr", "alignment"}

    def test_hotwords_restored_after_job(self, monkeypatch, tmp_path):
        result = {"language": "en", "segments": [{"text": "Hi", "start": 0.0, "end": 1.0}]}
        model, _, env = _whisperx_setup(monkeypatch, tmp_path, result=result)
        job, workdir = _job(
            tmp_path, environment=env, payload={"audio_hours": 1, "hotwords": ["gpu"]}
        )
        transcription.WhisperXExecutor().execute(job=job, workdir=workdir, gpu_index=0)
        assert model.options.hotwords is None

    def test_diarization_assigns_speakers(self, monkeypatch, tmp_path):
        result = {
            "language": "en",
            "segments": [{"text": "Hello world", "start": 0.0, "end": 2.0}],
        }

        class Frame:
            def __init__(self):
                self.rows = [{"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"}]

            def __len__(self):
                return len(self.rows)

            def iterrows(self):
                return iter(enumerate(self.rows))

        _, calls, env = _whisperx_setup(
            monkeypatch, tmp_path, result=result, diarize_df=Frame()
        )
        job, workdir = _job(
            tmp_path,
            environment=env,
            payload={
                "audio_hours": 1,
                "language": "en",
                "diarize": True,
                "num_speakers": 1,
            },
        )
        outputs = transcription.WhisperXExecutor().execute(
            job=job, workdir=workdir, gpu_index=0
        )
        document = json.loads(Path(outputs[0]["path"]).read_text())
        assert calls[-1]["num_speakers"] == 1
        assert document["diarized"] is True
        assert document["speaker_count"] == 1
        segment = document["segments"][0]
        assert segment["speaker"] == "SPEAKER_00"
        assert segment["words"][0]["speaker"] == "SPEAKER_00"
        roles = {model["role"] for model in document["models"]}
        assert roles == {"asr", "alignment", "diarization"}

    def test_diarize_rejected_without_pack(self, monkeypatch, tmp_path):
        result = {"language": "en", "segments": []}
        _, _, env = _whisperx_setup(
            monkeypatch,
            tmp_path,
            result=result,
            env_overrides={"WHISPERX_DIARIZATION": "false"},
        )
        job, workdir = _job(
            tmp_path, environment=env, payload={"audio_hours": 1, "diarize": True}
        )
        with pytest.raises(ExecutionError) as excinfo:
            transcription.WhisperXExecutor().execute(
                job=job, workdir=workdir, gpu_index=0
            )
        assert _code(excinfo) == "unsupported_option"

    def test_unsupported_declared_language(self, monkeypatch, tmp_path):
        result = {"language": "en", "segments": []}
        _, _, env = _whisperx_setup(monkeypatch, tmp_path, result=result)
        job, workdir = _job(
            tmp_path, environment=env, payload={"audio_hours": 1, "language": "de"}
        )
        with pytest.raises(ExecutionError) as excinfo:
            transcription.WhisperXExecutor().execute(
                job=job, workdir=workdir, gpu_index=0
            )
        assert _code(excinfo) == "unsupported_language"

    def test_unsupported_detected_language(self, monkeypatch, tmp_path):
        result = {
            "language": "fr",
            "segments": [{"text": "Bonjour", "start": 0.0, "end": 1.0}],
        }
        _, _, env = _whisperx_setup(monkeypatch, tmp_path, result=result)
        job, workdir = _job(tmp_path, environment=env)
        with pytest.raises(ExecutionError) as excinfo:
            transcription.WhisperXExecutor().execute(
                job=job, workdir=workdir, gpu_index=0
            )
        assert _code(excinfo) == "unsupported_language"

    def test_segment_bounds_validated(self, monkeypatch, tmp_path):
        result = {
            "language": "en",
            "segments": [{"text": "runs past the end", "start": 0.0, "end": 99.0}],
        }
        _, _, env = _whisperx_setup(monkeypatch, tmp_path, result=result)
        job, workdir = _job(tmp_path, environment=env)
        with pytest.raises(ExecutionError) as excinfo:
            transcription.WhisperXExecutor().execute(
                job=job, workdir=workdir, gpu_index=0
            )
        assert _code(excinfo) == "output_validation_failed"

    def test_diarization_empty_output_fails(self, monkeypatch, tmp_path):
        result = {
            "language": "en",
            "segments": [{"text": "Hello", "start": 0.0, "end": 1.0}],
        }

        class EmptyFrame:
            def __len__(self):
                return 0

            def iterrows(self):
                return iter(())

        _, _, env = _whisperx_setup(
            monkeypatch, tmp_path, result=result, diarize_df=EmptyFrame()
        )
        job, workdir = _job(
            tmp_path,
            environment=env,
            payload={"audio_hours": 1, "language": "en", "diarize": True},
        )
        with pytest.raises(ExecutionError) as excinfo:
            transcription.WhisperXExecutor().execute(
                job=job, workdir=workdir, gpu_index=0
            )
        assert _code(excinfo) == "output_validation_failed"


class TestHelpers:
    def test_assign_speakers_overlap_and_nearest(self):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "a", "speaker": None, "words": []},
            {"start": 5.0, "end": 6.0, "text": "b", "speaker": None, "words": []},
        ]
        intervals = [(0.0, 1.0, "S1"), (4.5, 6.5, "S2")]
        count = transcription._assign_speakers(segments, intervals)
        assert count == 2
        assert segments[0]["speaker"] == "S1"
        assert segments[1]["speaker"] == "S2"

    def test_segment_validation_word_outside(self):
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "hello",
                "speaker": None,
                "words": [{"word": "hi", "start": 0.0, "end": 5.0, "speaker": None}],
            }
        ]
        with pytest.raises(ExecutionError) as excinfo:
            transcription._validate_segments(segments, 10.0)
        assert _code(excinfo) == "output_validation_failed"

    def test_error_classification(self):
        assert transcription._classify("CUDA out of memory") == "gpu_oom"
        assert transcription._classify("cuDNN error") == "cuda_context_failure"
        assert transcription._classify("connection refused") == "model_load_failed"
        assert transcription._classify("gated repo access denied") == "model_load_failed"
        assert transcription._classify("something else") == "executor_failed"

    def test_release_gpu_drops_only_matching_index(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            transcription, "_verify_pack", lambda directory, **kw: Path(directory)
        )
        monkeypatch.setattr(
            transcription,
            "_load_parakeet",
            lambda *, model_dir, gpu_index: object(),
        )
        model_dir = tmp_path / "pack"
        model_dir.mkdir()
        transcription._parakeet_for_gpu(
            0, model_dir=model_dir, model_pack="p", model_id="m", revision="r" * 40
        )
        transcription._parakeet_for_gpu(
            1, model_dir=model_dir, model_pack="p", model_id="m", revision="r" * 40
        )
        transcription.ParakeetExecutor().release_gpu(0)
        remaining = {key[0] for key in transcription._PARAKEET_MODELS}
        assert remaining == {1}

    def test_failed_allocator_release_propagates(self, monkeypatch):
        import sys

        def fail_release():
            raise RuntimeError("CUDA context failed during release")

        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(cuda=SimpleNamespace(
                is_available=lambda: True, empty_cache=fail_release,
            )),
        )
        with pytest.raises(RuntimeError):
            transcription.WhisperXExecutor().release_gpu(0)


def test_retained_optional_import_frame_cannot_pin_released_model(monkeypatch, tmp_path):
    package = tmp_path / "whisperx"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from importlib import import_module\n"
        "def load_model(*args, **kwargs):\n"
        "    return import_module('whisperx.asr').Model()\n"
    )
    (package / "asr.py").write_text(
        "import sys\n"
        "# Optional dependency loaders can retain their import call stack.\n"
        "retained_frame = sys._getframe()\n"
        "class Model: pass\n"
    )
    for name in ("alignment", "diarize"):
        (package / f"{name}.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("whisperx", "whisperx.asr", "whisperx.alignment", "whisperx.diarize"):
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(transcription, "_verify_pack", lambda directory, **kw: directory)

    executor = transcription.WhisperXExecutor()
    model = transcription._whisperx_for_gpu(
        0, model_dir=tmp_path, model_pack="p", model_id="m",
        revision="a" * 40, compute_type="float16",
    )
    import weakref

    reference = weakref.ref(model)
    del model
    executor.release_gpu(0)
    assert reference() is None
