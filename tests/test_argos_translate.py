from __future__ import annotations

import hashlib
import io
import json
import zipfile
from types import SimpleNamespace

import pytest

from gpu_orchestrator_workloads import argos_pack, argos_translate


def _model_pack(tmp_path, *, name="pack", model_pack="test-pack", source="en", target="es"):
    model = tmp_path / name
    (model / "model").mkdir(parents=True)
    (model / "model" / "model.bin").write_bytes(b"model")
    (model / "sentencepiece.model").write_bytes(b"tokenizer")
    files = []
    for relative in ("model/model.bin", "sentencepiece.model"):
        data = (model / relative).read_bytes()
        files.append(
            {"path": relative, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
    (model / "manifest.json").write_text(
        json.dumps(
            {
                "model_pack": model_pack,
                "archive_sha256": "a" * 64,
                "metadata": {"from_code": source, "to_code": target},
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return model


def test_executor_preserves_order_and_reuses_gpu_runtime(monkeypatch, tmp_path):
    model = _model_pack(tmp_path)
    loads = []
    calls = []

    class Tokenizer:
        def encode(self, text, out_type):
            assert out_type is str
            return text.split()

        def decode(self, tokens):
            return " ".join(tokens)

    class Translator:
        def translate_batch(self, tokenized, **kwargs):
            calls.append((tokenized, kwargs))
            return [SimpleNamespace(hypotheses=[["es:" + "_".join(tokens)]]) for tokens in tokenized]

    def load_runtime(**kwargs):
        loads.append(kwargs)
        return Translator(), Tokenizer()

    monkeypatch.setattr(argos_translate, "_load_runtime", load_runtime)
    argos_translate._RUNTIMES.clear()
    job = {
        "id": "job-translate-1",
        "payload": {"segments": ["Hello world", "A second sentence"]},
        "executor": {
            "environment": {
                "ARGOS_MODEL_DIR": str(model),
                "ARGOS_MODEL_PACK": "test-pack",
                "ARGOS_MODEL_ARCHIVE_SHA256": "a" * 64,
                "ARGOS_SOURCE_LANGUAGE": "en",
                "ARGOS_TARGET_LANGUAGE": "es",
                "ARGOS_DEVICE": "cuda",
                "ARGOS_GPU_COMPUTE_TYPE": "int8_float16",
                "ARGOS_CPU_COMPUTE_TYPE": "int8",
                "ARGOS_BEAM_SIZE": "1",
                "ARGOS_MAX_BATCH_SIZE": "512",
            }
        },
    }

    outputs = argos_translate.ArgosTranslateExecutor().execute(
        job=job, workdir=tmp_path, gpu_index=2
    )
    argos_translate.ArgosTranslateExecutor().execute(job=job, workdir=tmp_path, gpu_index=2)

    assert len(loads) == 1
    assert loads[0]["device"] == "cuda"
    assert loads[0]["gpu_index"] == 2
    assert calls[0][1]["beam_size"] == 1
    assert calls[0][1]["batch_type"] == "tokens"
    document = json.loads(outputs[0]["path"].read_text())
    assert document["translations"] == ["es:Hello_world", "es:A_second_sentence"]
    assert document["segment_count"] == 2
    assert document["source_token_count"] == 5


def test_executor_verifies_route_language_pair(monkeypatch, tmp_path):
    model = _model_pack(tmp_path, source="fr", target="en")
    job = {
        "id": "job-wrong-pair",
        "payload": {"segments": ["Bonjour"]},
        "executor": {
            "environment": {
                "ARGOS_MODEL_DIR": str(model),
                "ARGOS_MODEL_PACK": "test-pack",
                "ARGOS_MODEL_ARCHIVE_SHA256": "a" * 64,
                "ARGOS_SOURCE_LANGUAGE": "en",
                "ARGOS_TARGET_LANGUAGE": "fr",
                "ARGOS_DEVICE": "cuda",
                "ARGOS_GPU_COMPUTE_TYPE": "int8_float16",
                "ARGOS_CPU_COMPUTE_TYPE": "int8",
            }
        },
    }
    argos_translate._RUNTIMES.clear()
    with pytest.raises(Exception, match="model language pair is not en->fr"):
        argos_translate.ArgosTranslateExecutor().execute(
            job=job, workdir=tmp_path, gpu_index=0
        )


def test_runtime_cache_is_lru_bounded_per_gpu(monkeypatch, tmp_path):
    models = [
        _model_pack(
            tmp_path,
            name=f"pack-{code}",
            model_pack=f"test-{code}",
            target=code,
        )
        for code in ("es", "fr", "de")
    ]
    loads = []

    class Tokenizer:
        def encode(self, text, out_type):
            return [text]

        def decode(self, tokens):
            return tokens[0]

    class Translator:
        def translate_batch(self, tokenized, **kwargs):
            return [SimpleNamespace(hypotheses=[tokens]) for tokens in tokenized]

    def load_runtime(**kwargs):
        loads.append((kwargs["model_dir"], kwargs["gpu_index"]))
        return Translator(), Tokenizer()

    monkeypatch.setattr(argos_translate, "_load_runtime", load_runtime)
    argos_translate._RUNTIMES.clear()

    def execute(model, target, gpu_index):
        job = {
            "id": f"job-{target}-{gpu_index}",
            "payload": {"segments": ["text"]},
            "executor": {
                "environment": {
                    "ARGOS_MODEL_DIR": str(model),
                    "ARGOS_MODEL_PACK": f"test-{target}",
                    "ARGOS_MODEL_ARCHIVE_SHA256": "a" * 64,
                    "ARGOS_SOURCE_LANGUAGE": "en",
                    "ARGOS_TARGET_LANGUAGE": target,
                    "ARGOS_DEVICE": "cuda",
                    "ARGOS_GPU_COMPUTE_TYPE": "int8_float16",
                    "ARGOS_CPU_COMPUTE_TYPE": "int8",
                    "ARGOS_MAX_CACHED_MODELS_PER_GPU": "2",
                }
            },
        }
        out = tmp_path / f"out-{target}-{gpu_index}-{len(loads)}"
        out.mkdir()
        argos_translate.ArgosTranslateExecutor().execute(
            job=job, workdir=out, gpu_index=gpu_index
        )

    execute(models[0], "es", 0)
    execute(models[1], "fr", 0)
    execute(models[0], "es", 0)  # es is now most recently used.
    execute(models[2], "de", 0)  # evicts fr, not es.
    execute(models[1], "fr", 0)  # reloads the evicted runtime.
    execute(models[0], "es", 1)  # GPU 1 has an independent allowance.

    assert len(loads) == 5
    gpu_zero_dirs = [key[0] for key in argos_translate._RUNTIMES if key[2] == 0]
    assert str(models[1]) in gpu_zero_dirs
    assert str(models[2]) in gpu_zero_dirs
    assert len(gpu_zero_dirs) == 2
    assert sum(1 for key in argos_translate._RUNTIMES if key[2] == 1) == 1


def test_safe_extract_rejects_parent_path(tmp_path):
    archive = tmp_path / "bad.argosmodel"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("en_es/../../escape", "bad")
    with pytest.raises(ValueError, match="unsafe archive member"):
        argos_pack._safe_extract(archive, tmp_path / "output", "en_es")


def test_safe_extract_strips_one_locked_root(tmp_path):
    archive = tmp_path / "good.argosmodel"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("en_es/metadata.json", "{}")
        bundle.writestr("en_es/model/model.bin", b"model")
    output = tmp_path / "output"
    output.mkdir()
    files = argos_pack._safe_extract(archive, output, "en_es")
    assert [item["path"] for item in files] == ["metadata.json", "model/model.bin"]
    assert (output / "model" / "model.bin").read_bytes() == b"model"


def test_releasing_gpu_drops_only_its_translators(monkeypatch):
    class Translator:
        loaded = True

        def unload_model(self):
            self.loaded = False

    released = Translator()
    other_gpu = Translator()
    cpu = Translator()
    cache = argos_translate.OrderedDict(
        {
            ("first", "cuda", 0, "int8_float16"): (released, object()),
            ("second", "cuda", 1, "int8_float16"): (other_gpu, object()),
            ("third", "cpu", -1, "int8"): (cpu, object()),
        }
    )
    monkeypatch.setattr(argos_translate, "_RUNTIMES", cache)

    executor = argos_translate.ArgosTranslateExecutor()
    executor.release_gpu(0)
    executor.release_gpu(0)

    assert not released.loaded
    assert other_gpu.loaded and cpu.loaded
    assert set(cache) == {
        ("second", "cuda", 1, "int8_float16"),
        ("third", "cpu", -1, "int8"),
    }
