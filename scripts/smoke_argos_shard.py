#!/usr/bin/env python3
"""Exercise every baked translation pair in a shard image on CUDA GPU 0."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from sparehour_workloads import argos_translate


SAMPLES = {
    "ar": "مرحبا، هذا اختبار للترجمة.",
    "de": "Hallo, dies ist ein Übersetzungstest.",
    "en": "Hello, this is a translation test.",
    "es": "Hola, esta es una prueba de traducción.",
    "fr": "Bonjour, ceci est un test de traduction.",
    "hi": "नमस्ते, यह अनुवाद का परीक्षण है।",
    "it": "Ciao, questo è un test di traduzione.",
    "ja": "こんにちは、これは翻訳のテストです。",
    "ko": "안녕하세요. 이것은 번역 테스트입니다.",
    "pt": "Olá, este é um teste de tradução.",
    "ru": "Здравствуйте, это проверка перевода.",
    "zh": "你好，这是一个翻译测试。",
}


def main() -> None:
    root = Path("/opt/models")
    results = []
    argos_translate._RUNTIMES.clear()
    with tempfile.TemporaryDirectory(prefix="sparehour-shard-smoke-") as temporary:
        work = Path(temporary)
        for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
            metadata = manifest["metadata"]
            source = metadata["from_code"]
            target = metadata["to_code"]
            output = work / f"{source}_{target}"
            output.mkdir()
            job = {
                "id": f"smoke-{source}-{target}",
                "payload": {"segments": [SAMPLES[source]]},
                "executor": {"environment": {
                    "ARGOS_MODEL_DIR": str(model_dir),
                    "ARGOS_MODEL_PACK": manifest["model_pack"],
                    "ARGOS_MODEL_ARCHIVE_SHA256": manifest["archive_sha256"],
                    "ARGOS_SOURCE_LANGUAGE": source,
                    "ARGOS_TARGET_LANGUAGE": target,
                    "ARGOS_DEVICE": "cuda",
                    "ARGOS_GPU_COMPUTE_TYPE": "int8_float16",
                    "ARGOS_CPU_COMPUTE_TYPE": "int8",
                    "ARGOS_BEAM_SIZE": "4" if (model_dir / "bpe.model").is_file() else "1",
                    "ARGOS_MAX_BATCH_SIZE": "1024",
                    "ARGOS_MAX_CACHED_MODELS_PER_GPU": "1",
                }},
            }
            argos_translate.ArgosTranslateExecutor().execute(
                job=job, workdir=output, gpu_index=0
            )
            document = json.loads((output / "translations.json").read_text(encoding="utf-8"))
            translation = document["translations"][0]
            assert document["device"] == "cuda"
            assert document["compute_type"] == "int8_float16"
            assert document["source_language"] == source
            assert document["target_language"] == target
            assert translation.strip()
            assert "▁" not in translation
            assert len(translation) < 10 * len(SAMPLES[source])
            cache_entries = [
                key for key in argos_translate._RUNTIMES
                if key[1] == "cuda" and key[2] == 0
            ]
            assert len(cache_entries) == 1
            results.append({
                "route": f"argos-{source}-{target}-fast",
                "elapsed_seconds": document["elapsed_seconds"],
                "translation": translation,
            })
    print(json.dumps({"routes_tested": len(results), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
