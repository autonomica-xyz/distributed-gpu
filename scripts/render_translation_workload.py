#!/usr/bin/env python3
"""Render all locked popular translation routes into the workload registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    models = [model for shard in catalog["shards"].values() for model in shard]
    pairs = [(model["from_code"], model["to_code"]) for model in models]
    if len(pairs) != len(set(pairs)):
        raise ValueError("locked translation catalog contains duplicate pairs")
    language_codes = sorted({code for pair in pairs for code in pair})
    model_packs = sorted(model["model_pack"] for model in models)
    output_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "job_id", "source_language", "target_language", "model_pack", "device",
            "compute_type", "beam_size", "segment_count", "source_character_count",
            "source_token_count", "target_token_count", "elapsed_seconds", "translations",
        ],
        "properties": {
            "job_id": {"type": "string", "minLength": 1},
            "source_language": {"enum": language_codes},
            "target_language": {"enum": language_codes},
            "model_pack": {"enum": model_packs},
            "device": {"enum": ["cpu", "cuda"]},
            "compute_type": {"type": "string", "minLength": 1},
            "beam_size": {"type": "integer", "minimum": 1, "maximum": 8},
            "segment_count": {"type": "integer", "minimum": 1, "maximum": 4096},
            "source_character_count": {"type": "integer", "minimum": 1, "maximum": 8388608},
            "source_token_count": {"type": "integer", "minimum": 1},
            "target_token_count": {"type": "integer", "minimum": 1},
            "elapsed_seconds": {"type": "number", "minimum": 0},
            "translations": {
                "type": "array", "minItems": 1, "maxItems": 4096,
                "items": {"type": "string"},
            },
        },
    }
    routes = {}
    for model in sorted(models, key=lambda item: (item["from_code"], item["to_code"])):
        source, target = model["from_code"], model["to_code"]
        routes[f"argos-{source}-{target}-fast"] = {
            "compatibility": {
                "capability": "translate-batch",
                "model_packs": [model["model_pack"]],
            },
            "resources": {
                "gpu_count": 1, "vendor": "nvidia", "min_vram_mb": 4096,
                "min_compute_capability": "7.0", "min_cpu_cores": 2,
                "min_ram_mb": 4096, "min_disk_mb": 1024,
            },
            "executor": {
                "type": "plugin",
                "handler": "gpu_orchestrator_workloads.argos_translate:ArgosTranslateExecutor",
                "timeout_seconds": 600,
                "environment": {
                    "ARGOS_MODEL_DIR": f"/opt/models/{source}_{target}",
                    "ARGOS_MODEL_PACK": model["model_pack"],
                    "ARGOS_MODEL_ARCHIVE_SHA256": model["sha256"],
                    "ARGOS_SOURCE_LANGUAGE": source,
                    "ARGOS_TARGET_LANGUAGE": target,
                    "ARGOS_DEVICE": "cuda",
                    "ARGOS_GPU_COMPUTE_TYPE": "int8_float16",
                    "ARGOS_CPU_COMPUTE_TYPE": "int8",
                    "ARGOS_BEAM_SIZE": str(model.get("fast_beam_size", 1)),
                    "ARGOS_MAX_BATCH_SIZE": "1024",
                    "ARGOS_MAX_CACHED_MODELS_PER_GPU": "1",
                },
            },
            "cost": {
                "unit": "segment", "default_units_per_job": 1,
                "payload_items_field": "segments", "default_units_per_gpu_hour": 200000,
            },
            "scaling": {
                "min_slots": 0, "max_slots": 100,
                "target_completion_hours": 2, "startup_minutes": 8,
            },
        }
    workload = {
        "api_version": "distributed-gpu.example/v1alpha1",
        "name": "translate-batch",
        "version": "2",
        "description": "Ordered, allowlisted popular-language segment batches using reviewed Argos CTranslate2 model packs.",
        "default_route": "argos-en-es-fast",
        "payload_schema": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object", "additionalProperties": False, "required": ["segments"],
            "properties": {"segments": {
                "type": "array", "minItems": 1, "maxItems": 4096,
                "items": {"type": "string", "minLength": 1, "maxLength": 8192},
            }},
        },
        "outputs": [{
            "name": "translations", "media_types": ["application/json"],
            "max_bytes": 33554432, "validator": "json_schema", "schema": output_schema,
        }],
        "routes": routes,
        "retry": {
            "max_attempts": 3,
            "retryable_error_codes": [
                "worker_lost", "provider_interrupted", "artifact_transport_failed",
                "gpu_oom", "model_load_failed",
            ],
        },
    }
    args.output.write_text(
        "# Generated by scripts/render_translation_workload.py; edit the lock catalog, not this file.\n"
        + yaml.safe_dump(workload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
