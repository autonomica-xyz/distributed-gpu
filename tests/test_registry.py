from __future__ import annotations

import pytest

from gpu_orchestrator.registry import AdmissionError


def test_loads_example_workloads(registry):
    assert registry.names() == [
        "demo-gpu",
        "image-generate",
        "llm-batch",
        "transcribe",
        "translate-batch",
        "translate-segment",
        "translate-text-batch",
    ]


def test_image_admission_snapshots_units_and_inventory(registry):
    admitted = registry.admit(
        "image-generate",
        {"prompt": "a red ceramic mug", "width": 1024, "height": 1024, "seed": 42},
    )

    assert admitted.route == "diffusers-sdxl"
    assert admitted.cost_unit == "image"
    assert admitted.estimated_units == 1
    assert admitted.requirements["min_vram_mb"] == 16384
    assert admitted.requirements["model_packs"] == ["sdxl-base-1.0-fp16-76d28af"]


@pytest.mark.parametrize(
    "payload, marker",
    [
        ({}, "prompt"),
        ({"prompt": "x"}, "seed"),
        ({"prompt": "x", "seed": 1, "width": 1000}, "width"),
        ({"prompt": "x", "seed": 1, "count": 5}, "count"),
        ({"prompt": "x", "seed": 1, "surprise": True}, "surprise"),
    ],
)
def test_rejects_invalid_image_payload_before_queueing(registry, payload, marker):
    with pytest.raises(AdmissionError, match=marker):
        registry.admit("image-generate", payload)


def test_unknown_route_is_rejected(registry):
    with pytest.raises(AdmissionError, match="unknown route"):
        registry.admit("demo-gpu", {"message": "hello"}, "does-not-exist")


def test_translation_admission_is_pair_and_model_pack_fixed(registry):
    admitted = registry.admit(
        "translate-batch", {"segments": ["Hello world", "How are you?"]}
    )

    assert admitted.route == "argos-en-es-fast"
    assert admitted.cost_unit == "segment"
    assert admitted.estimated_units == 2
    assert admitted.requirements["min_vram_mb"] == 4096
    assert admitted.requirements["model_packs"] == ["argos-en-es-1.0-d698d0ef"]
    assert "language" not in admitted.payload

    french = registry.admit(
        "translate-batch",
        {"segments": ["A trusted route chooses the model."]},
        "argos-en-fr-fast",
    )
    assert len(registry.get("translate-batch").routes) == 22
    assert french.requirements["model_packs"] == ["argos-en-fr-1.9-3a65ed83"]
    assert french.executor["environment"]["ARGOS_SOURCE_LANGUAGE"] == "en"
    assert french.executor["environment"]["ARGOS_TARGET_LANGUAGE"] == "fr"
    assert french.executor["environment"]["ARGOS_MAX_CACHED_MODELS_PER_GPU"] == "1"
    spanish_to_english = registry.admit(
        "translate-batch", {"segments": ["Hola mundo."]}, "argos-es-en-fast"
    )
    assert spanish_to_english.executor["environment"]["ARGOS_BEAM_SIZE"] == "4"


def test_llm_batch_cost_reserves_tokens_for_every_prompt(registry):
    admitted = registry.admit(
        "llm-batch",
        {
            "prompts": ["first", "second", "third"],
            "max_output_tokens": 256,
        },
    )

    assert admitted.cost_unit == "reserved_output_token"
    assert admitted.estimated_units == 768
