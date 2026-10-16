# Workload definitions

`translate-segment.yaml` routes one bounded text segment to a pinned local
CUDA LibreTranslate service (capability `libretranslate`). Each worker host
runs its own loopback-only service on `127.0.0.1:5001` and advertises the
capability plus the `libretranslate-1.9.6-bounded-sbd` model pack; any number
of heterogeneous workers may serve the same route concurrently. Reviewed 8 GB
GTX 1070/1080-class Pascal cards are eligible at cc 6.1, but lack the RTX
Tensor Core fast path and must be benchmarked as a separate economics tier:

```sh
sparehour agent --endpoint https://gpu-control.example.com --agent-id host-a \
  --capability libretranslate --model-pack libretranslate-1.9.6-bounded-sbd
```

Jobs choose only a schema-bounded `text`, `source`, and `target` pair; the
endpoint and wire format are fixed by the route. For bulk use
`translate-text-batch.yaml` with capability `libretranslate-batch`: it accepts
an ordered same-language `segments` array and amortizes claims, HTTP setup,
artifacts, and state transitions. It remains model-server agnostic and contains
no producer-specific identifiers. `translate-batch.yaml` is the separate
true CTranslate2/Argos token-batching contract for audited model-pack workers.

`image-generate.yaml` (v3) routes to the bundled fixed SDXL Diffusers
plugin. A worker must install the pinned Diffusers runtime and provision
the `sdxl-base-1.0-fp16-76d28af` pack under `SDXL_MODEL_DIR`; the executor
verifies the pack's manifest against the route's pinned model id and
revision before loading, and refuses jobs when it is absent. It never
falls back to downloading weights itself.

`transcribe.yaml` (v2) carries two fixed routes. `parakeet` restores
`nvidia/parakeet-tdt-0.6b-v3` from its verified pack; hotwords and
diarization are rejected as `unsupported_option`, never silently ignored.
`whisperx` runs faster-whisper `large-v3` with forced English wav2vec2
alignment and optional real pyannote `speaker-diarization-community-1`
diarization; all three packs are required by the route. Jobs upload audio
through the private input plane (`POST /v1/projects/{project}/inputs`) and
declare `audio_hours <= 4`; the old `expected_language` field is gone.

`llm-batch.yaml` (v2) sends up to 64 prompts through a warm loopback vLLM
server the operator pre-provisions against the verified
`qwen3.5-9b-awq-156edc4` pack (8192-token context, 4096 max output
tokens). The executor renders prompts with the server's `/tokenize`, sends
one true batched `/v1/completions` request, and uses `/sleep`/`/wake_up`
(vLLM `--enable-sleep-mode` + `VLLM_SERVER_DEV_MODE=1`, loopback only) to
hand VRAM back when the slot switches handlers.

This repository does not bundle large model weights or grant their
licenses; see `models/` for the reviewed immutable locks.

An agent must advertise a capability and model pack only after its image has:

- installed the named plugin handler;
- pinned and verified the model files;
- started and warmed any local model server;
- passed a canary for the declared input/output contract;
- measured initial throughput for its GPU class.

The SDXL executor keeps one pipeline cached per GPU and accepts no checkpoint,
LoRA, Python handler, or workflow graph from a job. ComfyUI is not required for
this route. It can be added later as a separate adapter when allowlisted graph
workflows, ControlNet, or multi-stage pipelines justify its larger operational
surface.
