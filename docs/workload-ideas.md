# Workload opportunities

## Recommended sequence

### 1. Image generation

This is the best first new production pack. It proves non-JSON binary outputs,
model-pack inventory, profile constraints, image validation, warm model
servers, and GPU-class routing while remaining inexpensive enough to test.

Start with a fixed Diffusers adapter for one exact model and revision. Jobs
choose only bounded fields such as prompt, dimensions, steps, and seed—not
Python code, workflow JSON, model filenames, or LoRAs. Cache one pipeline per
GPU so independent jobs reuse the warm model.

Add a pinned ComfyUI adapter only when the product needs allowlisted graph
workflows, ControlNet, LoRA combinations, or multi-stage image pipelines. Its
queue and history APIs are useful then, but they are unnecessary complexity
for a single fixed SDXL route.

Suggested first unit: `image`; evolve to `megapixel_step` if render shapes vary
widely.

### 2. Offline/batch LLM inference

The small model mentioned in discussion is most likely `Qwen3.5-9B`; the
current official catalog also has a larger `Qwen3.6-35B-A3B`, but not a
Qwen3.6-9B checkpoint. A quantized Qwen3.5-9B worker is a good 16–24 GB GPU
profile.

Start with asynchronous batches for extraction, classification, summarization,
synthetic data, or evaluation. Keep vLLM warm inside each worker and send jobs
to its local API. This fits durable queue semantics. Interactive chat serving
is a later service scheduler because it needs continuous batching, token-level
streaming, latency targets, and load balancing rather than one durable batch
job per request.

Suggested unit: input plus reserved/generated tokens.

### 3. Image upscale and restoration

This is an excellent low-cost validation workload:

- clear input/output contracts;
- deterministic dimension checks;
- cost proportional to megapixels;
- broad consumer and media-pipeline utility;
- works on smaller GPUs.

It is useful for testing artifact brokering and inventory scheduling before
expensive generative workloads.

### 4. Embeddings and reranking

Batch document embedding, cross-encoder reranking, and semantic deduplication
are predictable, measurable, and easy to retry. They exercise many-small-item
batching and throughput-based autoscaling.

Suggested unit: token or document.

### 5. OCR and document vision

PDF page rendering, OCR, layout extraction, table recognition, and vision-LLM
structured extraction combine large inputs with JSON and preview outputs. This
is commercially useful and tests multi-artifact jobs without video-sized cost.

Suggested unit: page or megapixel.

### 6. Dataset captioning and moderation

Batch image/video captioning, tagging, aesthetic scoring, deduplication, and
safety classifiers are naturally asynchronous. They can run during cheap
provider windows and create training/evaluation datasets.

### 7. TTS, voice conversion, and audio enhancement

Text-to-speech, denoising, source separation, loudness normalization, and
speaker embedding reuse the legacy system's audio experience while producing different
artifact shapes. Voice cloning requires explicit consent and abuse controls.

Suggested unit: output audio minute.

### 8. Evaluation and synthetic-data jobs

Run prompt suites, model comparisons, judge models, regression benchmarks, and
data generation as durable batches. These benefit from idempotency, immutable
model packs, reproducible seeds, and explicit spend plans.

### 9. Single-GPU LoRA fine-tuning

Fine-tuning adds checkpoints, periodic progress, long leases, resumability, and
larger disk requirements. It still fits one GPU per job for small models and is
a good step after artifacts and cancellation are mature.

Suggested unit: training token or optimizer step.

### 10. Video generation

Video is the strongest abstraction test but not the first operational test. It
forces:

- 24 GB or larger GPU routes;
- large validated outputs;
- long-running cancellation and progress;
- duration/resolution admission gates;
- expensive retry policy;
- startup and drain-cost accuracy.

Suggested unit: output megapixel-second. Build it after image generation and
the S3-compatible artifact broker are reliable.

### 11. Music generation

Text-to-music, continuation, remix, and stem generation resemble video in cost
and artifact policy. Validate duration, codec, loudness/non-silence, and
manifest provenance. Treat remix/repair as explicit spend.

### 12. 3D reconstruction and Gaussian splats

Photogrammetry, single-image reconstruction, NeRF/Gaussian-splat training, mesh
generation, and texture baking produce several large related artifacts and are
good later-stage workload packs.

## Selection rubric

| Workload | Queue fit | First-pack difficulty | Useful cost unit | Key validation |
|---|---:|---:|---|---|
| Image generation | excellent | medium | image / megapixel-step | decode, dimensions, count |
| Batch Qwen inference | excellent | medium | token | JSON/schema, item count |
| Upscale/restoration | excellent | low | megapixel | decode, output dimensions |
| Embeddings/rerank | excellent | low | token/document | vector shape/count |
| OCR/document vision | excellent | medium | page | JSON schema, page coverage |
| TTS/audio enhancement | excellent | medium | audio minute | ffprobe, duration, silence |
| LoRA training | good | high | training token | checkpoint integrity |
| Video generation | excellent | high/costly | megapixel-second | ffprobe, duration, dimensions |
| Interactive chat | poor initially | high | generated token | latency/stream correctness |

## Recommended product demonstration

Run three routes through the same control plane without scheduler code changes:

1. legacy-compatible transcription (`audio_hour`).
2. Fixed Diffusers image generation (`image`).
3. Qwen3.5-9B asynchronous prompt batches (`reserved_output_token`).

That combination proves the registry is real: different inputs, output media,
model lifecycles, resource classes, validators, and cost units all share one
queue, artifact boundary, worker identity model, and provider planner.
