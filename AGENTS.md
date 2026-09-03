# Distributed GPU agent notes

This file applies to the entire repository. The project is a standalone
generalization of patterns studied in `/opt/legacy`; do not modify the legacy system's live
services, configuration, tokens, or provider resources while working here.

## Product intent

- The primary workload model is many independent jobs distributed among GPU
  workers. Each physical GPU registers as its own schedulable slot.
- A single job currently owns one GPU. Tensor parallelism, distributed
  training, and other gang-scheduled jobs are a later, separate extension.
- Local towers and rented machines run the same agent protocol. Provider
  drivers own only capacity lifecycle and inventory reconciliation.
- Workload routes are the trust boundary. A job may supply only schema-bounded
  data; it must never select arbitrary code, model files, container images,
  LoRAs, or workflow graphs.

## Proven implementation choices

- Prefer a small direct adapter for a fixed model route. The first production
  example uses Diffusers and SDXL Base 1.0 directly; ComfyUI is not required.
- Add ComfyUI only as a separately routed adapter when allowlisted graph
  workflows, ControlNet, LoRA combinations, or multi-stage pipelines justify
  its larger operational surface.
- Keep one loaded model pipeline per GPU and reuse it across sequential jobs.
  `src/gpu_orchestrator_workloads/sdxl.py` implements this with a module-level
  cache and lock.
- Pin the trusted route to an exact model ID and immutable revision. The tested
  SDXL route uses `stabilityai/stable-diffusion-xl-base-1.0`, revision
  `76d28af79639c28a79fa5c6c6468febd3490a37e`, fp16 safetensors.
- The wheel must contain both `gpu_orchestrator` and
  `gpu_orchestrator_workloads`; preserve both entries in `pyproject.toml`.
- The public bootstrap endpoint may serve `.whl` files only. Never place
  credentials in a wheel, image, public URL, or job payload.
- Outputs cross an authenticated, bounded, content-addressed artifact plane.
  Validate the decoded media and confirm its SHA-256, not merely job success.

## Salad SDXL run learnings

The end-to-end run on 2026-07-15 used one real Salad RTX 4090 with 24,564 MiB
VRAM and compute capability 8.9. Three independent 1024x1024 jobs succeeded in
one attempt.

- Provider startup has distinct `pending`, `deploying`, image `downloading`,
  `creating`, and `running` phases. Inspect per-instance state and image-pull
  progress before declaring a deployment stuck.
- The Salad availability endpoint returned HTTP 403 for the tested account.
  Treat it as an optional optimization, not a dependency; provider group and
  instance inventory remain authoritative.
- An RTX 4090 initially appeared unavailable, then allocated without changing
  class. Do not churn a progressing deployment merely because `running=0`.
- Pulling the public PyTorch base image took several minutes. The agent then
  registered correctly and reported its actual GPU inventory through
  `nvidia-smi`.
- The immutable fp16 SDXL revision requires about 7.1 GB of weights. The first
  leased job took 561.9 seconds because it downloaded and loaded the model;
  warm jobs took 7.0 and 7.2 seconds. A production worker image should pre-bake
  dependencies and preferably the reviewed model pack to reduce paid startup.
- Installing the wheel and pinned Diffusers dependencies at container startup
  is suitable for an integration test, but a production image should contain
  them already and be referenced by digest.
- A Cloudflare quick tunnel worked only after forcing HTTP/2 because outbound
  QUIC was unavailable. Always verify both `/healthz` and the wheel download
  through the public URL before renting capacity.
- Tailscale Funnel accepted the node config (capability held for
  443/8443/10000) but the public `ts.net` DNS record never published
  (NXDOMAIN at public resolvers, checked via DoH). Do not assume Funnel works
  because the command succeeds; verify public resolution before relying on
  it. `tailscale funnel --https=443 off` also removes the serve config —
  re-add the tailnet serve afterwards to restore the dashboard.
- The worker did not accept the local SSH key. Queue and provider heartbeats
  were sufficient telemetry; do not weaken worker access simply to obtain
  extra logs.

## Argos Translate workload findings

Argos Translate is a promising offline batch workload, especially on cheap
NVIDIA cards, but use its packages as reviewed model assets rather than
exposing its general package manager directly to jobs.

- The latest published package inspected on 2026-07-16 was Argos Translate
  1.11.0. The repository's `master` identified itself as 1.11.1, so do not pin
  an unreleased branch accidentally. Pin the Python package, CTranslate2,
  CUDA runtime, worker image digest, and every language model independently.
- CTranslate2 CUDA can execute on reviewed 8 GB GTX 1070/1080/1080 Ti cards
  at compute capability 6.1, but they lack the >=7.0 Tensor Core fast path.
  Keep them visible as a separate economics tier and benchmark throughput;
  never infer cost per translated unit from hourly price alone.
- `ARGOS_DEVICE_TYPE=cuda` enables the documented GPU path, but Argos'
  `PackageTranslation` does not pass CTranslate2's `device_index`. It therefore
  does not satisfy this project's one-process/many-GPU slot isolation by
  itself. Use one process with one visible GPU or a thin adapter that constructs
  `ctranslate2.Translator(..., device_index=gpu_index)`.
- The high-level Argos path defaults to beam size 4 and `max_batch_size=32`
  with token batching. For inexpensive bulk throughput, create separate trusted
  route profiles: benchmark a beam-1 fast route with larger token batches and
  retain a beam-4 quality route only where its measured quality gain is worth
  the cost.
- Cache one translator per `(GPU, language pair, route profile)`. Submit many
  bounded segments per job; a rented GPU is inefficient for single-sentence
  jobs. Preserve segment order in a JSON output and report source/target token
  counts for economics.
- The pinned LibreTranslate CUDA integration image uses Python 3.10. The DGPU
  wheel supports Python >=3.10 and must be smoke-tested inside that exact image;
  an earlier >=3.11 metadata bound caused repeated Salad exit-code-1 retries
  even though the model service and Salad startup probes initially passed.
- The public Argos package index had 100 direct language pairs but no
  cryptographic hashes. Sample packages were roughly 65-285 MB compressed.
  Download only during a trusted build, reject unsafe ZIP members, inspect the
  metadata and license/provenance, compute a SHA-256, and bake the verified
  model pack. Never install a user-supplied `.argosmodel` at runtime.
- A complete size audit of the 2026-06-27 package index found 100 direct packs
  covering 50 language codes: 8.371 GiB downloaded and 9.587 GiB extracted.
  An English-centered hub still needs 98 packs and 9.450 GiB extracted, so it
  barely reduces an all-language Argos bundle.
- The official CTranslate2 CUDA 12.8 image was 1.688 GiB in compressed OCI
  layers. Therefore an optimized all-package image has a hard lower bound near
  10.1 GiB transferred before adding this agent and tokenizer/segmentation
  dependencies; budget roughly 10-11 GiB compressed and 14-16 GiB unpacked.
- Do not naively `pip install argostranslate` in that CUDA image. Its mandatory
  Stanza dependency resolves through PyTorch to a second CUDA stack; the
  observed dependency download added roughly 2.8 GB before extraction. A thin
  runtime should use CTranslate2, SentencePiece, and a bundled segmenter
  directly, while treating Argos packages as build-time model assets.
- Prefer a shared digest-pinned runtime layer plus small language-pair or
  regional model layers. A single 10+ GiB all-language image increases cold
  starts and paid interruption recovery while most workers will use only a few
  pairs.
- Offline behavior must be tested with egress disabled. Argos sentence-boundary
  detection can try to fetch Stanza or MiniSBD resources even after a language
  package is installed. Pre-bake every SBD asset or bypass that layer with a
  deterministic, bundled segmenter in the adapter.
- Argos code is permissively licensed, but language packages inherit model,
  corpus, and tokenizer/SBD provenance that varies by pair. Approval is
  pair-specific; the library license is not sufficient evidence for a model.
- CTranslate2 supports older NVIDIA cards, but its documented fast path uses
  Tensor Cores and compute capability >= 7.0. Start the price/performance
  benchmark with batch-priority RTX 2070/2060-class cards, then compare RTX
  3050/3060-class cards and a 4-vCPU int8 CPU baseline. Do not assume the
  cheapest hourly card is cheapest per miloperatorn translated characters.
- Measure cold model load, source and target tokens/second, GPU utilization,
  peak VRAM/RAM, interruption/retry rate, and dollars per miloperatorn source
  characters separately for each language pair, beam profile, batch size, and
  GPU class. Quality must be tested on a domain-representative bilingual set.
- Direct pairs are preferred. Argos can pivot through an intermediate language,
  but this loads two packs, roughly doubles inference work, and can compound
  quality loss; declare pivot routes and their two model-pack requirements
  explicitly.

## Argos translation image verification

The local image preparation and CPU integration run on 2026-07-22 established
the following additional facts. Full evidence is in
`docs/argos-translation-image.md`.

- The first fixed route is `translate-batch/argos-en-es-fast`. It uses the
  Argos 1.0 English-to-Spanish archive with SHA-256
  `d698d0ef87ad70d5d184b7fa6965905bf4368f09a2bb9ffb165a79bac96af0c4`.
  The safe builder installed 7 files totaling 96,243,365 bytes.
- The thin CPU worker image is 200.9 MiB and successfully loaded the real model
  with Docker networking disabled. It translated three ordered segments using
  CTranslate2 4.8.1, SentencePiece 0.2.2, and CPU int8 without attempting any
  hidden runtime download.
- The same image completed a real local controller lifecycle in one attempt:
  authenticated submission, placement, lease, plugin execution, bounded upload,
  JSON Schema validation, authenticated download, and artifact size/SHA-256
  verification all passed.
- Resource contracts must describe free capacity inside the running container,
  not host disk size or image size. The initial 4,096 MiB free-disk requirement
  kept a healthy container reporting 3,149 MiB free in `pending`; the reviewed
  92 MiB model pack justifies the corrected 1,024 MiB route floor.
- Current CTranslate2 GPU wheels need cuDNN in addition to CUDA. The GPU
  Dockerfile therefore pins the linux/amd64 CUDA 12.8.1 cuDNN runtime manifest
  `sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9`.
  Do not regress it to the smaller plain CUDA runtime.
- The exact CUDA image was built and tested on the owned `worker` RTX A5000
  Laptop GPU (16,384 MiB, compute capability 8.6). Direct inference worked with
  Docker networking disabled and reported `cuda` plus `int8_float16`.
- A controller/agent test used real `nvidia-smi` discovery, succeeded in one
  attempt on `worker-translation-gpu-gpu-0`, and passed authenticated artifact
  download, size, JSON Schema, and SHA-256 verification. No cloud resource was
  created.
- Same-process 4,096-segment runs processed 49,152 source tokens in 1.588
  seconds cold and 1.138 seconds warm (about 30,952 and 43,193 source
  tokens/second). This confirms that the module-level translator cache is used.
- The GPU already had a llama server using about 13,930 MiB. Translation still
  succeeded and total observed memory rose by roughly 371 MiB, but this is not
  a clean peak-VRAM benchmark. Never stop an unrelated workload to improve a
  test; repeat economics and memory measurement on an isolated worker.
- The verified local GPU image ID was
  `sha256:c29a4d952aedd08317d494604a26587c982b03da4be435751d835a6b0f682978`
  and its local size was 3,046,463,293 bytes. This is test evidence, not a
  registry distribution digest; publish a reviewed production build by
  immutable registry digest.
- The reviewed English-to-Spanish pack remains `evaluation_only` until its
  model/corpus license and provenance are approved for distribution.

## Translation image sharding

The official Argos index was rechecked on 2026-07-22 and still contained 100
direct packs over 50 language codes totaling 8.371 GiB downloaded. The concrete
shard proposal is recorded in `docs/translation-model-shards.md`.

- A core Western shard with English both ways for Spanish, French, German,
  Portuguese, and Italian is 10 packs totaling 1.040 GiB compressed.
- A global-demand shard with English both ways for Chinese, Japanese, Korean,
  Arabic, Hindi, and Russian is 12 packs totaling 1.261 GiB compressed.
- Combining those produces a viable 22-pack popular image with 2.301 GiB of
  model archives, but two shared-runtime shards should improve cold starts and
  route affinity.
- The remaining 78 packs still total 6.071 GiB. Do not create a monolithic
  `exotic-rest` image by default; with the 2.666 GiB CUDA/cuDNN base it recreates
  most of the all-language image penalty. Add regional long-tail shards only
  from observed demand, and consider persistent CPU/local workers for rare
  pairs.
- Before adding a second pair, generalize the executor's hard-coded `en -> es`
  manifest check to compare manifest values against trusted route environment
  values. Language codes and model locations must remain unavailable to job
  payloads.
- Multi-pair workers must use the implemented per-GPU LRU limit or route-affine
  processes so diverse traffic cannot accumulate translators until VRAM
  exhaustion. Production shard routes currently pin the limit to one.

## Popular translation shard implementation

Both proposed popular shards were implemented and tested on 2026-07-22. Full
evidence is in `docs/translation-shards-e2e.md`.

- `models/argos-popular.lock.json` pins 22 directional archives, including the
  actual top-level directory inside each archive. Do not derive the root from
  the pair: older archives use roots such as `en_es`, while newer archives use
  roots such as `translate-es_en-1_9`.
- Workload version 2 has 22 allowlisted routes. The payload still contains only
  bounded segments; source, target, directory, digest, pack, tokenizer profile,
  and beam profile all come from trusted route data.
- The executor now compares manifest source/target metadata with the trusted
  route instead of hard-coding `en -> es`. It rejects empty or duplicate file
  manifests and rechecks every installed file digest.
- The runtime cache is now an access-ordered per-GPU LRU. Trusted route data can
  set a limit from one through four; every production shard route currently
  pins one. Tests prove recency eviction and independent GPU limits.
- Argos packages do not use one tokenizer format. Twenty-one popular packs use
  SentencePiece; `es -> en` 1.9 uses legacy BPE. The thin image pins
  SentencePiece 0.2.2, Sacremoses 0.1.1, and subword-nmt 0.3.8 and does not add
  the Argos/Stanza/PyTorch runtime. SentencePiece decoding must remove visible
  `▁` boundary markers.
- `es -> en` 1.9 is unusable with greedy beam 1 despite correct BPE tokens; it
  produced long repetitive hallucinations. Its allowlisted fast route pins
  beam 4, which produced bounded plausible translations. Never assume one beam
  profile works for every package—quality-smoke each direction.
- All 10 core-western and 12 global routes loaded and translated sequentially
  on CUDA with network disabled and a one-model cache. The test images were
  3.798 GiB and 4.020 GiB locally; these are local test IDs, not publishable
  registry digests.
- Model-pack inventory correctly routed representative French and Japanese
  jobs to their respective shard agents, and authenticated artifacts passed
  schema, size, and digest checks.
- Physical VRAM is insufficient admission data on a shared GPU. When an
  unrelated llama server grew to 15,918 MiB and left 46 MiB free, both real
  agent attempts returned `gpu_oom` and remained retryable even though the slot
  advertises 16,384 MiB. Do not stop unrelated services for a test; add live
  free-VRAM telemetry or use isolated workers. The final placement/artifact
  proof used the explicit CPU override after all 22 CUDA routes had passed.

## Provider and credential safety

- Provider mutations are billable. Do not create resources without explicit
  user authorization and a bounded test plan.
- Never send the Salad API key to a worker. A worker receives only its public
  controller URL, public wheel URL, fixed identity, and short-lived agent token
  scoped to the required route.
- Use a unique container-group name and one replica per scoped worker identity.
- Before teardown, ensure successful artifacts have reached the controller.
  Then delete the provider resource and require authoritative inventory absence
  before disabling the agent credential. Client credentials may remain only
  long enough to download and verify retained artifacts.
- Revoke all temporary tokens, remove their plaintext files, stop the public
  tunnel and controller, and perform a final provider inventory check.
- The verified Salad group has been deleted, its inventory query is empty, and
  the test database contains no active credentials. Never infer current cloud
  resources from stale local agent rows; query the provider.

## Verification expectations

Run these after relevant changes:

```bash
.venv/bin/pytest
.venv/bin/dgpu validate
.venv/bin/python -m compileall -q src tests scripts
```

The current baseline is 97 passing tests. The Starlette `TestClient` warning
from FastAPI's dependency stack is known and does not indicate a test failure.

For a real image run, additionally prove all of the following:

1. Provider inventory reports the expected physical GPU.
2. Every job reaches `succeeded` without losing its lease.
3. Every output downloads through the authenticated API.
4. PNG dimensions, mode, embedded provenance, size, and SHA-256 match the job
   record.
5. Images are inspected visually rather than accepted from metadata alone.
6. Provider absence and credential retirement are verified after teardown.

## Important paths

- `workloads/image-generate.yaml`: bounded SDXL route and resource contract.
- `src/gpu_orchestrator_workloads/sdxl.py`: fixed Diffusers executor.
- `src/gpu_orchestrator/dashboard.py` + `dashboard.html`: read-only operator
  workbench served at `/dashboard`; all mutations go through the authenticated
  `/v1` API, never a back door.
- `docs/dashboard.md`: workbench enablement, role behavior, and the new
  read-only overview/recent endpoints.
- `src/gpu_orchestrator/deploy.py` + `store.py` deployments table:
  operator-initiated provider worker deploys with a durable ledger,
  reconciliation against provider inventory, and verified teardown ordering.
  Autonomous create/destroy remains disabled; every deploy is a human action.
- `docs/provider-workers.md`: public controller ingress for non-tailnet
  (Vast/Salad) workers, the pre-rental verification checklist, and the
  scoped worker bootstrap.
- `deploy/salad-sdxl-worker.json`: digest-pinned integration-test template.
- `scripts/salad_sdxl.py`: explicit create/status/destroy helper.
- `examples/sdxl-prompts.json`: deterministic SDXL verification prompts.
- `workloads/translate-batch.yaml`: fixed English-to-Spanish route contract.
- `src/gpu_orchestrator_workloads/argos_translate.py`: offline CTranslate2
  executor with explicit GPU-index selection and runtime caching.
- `src/gpu_orchestrator_workloads/argos_pack.py`: bounded, digest-verified model
  pack builder.
- `models/argos-en-es-1.0.lock.json`: reviewed evaluation model lock.
- `deploy/translation/`: digest-pinned CPU smoke and CUDA/cuDNN images.
- `docs/argos-translation-image.md`: image proof and remaining GPU test gate.
- `docs/translation-model-shards.md`: popular and long-tail model image design.
- `models/argos-popular.lock.json`: exact 22-pack evaluation lock catalog.
- `scripts/audit_argos_popular.py`: official-index archive audit and lock emitter.
- `scripts/build_argos_shard.py`: safe locked multi-pack extraction.
- `scripts/render_translation_workload.py`: generated trusted route registry.
- `scripts/smoke_argos_shard.py`: all-pair offline CUDA and LRU smoke test.
- `deploy/translation/Dockerfile.gpu-shard`: shared-runtime shard image.
- `docs/translation-shards-e2e.md`: two-shard GPU and placement evidence.
- `docs/internal/legacy-system-study.md`: what was generalized from the legacy system and what was coupled.

A `.git` directory now exists, but the initial files may still be untracked.
Inspect status before relying on diffs; never treat an empty diff as evidence
that the working tree contents are preserved or committed.
