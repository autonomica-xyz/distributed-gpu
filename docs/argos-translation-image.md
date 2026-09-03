# Argos-derived translation worker image

This workload uses a reviewed Argos language package as a build-time model
asset, but it does not install or invoke the Argos package manager at runtime.
The trusted executor calls CTranslate2 and SentencePiece directly, receives an
already segmented batch, and explicitly binds CUDA execution to the GPU index
owned by the worker slot.

## Current proof

The English-to-Spanish fast route was prepared and tested locally on
2026-07-22. No rented capacity or provider resource was created.

| Item | Result |
| --- | --- |
| Reviewed model archive | Argos `translate-en_es-1_0`, SHA-256 `d698d0ef87ad70d5d184b7fa6965905bf4368f09a2bb9ffb165a79bac96af0c4` |
| Extracted model pack | 7 files, 96,243,365 bytes |
| Runtime | CTranslate2 4.8.1 and SentencePiece 0.2.2, without PyTorch, Stanza, or the Argos runtime |
| CPU smoke image | `distributed-gpu-translate@sha256:e72771bcb9bf57206c185c310894a02de067c530800dc47e75784e3cb17757b4`, 210,676,808 bytes (200.9 MiB), linux/amd64 |
| Offline inference | Passed with Docker networking disabled; three ordered segments translated on CPU int8 |
| Queue integration | Passed in one attempt through admission, lease, execution, authenticated upload/download, JSON Schema validation, byte count, and SHA-256 validation |
| CUDA definition | Docker build checks pass against a digest-pinned CUDA 12.8.1 cuDNN runtime for linux/amd64 |
| CUDA execution | Passed on owned hardware: NVIDIA RTX A5000 Laptop GPU, 16,384 MiB, compute capability 8.6, driver 595.71.05 |

The queue test produced artifact SHA-256
`a39816c1bd183f18169a970a0718da945dd23a293ccc080e0e4a7196e1f9aeb6`
with 488 bytes. Its translations were:

```json
[
  "Hola mundo.",
  "El invernadero es cálido aunque está nevando afuera.",
  "Por favor, traduzca estas frases sin usar el Internet."
]
```

The first integration attempt remained pending rather than bypassing
placement: the container reported 3,149 MiB free while the initial route asked
for 4,096 MiB. The baked model is only about 92 MiB, so the route now requires
1,024 MiB free. The repeated test then succeeded in one attempt.

## Real GPU verification

The exact GPU Dockerfile was built from a streamed, minimal context on the
owned `worker` host on 2026-07-22. No cloud capacity was rented and the host's
existing containers were not changed. The resulting local image was linux/amd64,
3,046,463,293 bytes, with image ID
`sha256:c29a4d952aedd08317d494604a26587c982b03da4be435751d835a6b0f682978`.

First, direct execution passed with Docker networking disabled. CTranslate2
reported `device=cuda` and `compute_type=int8_float16`, and returned the same
three expected translations in 0.480 seconds.

Second, the container registered through the normal agent using real
`nvidia-smi` discovery, not fake inventory. The controller placed the job on
`worker-translation-gpu-gpu-0`; it succeeded in one attempt. The authenticated
497-byte output downloaded successfully and matched SHA-256
`1ce2357d6a155fbeb2ca601b68184f8bdc53d9b366b730735f158d90d41c859b`.

A same-process cold/warm throughput check used 4,096 repeated segments,
233,472 source characters, and 49,152 source tokens per run:

| Run | Elapsed | Source tokens/second |
| --- | ---: | ---: |
| First | 1.588 s | 30,952 |
| Warm | 1.138 s | 43,193 |

The GPU reached 96% observed utilization. A pre-existing llama-server process
already occupied about 13,930 MiB of this 16 GiB GPU; total observed memory
rose from about 14,193 MiB idle to 14,564 MiB during the translation test.
This proves the route fits in the small remaining margin, but it is not a clean
standalone peak-VRAM measurement. Do not stop that unrelated process merely to
improve benchmark data; repeat on an isolated worker for production economics.

## Rebuild

Build the reviewed pack from its lock file, build the project wheel, and then
build one of the images:

```bash
.venv/bin/python scripts/build_argos_model_pack.py \
  --lock models/argos-en-es-1.0.lock.json \
  --output state/model-packs/en_es
.venv/bin/pip wheel --no-deps --wheel-dir dist .
sudo docker build -f deploy/translation/Dockerfile.cpu \
  -t distributed-gpu-translate:cpu-smoke .
sudo docker build -f deploy/translation/Dockerfile.gpu \
  -t distributed-gpu-translate:en-es-gpu .
```

The pack builder downloads only the URL in the trusted lock, enforces the
archive SHA-256, rejects traversal and link members, bounds file count and
expanded size, records every extracted file digest, and installs atomically.
The lock currently marks this model `evaluation_only`; model/corpus licensing
and provenance must be approved before production distribution.

The GPU image uses the linux/amd64 manifest
`nvidia/cuda@sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9`.
It is the CUDA 12.8.1 **cuDNN** runtime: the plain CUDA runtime is insufficient
for current CTranslate2 GPU wheels. Its base layers alone are 2.666 GiB
compressed. It was not pulled on this host because only about 3.1 GiB was free
and unrelated legacy-system Docker data must not be pruned.

## Next production gate

Before publishing the image, use one isolated worker and one short-lived agent
credential to prove production economics:

1. Build and push the image by immutable registry digest.
2. Start one worker with egress disabled after it reaches the controller.
3. Run cold and warm batches large enough to measure tokens/second, clean peak VRAM,
   GPU utilization, and dollars per miloperatorn source characters.
4. Download and hash the authenticated output, then destroy the worker and
   disable both temporary credentials.

Start with this single pair instead of an all-language image. Once its
economics and quality are acceptable, add independently reviewed language-pair
or regional model layers over the shared runtime.
