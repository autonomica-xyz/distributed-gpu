# Popular translation shard verification

Two evaluation-only CUDA images were implemented and tested on the owned
`worker` RTX A5000 Laptop GPU on 2026-07-22. No provider resource was created.

## Shards

| Shard | Routes | Installed model bytes | Test image size | Test image ID |
| --- | ---: | ---: | ---: | --- |
| `core-western` | 10 | 1,265,644,579 | 3.798 GiB | `sha256:cb2c5af9ec0ee75ddac4941906179f4a807915b1d2303a624d8b76edca0c488d` |
| `global` | 12 | 1,529,332,608 | 4.020 GiB | `sha256:4b78ca9ca5b7446093eab59ca62a3695e4056459f587f2ff645ee9981511a7d0` |

The IDs above identify local test images, not published registry manifests.
Production images must be rebuilt from `Dockerfile.gpu-shard`, reviewed, and
published by immutable registry digest.

The lock catalog records the official URL, actual archive root, archive size,
SHA-256, language metadata, and model-pack ID for every pair. Safe extraction
rehashes every archive, forbids traversal and links, verifies metadata, and
records every installed file hash.

## CUDA route tests

Every route ran sequentially in its image with Docker networking disabled and
`device=cuda`, `compute_type=int8_float16`. The per-GPU cache limit was one, so
the sequence also exercised repeated translator eviction.

- Core: English both ways with German, Spanish, French, Italian, and
  Portuguese—10/10 passed.
- Global: English both ways with Arabic, Hindi, Japanese, Korean, Russian, and
  Chinese—12/12 passed.
- The smoke test rejects empty output, visible SentencePiece boundary markers,
  output expansion over 10x the input, wrong device/profile, wrong pair, or
  more than one cached runtime on GPU 0.

The audit exposed two package-family differences:

1. Most packs use `sentencepiece.model`, but Spanish-to-English 1.9 uses
   `bpe.model`. The thin runtime now supports both with pinned SentencePiece or
   pinned Sacremoses plus subword-nmt; it still does not install Argos.
2. Spanish-to-English 1.9 loops and hallucinates with beam 1 even when its BPE
   tokens are correct. Beam 4 returned bounded, plausible output, so its trusted
   route pins beam 4. Other tested routes retain beam 1.

## Placement and artifact tests

Two independent agents advertised only the model-pack IDs baked into their own
image. The controller correctly routed English-to-French to `core-western` and
English-to-Japanese to `global`. Both jobs succeeded and their authenticated
JSON artifacts passed schema, byte-count, and SHA-256 verification:

- English-to-French: `03248ec6327a85d487b82ddb43111739ae968ea359fd038f41cec2e2214addb7`
- English-to-Japanese: `d87a31c92649b88132ee520d3ba6a67c5d96d95a3a6d235d8c929879d6eb5dc3`

Those final agent tests used the explicit CPU override because an unrelated
llama server increased its allocation to 15,918 MiB and left only 46 MiB free.
An earlier attempt correctly returned `gpu_oom` and left both jobs pending for
retry. The unrelated service was not stopped. This demonstrates a remaining
scheduler limitation: physical VRAM inventory is not live free-VRAM admission
telemetry. The direct offline CUDA tests above completed before that pressure
increase and remain the GPU execution proof.

All temporary credentials were disabled after both attempts. The test images,
containers, archive cache, and extracted remote workspace were removed after
evidence collection.
