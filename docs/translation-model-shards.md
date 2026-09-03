# Translation model shard design

The model distribution unit should follow traffic and worker affinity rather
than putting all 100 Argos pairs into one image. Every language direction is a
separate reviewed model pack and a separate trusted route.

The official package index was re-audited on 2026-07-22. It still contained
100 direct packs and 50 language codes totaling 8.371 GiB downloaded. The
source is the [Argos package index](https://github.com/argosopentech/argospm-index/blob/main/index.json).

## Recommended initial shards

| Shard | Direct routes | Packs | Compressed model archives |
| --- | --- | ---: | ---: |
| `translate-core-western` | English both ways with Spanish, French, German, Portuguese, and Italian | 10 | 1.040 GiB |
| `translate-global` | English both ways with Chinese, Japanese, Korean, Arabic, Hindi, and Russian | 12 | 1.261 GiB |
| Combined popular set | Both shards above | 22 | 2.301 GiB |
| Remaining long tail | All other direct pairs | 78 | 6.071 GiB |

A single combined popular image is technically reasonable, but the two-shard
layout is preferable: it pulls less unused data, makes regional capacity easier
to scale, and increases the chance that sequential jobs reuse the already
loaded translator. Both images should share the exact same digest-pinned
CUDA/cuDNN runtime layer, so a host that needs both can reuse that layer.

Do not create one `exotic-rest` image by default. Its 6.071 GiB of model
archives plus the 2.666 GiB compressed CUDA/cuDNN base approaches the same cold
start and interruption-recovery penalty as the all-language image. Split the
long tail only after traffic is observed, for example into European,
Asian-Pacific, and MENA/African shards. Very low-volume pairs can remain on
CPU workers or on persistent local workers where the shared model store is
already warm.

## Scheduler and executor changes

The job must select an allowlisted route such as `translate-en-fr-fast`; it
must not supply arbitrary source/target codes or model paths. Each route pins:

- source and target language;
- model-pack ID, archive SHA-256, and model directory;
- beam and token-batch profile;
- eligible worker shard/capability.

The current English-to-Spanish executor verifies `en -> es` explicitly. Before
adding more packs, generalize that check to compare the trusted route values
with each pack manifest. Do not weaken it into accepting language codes from
the payload.

The current module cache is also unbounded across `(GPU, language pair,
profile)`. That is safe for the single-pair image but unsuitable for a 22-pair
worker. Add a per-GPU LRU limit—starting with one or two loaded translators—or
run route-affine worker processes. Eviction must release the translator and be
measured with `nvidia-smi`; otherwise a diverse job stream can steadily consume
VRAM.

Every new pair still needs its own safe-build lock, digest audit, offline test,
license/provenance approval, quality corpus, and cold/warm benchmark. Image
membership is not approval to execute an unreviewed model.
