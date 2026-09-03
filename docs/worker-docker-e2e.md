# Docker worker end-to-end verification

On 2026-08-28, the owned `worker` host registered through
`https://gpu-control.example.com` and completed real bounded translation jobs
before and after transitioning to the reusable generic Docker images. No
provider resource was created.

## Worker inventory

- Agent: `worker-translate-01`
- Slot: `worker-translate-01-gpu-0`
- GPU: NVIDIA RTX A5000 Laptop GPU
- VRAM: 16,384 MiB
- Compute capability: 8.6
- Capability: `libretranslate`
- Model pack: `libretranslate-1.9.6-bounded-sbd`
- Agent token: 7-day expiry, subject fixed to the agent, scope fixed to
  `routes=libretranslate`

The model API runs in `dgpu-worker-translate-model`, published only on
`127.0.0.1:5001`. The agent runs in `dgpu-worker-translate-agent`, reaches that
loopback service through Docker host networking, and sends queue/artifact
traffic to the stable public API. Final local image IDs were
`dgpu-libretranslate:1.9.6-bounded-sbd` → `sha256:128a235fcd0e…` and
`dgpu-agent:0.1.0` → `sha256:fecde58f14b3…`; these are local integration image
IDs, not published registry digests. The existing legacy LibreTranslate
containers were not modified; their 5.9 GiB model cache was cloned into a
DGPU-owned volume.

## Job and artifact evidence

- Job: `job_0000000000000000_000000000000`
- Project: `worker-e2e`
- Workload/route: `translate-segment/libretranslate-v1`
- Payload: `{"text":"Merhaba dünya","source":"tr","target":"en"}`
- Attempts: 1
- Worker/slot: `worker-translate-01` / `worker-translate-01-gpu-0`
- Created: `2026-08-28T11:17:38.332915Z`
- Finished: `2026-08-28T11:17:38.989012Z`
- Artifact text: `Hello world`
- Media type: `text/plain`
- Bytes: 11
- SHA-256: `64ec88ca00b268e5ba1a35678a1b5316d212f4f366b2477232534a8aeca37f3c`

The output was downloaded through the authenticated public artifact endpoint;
its text, byte count, and SHA-256 all matched the terminal job record. After
building and switching to the reusable images from `deploy/agent` and
`deploy/libretranslate`, a second job (`job_0000000000000000_000000000000`)
succeeded in one attempt on the same worker/slot from
`2026-08-28T11:26:48.301216Z` to `2026-08-28T11:26:50.109586Z`; its artifact
was again 11-byte `Hello world` with the same SHA-256. All temporary client
credentials were disabled. The worker credential remains active because the
user requested a running worker.

## Placement correction

Real `nvidia-smi` inventory names the card `NVIDIA RTX A5000 Laptop GPU`. The
route previously used start-anchored patterns such as `RTX A5000*`, which
would reject the real vendor-prefixed name. The allowlist now uses
`*RTX A5000*`-style patterns. The route now also admits reviewed 8 GB GTX
1070/1080/1080 Ti cards at cc 6.1; they can execute CUDA translation but lack
the cc>=7 Tensor Core fast path, so economics must be benchmarked separately.

See `docs/docker-workers.md` for the reusable copy/paste deployment and
teardown commands.
