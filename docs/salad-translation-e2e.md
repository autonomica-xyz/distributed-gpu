# LibreTranslate on Salad: warm end-to-end canary

On 2026-08-28, the operator-deployed Salad group `translatelive-example`
registered one real `NVIDIA GeForce RTX 2070 SUPER` slot with 8,192 MiB VRAM
and compute capability 7.5. The selected high-priority class cost was
$0.060/hour ($43.80 per 730-hour month).

## Startup findings

The pinned LibreTranslate image uses Python 3.10.12. The first DGPU wheel
required Python >=3.11, causing repeated `Instance Exited:1` cycles after
Salad's startup/readiness probes had already passed. The wheel was made Python
3.10 compatible and smoke-tested inside the exact pinned image.

Agent registration still occurred before LibreTranslate finished downloading
68 selected language packs because the deployed group retained the old fixed
20-second delay. A forced canary at 12:40 UTC correctly reached the Salad slot
but exhausted three attempts with `model_server_unavailable: [Errno 111]
Connection refused`. The slot was manually drained while downloads continued.
The reviewed pool command now waits for the loopback `/languages` health check
and verifies the model process remains alive before starting the agent.

## Successful warm canary

After model download completed, the owned-host worker agent was temporarily stopped
so Salad was the only claimant. The Salad agent was enabled for one bounded job:

- Job: `job_0000000000000000_000000000000`
- Workload/route: `translate-segment/libretranslate-v1`
- Input: `Merhaba dünya` (`tr` -> `en`)
- Worker: `translatelive-example`
- Slot: `translatelive-example-gpu-0`
- Created: `2026-08-28T13:00:30.089475+00:00`
- Started: `2026-08-28T13:00:30.413244+00:00`
- Finished: `2026-08-28T13:00:31.804907+00:00`
- Attempts: one
- Output: `Hello world`
- Artifact size: 11 bytes
- SHA-256: `64ec88ca00b268e5ba1a35678a1b5316d212f4f366b2477232534a8aeca37f3c`

The artifact was downloaded through the authenticated public API and its bytes
and SHA-256 matched the job record. The temporary client credential was revoked
and Worker was restored. No Salad resource was created, replaced, or destroyed
for this canary; the existing group remains active.

The agent now self-drains after fatal `model_server_unavailable` or
`model_load_failed` errors so one unhealthy worker cannot consume every retry.
