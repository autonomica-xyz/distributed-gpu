# Standalone architecture

## Product boundary

The platform distributes independent asynchronous jobs across a heterogeneous
fleet. A machine with four GPUs registers four slots; it can run four unrelated
jobs concurrently. A provider container normally registers one slot.

The trusted control plane decides admission, route, retry policy, cost policy,
and capacity. Workers execute the selected route and cannot choose provider
spend or arbitrary commands from job payloads.

```text
client
  -> authenticated API
       -> workload registry + JSON Schema admission
       -> SQLite queue and event log
       -> content-addressed artifact broker

GPU host agent
  -> registers one inventory slot per GPU
  -> atomically claims a compatible job
  -> confirms, executes, heartbeats, uploads, completes

fleet planner
  -> sums route-specific workload units
  -> counts explicitly owned local/provider slots
  -> checks provider credit and offers
  -> ranks total cost to drain
  -> proposes local/Vast/Salad capacity
```

## Workload definition

A workload YAML is the stable extension point. It declares:

- payload JSON Schema;
- input and output artifact contracts;
- one or more routes;
- trusted executor type and configuration;
- required capability and model packs;
- GPU/CPU/RAM/disk/provider/label requirements;
- workload cost unit and estimated throughput;
- scaling window and bounds;
- centrally controlled retryable error codes.

Jobs persist a snapshot of the selected definition. Updating YAML therefore
does not silently reinterpret already-queued work.

## Scheduling

Scheduling is pull-based and slot-aware:

1. An agent registers hardware inventory and installed model packs.
2. An idle slot asks for work.
3. A short `BEGIN IMMEDIATE` transaction scans priority/FIFO pending jobs.
4. The control plane matches capability, model packs, vendor, VRAM, compute
   capability, GPU model patterns, CPU, RAM, disk, provider, and labels.
5. The winning job and slot are updated atomically.
6. The claimant receives a random lease secret; only its hash is stored.
7. The lease secret is required for confirm, heartbeat, upload, completion,
   failure, or release.

This avoids the claim/release churn that occurs when broad worker capabilities
receive jobs for unloaded models.

## State machine

```text
pending -> leased -> running -> succeeded
             |          |          
             |          +-> pending / dead-letter / cancelled
             +-> pending (release or pre-run lease expiry)
```

Attempts begin at `confirm`, not at claim. A pre-run release therefore does not
consume an execution attempt. Running lease expiry becomes `worker_lost` and is
retried only within the workload policy.

Every transition emits an append-only event, and each confirmed attempt has a
separate duration/error record.

## Stale-worker fencing

Worker ID is insufficient ownership proof after a job is requeued. Each claim
therefore receives a unique lease token. When a lease expires, its hash is
cleared. Even if the old process wakes up later with the same agent identity,
it cannot complete or upload for the new claim.

## Artifact boundary

The current broker accepts running-job outputs through the authenticated API:

- the agent must own the running job and present its lease secret;
- the output name and media type must be declared;
- streaming upload is bounded by `max_bytes`;
- content is hashed and stored by SHA-256;
- JSON, JSON Schema, image, and ffprobe validators are supported;
- only artifacts attached to a completed job can be downloaded by that
  project's client.

Input artifact upload, external S3-compatible backing, signed direct-transfer
URLs, retention leases, and abandoned-upload GC are the next artifact phase.

## Authentication

Credentials are role and subject scoped:

- `client`: project submission/read/cancel, optionally restricted further to
  explicit workload and route allowlists;
- `agent`: one agent identity and an optional capability scope;
- `controller`: fleet observation/reaping;
- `admin`: credential and global administration.

Secrets are random and stored using per-token scrypt salts. Provider machines
receive only an agent credential, control-plane address, route/model settings,
and job-scoped artifact access.

## Executors

The trusted route—not the job payload—chooses one executor:

- `command`: a pinned argv inside the agent environment;
- `container`: a digest-pinned OCI image with one selected GPU;
- `http`: a warm local model service, useful for batch LLM inference;
- `plugin`: a workload-pack adapter, used by the fixed SDXL Diffusers executor
  and suitable for transcription or an optional asynchronous ComfyUI adapter.

The generic command contract writes `outputs.json`; the agent hashes and
uploads the named files. Long-lived model servers avoid loading a 9B model or
diffusion checkpoint for every job.

## Local and provider capacity

Local towers and provider workers run the same agent protocol. Their only
difference is lifecycle ownership:

- local slots register and remain operator managed;
- Vast is an offer/instance driver;
- Salad is a container-group driver with one group per scoped worker identity.

Provider drivers expose search, launch, inventory, destroy, and credit. The
cross-provider destroy helper returns only when authoritative inventory no
longer contains the resource. Local credentials must not be disabled earlier.

The implemented fleet planner is read-only. It counts slots explicitly labeled
for one pool, sums queued/running route units, checks credit, searches enabled
providers, and ranks expected total drain cost. Applying autonomous create and
destroy actions is deliberately a later, audited layer.

## Scaling units

Raw queue length is not comparable across workloads. Examples:

- transcription: audio hour;
- image generation: image or megapixel-step;
- batch LLM: reserved/generated token;
- upscale: input megapixel;
- video: output megapixel-second.

Definitions provide an initial units/GPU-hour prior. Production measurements
will replace it with conservative route/model/GPU-specific history.

## Pool isolation

An agent has an explicit `pool` label for autoscaling ownership. A slot may
technically advertise several capabilities, but it must not be counted as
dedicated capacity by two independent controllers. Generated pool configuration
and overlap tests should enforce that invariant before autonomous scaling.

## Why SQLite first

The target initial fleet is small enough that a short serialized claim is not a
bottleneck. SQLite offers simple recovery, transactions, inspection, and
backup. PostgreSQL can replace it when multi-region control-plane availability
or thousands of claims per second are real requirements; it is not needed to
prove workload abstraction or provider safety.

## Deferred gang scheduling

The schema keeps `gpu_count`, currently constrained to one. Supporting a single
job across several GPUs would require atomic multi-slot reservations, placement
groups, rendezvous information, coordinated startup, and group failure. That is
valuable for training or tensor-parallel serving, but separate from the primary
independent-job fleet.
