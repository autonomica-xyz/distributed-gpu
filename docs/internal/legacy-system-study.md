# The legacy system architecture and operations study

Study date: 2026-07-13

This document records what was learned from the live `/opt/legacy` checkout. The
checkout was treated as read-only because it is an active operational tree and
contains unrelated uncommitted work.

## What the legacy system actually is

The legacy system is not merely a transcription script. It is a compact, production-shaped
GPU batch platform assembled from four layers:

1. A generic Go queue kernel.
2. A transcription admission and execution pack.
3. An artifact-security boundary for untrusted rented workers.
4. A provider-aware fleet control and operations layer.

Its strongest work is in the operational invariants around that fourth layer.
The model code can be replaced; the spend, identity, recovery, and artifact
rules are the valuable part.

## Runtime topology

The inspected host runs the following control paths:

- `workqueue-daemon.service` is active. It owns the queue SQLite database,
  token database, HTTP lifecycle API, stale-heartbeat scheduler, and blob GC.
- `pipeline-storage-broker.service` is active. It holds object-store
  credentials and brokers job-scoped artifact access to rented workers.
- `pipeline-staging-watch.timer` periodically scans staging storage, enriches
  eligible episodes, chooses a backend, and submits work.
- `infrastructure-control-plane.service` is the active Parakeet provider
  controller. Its current production policy uses Vast and disables Salad for
  that pool.
- `infrastructure-control-plane-whisperx.service` exists as an isolated
  non-English pool but is inactive by policy.
- `jim-worker-governor.timer` evaluates suspicious workers and provider
  cleanup.
- `jim-workqueue-reconciler.timer` compares queue state with durable transcript
  state. Its production repair behavior is report-first rather than silently
  spending GPU money.
- `infrastructure-control-plane-auditor.timer` checks service, timer, policy,
  dashboard, provider-market, queue-progress, lease, and telemetry health and
  can apply narrowly defined remediation.
- Grafana/Alloy publish queue, fleet, provider, worker, and control health.

Systemd currently supplies a very large portion of the application wiring:
pool boundaries, worker-name regular expressions, storage locations, provider
limits, route selection, cost assumptions, status files, metrics files, and
auditor assertions. That made a single deployment inspectable, but it is one of
the main obstacles to adding many workloads safely.

## End-to-end transcription flow

```text
staging object appears
  -> staging-watch verifies audio + metadata
  -> metadata language admission/routing
  -> Go workqueue pending job
  -> route-specific provider control plane observes canonical backlog
  -> local/Vast/Salad worker capacity becomes available
  -> worker claims job
  -> worker downloads and verifies source artifact
  -> worker confirms running state
  -> GPU transcription with heartbeats and one-job lookahead
  -> worker uploads output through storage broker
  -> broker validates and promotes output
  -> broker completes workqueue job
  -> fleet metrics become measured throughput/cost history
  -> reconciler audits queue/output consistency
```

Admission happens before GPU work. English and default-English jobs route to
Parakeet. Supported non-English jobs route to WhisperX with a forced language.
Unsupported languages and metadata lookup failures are held rather than allowed
to spend GPU time.

## Workqueue kernel

The Go workqueue is the most reusable component.

### Job model

A job carries:

- a coarse `kind`;
- arbitrary JSON payload;
- named content-addressed inputs and outputs;
- priority;
- attempt limit;
- requester identity;
- requester-scoped idempotency key;
- worker ownership and timestamps.

### Lifecycle

```text
pending
  -> claimed-pending-blob
       -> pending                 release / missing replicated blob
       -> running                 worker confirms inputs are local
            -> done               outputs recorded
            -> pending            retryable failure or stale heartbeat
            -> dead-letter        attempts exhausted
            -> cancelled          client request observed by worker
```

Claim is one atomic SQLite `UPDATE ... RETURNING` ordered by priority and
creation time. The race test uses 50 concurrent claimers under Go's race
detector. A worker ID may hold at most three active jobs, limiting the blast
radius of one credential.

The two-step claim/confirm state is important for large artifacts: a worker can
reserve a job while obtaining input, but a job is not considered GPU-running
until the input is available.

### Authentication

The queue supports:

- submitter tokens scoped to job kinds;
- worker tokens tied to a worker ID and capability set;
- brokered-worker tokens for untrusted provider machines;
- admin tokens;
- per-token and optional per-IP rate limits;
- YAML compatibility or a reloadable SQLite token store.

New secrets are scrypt-derived. Authenticated requesters cannot spoof the
`requester` field, and workers cannot act as a different worker ID.

### Generic worker

The Go worker maps kinds to trusted YAML-defined command or local HTTP handlers.
Commands receive payload and input paths through templates; job payloads cannot
choose an arbitrary executable. Outputs are discovered, hashed, and ingested
into the blob store. This can already run transcription, transcoding, local LLM
calls, and other simple workloads.

### Queue limitations for a generalized product

- Claims match only `kind`; there is no GPU inventory, VRAM, model-pack, or
  bounded-label matching.
- Project and tenant isolation are not first-class fields.
- A worker action is fenced by worker ID, not by a unique claim/lease secret.
  A delayed old worker therefore has a weaker ownership proof than desirable.
- Errors are human text rather than stable error codes.
- There is no formal lifecycle event stream or webhook delivery.
- Blob metadata is intentionally small and insufficient for a user-facing
  artifact product.

## Transcription worker

`pipeline/workqueue_worker.py` is a capable but specialized runtime. It adds:

- Parakeet and WhisperX execution;
- language and backend compatibility guards;
- audio download and minimum-size validation;
- ffmpeg preprocessing;
- CUDA OOM batch-size fallback;
- chunked WhisperX language behavior and alignment fallback;
- local disk floors;
- graceful `SIGUSR1` drain;
- one-job lookahead so CPU/network preparation overlaps current GPU work;
- stage timings and cumulative content-duration metrics;
- brokered and direct storage modes.

It should become a transcription executor using a generic worker SDK. Adding
image or LLM branches directly to this file would preserve the coupling and
make every worker image carry unrelated policy.

## Storage broker

Rented workers are treated as untrusted. They get a worker identity, scoped
queue credential, broker endpoint, and job-specific URLs. They do not get
bucket credentials.

For a validated upload:

1. The broker verifies that the authenticated worker owns a running
   transcription job.
2. It issues a source-audio GET URL and a quarantine-output PUT URL.
3. It records an expiring storage lease.
4. The worker uploads and reports digest/size.
5. The broker checks object size and digest, parses JSON, and validates the
   Deepgram-compatible shape.
6. It promotes the quarantine object to the final key.
7. It completes the workqueue job using a broker-held queue credential.
8. It closes superseded leases and garbage-collects abandoned uploads.

A trusted-direct mode exists for explicitly allowlisted trusted workers, but it
must not be used for marketplace machines.

This pattern is directly reusable. The coupling is in the assumptions that an
input is audio, an output is a transcript, and validation is Deepgram JSON.

## Provider control plane

The active controller is a long-running Python monitor with route-specific
configuration. It:

- reads queue, broker, and provider signals;
- classifies canonical deployable backlog rather than using raw queue depth;
- observes the Vast and Salad markets even when idle;
- records offer history, startup attempts, failures, worker tokens, and
  performance in a fleet SQLite database;
- ranks provider candidates using expected cost to complete work, not only
  hourly price;
- gates deployment on hourly caps, drain cost, provider credit, startup
  cooldowns, and capacity policy;
- creates one scoped identity per provider worker;
- waits for useful worker readiness;
- scales down or destroys unhealthy workers;
- publishes status, Prometheus metrics, JSONL telemetry, and reports.

The file is over 21,000 lines and its principal resilience test is over 22,000
lines. That reflects substantial hardening, but also too many responsibilities
inside one module. Provider drivers, workload policy, planning, reconciliation,
telemetry, systemd auditing, and dashboard drift should be separate packages.

### Vast

Vast is a marketplace driver:

- search offers using GPU RAM, CPU, RAM, disk, reliability, verification, and
  hourly price constraints;
- rank usable offers;
- create an instance from an immutable worker image and scoped environment;
- poll readiness;
- destroy failed or excess instances;
- confirm the instance is absent from Vast inventory.

### Salad

Salad is a container-group driver. The legacy system creates one replica-one group per worker
identity instead of a multi-replica group, because every untrusted worker needs
an independently revocable queue credential. GPU class availability and
priority are checked before creation.

### The critical billing invariant

Provider inventory is billing truth. A local database row, missing queue
heartbeat, or deleted token does not prove a rented machine stopped billing.
Destroy must be followed by provider-inventory absence. Only then may local
identity and accounting state be retired.

## Autoscaling and measured economics

The legacy system originally scaled on transcript counts and constants. It now records
cumulative worker counters and converts samples into workload-scoped observed
throughput:

- jobs per GPU hour;
- audio hours per GPU hour;
- cost per audio hour;
- startup duration and failure behavior.

Cost estimation prefers a conservative percentile of recent measured fleet
performance. This is materially safer than assuming every episode has the same
length or every GPU/model combination matches a constant.

The generalized form is:

- each workload declares a unit (`audio_hour`, `image`, `output_token`,
  `megapixel`, `video_megapixel_second`);
- pending work is the sum of estimated units, not raw jobs;
- measured units/GPU-hour are scoped by workload, route, model pack, GPU, and
  provider;
- a candidate is ranked on startup plus drain cost and reliability;
- unknown required credit blocks deployment.

## Reconciliation and governance

The legacy system separates ordinary retry from explicit repair spend:

- the queue owns bounded execution retries;
- the reconciler checks that durable output actually exists and is plausible;
- repair/canary tooling builds a plan with routing and cost evidence;
- applying that plan is an explicit operator action;
- the governor identifies worker-fault patterns and coordinates provider
  cleanup.

This distinction should remain. A corrupt output must not trigger unbounded,
silent GPU spending.

## Generic and coupled surfaces

| Surface | Reusable now | Coupled today |
|---|---|---|
| Queue | lifecycle, priority, idempotency, attempts, heartbeat, cancellation | kind-only matching, no tenant/project |
| Worker | trusted command/HTTP mapping | production Python runtime assumes ASR |
| Artifacts | job ownership, scoped URLs, quarantine, leases | audio/transcript paths and validation |
| Autoscaling | ETA, budget and credit gates, measured throughput | audio/transcript names and backend classifiers |
| Providers | Vast/Salad lifecycle and billing truth | worker templates and naming are transcription-oriented |
| API | API keys, idempotency, attempt mirroring | transcription-only resources and staging layout |
| Operations | audits, governor, reconciler, metrics | hand-wired systemd paths and route-specific assertions |

## What is possible

With a workload registry, inventory-aware claims, a generic artifact broker,
and a worker SDK, the same platform can safely run:

- transcription, diarization, TTS, denoising, and source separation;
- image generation, upscaling, restoration, segmentation, and captioning;
- asynchronous LLM prompt batches, evaluations, embeddings, and reranking;
- OCR and document-vision extraction;
- video generation, interpolation, upscaling, and encoding;
- dataset annotation and synthetic-data production;
- LoRA fine-tuning and bounded single-GPU training;
- 3D reconstruction and Gaussian-splat generation;
- scientific batch inference and simulation.

The primary model remains one independently leased GPU slot per job. A future
gang scheduler can reserve several slots for distributed training, but it is
not necessary for the main workload fleet and should not complicate the first
product.

## Conclusions carried into this repository

The standalone implementation preserves:

- atomic pull scheduling;
- two-step lease and running states;
- heartbeat recovery and cancellation;
- scoped credentials;
- untrusted provider workers;
- job-scoped, validated output brokering;
- provider inventory as billing truth;
- workload-defined cost units;
- credit fail-closed policy;
- immutable worker images;
- explicit route and pool boundaries.

It improves the kernel with claim-time GPU/model-pack matching and unique lease
tokens that fence stale workers. It intentionally leaves `/opt/legacy` unchanged.

