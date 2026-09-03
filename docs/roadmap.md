# Implementation roadmap

## Implemented foundation

- workload YAML and payload JSON Schema admission;
- persisted route/executor/resource/artifact/cost snapshots;
- project-scoped idempotent jobs;
- atomic inventory-aware one-GPU-slot claims;
- confirm, heartbeat, cancellation, retry, dead-letter, release, and stale
  lease recovery;
- unique claim secrets that fence stale workers;
- attempt and event history;
- scoped scrypt-backed credentials;
- local GPU discovery and one executor loop per GPU;
- trusted command, container, HTTP, and plugin executor boundaries;
- streamed, size-bounded, content-addressed output uploads;
- JSON, JSON Schema, image, and ffprobe validation;
- authenticated artifact download;
- generic workload-unit autoscaling and cost-to-drain ranking;
- local capacity accounting plus Vast and Salad provider drivers;
- read-only multi-provider fleet planning with credit gates;
- executable demo and fixed SDXL Diffusers plugin plus transcription and Qwen
  batch contract scaffolds;
- one verified Salad RTX 4090 run with authenticated queueing, three PNG
  uploads/downloads, provider-absence verification, and credential retirement.

## Next: complete artifact plane

- client input upload sessions;
- job-scoped input download leases;
- S3-compatible backing with presigned direct transfer;
- quarantine-to-final promotion;
- digest/size confirmation on direct uploads;
- abandoned lease cleanup and retention policy;
- dimensions, duration, and other extracted artifact metadata;
- optional custom validator plugins in a sandbox.

## Implemented image workload slice

- fixed SDXL Base 1.0 model ID and immutable revision;
- bounded prompt/seed/dimension/step fields;
- one cached Diffusers pipeline per GPU;
- pinned Salad base-image digest and Python runtime versions;
- PNG validation, content addressing, and embedded provenance;
- successful real-provider cold start followed by warm sequential jobs.

Production image-pack follow-up includes a license manifest, pre-baked worker
image, startup warmup/probes, throughput history, cancellation during
inference, and optional separately routed ComfyUI workflows.

## Implemented provider reconciliation foundation

- durable provider-resource/token records and operator-confirmed scale-out;
- automatic bounded provider reconciliation in the private controller;
- separate provisioning, provider-running, DGPU-connected, stale, and absent states;
- Salad per-instance state and selected-price reconciliation;
- failed-launch credential cleanup;
- destroy plus inventory-absence verification before credential retirement.

Remaining provider lifecycle work includes startup timeouts, provider-account
circuit breakers, orphan adoption/alerts, idle/drained scale-down, and richer
apply/audit events. Autonomous provider mutation remains deliberately disabled.

## Next: measured fleet economics

- cumulative agent counters by workload/route/model/GPU;
- conservative recent throughput percentiles;
- startup and interruption history;
- measured cost per workload unit;
- provider market history and availability;
- budget reservation and actual cost ledger.

## Next: Qwen batch pack

- pinned Qwen3.5-9B quantized model pack;
- warm vLLM process and health checks;
- bounded prompts/context/output admission;
- token-aware batch contract;
- structured output validation;
- generated-token throughput and cost metrics.

## Product hardening

- tenant/project database and API-key administration;
- per-project quotas, budgets, retention, and audit log;
- formal stable error taxonomy;
- webhooks/event delivery;
- Prometheus metrics and generated dashboards;
- PostgreSQL option and control-plane HA when scale requires it;
- explicit repair/canary plans for every costly workload;
- pool overlap and configuration-generation tests;
- supply-chain signing and worker image/model provenance.

## Deferred

Gang scheduling for a single multi-GPU job remains a separate extension. It
requires multi-slot reservation, rendezvous, topology placement, coordinated
startup, and group failure semantics; none are needed for the primary
independent-job fleet.
