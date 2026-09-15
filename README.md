# Distributed GPU

A standalone control plane for distributing independent asynchronous jobs
across local and rented GPU workers. A host registers one independently
schedulable slot per GPU, so a four-GPU machine can execute four unrelated jobs
at once.

## What exists now

- versioned workload YAML and JSON Schema admission;
- an operator **workbench** (web dashboard) for clients to watch the fleet,
  inspect jobs, and submit schema-bounded work, served at `/dashboard`;
- durable SQLite jobs, attempts, events, priorities, and project-scoped
  idempotency;
- atomic claims matched on VRAM, compute capability, GPU model, provider,
  labels, capabilities, and installed model packs;
- `pending -> leased -> running -> succeeded/dead-letter/cancelled` lifecycle;
- expiring per-claim secrets, heartbeat recovery, cancellation, and centrally
  bounded retry policy;
- one concurrent executor loop per physical GPU;
- trusted command, digest-pinned container, local HTTP, and plugin executors;
- scoped client, agent, controller, and admin credentials;
- streamed, bounded, content-addressed output uploads with JSON, schema, image,
  and media validation;
- workload-unit autoscaling and total-cost-to-drain candidate ranking;
- local capacity plus Vast and Salad provider-driver boundaries;
- a read-only, credit-gated multi-provider fleet planner;
- executable demo, fixed SDXL Diffusers, and offline English-to-Spanish batch
  translation routes, plus transcription and Qwen batch contract definitions.

The queue, local agent, output broker, SDXL plugin, Argos-derived CTranslate2
adapter, and dry-run fleet planner are runnable. The translation CPU image has
passed offline and full queue smoke tests; its CUDA image has passed offline
and full queue tests on an RTX A5000 Laptop GPU. Qwen and transcription remain
workload-pack scaffolds. The
narrow `scripts/salad_sdxl.py` helper can create and authoritatively destroy
one explicit Salad test worker; unattended autoscaler apply remains disabled
until there is a durable resource/cost ledger and full reconciliation audit.

The fixed SDXL route intentionally does not use ComfyUI. Jobs may choose only
bounded inference fields such as prompt, seed, dimensions, and steps; the
trusted route fixes the model, revision, and executor code.

## Install and verify

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/dgpu validate
.venv/bin/pytest
```

`ffprobe` is additionally required when a workload uses the `media_probe`
artifact validator. A real NVIDIA agent discovers its slots with `nvidia-smi`.

## Local two-worker smoke test

The included demo proves scheduling, leases, execution, upload validation, and
artifact attachment without requiring a GPU. It uses the no-GPU `demo-gpu`
workload, which lives in the test fixtures (`tests/fixtures/workloads`) and is
intentionally kept out of the operational `workloads/` directory. Start from a
clean local state:

```bash
mkdir -p state
export CLIENT_TOKEN="$(.venv/bin/dgpu token \
  --role client --subject demo-project)"
export AGENT_TOKEN="$(.venv/bin/dgpu token \
  --role agent --subject demo-host --scope routes=demo-gpu)"
```

Run the control plane in one terminal (pointing it at the fixture workloads so
the `demo-gpu` route is available):

```bash
DGPU_WORKLOADS=tests/fixtures/workloads \
  .venv/bin/dgpu serve --host 127.0.0.1 --port 8088
```

The operator workbench is served at `http://127.0.0.1:8088/dashboard`. Open it
and sign in with any client/admin/controller token (for example the
`$CLIENT_TOKEN` issued below). Disable it with `--no-dashboard` or
`DGPU_DASHBOARD=0`. To expose it to your tailnet over HTTPS with no code
change, put `tailscale serve` in front of `http://127.0.0.1:8088`.

Submit two independent jobs in another terminal:

```bash
for n in 1 2; do
  curl --fail-with-body http://127.0.0.1:8088/v1/jobs \
    -H "Authorization: Bearer $CLIENT_TOKEN" \
    -H 'Content-Type: application/json' \
    -H "Idempotency-Key: demo-$n" \
    --data "{\"project_id\":\"demo-project\",\"workload\":\"demo-gpu\",\"payload\":{\"message\":\"job $n\"}}"
done
```

Then register two fake slots and process the queue:

```bash
export TOWER_TOKEN="$(.venv/bin/dgpu token \
  --role agent --subject tower-a --scope routes=demo-gpu)"
DGPU_TOKEN="$TOWER_TOKEN" .venv/bin/dgpu agent \
  --endpoint http://127.0.0.1:8088 \
  --agent-id demo-host \
  --capability demo-gpu \
  --fake-gpus 2 \
  --once
```

For a real local host, omit `--fake-gpus`; each GPU reported by `nvidia-smi`
becomes a separate slot. Advertise only capabilities and model packs actually
installed on that host.

```bash
DGPU_TOKEN="$AGENT_TOKEN" .venv/bin/dgpu agent \
  --endpoint https://gpu-control.example.com \
  --agent-id tower-a \
  --provider local \
  --capability demo-gpu
```

## Provider planning

Provider machines are outside the tailnet and reach the API through
`https://gpu-control.example.com`, while the dashboard remains tailnet-only.
The reverse proxy terminates TLS and forwards to a separate API-only listener
(`--no-dashboard`, no `DGPU_PUBLIC_URL`). Verify that public ingress before
renting capacity. The
workbench's Admin tab can deploy Vast/Salad workers per pool policy — Salad
shows live GPU-class/priority availability and hourly/daily/monthly estimates,
then requires the selected trusted offer in the confirmation flow. Provider
resources and DGPU agent heartbeats reconcile automatically in the private
controller. Every deploy is operator-confirmed, ledgered, and torn down with verified provider
absence before its credential is retired. Pool files pin the worker image,
own an explicit set of slots, and set provider-specific price and credit
policy. Planning is read-only:

```bash
.venv/bin/dgpu plan \
  --pool pools/image-standard.example.yaml \
  --vast-api-key-file /run/secrets/vast-api-key
```

The planner sums queued and running workload units, subtracts current pool
slots, fails closed when required credit is unknown, searches eligible offers,
and ranks them by startup-adjusted estimated drain cost. It never launches an
instance.

## Repository map

- `src/gpu_orchestrator/store.py`: durable queue and placement kernel;
- `src/gpu_orchestrator/dashboard.py` + `dashboard.html`: read-only operator
  workbench (overview aggregation and the self-contained web UI);
- `src/gpu_orchestrator/agent.py`: GPU discovery, slot loops, leases, and
  executor boundary;
- `src/gpu_orchestrator_workloads/sdxl.py`: fixed, cached SDXL Diffusers
  executor;
- `src/gpu_orchestrator_workloads/argos_translate.py`: fixed, cached,
  GPU-index-aware CTranslate2 translation executor;
- `src/gpu_orchestrator/artifacts.py`: local content-addressed output broker;
- `src/gpu_orchestrator/controller.py`: cross-provider dry-run fleet planning;
- `src/gpu_orchestrator/providers/`: Vast and Salad lifecycle drivers;
- `workloads/`: declarative workload and route examples;
- `deploy/`: pinned worker definitions, reusable Docker images, and
  private/public systemd units;
- `scripts/verify_public_ingress.sh`: fail-closed TLS/API/wheel/dashboard
  boundary verification before provider deployment;
- `scripts/salad_sdxl.py`: explicit Salad SDXL create/status/destroy lifecycle;
- `pools/`: provider-pool policy examples;
- `tests/`: race, fencing, placement, API, artifacts, economics, and provider
  contract tests.

The initial scheduler deliberately supports one GPU per independent job.
Coordinated multi-GPU jobs require a separate gang-scheduling extension and do
not complicate this primary use case.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, test commands, and
the release checklist. Security concerns should follow the process in
[SECURITY.md](SECURITY.md).

## License

Copyright (C) 2026 Aljaz Ceru

This project is licensed under the GNU General Public License v3.0 or later
(GPL-3.0-or-later). See [LICENSE](LICENSE) for the full text.
