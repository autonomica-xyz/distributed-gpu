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
- scoped client, agent, controller, and admin credentials, with per-route
  queue depth in `/v1/workloads` exposed only to tokens granted
  `--scope perms=dynamic` (admins implicitly);
- streamed, bounded, content-addressed output uploads with JSON, schema, image,
  and media validation, plus private client input uploads that agents fetch
  only while holding a lease on a job that references them;
- workload-unit autoscaling and total-cost-to-drain candidate ranking;
- local capacity plus Vast and Salad provider-driver boundaries;
- a read-only, credit-gated multi-provider fleet planner;
- executable demo, fixed SDXL Diffusers, offline English-to-Spanish batch
  translation, two-route GPU transcription, and vLLM-backed Qwen batch
  routes, all loading weights only from hash-verified model packs.

The narrow `scripts/salad_sdxl.py` helper can create and authoritatively destroy
one explicit Salad test worker; unattended autoscaler apply remains disabled
until there is a durable resource/cost ledger and full reconciliation audit.

The fixed model routes intentionally do not take model code from a job.
The SDXL route in particular does not use ComfyUI. Jobs may choose only
schema-bounded fields (prompt, seed, dimensions, steps, language,
diarization bounds, sampling knobs, an uploaded audio input); the
route pins the model id, immutable revision, executor handler, and endpoint.

## Model packs and worker runtime profiles

Weights are never fetched at job time. Operators provision each pack with
`gpu_orchestrator_workloads.model_packs.install_hf_pack(lock, target)`
(requires the optional `models` extra, `huggingface-hub`), which downloads
the pinned Hugging Face snapshot and verifies every file against the repo's
own LFS/blob digests; executors then re-run `verify_hf_pack` on the pack's
`manifest.json` before weights load, so verification at load time is pure
local I/O with no token. Pinned locks live in `models/`:

- `sdxl.lock.json`: `sdxl-base-1.0-fp16-76d28af` —
  `stabilityai/stable-diffusion-xl-base-1.0@76d28af…`, fp16 safetensors
  only, no gate (OpenRAIL++ license);
- `qwen.lock.json`: `qwen3.5-9b-awq-156edc4` —
  `cyankiwi/Qwen3.5-9B-AWQ-4bit@156edc4…`, compressed-tensors W4, no gate
  (Apache-2.0);
- `transcription.lock.json`: four packs —
  `nvidia/parakeet-tdt-0.6b-v3@541d1f9…` (CC-BY-4.0),
  `Systran/faster-whisper-large-v3@edaa852…` (MIT),
  `facebook/wav2vec2-base-960h@22aad52…` (Apache-2.0), and
  `pyannote/speaker-diarization-community-1@3533c8c…`. The diarization pack
  is gated (HF contact-details form, CC-BY-4.0 attribution required), so a
  fresh cold install needs an approved HF token at provisioning time; the
  token is used only then and never appears in job payloads.

Install packs under the route's `*_MODEL_DIR` paths (conventionally
`/models/<pack>`), owned by the worker user and kept read-only at job time.
The installer never overwrites content it cannot prove is one of its own
pack dirs, and a symlink anywhere in the traversal refuses the operation.

ML runtimes are deliberately separate profiles, not controller
dependencies: root development installs only `-e '.[dev]'` (which includes
`huggingface-hub` for pack provisioning), while `'.[models]'` is the
production install for pack tooling alone. The transcription worker image
installs `deploy/transcription/requirements.txt` (Python 3.12, torch 2.8,
NeMo, WhisperX, pyannote) plus ffmpeg/ffprobe binaries and baked-in NLTK
`punkt_tab` English data — workers run egress-off and must never call
`nltk.download` at job time. The Qwen worker runs the isolated vLLM stack
in `deploy/llm/requirements.txt` (vLLM 0.20.1, torch 2.11, transformers
5.14.1). For `llm-batch` the operator pre-provisions the verified pack and
starts one fixed loopback server (`VLLM_SERVER_DEV_MODE=1`,
`--enable-sleep-mode`, `--host 127.0.0.1`); the plugin then issues only
bounded requests — `/tokenize`, one true batched `/v1/completions`, and
`/sleep`/`/wake_up` for VRAM handoff — and never arbitrary payload-chosen
models or endpoints.

## Job inputs

Clients upload private, project-scoped blobs with
`POST /v1/projects/{project}/inputs?workload=<name>&name=<input>`; the
controller validates the blob against the workload's declared input spec
(media type, `max_bytes`, validator) and returns a content-addressed
`/v1/projects/{project}/inputs/{sha256}` URI. A job references inputs only
by these URIs. The agent downloads each declared input under its active
lease — the controller authorizes the download only when the leased job's
inputs reference that exact blob — verifies size and SHA-256, and
materializes the file under `workdir/inputs/` before the executor runs.
External, `file://`, or off-origin URIs are refused, so an input can never
smuggle the bearer token or read worker-local files.

## Workload catalog and dynamic inventory

`dgpu serve` loads declarative route YAML from `DGPU_WORKLOADS`
(default `workloads/`). `GET /v1/workloads` returns the live catalog;
per-route queue depth is exposed only to tokens granted
`--scope perms=dynamic` (and admins). Tokens are further scoped by role and
`--scope routes=<workload>` so an agent credential reaches only the routes
it serves. Model servers stay private: worker-local HTTP endpoints bind to
`127.0.0.1` and never leave the worker host.

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
- `src/gpu_orchestrator_workloads/model_packs.py`: verified Hugging Face
  pack install (`install_hf_pack`) and offline manifest verification
  (`verify_hf_pack`);
- `src/gpu_orchestrator_workloads/transcription.py`: parakeet and WhisperX
  executors for the `transcribe` routes;
- `src/gpu_orchestrator_workloads/qwen.py`: bounded adapter for a fixed
  loopback vLLM server;
- `src/gpu_orchestrator_workloads/argos_translate.py`: fixed, cached,
  GPU-index-aware CTranslate2 translation executor;
- `src/gpu_orchestrator/artifacts.py`: local content-addressed broker for
  outputs and private job inputs;
- `src/gpu_orchestrator/controller.py`: cross-provider dry-run fleet planning;
- `src/gpu_orchestrator/providers/`: Vast and Salad lifecycle drivers;
- `workloads/`: declarative workload and route examples;
- `deploy/`: pinned worker definitions, reusable Docker images, ML runtime
  requirement sets, and private/public systemd units;
- `models/`: reviewed immutable pack locks consumed by `install_hf_pack`;
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
