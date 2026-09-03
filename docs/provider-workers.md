# Provider worker access (Vast.ai, Salad.com, any non-tailnet host)

Local towers and operators reach the control plane privately. Provider
machines cannot: they are provisioned on demand outside the tailnet and need a
**public controller URL**. The host runs two API processes over the same WAL
SQLite database and artifact directory so the trust boundary is explicit:

```text
operators ──► https://controller.internal.example (tailnet only)
                 └──► 127.0.0.1:8088  dgpu-private
                       dashboard + authenticated API + provider lifecycle

workers ────► https://gpu-control.example.com (public reverse proxy/TLS)
                 └──► 10.0.0.10:8089 dgpu-public
                       authenticated API; no dashboard; no provider keys
```

`deploy/systemd/dgpu-private.service` and
`deploy/systemd/dgpu-public.service` define these processes. SQLite WAL mode
and short transactions permit the two listeners to share state safely. The
public service runs as the dedicated `dgpu-public` Unix account; read-only
bind mounts expose only its Python runtime, workload registry, pools, and
wheel directory. Group permissions expose the live SQLite file and artifact
directory, while `/home`, `/opt/legacy`, `/etc/dgpu`, historical test state, and
the private process environment are inaccessible. The public service also
clears `DGPU_PUBLIC_URL`, so deployment endpoints and provider drivers are not
constructed there, and starts with `--no-dashboard`; `/dashboard` and
`/v1/dashboard/*` therefore return 404.

## Reverse-proxy contract

**Current activation state (2026-08-28):** stable ingress is active at
`https://gpu-control.example.com` and forwards to the API-only backend on
`10.0.0.10:8089`. Internal split-horizon DNS reaches the reverse proxy at
`10.0.0.5`; public DNS reaches the WAN edge. Future provider deployments are
configured with the stable URL and wheel URL.

Create a TLS frontend for `gpu-control.example.com` and forward HTTP to
`10.0.0.10:8089`. Preserve the `Authorization` header and `Host`, set
`X-Forwarded-Proto: https`, and use timeouts large enough for artifact
transfers (at least 120 seconds). Prefer an allowlist of these public paths:

- `/healthz`
- `/bootstrap/*.whl`
- `/v1/*`

Return 404 for `/dashboard`, `/docs`, `/redoc`, and `/openapi.json` at the
proxy as defense in depth. Do not add proxy-level bearer credentials: workers
already use their short-lived route-scoped DGPU bearer token. The backend is
plain HTTP by design because TLS terminates at the trusted reverse proxy.

The private endpoint remains:

```sh
tailscale serve --bg --yes --https 443 http://127.0.0.1:8088
```

The temporary API-only Cloudflare quick tunnel was retired after stable
activation. The dashboard remains reachable only through Tailscale.

## Verify before renting capacity

```sh
scripts/verify_public_ingress.sh https://gpu-control.example.com
```

This proves strict public TLS, `/healthz`, byte-identical wheel download, 401
for an unauthenticated API request, 404 for the public dashboard, and 404 for
a non-wheel bootstrap path. The public bootstrap endpoint contains no
credential; authenticated API operations require a bearer token.

After that passes, atomically switch future provider deployments to the stable
hostname (the helper verifies everything *before* editing configuration):

```sh
scripts/activate_public_ingress.sh https://gpu-control.example.com
sudo grep '^DGPU_.*URL=' /etc/dgpu/controller.env
```

Existing workers keep their injected endpoint, so retire any old-fallback
workers before stopping that tunnel. If the deployment ledger has no live
rows, stop the fallback with `kill $(cat /tmp/cloudflared.pid)`.

## Provision a provider worker

The worker receives exactly four things (never a provider API key):

1. the public controller URL,
2. the public wheel URL,
3. a fixed agent identity,
4. a short-lived agent token scoped to the routes it may serve.

```sh
# on the control plane: scoped, expiring worker credential
dgpu token --role agent --subject vast-host-7 \
  --scope routes=libretranslate --expires-hours 6

# on the provider machine:
python3 -m venv .venv
.venv/bin/pip install "$PUBLIC_URL/bootstrap/distributed_gpu-0.1.0-py3-none-any.whl"
DGPU_TOKEN="$AGENT_TOKEN" .venv/bin/dgpu agent \
  --endpoint "$PUBLIC_URL" \
  --agent-id vast-host-7 --provider vast --provider-resource-id "$INSTANCE_ID" \
  --capability libretranslate --model-pack libretranslate-1.9.6-bounded-sbd
```

On a real GPU host omit `--fake-gpus`; every GPU reported by `nvidia-smi`
registers as its own schedulable slot. For translation workers, run the
LibreTranslate service on the same host (loopback-only, `127.0.0.1:5001`) —
see `workloads/README.md`. For a complete Docker build/run/teardown path, use
`docs/docker-workers.md`. Registration rejects any capability outside the
token's route scope, so a leaked worker token cannot enlarge its blast radius.

Idle slots poll for work once per second by default; over public tunnels and
large fleets prefer `--poll-seconds 3` to reduce chatter.

## Deploy provider workers from the workbench

The dashboard's **Admin → Capacity & deploy** section lists every pool in
`pools/` with its live plan (current vs target slots, queued units, provider
credit, blocked reasons, top offers) and offers an explicit, human-confirmed
**Deploy…** action. This is deliberately *not* an autonomous autoscaler
(AGENTS.md keeps unattended apply disabled); every deploy is an operator
action recorded in a durable `deployments` ledger.

Each deployed worker gets a unique identity
`<worker_prefix>-<hex>`, a short-lived agent token scoped to the pool's route
capability, the public controller URL, and (if configured) the public wheel
URL — nothing else. Provider API keys never leave the control-plane process.

Enable on the control plane:

```sh
DGPU_PUBLIC_URL=https://<public-url> \   # required to mount the endpoints
DGPU_WHEEL_URL=https://<public-url>/bootstrap/distributed_gpu-0.1.0-py3-none-any.whl
DGPU_DEPLOYMENT_RECONCILE_SECONDS=15      # private controller only
VAST_API_KEY=...                          # per-provider, optional
SALAD_API_KEY=... SALAD_ORGANIZATION=... SALAD_PROJECT=...
dgpu serve
```

Without provider keys the capacity view still renders (with
`driver_not_configured`) and deploys fail closed with a clear error. Salad
pools need a `provider_templates.salad` container-group template in the pool
file (see `pools/translate-live.example.yaml`). The pinned LibreTranslate
integration image runs Python 3.10, so the DGPU wheel intentionally supports
Python >=3.10 and is smoke-tested inside that exact image. Salad startup/readiness
probe events alone do not prove DGPU readiness; only agent registration and a
fresh heartbeat transition the ledger to `active`.

### Salad price and availability selection

For Salad, the workbench displays the cheapest eligible GPU-class/priority
combinations before confirmation. The driver refreshes the organization GPU
class catalog and priority prices from Salad (five-minute cache), filters it
through the workload's trusted VRAM, compute-capability, model, provider, and
pool cost constraints, then checks the four cheapest eligible classes through
Salad's availability preflight (30-second cache). The translation route includes
8 GB GTX 1070/1080/1080 Ti (cc 6.1) as legacy-capable options as well as RTX;
Pascal works but lacks Tensor Cores, so compare cost per translated unit rather
than hourly price alone. Each row shows:

- GPU class and priority (`batch`, `low`, `medium`, or `high`),
- live available count, or `unknown` if the optional endpoint is unavailable,
- price per worker-hour plus 24-hour and monthly estimates,
- whether price and availability came from the live API or reviewed fallback.

Monthly estimates use 730 continuously billed hours. They appear on every
listed offer, in the confirmation modal (including worker count), and on every
deployment ledger row; they are estimates rather than a commitment to run for
a full month.

The deploy modal requires an `offer_id` from that server-generated list and
shows per-worker and total fleet hourly cost. The controller re-fetches and
validates the offer, cost cap, credit policy, and known availability before
issuing worker credentials. It then replaces the trusted Salad template's GPU
class and priority with the selected offer; jobs cannot supply either value.
Without an explicit choice, API clients receive the cheapest currently
eligible hourly offer. Known-zero capacity cannot be deployed; unknown
availability remains operator-selectable because Salad preflight is advisory,
not allocation truth.

`batch` is normally cheapest but has the weakest reliability prior and may be
interrupted or slow to allocate. Availability counts are a preflight signal,
not a reservation. Billing continues until Destroy; token expiry does not
stop a Salad group. The displayed rate is the Salad GPU-class priority rate;
provider billing records remain authoritative.

The private controller reconciles automatically every 15 seconds by default;
**Reconcile now** remains available for an immediate check. One inventory read
per provider updates provisioning/starting/active/stale/absent state. Salad's
per-instance counts override its misleading group-level `running` status, and
a provider resource becomes `active` only after its DGPU agent has registered
and has a fresh heartbeat. Reconciliation also refreshes the selected Salad
GPU-class/priority price, repairing older synthetic ledger estimates.
Provider inventory remains billing truth.

**Destroy** enforces the verified teardown ordering: refuse while the worker
holds an active job; destroy the resource and require inventory absence; only
then disable the credential and close the ledger row.

## Verified end-to-end (2026-07-22)

A simulated provider worker ran the full loop through a fresh quick tunnel
against a scratch control plane (no real resources touched): job submitted
through the public URL → fake-GPU agent (`--provider vast`) registered,
claimed, confirmed, executed, completed → artifact downloaded through the
public URL, content and contract intact. The same loop is covered by the
test suite (`tests/test_agent.py`, `tests/test_api.py`).

## Service operations

```sh
systemctl status dgpu-private dgpu-public
sudo journalctl -u dgpu-private -u dgpu-public -f
sudo systemctl restart dgpu-private dgpu-public
```

Both services are enabled at boot. To retire public ingress, first ensure the
deployment ledger has no live provider rows, stop any fallback tunnel, then
`sudo systemctl stop dgpu-public`. Keep `dgpu-private` and Tailscale Serve up
for the operator dashboard. `tailscale serve --https=443 off` removes the
private proxy configuration and should not be part of normal API teardown.
