# Operator workbench (web dashboard)

The workbench is a dependency-free web UI served by the control plane so a
deployed client can watch the fleet, inspect jobs, and submit new work without a
separate frontend service. It is the same trust boundary as the CLI: every read
and mutation travels the authenticated `/v1` API.

## Enable

`dgpu serve` turns it on by default. Disable it with `--no-dashboard` or
`DGPU_DASHBOARD=0`. The page itself is an unauthenticated HTML shell; every data
and mutation call still requires a valid bearer token supplied in the browser.

```bash
.venv/bin/dgpu serve --host 0.0.0.0 --port 8088
# open http://gpu-control.example.com:8088/dashboard
```

## Sign in

Paste a control-plane URL and a bearer token into the connect dialog. The token
is kept only in that browser tab's `sessionStorage`.

- a **client** token (subject = project id) sees only its own jobs and aggregate
  fleet capacity. Its submit form is locked to its project.
- a **controller** or **admin** token additionally sees raw agent/slot detail,
  the cross-tenant backlog, the activity feed, and reap-expired-leases. Admins
  can also issue credentials and audit, edit, or revoke every retained token.

## Refresh behavior

The workbench refreshes every five seconds by patching existing cards and
keyed table rows in place. It preserves the active view, scroll position,
focus, and unchanged row nodes instead of replacing the page and blinking.

## What it shows

- **Overview** — job counts by state, GPU capacity (ready/busy/offline, VRAM,
  provider/model/capability breakdown), the route-grouped backlog
  (operators), and a live activity feed.
- **GPUs** — current agents and slots are separated from past provisioned GPUs.
  Provider-absent, failed, and destroyed deployments never contribute live
  capacity, even though their durable agent/slot records and job links remain
  visible under **Past GPUs**. Current disconnected workers are shown offline
  rather than ready.
- **Queues** — every registered workload/route with pending and active depth,
  queued units, recent completions and failures, observed units per GPU busy
  hour, effective throughput, drain ETA, and active/recent worker assignments.
  The statistics window is selectable from 1–168 hours.
- **Jobs** — a server-paginated queue (50 rows per page) filterable by state,
  workload, assigned GPU/worker, and free text. GPU filtering includes any
  recorded attempt, so terminal retries remain attributable even when the
  final lease is cleared. A job drawer
  shows payload, requirements, executor contract, artifacts with authenticated
  download, the per-attempt timeline, and the full event log. Running or pending
  jobs can be cancelled; any job can be resubmitted like-for-like.
- **Workloads** — the trusted, route-bound contracts with cost/throughput and
  scaling hints, and a **Submit** button that opens a form generated from the
  workload's payload JSON Schema (only schema-bounded fields; the trust
  boundary is preserved).
- **Admin** — see deployed GPUs first, page through provider offers, reap
  expired leases, inspect the backlog, issue scoped tokens, list the complete
  credential ledger (including last successful use), sort it by status,
  expiration, issued, updated, or last-used time, edit live claims/expiry,
  revoke credentials while retaining their audit rows, and deploy provider
  workers (admin only). Token secrets and
  password hashes are never returned. Salad choices show GPU class,
  priority, live/unknown availability, hourly rate, fleet-hourly cost, and
  24-hour/monthly estimates before confirmation. Provider state and DGPU agent
  heartbeat/slot state reconcile automatically in the background.

## Access over Tailscale

The control plane binds to `127.0.0.1` by default. Expose it to your tailnet
with **Tailscale Serve**, which keeps the control plane on localhost and lets
Tailscale handle exposure, TLS, and tailnet ACLs. No code change is required:
the workbench uses the page's own origin for API calls, so it works as soon as
you load it through the tailnet URL.

This was verified on node `controller.internal.example`.

One-time, on the host running the control plane (requires root/operator):

```bash
# let your user manage serve without sudo each time
sudo tailscale set --operator="$USER"

# proxy HTTPS on the tailnet name to the localhost control plane
# (cert is provisioned automatically)
tailscale serve --bg --https 443 http://127.0.0.1:8088
tailscale serve status   # -> https://<host>.<tailnet>.ts.net
```

Then open `https://<host>.<tailnet>.ts.net/dashboard` from any device on the
tailnet and sign in with a client/admin/controller token. The connect dialog
auto-fills the endpoint from the page origin.

Keep the control plane running with a process manager. This host uses the
hardened split units in `deploy/systemd/`: `dgpu-private.service` owns the
localhost dashboard and provider lifecycle, while `dgpu-public.service` owns
the dashboard-disabled LAN backend. See `docs/provider-workers.md`. For a
single-listener installation, a minimal unit looks like:

```ini
# /etc/systemd/system/dgpu-serve.service
[Unit]
Description=Distributed GPU control plane
After=network-online.target

[Service]
Type=simple
User=REPLACE_ME
WorkingDirectory=/opt/distributed-gpu
Environment=DGPU_DB=/var/lib/dgpu/control.sqlite3
Environment=DGPU_ARTIFACTS=/var/lib/dgpu/artifacts
Environment=DGPU_WORKLOADS=/var/lib/dgpu/workloads
ExecStart=/opt/distributed-gpu/.venv/bin/dgpu serve --host 127.0.0.1 --port 8088
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

The `tailscale serve` config persists across reboots; `dgpu serve` is the only
thing that needs a unit.

### Notes and alternatives

- Serve is **tailnet-only**. WireGuard already encrypts traffic, so even plain
  HTTP serve is safe; HTTPS simply avoids browser warnings.
- Do not expose this dashboard listener with Funnel or a public reverse proxy.
  For provider workers use the separate `--no-dashboard` listener and stable
  TLS reverse-proxy pattern in `docs/provider-workers.md`.
- To bind the control plane directly to the Tailscale interface instead of
  proxying: `dgpu serve --host $(tailscale ip -4) --port 8088`, then open
  `http://<host>.<tailnet>.ts.net:8088/dashboard`. This also exposes the port on
  the tailnet IP and skips TLS, so Serve is preferred.
- Disable the proxy with `tailscale serve --https=443 off` or
  `tailscale serve reset`.

## Queue ETA methodology

The ETA uses completed work in the selected window. DGPU divides completed
workload units by successful GPU busy time to obtain units per worker-hour,
then multiplies by currently active slots (or recent serving slots during a
poll gap). Backlog units divided by that effective rate gives the drain ETA.
Empty queues report zero; queues without successful history or serving workers
report **insufficient data**. The estimate assumes the observed workload mix
and worker allocation remain stable.

## New endpoints

The dashboard reuses the existing `/v1` API. Read-only aggregations are
provided because clients cannot otherwise observe capacity or activity in one call:

- `GET /v1/dashboard/overview` — role-aware histogram + fleet capacity summary +
  backlog. Available to `client`, `controller`, and `admin` (not `agent`).
  Clients receive only their own job counts and aggregate capacity; the
  cross-tenant backlog is omitted.
- `GET /v1/dashboard/queues` — route queue depth, outcomes, observed service
  rate, ETA, and active/recent workers. Client results are project-scoped;
  operator results span the control plane.
- `GET /v1/dashboard/recent` — recent fleet events, scoped to the caller's
  project for clients and global for operators.
- `GET /v1/control/tokens` — safe metadata for every active, expired, or revoked
  credential, including its last successful authentication time (admin only).
- `PUT /v1/control/tokens/{token_id}` — update a live token's role, subject,
  scopes, and expiry without exposing or rotating its secret (admin only).
- `DELETE /v1/control/tokens/{token_id}` — immediately revoke a token while
  retaining its metadata and revocation timestamp (admin only).

These endpoints are hidden from the OpenAPI schema (`include_in_schema=False`) and are only
mounted when the dashboard is enabled.
