# Autonomous multi-provider fleet management plan

Status: design only; autonomous provider mutation remains disabled.

This plan adds manual GPU deployment to the **GPUs** tab and introduces bounded,
operator-authorized scaling rules. It generalizes the useful provider-control
patterns studied in `/opt/legacy/pipeline/` and its referenced infrastructure
control-plane scripts without importing the legacy system's transcription coupling or
modifying the legacy system.

## Product outcomes

1. An operator can deploy a compatible GPU directly from the **GPUs** tab.
2. An administrator can create a standing rule such as:
   > If the `translate-text-batch/libretranslate-batch-v1` drain ETA remains
   > above two hours, add capacity from Vast or Salad using the cheapest
   > eligible expected cost-to-drain candidate.
3. Every decision is explainable before and after execution: inputs, candidate
   offers, exclusions, selected offer, budget, provider result, readiness, and
   teardown remain durable audit data.
4. Automatic actions remain workload-route bounded. A rule can select only a
   reviewed pool; it cannot choose arbitrary images, commands, models, LoRAs,
   graphs, or worker capabilities.

## Lessons retained from the legacy system

The legacy system's mature control path contains several patterns worth preserving:

- **One provider-neutral owner.** Provider helpers manage capacity mechanics,
  while the control plane owns queue policy and fallback. DGPU should keep this
  split between `DeploymentService`, provider drivers, and a new policy engine.
- **Provider inventory is billing truth; worker heartbeat is readiness truth.**
  Provider `running` is not usable capacity until DGPU receives a fresh scoped
  agent heartbeat.
- **Count active and in-flight capacity separately.** Never deploy duplicates
  merely because a new worker has not registered yet.
- **Rank total economics, not hourly price.** the legacy system combines measured throughput,
  startup time, startup cost, workload size, reliability, and hourly price.
- **Persist market and outcome history.** Provider observations, deployment
  attempts, readiness times, failures, throughput samples, and price trends
  improve later decisions.
- **Failure cooldowns and provider retry windows.** Repeated startup failures for
  one provider/GPU/priority shape temporarily remove it from consideration.
- **Conservative stale/default performance priors.** Unknown or old throughput
  must not be treated as precise measured performance.
- **Fail closed on policy and credit.** Revalidate provider health, offer price,
  availability, credit, and spend caps immediately before mutation.
- **Graceful scale-down.** Stop claims, wait for active jobs, destroy and verify
  provider absence, then retire the credential.
- **Report-only first.** Dry-run decisions, status snapshots, metrics, and
  append-only events precede unattended apply.
- **Bounded recovery.** Stale startup cleanup, orphan/token reconciliation, and
  provider fallback need explicit retry ceilings and durable reasons.

DGPU should not copy the legacy system's 20,000-line control-plane monolith, transcription
fields (`audio_hours`, worker regexes), duplicate token stores, or separate
provider state databases. DGPU already has generic workload units, route
contracts, a deployment ledger, provider drivers, queue statistics, and scoped
tokens. The new engine should be small modules over those primitives.

## Safety model

Enabling an `apply` rule is a standing, bounded operator authorization. It is
not permission for arbitrary capacity changes.

- Global default and migration default: `automation_mode=off`.
- Rule modes: `off`, `report-only`, and `apply`.
- Only an admin can create, edit, enable, or delete an apply rule.
- A controller may inspect rules, evaluations, and actions; manual deploy
  remains available to the existing fleet-operator roles.
- A global kill switch stops new automatic actions without stopping workers.
- Rules reference one reviewed pool and therefore one workload, route, image,
  command, capability, model pack, and resource contract.
- Per-rule limits: maximum provider slots, additions per evaluation, hourly
  spend, projected drain spend, daily spend, token lifetime, startup deadline,
  and provider retry count.
- Global limits apply across all rules to prevent individually safe policies
  from overspending together.
- Automatic teardown applies only to deployments created by that rule (or
  explicitly adopted), never owned towers or manually managed provider groups.
- Ambiguous provider create errors trigger an inventory lookup by deterministic
  resource name before credential retirement or fallback, preventing orphaned
  double billing.

## Rule contract

Rules are durable database records with optimistic versions. A representative
policy is:

```yaml
name: translation-drain-under-two-hours
pool_id: translate-batch-live
mode: report-only                 # promote to apply after observation
trigger:
  metric: drain_eta
  above_hours: 2
  sustain_seconds: 600
  require_observed_throughput: true
scale_up:
  target_eta_hours: 1.5
  max_provider_slots: 4
  max_additions_per_evaluation: 1
  cooldown_seconds: 1800
  startup_deadline_seconds: 1800
providers:
  allowed: [vast, salad]
  require_known_availability: false
  vast:
    require_known_credit: true
    max_hourly_cost_usd: 0.10
  salad:
    require_known_credit: false
    max_hourly_cost_usd: 0.08
budgets:
  max_rule_hourly_usd: 0.20
  max_projected_drain_usd: 1.00
  max_daily_actions: 6
scale_down:
  enabled: true
  below_eta_hours: 0.5
  sustain_seconds: 1800
  idle_seconds: 900
  min_provider_slots: 0
```

The scale-up and scale-down thresholds create hysteresis. A rule must stay over
or under its threshold for the configured duration; one noisy ETA sample cannot
rent or destroy a GPU.

### Credit semantics

Vast can be selected automatically only when its authenticated credit is known,
fresh, and sufficient after existing budget reservations.

Salad currently has no dependable account-credit endpoint. DGPU must not label
Salad credit as known. An admin may explicitly permit Salad with an unknown
credit state, but only inside hourly, projected-drain, daily-action, and fleet
caps. A future working Salad billing endpoint can tighten this without changing
the rule model.

## Decision pipeline

The evaluator runs only in the private control-plane process, initially every
60 seconds.

1. **Acquire a singleton lease.** A database lease with expiry prevents two
   private controllers from applying the same rule. Every action also has a
   unique idempotency key `(rule, evaluation, ordinal)`.
2. **Snapshot inputs.** Persist queue depth/units, drain ETA, throughput window,
   current slots, active provider deployments, in-flight deployments, active
   jobs, provider registry state, credits, budgets, and rule version.
3. **Validate signal quality.** Autonomous scale-out is blocked when throughput
   or serving-capacity history is insufficient unless the rule explicitly has
   a bounded bootstrap prior. Stale data is discounted conservatively.
4. **Apply sustain/cooldown gates.** Track threshold crossings durably across
   evaluations. Do not act while the rule, provider, or candidate shape is in a
   retry window.
5. **Compute target slots.** Use generic workload units and account for useful
   work completed by current capacity during startup. Subtract active and
   in-flight compatible slots before requesting additions.
6. **Collect fresh candidates.** Query each allowed provider through the current
   route/resource and pool cost constraints. Known-zero availability is
   excluded; unknown availability is allowed only by policy.
7. **Apply provider gates.** Exclude disabled/unhealthy providers, stale offers,
   insufficient/unknown required credit, exceeded budgets, and candidate shapes
   in startup-failure cooldown.
8. **Rank expected incremental cost-to-target.** For each candidate estimate:
   - route/provider/GPU throughput from conservative recent successful history;
   - startup p50/p95 and readiness success;
   - interruption/failure penalty;
   - ETA after the candidate becomes ready;
   - runtime plus startup cost required to reach the target ETA.

   For remaining units `U`, current throughput `C`, candidate throughput `c`,
   and startup hours `S`, the first-order ETA is:

   `S + max(0, U - C*S) / (C + c)`

   Ranking uses expected total incremental drain cost, then completion time,
   reliability, fresh availability, and stable provider/offer identifiers. It
   does not simply choose the lowest hourly rate.
9. **Write a proposal.** In `report-only`, stop here. The dashboard shows the
   selected candidate, alternatives, exclusions, expected ETA improvement, and
   cost.
10. **Revalidate and reserve.** In `apply`, atomically reserve budget and
    re-fetch the selected offer, availability, and credit immediately before
    calling `DeploymentService`.
11. **Apply one bounded action.** Start with at most one worker per evaluation.
    Record origin `autoscaler:<rule_id>` in the normal deployment ledger.
12. **Observe readiness.** Provider creation, allocation, start, DGPU heartbeat,
    and first completion are distinct states. In-flight workers count toward the
    target but not serving throughput.
13. **Fallback carefully.** On confirmed absence after a failed create, retire
    the credential, cool down that candidate, and optionally try the next ranked
    provider within the same action/retry cap. Never fallback after an ambiguous
    failure until inventory proves absence.

## Scale-down pipeline

Scale-down is a separate decision, not the inverse of scale-up.

1. Require ETA below the lower threshold for the full sustain interval.
2. Select only rule-owned deployments, preferring the highest expected future
   cost and lowest useful/reliable throughput.
3. Mark one agent draining so it cannot claim new work.
4. Wait for its active lease to finish; never interrupt a successful artifact
   transfer merely to save minutes of billing.
5. Destroy through the provider driver and require authoritative absence.
6. Disable the agent token only after absence is proven.
7. Retain deployment, attempts, jobs, artifacts, evaluations, and actions as
   historical audit data.
8. If provider destruction is ambiguous, keep the credential disabled from new
   claims only when safe, alert, and continue inventory verification; do not
   mark the deployment destroyed prematurely.

## Data model

Extend the existing SQLite schema rather than adding a second fleet database.

### `scaling_rules`

- id, name, pool_id, mode, enabled, version;
- trigger/scale-up/provider/budget/scale-down JSON validated by strict models;
- created/updated actor and timestamps;
- last evaluation/action timestamps and global kill-switch generation.

### `scaling_evaluations`

Append-only records containing rule version, complete bounded input snapshot,
threshold state, desired/current/in-flight slots, decision, blocked reason,
selected candidate, alternatives, projected ETA/cost, and timestamp.

### `scaling_actions`

- evaluation/rule/idempotency key;
- action (`deploy`, `drain`, `destroy`), state, actor/origin;
- deployment and token IDs;
- reservation, provider request/result, trace ID, retry/fallback lineage;
- created/started/finished timestamps.

### Extend `deployments`

Persist offer ID, GPU class/model, priority/region, availability at selection,
price source, rule/action origin, provider create response, startup deadline,
first heartbeat, first completion, and terminal failure classification. Manual
and automatic deployments use the same ledger.

### Market, startup, and performance observations

Persist route-scoped observations:

- provider offer price and availability over time;
- credit state and freshness (never secrets);
- create-to-provider-running and create-to-heartbeat durations;
- interruptions/startup failures by provider/GPU/priority;
- successful units per GPU busy hour and failure rates.

Use bounded retention/aggregation for market samples while keeping deployment
and action audit records permanently.

### Budget reservations

A small reservation ledger prevents concurrent rules from each spending the
same credit or hourly budget. Reservations expire if no provider action starts
and reconcile to actual deployments after creation.

## API and dashboard

### GPUs tab

Add a **Deploy GPU** button above current fleet inventory. It opens the existing
reviewed pool/provider/offer modal, with:

- workload/route/pool;
- compatible live offers across providers;
- credit state and freshness;
- availability and reliability;
- hourly, 24-hour, 730-hour, expected startup, and expected drain cost;
- explicit count, token lifetime, total cap, and confirmation.

The action calls the existing deployment service. The Admin tab may retain a
compact deployment ledger, but operators should not need to leave **GPUs** for
ordinary fleet work.

### Automation panel

On the GPUs tab, administrators see:

- rules with mode, route, trigger, current signal, sustain progress, bounds, and
  last decision;
- **Create rule**, **Edit**, **Enable report-only**, **Promote to apply**,
  **Pause**, and **Delete**;
- a simulation preview against current data;
- current/in-flight/desired capacity and projected ETA/cost;
- candidate table with explicit exclusion reasons;
- global kill switch;
- append-only evaluation/action history.

Promoting to `apply` requires a confirmation summarizing the maximum hourly and
daily exposure. Editing spend/provider/capacity bounds increments the rule
version and requires reconfirmation.

### Proposed private endpoints

- `GET /v1/control/scaling-rules`
- `POST /v1/control/scaling-rules`
- `PUT /v1/control/scaling-rules/{id}`
- `DELETE /v1/control/scaling-rules/{id}`
- `POST /v1/control/scaling-rules/{id}/simulate`
- `POST /v1/control/scaling-rules/{id}/evaluate`
- `GET /v1/control/scaling-evaluations`
- `GET /v1/control/scaling-actions`
- `POST /v1/control/automation/pause`

These remain private/controller routes and must not be mounted on public worker
ingress.

## Implementation structure

Avoid a new monolith:

- `autoscaler.py`: pure target-slot and candidate economics functions;
- `scaling_policy.py`: strict rule models and validation;
- `fleet_economics.py`: observation aggregation and conservative priors;
- `scaling_engine.py`: evaluation state machine, locking, reservations, and
  action orchestration;
- `deploy.py`: shared manual/automatic apply and verified teardown primitives;
- `store.py`: rule/evaluation/action/observation persistence;
- provider drivers: inventory, offers, credit, create/destroy only;
- `dashboard.html`: GPUs-tab manual deployment, rules, simulation, and history.

Pure decision functions should receive immutable snapshots and return a plan;
provider calls occur only in the action executor. This makes simulations and
unit tests identical to live decision logic.

## Delivery phases

### Phase 1 — Manual deployment on GPUs tab

- Move/reuse the current capacity and deploy flow on **GPUs**.
- Preserve server-side offer revalidation and confirmation.
- Enrich deployment attempts with selected-offer and startup audit fields.
- Add provider create ambiguity inventory checks.

Acceptance: a controller/admin can deploy from GPUs; no new arbitrary inputs;
all existing deploy/teardown tests pass.

### Phase 2 — Rules and report-only engine

- Add schemas, migrations, CRUD, singleton evaluator, simulations, and durable
  evaluations.
- Implement ETA threshold, sustain, hysteresis, in-flight accounting, and
  blocked reasons.
- No provider mutations regardless of rule configuration during initial burn-in.

Acceptance: replayed queue snapshots produce deterministic decisions; dashboard
explains every non-action.

### Phase 3 — Provider economics and circuit breakers

- Persist market, credit, startup, interruption, and route-throughput history.
- Rank Vast and Salad by expected cost-to-target.
- Add freshness requirements, conservative priors, failure cooldowns, retry
  windows, and global/per-rule budget reservations.

Acceptance: candidate rankings are reproducible from stored observations and
never select known-zero, over-budget, disabled, stale, or credit-blocked offers.

### Phase 4 — Bounded automatic scale-up

- Add global apply kill switch and admin promotion workflow.
- Enable one rule for `translate-text-batch`, one addition per evaluation, and
  a low fleet/hourly cap.
- Revalidate immediately before apply; verify ambiguous failures by inventory.
- Require provider heartbeat readiness and run a bounded canary if configured.

Acceptance: an ETA-over-threshold fixture deploys exactly one idempotent worker;
repeated evaluations and controller restarts cannot duplicate it.

### Phase 5 — Graceful automatic scale-down

- Add durable drain state and rule-owned worker selection.
- Wait for active jobs, verify provider absence, then retire credentials.
- Add stuck-drain alerts and operator override.

Acceptance: scale-down cannot interrupt an active job and cannot mark destroyed
before authoritative absence.

### Phase 6 — Multi-rule hardening

- Shared global budgets, fairness between routes, action concurrency limits,
  PostgreSQL/leader-election preparation, alerts, Prometheus metrics, and
  long-window economic reports.
- Optional explicitly bounded hedged startup can be considered only after
  measured interruption/startup economics justify its temporary double cost.

## Verification plan

Unit tests:

- ETA target math with zero/current/in-flight capacity and startup delay;
- hysteresis and sustain timers;
- unknown/stale throughput blocking and conservative priors;
- cross-provider cost-to-target ranking;
- Vast known-credit and Salad unknown-credit policy;
- hourly/drain/daily/global budget reservations;
- candidate cooldown and retry windows;
- idempotent actions and expired evaluator leases;
- ambiguous create response with resource present versus absent;
- graceful drain and teardown ordering.

Integration tests with fake drivers:

- report-only never mutates;
- apply deploys one selected provider and records full audit;
- primary failure falls back only after confirmed absence;
- known-zero/insufficient-credit providers are skipped;
- in-flight capacity suppresses duplicate deploy;
- controller restart resumes, rather than repeats, an action;
- scale-down waits for completion and verifies absence before token retirement.

Real-provider canary (separate explicit authorization):

1. Verify public ingress, provider inventory, offer, credit policy, and cap.
2. Enable one report-only translation rule and compare decisions for a burn-in
   window.
3. Promote to one-worker apply with a hard hourly/daily bound.
4. Submit a bounded batch, verify ordered authenticated artifact output.
5. Exercise drain, destroy, inventory absence, token retirement, and retained
   history.
6. Confirm no unexpected resources in both Vast and Salad inventories.

## Initial recommendation

Implement Phase 1 and Phase 2 first. Run the two-hour translation rule in
`report-only` until DGPU has enough route/GPU startup and throughput history to
rank expected cost reliably. Vast automatic selection should require known
credit. Salad should remain eligible only under an explicit unknown-credit
allowance and hard spend caps. Promote one rule to apply only after simulations
match operator decisions and the external public TLS route is verified from a
non-tailnet network.
