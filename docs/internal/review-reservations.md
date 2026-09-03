# Codex review reservations and budget-reservation design

This file records the concerns ("reservations") raised by the `codex exec review`
passes during the autonomous fleet-management implementation, and the
design/behavior of the `budget_reservations` subsystem that was added to address
the spend-safety findings.

## Latest codex review findings (most recent pass)

The final focused review reported four high-severity findings:

1. **Spend limits are projections, not enforced caps.**
   - `budget_reservations` expire at the startup deadline, but a worker can
     remain billable long after the reservation expires if scale-down is
     disabled, paused, or slow. The rule's `max_projected_drain_usd` can
     therefore be materially exceeded.
   - *Files:* `src/gpu_orchestrator/scaling_engine.py` (apply paths),
     `src/gpu_orchestrator/store.py` (`reserve_budget`).

2. **Provider disappearance leaves the worker credential active.**
   - `DeploymentService.reconcile()` marks a missing provider resource `absent`
     but previously did not disable the worker token, allowing a compromised or
     orphaned worker to keep claiming work until the token expiry.
   - *Fix applied:* `reconcile()` now calls `store.disable_token()` when marking
     a deployment `absent`.

3. **Scaling lease not safely fenced across controllers.**
   - The default lease owner was `controller`, so a stale/stuck evaluator could
     release a successor's lease (same owner string) and allow concurrent
     billable mutations.
   - *Fix applied:* each evaluator now generates a unique owner
     (`autoscaler-{pid}-{random}`), and `acquire_scaling_lease` refuses to
     overwrite an unexpired lease held by a different owner.

4. **Draining workers are counted as usable capacity.**
   - `FleetPlanner._current_slots()` counted `draining` agents as ready slots,
     which could suppress needed scale-out while a worker is being torn down.
   - *Fix applied:* `_current_slots()` now counts only agents in state `ready`.

## Earlier codex findings that were addressed

Over the review loop, the following high-severity items were also raised and
fixed:

- **Ambiguous-create recovery can launch duplicate paid workers.**
  - Fixed by recording `preparing` deployments before `driver.launch`, treating
    `preparing` as in-flight capacity, and resolving multi-worker actions from
    provider inventory.

- **Non-finite provider values bypass autonomous cost controls.**
  - Fixed by adding `allow_inf_nan=False` to policy/model float fields and
    rejecting `NaN`/`inf` in Vast/Salad float parsing.

- **`require_known_availability` bypassed for Vast.**
  - Fixed by setting Vast `ProviderOffer.available_count = None` when no
    availability preflight is performed.

- **Drain actions not recoverable/idempotent.**
  - Fixed by leaving the action `pending` on transient conflicts (active jobs,
    provider unavailability) and resuming via `_recover_pending_drain()`.

- **Vast contract-id mismatch in cleanup.**
  - Fixed by resolving the actual `resource_id` from inventory by matching
    `ProviderResource.worker_id` before calling `destroy_and_verify_absent`.

- **`min_provider_slots` enforced against all pool slots.**
  - Fixed by counting only rule-owned `active`/`draining`/`stale` deployments.

- **Concurrent rules for the same pool can both scale out.**
  - Fixed by enforcing one enabled `apply` rule per pool in
    `create_scaling_rule` / `update_scaling_rule`.

- **Controller token can create an `apply` rule.**
  - Fixed by requiring `admin_principal` for `POST /v1/control/scaling-rules`.

- **Stale/expired autonomous workers never eligible for teardown.**
  - Fixed by adding `stale` to the rule-owned deployments considered for
    scale-down.

- **Global kill switch has a launch race.**
  - Fixed by adding `automation_state.generation`, gating `reserve_budget` on
    the generation, and re-checking `paused` immediately before provider
    mutation.

- **Teardown can race a stale worker claiming work.**
  - Fixed by rejecting stale-heartbeat slots in `store.claim_next()`.

## Budget-reservation design

The `budget_reservations` table was introduced to make concurrent, credit-aware
scale-up safer. It is not a full spend ledger; it reserves the *projected drain
cost* of an action before provider mutation and reconciles on success/failure.

### Schema

```text
CREATE TABLE budget_reservations (
    id          TEXT PRIMARY KEY,
    rule_id     TEXT,
    provider    TEXT,          -- added to track per-provider credit exposure
    amount_usd  REAL NOT NULL,
    reserved_at TEXT NOT NULL,
    expires_at  TEXT NOT NULL,
    reconciled_at TEXT         -- set when the reservation is released or
                                 -- the associated deployment is finalized
);
```

Indexes exist on `rule_id` and `provider`.

### Store methods

- `reserve_budget(rule_id, amount_usd, ttl_seconds, provider=None,
  automation_generation=None)`
  - Inserts a new reservation.
  - If `automation_generation` is supplied, it checks the `automation_state`
    table inside the same transaction and raises `ConflictError` if
    `paused != 0` or `generation != automation_generation`.

- `sum_unexpired_reservations(rule_id=None, provider=None)`
  - Returns the total un-reconciled, un-expired reservation amount for a rule
    and/or a provider.

- `release_budget_reservation(reservation_id)`
  - Marks a reservation as reconciled (e.g. after a deploy or a failure).

### Enforcement in `ScalingEngine.apply_scale_up`

1. `automation_state()` is read before the reservation.
2. If the switch is paused, the evaluation is blocked.
3. `sum_unexpired_reservations(rule_id=...)` is compared to
   `rule.budgets.max_projected_drain_usd`.
4. `sum_unexpired_reservations(provider=...)` is compared to the provider's
   known `account_credit()` when `require_known_credit` is true.
5. `reserve_budget(..., automation_generation=...)` only succeeds if the
   `automation_state` has not changed between the read and the insert.
6. A final `paused` check happens immediately before `DeploymentService.deploy()`.

### Remaining design gap

The reservation is still a *projected* cap. It does **not** enforce a hard
lifetime or total billed cost on a running worker. If `scale_down` is disabled
or cannot keep up, a worker can outlive its reserved projection. Closing that
gap requires either a background lifetime/total-cost reaper or a hard
per-deployment token/contract expiry enforced by the provider.
