"""Workload-unit autoscaling and provider cost-to-drain ranking."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite


@dataclass(frozen=True)
class ScalingInputs:
    queued_units: float
    running_units: float = 0.0
    current_slots: int = 0
    units_per_gpu_hour: float = 1.0
    target_completion_hours: float = 6.0
    startup_hours: float = 0.25
    min_slots: int = 0
    max_slots: int = 10
    hourly_cost_usd: float = 0.0
    hourly_budget_usd: float | None = None
    available_credit_usd: float | None = None
    require_known_credit: bool = False


@dataclass(frozen=True)
class ScalingDecision:
    target_slots: int
    requested_slots: int
    estimated_eta_hours_current: float | None
    estimated_eta_hours_target: float | None
    projected_hourly_cost_usd: float
    projected_drain_cost_usd: float | None
    blocked_reason: str | None
    reason: str


@dataclass(frozen=True)
class ProviderCandidate:
    provider: str
    offer_id: str
    gpu_model: str
    hourly_cost_usd: float
    startup_hours: float
    units_per_gpu_hour: float
    reliability: float = 1.0
    available_count: int | None = None


@dataclass(frozen=True)
class RankedCandidate:
    candidate: ProviderCandidate
    estimated_completion_hours: float
    estimated_total_cost_usd: float
    score: float


def estimate_target_slots(inputs: ScalingInputs) -> ScalingDecision:
    queued = max(0.0, float(inputs.queued_units))
    running = max(0.0, float(inputs.running_units))
    remaining = queued + running
    current = max(0, int(inputs.current_slots))
    minimum = max(0, int(inputs.min_slots))
    maximum = max(minimum, int(inputs.max_slots))
    throughput = max(0.0, float(inputs.units_per_gpu_hour))

    current_eta = remaining / (current * throughput) if current > 0 and throughput > 0 else None
    if remaining <= 0:
        target = minimum
        return _decision(inputs, current, target, 0.0, 0.0, None, "queue is empty")

    if throughput <= 0:
        target = max(minimum, current)
        return _decision(
            inputs,
            current,
            target,
            current_eta,
            None,
            "throughput_unknown",
            "no usable throughput estimate; autonomous scale-out is blocked",
        )

    useful_hours = max(0.05, inputs.target_completion_hours - max(0.0, inputs.startup_hours))
    work_after_startup = max(0.0, remaining - current * throughput * max(0.0, inputs.startup_hours))
    required_capacity = work_after_startup / useful_hours
    target = max(minimum, current, ceil(required_capacity / throughput))
    target = min(maximum, target)

    blocked: str | None = None
    if inputs.require_known_credit and inputs.available_credit_usd is None and target > current:
        target = current
        blocked = "provider_credit_unknown"
    if inputs.hourly_cost_usd > 0 and inputs.hourly_budget_usd is not None:
        affordable = max(0, int(inputs.hourly_budget_usd // inputs.hourly_cost_usd))
        if target > affordable:
            target = max(minimum, min(current, affordable))
            blocked = blocked or "hourly_budget_exceeded"

    target_eta: float | None
    if target <= 0:
        target_eta = None
    elif target <= current:
        target_eta = remaining / (target * throughput)
    else:
        during_startup = current * throughput * max(0.0, inputs.startup_hours)
        target_eta = max(0.0, inputs.startup_hours) + max(0.0, remaining - during_startup) / (
            target * throughput
        )

    projected_drain = (
        target_eta * target * max(0.0, inputs.hourly_cost_usd)
        if target_eta is not None
        else None
    )
    if (
        target > current
        and inputs.available_credit_usd is not None
        and projected_drain is not None
        and projected_drain > inputs.available_credit_usd
    ):
        target = current
        target_eta = current_eta
        blocked = "provider_credit_insufficient"

    reason = (
        "deployment blocked by spend policy"
        if blocked
        else "capacity selected to meet the target completion window"
    )
    return _decision(inputs, current, target, current_eta, target_eta, blocked, reason)


def _decision(
    inputs: ScalingInputs,
    current: int,
    target: int,
    current_eta: float | None,
    target_eta: float | None,
    blocked: str | None,
    reason: str,
) -> ScalingDecision:
    hourly_cost = target * max(0.0, inputs.hourly_cost_usd)
    drain_cost = target_eta * hourly_cost if target_eta is not None else None
    return ScalingDecision(
        target_slots=target,
        requested_slots=target - current,
        estimated_eta_hours_current=current_eta,
        estimated_eta_hours_target=target_eta,
        projected_hourly_cost_usd=hourly_cost,
        projected_drain_cost_usd=drain_cost,
        blocked_reason=blocked,
        reason=reason,
    )


def rank_candidates(
    candidates: list[ProviderCandidate],
    *,
    units_to_drain: float,
) -> list[RankedCandidate]:
    """Rank on total expected drain cost, not headline hourly price.

    Reliability is a conservative penalty: a 90% reliable offer is treated as
    needing roughly 1/0.9 times the nominal runtime and cost.
    """

    ranked: list[RankedCandidate] = []
    units = max(0.0, float(units_to_drain))
    for candidate in candidates:
        if (
            candidate.hourly_cost_usd < 0
            or candidate.startup_hours < 0
            or candidate.units_per_gpu_hour <= 0
            or not all(
                isfinite(value)
                for value in (
                    candidate.hourly_cost_usd,
                    candidate.startup_hours,
                    candidate.units_per_gpu_hour,
                    candidate.reliability,
                )
            )
        ):
            continue
        reliability = min(1.0, max(0.05, candidate.reliability))
        runtime = candidate.startup_hours + units / candidate.units_per_gpu_hour
        adjusted_runtime = runtime / reliability
        total_cost = adjusted_runtime * candidate.hourly_cost_usd
        ranked.append(
            RankedCandidate(
                candidate=candidate,
                estimated_completion_hours=adjusted_runtime,
                estimated_total_cost_usd=total_cost,
                score=total_cost,
            )
        )
    return sorted(
        ranked,
        key=lambda item: (
            item.score,
            item.estimated_completion_hours,
            -item.candidate.reliability,
            item.candidate.provider,
            item.candidate.offer_id,
        ),
    )
