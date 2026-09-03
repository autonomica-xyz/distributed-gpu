"""Observation aggregation and conservative fleet economics.

Pure functions for ranking GPU provider candidates by expected cost-to-drain
using recent market, startup, and throughput observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


__all__ = [
    "MarketObservation",
    "ThroughputObservation",
    "StartupObservation",
    "Candidate",
    "RankedCandidate",
    "conservative_throughput",
    "estimate_eta_after_startup",
    "expected_drain_cost",
    "rank_by_cost_to_target",
]


@dataclass(frozen=True)
class MarketObservation:
    """Provider offer snapshot: price, availability, and reliability."""

    provider: str
    offer_id: str
    gpu_model: str
    hourly_cost_usd: float
    available_count: int
    reliability: float
    observed_at: float


@dataclass(frozen=True)
class ThroughputObservation:
    """Measured route throughput for a provider/offer/GPU."""

    provider: str
    offer_id: str
    gpu_model: str
    route: str
    units_per_gpu_hour: float
    window_hours: float
    observed_at: float


@dataclass(frozen=True)
class StartupObservation:
    """Create-to-heartbeat timing and success/failure sample."""

    provider: str
    offer_id: str
    gpu_model: str
    create_to_heartbeat_seconds: float
    success: bool
    observed_at: float


@dataclass(frozen=True)
class Candidate:
    """A provider offer with enough data to rank by expected drain cost."""

    provider: str
    offer_id: str
    gpu_model: str
    hourly_cost_usd: float
    startup_hours: float
    units_per_gpu_hour: float
    reliability: float = 1.0


@dataclass(frozen=True)
class RankedCandidate:
    """A candidate annotated with cost-to-target economics."""

    candidate: Any
    completion_hours: float
    total_cost_usd: float
    score: float


def conservative_throughput(
    observations: list[ThroughputObservation],
    default: float,
    staleness_discount: float,
) -> float:
    """Return the most recent throughput, damped toward a default prior.

    ``staleness_discount`` is a [0, 1] factor: 0 means fresh (return the most
    recent observation), 1 means fully discounted (return the default). Values
    in between linearly blend from the most recent observation toward the
    default.
    """
    default = float(default)
    if not observations:
        return default

    most_recent = max(observations, key=lambda obs: obs.observed_at)
    value = float(most_recent.units_per_gpu_hour)
    if not isfinite(value):
        return default

    discount = max(0.0, min(1.0, float(staleness_discount)))
    if discount == 0.0:
        return value
    return value + (default - value) * discount


def estimate_eta_after_startup(
    units: float,
    current_slots: int,
    current_units_per_hour: float,
    candidate_units_per_hour: float,
    startup_hours: float,
) -> float:
    """First-order ETA: S + max(0, U - C*S) / (C + c)."""
    U = max(0.0, float(units))
    S = max(0.0, float(startup_hours))
    C = max(0, int(current_slots)) * max(0.0, float(current_units_per_hour))
    c = max(0.0, float(candidate_units_per_hour))
    total_throughput = C + c
    if total_throughput <= 0.0:
        return float("inf")
    return S + max(0.0, U - C * S) / total_throughput


def expected_drain_cost(
    units: float,
    startup_hours: float,
    units_per_gpu_hour: float,
    hourly_cost_usd: float,
    reliability: float,
    *,
    current_slots: int = 0,
    current_units_per_gpu_hour: float | None = None,
) -> tuple[float, float]:
    """Return (completion_hours, total_cost) for draining ``units``.

    Reliability is applied as a conservative multiplier, e.g. 90% reliability
    inflates both time and cost by 1/0.9.
    """
    U = max(0.0, float(units))
    S = max(0.0, float(startup_hours))
    uph = float(units_per_gpu_hour)
    cost = float(hourly_cost_usd)
    rel = float(reliability)

    if uph <= 0.0 or not all(
        isfinite(v) for v in (uph, cost, rel, S, U)
    ):
        return float("inf"), float("inf")

    rel = min(1.0, max(0.05, rel))
    completion = estimate_eta_after_startup(
        U, current_slots, current_units_per_gpu_hour if current_units_per_gpu_hour is not None else uph,
        uph, S,
    )
    # This is incremental spend: existing fleet cost is sunk for this choice.
    total_cost = completion * cost / rel
    return completion, total_cost


def rank_by_cost_to_target(
    candidates: list,
    units_to_drain: float,
    *,
    current_slots: int = 0,
    current_units_per_gpu_hour: float | None = None,
) -> list[RankedCandidate]:
    """Rank candidates on expected total incremental cost-to-drain.

    Mirrors the tie-breaker ordering in ``autoscaler.py``: lower total drain
    cost wins, then shorter completion time, then higher reliability, and
    finally stable provider/offer identifiers.
    """
    units = max(0.0, float(units_to_drain))
    ranked: list[RankedCandidate] = []

    for candidate in candidates:
        values = (
            candidate.hourly_cost_usd,
            candidate.startup_hours,
            candidate.units_per_gpu_hour,
            candidate.reliability,
            current_units_per_gpu_hour if current_units_per_gpu_hour is not None else candidate.units_per_gpu_hour,
        )
        if (
            candidate.hourly_cost_usd < 0
            or candidate.startup_hours < 0
            or candidate.units_per_gpu_hour <= 0
            or not all(isfinite(v) for v in values)
        ):
            continue

        completion, total_cost = expected_drain_cost(
            units,
            candidate.startup_hours,
            candidate.units_per_gpu_hour,
            candidate.hourly_cost_usd,
            candidate.reliability,
            current_slots=current_slots,
            current_units_per_gpu_hour=current_units_per_gpu_hour,
        )

        ranked.append(
            RankedCandidate(
                candidate=candidate,
                completion_hours=completion,
                total_cost_usd=total_cost,
                score=total_cost,
            )
        )

    return sorted(
        ranked,
        key=lambda item: (
            item.score,
            item.completion_hours,
            -item.candidate.reliability,
            item.candidate.provider,
            item.candidate.offer_id,
        ),
    )
