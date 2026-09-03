from gpu_orchestrator.autoscaler import (
    ProviderCandidate,
    ScalingInputs,
    estimate_target_slots,
    rank_candidates,
)


def test_generic_units_drive_target_capacity():
    decision = estimate_target_slots(
        ScalingInputs(
            queued_units=600,
            current_slots=1,
            units_per_gpu_hour=100,
            target_completion_hours=2,
            startup_hours=0.5,
            max_slots=10,
            hourly_cost_usd=0.25,
        )
    )
    assert decision.target_slots == 4
    assert decision.requested_slots == 3
    assert decision.estimated_eta_hours_target == 1.875


def test_unknown_required_credit_blocks_scale_out():
    decision = estimate_target_slots(
        ScalingInputs(
            queued_units=100,
            current_slots=0,
            units_per_gpu_hour=10,
            require_known_credit=True,
            available_credit_usd=None,
        )
    )
    assert decision.target_slots == 0
    assert decision.blocked_reason == "provider_credit_unknown"


def test_total_drain_cost_beats_low_hourly_price():
    ranked = rank_candidates(
        [
            ProviderCandidate("slow", "1", "old", 0.10, 0.5, 1),
            ProviderCandidate("fast", "2", "new", 0.30, 0.1, 10),
        ],
        units_to_drain=10,
    )
    assert ranked[0].candidate.provider == "fast"
    assert ranked[0].estimated_total_cost_usd < ranked[1].estimated_total_cost_usd

