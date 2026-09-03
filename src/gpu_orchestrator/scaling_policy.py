"""Strict Pydantic contracts for autonomous fleet scaling rules.

Rules are durable database records validated before persistence. Every
sub-document uses ``extra="forbid"`` so unexpected fields are rejected.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ProviderName = Literal["vast", "salad", "local"]


class TriggerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str = Field(min_length=1)
    above_hours: float = Field(gt=0, allow_inf_nan=False)
    sustain_seconds: int = Field(ge=0)
    require_observed_throughput: bool


class ScaleUpSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_eta_hours: float = Field(gt=0, allow_inf_nan=False)
    max_provider_slots: int = Field(ge=1)
    max_additions_per_evaluation: int = Field(ge=1)
    cooldown_seconds: int = Field(ge=0)
    startup_deadline_seconds: int = Field(ge=0)

    @model_validator(mode="after")
    def max_additions_fit_in_max_slots(self) -> "ScaleUpSpec":
        if self.max_additions_per_evaluation > self.max_provider_slots:
            raise ValueError(
                "max_additions_per_evaluation cannot exceed max_provider_slots"
            )
        return self


class ScaleDownSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    below_eta_hours: float = Field(gt=0, allow_inf_nan=False)
    sustain_seconds: int = Field(ge=0)
    idle_seconds: int = Field(ge=0)
    min_provider_slots: int = Field(ge=0)


class ProviderLimitsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    require_known_credit: bool
    max_hourly_cost_usd: float = Field(ge=0, allow_inf_nan=False)


class ProvidersSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed: list[ProviderName]
    require_known_availability: bool
    vast: ProviderLimitsSpec | None = None
    salad: ProviderLimitsSpec | None = None

    @model_validator(mode="after")
    def at_least_one_allowed_provider(self) -> "ProvidersSpec":
        if not self.allowed:
            raise ValueError("at least one provider must be allowed")
        if "vast" in self.allowed and self.vast is None:
            raise ValueError("vast limits are required when vast is allowed")
        if "salad" in self.allowed and self.salad is None:
            raise ValueError("salad limits are required when salad is allowed")
        return self


class BudgetsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_rule_hourly_usd: float = Field(ge=0, allow_inf_nan=False)
    max_projected_drain_usd: float = Field(ge=0, allow_inf_nan=False)
    max_daily_actions: int = Field(ge=0)


class ScalingRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    pool_id: str = Field(min_length=1)
    mode: Literal["off", "report-only", "apply"]
    enabled: bool
    version: int = Field(ge=1)
    trigger: TriggerSpec
    scale_up: ScaleUpSpec
    providers: ProvidersSpec
    budgets: BudgetsSpec
    scale_down: ScaleDownSpec
    created_at: str
    updated_at: str
    actor: str
    last_evaluation_at: str | None = None
    last_action_at: str | None = None
    threshold_state: str | None = None
    threshold_since: str | None = None
    deleted_at: str | None = None


def load_rule(document: dict[str, Any]) -> ScalingRule:
    """Validate a raw rule document and return a ``ScalingRule`` instance."""
    return ScalingRule.model_validate(document)


def rule_to_document(rule: ScalingRule) -> dict[str, Any]:
    """Serialize a validated rule to a JSON-safe dictionary."""
    return rule.model_dump(mode="json")


__all__ = [
    "BudgetsSpec",
    "ProviderLimitsSpec",
    "ProvidersSpec",
    "ScaleDownSpec",
    "ScaleUpSpec",
    "ScalingRule",
    "TriggerSpec",
    "load_rule",
    "rule_to_document",
]
