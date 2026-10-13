# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local ATIF domain models for Intake ingest.

These models mirror the small Pydantic schema surface Intake needs from
``nvidia-nat-atif`` without adding a service dependency on NAT packages.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from nhx.intake.spans.ingest.evaluation_context import EvaluationContext
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import TypeAliasType

ATIF_VERSION = "ATIF-v1.7"
AtifSchemaVersion = Literal[
    "ATIF-v1.0",
    "ATIF-v1.1",
    "ATIF-v1.2",
    "ATIF-v1.3",
    "ATIF-v1.4",
    "ATIF-v1.5",
    "ATIF-v1.6",
    "ATIF-v1.7",
]


class AtifImageSource(BaseModel):
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    path: str

    model_config = ConfigDict(extra="forbid")


AtifTimestamp = Annotated[str, Field(json_schema_extra={"format": "date-time"})]


class AtifContentPartText(BaseModel):
    type: Literal["text"]
    text: str

    model_config = ConfigDict(extra="forbid")


class AtifContentPartImage(BaseModel):
    type: Literal["image"]
    source: AtifImageSource

    model_config = ConfigDict(extra="forbid")


AtifContentPart = TypeAliasType(
    "AtifContentPart",
    Annotated[AtifContentPartText | AtifContentPartImage, Field(discriminator="type")],
)


class AtifAgent(BaseModel):
    name: str
    version: str
    model_name: str | None = None
    tool_definitions: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")


class AtifToolCall(BaseModel):
    tool_call_id: str
    function_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # ATIF v1.7 makes per-tool-call metadata first-class: NAT publishers write
    # ancestry / invocation timing here, and the spec requires consumers to
    # tolerate absent and unknown keys (nvidia_nat_atif ``AtifToolCallExtra``).
    extra: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")


class AtifSubagentTrajectoryRef(BaseModel):
    trajectory_id: str | None = None
    trajectory_path: str | None = None
    session_id: str | None = None
    extra: dict[str, Any] | None = None

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "anyOf": [
                {"required": ["trajectory_id"]},
                {"required": ["trajectory_path"]},
                {"required": ["session_id"]},
            ],
        },
    )

    @model_validator(mode="after")
    def validate_identifier(self) -> AtifSubagentTrajectoryRef:
        """Require at least one supported subagent identifier."""
        if self.trajectory_id is None and self.trajectory_path is None and self.session_id is None:
            raise ValueError(
                "SubagentTrajectoryRef MUST set at least one of trajectory_id, trajectory_path, or session_id"
            )
        return self


class AtifObservationResult(BaseModel):
    source_call_id: str | None = None
    content: str | list[AtifContentPart] | None = None
    subagent_trajectory_ref: list[AtifSubagentTrajectoryRef] | None = None
    extra: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")


class AtifObservation(BaseModel):
    results: list[AtifObservationResult] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class AtifMetrics(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    extra: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")


class AtifFinalMetrics(BaseModel):
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_cached_tokens: int | None = None
    total_cost_usd: float | None = None
    total_steps: int | None = Field(default=None, ge=0)
    extra: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")


class AtifStepBase(BaseModel):
    step_id: int = Field(ge=1)
    timestamp: AtifTimestamp | None = Field(default=None, json_schema_extra={"format": "date-time"})
    message: str | list[AtifContentPart] = ""
    is_copied_context: bool | None = None
    extra: dict[str, Any] | None = None
    llm_call_count: int | None = Field(default=None, ge=0)
    observation: AtifObservation | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: str | None) -> str | None:
        """Validate an optional ISO 8601 step timestamp."""
        if value is not None:
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(f"Invalid ISO 8601 timestamp: {error}") from error
        return value


class AtifStepSystem(AtifStepBase):
    source: Literal["system"]


class AtifStepUser(AtifStepBase):
    source: Literal["user"]


class AtifStepAgent(AtifStepBase):
    source: Literal["agent"]
    model_name: str | None = None
    reasoning_effort: str | float | None = None
    reasoning_content: str | None = None
    tool_calls: list[AtifToolCall] | None = None
    metrics: AtifMetrics | None = None


AtifStep = TypeAliasType(
    "AtifStep",
    Annotated[AtifStepSystem | AtifStepUser | AtifStepAgent, Field(discriminator="source")],
)


def validate_atif_step_ids(steps: list[AtifStep]) -> None:
    """Require one-based sequential step identifiers."""
    # Keep this aligned with NAT's current ATIF model until the format
    # explicitly allows compacted or branched trajectories with gaps.
    for index, step in enumerate(steps):
        expected_step_id = index + 1
        if step.step_id != expected_step_id:
            raise ValueError(
                f"steps[{index}].step_id: expected {expected_step_id} (sequential from 1), got {step.step_id}"
            )


def validate_atif_tool_call_references(steps: list[AtifStep]) -> None:
    """Require unique calls and resolvable observation call references."""
    for step in steps:
        tool_call_ids: set[str] = set()
        tool_calls = step.tool_calls if isinstance(step, AtifStepAgent) else None
        for tool_call in tool_calls or []:
            if tool_call.tool_call_id in tool_call_ids:
                raise ValueError(f"Duplicate tool_call_id '{tool_call.tool_call_id}' in step {step.step_id}")
            tool_call_ids.add(tool_call.tool_call_id)
        if step.observation is None:
            continue
        for result in step.observation.results:
            if result.source_call_id is not None and result.source_call_id not in tool_call_ids:
                raise ValueError(
                    f"Observation result references source_call_id '{result.source_call_id}' "
                    f"which is not found in step {step.step_id}'s tool_calls"
                )


def validate_atif_v17_subagent_ref_resolution_keys(steps: list[AtifStep]) -> None:
    """Require v1.7 subagent references to include a resolvable key."""
    for step in steps:
        if step.observation is None:
            continue
        for result in step.observation.results:
            for subagent_ref in result.subagent_trajectory_ref or []:
                if subagent_ref.trajectory_id is None and subagent_ref.trajectory_path is None:
                    raise ValueError(
                        "ATIF-v1.7 SubagentTrajectoryRef MUST set at least one of "
                        "trajectory_id or trajectory_path; session_id alone is informational"
                    )


class AtifTrajectory(BaseModel):
    schema_version: AtifSchemaVersion = ATIF_VERSION
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    trajectory_id: str | None = None
    agent: AtifAgent
    steps: list[AtifStep] = Field(default_factory=list)
    notes: str | None = None
    final_metrics: AtifFinalMetrics | None = None
    continued_trajectory_ref: str | None = None
    extra: dict[str, Any] | None = None
    subagent_trajectories: list[AtifTrajectory] | None = Field(
        default=None,
        description=(
            "Embedded ATIF-v1.7 subagent trajectories. Intake expands these into the parent trajectory's "
            "trace, resolving subagent_trajectory_ref entries by trajectory_id."
        ),
    )
    evaluation_context: EvaluationContext | None = None

    model_config = ConfigDict(extra="forbid")

    def to_json_dict(self, *, exclude_none: bool = True) -> dict[str, Any]:
        """Serialize the trajectory to JSON-compatible values."""
        return self.model_dump(exclude_none=exclude_none, mode="json")

    @model_validator(mode="after")
    def validate_step_ids(self) -> AtifTrajectory:
        """Validate this trajectory's sequential step identifiers."""
        validate_atif_step_ids(self.steps)
        return self

    @model_validator(mode="after")
    def validate_tool_call_references(self) -> AtifTrajectory:
        """Validate this trajectory's tool-call references."""
        validate_atif_tool_call_references(self.steps)
        return self

    @model_validator(mode="after")
    def validate_subagent_ref_resolution_keys(self) -> AtifTrajectory:
        """Validate v1.7 subagent reference resolution keys."""
        if self.schema_version == "ATIF-v1.7":
            validate_atif_v17_subagent_ref_resolution_keys(self.steps)
        return self

    @model_validator(mode="after")
    def validate_subagent_trajectory_ids(self) -> AtifTrajectory:
        """Validate IDs for embedded trajectories at this tree level."""
        validate_atif_subagent_trajectory_ids(self.subagent_trajectories)
        return self


def validate_atif_subagent_trajectory_ids(subagent_trajectories: list[AtifTrajectory] | None) -> None:
    """Require unique trajectory IDs for embedded sibling trajectories."""
    if not subagent_trajectories:
        return
    seen: set[str] = set()
    for index, subagent in enumerate(subagent_trajectories):
        if subagent.trajectory_id is None:
            raise ValueError(
                f"subagent_trajectories[{index}].trajectory_id is required on embedded ATIF-v1.7 subagents"
            )
        if subagent.trajectory_id in seen:
            raise ValueError(f"subagent_trajectories[{index}].trajectory_id duplicates {subagent.trajectory_id!r}")
        seen.add(subagent.trajectory_id)
