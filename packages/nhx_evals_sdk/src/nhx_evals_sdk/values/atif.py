# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Permissive ATIF read models for the evals SDK.

The evaluator ingests traces that producers emit in the Agent Trajectory
Interchange Format (ATIF; RFC 0001, schema_version ``ATIF-v1.x``). Selected
fields are derived from Harbor's reference models at commit
``aaf0561340fd2f03257ec3084732f98537a2d2b1``. They are intentionally not a
byte-for-byte vendoring: this SDK keeps ``extra="ignore"``, makes newly consumed
fields optional where possible, and omits producer-side cross-field validators,
multimodal content models, and embedded subagent trajectories.

Validation here means "this payload carries the ATIF fields evaluator metrics
read", not full RFC conformance. The producer's original dictionary remains the
authoritative persistence representation; these models provide typed read access.
"""

# ruff: noqa: I001 - the vendored SDK mirror uses different import-order settings.

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolCall(BaseModel):
    """A single tool/function invocation within an agent step."""

    model_config = ConfigDict(extra="ignore")

    tool_call_id: str | None = Field(default=None, description="Producer-assigned tool call id, if any.")
    function_name: str = Field(description="Name of the invoked tool/function.")
    arguments: dict[str, Any] | None = Field(default=None, description="Arguments passed to the tool.")
    extra: dict[str, Any] | None = Field(default=None, description="Custom tool-call metadata.")


class ObservationResult(BaseModel):
    """One tool or environment result attached to an observation."""

    model_config = ConfigDict(extra="ignore")

    source_call_id: str | None = Field(default=None, description="Related ToolCall.tool_call_id, if any.")
    content: Any | None = Field(default=None, description="Tool result content retained for evidence readers.")
    extra: dict[str, Any] | None = Field(default=None, description="Custom result-level metadata.")


class Observation(BaseModel):
    """Environment feedback following tool calls or other actions."""

    model_config = ConfigDict(extra="ignore")

    results: list[ObservationResult] = Field(default_factory=list, description="Results produced by the action.")


class Metrics(BaseModel):
    """Per-step token metrics."""

    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    extra: dict[str, Any] | None = None


class FinalMetrics(BaseModel):
    """Trajectory-level aggregate token metrics."""

    model_config = ConfigDict(extra="ignore")

    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_cached_tokens: int | None = None
    total_cost_usd: float | None = None
    total_steps: int | None = Field(default=None, ge=0)
    extra: dict[str, Any] | None = None


class Agent(BaseModel):
    """ATIF protocol producer recorded in a trajectory, not an inference target."""

    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    version: str | None = None
    model_name: str | None = None
    tool_definitions: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None


class Step(BaseModel):
    """One step in an agent trajectory."""

    model_config = ConfigDict(extra="ignore")

    step_id: int | None = Field(default=None, ge=1, description="Producer-assigned ordinal step index.")
    timestamp: str | None = Field(default=None, description="Producer-reported ISO 8601 timestamp.")
    source: Literal["system", "user", "agent"] = Field(description="Who produced this step.")
    model_name: str | None = Field(default=None, description="Model used for this step, if reported.")
    reasoning_effort: str | float | None = Field(default=None, description="Reported reasoning effort.")
    message: str = Field(default="", description="Step text content.")
    reasoning_content: str | None = Field(default=None, description="Explicit reasoning content, if exposed.")
    tool_calls: list[ToolCall] | None = Field(default=None, description="Tool calls issued in this step.")
    observation: Observation | None = Field(default=None, description="Environment feedback for this step.")
    metrics: Metrics | None = Field(default=None, description="Per-step token metrics, if reported.")
    is_copied_context: bool | None = Field(default=None, description="Whether the step was copied as context.")
    llm_call_count: int | None = Field(default=None, ge=0, description="LLM calls represented by this step.")
    extra: dict[str, Any] | None = Field(default=None, description="Custom step-level metadata.")


class Trajectory(BaseModel):
    """An ATIF trajectory read view over the fields evaluator metrics consume."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field(description="ATIF schema version, e.g. 'ATIF-v1.7'.")
    session_id: str | None = Field(default=None, description="Identifier for the logical agent run.")
    trajectory_id: str | None = Field(default=None, description="Identifier for this trajectory document.")
    agent: Agent | None = Field(default=None, description="Producer-recorded agent configuration.")
    steps: list[Step] = Field(min_length=1, description="Ordered trajectory steps.")
    notes: str | None = Field(default=None, description="Producer notes about the trajectory.")
    final_metrics: FinalMetrics | None = Field(default=None, description="Aggregate token metrics, if reported.")
    continued_trajectory_ref: str | None = Field(default=None, description="Reference to a continuation trace.")
    extra: dict[str, Any] | None = Field(default=None, description="Custom trajectory-level metadata.")

    @field_validator("schema_version")
    @classmethod
    def _looks_like_atif(cls, value: str) -> str:
        # Cheap sanity gate so arbitrary JSON isn't silently accepted as a trace.
        if not value.startswith("ATIF-"):
            raise ValueError(f"unexpected trace schema_version {value!r}; expected an 'ATIF-*' version")
        return value
