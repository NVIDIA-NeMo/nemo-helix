# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request/response types for the Guardrails service.

Single source of truth for the HTTP contract. Replaces the Stainless-generated
``nemo_platform.types.guardrail`` module.
"""

from __future__ import annotations

from collections.abc import ItemsView, KeysView
from datetime import datetime
from typing import Any, Literal, NotRequired, TypedDict

from nemo_platform_plugin.schema import Page
from pydantic import BaseModel, ConfigDict, Field


class _GuardrailValue(BaseModel):
    """Forward-compatible base for guardrails wire DTOs."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    def _mapping_values(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)

    def __getitem__(self, key: str) -> Any:
        values = self._mapping_values()
        try:
            return values[key]
        except KeyError:
            raise KeyError(key) from None

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return self._mapping_values().get(key, default)

    def keys(self) -> KeysView[str]:
        return self._mapping_values().keys()

    def items(self) -> ItemsView[str, Any]:
        return self._mapping_values().items()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            return self._mapping_values() == other
        return super().__eq__(other)


# ---------------------------------------------------------------------------
# Rails config value types
# ---------------------------------------------------------------------------


class ModelParameters(_GuardrailValue):
    """Parameters for configuring how to interact with a model in a guardrails config."""

    base_url: str | None = None
    default_headers: dict[str, str] | None = None


class Model(_GuardrailValue):
    """Configuration of a model used by the rails engine."""

    engine: str
    type: str
    cache: dict[str, Any] | None = None
    mode: Literal["chat", "text"] | None = None
    model: str | None = None
    parameters: ModelParameters | None = None


class InputRails(_GuardrailValue):
    """Configuration of input rails."""

    flows: list[str] | None = None
    parallel: bool | None = None


class OutputRailsStreamingConfig(_GuardrailValue):
    """Configuration for managing streaming output of LLM tokens."""

    chunk_size: int | None = None
    context_size: int | None = None
    enabled: bool | None = None
    stream_first: bool | None = None


class OutputRails(_GuardrailValue):
    """Configuration of output rails."""

    apply_to_reasoning_traces: bool | None = None
    flows: list[str] | None = None
    parallel: bool | None = None
    streaming: OutputRailsStreamingConfig | None = None


class Rails(_GuardrailValue):
    """Configuration of specific rails."""

    actions: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    dialog: dict[str, Any] | None = None
    input: InputRails | None = None
    output: OutputRails | None = None
    retrieval: dict[str, Any] | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: dict[str, Any] | None = None


class RailsConfig(_GuardrailValue):
    """Configuration object for the models and the rails."""

    actions_server_url: str | None = None
    colang_version: str | None = None
    custom_data: dict[str, Any] | None = None
    enable_multi_step_generation: bool | None = None
    enable_rails_exceptions: bool | None = None
    instructions: list[dict[str, Any]] | None = None
    lowest_temperature: float | None = None
    models: list[Model] | None = None
    passthrough: bool | None = None
    prompting_mode: str | None = None
    prompts: list[dict[str, Any]] | None = None
    rails: Rails | None = None
    sample_conversation: str | None = None
    tracing: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Generation / guardrails data types
# ---------------------------------------------------------------------------


class GenerationLogOptionsParam(TypedDict, total=False):
    """Options for what should be included in the generation log."""

    activated_rails: bool
    colang_history: bool
    internal_events: bool
    llm_calls: bool
    stats: bool


class GenerationStats(_GuardrailValue):
    """General stats about a guardrails generation."""

    dialog_rails_duration: float | None = None
    generation_rails_duration: float | None = None
    input_rails_duration: float | None = None
    llm_calls_count: int | None = None
    llm_calls_duration: float | None = None
    llm_calls_total_completion_tokens: int | None = None
    llm_calls_total_prompt_tokens: int | None = None
    llm_calls_total_tokens: int | None = None
    output_rails_duration: float | None = None
    total_duration: float | None = None


class LLMCallInfo(_GuardrailValue):
    """Information about an LLM call made while evaluating rails."""

    id: str | None = None
    completion: str | None = None
    completion_tokens: int | None = None
    duration: float | None = None
    finished_at: float | None = None
    llm_model_name: str | None = None
    prompt: str | None = None
    prompt_tokens: int | None = None
    raw_response: dict[str, Any] | None = None
    started_at: float | None = None
    task: str | None = None
    total_tokens: int | None = None


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------


class GuardrailConfig(_GuardrailValue):
    """A guardrail configuration entity."""

    name: str = ""
    workspace: str
    project: str | None = None
    description: str | None = None
    data: RailsConfig = Field(default_factory=RailsConfig, description="Guardrail configuration data")
    id: str
    created_at: datetime
    created_by: str | None = None
    updated_at: datetime
    updated_by: str | None = None


GuardrailConfigPage = Page[GuardrailConfig]


class RailStatus(_GuardrailValue):
    """Status of an individual rail."""

    status: str = Field(description="Status of the individual rail: success, blocked, or unknown.")


class ActivatedRail(_GuardrailValue):
    """A rail that ran during a check, as reported in the generation log."""

    name: str = ""
    type: str = ""
    additional_info: dict[str, Any] | None = None
    decisions: list[str] | None = None
    duration: float | None = None
    executed_actions: list[dict[str, Any]] | None = None
    finished_at: float | None = None
    started_at: float | None = None
    stop: bool | None = None


class GenerationLog(_GuardrailValue):
    """Logging information about a guardrails generation."""

    activated_rails: list[ActivatedRail] | None = None
    colang_history: str | None = None
    internal_events: list[dict[str, Any]] | None = None
    llm_calls: list[LLMCallInfo] | None = None
    stats: GenerationStats | None = None


class GuardrailsData(_GuardrailValue):
    """Guardrails-specific output attached to a check or chat response."""

    llm_output: dict[str, Any] | None = None
    config_ids: list[str] | None = Field(default=None, description="Configuration ids that were used.")
    output_data: dict[str, Any] | None = None
    log: GenerationLog | None = Field(default=None, description="Populated when guardrails log options are requested.")


class GuardrailsDataOutput(GuardrailsData):
    """Guardrails-specific output returned by the check API."""


class GuardrailCheckResponse(_GuardrailValue):
    """Response from a guardrail check request."""

    status: str = Field(description="Overall status: success if all rails passed, blocked if any failed.")
    rails_status: dict[str, RailStatus] = Field(
        default_factory=dict, description="Status of each rail, keyed by rail name."
    )
    guardrails_data: GuardrailsDataOutput | None = None


# ---------------------------------------------------------------------------
# Request types
# ---------------------------------------------------------------------------


class CreateGuardrailConfigRequest(BaseModel):
    """Input schema for creating a guardrail config."""

    name: str
    description: str | None = None
    data: RailsConfig | dict[str, Any] | None = Field(default=None, description="Guardrail configuration data")


class UpdateGuardrailConfigRequest(BaseModel):
    """Input schema for updating a guardrail config."""

    description: str | None = None
    data: RailsConfig | dict[str, Any] | None = None


class GuardrailCheckRequest(_GuardrailValue):
    """Guardrail check request body.

    Shaped like an OpenAI chat-completions request. ``extra="allow"`` passes
    through any additional sampling parameters the backend accepts.
    """

    model: str = Field(description="The model the checked conversation targets.")
    messages: list[dict[str, Any]] = Field(description="The conversation to check, in OpenAI chat format.")
    guardrails: dict[str, Any] = Field(
        default_factory=dict,
        description="Guardrails options for the request, e.g. config_id, config, or options.",
    )
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


# ---------------------------------------------------------------------------
# Query parameter types
# ---------------------------------------------------------------------------


class ListGuardrailConfigsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]
    project: NotRequired[str]
