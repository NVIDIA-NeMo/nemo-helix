# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed extension point for translating agent stream frames into ATIF evidence."""

# ruff: noqa: I001 - the vendored SDK mirror uses different import-order settings.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nhx_evals_sdk.values.atif import Trajectory
from nhx_evals_sdk.values.evidence import EvidenceDescriptor


class SseFrame(BaseModel):
    """One parsed field from an agent's JSON SSE response."""

    model_config = ConfigDict(extra="forbid")

    channel: str
    payload: Any
    raw: str


class AgentStreamTranslationContext(BaseModel):
    """Non-secret invocation context supplied to an agent stream translator."""

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    endpoint: str
    request_payload: dict[str, Any]
    final_payload: Any | None = None
    output_text: str | None = None
    run_id: str | None = None
    task_id: str | None = None
    invocation_id: str | None = None
    conversation_id: str | None = None
    http_status: int | None = None
    stream_error: str | None = None


class AgentStreamTranslation(BaseModel):
    """Canonical ATIF plus optional client-owned derived evidence.

    The trajectory stays as the producer-emitted dictionary so fields outside
    the evals SDK's lightweight ATIF read model are not discarded.
    """

    model_config = ConfigDict(extra="forbid")

    trajectory: dict[str, Any]
    evidence: dict[str, EvidenceDescriptor] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("trajectory")
    @classmethod
    def _validate_atif(cls, value: dict[str, Any]) -> dict[str, Any]:
        Trajectory.model_validate(value)
        return value


@runtime_checkable
class AgentStreamTranslator(Protocol):
    """Translate captured agent stream frames into canonical ATIF evidence."""

    def __call__(
        self,
        frames: Sequence[SseFrame],
        *,
        context: AgentStreamTranslationContext,
    ) -> AgentStreamTranslation: ...
