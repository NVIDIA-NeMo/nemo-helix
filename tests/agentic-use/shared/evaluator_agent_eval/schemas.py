# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schemas for captured agent attempts and Evaluator scoring rows."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SurfaceName = Literal["cli", "plugin_sdk", "standalone_sdk", "legacy_service", "docs", "unknown"]


class AgentAttemptInput(BaseModel):
    """Input shown to a candidate agent."""

    model_config = ConfigDict(extra="forbid")

    instruction_text: str
    instruction_path: str | None = None


class AgentAttemptOutput(BaseModel):
    """Output captured from a candidate agent."""

    model_config = ConfigDict(extra="forbid")

    final_text: str = ""
    final_answer_extracted: bool = False
    final_answer_source: str | None = None
    raw_log_paths: list[str] = Field(default_factory=list)


class AgentAttemptTrace(BaseModel):
    """Trajectory artifacts captured for a candidate agent."""

    model_config = ConfigDict(extra="forbid")

    atif_path: str | None = None


class AgentAttemptMetadata(BaseModel):
    """Runtime metadata captured for a candidate agent attempt."""

    model_config = ConfigDict(extra="forbid")

    agent_runtime: str
    agent_model: str
    agent_runtime_version: str | None = None
    repo_revision: str | None = None
    run_id: str | None = None
    exit_code: int | None = None
    duration_ms: int | None = None

    @field_validator("duration_ms")
    @classmethod
    def _duration_non_negative(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("duration_ms must be non-negative")
        return value


class CapturedAgentAttempt(BaseModel):
    """One normalized captured candidate-agent attempt.

    This is the portable attempt record. It describes what the agent was asked,
    what it returned, optional trajectory artifacts, and runtime metadata. It
    deliberately does not encode evaluator-specific scoring policy.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str
    input: AgentAttemptInput
    output: AgentAttemptOutput
    metadata: AgentAttemptMetadata
    trace: AgentAttemptTrace | None = None


class TrajectorySummary(BaseModel):
    """Small trajectory summary consumed by trajectory metrics."""

    model_config = ConfigDict(extra="forbid")

    tool_call_count: int = 0
    failed_command_count: int = 0
    recovery_event_count: int = 0

    @field_validator("tool_call_count", "failed_command_count", "recovery_event_count")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("trajectory counts must be non-negative")
        return value


class EvaluatorScoringRow(BaseModel):
    """Evaluator-specific dataset row derived from one captured attempt.

    This is the row shape passed to ``Evaluator().run_sync(dataset=...)`` for
    offline scoring. It combines generic attempt data with task scoring policy.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_dir: str | None = None
    task_suite_id: str = "evaluator_agent_benchmark"
    task_suite_version: str = "v0"
    instruction_text: str = ""
    instruction_path: str | None = None
    agent_log_dir: str | None = None
    workspace_dir: str | None = None
    agent_runtime: str
    agent_model: str
    agent_runtime_version: str | None = None
    surface_constraint: SurfaceName
    allowed_surfaces: list[SurfaceName]
    forbidden_surfaces: list[SurfaceName] = Field(default_factory=lambda: ["legacy_service"])
    repo_revision: str | None = None
    run_id: str | None = None
    output_text: str = ""
    final_answer_extracted: bool = False
    final_answer_source: str | None = None
    raw_log_paths: list[str] = Field(default_factory=list)
    observed_surfaces: list[SurfaceName] = Field(default_factory=list)
    forbidden_surface_hits: list[str] = Field(default_factory=list)
    atif_trajectory_path: str | None = None
    trajectory_summary: TrajectorySummary | None = None
    exit_code: int | None = None
    duration_ms: int | None = None

    @field_validator("allowed_surfaces")
    @classmethod
    def _allowed_surfaces_not_empty(cls, value: list[SurfaceName]) -> list[SurfaceName]:
        if not value:
            raise ValueError("allowed_surfaces must not be empty")
        return value

    def to_dataset_row(self) -> dict[str, object]:
        """Serialize as a JSON-compatible Evals SDK dataset row."""
        return self.model_dump(mode="json")


def load_evaluator_scoring_rows_jsonl(path: str | Path) -> list[EvaluatorScoringRow]:
    """Load Evaluator scoring rows from a JSONL file."""
    rows: list[EvaluatorScoringRow] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(EvaluatorScoringRow.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"Invalid evaluator scoring row at {path}:{line_number}: {exc}") from exc
    return rows
