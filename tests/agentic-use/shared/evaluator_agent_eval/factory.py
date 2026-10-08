# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct captured attempts and Evaluator scoring rows."""

from pathlib import Path

from evaluator_agent_eval.artifacts import AgentArtifacts
from evaluator_agent_eval.atif import summarize_atif_trajectory
from evaluator_agent_eval.schemas import (
    AgentAttemptInput,
    AgentAttemptMetadata,
    AgentAttemptOutput,
    AgentAttemptTrace,
    CapturedAgentAttempt,
    EvaluatorScoringRow,
)
from evaluator_agent_eval.surfaces import detect_surfaces, evidence_from_artifacts
from evaluator_agent_eval.task_config import AgenticUseTaskConfig, load_agentic_use_task_config
from pydantic import BaseModel, ConfigDict


class AgentRunMetadata(BaseModel):
    """Runtime metadata captured by the benchmark runner."""

    model_config = ConfigDict(extra="forbid")

    agent_runtime: str = "unknown"
    agent_model: str = "unknown"
    agent_runtime_version: str | None = None
    repo_revision: str | None = None
    run_id: str | None = None
    exit_code: int | None = None
    duration_ms: int | None = None


def capture_agent_attempt(
    *,
    task_dir: str | Path,
    artifacts: AgentArtifacts,
    metadata: AgentRunMetadata | None = None,
) -> CapturedAgentAttempt:
    """Build a generic captured attempt."""
    root = Path(task_dir)
    run_metadata = metadata or AgentRunMetadata()
    instruction_path = root / "instruction.md"
    instruction_text = instruction_path.read_text(encoding="utf-8") if instruction_path.exists() else ""
    raw_log_paths = [
        str(path.relative_to(artifacts.agent_log_dir))
        for path in sorted(artifacts.agent_log_dir.iterdir())
        if path.is_file()
    ]
    trace = (
        AgentAttemptTrace(atif_path=str(artifacts.atif_trajectory_path))
        if artifacts.atif_trajectory_path is not None
        else None
    )

    return CapturedAgentAttempt(
        task_id=root.name,
        input=AgentAttemptInput(instruction_text=instruction_text, instruction_path=str(instruction_path)),
        output=AgentAttemptOutput(
            final_text=artifacts.final_answer.text,
            final_answer_extracted=artifacts.final_answer.extracted,
            final_answer_source=artifacts.final_answer.source,
            raw_log_paths=raw_log_paths,
        ),
        metadata=AgentAttemptMetadata(
            agent_runtime=run_metadata.agent_runtime,
            agent_model=run_metadata.agent_model,
            agent_runtime_version=run_metadata.agent_runtime_version,
            repo_revision=run_metadata.repo_revision,
            run_id=run_metadata.run_id,
            exit_code=run_metadata.exit_code,
            duration_ms=run_metadata.duration_ms,
        ),
        trace=trace,
    )


def build_evaluator_scoring_row(
    *,
    task_dir: str | Path,
    attempt: CapturedAgentAttempt,
    artifacts: AgentArtifacts,
    task_config: AgenticUseTaskConfig | None = None,
) -> EvaluatorScoringRow:
    """Build an Evals SDK dataset row for one captured attempt."""
    root = Path(task_dir)
    config = task_config or load_agentic_use_task_config(root)
    evidence = evidence_from_artifacts(
        final_answer_text=attempt.output.final_text,
        raw_text=_surface_raw_text(artifacts),
        command_json_path=artifacts.agent_log_dir / "command.json",
    )
    surface_detection = detect_surfaces(
        evidence,
        forbidden_patterns=config.evaluator.forbidden_patterns,
    )
    trajectory_summary = (
        summarize_atif_trajectory(artifacts.atif_trajectory_path)
        if artifacts.atif_trajectory_path is not None
        else None
    )

    surface = config.evaluator.surface
    return EvaluatorScoringRow(
        task_id=attempt.task_id,
        task_dir=str(root),
        task_suite_id=config.suite_id,
        task_suite_version=config.suite_version,
        instruction_text=attempt.input.instruction_text,
        instruction_path=attempt.input.instruction_path,
        agent_log_dir=str(artifacts.agent_log_dir),
        workspace_dir=str(artifacts.workspace_dir) if artifacts.workspace_dir is not None else None,
        agent_runtime=attempt.metadata.agent_runtime,
        agent_model=attempt.metadata.agent_model,
        agent_runtime_version=attempt.metadata.agent_runtime_version,
        surface_constraint=surface.constraint,
        allowed_surfaces=surface.allowed,
        forbidden_surfaces=surface.forbidden,
        repo_revision=attempt.metadata.repo_revision,
        run_id=attempt.metadata.run_id,
        output_text=attempt.output.final_text,
        final_answer_extracted=attempt.output.final_answer_extracted,
        final_answer_source=attempt.output.final_answer_source,
        raw_log_paths=attempt.output.raw_log_paths,
        observed_surfaces=surface_detection.observed_surfaces,
        forbidden_surface_hits=surface_detection.forbidden_surface_hits,
        atif_trajectory_path=attempt.trace.atif_path if attempt.trace else None,
        trajectory_summary=trajectory_summary,
        exit_code=attempt.metadata.exit_code,
        duration_ms=attempt.metadata.duration_ms,
    )


def _surface_raw_text(artifacts: AgentArtifacts) -> str:
    parts: list[str] = []
    solution_path = artifacts.workspace_artifact("solution.py")
    if solution_path is not None and solution_path.is_file():
        parts.append(solution_path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(part for part in parts if part)
