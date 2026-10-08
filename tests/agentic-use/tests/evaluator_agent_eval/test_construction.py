# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluator_agent_eval construction helpers."""

import json
from pathlib import Path

from evaluator_agent_eval.artifacts import AgentArtifacts
from evaluator_agent_eval.factory import AgentRunMetadata, build_evaluator_scoring_row, capture_agent_attempt


def test_capture_agent_attempt_from_task_config_and_artifacts(evaluator_task_dir: Path, agent_log_dir: Path):
    (agent_log_dir / "final_message.txt").write_text(
        "Use packages/nhx_evals_sdk with Evaluator, run_sync, and ExactMatchMetric.",
        encoding="utf-8",
    )

    captured_attempt = capture_agent_attempt(
        task_dir=evaluator_task_dir,
        artifacts=AgentArtifacts.from_dir(agent_log_dir),
        metadata=AgentRunMetadata(agent_runtime="claude-code", agent_model="sonnet"),
    )

    assert captured_attempt.task_id == "task"
    assert captured_attempt.input.instruction_path is not None
    assert (
        captured_attempt.output.final_text
        == "Use packages/nhx_evals_sdk with Evaluator, run_sync, and ExactMatchMetric."
    )
    assert captured_attempt.output.final_answer_extracted is True
    assert captured_attempt.metadata.agent_runtime == "claude-code"
    assert captured_attempt.metadata.agent_model == "sonnet"


def test_build_evaluator_scoring_row_from_task_config_and_artifacts(evaluator_task_dir: Path, agent_log_dir: Path):
    (agent_log_dir / "final_message.txt").write_text(
        "Use packages/nhx_evals_sdk with Evaluator, run_sync, and ExactMatchMetric.",
        encoding="utf-8",
    )
    artifacts = AgentArtifacts.from_dir(agent_log_dir)
    attempt = capture_agent_attempt(
        task_dir=evaluator_task_dir,
        artifacts=artifacts,
        metadata=AgentRunMetadata(agent_runtime="claude-code", agent_model="sonnet"),
    )

    scoring_row = build_evaluator_scoring_row(
        task_dir=evaluator_task_dir,
        attempt=attempt,
        artifacts=artifacts,
    )

    assert scoring_row.task_id == "task"
    assert scoring_row.task_dir == str(evaluator_task_dir)
    assert scoring_row.task_suite_id == "custom_suite"
    assert scoring_row.task_suite_version == "v-test"
    assert scoring_row.instruction_path == str(evaluator_task_dir / "instruction.md")
    assert scoring_row.agent_log_dir == str(agent_log_dir)
    assert scoring_row.agent_runtime == "claude-code"
    assert scoring_row.agent_model == "sonnet"
    assert scoring_row.final_answer_extracted is True
    assert scoring_row.final_answer_source == "final_message.txt"
    assert scoring_row.observed_surfaces == ["standalone_sdk"]


def test_build_evaluator_scoring_row_allows_sdk_task_metrics_to_populate_scores(
    evaluator_task_dir: Path, agent_log_dir: Path
):
    (agent_log_dir / "final_message.txt").write_text(
        "Use packages/nhx_evals_sdk with Evaluator, run_sync, and ExactMatchMetric.",
        encoding="utf-8",
    )
    artifacts = AgentArtifacts.from_dir(agent_log_dir)
    attempt = capture_agent_attempt(
        task_dir=evaluator_task_dir,
        artifacts=artifacts,
        metadata=AgentRunMetadata(agent_runtime="claude-code", agent_model="sonnet"),
    )

    scoring_row = build_evaluator_scoring_row(
        task_dir=evaluator_task_dir,
        attempt=attempt,
        artifacts=artifacts,
    )

    assert scoring_row.output_text == "Use packages/nhx_evals_sdk with Evaluator, run_sync, and ExactMatchMetric."


def test_build_evaluator_scoring_row_detects_surface_from_workspace_artifact(
    evaluator_task_dir: Path, agent_log_dir: Path, tmp_path: Path
):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "solution.py").write_text(
        "from nhx_evals_sdk import Evaluator, ExactMatchMetric\n",
        encoding="utf-8",
    )
    (agent_log_dir / "final_message.txt").write_text("Wrote the requested solution.", encoding="utf-8")
    artifacts = AgentArtifacts.from_dir(agent_log_dir, workspace_dir=workspace_dir)
    attempt = capture_agent_attempt(
        task_dir=evaluator_task_dir,
        artifacts=artifacts,
        metadata=AgentRunMetadata(agent_runtime="claude-code", agent_model="sonnet"),
    )

    scoring_row = build_evaluator_scoring_row(
        task_dir=evaluator_task_dir,
        attempt=attempt,
        artifacts=artifacts,
    )

    assert scoring_row.observed_surfaces == ["standalone_sdk"]


def test_build_evaluator_scoring_row_fails_success_when_final_answer_missing(
    evaluator_task_dir: Path, agent_log_dir: Path
):
    (agent_log_dir / "final_message.txt").write_text(json.dumps({"result": ""}), encoding="utf-8")
    artifacts = AgentArtifacts.from_dir(agent_log_dir)
    attempt = capture_agent_attempt(
        task_dir=evaluator_task_dir,
        artifacts=artifacts,
        metadata=AgentRunMetadata(agent_runtime="claude-code", agent_model="sonnet"),
    )

    scoring_row = build_evaluator_scoring_row(
        task_dir=evaluator_task_dir,
        attempt=attempt,
        artifacts=artifacts,
    )

    assert scoring_row.output_text == ""


def test_capture_agent_attempt_carries_runtime_metadata(evaluator_task_dir: Path, agent_log_dir: Path):
    (agent_log_dir / "final_message.txt").write_text(
        "Use packages/nhx_evals_sdk with Evaluator, run_sync, and ExactMatchMetric.",
        encoding="utf-8",
    )

    captured_attempt = capture_agent_attempt(
        task_dir=evaluator_task_dir,
        artifacts=AgentArtifacts.from_dir(agent_log_dir),
        metadata=AgentRunMetadata(
            agent_runtime="aut",
            agent_model="candidate",
            agent_runtime_version="1.2.3",
            repo_revision="abc123",
            run_id="run-1",
            exit_code=0,
            duration_ms=42,
        ),
    )

    assert captured_attempt.metadata.agent_runtime_version == "1.2.3"
    assert captured_attempt.metadata.repo_revision == "abc123"
    assert captured_attempt.metadata.run_id == "run-1"
    assert captured_attempt.metadata.exit_code == 0
    assert captured_attempt.metadata.duration_ms == 42


def test_build_evaluator_scoring_row_validates_atif_and_populates_trajectory_summary(
    evaluator_task_dir: Path,
    agent_log_dir: Path,
    atif_payload: dict[str, object],
):
    (agent_log_dir / "final_message.txt").write_text(
        "Use packages/nhx_evals_sdk with Evaluator, run_sync, and ExactMatchMetric.",
        encoding="utf-8",
    )
    (agent_log_dir / "trajectory.json").write_text(json.dumps(atif_payload), encoding="utf-8")
    artifacts = AgentArtifacts.from_dir(agent_log_dir)
    attempt = capture_agent_attempt(
        task_dir=evaluator_task_dir,
        artifacts=artifacts,
        metadata=AgentRunMetadata(agent_runtime="claude-code", agent_model="sonnet"),
    )

    scoring_row = build_evaluator_scoring_row(
        task_dir=evaluator_task_dir,
        attempt=attempt,
        artifacts=artifacts,
    )

    assert scoring_row.atif_trajectory_path == str(agent_log_dir / "trajectory.json")
    assert scoring_row.trajectory_summary is not None
    assert scoring_row.trajectory_summary.tool_call_count == 2
    assert scoring_row.trajectory_summary.failed_command_count == 1
    assert scoring_row.trajectory_summary.recovery_event_count == 1


def test_build_evaluator_scoring_row_ignores_raw_log_prompt_echo_for_surface_detection(
    evaluator_task_dir: Path, agent_log_dir: Path
):
    (agent_log_dir / "final_message.txt").write_text(
        "Use only packages/nhx_evals_sdk and do not use services/*.",
        encoding="utf-8",
    )
    (agent_log_dir / "nat_agent.log").write_text(
        "Prompt reminder: do not use the nemo evaluation CLI, plugin SDK APIs, or services/*.",
        encoding="utf-8",
    )
    artifacts = AgentArtifacts.from_dir(agent_log_dir)
    attempt = capture_agent_attempt(
        task_dir=evaluator_task_dir,
        artifacts=artifacts,
        metadata=AgentRunMetadata(agent_runtime="claude-code", agent_model="sonnet"),
    )

    scoring_row = build_evaluator_scoring_row(
        task_dir=evaluator_task_dir,
        attempt=attempt,
        artifacts=artifacts,
    )

    assert scoring_row.observed_surfaces == ["standalone_sdk"]
    assert scoring_row.forbidden_surface_hits == []
