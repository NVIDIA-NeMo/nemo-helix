# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluator_agent_eval.artifacts."""

import json
from pathlib import Path

from evaluator_agent_eval.artifacts import AgentArtifacts, FinalAnswer


def test_agent_artifacts_do_not_treat_json_without_final_answer_as_text(agent_log_dir: Path):
    (agent_log_dir / "final_message.txt").write_text(
        json.dumps({"result": "", "permission_denials": [{"tool_name": "ExitPlanMode"}]}),
        encoding="utf-8",
    )

    artifacts = AgentArtifacts.from_dir(agent_log_dir)

    assert artifacts.final_answer == FinalAnswer(extracted=False, source="final_message.txt")


def test_agent_artifacts_extract_json_result(agent_log_dir: Path):
    (agent_log_dir / "nat_agent.log").write_text(
        json.dumps({"result": "Use packages/nhx_evals_sdk with Evaluator.run_sync and ExactMatchMetric."}),
        encoding="utf-8",
    )

    artifacts = AgentArtifacts.from_dir(agent_log_dir)

    assert artifacts.final_answer.extracted is True
    assert "ExactMatchMetric" in artifacts.final_answer.text
    assert artifacts.final_answer.source == "nat_agent.log:json.result"


def test_agent_artifacts_extract_latest_jsonl_result(agent_log_dir: Path):
    (agent_log_dir / "nat_agent.log").write_text(
        "\n".join(
            [
                json.dumps({"result": "first"}),
                json.dumps({"result": "second"}),
            ]
        ),
        encoding="utf-8",
    )

    artifacts = AgentArtifacts.from_dir(agent_log_dir)

    assert artifacts.final_answer == FinalAnswer(extracted=True, text="second", source="nat_agent.log:jsonl.result")


def test_plain_final_message_is_final_answer(agent_log_dir: Path):
    (agent_log_dir / "final_message.txt").write_text("plain answer", encoding="utf-8")

    artifacts = AgentArtifacts.from_dir(agent_log_dir)

    assert artifacts.final_answer == FinalAnswer(extracted=True, text="plain answer", source="final_message.txt")


def test_agent_artifacts_find_atif_trajectory(agent_log_dir: Path):
    trajectory_path = agent_log_dir / "trajectory.json"
    trajectory_path.write_text("{}", encoding="utf-8")

    artifacts = AgentArtifacts.from_dir(agent_log_dir)

    assert artifacts.atif_trajectory_path == trajectory_path


def test_agent_artifacts_resolve_workspace_artifact(agent_log_dir: Path, tmp_path: Path):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    solution_path = workspace_dir / "solution.py"
    solution_path.write_text("print('ok')", encoding="utf-8")

    artifacts = AgentArtifacts.from_dir(agent_log_dir, workspace_dir=workspace_dir)

    assert artifacts.workspace_artifact("solution.py") == solution_path
    assert artifacts.workspace_artifact("../escape.py") is None
