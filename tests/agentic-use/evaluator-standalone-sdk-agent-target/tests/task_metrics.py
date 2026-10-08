# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-specific metrics for the standalone SDK agent-target task."""

import ast
import asyncio
import subprocess
from pathlib import Path

from evaluator_agent_eval.artifacts import AgentArtifacts
from evaluator_agent_eval.task_config import load_agentic_use_task_config
from evaluator_agent_eval.task_metric_utils import (
    contains_all,
    extract_fenced_python_code,
    extract_json_object_from_stdout,
    find_workspace_python_code,
    object_dict,
    run_python_code,
    score_checks,
)
from nhx_evals_sdk.values.results import MetricResult, MetricScore


class AgentTargetConfigurationMetric:
    """Verify the candidate configured an Evals SDK Agent target."""

    @property
    def type(self) -> str:
        return "agent_eval/agent_target_configuration"

    def score_names(self) -> list[str]:
        return [
            "task_success",
            "verification_score",
            "output_schema_valid",
            "code_block_present",
            "code_ran",
            "required_terms_present",
            "agent_target_terms_present",
            "agent_target_used",
            "extracted_answer",
            "score_correct",
            "trajectory_captured",
            "candidate_metadata_present",
        ]

    async def compute_scores(self, item: dict, sample: dict) -> MetricResult:
        final_text = str(item.get("output_text", ""))
        artifacts = _artifacts_from_item(item)
        code = _extract_python_code(final_text) or _extract_workspace_python_code(artifacts)
        process = await asyncio.to_thread(_run_candidate_code, code) if code else None
        stdout = process.stdout if process is not None else ""
        output_payload = _extract_output_payload(stdout)

        text_to_check = f"{final_text}\n{code or ''}"
        required_terms_present = contains_all(text_to_check, _required_terms(item))
        code_ran = process is not None and process.returncode == 0
        extracted_answer = output_payload.get("answer") == "4" if output_payload else False
        score_correct = output_payload.get("exact_match") == 1.0 if output_payload else False
        trajectory_tool_calls = output_payload.get("trajectory_tool_calls") if output_payload else None
        trajectory_captured = "calculator" in _tool_call_names(trajectory_tool_calls)
        agent_target_used = _uses_agent_as_run_sync_target(code) if code else False
        candidate_metadata_present = bool(
            output_payload
            and output_payload.get("candidate_agent_runtime")
            and output_payload.get("candidate_agent_model")
        )
        agent_target_terms_present = all(
            term in text_to_check
            for term in (
                "Agent",
                "AgentFormat.GENERIC",
                "body",
                "response_path",
                "$.answer",
                "trajectory_path",
                "$.trajectory",
            )
        )
        output_schema_valid = bool(
            code and code_ran and extracted_answer and score_correct and trajectory_captured and agent_target_used
        )
        task_success = bool(
            item.get("final_answer_extracted") is True
            and required_terms_present
            and agent_target_used
            and candidate_metadata_present
            and output_schema_valid
        )
        checks = [
            item.get("final_answer_extracted") is True,
            required_terms_present,
            code is not None,
            code_ran,
            agent_target_used,
            extracted_answer,
            score_correct,
            trajectory_captured,
            candidate_metadata_present,
        ]

        return MetricResult(
            scores=[
                MetricScore(name="task_success", value=float(task_success)),
                MetricScore(name="verification_score", value=score_checks(checks)),
                MetricScore(name="output_schema_valid", value=float(output_schema_valid)),
                MetricScore(name="code_block_present", value=float(code is not None)),
                MetricScore(name="code_ran", value=float(code_ran)),
                MetricScore(name="required_terms_present", value=float(required_terms_present)),
                MetricScore(name="agent_target_terms_present", value=float(agent_target_terms_present)),
                MetricScore(name="agent_target_used", value=float(agent_target_used)),
                MetricScore(name="extracted_answer", value=float(extracted_answer)),
                MetricScore(name="score_correct", value=float(score_correct)),
                MetricScore(name="trajectory_captured", value=float(trajectory_captured)),
                MetricScore(name="candidate_metadata_present", value=float(candidate_metadata_present)),
            ]
        )


def _extract_python_code(text: str) -> str | None:
    return extract_fenced_python_code(text, predicate=_looks_like_agent_target_code)


def _artifacts_from_item(item: dict) -> AgentArtifacts:
    agent_log_dir = item.get("agent_log_dir")
    if not isinstance(agent_log_dir, str):
        raise ValueError("agent_log_dir is required for agent-target metric")
    workspace_dir = item.get("workspace_dir")
    return AgentArtifacts.from_dir(
        agent_log_dir, workspace_dir=workspace_dir if isinstance(workspace_dir, str) else None
    )


def _required_terms(item: dict) -> list[str]:
    task_dir = item.get("task_dir")
    if not isinstance(task_dir, str):
        return []
    return load_agentic_use_task_config(Path(task_dir)).evaluator.expected.required_terms


def _extract_workspace_python_code(artifacts: AgentArtifacts) -> str | None:
    return find_workspace_python_code(
        artifacts,
        preferred_names=["agent_target_exact_match.py", "solution.py"],
        predicate=_looks_like_agent_target_code,
    )


def _run_candidate_code(code: str) -> subprocess.CompletedProcess[str]:
    return run_python_code(
        code,
        filename="candidate_agent_target.py",
        timeout=30,
        timeout_stderr="candidate agent-target code timed out",
    )


def _looks_like_agent_target_code(code: str) -> bool:
    return "Evaluator" in code and "Agent" in code and "run_sync" in code


def _uses_agent_as_run_sync_target(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    agent_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and _call_name(node.value) == "Agent":
            if not _agent_call_has_required_paths(node.value):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    agent_names.add(target.id)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node) != "run_sync":
            continue
        for keyword in node.keywords:
            if keyword.arg != "target":
                continue
            if isinstance(keyword.value, ast.Name) and keyword.value.id in agent_names:
                return True
            if isinstance(keyword.value, ast.Call) and _call_name(keyword.value) == "Agent":
                return _agent_call_has_required_paths(keyword.value)
    return False


def _agent_call_has_required_paths(call: ast.Call) -> bool:
    keyword_values = {
        keyword.arg: keyword.value.value
        for keyword in call.keywords
        if keyword.arg is not None and isinstance(keyword.value, ast.Constant)
    }
    return keyword_values.get("response_path") == "$.answer" and keyword_values.get("trajectory_path") == "$.trajectory"


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _extract_output_payload(stdout: str) -> dict[str, object] | None:
    return extract_json_object_from_stdout(stdout)


def _tool_call_names(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    names: list[str] = []
    for entry in value:
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, dict):
            name = object_dict(entry).get("name")
            if isinstance(name, str):
                names.append(name)
    return names
