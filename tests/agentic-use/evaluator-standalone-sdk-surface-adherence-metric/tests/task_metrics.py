# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-specific metrics for the standalone SDK surface-adherence metric task."""

import asyncio
import subprocess
from pathlib import Path

from evaluator_agent_eval.artifacts import AgentArtifacts
from evaluator_agent_eval.task_config import load_agentic_use_task_config
from evaluator_agent_eval.task_metric_utils import (
    contains_all,
    extract_fenced_python_code,
    extract_marker_json_object,
    find_workspace_python_code,
    object_dict,
    run_python_code,
    score_checks,
    string_list,
)
from nhx_evals_sdk.values.results import MetricResult, MetricScore

HARNESS_RESULT_MARKER = "__SURFACE_METRIC_RESULT__="


class SurfaceAdherenceMetricAuthoringMetric:
    """Verify the candidate authored an SDK-compatible surface-adherence metric."""

    @property
    def type(self) -> str:
        return "agent_eval/surface_adherence_metric_authoring"

    def score_names(self) -> list[str]:
        return [
            "task_success",
            "verification_score",
            "output_schema_valid",
            "code_block_present",
            "code_ran",
            "required_terms_present",
            "protocol_compatible",
            "type_property_valid",
            "emits_expected_scores",
            "pass_case_correct",
            "fail_case_correct",
        ]

    async def compute_scores(self, item: dict, sample: dict) -> MetricResult:
        final_text = str(item.get("output_text", ""))
        artifacts = _artifacts_from_item(item)
        code = _extract_python_code(final_text) or _extract_workspace_python_code(artifacts)
        process = await asyncio.to_thread(_run_candidate_code_with_harness, code) if code else None
        stdout = process.stdout if process is not None else ""
        harness_result = _extract_harness_result(stdout)

        text_to_check = f"{final_text}\n{code or ''}"
        required_terms_present = contains_all(text_to_check, _required_terms(item))
        code_ran = process is not None and process.returncode == 0
        score_names = string_list(harness_result.get("score_names")) if harness_result else []
        pass_scores = object_dict(harness_result.get("pass_scores")) if harness_result else {}
        fail_scores = object_dict(harness_result.get("fail_scores")) if harness_result else {}

        protocol_compatible = harness_result.get("protocol_compatible") is True if harness_result else False
        metric_type = harness_result.get("metric_type") if harness_result else None
        type_property_valid = isinstance(metric_type, str) and bool(metric_type.strip())
        emits_expected_scores = "surface_adherence" in score_names and "surface_violation_count" in score_names
        pass_case_correct = (
            pass_scores.get("surface_adherence") == 1.0 and pass_scores.get("surface_violation_count") == 0.0
        )
        fail_adherence = fail_scores.get("surface_adherence")
        fail_violation_count = fail_scores.get("surface_violation_count")
        fail_case_correct = (
            isinstance(fail_adherence, int | float)
            and fail_adherence < 1.0
            and isinstance(fail_violation_count, int | float)
            and fail_violation_count > 0.0
        )
        output_schema_valid = bool(
            code
            and code_ran
            and protocol_compatible
            and type_property_valid
            and emits_expected_scores
            and pass_case_correct
            and fail_case_correct
        )
        task_success = bool(
            item.get("final_answer_extracted") is True and required_terms_present and output_schema_valid
        )
        checks = [
            item.get("final_answer_extracted") is True,
            required_terms_present,
            code is not None,
            code_ran,
            protocol_compatible,
            type_property_valid,
            emits_expected_scores,
            pass_case_correct,
            fail_case_correct,
        ]

        return MetricResult(
            scores=[
                MetricScore(name="task_success", value=float(task_success)),
                MetricScore(name="verification_score", value=score_checks(checks)),
                MetricScore(name="output_schema_valid", value=float(output_schema_valid)),
                MetricScore(name="code_block_present", value=float(code is not None)),
                MetricScore(name="code_ran", value=float(code_ran)),
                MetricScore(name="required_terms_present", value=float(required_terms_present)),
                MetricScore(name="protocol_compatible", value=float(protocol_compatible)),
                MetricScore(name="type_property_valid", value=float(type_property_valid)),
                MetricScore(name="emits_expected_scores", value=float(emits_expected_scores)),
                MetricScore(name="pass_case_correct", value=float(pass_case_correct)),
                MetricScore(name="fail_case_correct", value=float(fail_case_correct)),
            ]
        )


def _extract_python_code(text: str) -> str | None:
    return extract_fenced_python_code(text, predicate=_looks_like_surface_metric_code)


def _artifacts_from_item(item: dict) -> AgentArtifacts:
    agent_log_dir = item.get("agent_log_dir")
    if not isinstance(agent_log_dir, str):
        raise ValueError("agent_log_dir is required for surface-adherence metric")
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
        preferred_names=["surface_adherence_metric.py", "solution.py"],
        predicate=_looks_like_surface_metric_code,
    )


def _run_candidate_code_with_harness(code: str) -> subprocess.CompletedProcess[str]:
    return run_python_code(
        code,
        filename="candidate_surface_adherence_metric.py",
        timeout=20,
        appended_code=_HARNESS_CODE,
        timeout_stderr="candidate surface-adherence metric code timed out",
    )


def _looks_like_surface_metric_code(code: str) -> bool:
    return "compute_scores" in code and "MetricResult" in code and "MetricScore" in code


def _extract_harness_result(stdout: str) -> dict[str, object] | None:
    return extract_marker_json_object(stdout, marker=HARNESS_RESULT_MARKER)


_HARNESS_CODE = r"""
import asyncio
import inspect
import json

from nhx_evals_sdk.metrics.base import Metric


def __scores_to_dict(metric_result):
    return {score.name: float(score.value) for score in metric_result.scores}


async def __surface_metric_harness():
    metric_class = None
    for value in list(globals().values()):
        if not inspect.isclass(value):
            continue
        if value.__module__ != "__main__":
            continue
        if hasattr(value, "compute_scores") and hasattr(value, "score_names"):
            metric_class = value
            break

    if metric_class is None:
        raise RuntimeError("No candidate metric class with compute_scores and score_names was found")

    metric = metric_class()
    pass_result = await metric.compute_scores(
        {
            "observed_surfaces": ["standalone_sdk"],
            "allowed_surfaces": ["standalone_sdk"],
            "forbidden_surfaces": ["legacy_service"],
        },
        {},
    )
    fail_result = await metric.compute_scores(
        {
            "observed_surfaces": ["standalone_sdk", "legacy_service"],
            "allowed_surfaces": ["standalone_sdk"],
            "forbidden_surfaces": ["legacy_service"],
        },
        {},
    )
    print(
        "__SURFACE_METRIC_RESULT__="
        + json.dumps(
            {
                "class_name": metric_class.__name__,
                "protocol_compatible": isinstance(metric, Metric),
                "metric_type": metric.type,
                "score_names": list(metric.score_names()),
                "pass_scores": __scores_to_dict(pass_result),
                "fail_scores": __scores_to_dict(fail_result),
            },
            sort_keys=True,
        )
    )


asyncio.run(__surface_metric_harness())
"""
