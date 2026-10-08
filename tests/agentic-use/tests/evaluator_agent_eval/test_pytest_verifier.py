# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluator_agent_eval.pytest_verifier."""

import pytest
from evaluator_agent_eval.pytest_verifier import _assert_evaluator_scores
from evaluator_agent_eval.runner import score_evaluator_rows
from evaluator_agent_eval.schemas import EvaluatorScoringRow
from nhx_evals_sdk.values.results import MetricResult, MetricScore


class ScriptedTaskMetric:
    """SDK metric used to verify task-specific metric normalization."""

    def __init__(
        self,
        *,
        metric_type: str = "agent_eval/task_specific",
        task_success: float = 1.0,
        verification_score: float = 1.0,
        output_schema_valid: float = 1.0,
    ) -> None:
        self._type = metric_type
        self._task_success = task_success
        self._verification_score = verification_score
        self._output_schema_valid = output_schema_valid

    @property
    def type(self) -> str:
        return self._type

    def score_names(self) -> list[str]:
        return ["task_success", "verification_score", "output_schema_valid"]

    async def compute_scores(self, item: dict, sample: dict) -> MetricResult:
        return MetricResult(
            scores=[
                MetricScore(name="task_success", value=self._task_success),
                MetricScore(name="verification_score", value=self._verification_score),
                MetricScore(name="output_schema_valid", value=self._output_schema_valid),
            ]
        )


class MissingRequiredScoreMetric:
    """SDK metric that omits the required task scores."""

    @property
    def type(self) -> str:
        return "agent_eval/missing_required_score"

    def score_names(self) -> list[str]:
        return ["unrelated_score"]

    async def compute_scores(self, item: dict, sample: dict) -> MetricResult:
        return MetricResult(scores=[MetricScore(name="unrelated_score", value=1.0)])


def _row() -> EvaluatorScoringRow:
    return EvaluatorScoringRow.model_validate(
        {
            "task_id": "task",
            "agent_runtime": "codex",
            "agent_model": "gpt-5.4",
            "surface_constraint": "standalone_sdk",
            "allowed_surfaces": ["standalone_sdk"],
            "forbidden_surfaces": ["legacy_service"],
            "output_text": "Used nhx_evals_sdk only.",
            "observed_surfaces": ["standalone_sdk"],
        }
    )


def test_task_specific_metrics_run_through_evaluator_sdk():
    task_metrics = [
        ScriptedTaskMetric(metric_type="agent_eval/task_specific_a"),
        ScriptedTaskMetric(metric_type="agent_eval/task_specific_b"),
    ]
    scored = score_evaluator_rows([_row()], additional_metrics=task_metrics)

    _assert_evaluator_scores(scored)

    aggregate_names = {score.name for score in scored.aggregate_scores.scores}
    assert "agent_eval/surface_adherence.surface_adherence" in aggregate_names
    assert "agent_eval/legacy_surface_avoidance.legacy_surface_avoidance" in aggregate_names
    assert "agent_eval/task_specific_a.task_success" in aggregate_names
    assert "agent_eval/task_specific_b.verification_score" in aggregate_names


def test_task_specific_metrics_require_sdk_scores():
    task_metrics = [MissingRequiredScoreMetric()]
    scored = score_evaluator_rows([_row()], additional_metrics=task_metrics)

    with pytest.raises(AssertionError, match="did not emit required scores"):
        _assert_evaluator_scores(scored)
