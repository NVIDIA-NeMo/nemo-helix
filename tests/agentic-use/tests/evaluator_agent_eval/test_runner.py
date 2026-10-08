# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluator_agent_eval.runner."""

from evaluator_agent_eval.runner import score_evaluator_rows
from evaluator_agent_eval.schemas import EvaluatorScoringRow
from nhx_evals_sdk.values.results import MetricResult, MetricScore


class ExtraMetric:
    @property
    def type(self) -> str:
        return "agent_eval/extra"

    def score_names(self) -> list[str]:
        return ["extra_score"]

    async def compute_scores(self, item: dict, sample: dict) -> MetricResult:
        return MetricResult(scores=[MetricScore(name="extra_score", value=1.0)])


def _row(**overrides: object) -> EvaluatorScoringRow:
    data = {
        "task_id": "run-simple-exact-match",
        "agent_runtime": "codex",
        "agent_model": "gpt-5.4",
        "surface_constraint": "standalone_sdk",
        "allowed_surfaces": ["standalone_sdk"],
        "forbidden_surfaces": ["legacy_service"],
        "output_text": "Used nhx_evals_sdk only.",
        "observed_surfaces": ["standalone_sdk"],
        "trajectory_summary": {"tool_call_count": 3, "failed_command_count": 1, "recovery_event_count": 1},
    }
    data.update(overrides)
    return EvaluatorScoringRow.model_validate(data)


def test_score_evaluator_rows_uses_evaluator_sdk():
    result = score_evaluator_rows([_row(), _row(output_text="Still used nhx_evals_sdk.")])

    assert len(result.row_scores) == 2
    aggregate_names = {score.name for score in result.aggregate_scores.scores}
    assert "agent_eval/surface_adherence.surface_adherence" in aggregate_names
    assert "agent_eval/legacy_surface_avoidance.legacy_surface_avoidance" in aggregate_names
    assert "agent_eval/trajectory_evidence.trajectory_present" in aggregate_names


def test_score_evaluator_rows_appends_additional_metrics():
    result = score_evaluator_rows([_row()], additional_metrics=[ExtraMetric()])

    aggregate_names = {score.name for score in result.aggregate_scores.scores}
    assert "agent_eval/surface_adherence.surface_adherence" in aggregate_names
    assert "agent_eval/extra.extra_score" in aggregate_names
