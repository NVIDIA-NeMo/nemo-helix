# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluator_agent_eval.metrics."""

from math import isnan

import pytest
from evaluator_agent_eval.metrics import (
    LegacySurfaceAvoidanceMetric,
    SurfaceAdherenceMetric,
    TrajectoryEvidenceMetric,
    default_agent_eval_metrics,
)
from evaluator_agent_eval.schemas import EvaluatorScoringRow
from nhx_evals_sdk.metrics.base import Metric

SURFACE_FIELD_KEYS = {
    "observed_surfaces_key": "observed_surfaces",
    "allowed_surfaces_key": "allowed_surfaces",
    "forbidden_surfaces_key": "forbidden_surfaces",
    "forbidden_surface_hits_key": "forbidden_surface_hits",
}
TRAJECTORY_FIELD_KEYS = {
    "trajectory_summary_key": "trajectory_summary",
    "tool_call_count_key": "tool_call_count",
    "failed_command_count_key": "failed_command_count",
    "recovery_event_count_key": "recovery_event_count",
}


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


def _scores_by_name(result) -> dict[str, float]:
    return {score.name: score.value for score in result.scores}


@pytest.mark.asyncio
async def test_surface_adherence_penalizes_forbidden_surface_hits():
    metric = SurfaceAdherenceMetric(**SURFACE_FIELD_KEYS)
    result = await metric.compute_scores(
        _row(
            observed_surfaces=["standalone_sdk", "legacy_service"], forbidden_surface_hits=["services/"]
        ).to_dataset_row(),
        {},
    )

    scores = _scores_by_name(result)
    assert scores["surface_adherence"] == 0.0
    assert scores["surface_violation_count"] == 2.0


@pytest.mark.asyncio
async def test_legacy_surface_avoidance_scans_output_text():
    metric = LegacySurfaceAvoidanceMetric(
        observed_surfaces_key="observed_surfaces",
        forbidden_surface_hits_key="forbidden_surface_hits",
        output_text_key="output_text",
        legacy_surface="legacy_service",
        legacy_text_patterns=("services/", "services\\"),
    )
    result = await metric.compute_scores(
        _row(output_text="I inspected services/evaluator to solve it.").to_dataset_row(), {}
    )

    scores = _scores_by_name(result)
    assert scores["legacy_surface_avoidance"] == 0.0
    assert scores["legacy_surface_hit_count"] == 1.0


@pytest.mark.asyncio
async def test_legacy_surface_avoidance_ignores_negated_mentions():
    metric = LegacySurfaceAvoidanceMetric(
        observed_surfaces_key="observed_surfaces",
        forbidden_surface_hits_key="forbidden_surface_hits",
        output_text_key="output_text",
        legacy_surface="legacy_service",
        legacy_text_patterns=("services/", "services\\"),
    )
    result = await metric.compute_scores(
        _row(output_text="Do not use services/evaluator or any legacy services path.").to_dataset_row(),
        {},
    )

    scores = _scores_by_name(result)
    assert scores["legacy_surface_avoidance"] == 1.0
    assert scores["legacy_surface_hit_count"] == 0.0


@pytest.mark.asyncio
async def test_surface_metrics_can_use_configured_field_keys():
    metric = SurfaceAdherenceMetric(
        observed_surfaces_key="actual",
        allowed_surfaces_key="allowed",
        forbidden_surfaces_key="blocked",
        forbidden_surface_hits_key="blocked_hits",
    )
    result = await metric.compute_scores(
        {
            "actual": ["standalone_sdk", "legacy_service"],
            "allowed": ["standalone_sdk"],
            "blocked": ["legacy_service"],
            "blocked_hits": ["services/"],
        },
        {},
    )

    scores = _scores_by_name(result)
    assert scores["surface_adherence"] == 0.0
    assert scores["surface_violation_count"] == 2.0


@pytest.mark.asyncio
async def test_surface_adherence_returns_nan_for_malformed_surface_fields():
    metric = SurfaceAdherenceMetric(**SURFACE_FIELD_KEYS)
    result = await metric.compute_scores(
        _row().to_dataset_row() | {"observed_surfaces": "standalone_sdk"},
        {},
    )

    scores = _scores_by_name(result)
    assert isnan(scores["surface_adherence"])
    assert isnan(scores["surface_violation_count"])


@pytest.mark.asyncio
async def test_legacy_surface_avoidance_returns_nan_for_malformed_fields():
    metric = LegacySurfaceAvoidanceMetric(
        observed_surfaces_key="observed_surfaces",
        forbidden_surface_hits_key="forbidden_surface_hits",
        output_text_key="output_text",
        legacy_surface="legacy_service",
        legacy_text_patterns=("services/", "services\\"),
    )
    result = await metric.compute_scores(
        _row().to_dataset_row() | {"output_text": 123},
        {},
    )

    scores = _scores_by_name(result)
    assert isnan(scores["legacy_surface_avoidance"])
    assert isnan(scores["legacy_surface_hit_count"])


def test_metrics_implement_sdk_protocol():
    for metric in default_agent_eval_metrics():
        assert isinstance(metric, Metric)


@pytest.mark.asyncio
async def test_trajectory_metric_requires_summary_not_just_path():
    metric = TrajectoryEvidenceMetric(**TRAJECTORY_FIELD_KEYS)
    with_summary = await metric.compute_scores(_row().to_dataset_row(), {})
    path_only = await metric.compute_scores(
        _row(trajectory_summary=None, atif_trajectory_path="/tmp/trajectory.json").to_dataset_row(),
        {},
    )

    assert _scores_by_name(with_summary)["trajectory_present"] == 1.0
    assert _scores_by_name(path_only)["trajectory_present"] == 0.0


@pytest.mark.asyncio
async def test_trajectory_metric_can_use_configured_field_keys():
    metric = TrajectoryEvidenceMetric(
        trajectory_summary_key="trace_counts",
        tool_call_count_key="tools",
        failed_command_count_key="failures",
        recovery_event_count_key="recoveries",
    )
    result = await metric.compute_scores(
        {"trace_counts": {"tools": 2, "failures": 1, "recoveries": 1}},
        {},
    )

    scores = _scores_by_name(result)
    assert scores["trajectory_present"] == 1.0
    assert scores["tool_call_count"] == 2.0
    assert scores["failed_command_count"] == 1.0
    assert scores["recovery_event_count"] == 1.0


@pytest.mark.asyncio
async def test_trajectory_metric_returns_nan_for_malformed_summary():
    metric = TrajectoryEvidenceMetric(**TRAJECTORY_FIELD_KEYS)
    result = await metric.compute_scores(
        _row().to_dataset_row() | {"trajectory_summary": {"tool_call_count": -1}},
        {},
    )

    scores = _scores_by_name(result)
    assert isnan(scores["trajectory_present"])
    assert isnan(scores["tool_call_count"])
    assert isnan(scores["failed_command_count"])
    assert isnan(scores["recovery_event_count"])
