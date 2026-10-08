# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest verifier for Evaluator agent benchmark tasks."""

import json
import os
from pathlib import Path
from typing import Sequence

from evaluator_agent_eval.artifacts import AgentArtifacts
from evaluator_agent_eval.factory import AgentRunMetadata, build_evaluator_scoring_row, capture_agent_attempt
from evaluator_agent_eval.runner import score_evaluator_rows
from evaluator_agent_eval.task_config import load_agentic_use_task_config
from nhx_evals_sdk.metrics.base import Metric
from nhx_evals_sdk.values.multi_metric_results import BenchmarkEvaluationResult

AGENT_LOG_DIR = Path("/logs/agent")
VERIFIER_LOG_DIR = Path("/logs/verifier")
TASK_DIR = Path(os.environ.get("AGENTIC_USE_TASK_DIR", "/task"))
WORKSPACE_DIR = Path(os.environ.get("AGENTIC_USE_WORKSPACE_DIR", "/app/workspace"))
TASK_SUCCESS_SCORE = "task_success"
VERIFICATION_SCORE = "verification_score"
OUTPUT_SCHEMA_VALID_SCORE = "output_schema_valid"


def verify_agent_attempt_scores_with_evaluator_sdk(*, task_specific_metrics: Sequence[Metric]) -> None:
    """Verify one captured agent attempt with the shared Evaluator benchmark metrics."""
    task_config = load_agentic_use_task_config(TASK_DIR)
    artifacts = AgentArtifacts.from_dir(AGENT_LOG_DIR, workspace_dir=WORKSPACE_DIR)
    attempt = capture_agent_attempt(
        task_dir=TASK_DIR,
        artifacts=artifacts,
        metadata=AgentRunMetadata(
            agent_runtime=os.environ.get("NAT_AGENT_BACKEND", "unknown"),
            agent_model=os.environ.get("NAT_AGENT_MODEL", "unknown"),
        ),
    )
    scoring_row = build_evaluator_scoring_row(
        task_dir=TASK_DIR,
        attempt=attempt,
        artifacts=artifacts,
        task_config=task_config,
    )

    scored = score_evaluator_rows(
        [scoring_row],
        additional_metrics=task_specific_metrics,
    )
    _write_evaluator_scores(scored=scored)
    _assert_evaluator_scores(scored)


def _assert_evaluator_scores(scored: BenchmarkEvaluationResult) -> None:
    if scored.row_scores and scored.row_scores[0].error:
        raise AssertionError(f"Evaluator metric failed: {scored.row_scores[0].error}")

    score_values = _task_score_values(scored)
    missing = [
        name for name in (TASK_SUCCESS_SCORE, VERIFICATION_SCORE, OUTPUT_SCHEMA_VALID_SCORE) if not score_values[name]
    ]
    if missing:
        raise AssertionError(f"Task-specific Evaluator metrics did not emit required scores: {', '.join(missing)}")

    aggregate_scores = _aggregate_scores(scored)
    assert all(value == 1.0 for value in score_values[TASK_SUCCESS_SCORE])
    assert sum(score_values[VERIFICATION_SCORE]) / len(score_values[VERIFICATION_SCORE]) == 1.0
    assert all(value == 1.0 for value in score_values[OUTPUT_SCHEMA_VALID_SCORE])
    assert aggregate_scores["agent_eval/surface_adherence.surface_adherence"] == 1.0
    assert aggregate_scores["agent_eval/legacy_surface_avoidance.legacy_surface_avoidance"] == 1.0


def _task_score_values(scored: BenchmarkEvaluationResult) -> dict[str, list[float]]:
    score_values: dict[str, list[float]] = {
        TASK_SUCCESS_SCORE: [],
        VERIFICATION_SCORE: [],
        OUTPUT_SCHEMA_VALID_SCORE: [],
    }
    if not scored.row_scores:
        return score_values

    for metric_scores in scored.row_scores[0].metrics.values():
        for score in metric_scores:
            if score.name in score_values:
                score_values[score.name].append(float(score.value))
    return score_values


def _aggregate_scores(scored: BenchmarkEvaluationResult) -> dict[str, float]:
    return {str(score.name): float(score.mean) for score in scored.aggregate_scores.scores if score.mean is not None}


def _write_evaluator_scores(*, scored: BenchmarkEvaluationResult) -> None:
    VERIFIER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "aggregate_scores": [
            {"name": name, "mean": value} for name, value in sorted(_aggregate_scores(scored).items())
        ],
    }
    (VERIFIER_LOG_DIR / "evaluator_scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
