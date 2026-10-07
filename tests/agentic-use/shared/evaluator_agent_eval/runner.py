# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Evals SDK scoring over normalized Evaluator rows."""

from collections.abc import Iterable, Sequence

from evaluator_agent_eval.metrics import default_agent_eval_metrics
from evaluator_agent_eval.schemas import EvaluatorScoringRow
from nhx_evals_sdk import Evaluator
from nhx_evals_sdk.metrics.base import Metric
from nhx_evals_sdk.values.multi_metric_results import BenchmarkEvaluationResult


def score_evaluator_rows(
    rows: Iterable[EvaluatorScoringRow],
    *,
    additional_metrics: Sequence[Metric] = (),
) -> BenchmarkEvaluationResult:
    """Score normalized Evaluator rows with SDK metrics."""
    dataset = [row.to_dataset_row() for row in rows]
    metrics = [*default_agent_eval_metrics(), *additional_metrics]
    return Evaluator().run_sync(metrics=metrics, dataset=dataset)
