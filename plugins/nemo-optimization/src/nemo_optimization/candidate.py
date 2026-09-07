# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared candidate-evaluation contracts for optimizer backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from nemo_evaluator_sdk.agent_eval.scores import AgentEvalScoreStatus, AgentEvalTaskScore

REASONING_OUTPUT_NAME = "reasoning"


class CandidateEvaluationError(RuntimeError):
    """Raised when candidate configuration or evaluation fails."""


class CandidateEvaluator(Protocol):
    """Evaluate one immutable optimizer candidate."""

    def evaluate(
        self,
        *,
        trial_number: int,
        suggestions: dict[str, Any],
        trial_overlay: dict[str, Any],
        rep: int,
    ) -> CandidateEvaluationResult:
        """Return aggregate metrics and row-level score records."""
        ...


@dataclass(frozen=True)
class RowReasoning:
    """Reasoning paired with the objective score from the same row score."""

    task_id: str
    metric_type: str
    objective_name: str
    objective_value: float
    reasoning: str


@dataclass(frozen=True)
class CandidateEvaluationResult:
    """Aggregate metrics plus raw row-level scores for one candidate."""

    aggregate_metrics: dict[str, float]
    scores: tuple[AgentEvalTaskScore, ...] = ()

    def reasoning_for_metric(self, metric_name: str) -> tuple[RowReasoning, ...]:
        """Return structured evaluator reasoning paired with *metric_name* values."""

        rows: list[RowReasoning] = []
        for score in self.scores:
            if score.status != AgentEvalScoreStatus.COMPLETED:
                continue
            outputs = _outputs_by_name(score)
            if metric_name not in outputs or REASONING_OUTPUT_NAME not in outputs:
                continue
            reasoning = outputs[REASONING_OUTPUT_NAME]
            if not isinstance(reasoning, str) or not reasoning.strip():
                continue
            try:
                objective_value = numeric_metric_value(outputs[metric_name], metric_name=metric_name)
            except CandidateEvaluationError:
                continue
            rows.append(
                RowReasoning(
                    task_id=score.task_id,
                    metric_type=score.metric_type,
                    objective_name=metric_name,
                    objective_value=objective_value,
                    reasoning=reasoning,
                )
            )
        return tuple(rows)


def numeric_metric_value(value: Any, *, metric_name: str) -> float:
    """Return a finite numeric metric value or raise a candidate error."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateEvaluationError(f"Metric output {metric_name!r} must be numeric.")
    try:
        numeric_value = float(value)
    except OverflowError as exc:
        raise CandidateEvaluationError(f"Metric output {metric_name!r} must be finite.") from exc
    if not math.isfinite(numeric_value):
        raise CandidateEvaluationError(f"Metric output {metric_name!r} must be finite.")
    return numeric_value


def _outputs_by_name(score: AgentEvalTaskScore) -> dict[str, Any]:
    return {output.name: output.value for output in score.outputs}
