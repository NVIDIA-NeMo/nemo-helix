# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declared metric outputs on each trial, used by agent-evaluation summaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from nemo_evaluator_sdk.agent_eval.scores import (
    AgentEvalScoreStatus,
    AgentEvalTaskScore,
    is_trial_failure,
)
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_evaluator_sdk.metrics.protocol import MetricOutput
from nemo_evaluator_sdk.metrics.utils import metric_type_name

#: What became of one declared output on one trial. Named because it is spelled at every producer
#: and consumer of an observation, and because "failed" alone would not say whose failure it was.
ObservationState = Literal["observed", "missing", "metric_failed", "trial_failed"]


@dataclass(frozen=True)
class OutputObservation:
    """One declared metric output on one trial, and whether it was scored."""

    task_id: str
    trial_id: str
    metric_type: str
    output_name: str
    state: ObservationState
    output: MetricOutput | None = None


def project_output_observations(
    scores: Sequence[AgentEvalTaskScore],
    tasks: Sequence[AgentEvalTask] | None,
) -> tuple[OutputObservation, ...]:
    """Expand each per-trial metric score into one row per applicable output name.

    A score record is one metric on one trial and may omit optional outputs. This function
    lists the names that apply, then records whether each name was scored:

    - ``observed`` — the output is present
    - ``missing`` — the output was applicable but not emitted
    - ``metric_failed`` / ``trial_failed`` — scoring did not run to completion

    With ``tasks``, applicable names are that metric's ``output_spec()`` on that task
    (undeclared names are ignored; a task that does not declare an output contributes
    no rows for it). Without ``tasks``, names are the union of outputs actually emitted
    by that metric type across the run, including on failed scores that emitted none.
    """
    names_by_metric: dict[str, set[str]] = {}
    if tasks is None:
        for score in scores:
            for output in score.outputs:
                names_by_metric.setdefault(score.metric_type, set()).add(output.name)
        applicable_by_task_metric: dict[tuple[str, str], tuple[str, ...]] | None = None
    else:
        applicable_by_task_metric = {}
        for task in tasks:
            for metric in task.metrics:
                applicable_by_task_metric[(task.id, metric_type_name(metric))] = tuple(
                    output.name for output in metric.output_spec()
                )

    observations: list[OutputObservation] = []
    for score in scores:
        if applicable_by_task_metric is None:
            output_names = tuple(sorted(names_by_metric.get(score.metric_type, ())))
        else:
            output_names = applicable_by_task_metric.get((score.task_id, score.metric_type), ())
        if not output_names:
            continue

        outputs: dict[str, MetricOutput] = {}
        for output in score.outputs:
            outputs.setdefault(output.name, output)

        for output_name in output_names:
            output = outputs.get(output_name)
            if is_trial_failure(score):
                state: ObservationState = "trial_failed"
            elif score.status is AgentEvalScoreStatus.FAILED:
                state = "metric_failed"
            elif output is None:
                state = "missing"
            else:
                state = "observed"
            observations.append(
                OutputObservation(
                    task_id=score.task_id,
                    trial_id=score.trial_id,
                    metric_type=score.metric_type,
                    output_name=output_name,
                    state=state,
                    output=output if state == "observed" else None,
                )
            )
    return tuple(observations)
