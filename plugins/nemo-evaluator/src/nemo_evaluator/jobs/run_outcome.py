# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decide whether an evaluation run produced a usable result, and report the counts to the job.

Both evaluator jobs tolerate per-unit failures (``ignore_request_failure``, failed metric scores) so
one bad row cannot abort a run. The cost is that a run where *nothing* scored still ends with every
artifact written. :class:`RunOutcome` is the rollup that tells the two apart: it is written to the
job's status details under :data:`STATUS_DETAILS_KEY` and decides the job's final status.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Literal

from nemo_evaluator.jobs.utils import run_with_isolated_async_client
from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult
from nemo_evaluator_sdk.agent_eval.scores import AgentEvalScoreStatus
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrialStatus
from nemo_evaluator_sdk.values.multi_metric_results import BenchmarkEvaluationResult
from nemo_evaluator_sdk.values.protocol import MetricOutput
from nemo_evaluator_sdk.values.results import EvaluationResult, row_status
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

#: Key under which the outcome is merged into the job's ``status_details``.
STATUS_DETAILS_KEY = "evaluation"


class RunOutcome(BaseModel):
    """Did the run score anything? Counts over its units (agent-eval trials or row-eval rows)."""

    model_config = ConfigDict(extra="forbid")

    unit: Literal["trials", "rows"] = Field(description="What ``total``, ``errored`` and ``scored`` count.")
    total: int = Field(ge=0, description="Units the run produced.")
    errored: int = Field(ge=0, description="Units that reported an error while being produced or scored.")
    scored: int = Field(ge=0, description="Units with at least one usable (non-failed, non-NaN) metric value.")

    @property
    def failed(self) -> bool:
        """True when the run produced units but none of them scored."""
        return self.total > 0 and self.scored == 0

    @property
    def message(self) -> str:
        if self.failed:
            return (
                f"No usable scores across {self.total} {self.unit} ({self.errored} reported errors). "
                "See the job logs for the underlying request or metric failures."
            )
        return f"{self.scored} of {self.total} {self.unit} scored; {self.errored} reported errors."

    def details(self) -> dict[str, object]:
        return {**self.model_dump(), "failed": self.failed, "message": self.message}

    def status_details(self) -> dict[str, object]:
        return {STATUS_DETAILS_KEY: self.details()}


def agent_eval_outcome(result: AgentEvalResult) -> RunOutcome:
    """Trials count as errored when they failed to produce or carry a producer error."""
    errored = sum(
        1 for trial in result.trials if trial.status is AgentEvalTrialStatus.FAILED or trial.error is not None
    )
    scored_trial_ids = {
        score.trial_id
        for score in result.scores
        if score.status is not AgentEvalScoreStatus.FAILED and _has_usable_value(score.outputs)
    }
    scored = sum(1 for trial in result.trials if trial.id in scored_trial_ids)
    return RunOutcome(unit="trials", total=len(result.trials), errored=errored, scored=scored)


def row_eval_outcome(result: EvaluationResult | BenchmarkEvaluationResult) -> RunOutcome:
    """Rows count as errored when inference or any metric recorded an error for them."""
    rows = result.row_scores
    errored = sum(1 for row in rows if row_status(row) == "error" or "inference_error" in row.sample)
    scored = sum(1 for row in rows if any(_has_usable_value(outputs) for outputs in row.metrics.values()))
    return RunOutcome(unit="rows", total=len(rows), errored=errored, scored=scored)


def _has_usable_value(outputs: Iterable[MetricOutput]) -> bool:
    for output in outputs:
        value = output.value
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        return True
    return False


def report_run_outcome(outcome: RunOutcome, *, ctx: JobContext, async_client: AsyncNemoClient | None) -> None:
    """Merge the outcome into the platform job's status details. Best-effort; local runs skip it."""
    if ctx.job_id is None or async_client is None:
        return
    job_id = ctx.job_id

    async def _update(client: AsyncNemoClient) -> None:
        await AsyncJobsClient.from_client(client).update_status_details(
            job_id, workspace=ctx.workspace, body=outcome.status_details()
        )

    try:
        run_with_isolated_async_client(async_client, _update)
    except Exception:
        logger.warning("Failed to report the evaluation outcome for job %r", job_id, exc_info=True)


__all__ = [
    "STATUS_DETAILS_KEY",
    "RunOutcome",
    "agent_eval_outcome",
    "report_run_outcome",
    "row_eval_outcome",
]
