# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decide whether an evaluation run produced a usable result, and report the counts to the job.

Both evaluator jobs tolerate per-unit failures (``ignore_request_failure``, failed metric scores) so
one bad row cannot abort a run. The cost is that a run where *nothing* scored still ends with every
artifact written. :class:`RunOutcome` is the rollup that tells the two apart: it is written to the
job's status details under :data:`STATUS_DETAILS_KEY` and decides the job's final status.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import math
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from nemo_evals.jobs.utils import run_with_isolated_async_client
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nhx_evals_sdk.agent_eval.results import AgentEvalResult
from nhx_evals_sdk.agent_eval.scores import AgentEvalScoreStatus
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrialStatus
from nhx_evals_sdk.values.multi_metric_results import BenchmarkEvaluationResult
from nhx_evals_sdk.values.protocol import MetricOutput
from nhx_evals_sdk.values.results import EvaluationResult, row_status
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

#: Key under which the outcome is merged into the job's ``status_details``.
STATUS_DETAILS_KEY = "evaluation"
PUBLISH_TIMEOUT_SEC = 15.0
"""How long a progress report is waited for; a slower one keeps going in the background."""
PUBLICATION_STATUS_KEY = "publication"
"""Where an agent evaluation that publishes to Intake records the publish outcome in its status details."""


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
    # Trial ids may repeat (Gym derives them from a rollout index), and a score names only the id,
    # so a usable score cannot be pinned to one duplicate. Cap scored trials per id at the number of
    # usable scores for that id: exact for one metric per trial, never above the trial count otherwise.
    usable_scores_by_id: Counter[str] = Counter(
        score.trial_id
        for score in result.scores
        if score.status is not AgentEvalScoreStatus.FAILED and _has_usable_value(score.outputs)
    )
    trials_by_id = Counter(trial.id for trial in result.trials)
    scored = sum(min(count, usable_scores_by_id[trial_id]) for trial_id, count in trials_by_id.items())
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
    report_status_details(outcome.status_details(), ctx=ctx, async_client=async_client)


def report_status_details(details: dict[str, object], *, ctx: JobContext, async_client: AsyncNemoClient | None) -> None:
    """Merge ``details`` into the platform job's status details. Best-effort; local runs skip it."""
    if ctx.job_id is None or async_client is None:
        return
    job_id = ctx.job_id
    attempt_id = ctx.attempt_id
    if attempt_id is None:
        logger.warning("Cannot report the evaluation outcome for job %r without an attempt ID", job_id)
        return

    async def _update(client: AsyncNemoClient) -> None:
        jobs = AsyncJobsClient.from_client(client)
        await jobs.update_attempt_status_details(
            job_id,
            attempt_id,
            workspace=ctx.workspace,
            body=details,
        )

    try:
        run_with_isolated_async_client(async_client, _update)
    except Exception:
        logger.warning("Failed to report status details %s for job %r", sorted(details), job_id, exc_info=True)


def status_details_writer(
    ctx: JobContext, async_client: AsyncNemoClient | None
) -> Callable[[dict[str, Any]], Awaitable[None]]:
    """Writes snapshots of the job's progress to its status details, in order, keeping only the newest queued one.

    Each snapshot replaces the one before under the same keys, so a snapshot still queued when a newer one
    arrives is skipped: at most one write is in flight and one waits, however slow the writes are. One thread
    keeps them in order; the write is synchronous, so it stays off the event loop.
    """
    writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="status-details")
    newest = itertools.count()
    latest = 0

    def write(ticket: int, details: dict[str, Any]) -> None:
        if ticket == latest:
            report_status_details(details, ctx=ctx, async_client=async_client)

    async def publish(details: dict[str, Any]) -> None:
        nonlocal latest
        latest = next(newest)
        # Shielded: when the caller stops waiting, the write keeps its place in line rather than being dropped.
        await asyncio.shield(asyncio.get_running_loop().run_in_executor(writer, write, latest, details))

    return publish


async def publish_best_effort(
    publish: Callable[[dict[str, Any]], Awaitable[None]], details: dict[str, Any], *, timeout: float
) -> None:
    """Publish ``details``, logging rather than raising when the write fails or outlasts ``timeout``."""
    try:
        async with asyncio.timeout(timeout):
            await publish(details)
    except TimeoutError:
        logger.warning("Reporting progress took over %ss; continuing", timeout)
    except Exception:
        logger.warning("Could not report progress", exc_info=True)


__all__ = [
    "PUBLICATION_STATUS_KEY",
    "PUBLISH_TIMEOUT_SEC",
    "STATUS_DETAILS_KEY",
    "RunOutcome",
    "agent_eval_outcome",
    "publish_best_effort",
    "report_run_outcome",
    "report_status_details",
    "row_eval_outcome",
    "status_details_writer",
]
