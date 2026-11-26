# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report each Harbor trial's progress while a job runs, so a task Job's trials can be watched live.

The reporter follows Harbor's trial lifecycle hooks and, after each event, reports the trials counted by phase, and
optionally the whole trial list (see :mod:`nemo_evals.taskset_runs.trials`), through a ``publish`` callback: the
job's status details. Reports go out in the background, so a slow status write never holds up a trial; they are best
effort and never fail the job. The trial results Harbor writes stay authoritative.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from harbor.job import CANCELLED_ERROR_TYPE, Job
from harbor.models.job.config import RetryConfig
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import ExceptionInfo, TrialResult
from harbor.trial.hooks import TrialHookEvent
from nemo_evals.jobs.run_outcome import PUBLISH_TIMEOUT_SEC, publish_best_effort
from nemo_evals.taskset_runs.trials import (
    SETTLED_TRIAL_PHASES,
    TRIAL_COUNTS_STATUS_KEY,
    TRIALS_STATUS_KEY,
    TrialPhase,
    TrialProgress,
    count_trials,
)
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborJobScope
from nhx_evals_sdk.agent_eval.runtimes.harbor.trial_adapter import with_sandbox_terminations
from nhx_evals_sdk.agent_eval.trials import TrialError

logger = logging.getLogger(__name__)

MAX_ERROR_CHARS = 2000
"""Longest error message kept per trial. Errors are rare and their details are what debugging needs, so it's generous."""

Publish = Callable[[dict[str, Any]], Awaitable[None]]


def will_retry(retry: RetryConfig, error: ExceptionInfo | None, attempts: int) -> bool:
    """Whether Harbor runs a trial again after an attempt, by the rule its trial queue applies.

    ``attempts`` counts the attempts that ended, this one included.
    """
    if error is None:
        return False
    if retry.exclude_exceptions and error.exception_type in retry.exclude_exceptions:
        return False
    if retry.include_exceptions and error.exception_type not in retry.include_exceptions:
        return False
    return attempts <= retry.max_retries


def trial_error(result: TrialResult) -> ExceptionInfo | None:
    """The attempt's error; a multi-step trial records it on the step that raised."""
    if result.exception_info is not None:
        return result.exception_info
    return next((step.exception_info for step in result.step_results or () if step.exception_info is not None), None)


def reported_error(result: TrialResult) -> TrialError | None:
    """The attempt's error as its final result reports it, so live progress and results agree on the type."""
    error = trial_error(result)
    if error is None:
        return None
    harbor = TrialError(type=error.exception_type, message=error.exception_message)
    uri = urlparse(result.trial_uri)
    return with_sandbox_terminations(harbor, Path(url2pathname(uri.path))) if uri.scheme == "file" else harbor


def trial_reward(result: TrialResult, reward_key: str) -> float | None:
    """The attempt's primary reward: the verifier's ``reward_key`` and nothing else.

    A missing key is no reward, even when the verifier emitted one differently named: the final evaluation
    selects the primary reward the same way, and a live score that final results then lack would mislead.
    """
    verifier = result.verifier_result
    rewards = verifier.rewards if verifier is not None else None
    if not rewards:
        return None

    value = rewards.get(reward_key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


class TrialProgressReporter:
    """Tracks one Harbor job's trials through their lifecycle hooks and reports them after each change.

    Every report carries the trials counted by phase. ``report_trials`` adds the full trial list, which a task
    Job's readers show trial by trial; a job running thousands of trials leaves it off, since each report
    rewrites it whole. ``session_id`` names each trial's Intake session, so readers can link a settled trial to
    its traces.

    Harbor waits for each hook before the trial moves on, so a hook only records the change and schedules its
    report; reports start in the order they were made. The job's scope waits for them when it exits, so the last
    counts are written before the job ends.
    """

    def __init__(
        self,
        publish: Publish,
        *,
        reward_key: str,
        session_id: Callable[[str], str] | None = None,
        report_trials: bool = True,
        publish_timeout: float = PUBLISH_TIMEOUT_SEC,
    ) -> None:
        self._publish = publish
        self._reward_key = reward_key
        self._report_trials = report_trials
        self._session_id = session_id
        self._publish_timeout = publish_timeout
        self._trials: dict[str, TrialProgress] = {}
        self._ended: dict[str, int] = {}
        self._retry = RetryConfig()
        self._reports: list[asyncio.Task[None]] = []

    def job_scope(self) -> HarborJobScope:
        """The scope a Harbor runner enters around each job's ``run()``."""
        return self._watching

    @property
    def trials(self) -> list[TrialProgress]:
        return list(self._trials.values())

    @contextlib.asynccontextmanager
    async def _watching(self, job: Job) -> AsyncGenerator[None]:
        self._retry = job.config.retry

        # Harbor plans its trials in Job.create and keeps them in private attributes. A resumed job (a rerun or
        # resumed task Job on the same storage) runs only the unfinished ones; report the finished ones too, or the
        # task's trials would shrink to what's left.
        for result in job._existing_trial_results:
            self._finished(result)
        for config in job._remaining_trial_configs:
            self._plan(config)

        job.on_trial_started(self._on(TrialPhase.STARTING))
        job.on_agent_started(self._on(TrialPhase.AGENT_RUNNING))
        job.on_verification_started(self._on(TrialPhase.VERIFYING))
        job.on_trial_ended(self._ended_attempt)
        job.on_trial_cancelled(self._cancelled)
        self._report()

        try:
            yield
        finally:
            # A task Job that stopped or raised ends no trial: whatever hadn't settled never will.
            now = datetime.now(UTC)
            stopped = {
                name: trial.model_copy(update={"phase": TrialPhase.CANCELLED, "finished_at": now})
                for name, trial in self._trials.items()
                if trial.phase not in SETTLED_TRIAL_PHASES
            }

            if stopped:
                self._trials.update(stopped)
                self._report()

            reported = asyncio.gather(*self._reports)
            try:
                await asyncio.shield(reported)
            except asyncio.CancelledError:
                await reported
                raise

    def _plan(self, config: TrialConfig) -> None:
        name = config.trial_name
        self._trials.setdefault(
            name,
            TrialProgress(
                trial=name,
                session_id=self._session_id(name) if self._session_id is not None else None,
                phase=TrialPhase.PENDING,
            ),
        )

    def _finished(self, result: TrialResult) -> None:
        """Record a trial an earlier run of this job already settled, from the result Harbor kept."""
        error = trial_error(result)
        reported = reported_error(result)
        if error is None:
            phase = TrialPhase.SUCCEEDED
        elif error.exception_type == CANCELLED_ERROR_TYPE:
            phase = TrialPhase.CANCELLED  # Harbor records a cancelled trial as one that raised CancelledError
        else:
            phase = TrialPhase.ERRORED

        self._trials[result.trial_name] = TrialProgress(
            trial=result.trial_name,
            session_id=self._session_id(result.trial_name) if self._session_id is not None else None,
            phase=phase,
            attempts=1,
            started_at=result.started_at,
            finished_at=result.finished_at,
            reward=trial_reward(result, self._reward_key),
            error_type=reported.type if reported is not None else None,
            error=reported.message[:MAX_ERROR_CHARS] if reported is not None and reported.message else None,
        )

    def _on(self, phase: TrialPhase) -> Callable[[TrialHookEvent], Awaitable[None]]:
        async def hook(event: TrialHookEvent) -> None:
            trial = self._trial(event)
            update: dict[str, object] = {"phase": phase}
            if phase is TrialPhase.STARTING:
                update["attempts"] = trial.attempts + 1
                update["started_at"] = trial.started_at or event.result.started_at or event.timestamp

            self._trials[trial.trial] = trial.model_copy(update=update)
            self._report()

        return hook

    async def _ended_attempt(self, event: TrialHookEvent) -> None:
        trial = self._trial(event)
        self._ended[trial.trial] = self._ended.get(trial.trial, 0) + 1
        error = trial_error(event.result)
        reported = reported_error(event.result)

        if trial.phase is TrialPhase.CANCELLED:
            phase = TrialPhase.CANCELLED
        # Harbor retries on the trial's own exception only. A multi-step trial's error can sit on a step alone, which
        # Harbor doesn't retry, so predict the retry from the top-level one but report the step's error.
        elif will_retry(self._retry, event.result.exception_info, self._ended[trial.trial]):
            phase = TrialPhase.RETRYING
        else:
            phase = TrialPhase.ERRORED if error is not None else TrialPhase.SUCCEEDED

        settled = phase in SETTLED_TRIAL_PHASES
        self._trials[trial.trial] = trial.model_copy(
            update={
                "phase": phase,
                "finished_at": (event.result.finished_at or event.timestamp) if settled else None,
                "reward": trial_reward(event.result, self._reward_key),
                "error_type": reported.type if reported is not None else None,
                "error": reported.message[:MAX_ERROR_CHARS] if reported is not None and reported.message else None,
            }
        )

        self._report()

    async def _cancelled(self, event: TrialHookEvent) -> None:
        trial = self._trial(event)
        self._trials[trial.trial] = trial.model_copy(
            update={"phase": TrialPhase.CANCELLED, "finished_at": event.timestamp}
        )
        self._report()

    def _trial(self, event: TrialHookEvent) -> TrialProgress:
        """The tracked trial of ``event``, planning it first if Harbor runs a trial it didn't plan."""
        self._plan(event.config)
        return self._trials[event.config.trial_name]

    def _report(self) -> None:
        """Report every trial as it stands now, in the background; a slow or failed report is logged, never raised."""
        trials = list(self._trials.values())
        details: dict[str, Any] = {TRIAL_COUNTS_STATUS_KEY: count_trials(trials).model_dump(mode="json")}
        if self._report_trials:
            details[TRIALS_STATUS_KEY] = [trial.model_dump(mode="json") for trial in trials]
        self._reports = [report for report in self._reports if not report.done()]
        self._reports.append(
            asyncio.ensure_future(publish_best_effort(self._publish, details, timeout=self._publish_timeout))
        )
