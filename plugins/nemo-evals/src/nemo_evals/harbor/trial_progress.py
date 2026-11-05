# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report each Harbor trial's progress while a job runs, so a task Job's trials can be watched live.

The reporter follows Harbor's trial lifecycle hooks and, after each event, reports the whole trial list (see
:mod:`nemo_evals.taskset_runs.trials`) through a ``publish`` callback: the task Job's status details. Reports are
best effort and never fail or stall the job for long; the trial results Harbor writes stay authoritative.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from harbor.job import Job
from harbor.models.job.config import RetryConfig
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import ExceptionInfo, TrialResult
from harbor.trial.hooks import TrialHookEvent
from nemo_evals.taskset_runs.trials import (
    SETTLED_TRIAL_PHASES,
    TRIALS_STATUS_KEY,
    TrialPhase,
    TrialProgress,
)
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborJobScope

logger = logging.getLogger(__name__)

PUBLISH_TIMEOUT_SEC = 15.0
"""How long a trial event waits for its report; a slower report keeps going in the background."""
MAX_ERROR_CHARS = 500

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


def trial_reward(result: TrialResult, reward_key: str) -> float | None:
    """The attempt's primary reward: ``reward_key`` from the verifier's rewards, or their only value."""
    verifier = result.verifier_result
    rewards = verifier.rewards if verifier is not None else None
    if not rewards:
        return None
    value = rewards.get(reward_key)
    if value is None and len(rewards) == 1:
        value = next(iter(rewards.values()))
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


class TrialProgressReporter:
    """Tracks one Harbor job's trials through their lifecycle hooks and reports them after each change.

    ``session_id`` names each trial's Intake session, so readers can link a settled trial to its traces.
    """

    def __init__(
        self,
        publish: Publish,
        *,
        reward_key: str,
        session_id: Callable[[str], str] | None = None,
        publish_timeout: float = PUBLISH_TIMEOUT_SEC,
    ) -> None:
        self._publish = publish
        self._reward_key = reward_key
        self._session_id = session_id
        self._publish_timeout = publish_timeout
        self._trials: dict[str, TrialProgress] = {}
        self._ended: dict[str, int] = {}
        self._retry = RetryConfig()
        self._lock = asyncio.Lock()

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
        await self._report()
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
                report = asyncio.ensure_future(self._report())
                try:
                    await asyncio.shield(report)
                except asyncio.CancelledError:
                    await report
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
        self._trials[result.trial_name] = TrialProgress(
            trial=result.trial_name,
            session_id=self._session_id(result.trial_name) if self._session_id is not None else None,
            phase=TrialPhase.ERRORED if error is not None else TrialPhase.SUCCEEDED,
            attempts=1,
            started_at=result.started_at,
            finished_at=result.finished_at,
            reward=trial_reward(result, self._reward_key),
            error_type=error.exception_type if error is not None else None,
            error=error.exception_message[:MAX_ERROR_CHARS] if error is not None else None,
        )

    def _on(self, phase: TrialPhase) -> Callable[[TrialHookEvent], Awaitable[None]]:
        async def hook(event: TrialHookEvent) -> None:
            trial = self._trial(event)
            update: dict[str, object] = {"phase": phase}
            if phase is TrialPhase.STARTING:
                update["attempts"] = trial.attempts + 1
                update["started_at"] = trial.started_at or event.result.started_at or event.timestamp
            self._trials[trial.trial] = trial.model_copy(update=update)
            await self._report()

        return hook

    async def _ended_attempt(self, event: TrialHookEvent) -> None:
        trial = self._trial(event)
        self._ended[trial.trial] = self._ended.get(trial.trial, 0) + 1
        error = trial_error(event.result)
        if trial.phase is TrialPhase.CANCELLED:
            phase = TrialPhase.CANCELLED
        elif will_retry(self._retry, error, self._ended[trial.trial]):
            phase = TrialPhase.RETRYING
        else:
            phase = TrialPhase.ERRORED if error is not None else TrialPhase.SUCCEEDED
        settled = phase in SETTLED_TRIAL_PHASES
        self._trials[trial.trial] = trial.model_copy(
            update={
                "phase": phase,
                "finished_at": (event.result.finished_at or event.timestamp) if settled else None,
                "reward": trial_reward(event.result, self._reward_key),
                "error_type": error.exception_type if error is not None else None,
                "error": error.exception_message[:MAX_ERROR_CHARS] if error is not None else None,
            }
        )
        await self._report()

    async def _cancelled(self, event: TrialHookEvent) -> None:
        trial = self._trial(event)
        self._trials[trial.trial] = trial.model_copy(
            update={"phase": TrialPhase.CANCELLED, "finished_at": event.timestamp}
        )
        await self._report()

    def _trial(self, event: TrialHookEvent) -> TrialProgress:
        """The tracked trial of ``event``, planning it first if Harbor runs a trial it didn't plan."""
        self._plan(event.config)
        return self._trials[event.config.trial_name]

    async def _report(self) -> None:
        """Report every trial as it stands now; a slow or failed report is logged, never raised."""
        async with self._lock:
            details = {TRIALS_STATUS_KEY: [trial.model_dump(mode="json") for trial in self._trials.values()]}
            try:
                async with asyncio.timeout(self._publish_timeout):
                    await self._publish(details)
            except TimeoutError:
                logger.warning("Reporting trial progress took over %ss; continuing", self._publish_timeout)
            except Exception:
                logger.warning("Could not report trial progress", exc_info=True)


def combined_job_scope(*scopes: HarborJobScope | None) -> HarborJobScope | None:
    """One job scope entering each of ``scopes`` in order around a Harbor job; None when every scope is None."""
    present = [scope for scope in scopes if scope is not None]
    if not present:
        return None
    if len(present) == 1:
        return present[0]

    @contextlib.asynccontextmanager
    async def combined(job: Job) -> AsyncGenerator[None]:
        async with contextlib.AsyncExitStack() as stack:
            for scope in present:
                await stack.enter_async_context(scope(job))
            yield

    return combined
