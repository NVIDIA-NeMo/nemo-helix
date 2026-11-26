# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report a run's trials in the job's status details while it produces them."""

from __future__ import annotations

import asyncio
import math
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from nemo_evals.jobs.run_outcome import PUBLISH_TIMEOUT_SEC, publish_best_effort
from nemo_evals.taskset_runs.trials import (
    TRIAL_COUNTS_STATUS_KEY,
    TrialCounts,
    TrialPhase,
    TrialProgress,
)
from nhx_evals_sdk.agent_eval.runtimes.gym.records import NG_ATTEMPT_INDEX, NG_TASK_INDEX
from nhx_evals_sdk.agent_eval.runtimes.gym.results import rollout_failure
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus


class _CountReporter:
    """Counts a run's finished trials against its plan and reports them under :data:`TRIAL_COUNTS_STATUS_KEY`.

    Reports go out in the background, in order, so a slow status write never holds up the run; ``stopped``
    waits for them. A planned trial not reported by then counts as cancelled.
    """

    def __init__(
        self,
        publish: Callable[[dict[str, Any]], Awaitable[None]],
        *,
        publish_timeout: float = PUBLISH_TIMEOUT_SEC,
    ) -> None:
        self._publish = publish
        self._publish_timeout = publish_timeout
        self._planned = 0
        self._unfinished = TrialPhase.PENDING
        self._phases: Counter[TrialPhase] = Counter()
        self._errors: Counter[str] = Counter()
        self._rewarded = 0
        self._reward_sum = 0.0
        self._settled = 0
        self._reports: list[asyncio.Task[None]] = []

    async def planned(self, trials: int) -> None:
        self._planned = trials
        self._report()

    async def stopped(self) -> None:
        self._unfinished = TrialPhase.CANCELLED
        self._report()
        await asyncio.gather(*self._reports)

    def counts(self) -> TrialCounts:
        """The finished trials counted as recorded; the rest pending, or cancelled once the run stopped."""
        phases = dict(self._phases)
        unfinished = self._planned - self._settled
        if unfinished > 0:
            phases[self._unfinished] = unfinished
        return TrialCounts(
            total=sum(phases.values()),
            phases=phases,
            rewarded=self._rewarded,
            reward=self._reward_sum / self._rewarded if self._rewarded else None,
            errors=dict(self._errors),
        )

    def _add(self, trials: Sequence[TrialProgress], *, settled: int) -> None:
        """Record finished ``trials``, of which ``settled`` each settle one planned trial."""
        for trial in trials:
            self._phases[trial.phase] += 1
            if trial.phase is TrialPhase.ERRORED:
                self._errors[trial.error_type or "error"] += 1
            if trial.reward is not None:
                self._rewarded += 1
                self._reward_sum += trial.reward
        self._settled += settled
        self._report()

    def _report(self) -> None:
        details = {TRIAL_COUNTS_STATUS_KEY: self.counts().model_dump(mode="json")}
        self._reports = [report for report in self._reports if not report.done()]
        self._reports.append(
            asyncio.ensure_future(publish_best_effort(self._publish, details, timeout=self._publish_timeout))
        )


class TrialCounter(_CountReporter):
    """Counts a Fabric runner's or an endpoint target's trials as they finish.

    A trial failed when it reports an error or failed status, as the job's final outcome counts it. Satisfies
    :class:`~nhx_evals_sdk.agent_eval.progress.TrialProgressListener`.
    """

    async def finished(self, trials: Sequence[AgentEvalTrial]) -> None:
        self._add([_progress_of(trial) for trial in trials], settled=len(trials))


class GymTrialCounter(_CountReporter):
    """Counts a Gym run's rollouts as they finish.

    A run plans one trial per rollout (tasks times repeats), and every record Gym writes is a trial, as in the
    final evaluation: a retried rollout counts once per attempt, and only a first attempt settles a planned one.
    Satisfies :class:`~nhx_evals_sdk.agent_eval.runtimes.gym.progress.RolloutProgress`.
    """

    def __init__(
        self,
        publish: Callable[[dict[str, Any]], Awaitable[None]],
        *,
        reward_key: str,
        publish_timeout: float = PUBLISH_TIMEOUT_SEC,
    ) -> None:
        super().__init__(publish, publish_timeout=publish_timeout)
        self._reward_key = reward_key

    async def finished(self, records: Sequence[Mapping[str, Any]]) -> None:
        attributed = [record for record in records if _task_index(record) is not None]
        self._add(
            [self._trial(record) for record in attributed],
            settled=sum(1 for record in attributed if not record.get(NG_ATTEMPT_INDEX)),
        )

    def _trial(self, record: Mapping[str, Any]) -> TrialProgress:
        failure = rollout_failure(record)
        if failure is not None:
            return TrialProgress(trial="", phase=TrialPhase.ERRORED, error_type=failure)
        return TrialProgress(trial="", phase=TrialPhase.SUCCEEDED, reward=_reward(record.get(self._reward_key)))


def _progress_of(trial: AgentEvalTrial) -> TrialProgress:
    if trial.error is not None or trial.status is AgentEvalTrialStatus.FAILED:
        return TrialProgress(
            trial=trial.id, phase=TrialPhase.ERRORED, error_type=trial.error.type if trial.error else None
        )
    return TrialProgress(trial=trial.id, phase=TrialPhase.SUCCEEDED)


def _task_index(record: Mapping[str, Any]) -> int | None:
    """The record's task index; None when Gym wrote none it can be attributed by, as the final evaluation skips it."""
    index = record.get(NG_TASK_INDEX)
    return index if isinstance(index, int) and not isinstance(index, bool) else None


def _reward(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)
