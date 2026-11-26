# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report a Gym run's trials in the job's status details while it collects them."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from nemo_evals.jobs.run_outcome import PUBLISH_TIMEOUT_SEC, publish_best_effort
from nemo_evals.taskset_runs.trials import (
    TRIAL_COUNTS_STATUS_KEY,
    TrialCounts,
    TrialPhase,
    TrialProgress,
    count_trials,
)
from nhx_evals_sdk.agent_eval.runtimes.gym.records import NG_ROLLOUT_INDEX, NG_TASK_INDEX
from nhx_evals_sdk.agent_eval.runtimes.gym.results import rollout_failure


class GymTrialCounter:
    """Counts a Gym run's rollouts as they finish and reports them under :data:`TRIAL_COUNTS_STATUS_KEY`.

    One trial is one rollout: a task run once, so a run has tasks times repeats of them. Gym writes a record
    per attempt, so a retried rollout's latest record stands. Satisfies
    :class:`~nhx_evals_sdk.agent_eval.runtimes.gym.progress.RolloutProgress`.
    """

    def __init__(
        self,
        publish: Callable[[dict[str, Any]], Awaitable[None]],
        *,
        reward_key: str,
        publish_timeout: float = PUBLISH_TIMEOUT_SEC,
    ) -> None:
        self._publish = publish
        self._reward_key = reward_key
        self._publish_timeout = publish_timeout
        self._planned = 0
        self._unfinished = TrialPhase.PENDING
        self._rollouts: dict[str, TrialProgress] = {}
        self._lock = asyncio.Lock()

    async def planned(self, rollouts: int) -> None:
        self._planned = rollouts
        await self._report()

    async def finished(self, records: Sequence[Mapping[str, Any]]) -> None:
        for record in records:
            rollout = self._rollout(record)
            self._rollouts[rollout.trial] = rollout
        await self._report()

    async def stopped(self) -> None:
        self._unfinished = TrialPhase.CANCELLED
        await self._report()

    def counts(self) -> TrialCounts:
        """The finished rollouts counted as recorded; the rest pending, or cancelled once collection stopped."""
        counts = count_trials(list(self._rollouts.values()))
        unfinished = self._planned - counts.total
        if unfinished <= 0:
            return counts
        return counts.model_copy(
            update={"total": self._planned, "phases": {**counts.phases, self._unfinished: unfinished}}
        )

    def _rollout(self, record: Mapping[str, Any]) -> TrialProgress:
        task, rollout = record.get(NG_TASK_INDEX), record.get(NG_ROLLOUT_INDEX)
        name = f"{task}:{rollout}" if isinstance(task, int) and isinstance(rollout, int) else f"#{len(self._rollouts)}"
        failure = rollout_failure(record)
        if failure is not None:
            return TrialProgress(trial=name, phase=TrialPhase.ERRORED, error_type=failure)
        return TrialProgress(trial=name, phase=TrialPhase.SUCCEEDED, reward=_reward(record.get(self._reward_key)))

    async def _report(self) -> None:
        async with self._lock:
            details = {TRIAL_COUNTS_STATUS_KEY: self.counts().model_dump(mode="json")}
            await publish_best_effort(self._publish, details, timeout=self._publish_timeout)


def _reward(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)
