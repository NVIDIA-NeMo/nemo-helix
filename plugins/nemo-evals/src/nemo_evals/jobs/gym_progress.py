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
from nhx_evals_sdk.agent_eval.runtimes.gym.results import rollout_failure


class GymTrialCounter:
    """Counts a Gym run's rollouts as they finish and reports them under :data:`TRIAL_COUNTS_STATUS_KEY`.

    A run plans one trial per rollout (tasks times repeats), and every record Gym writes is a trial, as in the
    final evaluation: a retried rollout counts once per attempt. Satisfies
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
        self._finished: list[TrialProgress] = []
        self._lock = asyncio.Lock()

    async def planned(self, rollouts: int) -> None:
        self._planned = rollouts
        await self._report()

    async def finished(self, records: Sequence[Mapping[str, Any]]) -> None:
        self._finished.extend(self._trial(record) for record in records)
        await self._report()

    async def stopped(self) -> None:
        self._unfinished = TrialPhase.CANCELLED
        await self._report()

    def counts(self) -> TrialCounts:
        """The finished rollouts counted as recorded; the rest pending, or cancelled once collection stopped."""
        counts = count_trials(self._finished)
        unfinished = self._planned - counts.total
        if unfinished <= 0:
            return counts
        return counts.model_copy(
            update={"total": self._planned, "phases": {**counts.phases, self._unfinished: unfinished}}
        )

    def _trial(self, record: Mapping[str, Any]) -> TrialProgress:
        failure = rollout_failure(record)
        if failure is not None:
            return TrialProgress(trial="", phase=TrialPhase.ERRORED, error_type=failure)
        return TrialProgress(trial="", phase=TrialPhase.SUCCEEDED, reward=_reward(record.get(self._reward_key)))

    async def _report(self) -> None:
        async with self._lock:
            details = {TRIAL_COUNTS_STATUS_KEY: self.counts().model_dump(mode="json")}
            await publish_best_effort(self._publish, details, timeout=self._publish_timeout)


def _reward(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)
