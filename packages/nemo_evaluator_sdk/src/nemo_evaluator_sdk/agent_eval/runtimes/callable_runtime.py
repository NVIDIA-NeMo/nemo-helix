# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal, dependency-light AgentTaskRunner backed by a user-supplied callable."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import (
    AgentEvalTrial,
    AgentEvalTrialStatus,
    AgentOutput,
    RunnerInfo,
    TrialMeasurements,
    callable_identity,
)
from nemo_evaluator_sdk.values.evidence import CandidateEvidence


@dataclass(slots=True)
class TrialDraft:
    """What an agent callable returns for one task: final output plus optional evidence.

    The runtime wraps this into a completed :class:`AgentEvalTrial`. Returning a
    :class:`AgentOutput` or a plain string is also accepted as shorthand.
    """

    output: AgentOutput
    evidence: CandidateEvidence | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    measurements: TrialMeasurements = field(default_factory=TrialMeasurements)


AgentTaskFn = Callable[[AgentEvalTask], Awaitable[TrialDraft | AgentOutput | str]]


class CallableAgentTaskRunner:
    """Smallest possible :class:`AgentTaskRunner`: delegate each task to an async callable.

    The callable receives an :class:`AgentEvalTask` and returns the agent's final output as
    a :class:`TrialDraft`, an :class:`AgentOutput`, or a plain string. This runtime adds only
    what the ``AgentTaskRunner`` contract needs: bounded concurrency, stable trial ids, and
    failure capture (an exception becomes a ``FAILED`` trial instead of aborting the batch).
    It requires no Docker or external agent SDK, so it doubles as a reference for richer
    runtimes and as the seam an ``AgentEvaluator`` drives via ``run(target=runner)``.
    """

    def __init__(
        self,
        agent_fn: AgentTaskFn,
        *,
        parallelism: int | None = None,
        trial_id_suffix: str = "trial",
    ) -> None:
        self._agent_fn = agent_fn
        self._parallelism = parallelism
        self._trial_id_suffix = trial_id_suffix

    def runner_info(self) -> RunnerInfo:
        """Identify this runner; the agent callable itself is the result-shaping detail."""
        return RunnerInfo(
            name="callable",
            kind="runner",
            config={
                "agent_fn": callable_identity(self._agent_fn),
                "parallelism": self._parallelism,
            },
        )

    async def run_tasks(
        self,
        tasks: Sequence[AgentEvalTask],
        config: AgentEvalRunConfig | None = None,
    ) -> list[AgentEvalTrial]:
        """Run every task through the callable and return one trial per task, in order."""
        parallelism = self._parallelism if self._parallelism is not None else (config.parallelism if config else 4)
        semaphore = asyncio.Semaphore(max(1, parallelism))

        async def run_one(task: AgentEvalTask) -> AgentEvalTrial:
            async with semaphore:
                try:
                    result = await self._agent_fn(task)
                except Exception as exc:  # noqa: BLE001 - surfaced as a FAILED trial, not a crash
                    return self._failed_trial(task, exc)
                return self._completed_trial(task, result)

        return list(await asyncio.gather(*(run_one(task) for task in tasks)))

    def _trial_id(self, task: AgentEvalTask) -> str:
        return f"{task.id}:{self._trial_id_suffix}"

    def _completed_trial(self, task: AgentEvalTask, result: TrialDraft | AgentOutput | str) -> AgentEvalTrial:
        draft = _as_trial_draft(result)
        return AgentEvalTrial(
            id=self._trial_id(task),
            task_id=task.id,
            status=AgentEvalTrialStatus.COMPLETED,
            output=draft.output,
            evidence=draft.evidence,
            measurements=draft.measurements,
            metadata=draft.metadata,
        )

    def _failed_trial(self, task: AgentEvalTask, exc: Exception) -> AgentEvalTrial:
        return AgentEvalTrial(
            id=self._trial_id(task),
            task_id=task.id,
            status=AgentEvalTrialStatus.FAILED,
            metadata={"error": f"{type(exc).__name__}: {exc}"},
        )


def _as_trial_draft(result: TrialDraft | AgentOutput | str) -> TrialDraft:
    if isinstance(result, TrialDraft):
        return result
    if isinstance(result, AgentOutput):
        return TrialDraft(output=result)
    if isinstance(result, str):
        return TrialDraft(output=AgentOutput(output_text=result))
    raise TypeError(f"agent callable must return TrialDraft, AgentOutput, or str; got {type(result).__name__}")
