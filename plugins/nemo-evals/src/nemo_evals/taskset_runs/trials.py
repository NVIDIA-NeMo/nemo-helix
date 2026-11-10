# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live trial progress of a task Job, as the Job reports it in its status details.

A task Job runs its task's Harbor trials one at a time and can take hours, while the run only learns a task's
outcome when its Job ends. So the Job reports each trial's progress under :data:`TRIALS_STATUS_KEY` as Harbor
moves through it (see ``nemo_evals.harbor.trial_progress``): the run's task API reads the list as it is, and the run
controller copies a :class:`TrialCounts` summary onto the task's member each time it reads the Job, so run and task
pages can show trial progress without reading every Job.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime
from enum import StrEnum
from statistics import fmean

from pydantic import BaseModel, ConfigDict, Field, ValidationError

TRIALS_STATUS_KEY = "trials"
"""Status-details key under which a task Job reports its trials, as a list of :class:`TrialProgress`."""


class TrialPhase(StrEnum):
    """Where a Harbor trial is in its lifecycle, following the events Harbor raises for it."""

    PENDING = "pending"  # Planned; Harbor hasn't started it.
    STARTING = "starting"  # Harbor started the trial and is preparing its environment.
    AGENT_RUNNING = "agent_running"  # The environment is up and the agent is working on the task.
    VERIFYING = "verifying"  # The agent finished; the verifier is scoring its work.
    RETRYING = "retrying"  # The attempt raised and Harbor will run the trial again.
    SUCCEEDED = "succeeded"  # The trial ended without an error; see reward.
    ERRORED = "errored"  # The last attempt raised; reward is set only if the verifier still scored it.
    CANCELLED = "cancelled"  # Harbor cancelled the trial, or the task Job stopped or raised before it ended.


SETTLED_TRIAL_PHASES = frozenset({TrialPhase.SUCCEEDED, TrialPhase.ERRORED, TrialPhase.CANCELLED})


class TrialProgress(BaseModel):
    """One trial of a task Job: its phase, timing, reward and error, as of the Job's last report."""

    model_config = ConfigDict(extra="ignore")

    trial: str = Field(description="Harbor's trial name, unique within the task Job.")
    session_id: str | None = Field(
        default=None,
        description="Intake session holding the trial's traces (<task Job>:<trial>), once anything is recorded there.",
    )
    phase: TrialPhase = Field(description="Where the trial is in its lifecycle.")
    attempts: int = Field(default=0, ge=0, description="Attempts Harbor started, retries included.")
    started_at: datetime | None = Field(default=None, description="When the trial's first attempt started.")
    finished_at: datetime | None = Field(default=None, description="When the trial settled; None until then.")
    reward: float | None = Field(default=None, description="The verifier's primary reward for the last attempt.")
    error_type: str | None = Field(default=None, description="Exception type of the last failed attempt.")
    error: str | None = Field(default=None, description="Exception message of the last failed attempt, truncated.")


class TrialCounts(BaseModel):
    """A task's trials counted by phase, with their mean reward: what run and task lists show per task."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    total: int = Field(ge=0, description="Trials the task Job planned.")
    phases: dict[TrialPhase, int] = Field(
        default_factory=dict,
        description=(
            'Trials by phase, for example {"pending": 1, "agent_running": 1, "succeeded": 2}; a phase with no '
            "trials is left out."
        ),
    )
    rewarded: int = Field(default=0, ge=0, description="Trials with a reward; reward averages over them.")
    reward: float | None = Field(default=None, description="Mean reward over the rewarded trials; None without any.")
    errors: dict[str, int] = Field(
        default_factory=dict, description="Errored trials by exception type, for example {'AgentTimeoutError': 1}."
    )


def count_trials(trials: Sequence[TrialProgress]) -> TrialCounts:
    """Count ``trials`` by phase and average their rewards."""
    rewards = [trial.reward for trial in trials if trial.reward is not None]
    errors = Counter(trial.error_type or "error" for trial in trials if trial.phase is TrialPhase.ERRORED)

    return TrialCounts(
        total=len(trials),
        phases=dict(Counter(trial.phase for trial in trials)),
        rewarded=len(rewards),
        reward=fmean(rewards) if rewards else None,
        errors=dict(errors),
    )


def reported_trials(status_details: object) -> list[TrialProgress] | None:
    """The trials a task Job reported in ``status_details``; None when it reported none or a shape we can't read.

    The Job reports them best effort, and an older dispatcher reports nothing: neither may break a read.
    """
    if not isinstance(status_details, dict):
        return None
    reported = status_details.get(TRIALS_STATUS_KEY)
    if not isinstance(reported, list):
        return None
    try:
        return [TrialProgress.model_validate(item) for item in reported]
    except ValidationError:
        return None


def sum_trial_counts(counts: Iterable[TrialCounts]) -> TrialCounts | None:
    """Sum trial counts across tasks; None when no task has any.

    ``reward`` is the mean of each task's mean reward, so every task counts equally regardless of how many trials
    it ran, as Harbor scores a benchmark.
    """
    counts = list(counts)
    if not counts:
        return None

    phases: Counter[TrialPhase] = Counter()
    errors: Counter[str] = Counter()
    for count in counts:
        phases.update(count.phases)
        errors.update(count.errors)
    task_rewards = [count.reward for count in counts if count.reward is not None]

    return TrialCounts(
        total=sum(count.total for count in counts),
        phases=dict(phases),
        rewarded=sum(count.rewarded for count in counts),
        reward=fmean(task_rewards) if task_rewards else None,
        errors=dict(errors),
    )
