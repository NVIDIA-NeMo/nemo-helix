# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live trial progress of a task Job, as the Job reports it in its status details.

A task Job runs its task's Harbor trials one at a time and can take hours, while the run only learns a task's
outcome when its Job ends. So the Job reports each trial's progress under :data:`TRIALS_STATUS_KEY` as Harbor
moves through it (see ``nemo_evals.harbor.trial_progress``).
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

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
