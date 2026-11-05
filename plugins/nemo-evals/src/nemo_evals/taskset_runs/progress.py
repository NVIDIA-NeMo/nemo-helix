# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A run's progress, counted from its members on each read: nothing is stored on the run, and no Jobs call."""

from collections import Counter
from collections.abc import Iterable

from nemo_evals.taskset_runs.entities import MemberPhase, TasksetRunMemberEntity, is_queued
from nemo_evals.taskset_runs.trials import TrialCounts, sum_trial_counts
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from pydantic import BaseModel, Field


class TasksetRunProgress(BaseModel):
    """How far a run's tasks got, as of the controller's last writes to its members."""

    tasks: int = Field(description="Members created; the plan's task count from the controller's first cycle on.")
    members: dict[MemberPhase, int] = Field(
        description="Members by phase, settled phases included; a phase with no members is left out."
    )
    queued: int = Field(description="Members waiting for a task-Job slot: image_ready without a create intent.")
    task_jobs: dict[HelixJobStatus, int] = Field(
        description=(
            "Registered members by their task Job's status as the controller last read it, "
            'for example {"active": 18, "pending": 2}.'
        )
    )
    trial_counts: TrialCounts | None = Field(
        default=None,
        description=(
            "Every task's trials counted together, as their task Jobs last reported them (planned trials count "
            "as pending until then). Unlike a task's, reward here is the mean of the tasks' mean rewards, so each "
            "task weighs the same whatever its trial count, as Harbor scores a benchmark."
        ),
    )


def build_progress(members: Iterable[TasksetRunMemberEntity]) -> TasksetRunProgress:
    """Count ``members`` by phase, queued tasks, and registered members' task Jobs by status.

    A queued member is ``image_ready`` with no ``task_job``; one with ``task_job`` set is a create intent
    and already holds a slot. A registered member with no ``job_status`` yet isn't in ``task_jobs``.
    Trials sum the counts the controller copied onto each member from its task Job.
    """
    members = list(members)
    return TasksetRunProgress(
        tasks=len(members),
        members=dict(Counter(member.phase for member in members)),
        queued=sum(is_queued(member) for member in members),
        task_jobs=dict(
            Counter(
                member.job_status
                for member in members
                if member.phase is MemberPhase.REGISTERED and member.job_status is not None
            )
        ),
        trial_counts=sum_trial_counts(member.trial_counts for member in members if member.trial_counts is not None),
    )
