# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``build_progress``: a run's progress counted from its members."""

import pytest
from nemo_evals.taskset_runs.entities import MemberPhase, TasksetRunMemberEntity
from nemo_evals.taskset_runs.progress import build_progress
from nemo_helix_plugin.jobs.schemas import HelixJobStatus


def member(
    index: int, phase: MemberPhase, *, job: bool = False, job_status: HelixJobStatus | None = None
) -> TasksetRunMemberEntity:
    return TasksetRunMemberEntity(
        name=f"task-{index}",
        workspace="default",
        task_index=index,
        phase=phase,
        task_job=f"run-x-task-{index}" if job else None,
        job_status=job_status,
    )


@pytest.mark.parametrize("phase", list(MemberPhase))
def test_every_phase_is_counted(phase):
    progress = build_progress([member(0, phase)])
    assert progress.tasks == 1
    assert progress.members == {phase: 1}
    assert progress.queued == (phase is MemberPhase.IMAGE_READY)


def test_a_create_intent_is_not_queued():
    """An image_ready member with task_job set holds a slot: it is counted in image_ready but not queued."""
    progress = build_progress([member(0, MemberPhase.IMAGE_READY), member(1, MemberPhase.IMAGE_READY, job=True)])
    assert progress.members == {MemberPhase.IMAGE_READY: 2}
    assert progress.queued == 1
    assert progress.task_jobs == {}


def test_task_jobs_count_registered_members_by_job_status():
    """Settled members keep their Job's end status, but only registered members' Jobs are counted."""
    members = [
        member(0, MemberPhase.REGISTERED, job=True, job_status=HelixJobStatus.ACTIVE),
        member(1, MemberPhase.REGISTERED, job=True, job_status=HelixJobStatus.ACTIVE),
        member(2, MemberPhase.REGISTERED, job=True, job_status=HelixJobStatus.PENDING),
        member(3, MemberPhase.REGISTERED, job=True),  # no status read yet
        member(4, MemberPhase.SUCCEEDED, job=True, job_status=HelixJobStatus.COMPLETED),
        member(5, MemberPhase.FAILED, job=True, job_status=HelixJobStatus.ERROR),
        member(6, MemberPhase.IMAGE_READY),
    ]
    progress = build_progress(members)
    assert progress.model_dump(mode="json") == {
        "tasks": 7,
        "members": {"registered": 4, "succeeded": 1, "failed": 1, "image_ready": 1},
        "queued": 1,
        "task_jobs": {"active": 2, "pending": 1},
    }


def test_no_members_counts_nothing():
    assert build_progress(iter([])).model_dump() == {"tasks": 0, "members": {}, "queued": 0, "task_jobs": {}}
