# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from nemo_helix_plugin.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_helix_plugin.jobs.types import JobControl
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.app.lifecycle import (
    active_ttl_anchor,
    attempt_can_transition,
    describe_paused_exit,
    effective_pause_ttl_seconds,
    image_digest_updates,
    pause_deadline_exceeded,
    storage_targets,
    storage_window_expired,
    with_lifecycle_timestamps,
)
from nhx.core.jobs.app.providers import ContainerSpec, CPUExecutionProvider
from nhx.core.jobs.app.schemas import HelixJobEnvironmentVariable, HelixJobSpec, HelixJobStepSpec
from nhx.core.jobs.config import JobsStorageConfig
from nhx.core.jobs.controllers.backends.base import JobBackend
from nhx.core.jobs.entities import HelixJobStep, JobStorageTarget


def test_lifecycle_timestamps_accept_the_step_entity_and_injected_clock():
    now = datetime.datetime(2026, 10, 9, 12, 0, tzinfo=datetime.timezone.utc)
    step = HelixJobStep(
        name="train",
        workspace="default",
        attempt_id="attempt-1",
        status=HelixJobStatus.PAUSING,
    )

    details = with_lifecycle_timestamps(step, HelixJobStatus.PAUSED, {}, now=now)

    assert details == {"stopped_at": now.isoformat()}


def test_same_status_does_not_move_the_clock():
    now = datetime.datetime(2026, 10, 9, 12, 0, tzinfo=datetime.timezone.utc)
    step = SimpleNamespace(status=HelixJobStatus.PAUSED, status_details={})

    assert with_lifecycle_timestamps(step, HelixJobStatus.PAUSED, {"message": "still"}, now=now) == {"message": "still"}


def test_pause_deadline_uses_the_injected_clock():
    requested = datetime.datetime(2026, 10, 9, 12, 0, tzinfo=datetime.timezone.utc)
    step = SimpleNamespace(
        status_details={"pause_requested_at": requested.isoformat()},
        step_spec=SimpleNamespace(lifecycle=SimpleNamespace(pause_deadline_seconds=10)),
    )

    assert pause_deadline_exceeded(step, now=requested + datetime.timedelta(seconds=10)) is False
    assert pause_deadline_exceeded(step, now=requested + datetime.timedelta(seconds=11)) is True


def test_timestamps_across_pause_resume_failure_and_second_pause():
    clock = [datetime.datetime(2026, 10, 9, 12, minute, tzinfo=datetime.timezone.utc) for minute in range(6)]
    step = SimpleNamespace(status=HelixJobStatus.ACTIVE, status_details={})

    def move(status: HelixJobStatus, now: datetime.datetime) -> None:
        step.status_details = {**step.status_details, **(with_lifecycle_timestamps(step, status, {}, now=now) or {})}
        step.status = status

    move(HelixJobStatus.PAUSING, clock[0])
    move(HelixJobStatus.PAUSED, clock[1])
    move(HelixJobStatus.RESUMING, clock[2])
    move(HelixJobStatus.ACTIVE, clock[3])
    assert step.status_details == {
        "pause_requested_at": clock[0].isoformat(),
        "stopped_at": clock[1].isoformat(),
        "resumed_at": clock[2].isoformat(),
    }

    move(HelixJobStatus.PAUSING, clock[4])
    move(HelixJobStatus.ERROR, clock[5])
    assert step.status_details == {
        "pause_requested_at": clock[4].isoformat(),
        "stopped_at": clock[5].isoformat(),
        "resumed_at": clock[2].isoformat(),
    }


@pytest.mark.parametrize(
    "exit_code,expected",
    [
        (75, HelixJobStatus.PAUSED),
        (0, HelixJobStatus.COMPLETED),
        (1, HelixJobStatus.ERROR),
        (137, HelixJobStatus.ERROR),
    ],
)
def test_pausing_exit_code_mapping(exit_code: int, expected: HelixJobStatus):
    status, _, error_details = describe_paused_exit(exit_code)
    assert status == expected
    assert bool(error_details) == (expected == HelixJobStatus.ERROR)


def test_pausing_transitions():
    assert HelixJobStatus.PAUSING.can_transition_to(HelixJobStatus.CANCELLING)
    assert HelixJobStatus.PAUSING.can_transition_to(HelixJobStatus.COMPLETED)
    # A stale active report from the reconciler must not undo a pause on the step.
    assert not HelixJobStatus.PAUSING.can_transition_to(HelixJobStatus.ACTIVE)
    # The attempt moves on when the step being paused finished instead.
    assert attempt_can_transition(HelixJobStatus.PAUSING, HelixJobStatus.ACTIVE)
    assert not attempt_can_transition(HelixJobStatus.COMPLETED, HelixJobStatus.ACTIVE)


def test_active_ttl_does_not_count_paused_time():
    created = datetime.datetime(2026, 10, 1, tzinfo=datetime.timezone.utc)
    resumed = created + datetime.timedelta(days=3)
    step = SimpleNamespace(created_at=created, status_details={"resumed_at": resumed.isoformat()})

    assert active_ttl_anchor(step) == resumed
    one_day = 24 * 3600
    assert not JobBackend.check_step_ttl(MagicMock(), step, one_day, now=resumed + datetime.timedelta(hours=1))
    assert JobBackend.check_step_ttl(MagicMock(), step, one_day, now=resumed + datetime.timedelta(days=2))


def test_stopping_snapshots_the_image_digest():
    now = datetime.datetime(2026, 10, 9, 12, 0, tzinfo=datetime.timezone.utc)
    step = SimpleNamespace(status=HelixJobStatus.ACTIVE, status_details={"image_digest": "sha256:old"})

    details = with_lifecycle_timestamps(step, HelixJobStatus.ERROR, {}, now=now)

    assert details == {"stopped_at": now.isoformat(), "image_digest_at_save": "sha256:old"}


def test_image_digest_comparison():
    assert image_digest_updates({}, "sha256:a") == {"image_digest": "sha256:a"}
    saved = {"image_digest_at_save": "sha256:a"}
    assert image_digest_updates(saved, "sha256:a") == {"image_digest": "sha256:a"}
    assert "image_digest_warning" in image_digest_updates(saved, "sha256:b")
    assert "image_digest_comparison_skipped" in image_digest_updates(saved, None)
    # The warning or skip note is written once.
    warned = {**saved, "image_digest_warning": "changed"}
    assert image_digest_updates(warned, "sha256:c") == {"image_digest": "sha256:c"}


def test_storage_targets_lists_each_backend_that_mounts_storage_once():
    storage = [HelixJobEnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value="/storage")]
    cpu = CPUExecutionProvider(provider="cpu", profile="default", container=ContainerSpec(image="busybox"))
    large = CPUExecutionProvider(provider="cpu", profile="large", container=ContainerSpec(image="busybox"))
    spec = HelixJobSpec(
        steps=[
            HelixJobStepSpec(name="download", executor=cpu, environment=storage),
            HelixJobStepSpec(name="train", executor=large, environment=storage),
            HelixJobStepSpec(name="upload", executor=cpu, environment=storage),
            HelixJobStepSpec(name="notify", executor=large),
        ]
    )

    assert storage_targets(spec) == [
        JobStorageTarget(provider="cpu", profile="default"),
        JobStorageTarget(provider="cpu", profile="large"),
    ]


def test_storage_window():
    stopped = datetime.datetime(2026, 10, 9, 12, 0, tzinfo=datetime.timezone.utc)
    assert not storage_window_expired(stopped, 3600, now=stopped + datetime.timedelta(minutes=59))
    assert storage_window_expired(stopped, 3600, now=stopped + datetime.timedelta(minutes=61))
    assert storage_window_expired(stopped, 0, now=stopped)


def test_effective_pause_window_combines_job_control_and_platform_settings():
    settings = JobsStorageConfig(paused_storage_ttl_seconds=600, maximum_pause_ttl_seconds=900)

    assert effective_pause_ttl_seconds(JobControl(), settings) == 600
    assert effective_pause_ttl_seconds(JobControl(pause_ttl_seconds=300), settings) == 300
    # A maximum lowered after a job saved its value still caps that value.
    assert effective_pause_ttl_seconds(JobControl(pause_ttl_seconds=1200), settings) == 900
