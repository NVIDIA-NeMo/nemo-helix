# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.app.lifecycle import (
    active_ttl_anchor,
    attempt_can_transition,
    describe_paused_exit,
    pause_deadline_exceeded,
    with_lifecycle_timestamps,
)
from nhx.core.jobs.controllers.backends.base import JobBackend
from nhx.core.jobs.entities import HelixJobStep


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
