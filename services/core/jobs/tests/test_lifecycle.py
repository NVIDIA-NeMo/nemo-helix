# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
from types import SimpleNamespace

from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.app.lifecycle import pause_deadline_exceeded, with_lifecycle_timestamps
from nhx.core.jobs.entities import HelixJobStep


def test_lifecycle_timestamps_accept_the_step_entity_and_injected_clock():
    now = datetime.datetime(2026, 10, 9, 12, 0, tzinfo=datetime.timezone.utc)
    step = HelixJobStep(
        name="train",
        workspace="default",
        attempt_id="attempt-1",
        status=HelixJobStatus.ACTIVE,
        status_details={"image_digest": "sha256:old"},
    )

    details = with_lifecycle_timestamps(step, HelixJobStatus.PAUSED, {}, now=now)

    assert details is not None
    assert details["stopped_at"] == now.isoformat()
    assert details["image_digest_at_save"] == "sha256:old"


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
