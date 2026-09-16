# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone

from nemo_helix_plugin.jobs.schemas import PlatformJobStatus, PlatformJobStatusResponse
from nemo_helix_plugin.jobs.watch import _status_event


def test_status_event_preserves_error_details_for_failed_job() -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    status_response = PlatformJobStatusResponse(
        id="job-id",
        name="job-name",
        status=PlatformJobStatus.ERROR,
        status_details={"phase": "failed"},
        error_details={"reason": "container exited"},
        steps=[],
        created_at=timestamp,
        updated_at=timestamp,
    )

    event = _status_event(status_response, "job-name")

    assert event.status == "error"
    assert event.status_details == {"phase": "failed"}
    assert event.error_details == {"reason": "container exited"}
    assert event.terminal is True
    assert event.successful is False
