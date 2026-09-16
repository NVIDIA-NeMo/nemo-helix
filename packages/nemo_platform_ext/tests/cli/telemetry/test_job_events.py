# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from nemo_platform_ext.cli.core import waiters
from nemo_platform_plugin.jobs.schemas import PlatformJobStatus, PlatformJobStatusResponse

WAITERS_MODULE = "nemo_platform_ext.cli.core.waiters"
WATCH_MODULE = "nemo_platform_plugin.jobs.watch"
EMIT_TARGET = "nemo_platform_ext.cli.telemetry.emit.emit_event"
JOB_TIMESTAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _DummyLive:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __enter__(self) -> "_DummyLive":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def update(self, *args: object) -> None:
        pass

    def stop(self) -> None:
        pass

    def start(self) -> None:
        pass


class _StatusResponse:
    def __init__(self, status: PlatformJobStatusResponse) -> None:
        self._status = status

    def data(self) -> PlatformJobStatusResponse:
        return self._status


def _status_response(status: str | PlatformJobStatus) -> _StatusResponse:
    return _StatusResponse(
        PlatformJobStatusResponse(
            id="job-a",
            name="job-a",
            status=PlatformJobStatus(status),
            status_details={"input_tokens": 512, "output_tokens": 2048, "model": "nemotron"},
            error_details=None,
            steps=[],
            created_at=JOB_TIMESTAMP,
            updated_at=JOB_TIMESTAMP,
        )
    )


@pytest.fixture(autouse=True)
def _quiet_rich_output() -> Iterator[None]:
    with (
        patch(f"{WAITERS_MODULE}.Live", _DummyLive),
        patch(f"{WAITERS_MODULE}.console.print"),
    ):
        yield


def test_wait_for_platform_job_does_not_emit_client_job_run_event() -> None:
    jobs = MagicMock()
    jobs.get_job_status.return_value = _status_response("completed")

    with patch(EMIT_TARGET) as emit_event:
        assert waiters.wait_for_platform_job(jobs, "job-a", workspace="default", resource_label="customization") is True

    emit_event.assert_not_called()


def test_wait_for_platform_job_timeout_still_does_not_emit_client_job_run_event() -> None:
    jobs = MagicMock()
    jobs.get_job_status.return_value = _status_response("active")

    with (
        patch(f"{WAITERS_MODULE}.time.time", return_value=0.0),
        patch(f"{WATCH_MODULE}.time.monotonic", side_effect=[0.0, 0.0, 4.0, 5.0]),
        patch(f"{WATCH_MODULE}.time.sleep"),
        patch(EMIT_TARGET) as emit_event,
    ):
        assert waiters.wait_for_platform_job(jobs, "job-a", workspace="default", timeout=5, poll_interval=10) is False

    emit_event.assert_not_called()
