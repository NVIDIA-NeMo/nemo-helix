# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs import notifications
from nhx.core.jobs.notifications import build_job_status_event, emit_job_status_event


def _event(status: HelixJobStatus, error_details: dict | None = None) -> dict:
    return build_job_status_event(
        job_id="job-123",
        job_name="ft-run",
        source="customization",
        attempt_id="attempt-1",
        previous_status=HelixJobStatus.ACTIVE,
        status=status,
        error_details=error_details,
    )


@pytest.mark.parametrize(
    ("status", "severity"),
    [
        (HelixJobStatus.COMPLETED, "success"),
        (HelixJobStatus.ERROR, "error"),
        (HelixJobStatus.CANCELLED, "warning"),
        (HelixJobStatus.PAUSED, "info"),
    ],
)
def test_build_job_status_event(status: HelixJobStatus, severity: str) -> None:
    event = _event(status)

    assert event["type"] == f"jobs.job.{status.value}"
    assert event["severity"] == severity
    assert event["data"] == {
        "job_id": "job-123",
        "job_name": "ft-run",
        "source": "customization",
        "attempt_id": "attempt-1",
        "previous_status": "active",
        "status": status.value,
    }
    assert "message" not in event


def test_build_job_status_event_carries_error_message() -> None:
    assert _event(HelixJobStatus.ERROR, {"message": "OOM"})["message"] == "OOM"
    assert "message" not in _event(HelixJobStatus.COMPLETED, {"message": "ignored"})


async def test_emit_posts_to_notif_as_jobs_service() -> None:
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.send = AsyncMock()
    event = _event(HelixJobStatus.COMPLETED)

    with patch.object(notifications, "get_async_nemo_client", return_value=client) as factory:
        emit_job_status_event("default", event)
        await asyncio.gather(*notifications._pending_posts)

    assert factory.call_args.kwargs["as_service"] == "jobs"
    request = client.send.await_args.args[0]
    assert request.method == "POST"
    assert request.path_template == "/apis/notif/v2/workspaces/{workspace}/events"
    assert request.path_params == {"workspace": "default"}
    assert json.loads(request.content) == event


async def test_emit_swallows_post_failure() -> None:
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.send = AsyncMock(side_effect=RuntimeError("notif not installed"))

    with patch.object(notifications, "get_async_nemo_client", return_value=client):
        emit_job_status_event("default", _event(HelixJobStatus.ERROR))
        await asyncio.gather(*notifications._pending_posts, return_exceptions=True)
        await asyncio.sleep(0)

    assert not notifications._pending_posts
