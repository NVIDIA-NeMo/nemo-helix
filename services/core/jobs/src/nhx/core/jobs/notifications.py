# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort job status events for the notif plugin.

Each job status change is posted to ``/apis/notif/v2/workspaces/{workspace}/events`` as
``service:jobs``. The notif plugin is optional: when it is not installed the post fails and is
dropped, and a failed post never affects the status update that triggered it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from nemo_helix_plugin.client.types import PreparedRequest
from nhx.common.client_factory import get_async_nemo_client
from nhx.common.jobs.schemas import HelixJobStatus

logger = logging.getLogger(__name__)

NOTIF_EVENTS_PATH = "/apis/notif/v2/workspaces/{workspace}/events"

_SEVERITY = {
    HelixJobStatus.COMPLETED: "success",
    HelixJobStatus.ERROR: "error",
    HelixJobStatus.CANCELLED: "warning",
}


def build_job_status_event(
    *,
    job_id: str,
    job_name: str,
    source: str,
    attempt_id: str,
    previous_status: HelixJobStatus,
    status: HelixJobStatus,
    error_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the notif publish body for a job moving from ``previous_status`` to ``status``."""
    event: dict[str, Any] = {
        "type": f"jobs.job.{status.value}",
        "title": f"Job {job_name} is {status.value}",
        "severity": _SEVERITY.get(status, "info"),
        "data": {
            "job_id": job_id,
            "job_name": job_name,
            "source": source,
            "attempt_id": attempt_id,
            "previous_status": previous_status.value,
            "status": status.value,
        },
    }
    message = (error_details or {}).get("message")
    if status == HelixJobStatus.ERROR and isinstance(message, str) and message:
        event["message"] = message[:2000]
    return event


async def _post_job_status_event(workspace: str, event: dict[str, Any]) -> None:
    request: PreparedRequest[None] = PreparedRequest(
        path_template=NOTIF_EVENTS_PATH,
        path_params={"workspace": workspace},
        method="POST",
        content=json.dumps(event).encode(),
        content_type="application/json",
        response_type=None,
    )
    async with get_async_nemo_client(as_service="jobs", internal=True, retry=None, timeout=5.0) as client:
        await client.send(request)


def emit_job_status_event(workspace: str, event: dict[str, Any]) -> None:
    """Post ``event`` to the notif plugin in the background."""
    try:
        task = asyncio.create_task(_post_job_status_event(workspace, event))
    except RuntimeError:
        logger.debug("No running event loop for job status notification", exc_info=True)
        return
    # The loop only keeps a weak reference to a task, so hold one until it finishes.
    _pending_posts.add(task)
    task.add_done_callback(_consume_task_exception)


_pending_posts: set[asyncio.Task[None]] = set()


def _consume_task_exception(task: asyncio.Task[None]) -> None:
    _pending_posts.discard(task)
    try:
        task.result()
    except Exception:
        logger.debug("Job status notification failed", exc_info=True)
