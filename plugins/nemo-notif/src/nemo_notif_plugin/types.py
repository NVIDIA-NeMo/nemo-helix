# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request and event models for the notif plugin."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal["info", "success", "warning", "error"]


class CreateNotifEventRequest(BaseModel):
    """Body of ``POST /v2/workspaces/{workspace}/events``."""

    type: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
        description="Dot-separated event type, e.g. 'jobs.job.failed'. Listeners can filter on it.",
    )
    title: str = Field(min_length=1, max_length=200)
    message: str | None = Field(default=None, max_length=2000)
    severity: Severity = "info"
    link: str | None = Field(default=None, max_length=2000, description="Relative Studio path to open on click.")
    data: dict[str, Any] = Field(default_factory=dict, description="Free-form payload for the listener.")


class NotifEvent(CreateNotifEventRequest):
    """An event as fanned out to listeners."""

    id: str
    workspace: str
    created_at: datetime
    created_by: str | None = None


class PublishNotifEventResponse(BaseModel):
    event: NotifEvent
    delivered: int = Field(description="Number of SSE listeners the event was handed to.")
