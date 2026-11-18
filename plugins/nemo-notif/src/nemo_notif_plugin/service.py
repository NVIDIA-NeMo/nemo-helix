# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Notif plugin service, mounted at ``/apis/notif``.

``POST /v2/workspaces/{workspace}/events`` publishes an ephemeral event and
``GET /v2/workspaces/{workspace}/events/stream`` is the SSE stream it fans out to. Nothing is
persisted, and ``Last-Event-ID`` is accepted but cannot replay missed events.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timezone
from typing import ClassVar

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.dependencies import get_effective_principal_id
from nemo_helix_plugin.service import NemoService, RouterSpec
from nemo_notif_plugin.authz import NotifEventPerms, scope
from nemo_notif_plugin.broker import Listener, NotifBroker, broker
from nemo_notif_plugin.types import CreateNotifEventRequest, NotifEvent, PublishNotifEventResponse

# Comment lines keep idle connections open through proxies that drop silent streams.
KEEPALIVE_SECONDS = 15.0

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",  # disable proxy buffering so events flush immediately
}


class NotifService(NemoService):
    """Ephemeral event fan-out to SSE listeners."""

    name: ClassVar[str] = "notif"
    dependencies: ClassVar[list[str]] = []

    def get_routers(self) -> list[RouterSpec]:
        return [
            RouterSpec(
                build_events_router(broker),
                tag="Notif Events",
                description="Publish ephemeral notification events and stream them over SSE.",
                prefix="/v2/workspaces/{workspace}",
            )
        ]


def format_sse(event: NotifEvent) -> str:
    # Studio's SSE reader only parses `id:` and `data:`, so the type travels inside `data` too.
    return f"id: {event.id}\nevent: {event.type}\ndata: {event.model_dump_json()}\n\n"


async def stream_events(
    listener: Listener,
    is_disconnected: Callable[[], Awaitable[bool]],
    keepalive_seconds: float = KEEPALIVE_SECONDS,
) -> AsyncIterator[str]:
    # Flush something immediately so clients and proxies see the stream open.
    yield ": connected\n\n"
    while True:
        try:
            event = await asyncio.wait_for(listener.queue.get(), timeout=keepalive_seconds)
        except TimeoutError:
            if await is_disconnected():
                return
            yield ": keepalive\n\n"
            continue
        if event is None:  # the broker cut this listener off for falling behind
            return
        yield format_sse(event)


def build_events_router(notif_broker: NotifBroker) -> APIRouter:
    router = APIRouter()

    @router.post("/events", status_code=201, response_model=PublishNotifEventResponse)
    @scope.write
    @path_rule(
        callers=[CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL],
        permissions=[NotifEventPerms.CREATE],
    )
    async def publish_event(
        workspace: str,
        body: CreateNotifEventRequest,
        effective_principal_id: str = Depends(get_effective_principal_id),
    ) -> PublishNotifEventResponse:
        """Fan an event out to every listener currently streaming this workspace."""
        event = NotifEvent(
            **body.model_dump(),
            id=str(uuid.uuid4()),
            workspace=workspace,
            created_at=datetime.now(timezone.utc),
            created_by=effective_principal_id or None,
        )
        return PublishNotifEventResponse(event=event, delivered=notif_broker.publish(event))

    @router.get("/events/stream", response_class=StreamingResponse)
    @scope.read
    @path_rule(
        callers=[CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL],
        permissions=[NotifEventPerms.READ],
    )
    async def stream_workspace_events(
        workspace: str,
        request: Request,
        types: list[str] | None = Query(default=None, description="Only stream these event types."),
    ) -> StreamingResponse:
        """SSE stream of events published to this workspace while connected."""

        async def body() -> AsyncIterator[str]:
            async with notif_broker.listen(workspace, frozenset(types) if types else None) as listener:
                async for chunk in stream_events(listener, request.is_disconnected):
                    yield chunk

        return StreamingResponse(body(), media_type="text/event-stream", headers=SSE_HEADERS)

    return router
