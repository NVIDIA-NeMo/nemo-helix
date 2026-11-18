# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Notif plugin service, mounted at ``/apis/notif``.

Events are stored as ``notif_event`` entities and fanned out live to SSE listeners:

- ``POST /v2/workspaces/{workspace}/events`` stores an event and pushes it to listeners.
- ``GET  /v2/workspaces/{workspace}/events`` lists recent events with the caller's read state.
- ``GET  /v2/workspaces/{workspace}/events/stream`` is the SSE stream.
- ``POST /v2/workspaces/{workspace}/events/{event_id}/read`` (and ``/unread``) set read state.
- ``POST /v2/workspaces/{workspace}/events/read-all`` marks every recent event read.

The stream does not replay: clients load ``GET /events`` first, then apply streamed events.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import ClassVar

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.dependencies import get_effective_principal_id
from nemo_helix_plugin.entity_client import NemoEntitiesClient, NemoEntityNotFoundError, get_entity_client
from nemo_helix_plugin.service import NemoService, RouterSpec
from nemo_notif_plugin import store
from nemo_notif_plugin.authz import NotifEventPerms, scope
from nemo_notif_plugin.broker import Listener, NotifBroker, broker
from nemo_notif_plugin.types import (
    CreateNotifEventRequest,
    MarkAllReadResponse,
    NotifEvent,
    NotifEventList,
    PublishNotifEventResponse,
)

CALLERS = [CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL]

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
    @path_rule(callers=CALLERS, permissions=[NotifEventPerms.CREATE])
    async def publish_event(
        workspace: str,
        body: CreateNotifEventRequest,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
    ) -> PublishNotifEventResponse:
        """Store an event, then fan it out to every listener currently streaming this workspace."""
        entity = await store.create_event(entity_client, workspace, body)
        # New events are unread for everyone, so one payload serves every listener.
        event = store.to_event(entity, principal_id=None)
        return PublishNotifEventResponse(event=event, delivered=notif_broker.publish(event))

    @router.get("/events", response_model=NotifEventList)
    @scope.read
    @path_rule(callers=CALLERS, permissions=[NotifEventPerms.READ])
    async def list_events(
        workspace: str,
        limit: int = Query(default=50, ge=1, le=store.SCAN_WINDOW),
        unread_only: bool = Query(default=False),
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
        effective_principal_id: str = Depends(get_effective_principal_id),
    ) -> NotifEventList:
        """Most recent events first, with the caller's read state and unread count."""
        events = [
            store.to_event(e, effective_principal_id) for e in await store.recent_events(entity_client, workspace)
        ]
        unread = [e for e in events if not e.read]
        return NotifEventList(data=(unread if unread_only else events)[:limit], unread_count=len(unread))

    @router.post("/events/read-all", response_model=MarkAllReadResponse)
    @scope.write
    @path_rule(callers=CALLERS, permissions=[NotifEventPerms.READ])
    async def mark_all_events_read(
        workspace: str,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
        effective_principal_id: str = Depends(get_effective_principal_id),
    ) -> MarkAllReadResponse:
        """Mark every recent event read for the caller."""
        return MarkAllReadResponse(marked=await store.mark_all_read(entity_client, workspace, effective_principal_id))

    async def _set_read(
        workspace: str,
        event_id: str,
        read: bool,
        entity_client: NemoEntitiesClient,
        principal_id: str,
    ) -> NotifEvent:
        try:
            entity = await store.set_read(entity_client, workspace, event_id, principal_id, read)
        except NemoEntityNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found") from exc
        return store.to_event(entity, principal_id)

    @router.post("/events/{event_id}/read", response_model=NotifEvent)
    @scope.write
    @path_rule(callers=CALLERS, permissions=[NotifEventPerms.READ])
    async def mark_event_read(
        workspace: str,
        event_id: str,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
        effective_principal_id: str = Depends(get_effective_principal_id),
    ) -> NotifEvent:
        """Mark one event read for the caller."""
        return await _set_read(workspace, event_id, True, entity_client, effective_principal_id)

    @router.post("/events/{event_id}/unread", response_model=NotifEvent)
    @scope.write
    @path_rule(callers=CALLERS, permissions=[NotifEventPerms.READ])
    async def mark_event_unread(
        workspace: str,
        event_id: str,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
        effective_principal_id: str = Depends(get_effective_principal_id),
    ) -> NotifEvent:
        """Mark one event unread for the caller."""
        return await _set_read(workspace, event_id, False, entity_client, effective_principal_id)

    @router.get("/events/stream", response_class=StreamingResponse)
    @scope.read
    @path_rule(callers=CALLERS, permissions=[NotifEventPerms.READ])
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
