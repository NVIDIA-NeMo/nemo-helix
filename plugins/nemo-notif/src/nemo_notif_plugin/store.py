# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistence of notif events in the entity store.

The entity store cannot filter on "principal not in read_by", so listing scans the most recent
``SCAN_WINDOW`` events of the workspace and filters in Python. Older events never count as unread.
"""

from __future__ import annotations

import uuid

from nemo_helix_plugin.entity_client import NemoEntitiesClient, NemoEntityConflictError
from nemo_notif_plugin.entities import NotifEventEntity
from nemo_notif_plugin.types import CreateNotifEventRequest, NotifEvent

# The entity store's maximum page size.
SCAN_WINDOW = 1000
# Concurrent read-state updates race on db_version; retry a few times before giving up.
UPDATE_ATTEMPTS = 3
# Shared reader key when the request has no principal (auth disabled).
ANONYMOUS_READER = "anonymous"


def reader_key(principal_id: str | None) -> str:
    return principal_id or ANONYMOUS_READER


def to_event(entity: NotifEventEntity, principal_id: str | None) -> NotifEvent:
    return NotifEvent(
        id=entity.name,
        workspace=entity.workspace,
        type=entity.type,
        title=entity.title,
        message=entity.message,
        severity=entity.severity,
        link=entity.link,
        data=entity.data,
        created_at=entity.created_at,
        created_by=entity.created_by,
        read=reader_key(principal_id) in entity.read_by,
    )


async def create_event(
    entity_client: NemoEntitiesClient, workspace: str, body: CreateNotifEventRequest
) -> NotifEventEntity:
    entity = NotifEventEntity(name=f"evt-{uuid.uuid4().hex}", workspace=workspace, **body.model_dump())
    return await entity_client.create(entity)


async def recent_events(entity_client: NemoEntitiesClient, workspace: str) -> list[NotifEventEntity]:
    result = await entity_client.list(
        NotifEventEntity, workspace=workspace, sort="-created_at", page=1, page_size=SCAN_WINDOW
    )
    return list(result.data)


async def set_read(
    entity_client: NemoEntitiesClient,
    workspace: str,
    event_id: str,
    principal_id: str | None,
    read: bool,
) -> NotifEventEntity:
    """Add or remove the caller from ``read_by``. Raises NemoEntityNotFoundError for unknown ids."""
    key = reader_key(principal_id)
    for attempt in range(UPDATE_ATTEMPTS):
        entity = await entity_client.get(NotifEventEntity, name=event_id, workspace=workspace)
        if (key in entity.read_by) == read:
            return entity
        entity.read_by = [*entity.read_by, key] if read else [r for r in entity.read_by if r != key]
        try:
            return await entity_client.update(entity)
        except NemoEntityConflictError:
            if attempt == UPDATE_ATTEMPTS - 1:
                raise
    raise AssertionError("unreachable")


async def mark_all_read(entity_client: NemoEntitiesClient, workspace: str, principal_id: str | None) -> int:
    key = reader_key(principal_id)
    marked = 0
    for entity in await recent_events(entity_client, workspace):
        if key not in entity.read_by:
            await set_read(entity_client, workspace, entity.name, principal_id, read=True)
            marked += 1
    return marked
