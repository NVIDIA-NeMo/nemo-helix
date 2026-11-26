# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live event relay for a war-game run: the run POSTs events here, Studio polls for them.

Each event is stored as an :class:`AgentHardenerRunEvent` child of its run in the Entity Store, so every
API replica serves the same history and no volume is shared. The API numbers events in arrival order
(``seq``, the id Studio polls after); the store's unique child name makes concurrent ingests for one run
take different numbers. Runs recorded before this kept their events in a fileset (``events_fileset``),
which is still served for a run that has no event children.
"""

from __future__ import annotations

import contextlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from nemo_agent_hardener_plugin._perms import AgentHardenerRunPerms
from nemo_agent_hardener_plugin.authz import scope
from nemo_agent_hardener_plugin.entities import AgentHardenerRun, AgentHardenerRunEvent, event_name
from nemo_agent_hardener_plugin.filesets import download_fileset
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    get_entity_client,
)
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

_EVENT_PAGE_SIZE = 200
_SEQ_ATTEMPTS = 5
_RUN_ID_CACHE_LIMIT = 4096
_run_ids: dict[tuple[str, str], str] = {}


def _get_client() -> NemoClient:
    from nemo_helix_plugin.client_provider import get_nemo_client

    return get_nemo_client(as_service="agent-hardener", internal=True)


class EventIn(BaseModel):
    """Body for ``POST /runs/{name}/events`` — one event emitted by the run's EventBus."""

    event: str
    payload: dict[str, Any] = {}
    sink_id: str | None = Field(
        default=None, description="Id of the sending process's sink; with `seq`, makes a resent event idempotent."
    )
    seq: int | None = Field(
        default=None, ge=1, description="The sender's own counter for this event, increasing within one `sink_id`."
    )


class EventsResponse(BaseModel):
    """Response for GET /runs/{name}/events — events after the given sequence id."""

    events: list[dict[str, Any]]


async def _run_id(entity_client: NemoEntitiesClient, workspace: str, name: str) -> str | None:
    """The run's entity id, or None when there is no such run. Cached: a run's id never changes."""
    key = (workspace, name)
    if (run_id := _run_ids.get(key)) is None:
        try:
            run = await entity_client.get(AgentHardenerRun, name=name, workspace=workspace)
        except NemoEntityNotFoundError:
            return None
        if len(_run_ids) >= _RUN_ID_CACHE_LIMIT:
            _run_ids.clear()
        run_id = _run_ids[key] = run.id
    return run_id


def forget_run(workspace: str, name: str) -> None:
    """Drop a deleted run from the id cache; the store deletes its event children with it."""
    _run_ids.pop((workspace, name), None)


def _children_of(run_id: str, *conditions: FilterOperation) -> FilterOperation:
    parent = ComparisonOperation(field="parent", operator=FilterOperator.EQ, value=run_id)
    return LogicalOperation(operator=FilterOperator.AND, operations=[parent, *conditions])


async def _list_events(
    entity_client: NemoEntitiesClient,
    workspace: str,
    run_id: str,
    *conditions: FilterOperation,
    sort: str,
    page_size: int,
) -> list[AgentHardenerRunEvent]:
    page = await entity_client.list(
        AgentHardenerRunEvent,
        workspace=workspace,
        filter_operation=_children_of(run_id, *conditions),
        sort=sort,
        page_size=page_size,
    )
    return page.data


async def _last_seq(entity_client: NemoEntitiesClient, workspace: str, run_id: str) -> int:
    latest = await _list_events(entity_client, workspace, run_id, sort="-name", page_size=1)
    return latest[0].seq if latest else 0


async def _already_stored(
    entity_client: NemoEntitiesClient, workspace: str, run_id: str, sink_id: str, sink_seq: int
) -> bool:
    match = await _list_events(
        entity_client,
        workspace,
        run_id,
        ComparisonOperation(field="data.sink_id", operator=FilterOperator.EQ, value=sink_id),
        ComparisonOperation(field="data.sink_seq", operator=FilterOperator.EQ, value=sink_seq),
        sort="name",
        page_size=1,
    )
    return bool(match)


async def _events_after(
    entity_client: NemoEntitiesClient, workspace: str, run_id: str, after: int
) -> list[AgentHardenerRunEvent]:
    """Every event with ``seq > after``, in order.

    Pages by name rather than by offset, so an ingest landing mid-read cannot shift a page boundary.
    """
    events: list[AgentHardenerRunEvent] = []
    cursor = event_name(after)
    while True:
        page = await _list_events(
            entity_client,
            workspace,
            run_id,
            ComparisonOperation(field="name", operator=FilterOperator.GT, value=cursor),
            sort="name",
            page_size=_EVENT_PAGE_SIZE,
        )
        events.extend(page)
        if len(page) < _EVENT_PAGE_SIZE:
            return events
        cursor = page[-1].name


router = APIRouter()


@router.post("/runs/{name}/events", status_code=204, tags=["Agent Hardener Events"])
@scope.write
# The war-game run posts its events here; allow both a human operator (local CLI) and the
# job's service principal (platform-executed run) as ingest callers.
@path_rule(
    callers=[CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL],
    permissions=[AgentHardenerRunPerms.EVENTS_WRITE],
)
async def ingest_event(
    workspace: str,
    name: str,
    body: EventIn,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> None:
    """Ingest one run event (the run's EventBus POSTs here); a resent event is stored once."""
    run_id = await _run_id(entity_client, workspace, name)
    if run_id is None:
        raise HTTPException(status_code=404, detail=f"AgentHardenerRun '{name}' not found in workspace '{workspace}'.")
    if body.sink_id and body.seq and await _already_stored(entity_client, workspace, run_id, body.sink_id, body.seq):
        return
    for _ in range(_SEQ_ATTEMPTS):
        seq = await _last_seq(entity_client, workspace, run_id) + 1
        event = AgentHardenerRunEvent(
            name=event_name(seq),
            workspace=workspace,
            seq=seq,
            event=body.event,
            payload=body.payload,
            sink_id=body.sink_id or "",
            sink_seq=body.seq or 0,
        )
        event._parent = run_id
        try:
            await entity_client.create(event)
            return
        except NemoEntityConflictError:
            continue  # another ingest took this seq; number after it
    raise HTTPException(status_code=503, detail=f"Too many concurrent events for run {name!r}; retry.")


@router.get("/runs/{name}/events", tags=["Agent Hardener Events"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AgentHardenerRunPerms.EVENTS_READ])
async def get_events(
    workspace: str,
    name: str,
    after: int = 0,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> EventsResponse:
    """Return the run's events with sequence id greater than *after*.

    A run recorded before events moved to the Entity Store is served from its ``events_fileset``.
    """
    run_id = await _run_id(entity_client, workspace, name)
    if run_id is None:
        return EventsResponse(events=[])
    stored = await _events_after(entity_client, workspace, run_id, after)
    if stored or (after > 0 and await _last_seq(entity_client, workspace, run_id)):
        return EventsResponse(events=[{"id": e.seq, "event": e.event, "payload": e.payload} for e in stored])

    try:
        run = await entity_client.get(AgentHardenerRun, name=name, workspace=workspace)
    except NemoEntityNotFoundError:
        return EventsResponse(events=[])
    if not run.events_fileset:
        return EventsResponse(events=[])
    # The download is blocking sync I/O that calls back into this same platform; on the event loop it
    # would self-deadlock (the server cannot answer its own request), so it runs on a worker thread.
    return EventsResponse(events=await run_in_threadpool(_legacy_events, name, run.events_fileset, after))


def _legacy_events(name: str, fileset_ref: str, after: int) -> list[dict[str, Any]]:
    """Blocking: the events of a run whose history is an ``events.jsonl`` fileset (id = line number).

    Empty only when the fileset is gone; a failed download is a 502, since an empty list would read as a
    run that emitted no events.
    """
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dest = download_fileset(_get_client(), fileset_ref, Path(tmp))
            log = next(iter(sorted(dest.rglob("*.jsonl"))), None)
            return _read_jsonl(log, after) if log else []
    except NotFoundError:
        return []
    except Exception as exc:
        logger.warning("Fileset fallback failed for run %r events", name, exc_info=True)
        raise HTTPException(
            status_code=502, detail=f"Could not load the persisted events for run {name!r}: {exc}"
        ) from exc


def _read_jsonl(path: Path, after: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line_no <= after or not line.strip():
                continue
            with contextlib.suppress(json.JSONDecodeError):
                events.append({"id": line_no, **json.loads(line)})
    return events
