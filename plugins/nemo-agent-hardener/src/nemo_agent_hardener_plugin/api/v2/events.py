# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable event relay for a war-game run: the run POSTs events here, Studio polls for them.

Poll model: agent-hardener's EventBus POSTs each event to ``POST /runs/{name}/events``; the
:class:`EventHub` appends it to a per-run ``events.jsonl`` (write-through). The file is the
durable history — the full per-agent transcript survives a service restart. Each event's id is
its 1-based line number in the file.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from nemo_agent_hardener_plugin._perms import AgentHardenerRunPerms
from nemo_agent_hardener_plugin.authz import scope
from nemo_agent_hardener_plugin.config import AgentHardenerConfig
from nemo_agent_hardener_plugin.entities import AGENT_HARDENER_RUN_TYPE
from nemo_agent_hardener_plugin.filesets import download_fileset
from nemo_platform_plugin.authz import CallerKind, path_rule
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.entities.client import EntitiesClient
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)


def _get_sdk() -> Any:
    from nemo_platform_plugin.sdk_provider import get_platform_sdk

    return get_platform_sdk(as_service="agent-hardener", internal=True)


def _events_path(workspace: str, run_name: str) -> Path:
    """Durable per-run events log: ``<state_dir>/run-events/<workspace>/<safe-run-name>.jsonl``."""
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in run_name) or "run"
    return AgentHardenerConfig.get().state_dir / "run-events" / workspace / f"{safe}.jsonl"


class EventIn(BaseModel):
    """Body for ``POST /runs/{name}/events`` — one event emitted by the run's EventBus."""

    event: str
    payload: dict[str, Any] = {}


class _RunStream:
    """One run's durable ``events.jsonl`` (all on the event loop).

    The file is the source of truth for history + sequence ids (id == 1-based line number); ``_seq`` is
    seeded from the existing line count so ids stay monotonic across a restart.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._seq = self._line_count()

    def _line_count(self) -> int:
        if not self._path.exists():
            return 0
        with self._path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    def publish(self, event: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        self._seq += 1

    def history(self, after_id: int) -> list[tuple[int, dict[str, Any]]]:
        """Replay persisted events with a line-number id greater than *after_id*."""
        if not self._path.exists():
            return []
        items: list[tuple[int, dict[str, Any]]] = []
        with self._path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if line_no <= after_id or not line.strip():
                    continue
                with contextlib.suppress(json.JSONDecodeError):
                    items.append((line_no, json.loads(line)))
        return items


class EventHub:
    """Per-run event streams for this plugin process (created on first access).

    Keyed by ``(workspace, run_name)`` so runs sharing a name across workspaces never cross streams.
    """

    def __init__(self) -> None:
        self._streams: dict[tuple[str, str], _RunStream] = {}

    def stream(self, workspace: str, run_name: str) -> _RunStream:
        key = (workspace, run_name)
        if key not in self._streams:
            self._streams[key] = _RunStream(_events_path(workspace, run_name))
        return self._streams[key]


hub = EventHub()
router = APIRouter()


@router.post("/runs/{name}/events", status_code=204, tags=["Agent Hardener Events"])
@scope.write
# The war-game run posts its events here; allow both a human operator (local CLI) and the
# job's service principal (platform-executed run) as ingest callers.
@path_rule(
    callers=[CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL],
    permissions=[AgentHardenerRunPerms.EVENTS_WRITE],
)
async def ingest_event(workspace: str, name: str, body: EventIn) -> None:
    """Ingest one run event (the run's EventBus POSTs here)."""
    hub.stream(workspace, name).publish({"event": body.event, "payload": body.payload})


class EventsResponse(BaseModel):
    """Response for GET /runs/{name}/events — events after the given sequence id."""

    events: list[dict[str, Any]]


@router.get("/runs/{name}/events", tags=["Agent Hardener Events"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AgentHardenerRunPerms.EVENTS_READ])
async def get_events(workspace: str, name: str, after: int = 0) -> EventsResponse:
    """Return all persisted run events with sequence id greater than *after*.

    Falls back to downloading from the run's ``events_fileset`` when the local
    file is absent (e.g. after a pod restart).
    """
    stream = hub.stream(workspace, name)
    result = stream.history(after_id=after)

    if not result and not stream._path.exists():
        # The fallback does blocking sync I/O (SDK entity lookup + fileset download) that calls back into
        # this same platform. Running it on the event loop self-deadlocks — the server can't answer its own
        # lookup, so the SDK retries for ~181s while every other request (incl. the inference gateway) is
        # frozen. Offload to a worker thread so the loop stays free and the lookup resolves promptly.
        result = await run_in_threadpool(_fileset_fallback, workspace, name, stream, after)

    return EventsResponse(events=[{"id": seq, **event} for seq, event in result])


def _fileset_fallback(workspace: str, name: str, stream: Any, after: int) -> list[tuple[int, dict[str, Any]]]:
    """Blocking: fetch the run's ``events_fileset`` and re-read history. Must run off the event loop."""
    try:
        sdk = _get_sdk()
        # get_entity_by_name returns a generic Entity — its domain fields live under `.data`
        # (same access pattern as sdk.py::_run_to_dict), not as top-level attributes.
        run = (
            client_from_platform(sdk, EntitiesClient)
            .get_entity_by_name(
                name=name,
                entity_type=AGENT_HARDENER_RUN_TYPE,
                workspace=workspace,
            )
            .data()
        )
        fileset_ref = (getattr(run, "data", None) or {}).get("events_fileset")
        if fileset_ref:
            download_fileset(sdk, fileset_ref, stream._path.parent)
            return stream.history(after_id=after)
    except Exception:
        logger.warning("Fileset fallback failed for run %r events; returning empty", name, exc_info=True)
    return []
