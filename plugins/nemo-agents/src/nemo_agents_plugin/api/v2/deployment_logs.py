# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-deployment log retrieval.

The in-memory runner backend writes one ``<deployment>.log`` per deployment
under ``system_dir()`` (typically ``~/.local/share/nemo/agents/system/``).
The ``nemo agents logs`` CLI already reads these files directly. This module
exposes the same content via the agents API so Studio can render them
without poking around on disk.

Two routes:

* ``GET /deployments/{name}/logs?tail=N`` — read the last *N* lines (default
  500) and return them as a JSON page.
* ``GET /deployments/{name}/logs/stream`` — long-lived SSE stream that
  tail-follows the file. Each new line becomes one ``id:``+``data:`` event,
  where the id is the file byte-offset after the line. Clients reconnect with
  the ``Last-Event-ID`` header to resume without gaps. The generated SDK
  doesn't model SSE, so clients use a fetch-based EventSource that can attach
  the ``Authorization`` header (native ``EventSource`` cannot).

Lines are shaped to match :class:`HelixJobLog` so Studio can reuse the
existing ``LogViewer`` component without a new schema.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from nemo_agents_plugin.api.v2._perms import DeploymentPerms
from nemo_agents_plugin.api.v2.dependencies import get_entity_client
from nemo_agents_plugin.authz import scope
from nemo_agents_plugin.entities import AgentDeployment
from nemo_agents_plugin.runner.registry import get_runner_backend
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.entity_client import NemoEntitiesClient, NemoEntityNotFoundError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# Match the timestamp prefix written by ``logging.basicConfig``-style emitters
# (e.g. ``2026-05-19 21:04:28 - INFO     - foo:11 - message``). Falls back to
# the empty string when the line is plain (uvicorn debug, tracebacks, …).
_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)")


class LogLine(BaseModel):
    """One line shaped to match ``HelixJobLog`` so Studio's LogViewer renders it as-is."""

    timestamp: str = Field(description="ISO-8601 timestamp parsed from the line; empty when absent.")
    job: str = Field(default="", description="Empty — kept for shape compatibility with jobs logs.")
    job_step: str = Field(default="", description="Empty — kept for shape compatibility.")
    job_task: str = Field(default="", description="Empty — kept for shape compatibility.")
    message: str = Field(description="The raw log line minus any parsed timestamp prefix.")


class DeploymentLogsResponse(BaseModel):
    """Response body for ``GET /deployments/{name}/logs``."""

    data: list[LogLine]
    total_lines: int = Field(description="Number of lines actually returned.")
    next_offset: int = Field(
        description="Byte offset just past the returned tail; pass as Last-Event-ID to resume the stream without gaps.",
    )


def _parse_line(raw: str) -> LogLine:
    line = raw.rstrip("\n")
    match = _TIMESTAMP_RE.match(line)
    if match:
        ts = match.group(1).replace(" ", "T")
        return LogLine(timestamp=ts, message=line[match.end() :].lstrip(" -:\t"))
    return LogLine(timestamp="", message=line)


async def _resolve_log_path(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient,
) -> Path:
    # Auth first — entity lookup scopes to the caller's workspace.
    try:
        await entity_client.get(AgentDeployment, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Deployment {name!r} not found in workspace {workspace!r}.",
        ) from exc
    from nemo_agents_plugin.runner.backend import ExternalLog, LocalLog, NotYetAvailable

    location = get_runner_backend().get_log_location(workspace, name)
    if isinstance(location, LocalLog):
        return location.path
    if isinstance(location, NotYetAvailable):
        raise HTTPException(
            status_code=404,
            detail=f"Deployment {name!r} hasn't produced any log output yet.",
        )
    if isinstance(location, ExternalLog):
        raise HTTPException(
            status_code=404,
            detail=f"Logs for {name!r} ship through a backend channel. {location.hint}".rstrip(),
        )
    raise HTTPException(status_code=404, detail=f"No log channel available for {name!r}.")


_TAIL_READ_BLOCK = 8192


def _read_tail(path: Path, n: int) -> tuple[list[str], int]:
    """Return the last *n* lines and the end byte-offset (the SSE resume cursor).

    Reads backward from EOF in blocks so cost is proportional to the requested
    tail, not the whole file — a multi-GB log with ``tail=500`` reads a few KB.
    """
    with path.open("rb") as fh:
        end_offset = fh.seek(0, 2)
        if n <= 0:
            return [], end_offset
        data = b""
        pos = end_offset
        # Read backward until we have more than n newlines (so the leading,
        # possibly-truncated fragment falls outside the last n) or hit the start.
        while pos > 0 and data.count(b"\n") <= n:
            read_size = min(_TAIL_READ_BLOCK, pos)
            pos -= read_size
            fh.seek(pos)
            data = fh.read(read_size) + data
    lines = data.decode("utf-8", errors="replace").splitlines(keepends=True)
    return lines[-n:], end_offset


_TAIL_LINE_CAP = 10_000


@router.get(
    "/deployments/{name}/logs",
    response_model=DeploymentLogsResponse,
    tags=["Agent Deployments"],
)
@scope.read
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[DeploymentPerms.READ],
)
async def get_deployment_logs(
    workspace: str,
    name: str,
    tail: int = Query(default=500, ge=0, le=_TAIL_LINE_CAP),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> DeploymentLogsResponse:
    """Return the most recent *tail* lines from the deployment's log file."""
    path = await _resolve_log_path(workspace, name, entity_client)
    raw_lines, end_offset = await asyncio.to_thread(_read_tail, path, tail)
    parsed = [_parse_line(line) for line in raw_lines]
    return DeploymentLogsResponse(data=parsed, total_lines=len(parsed), next_offset=end_offset)


def _parse_last_event_id(value: str | None) -> int | None:
    if not value:
        return None
    try:
        offset = int(value)
    except ValueError:
        return None
    return offset if offset >= 0 else None


def _open_at(path: Path, start_offset: int | None):  # noqa: ANN202 — TextIO handle
    """Open the log and seek to the resume point.

    ``start_offset is None`` → EOF (first connect; the tail covered history).
    A ``start_offset`` past EOF means the file was truncated/rotated since the
    id was issued, so replay the new file from the start instead of dropping it.
    """
    fh = path.open("r", encoding="utf-8", errors="replace")
    size = fh.seek(0, 2)  # EOF
    if start_offset is not None:
        fh.seek(start_offset if start_offset <= size else 0)
    return fh


def _read_next(fh) -> tuple[str | None, int, bool]:  # noqa: ANN001 — TextIO handle
    """Read one complete line. Returns (line|None, end_offset, truncated)."""
    pos = fh.tell()
    try:
        size = os.fstat(fh.fileno()).st_size
    except OSError:
        size = pos
    if size < pos:
        # File shrank (truncated/rotated) — restart from the top of the new file.
        fh.seek(0)
        return None, 0, True
    line = fh.readline()
    if line.endswith("\n"):
        return line, fh.tell(), False
    fh.seek(pos)  # partial line still being written — re-read once the newline lands
    return None, pos, False


async def _stream_log_lines(
    path: Path,
    start_offset: int | None,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
) -> AsyncIterator[str]:
    """Yield SSE events for new lines; each ``id:`` is the byte-offset to resume from.

    File I/O runs in a thread so the polling loop never blocks the event loop.
    The stream ends when the log file is deleted or the client disconnects.
    """
    poll_interval = 0.5
    keepalive_interval = 15.0  # heartbeat so proxies don't close idle connections
    fh = await asyncio.to_thread(_open_at, path, start_offset)
    try:
        last_keepalive = asyncio.get_running_loop().time()
        while True:
            if not path.exists():
                return  # deployment deleted — don't poll a dead file forever
            if is_disconnected is not None and await is_disconnected():
                return
            line, offset, truncated = await asyncio.to_thread(_read_next, fh)
            if truncated:
                continue
            if line is not None:
                payload = _parse_line(line).model_dump()
                yield f"id: {offset}\ndata: {json.dumps(payload)}\n\n"
                last_keepalive = asyncio.get_running_loop().time()
            else:
                now = asyncio.get_running_loop().time()
                if now - last_keepalive >= keepalive_interval:
                    yield f": keepalive {datetime.now(UTC).isoformat()}\n\n"
                    last_keepalive = now
                await asyncio.sleep(poll_interval)
    finally:
        await asyncio.to_thread(fh.close)


@router.get("/deployments/{name}/logs/stream", tags=["Agent Deployments"])
@scope.read
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[DeploymentPerms.READ],
)
async def stream_deployment_logs(
    workspace: str,
    name: str,
    request: Request,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> StreamingResponse:
    """SSE tail-follow of the deployment's log file; resumes via ``Last-Event-ID``."""
    path = await _resolve_log_path(workspace, name, entity_client)
    start_offset = _parse_last_event_id(request.headers.get("last-event-id"))
    return StreamingResponse(
        _stream_log_lines(path, start_offset, request.is_disconnected),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering so events flush immediately
        },
    )
