# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kern session controller — registered under ``nemo.controllers``.

Connects to the kern status Unix socket, reads the NDJSON stream of session
events, and upserts :class:`~nemo_kern.entities.KernSession` entities into
the platform entity store.  Prunes sessions that have not been updated in
more than :attr:`~nemo_kern.config.KernConfig.session_stale_seconds` seconds.

Wire protocol: ``packages/status/protocol.ts`` — messages are newline-delimited
JSON objects with a ``t`` field:

- ``hello``     → agent, model, cwd, pid, session, host
- ``state``     → state (working|idle), turn, ctx (0..1), cost, ts
- ``hold``      → id, summary, tool, tier, rule, options, via, ts
- ``release``   → id, verdict, via
- ``exit``      → code
- ``heartbeat`` → (no payload)

The controller maintains a local in-process cache (``_sessions``) while the
socket is open; on each reconcile cycle it flushes stale entries and upserts
active ones.

Reconnection: if the socket is missing or the connection drops (the kern
daemon is not running), the controller logs a warning and waits until the
next cycle.  It never crashes the platform reconcile loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, ClassVar

from nemo_helix_plugin.controller import NemoController

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-process session state
# ---------------------------------------------------------------------------


class _SessionState:
    """Tracks the current state for a single agent session."""

    __slots__ = (
        "agent",
        "model",
        "status",
        "ctx_fill",
        "cost",
        "hold",
        "updated_at",
    )

    def __init__(self, agent: str) -> None:
        self.agent: str = agent
        self.model: str | None = None
        self.status: str = "idle"
        self.ctx_fill: float | None = None
        self.cost: float = 0.0
        self.hold: str | None = None
        self.updated_at: float = time.time()

    def touch(self) -> None:
        self.updated_at = time.time()


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class KernSessionController(NemoController):
    """Reconcile-loop controller for kern live sessions.

    On each cycle:
    1. Drains buffered socket messages accumulated since the last cycle.
    2. Prunes sessions not updated within ``session_stale_seconds``.
    3. Upserts the remaining sessions as ``KernSession`` entities.

    Registered under the ``nemo.controllers`` entry-point group as
    ``kern-session-controller``.
    """

    name: ClassVar[str] = "kern-session-controller"
    dependencies: ClassVar[list[str]] = []

    def __init__(self) -> None:
        # In-process session cache: session_id → _SessionState
        self._sessions: dict[str, _SessionState] = {}
        # asyncio reader task (started in on_startup)
        self._reader_task: asyncio.Task[None] | None = None
        # Queue for messages parsed by the reader task
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=2048)
        # Config-driven state (set in on_startup)
        self._interval: float = 10.0
        self._stale_seconds: float = 300.0
        self._entities: Any | None = None  # NemoEntitiesClient

    @property
    def interval_seconds(self) -> float:
        return self._interval

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_startup(self) -> None:
        from nemo_kern.config import KernConfig

        config = KernConfig.get()
        self._stale_seconds = config.session_stale_seconds
        self._interval = min(self._stale_seconds / 2, 10.0)
        self._workspace = config.workspace

        # Build a service-principal entity client for background upserts.
        try:
            from nhx.common.sdk_factory import get_async_platform_sdk
            from nemo_helix_plugin.client.adapter import client_from_platform
            from nemo_helix_plugin.entities.client import AsyncEntitiesClient
            from nemo_helix_plugin.entity_client import NemoEntitiesClient

            sdk = get_async_platform_sdk(as_service="kern", internal=True)
            typed_client = client_from_platform(sdk, AsyncEntitiesClient)
            self._entities = NemoEntitiesClient(typed_client)
            logger.info("KernSessionController: entity client ready")
        except Exception:
            logger.warning(
                "KernSessionController: entity client unavailable — session upserts disabled",
                exc_info=True,
            )

        # Launch the background status-log tailer.
        self._reader_task = asyncio.create_task(
            self._socket_reader_loop(config.status_log),
            name="kern-status-log-tailer",
        )
        logger.info(
            "KernSessionController started (interval=%.1fs stale=%.1fs log=%s)",
            self._interval,
            self._stale_seconds,
            config.status_log,
        )

    async def on_shutdown(self) -> None:
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        logger.info("KernSessionController shut down.")

    # ------------------------------------------------------------------
    # Reconcile
    # ------------------------------------------------------------------

    async def list_objects(self) -> list:
        """Drain the message queue and return the sessions to reconcile."""
        self._drain_queue()
        self._prune_stale()
        return list(self._sessions.keys())

    async def reconcile_one(self, obj: object) -> None:
        """Upsert a single session entity."""
        session_id = str(obj)
        state = self._sessions.get(session_id)
        if state is None:
            return
        if self._entities is None:
            return

        from hashlib import sha1

        from nemo_kern.entities import KernSession

        # Entity names must match ^[a-z][a-z0-9-@.+_]{1,62}$ — the session id is a
        # path, so derive a stable, valid name from it instead.
        name = "kern-" + sha1(session_id.encode("utf-8")).hexdigest()[:16]

        try:
            # Try to get the existing entity and update it.
            existing = await self._entities.get(KernSession, name=name, workspace=self._workspace)
            existing.agent = state.agent
            existing.session_path = session_id
            existing.model = state.model
            existing.status = state.status
            existing.ctx_fill = state.ctx_fill
            existing.cost = state.cost
            existing.hold = state.hold
            existing.last_seen = state.updated_at
            await self._entities.update(existing)
        except Exception:
            # Entity does not exist yet — create it.
            try:
                session_entity = KernSession(
                    name=name,
                    workspace=self._workspace,
                    agent=state.agent,
                    session_path=session_id,
                    model=state.model,
                    status=state.status,
                    ctx_fill=state.ctx_fill,
                    cost=state.cost,
                    hold=state.hold,
                    last_seen=state.updated_at,
                )
                await self._entities.create(session_entity)
            except Exception:
                logger.warning(
                    "KernSessionController: upsert failed for session %s", session_id, exc_info=True
                )

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _drain_queue(self) -> None:
        """Drain all buffered NDJSON messages from the queue into _sessions."""
        while not self._queue.empty():
            try:
                msg = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._apply_message(msg)

    def _apply_message(self, msg: dict[str, Any]) -> None:
        """Apply a single protocol message to the session cache."""
        t = msg.get("t")
        if t == "hello":
            # hello messages identify the session; they carry the session path.
            # Use session path as the stable ID, falling back to agent+pid.
            session_path: str = msg.get("session") or ""
            session_id = session_path or f"{msg.get('agent', 'unknown')}:{msg.get('pid', 0)}"
            if session_id not in self._sessions:
                self._sessions[session_id] = _SessionState(
                    agent=msg.get("agent", "")
                )
            s = self._sessions[session_id]
            s.agent = msg.get("agent", s.agent)
            s.model = msg.get("model") or s.model
            s.touch()

        elif t == "state":
            # state messages don't directly carry session_id; they come on the
            # same connection as the preceding hello. We update whichever session
            # was most recently seen (the one whose reader task sent this message).
            # Because the socket is per-session, we tag messages with their
            # originating session_id in the reader task.
            session_id = msg.get("_session_id")
            if session_id and session_id in self._sessions:
                s = self._sessions[session_id]
                s.status = msg.get("state", s.status)
                ctx = msg.get("ctx")
                s.ctx_fill = float(ctx) if ctx is not None else s.ctx_fill
                s.cost = float(msg.get("cost", s.cost))
                s.touch()

        elif t == "hold":
            session_id = msg.get("_session_id")
            if session_id and session_id in self._sessions:
                s = self._sessions[session_id]
                s.hold = msg.get("summary")
                s.touch()

        elif t == "release":
            session_id = msg.get("_session_id")
            if session_id and session_id in self._sessions:
                self._sessions[session_id].hold = None
                self._sessions[session_id].touch()

        elif t == "exit":
            session_id = msg.get("_session_id")
            if session_id:
                self._sessions.pop(session_id, None)

    def _prune_stale(self) -> None:
        """Remove sessions that have not been updated recently."""
        cutoff = time.time() - self._stale_seconds
        stale = [sid for sid, s in self._sessions.items() if s.updated_at < cutoff]
        for sid in stale:
            logger.debug("KernSessionController: pruning stale session %s", sid)
            self._sessions.pop(sid, None)

    # ------------------------------------------------------------------
    # Background status-log tailer
    # ------------------------------------------------------------------

    async def _socket_reader_loop(self, log_path: object) -> None:
        """Tail the kern status NDJSON mirror file and forward messages.

        The pi-kern-status extension appends one JSON line per status message
        to ``status.jsonl`` (each stamped with ``_session_id``). On start we
        backfill the most recent lines so sessions active before this process
        started still show up, then follow the file tail. Runs until cancelled.
        """
        from pathlib import Path

        path = Path(str(log_path))
        retry_delay = 5.0
        poll_interval = 1.0
        last_size = 0
        backfilled = False

        def _enqueue(msg: dict[str, Any]) -> None:
            try:
                self._queue.put_nowait(msg)
            except asyncio.QueueFull:
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self._queue.put_nowait(msg)
                except asyncio.QueueFull:
                    pass

        while True:
            if not path.exists():
                logger.debug(
                    "KernSessionController: status log not present at %s, retrying in %.0fs",
                    path,
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)
                continue

            try:
                with open(path, "rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    if not backfilled and size > 0:
                        # Backfill the most recent ~64 KiB so recently active
                        # sessions appear immediately after startup.
                        f.seek(max(0, size - 65536))
                        tail = f.read().decode("utf-8", errors="replace")
                        for line in tail.splitlines():
                            if not line.strip():
                                continue
                            try:
                                _enqueue(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                        last_size = size
                        backfilled = True
                        logger.info(
                            "KernSessionController: backfilled %d lines from %s",
                            len(tail.splitlines()),
                            path,
                        )
                        continue

                    if size > last_size:
                        f.seek(last_size)
                        data = f.read().decode("utf-8", errors="replace")
                        for line in data.splitlines():
                            if not line.strip():
                                continue
                            try:
                                _enqueue(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                        last_size = size

            except OSError as exc:
                logger.warning(
                    "KernSessionController: status log read error (%s), retrying in %.0fs",
                    exc,
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)
                continue
            except asyncio.CancelledError:
                logger.debug("KernSessionController: status log tailer cancelled")
                return
            except Exception:
                logger.exception(
                    "KernSessionController: unexpected error in status log tailer, retrying in %.0fs",
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)
                continue

            await asyncio.sleep(poll_interval)
