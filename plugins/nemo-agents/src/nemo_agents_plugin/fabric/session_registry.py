# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-local logical session registry for Fabric runtimes."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_fabric import Runtime


@dataclass(slots=True)
class FabricRuntimeSession:
    """Platform session bound to one stateful Fabric runtime."""

    session_id: str
    runtime: Runtime
    created_at: float
    last_accessed_at: float
    closing: bool = False
    invocation_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


class FabricSessionNotFoundError(LookupError):
    """Raised when a logical Fabric session is not registered."""


class FabricSessionAlreadyExistsError(ValueError):
    """Raised when a logical Fabric session ID is registered twice."""


class FabricSessionRegistryClosedError(RuntimeError):
    """Raised when registering a session after shutdown has started."""


class FabricSessionRegistry:
    """Maintain the process-local mapping from Platform sessions to runtimes."""

    def __init__(self) -> None:
        self._sessions: dict[str, FabricRuntimeSession] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def register(
        self,
        runtime: Runtime,
        *,
        session_id: str | None = None,
    ) -> FabricRuntimeSession:
        """Bind a new opaque Platform session ID to a Fabric runtime."""
        resolved_session_id = session_id or str(uuid.uuid4())
        now = time.monotonic()
        session = FabricRuntimeSession(
            session_id=resolved_session_id,
            runtime=runtime,
            created_at=now,
            last_accessed_at=now,
        )

        async with self._lock:
            if self._closed:
                raise FabricSessionRegistryClosedError("Fabric session registry is closed.")
            if resolved_session_id in self._sessions:
                raise FabricSessionAlreadyExistsError(f"Fabric session '{resolved_session_id}' is already registered.")
            self._sessions[resolved_session_id] = session
        return session

    async def get(self, session_id: str) -> FabricRuntimeSession:
        """Return an active session and update its last-accessed time."""
        async with self._lock:
            try:
                session = self._sessions[session_id]
            except KeyError as error:
                raise FabricSessionNotFoundError(f"Fabric session '{session_id}' was not found.") from error
            session.last_accessed_at = time.monotonic()
            return session

    async def refresh_activity(self, session: FabricRuntimeSession) -> None:
        """Refresh activity for a session that is still registered."""
        async with self._lock:
            if self._sessions.get(session.session_id) is session:
                session.last_accessed_at = time.monotonic()

    async def remove(self, session_id: str) -> FabricRuntimeSession | None:
        """Mark a session as closing, then remove and return it."""
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is not None:
                session.closing = True
            return session

    async def remove_expired(self, *, idle_timeout_seconds: float) -> list[FabricRuntimeSession]:
        """Remove inactive sessions that are not currently invoking."""
        cutoff = time.monotonic() - idle_timeout_seconds
        expired: list[FabricRuntimeSession] = []
        async with self._lock:
            for session_id, session in list(self._sessions.items()):
                if session.last_accessed_at > cutoff or session.invocation_lock.locked():
                    continue
                session.closing = True
                expired.append(session)
                del self._sessions[session_id]
        return expired

    async def drain(self) -> list[FabricRuntimeSession]:
        """Close the registry and remove all remaining sessions."""
        async with self._lock:
            self._closed = True
            sessions = list(self._sessions.values())
            self._sessions.clear()
            for session in sessions:
                session.closing = True
            return sessions

    async def count(self) -> int:
        """Return the number of registered logical sessions."""
        async with self._lock:
            return len(self._sessions)
