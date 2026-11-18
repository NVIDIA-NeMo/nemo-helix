# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process fan-out of ephemeral events to SSE listeners.

Events are never stored: a listener only sees events published while it is connected, and only
listeners attached to the same API process see them. That is exact for the default local setup
(one process) and lossy once the API runs as several replicas.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from nemo_notif_plugin.types import NotifEvent

logger = logging.getLogger(__name__)

# A listener that falls this far behind is cut off rather than letting its queue grow without bound.
DEFAULT_QUEUE_SIZE = 256


@dataclass(eq=False)
class Listener:
    """One connected SSE client."""

    workspace: str
    types: frozenset[str] | None
    queue: asyncio.Queue[NotifEvent | None] = field(default_factory=lambda: asyncio.Queue(DEFAULT_QUEUE_SIZE))

    def wants(self, event: NotifEvent) -> bool:
        return self.types is None or event.type in self.types


class NotifBroker:
    """Workspace-keyed registry of listeners. Must be used from a single event loop."""

    def __init__(self) -> None:
        self._listeners: dict[str, set[Listener]] = defaultdict(set)

    def listener_count(self, workspace: str) -> int:
        return len(self._listeners.get(workspace, ()))

    @asynccontextmanager
    async def listen(self, workspace: str, types: frozenset[str] | None = None) -> AsyncIterator[Listener]:
        listener = Listener(workspace=workspace, types=types)
        self._listeners[workspace].add(listener)
        try:
            yield listener
        finally:
            listeners = self._listeners.get(workspace)
            if listeners is not None:
                listeners.discard(listener)
                if not listeners:
                    del self._listeners[workspace]

    def publish(self, event: NotifEvent) -> int:
        """Hand ``event`` to every matching listener in its workspace; return how many got it."""
        delivered = 0
        for listener in list(self._listeners.get(event.workspace, ())):
            if not listener.wants(event):
                continue
            try:
                listener.queue.put_nowait(event)
                delivered += 1
            except asyncio.QueueFull:
                logger.warning("Dropping slow notif listener on workspace %s", event.workspace)
                self._disconnect(listener)
        return delivered

    def _disconnect(self, listener: Listener) -> None:
        self._listeners.get(listener.workspace, set()).discard(listener)
        # Make room for the sentinel so the stream ends instead of hanging.
        while not listener.queue.empty():
            listener.queue.get_nowait()
        listener.queue.put_nowait(None)


broker = NotifBroker()
