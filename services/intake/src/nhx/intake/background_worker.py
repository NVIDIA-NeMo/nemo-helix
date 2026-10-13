# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A background task with a managed start/stop lifecycle.

Runs a subclass's ``_run()`` coroutine as a single asyncio ``Task``: ``start()`` launches it and
``stop()`` signals it (via ``self._stopping``) and awaits its exit. It never cancels mid-run, so an
in-flight iteration finishes before ``stop()`` returns. Subclasses own what the loop actually does.

Intake-agnostic; a candidate to move to ``nhx.common`` if another service wants the same lifecycle.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod


class BackgroundWorker(ABC):
    """Runs ``_run()`` as a single asyncio task with a start/stop lifecycle."""

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    def start(self) -> None:
        if self._task is None:
            self._stopping.clear()
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        # Signal the loop to exit and await it — we never cancel mid-run, so an in-flight iteration
        # finishes before stop() returns.
        self._stopping.set()
        if self._task is not None:
            await self._task
            self._task = None

    @abstractmethod
    async def _run(self) -> None:
        """The worker loop. Must return promptly once ``self._stopping`` is set."""
