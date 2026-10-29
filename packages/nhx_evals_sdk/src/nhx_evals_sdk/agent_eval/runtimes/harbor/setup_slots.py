# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cap how many local Docker trials start and set up their agent at once, across every run on the machine.

Each local Docker trial installs its agent from scratch, so setups that overlap slow each other down until
they pass Harbor's setup timeout. A trial takes one of :data:`MAX_CONCURRENT_SETUPS_ENV` slots (default
:data:`DEFAULT_MAX_CONCURRENT_SETUPS`) when its environment starts, and gives it back when its agent starts or
the trial ends. Waiting happens in Harbor's ``ENVIRONMENT_START`` hook, before Harbor starts the environment's
or setup's timers, so a queued trial never spends its setup budget waiting.

A slot is an exclusive ``flock`` on one of N files in :func:`default_slot_dir`. The kernel drops the lock when
its process exits, so a crashed run never keeps a slot. Trials in other environments (OpenSandbox and any
other ``import_path`` environment) don't take a slot.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import IO, Protocol

if sys.platform != "win32":  # Windows has no flock, so no cap there.
    import fcntl

logger = logging.getLogger(__name__)

MAX_CONCURRENT_SETUPS_ENV = "NHX_HARBOR_MAX_CONCURRENT_SETUPS"
"""Slots per machine; ``0`` turns the cap off. Every run on the machine should use the same value."""
DEFAULT_MAX_CONCURRENT_SETUPS = 4
POLL_INTERVAL_SEC = 0.5


class _Environment(Protocol):
    @property
    def type(self) -> object: ...

    @property
    def import_path(self) -> str | None: ...


class _TrialConfig(Protocol):
    @property
    def environment(self) -> _Environment: ...


class _TrialHookEvent(Protocol):
    @property
    def trial_name(self) -> str: ...

    @property
    def config(self) -> _TrialConfig: ...


class _HookedJob(Protocol):
    def on_environment_started(self, callback: Callable[[_TrialHookEvent], Awaitable[None]]) -> object: ...
    def on_agent_started(self, callback: Callable[[_TrialHookEvent], Awaitable[None]]) -> object: ...
    def on_trial_ended(self, callback: Callable[[_TrialHookEvent], Awaitable[None]]) -> object: ...
    def on_trial_cancelled(self, callback: Callable[[_TrialHookEvent], Awaitable[None]]) -> object: ...


def default_slot_dir() -> Path:
    """Shared by every run of this user on the machine."""
    return Path(tempfile.gettempdir()) / "nemo-harbor-setup-slots"


def max_concurrent_setups() -> int:
    """:data:`MAX_CONCURRENT_SETUPS_ENV`, or the default; raises on a value that isn't a whole number >= 0."""
    raw = os.environ.get(MAX_CONCURRENT_SETUPS_ENV, "").strip()
    if not raw:
        return DEFAULT_MAX_CONCURRENT_SETUPS
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < 0:
        raise ValueError(f"{MAX_CONCURRENT_SETUPS_ENV}={raw!r}: expected a whole number >= 0 (0 turns the cap off)")
    return value


class SetupSlots:
    """N machine-wide slots; one held per local Docker trial from environment start until its agent starts."""

    def __init__(self, slots: int, directory: Path) -> None:
        self._slots = slots
        self._directory = directory
        self._held: dict[str, IO[bytes]] = {}

    def attach(self, job: _HookedJob) -> None:
        """Take a slot when each local Docker trial's environment starts; give it back when it no longer needs it."""
        if self._slots == 0 or sys.platform == "win32":
            return
        job.on_environment_started(self._acquire)
        job.on_agent_started(self._release)
        job.on_trial_ended(self._release)
        job.on_trial_cancelled(self._release)

    async def _acquire(self, event: _TrialHookEvent) -> None:
        if not _is_local_docker(event.config.environment) or event.trial_name in self._held:
            return
        self._directory.mkdir(parents=True, exist_ok=True)
        waiting = False
        while (handle := self._try_lock()) is None:
            if not waiting:
                logger.info(
                    "Trial %s is waiting for an agent setup slot: all %d are in use (set %s to change)",
                    event.trial_name,
                    self._slots,
                    MAX_CONCURRENT_SETUPS_ENV,
                )
                waiting = True
            await asyncio.sleep(POLL_INTERVAL_SEC)
        self._held[event.trial_name] = handle

    async def _release(self, event: _TrialHookEvent) -> None:
        handle = self._held.pop(event.trial_name, None)
        if handle is not None:
            handle.close()  # closing the file drops its flock

    def _try_lock(self) -> IO[bytes] | None:
        if sys.platform == "win32":
            return None
        for index in range(self._slots):
            handle = (self._directory / f"slot-{index}.lock").open("ab")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            return handle
        return None


def _is_local_docker(environment: _Environment) -> bool:
    kind = getattr(environment.type, "value", environment.type)
    return environment.import_path is None and kind in (None, "docker")
