# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cap how many local Docker trials start and set up their agent at once, across every run of a user on the machine.

Each local Docker trial installs its agent from scratch, so setups that overlap slow each other down until
they pass Harbor's setup timeout. A trial takes one of :data:`MAX_CONCURRENT_SETUPS_ENV` slots (default
:data:`DEFAULT_MAX_CONCURRENT_SETUPS`) when its environment starts, and gives it back when its agent starts or
the trial ends. Waiting happens in Harbor's ``ENVIRONMENT_START`` hook, before Harbor starts the environment's
or setup's timers, so a queued trial never spends its setup budget waiting.

A slot is an exclusive ``flock`` on one of N files in :func:`default_slot_dir`, one directory per user. The kernel drops the lock when
its process exits, so a crashed run never keeps a slot. A trial takes one when it runs in Harbor's Docker
environment or an ``import_path`` environment built on it, such as the agent-layer environment; trials in any
other environment (OpenSandbox, other providers) don't.

``/tmp`` is shared, so another user may create the directory first, put a file in its place, or put a FIFO or link
at a slot's path. The directory is used only when it is this user's own and no one else can write to it;
otherwise the run sets up without the cap. A slot file that isn't a regular file of the directory is skipped, never opened in a way that
could block.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import stat
import sys
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import IO, Protocol

if sys.platform != "win32":  # Windows has no flock, so no cap there.
    import fcntl

logger = logging.getLogger(__name__)

MAX_CONCURRENT_SETUPS_ENV = "NHX_HARBOR_MAX_CONCURRENT_SETUPS"
"""Slots per user on the machine; ``0`` turns the cap off. Every run of the user should use the same value."""
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
    """Shared by every run of this user on the machine, whatever its ``TMPDIR``.

    Per user: another user couldn't open lock files this one created. In ``/tmp``, not the temp directory: runs
    with different ``TMPDIR`` values would each get their own slots.
    """
    if sys.platform == "win32":  # never used: Windows has no flock, so no cap
        return Path(tempfile.gettempdir()) / "nemo-harbor-setup-slots"
    return Path("/tmp") / f"nemo-harbor-setup-slots-{os.getuid()}"


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
        #: Whether the directory is this user's own, once checked; None before the first trial needs it.
        self._trusted: bool | None = None

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
        if self._trusted is None:
            try:
                self._directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            except OSError:  # something else is at the path, or it can't be made: as untrusted as another's
                self._trusted = False
            else:
                self._trusted = _owned_by_this_user(self._directory)
            if not self._trusted:
                logger.warning(
                    "Agent setup slot directory %s isn't this user's own, or others can write to it; trials set up "
                    "without the %s cap. Remove it to restore the cap.",
                    self._directory,
                    MAX_CONCURRENT_SETUPS_ENV,
                )
        if not self._trusted:
            return
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
            handle = _open_slot(self._directory / f"slot-{index}.lock")
            if handle is None:
                continue
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            return handle
        return None


def _owned_by_this_user(directory: Path) -> bool:
    """Whether ``directory`` is a directory, not a link, that this user owns and no one else can write to."""
    try:
        info = directory.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid() and not info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    )


def _open_slot(path: Path) -> IO[bytes] | None:
    """``path`` opened for its lock, or None when it is anything but a regular file: never follows a link, and never
    waits on a FIFO."""
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    except OSError:
        return None
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        return None
    return os.fdopen(descriptor, "ab")


def _is_local_docker(environment: _Environment) -> bool:
    kind = getattr(environment.type, "value", environment.type)
    if kind not in (None, "docker"):
        return False
    return environment.import_path is None or _is_docker_environment(environment.import_path)


def _is_docker_environment(import_path: str) -> bool:
    """Whether the environment class ``module:Class`` runs its trials in Harbor's local Docker environment."""
    from harbor.environments.docker.docker import DockerEnvironment

    module, _, name = import_path.partition(":")
    try:
        environment_class = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError, ValueError):
        return False
    return isinstance(environment_class, type) and issubclass(environment_class, DockerEnvironment)
