# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-exit hooks for a process that owns sandboxes.

Kept out of the orchestrator so a hook that destroys sandboxes can register
itself without the orchestrator and that hook importing each other.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
from collections.abc import Callable
from types import FrameType

LOGGER = logging.getLogger(__name__)

#: Signals a container runtime sends before SIGKILL. SIGKILL and node loss cannot be caught and
#: stay the sandbox ttl_s's problem.
TERMINATION_SIGNALS = (signal.SIGTERM, signal.SIGINT)

# Registered shutdown hooks for this process.
_TERMINATION_SHUTDOWNS: list[Callable[[], None]] = []


def install_termination_cleanup(shutdown: Callable[[], None]) -> None:
    """Run ``shutdown`` when this process exits without it having been called.

    A process that owns a sandbox is the only thing that can name it. Ray tears an actor's worker
    down without running user teardown, so a job that is cancelled, evicted or preempted otherwise
    leaves its sandbox running until ttl_s. ``shutdown`` must tolerate being called twice: an
    ordinary exit runs it directly and then again from ``atexit``.

    For a process the caller owns -- an actor, a CLI. Not for a library embedded in someone else's
    host, whose signal handling is not ours to replace.

    Each call adds a hook. The signal handler runs every hook registered so far, so arming
    the job reaper and then the session teardown does not leave the reaper installed only
    on ``atexit``.
    """
    _TERMINATION_SHUTDOWNS.append(shutdown)
    atexit.register(shutdown)

    def _terminate(signum: int, _frame: FrameType | None) -> None:
        # Restored before the cleanup runs, not after: a second signal arriving mid-shutdown then
        # takes the default action and terminates, rather than re-entering this handler on top of
        # an in-flight destroy. Two SIGTERMs mean the sender wants the process gone.
        signal.signal(signum, signal.SIG_DFL)
        LOGGER.warning("received signal %s; destroying sandboxed Gym host before exit", signum)
        error: BaseException | None = None
        for hook in list(_TERMINATION_SHUTDOWNS):
            try:
                hook()
            except BaseException as exc:
                # Keep going so one failed teardown cannot spare a sandbox another hook owns.
                # The first error is re-raised after the rest have run.
                if error is None:
                    error = exc
        try:
            if error is not None:
                raise error
        finally:
            # Re-raised even if cleanup failed, so the exit status still reports the signal --
            # swallowing it would make a cancelled job look like a clean stop.
            os.kill(os.getpid(), signum)

    for signum in TERMINATION_SIGNALS:
        try:
            signal.signal(signum, _terminate)
        except ValueError:
            # Only the main thread may install handlers, and Ray does not promise to call an
            # actor method there. atexit still covers the ordinary exit.
            LOGGER.debug("cannot install a %s handler off the main thread", signum)
