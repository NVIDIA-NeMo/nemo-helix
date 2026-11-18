# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Errors sandbox backends raise in place of their SDKs' own, so shared code can handle them the same way on any backend.

How a runner should treat them:

* ``TransientBackendError``: retry the same call, with backoff. Backends raise it only when the call
  definitely didn't take effect, so a retry never runs a command twice.
* ``SandboxLost`` and ``ExecInterrupted``: an infrastructure failure. Fail the trial as one and retry
  it in a new sandbox. Don't grade it, because the task never got to finish.
* ``UnsupportedOperation``: the task doesn't fit this backend. Refuse it and name the reason rather
  than retry. ``SandboxBackend.check`` catches most of these before a sandbox exists.
* Any other ``SandboxBackendError``: the operation failed, and a retry may or may not help.
"""

from __future__ import annotations


class SandboxBackendError(RuntimeError):
    """A sandbox backend failed an operation."""


class SandboxNotFound(SandboxBackendError):
    """The backend no longer knows the sandbox, because something deleted it or its lifetime ended."""


class SandboxLost(SandboxNotFound):
    """The sandbox went away while it was in use, without anyone deleting it.

    Some backends delete a sandbox themselves when a command overruns, memory runs out or its lease
    ends. Others lose one when its node or pod dies. Either way the task didn't fail on its own
    merits, so a runner retries the trial instead of grading it. A sandbox that's gone because of a
    ``ttl_s`` the caller set, or a ``delete`` it made, is a plain ``SandboxNotFound``.
    """


class TransientBackendError(SandboxBackendError):
    """The backend failed in a way that may clear up, such as a busy, restarting or unreachable server.

    Callers decide whether to retry by this type rather than by matching SDK exception names, which
    differ between backends and SDK releases.
    """


class RevisionConflict(SandboxBackendError):
    """A change named the revision it expected, and the sandbox has moved past it.

    ``EgressControl.set_egress`` raises it when ``expected_revision`` is stale. Read the current state
    and decide again; retrying the same change blindly would overwrite whatever changed it.
    """


class UnsupportedOperation(SandboxBackendError):
    """The backend can't do what was asked, and won't quietly do something weaker instead.

    For example, running a command as root on a backend that never runs anything as root. Failing
    keeps a task from being graded under conditions it didn't ask for.
    """


class ExecInterrupted(SandboxBackendError):
    """A command's output ended before it reported an exit code, so whether it succeeded is unknown.

    Backends stream a command's output and send its exit code last. If the stream ends without one,
    as it does when the sandbox dies mid-command, any exit code we reported would be made up. When
    the backend knows the sandbox is gone, it raises ``SandboxLost`` instead.
    """
