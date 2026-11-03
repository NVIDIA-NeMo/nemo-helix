# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Errors a sandbox backend raises in place of its SDK's own, so shared code can act on them whatever the backend."""

from __future__ import annotations


class SandboxBackendError(RuntimeError):
    """A sandbox backend failed an operation."""


class SandboxNotFound(SandboxBackendError):
    """The backend no longer knows the sandbox: something deleted it, or its lifetime ended."""


class ExecInterrupted(SandboxBackendError):
    """A command's output ended before it reported an exit code, so whether it succeeded is unknown.

    Backends stream a command's output and send its exit code last. When the stream ends without
    one, as it does when the sandbox dies under the command, the command never confirmed success,
    so reporting any exit code would invent an outcome.
    """
