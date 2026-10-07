# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Docker Compose sandbox provider façade."""

from __future__ import annotations

from ._compose_contracts import (
    ComposeCleanupError,
    ComposeCommandResult,
    ComposeServiceTopology,
    ProgressCallback,
    PullPolicy,
)
from ._compose_provider import (
    ComposeTeardownContext,
    DockerComposeSandboxProvider,
    TeardownHook,
)

for _public_class in (
    ComposeCleanupError,
    ComposeCommandResult,
    ComposeServiceTopology,
    ComposeTeardownContext,
    DockerComposeSandboxProvider,
):
    _public_class.__module__ = __name__

__all__ = [
    "ComposeCleanupError",
    "ComposeCommandResult",
    "ComposeServiceTopology",
    "ComposeTeardownContext",
    "DockerComposeSandboxProvider",
    "ProgressCallback",
    "PullPolicy",
    "TeardownHook",
]
