# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The deployment's backend, from its configuration."""

from __future__ import annotations

from collections.abc import Sequence

from nemo_builder_plugin.backend import Backend, BackendRejectedError
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.execution import ExecutionBackend
from nemo_builder_plugin.remote import RemoteBackend
from nemo_helix_plugin.jobs.endpoints import ExecutionProfile

#: Every value `builder.backend` takes.
BACKENDS = (ExecutionBackend.name, RemoteBackend.name)


def load_backend(config: BuilderConfig, profiles: Sequence[ExecutionProfile]) -> Backend:
    """The backend `builder.backend` names. An unknown name refuses every submit, rather than build elsewhere."""
    if config.backend == ExecutionBackend.name:
        return ExecutionBackend(config, profiles)
    if config.backend == RemoteBackend.name:
        return RemoteBackend(config, profiles)
    raise BackendRejectedError(
        f"builder.backend is {config.backend!r}, which is no backend; expected one of: {', '.join(BACKENDS)}"
    )
