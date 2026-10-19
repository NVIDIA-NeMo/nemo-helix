# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The deployment's backend, from its configuration."""

from __future__ import annotations

from collections.abc import Sequence

from nemo_builder_plugin.backend import Backend
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.execution import ExecutionBackend
from nemo_helix_plugin.jobs.endpoints import ExecutionProfile


def load_backend(config: BuilderConfig, profiles: Sequence[ExecutionProfile]) -> Backend:
    return ExecutionBackend(config, profiles)
