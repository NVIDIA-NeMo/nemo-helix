# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The deployment's backend, from its configuration.

The builder's routes and the credential broker both load it here, from the same ``builder:``
section, so they act on one backend.
"""

from __future__ import annotations

from nemo_builder_plugin.backend import Backend
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.execution import ExecutionBackend


def load_backend(config: BuilderConfig) -> Backend:
    return ExecutionBackend(config)
