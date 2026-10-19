# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executor selection for deployments-plugin model entities."""

from nhx.common.config import Runtime
from nhx.core.models.controllers.backends.deployments_plugin.config import DeploymentsPluginConfig


def executor_for_runtime(config: DeploymentsPluginConfig, runtime: Runtime) -> str | None:
    """Resolve a configured executor name for the platform runtime.

    ``platform.runtime`` (docker/kubernetes) is global deployment topology;
    executor names are per-backend config that map to entries in the
    deployments-plugin executor registry. This returns ``None`` when the
    operator enabled ``deployments_plugin`` but omitted ``docker_executor``,
    ``k8s_executor``, and ``default_executor`` for the active runtime —
    ``create_model_deployment`` surfaces that misconfiguration as ERROR.
    """
    if runtime == Runtime.DOCKER:
        return config.docker_executor or config.default_executor
    if runtime == Runtime.KUBERNETES:
        return config.k8s_executor or config.default_executor
    return config.default_executor
