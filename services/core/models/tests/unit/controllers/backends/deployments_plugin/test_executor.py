# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nhx.common.config import Runtime
from nhx.core.models.controllers.backends.deployments_plugin.config import DeploymentsPluginConfig
from nhx.core.models.controllers.backends.deployments_plugin.executor import executor_for_runtime


def test_executor_prefers_runtime_specific_value_then_default() -> None:
    config = DeploymentsPluginConfig(default_executor="default", docker_executor="docker", k8s_executor="k8s")
    assert executor_for_runtime(config, Runtime.DOCKER) == "docker"
    assert executor_for_runtime(config, Runtime.KUBERNETES) == "k8s"
    assert executor_for_runtime(DeploymentsPluginConfig(default_executor="default"), Runtime.DOCKER) == "default"
