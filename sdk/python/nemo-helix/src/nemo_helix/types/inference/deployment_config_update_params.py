# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Required, TypedDict

from .engine import Engine
from .container_executor_config_param import ContainerExecutorConfigParam
from .model_deployment_config_model_spec_param import ModelDeploymentConfigModelSpecParam

__all__ = ["DeploymentConfigUpdateParams"]


class DeploymentConfigUpdateParams(TypedDict, total=False):
    workspace: str

    engine: Required[Engine]
    """Inference engine selecting the compiler path for a deployment.

    The engine determines what command, image, and env a deployment compiles to. The
    fields a compiler consumes are not engine-specific; engines take the same inputs
    (model_spec + executor_config) and differ in what they do with them.
    """

    executor_config: Required[ContainerExecutorConfigParam]
    """Compute + container settings shared by the docker and k8s executors.

    Both the docker and k8s executors run containers and share this shape. A future
    non-container executor (e.g. subprocess) would warrant turning `executor_config`
    into a discriminated union.
    """

    model_spec: Required[ModelDeploymentConfigModelSpecParam]
    """What model to serve and how -- independent of the executor it runs on.

    Executor-invariant facts about the model. The compiler resolves the weight
    source per engine; serving fields override the model entity spec when set.
    """

    description: str
    """Optional description of the deployment configuration"""

    model_entity_id: str
    """Optional reference to the base model entity ID for this deployment"""
