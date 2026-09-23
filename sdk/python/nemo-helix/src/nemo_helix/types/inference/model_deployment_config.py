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

from typing import Optional
from datetime import datetime

from .engine import Engine
from ..._compat import PYDANTIC_V1, ConfigDict
from ..._models import BaseModel
from .container_executor_config import ContainerExecutorConfig
from .model_deployment_config_model_spec import ModelDeploymentConfigModelSpec

__all__ = ["ModelDeploymentConfig"]


class ModelDeploymentConfig(BaseModel):
    """
    ModelDeploymentConfig stores the configuration details for deploying a model.
    These objects are immutable with automatic versioning.

    The unique identifier is the combination of workspace/name/entity_version.
    """

    created_at: datetime
    """The timestamp of model entity creation"""

    engine: Engine
    """Inference engine selecting the compiler path for a deployment.

    The engine determines what command, image, and env a deployment compiles to. The
    fields a compiler consumes are not engine-specific; engines take the same inputs
    (model_spec + executor_config) and differ in what they do with them.
    """

    entity_version: int
    """Version of this deployment config. Automatically managed."""

    executor_config: ContainerExecutorConfig
    """Compute + container settings shared by the docker and k8s executors.

    Both the docker and k8s executors run containers and share this shape. A future
    non-container executor (e.g. subprocess) would warrant turning `executor_config`
    into a discriminated union.
    """

    model_spec: ModelDeploymentConfigModelSpec
    """What model to serve and how -- independent of the executor it runs on.

    Executor-invariant facts about the model. The compiler resolves the weight
    source per engine; serving fields override the model entity spec when set.
    """

    name: str
    """Name of the entity.

    Name/workspace combo must be unique across all entities. Name must start with a
    lowercase letter, be 2-63 characters, and use lowercase letters, digits,
    hyphens, and dots (no consecutive hyphens, cannot end with a hyphen).
    """

    updated_at: datetime
    """The timestamp of the last model entity update"""

    workspace: str
    """The workspace of the entity.

    Allowed characters: letters (a-z, A-Z), digits (0-9), underscores, hyphens, and
    dots.
    """

    id: Optional[str] = None
    """Unique identifier for the deployment config"""

    description: Optional[str] = None
    """Optional description of the deployment configuration"""

    model_entity_id: Optional[str] = None
    """Optional reference to the base model entity ID for this deployment"""

    project: Optional[str] = None
    """The URN of the project associated with this entity."""

    if not PYDANTIC_V1:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
