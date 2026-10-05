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

from typing import List, Optional
from datetime import datetime

from ..._compat import PYDANTIC_V1, ConfigDict
from ..._models import BaseModel
from ..shared.auth_context import AuthContext
from .model_deployment_status import ModelDeploymentStatus
from .model_deployment_status_history_item import ModelDeploymentStatusHistoryItem

__all__ = ["ModelDeployment"]


class ModelDeployment(BaseModel):
    """
    ModelDeployment represents a deployed instance of a model with a specific configuration.
    These objects are immutable with automatic versioning, except for status updates.

    The unique identifier is the combination of workspace/name/entity_version.
    """

    config: str
    """Reference to the ModelDeploymentConfig name"""

    config_version: int
    """Reference to the specific ModelDeploymentConfig version"""

    created_at: datetime
    """The timestamp of model entity creation"""

    entity_version: int
    """Version of this deployment. Automatically managed."""

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
    """Unique identifier for the deployment"""

    auth_context: Optional[AuthContext] = None
    """Auth context captured at resource creation for delegated access.

    Stores a snapshot of the creating principal's identity so that controllers can
    later act on their behalf (e.g., accessing secrets).
    """

    model_provider_id: Optional[str] = None
    """
    Optional reference to the auto-created ModelProvider workspace/name (format:
    workspace/name)
    """

    project: Optional[str] = None
    """The URN of the project associated with this entity."""

    status: Optional[ModelDeploymentStatus] = None
    """Status enum for ModelDeployment objects."""

    status_history: Optional[List[ModelDeploymentStatusHistoryItem]] = None
    """History of status changes, ordered chronologically (oldest first)"""

    status_message: Optional[str] = None
    """Detailed status message, populated by models controller"""

    if not PYDANTIC_V1:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
