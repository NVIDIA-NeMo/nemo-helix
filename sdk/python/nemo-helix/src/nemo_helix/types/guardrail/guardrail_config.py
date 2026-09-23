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

from ..._models import BaseModel
from .rails_config import RailsConfig

__all__ = ["GuardrailConfig"]


class GuardrailConfig(BaseModel):
    """A guardrail configuration entity."""

    id: str

    created_at: datetime

    created_by: Optional[str] = None

    db_version: int
    """Database version of the entity for optimistic locking."""

    entity_id: str
    """Alias for id for backwards compatibility."""

    parent: str
    """Parent entity ID for nested entities."""

    updated_at: datetime

    updated_by: Optional[str] = None

    workspace: str
    """Workspace identifier"""

    data: Optional[RailsConfig] = None
    """Configuration object for the models and the rails."""

    description: Optional[str] = None
    """Description of the guardrail config"""

    name: Optional[str] = None
    """Entity name within the workspace"""

    project: Optional[str] = None
    """The name of the project associated with this entity."""
