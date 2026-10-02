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

from typing import Dict, List, Optional
from datetime import datetime

from ..._compat import PYDANTIC_V1, ConfigDict
from ..._models import BaseModel
from ..shared.auth_context import AuthContext
from .served_model_mapping import ServedModelMapping
from .model_provider_status import ModelProviderStatus

__all__ = ["ModelProvider"]


class ModelProvider(BaseModel):
    """
    A ModelProvider defines a reachable network endpoint that provides an inference
    service for one or more Model Entities. Examples of Model Providers include
    OpenAI, NIMs, Bedrock, NVIDIA Build, etc. A ModelProvider may be provisioned
    automatically by Models Controller for ModelDeployments, or it may be provisioned
    manually by an end user for an endpoint that does not have its lifecycle managed
    by models service (like an external provider.)

    The unique identifier for a ModelProvider is the combination of workspace/name.
    """

    created_at: datetime
    """The timestamp of model entity creation"""

    host_url: str
    """The network endpoint URL for the model provider"""

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
    """Unique identifier for the model provider"""

    api_key_secret_name: Optional[str] = None
    """Reference to the API key stored in Secrets service"""

    auth_context: Optional[AuthContext] = None
    """Auth context captured at resource creation for delegated access.

    Stores a snapshot of the creating principal's identity so that controllers can
    later act on their behalf (e.g., accessing secrets).
    """

    auth_header_format: Optional[str] = None
    """
    Jinja2 template string controlling how the API key secret is sent to the
    upstream. Must contain exactly one variable named `auth_secret`, which is
    substituted with the resolved secret value at request time. Example:
    `'X-Api-Key: {{ auth_secret }}'`. If not set, defaults to
    `'Authorization: Bearer {{ auth_secret }}'`.
    """

    default_extra_body: Optional[Dict[str, object]] = None
    """Default body parameters for inference requests.

    Can be overridden by user requests.
    """

    default_extra_headers: Optional[Dict[str, str]] = None
    """Default headers for inference requests. Can be overridden by user requests."""

    description: Optional[str] = None
    """Optional description of the model provider"""

    enabled_models: Optional[List[str]] = None
    """Optional list of specific models to enable from this provider.

    If not set, all discovered models are enabled.
    """

    model_deployment_id: Optional[str] = None
    """
    Optional reference to the ModelDeployment ID if this provider was auto-created
    for a deployment
    """

    project: Optional[str] = None
    """The URN of the project associated with this entity."""

    required_extra_body: Optional[Dict[str, object]] = None
    """Required body parameters for inference requests.

    Cannot be overridden by user requests.
    """

    required_extra_headers: Optional[Dict[str, str]] = None
    """Required headers for inference requests. Cannot be overridden by user requests."""

    served_models: Optional[List[ServedModelMapping]] = None
    """List of models served by this provider with routing information for IGW"""

    status: Optional[ModelProviderStatus] = None
    """Status enum for ModelProvider objects."""

    status_message: Optional[str] = None
    """Detailed status message, populated by models service"""

    if not PYDANTIC_V1:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
