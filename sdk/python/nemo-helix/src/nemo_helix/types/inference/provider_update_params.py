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

from typing import Dict
from typing_extensions import Required, TypedDict

from ..._types import SequenceNotStr
from .model_provider_status import ModelProviderStatus

__all__ = ["ProviderUpdateParams"]


class ProviderUpdateParams(TypedDict, total=False):
    workspace: str

    host_url: Required[str]
    """The network endpoint URL for the model provider"""

    api_key_secret_name: str
    """Reference to an API key secret stored in the Secrets service.

    Create the secret first via secrets API, then pass the secret name here.
    """

    auth_header_format: str
    """
    Jinja2 template string controlling how the API key secret is sent to the
    upstream. Must contain exactly one variable named `auth_secret`, which is
    substituted with the resolved secret value at request time. Example:
    `'X-Api-Key: {{ auth_secret }}'`. If not set, defaults to
    `'Authorization: Bearer {{ auth_secret }}'`.
    """

    default_extra_body: Dict[str, object]
    """Default body parameters for inference requests.

    Can be overridden by user requests.
    """

    default_extra_headers: Dict[str, str]
    """Default headers for inference requests. Can be overridden by user requests."""

    description: str
    """Optional description of the model provider"""

    enabled_models: SequenceNotStr[str]
    """Optional list of specific models to enable from this provider"""

    model_deployment_id: str
    """
    Optional reference to the ModelDeployment ID if this provider is associated with
    a deployment
    """

    project: str
    """The URN of the project associated with this model provider"""

    required_extra_body: Dict[str, object]
    """Required body parameters for inference requests.

    Cannot be overridden by user requests.
    """

    required_extra_headers: Dict[str, str]
    """Required headers for inference requests. Cannot be overridden by user requests."""

    status: ModelProviderStatus
    """Status enum for ModelProvider objects."""

    status_message: str
    """Status message"""
