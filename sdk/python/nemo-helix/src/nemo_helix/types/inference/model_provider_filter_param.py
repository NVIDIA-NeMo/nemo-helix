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

from typing import Union
from typing_extensions import TypeAlias, TypedDict

from .model_provider_status import ModelProviderStatus
from ..shared_params.string_filter import StringFilter
from ..shared_params.datetime_filter import DatetimeFilter

__all__ = ["ModelProviderFilterParam", "Description", "HostURL", "Name"]

Description: TypeAlias = Union[StringFilter, str]

HostURL: TypeAlias = Union[StringFilter, str]

Name: TypeAlias = Union[StringFilter, str]


class ModelProviderFilterParam(TypedDict, total=False):
    """Filter for ModelProvider queries."""

    created_at: DatetimeFilter
    """Filter by creation date."""

    description: Description
    """Filter by description."""

    host_url: HostURL
    """Filter by host URL."""

    model_deployment_id: str
    """Filter by associated deployment ID."""

    name: Name
    """Filter by name."""

    project: str
    """Filter by project URN."""

    status: ModelProviderStatus
    """Status enum for ModelProvider objects."""

    updated_at: DatetimeFilter
    """Filter by update date."""

    workspace: str
    """Filter by workspace."""
