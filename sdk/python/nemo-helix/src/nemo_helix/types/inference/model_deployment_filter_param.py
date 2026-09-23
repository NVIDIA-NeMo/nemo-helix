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

from .model_deployment_status import ModelDeploymentStatus
from ..shared_params.string_filter import StringFilter
from ..shared_params.datetime_filter import DatetimeFilter

__all__ = ["ModelDeploymentFilterParam", "Config", "Name", "StatusMessage"]

Config: TypeAlias = Union[StringFilter, str]

Name: TypeAlias = Union[StringFilter, str]

StatusMessage: TypeAlias = Union[StringFilter, str]


class ModelDeploymentFilterParam(TypedDict, total=False):
    """Filter for ModelDeployment queries."""

    config: Config
    """Filter by config name."""

    created_at: DatetimeFilter
    """Filter by creation date."""

    model_provider_id: str
    """Filter by model provider ID."""

    name: Name
    """Filter by deployment name."""

    project: str
    """Filter by project URN."""

    status: ModelDeploymentStatus
    """Status enum for ModelDeployment objects."""

    status_message: StatusMessage
    """Filter by status message."""

    updated_at: DatetimeFilter
    """Filter by update date."""

    workspace: str
    """Filter by workspace."""
