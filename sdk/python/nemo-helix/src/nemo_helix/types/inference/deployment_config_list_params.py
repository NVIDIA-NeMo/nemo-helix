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

from typing_extensions import TypedDict

from .model_deployment_config_filter_param import ModelDeploymentConfigFilterParam

__all__ = ["DeploymentConfigListParams"]


class DeploymentConfigListParams(TypedDict, total=False):
    workspace: str

    filter: ModelDeploymentConfigFilterParam
    """
    Filter deployment configs by workspace, project, model_entity_id, name,
    description, created_at, and updated_at.
    """

    page: int
    """Page number."""

    page_size: int
    """Page size."""

    sort: str
    """The field to sort by.

    To sort in decreasing order, use `-` in front of the field name.
    """
