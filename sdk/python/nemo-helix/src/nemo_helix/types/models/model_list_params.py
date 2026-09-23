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

from .model_entity_sort_field import ModelEntitySortField
from .model_entity_filter_param import ModelEntityFilterParam

__all__ = ["ModelListParams"]


class ModelListParams(TypedDict, total=False):
    workspace: str

    filter: ModelEntityFilterParam
    """
    Filter models by name, project, workspace, base_model, adapters,
    finetuning_type, prompt, lora_enabled, description, created_at, and updated_at.
    """

    page: int
    """Page number."""

    page_size: int
    """Page size."""

    sort: ModelEntitySortField
    """The field to sort by.

    To sort in decreasing order, use `-` in front of the field name.
    """

    verbose: bool
    """Whether to include full spec details"""
