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

from ..shared.finetuning_type import FinetuningType
from ..shared_params.string_filter import StringFilter
from ..shared_params.datetime_filter import DatetimeFilter

__all__ = ["AdapterEntityFilterParam", "Description", "Model", "Name"]

Description: TypeAlias = Union[StringFilter, str]

Model: TypeAlias = Union[StringFilter, str]

Name: TypeAlias = Union[StringFilter, str]


class AdapterEntityFilterParam(TypedDict, total=False):
    """Filter for Adapter list queries."""

    created_at: DatetimeFilter
    """Filter entities based on creation date."""

    description: Description
    """Filter by description."""

    enabled: bool
    """Filter by whether the adapter is enabled for inference after training."""

    fileset: str
    """Filter by fileset reference in the form {workspace}/{fileset_name}."""

    finetuning_type: FinetuningType
    """Finetuning types."""

    model: Model
    """
    Filter by parent (base) model entity reference in the form
    {workspace}/{model_name}.
    """

    name: Name
    """Filter by adapter name."""

    updated_at: DatetimeFilter
    """Filter entities based on update date."""
