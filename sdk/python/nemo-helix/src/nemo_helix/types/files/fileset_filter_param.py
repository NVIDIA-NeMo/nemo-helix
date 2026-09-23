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

from .fileset_purpose import FilesetPurpose
from .storage_config_type import StorageConfigType
from ..shared_params.string_filter import StringFilter
from ..shared_params.datetime_filter import DatetimeFilter

__all__ = ["FilesetFilterParam", "Description", "Name"]

Description: TypeAlias = Union[StringFilter, str]

Name: TypeAlias = Union[StringFilter, str]


class FilesetFilterParam(TypedDict, total=False):
    """Filter schema for listing filesets."""

    created_at: DatetimeFilter
    """Filter by creation date.

    Supports '$gte' (on or after) and '$lte' (on or before) datetime filters.
    """

    description: Description
    """Filter by fileset description."""

    name: Name
    """Filter by fileset name."""

    purpose: FilesetPurpose
    """Filter by the purpose of the fileset (e.g., 'dataset', 'generic')."""

    storage_type: StorageConfigType
    """Filter by the storage backend type (e.g., 'local', 'ngc')."""

    updated_at: DatetimeFilter
    """Filter by update date.

    Supports '$gte' (on or after) and '$lte' (on or before) datetime filters.
    """
