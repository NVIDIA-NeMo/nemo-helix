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

from ..shared.generic_sort_field import GenericSortField

__all__ = ["WorkspaceListParams"]


class WorkspaceListParams(TypedDict, total=False):
    filter: str
    """Query filter expression. Supports text and JSON syntaxes:

    - Text: name:"value" AND status>500 with operators : ~ > >= < <= IN NOT IN AND
      OR and negation prefix -
    - Object (JSON): {"name":{"$like":"value"}} with operators $eq, $like, $lt,
      $lte, $gt, $gte, $in, $nin, $and, $or, $not
    - Bracket notation: ?filter[name][$like]=value
    - Relationship traversal: ?filter[relationship][$exists]=true or
      ?filter[relationship][field]=value
    """

    page: int
    """Page number"""

    page_size: int
    """Items per page"""

    sort: GenericSortField
    """Sort field"""
