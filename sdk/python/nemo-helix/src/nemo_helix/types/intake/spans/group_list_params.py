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

from typing_extensions import Required, TypedDict

from ..span_filter_param import SpanFilterParam
from .span_group_sort_field import SpanGroupSortField

__all__ = ["GroupListParams"]


class GroupListParams(TypedDict, total=False):
    workspace: str

    by: Required[str]
    """Comma-separated span fields to group by, e.g. trace_id or session_id,trace_id."""

    filter: SpanFilterParam
    """
    Filter spans by the same fields as the span list endpoint, then group matching
    spans by the comma-separated fields in the by query parameter.
    """

    page: int
    """Page number."""

    page_size: int
    """Page size."""

    sort: SpanGroupSortField
    """Sort groups by size or by start time.

    Use -started_at for the traces or sessions that began most recently, which
    answers 'what ran lately' in one call instead of paging through spans. A group's
    time is its earliest matching span, so this orders by when work started and not
    by when it was last active.
    """
