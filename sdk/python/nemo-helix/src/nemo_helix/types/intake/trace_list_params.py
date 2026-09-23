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

from typing_extensions import Literal, TypedDict

from .trace_sort_field import TraceSortField
from .trace_filter_param import TraceFilterParam

__all__ = ["TraceListParams"]


class TraceListParams(TypedDict, total=False):
    workspace: str

    filter: TraceFilterParam
    """
    Filter root-span-backed traces by id, session_id, root status, root span
    started_at, evaluation_name, test_case_name, and agent_name.
    """

    mode: Literal["summary", "preview", "detailed"]
    """Response mode.

    summary returns root-span fields without payloads or rollups; preview adds
    token, cost, and span-count rollups plus 300-character input/output previews;
    detailed returns rollups and full payloads.
    """

    page: int
    """Page number."""

    page_size: int
    """Page size."""

    sort: TraceSortField
