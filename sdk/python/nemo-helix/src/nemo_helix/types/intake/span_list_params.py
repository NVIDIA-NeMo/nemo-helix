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

from .span_sort_field import SpanSortField
from .span_filter_param import SpanFilterParam

__all__ = ["SpanListParams"]


class SpanListParams(TypedDict, total=False):
    workspace: str

    filter: SpanFilterParam
    """
    Filter spans by session_id, trace_id, parent_span_id, project, evaluation_name,
    test_case_name, source, kind, status, model, tool_name, provider, agent_id,
    agent_name, and started_at. Every field takes one exact value, except
    started_at, which takes gte and lte.
    """

    mode: Literal["summary", "preview", "detailed"]
    """Response mode.

    summary omits payloads and raw attributes; preview includes input and output
    truncated to 300 characters; detailed returns full payloads and raw attributes.
    """

    page: int
    """Page number."""

    page_size: int
    """Page size."""

    sort: SpanSortField
