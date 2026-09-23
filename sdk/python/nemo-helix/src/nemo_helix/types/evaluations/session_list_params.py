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

from .evaluation_session_filter_param import EvaluationSessionFilterParam

__all__ = ["SessionListParams"]


class SessionListParams(TypedDict, total=False):
    workspace: str

    filter: EvaluationSessionFilterParam
    """Filter sessions by test_case_name and status."""

    mode: Literal["summary", "preview", "detailed"]
    """Response mode.

    summary omits root-span input and output; preview includes both truncated to 300
    characters; detailed returns full root-span payloads.
    """

    page: int
    """Page number."""

    page_size: int
    """Page size."""

    sort: str
    """
    Comma-separated list of fields to sort by, applied in order (the first field
    dominates); prefix a field with '-' for descending — e.g.
    '-cost_total_usd,latency_ms'. Fields: test_case_name, started_at, ended_at,
    latency_ms, status, cost_total_usd, tokens. When omitted, sessions are ordered
    by started_at ascending.
    """
