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

from .trace_filter_param import TraceFilterParam
from .trace_metric_bucket_param import TraceMetricBucketParam

__all__ = ["TraceGetMetricsParams"]


class TraceGetMetricsParams(TypedDict, total=False):
    workspace: str

    bucket: TraceMetricBucketParam
    """Time bucket granularity. total collapses the filtered range into a single row."""

    filter: TraceFilterParam
    """Filter the traces the metrics are computed over.

    Accepts the same fields as the traces list, so agent_name scopes the rollup to
    one agent. Without a started_at lower bound the rollup covers the last 7 days.
    """

    timezone: str
    """IANA timezone the buckets are aligned to, e.g. America/Los_Angeles."""
