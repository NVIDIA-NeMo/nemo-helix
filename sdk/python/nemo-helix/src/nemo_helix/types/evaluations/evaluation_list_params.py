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

from .evaluation_filter_param import EvaluationFilterParam

__all__ = ["EvaluationListParams"]


class EvaluationListParams(TypedDict, total=False):
    workspace: str

    filter: EvaluationFilterParam
    """
    Filter evaluations by name, experiment_id (experiment group membership;
    experiment_group_id is a deprecated alias), dataset_name, dataset_version,
    created_by, created_at, or updated_at. Pass is_deleted=true to return only
    soft-deleted evaluations; omit to see only live ones. Pass is_pinned=true (or
    false) to filter by pinned state; omit to return both. Filter by a metadata
    key/value: filter[metadata.<key>]=<value>. Filter by a rollup metric with
    numeric range operators ($gte/$lte/$gt/$lt/$eq): filter[run_count][$gte]=5,
    filter[cost_usd.mean][$lte]=0.5, filter[latency_ms.p95][$lte]=1000,
    filter[tokens.mean][$lte]=5000, or filter[evaluators.<name>.mean][$gte]=0.8.
    """

    page: int
    """Page number."""

    page_size: int
    """Page size."""

    sort: str
    """
    Comma-separated list of fields to sort by, applied in order (the first field
    dominates); prefix any field with '-' for descending — e.g.
    '-evaluators.reward.mean,cost_usd.mean'. Each field is an evaluation attribute
    (name, created_at, updated_at, pinned_at) or an aggregate metric: run_count,
    test_case_count, cost_usd.<stat>, latency_ms.<stat>, tokens.<stat>, or
    evaluators.<name>.<stat>, where <stat> is one of mean, median, p90, p95, p99,
    sum, count. When omitted, defaults to -created_at with pinned evaluations first.
    """
