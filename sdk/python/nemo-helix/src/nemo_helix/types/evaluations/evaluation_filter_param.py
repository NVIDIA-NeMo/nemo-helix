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

from typing import Dict
from typing_extensions import TypedDict

from .number_filter_param import NumberFilterParam
from .metric_stat_filters_param import MetricStatFiltersParam
from ..shared_params.datetime_filter import DatetimeFilter

__all__ = ["EvaluationFilterParam"]


class EvaluationFilterParam(TypedDict, total=False):
    """Filter for listing Evaluations."""

    agent_name: str
    """Filter evaluations that observed this agent name in any ingested session."""

    agent_version: str
    """Filter evaluations that observed this agent version in any ingested session."""

    cost_usd: MetricStatFiltersParam
    """Numeric range filters keyed by rollup aggregate stat.

    Declaring each stat explicitly (rather than an open `dict[str, NumberFilter]`)
    makes the valid stats visible in the OpenAPI schema, e.g.
    `filter[cost_usd.mean][$lte]=0.5`. These stats must stay in sync with the
    runtime sort/filter grammar (`_METRIC_STATS` in the evaluations endpoints); a
    unit test guards the parity.
    """

    created_at: DatetimeFilter
    """
    Filter evaluations by creation timestamp; supports `$gte` and `$lte` for ranges.
    """

    created_by: str
    """Filter evaluations by the principal that created them."""

    dataset_name: str
    """Filter evaluations by dataset name."""

    dataset_version: str
    """Filter evaluations by dataset version."""

    evaluators: Dict[str, MetricStatFiltersParam]
    """Filter by an evaluator rollup stat, e.g.

    filter[evaluators.<name>.mean][$gte]=0.8.
    """

    experiment_group_id: str
    """
    Deprecated alias for `experiment_id`; filter evaluations by experiment group
    membership.
    """

    experiment_id: str
    """
    Filter evaluations by experiment group membership: matches evaluations whose
    `experiment_ids` include this group id (legacy rows still on
    `experiment_group_id` also match). This is the canonical membership filter — it
    mirrors the write-side `experiment_ids`.
    """

    is_deleted: bool
    """When true, returns only soft-deleted evaluations.

    Omit (or false) to see only live evaluations.
    """

    is_pinned: bool
    """When true, returns only pinned evaluations.

    When false, returns only unpinned evaluations. Omit to return both.
    """

    latency_ms: MetricStatFiltersParam
    """Numeric range filters keyed by rollup aggregate stat.

    Declaring each stat explicitly (rather than an open `dict[str, NumberFilter]`)
    makes the valid stats visible in the OpenAPI schema, e.g.
    `filter[cost_usd.mean][$lte]=0.5`. These stats must stay in sync with the
    runtime sort/filter grammar (`_METRIC_STATS` in the evaluations endpoints); a
    unit test guards the parity.
    """

    metadata: Dict[str, str]
    """Filter by a metadata key/value pair, e.g.

    filter[metadata.model]=claude-opus-4-8.
    """

    model_name: str
    """Filter evaluations that observed this model name in any ingested session."""

    name: str
    """Filter evaluations by name."""

    run_count: NumberFilterParam
    """Filter by run count, e.g. filter[run_count][$gte]=5."""

    tokens: MetricStatFiltersParam
    """Numeric range filters keyed by rollup aggregate stat.

    Declaring each stat explicitly (rather than an open `dict[str, NumberFilter]`)
    makes the valid stats visible in the OpenAPI schema, e.g.
    `filter[cost_usd.mean][$lte]=0.5`. These stats must stay in sync with the
    runtime sort/filter grammar (`_METRIC_STATS` in the evaluations endpoints); a
    unit test guards the parity.
    """

    updated_at: DatetimeFilter
    """
    Filter evaluations by last-updated timestamp; supports `$gte` and `$lte` for
    ranges.
    """
