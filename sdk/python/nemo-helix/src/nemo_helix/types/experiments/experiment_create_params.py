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
from typing_extensions import Required, TypedDict

from .column_layout_param import ColumnLayoutParam
from .pareto_config_param import ParetoConfigParam

__all__ = ["ExperimentCreateParams"]


class ExperimentCreateParams(TypedDict, total=False):
    workspace: str

    name: Required[str]
    """Workspace-unique experiment name."""

    column_layout: ColumnLayoutParam
    """
    A saved table layout for a group's evaluations list: column order and which
    columns are hidden.

    Column ids are Studio's and cannot be enumerated here — the table builds a
    column per evaluator and metadata key found in the rows — so ids are stored and
    echoed back unvalidated.

    Visibility is stored as the _hidden_ ids rather than a map over every column, so
    a column that appears later (a new evaluator, a new metadata key) shows up by
    default.
    """

    default_sort: str
    """
    Default sort for this experiment's evaluations list, as a `sort`-param string: a
    comma-separated, ordered list of fields where the first is the primary sort and
    the rest break ties (leading '-' on a field = descending), e.g.
    '-evaluators.reward.mean,cost_usd.mean'. Defaults to '-created_at'. Accepts any
    field the evaluations list `sort` param does; clients apply it as the list
    `sort` param.
    """

    description: str
    """Human-readable purpose of the experiment."""

    insight_id: str
    """Reference to an external insight that seeded this experiment, if any."""

    is_favorite: bool
    """Whether this Experiment is marked as a favorite.

    Defaults to false on create; omit on update to preserve the existing value.
    """

    metadata: Dict[str, str]
    """Free-form producer metadata for the experiment."""

    pareto: ParetoConfigParam
    """Default X/Y metrics for a group's cost-vs-accuracy Pareto view.

    Metric ids use the same vocabulary as the evaluations list sort/filter fields —
    `cost_usd`, `latency_ms`, or `evaluators.<name>`. Defaults to cost (x) vs
    latency (y): both exist for every group, so the chart always has something to
    render before anyone customizes it.
    """

    show_evaluations_over_time: bool
    """Whether Studio should display this Experiment's Evaluation results over time.

    Defaults to false on create; omit on update to preserve the existing value.
    """

    summary: str
    """Human- or agent-authored summary of the experiment's findings."""
