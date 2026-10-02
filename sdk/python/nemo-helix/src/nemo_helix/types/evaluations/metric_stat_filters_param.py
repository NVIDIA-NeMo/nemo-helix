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

from .number_filter_param import NumberFilterParam

__all__ = ["MetricStatFiltersParam"]


class MetricStatFiltersParam(TypedDict, total=False):
    """Numeric range filters keyed by rollup aggregate stat.

    Declaring each stat explicitly (rather than an open ``dict[str, NumberFilter]``) makes the valid
    stats visible in the OpenAPI schema, e.g. ``filter[cost_usd.mean][$lte]=0.5``. These stats must
    stay in sync with the runtime sort/filter grammar (``_METRIC_STATS`` in the evaluations
    endpoints); a unit test guards the parity.
    """

    count: NumberFilterParam

    mean: NumberFilterParam

    median: NumberFilterParam

    p90: NumberFilterParam

    p95: NumberFilterParam

    p99: NumberFilterParam

    sum: NumberFilterParam
