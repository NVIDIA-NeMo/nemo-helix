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

from typing import Optional

from ..._models import BaseModel

__all__ = ["ParetoConfig"]


class ParetoConfig(BaseModel):
    """Default X/Y metrics for a group's cost-vs-accuracy Pareto view.

    Metric ids use the same vocabulary as the evaluations list sort/filter fields — ``cost_usd``,
    ``latency_ms``, or ``evaluators.<name>``. Defaults to cost (x) vs latency (y): both exist for
    every group, so the chart always has something to render before anyone customizes it.
    """

    x_metric: Optional[str] = None
    """Metric plotted on the Pareto X axis."""

    y_metric: Optional[str] = None
    """Metric plotted on the Pareto Y axis."""
