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
from datetime import datetime

from ..._models import BaseModel
from .cost_rollup_response import CostRollupResponse
from .token_rollup_response import TokenRollupResponse
from .latency_rollup_response import LatencyRollupResponse

__all__ = ["TraceMetricPointResponse"]


class TraceMetricPointResponse(BaseModel):
    cached_tokens: TokenRollupResponse

    cost_usd: CostRollupResponse

    failed_run_count: int
    """Runs whose root span ended in error."""

    input_tokens: TokenRollupResponse

    latency_ms: LatencyRollupResponse

    output_tokens: TokenRollupResponse

    run_count: int
    """Agent runs started in this bucket."""

    total_tokens: TokenRollupResponse

    bucket_start: Optional[datetime] = None
    """Start of the bucket in the requested timezone. Omitted when bucket=total."""
