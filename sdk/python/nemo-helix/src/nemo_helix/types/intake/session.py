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
from .span_status import SpanStatus

__all__ = ["Session"]


class Session(BaseModel):
    """
    Aggregate telemetry for one Intake session; does not include traces or span payloads.
    """

    id: str

    span_count: int

    started_at: datetime

    status: SpanStatus

    trace_count: int

    workspace: str

    cached_tokens: Optional[int] = None

    cost_input_usd: Optional[float] = None

    cost_output_usd: Optional[float] = None

    cost_usd: Optional[float] = None

    duration_ms: Optional[float] = None

    ended_at: Optional[datetime] = None

    input_tokens: Optional[int] = None

    output_tokens: Optional[int] = None

    total_tokens: Optional[int] = None
