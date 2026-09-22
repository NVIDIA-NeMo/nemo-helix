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

from typing import List, Optional
from datetime import datetime

from ..._models import BaseModel
from .span_status import SpanStatus
from .evaluation_context import EvaluationContext

__all__ = ["Trace"]


class Trace(BaseModel):
    id: str

    session_id: str

    started_at: datetime

    status: SpanStatus

    workspace: str

    agent_name: Optional[str] = None

    agent_version: Optional[str] = None

    cached_tokens: Optional[int] = None

    cost_input_usd: Optional[float] = None

    cost_output_usd: Optional[float] = None

    cost_usd: Optional[float] = None

    duration_ms: Optional[float] = None

    ended_at: Optional[datetime] = None

    error_count: Optional[int] = None

    evaluation_context: Optional[EvaluationContext] = None
    """
    Identifies the Evaluation and optional test case associated with ingested
    telemetry.
    """

    input: Optional[str] = None
    """Root-span input text.

    Omitted in summary mode and truncated to 300 characters in preview mode.
    """

    input_tokens: Optional[int] = None

    models: Optional[List[str]] = None
    """Distinct models used across the trace's spans. Omitted in summary mode."""

    name: Optional[str] = None

    output: Optional[str] = None
    """Root-span output text.

    Omitted in summary mode and truncated to 300 characters in preview mode.
    """

    output_tokens: Optional[int] = None

    providers: Optional[List[str]] = None
    """Distinct inference providers used across the trace's spans.

    Omitted in summary mode.
    """

    root_span_id: Optional[str] = None

    span_count: Optional[int] = None

    total_tokens: Optional[int] = None
