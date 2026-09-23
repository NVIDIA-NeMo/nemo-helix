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

from typing import Dict, Optional
from datetime import datetime

from ..._models import BaseModel
from .span_kind import SpanKind
from .span_status import SpanStatus
from .span_evaluation_context import SpanEvaluationContext

__all__ = ["Span"]


class Span(BaseModel):
    ingested_at: datetime

    kind: SpanKind

    session_id: str

    source: str

    span_id: str

    started_at: datetime

    status: SpanStatus

    workspace: str

    agent_id: Optional[str] = None

    agent_name: Optional[str] = None

    cached_tokens: Optional[int] = None

    cost_details: Optional[Dict[str, float]] = None

    cost_input_usd: Optional[float] = None

    cost_output_usd: Optional[float] = None

    cost_total_usd: Optional[float] = None

    ended_at: Optional[datetime] = None

    error_message: Optional[str] = None
    """Normalized error message.

    In summary and preview modes this is truncated to 1000 characters.
    """

    error_type: Optional[str] = None

    evaluation_context: Optional[SpanEvaluationContext] = None

    input: Optional[str] = None
    """Span input text.

    Omitted in summary mode and truncated to 300 characters in preview mode.
    """

    input_tokens: Optional[int] = None

    model: Optional[str] = None

    name: Optional[str] = None

    output: Optional[str] = None
    """Span output text.

    Omitted in summary mode and truncated to 300 characters in preview mode.
    """

    output_tokens: Optional[int] = None

    parent_span_id: Optional[str] = None

    project: Optional[str] = None

    prompt_id: Optional[str] = None

    provider: Optional[str] = None

    raw_attributes: Optional[str] = None

    tool_name: Optional[str] = None

    total_tokens: Optional[int] = None

    trace_id: Optional[str] = None

    usage_details: Optional[Dict[str, int]] = None
