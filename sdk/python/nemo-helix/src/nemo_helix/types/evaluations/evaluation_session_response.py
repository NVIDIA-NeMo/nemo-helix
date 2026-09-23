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
from ..intake.span_status import SpanStatus

__all__ = ["EvaluationSessionResponse"]


class EvaluationSessionResponse(BaseModel):
    """One ingested session of an Evaluation — a single test case execution.

    Hydrated from ClickHouse at read time by reading root/session membership from
    ``trace_index`` and joining page-bounded span/evaluator rollups.
    """

    evaluation_name: str

    root_span_id: str

    session_id: str

    started_at: datetime

    status: SpanStatus
    """Root-span status: success, error, cancelled, or unknown."""

    trace_id: str

    workspace: str

    cached_tokens: Optional[int] = None
    """Sum of cached tokens across this session's spans."""

    cost_total_usd: Optional[float] = None
    """Sum of cost across this session's spans."""

    ended_at: Optional[datetime] = None

    evaluator_scores: Optional[Dict[str, float]] = None
    """Per-evaluator session-mean score.

    Includes NUMERIC and BOOLEAN evaluator results only; text/categorical results
    are omitted.
    """

    input: Optional[str] = None
    """Root-span input text.

    Omitted in summary mode and truncated to 300 characters in preview mode.
    """

    input_tokens: Optional[int] = None
    """Sum of input tokens across this session's spans."""

    latency_ms: Optional[float] = None

    output: Optional[str] = None
    """Root-span output text.

    Omitted in summary mode and truncated to 300 characters in preview mode.
    """

    output_tokens: Optional[int] = None
    """Sum of output tokens across this session's spans."""

    test_case_id: Optional[str] = None
    """Deprecated alias for test_case_name. Use test_case_name instead."""

    test_case_name: Optional[str] = None
    """Test case name; null when the producer did not set one."""
