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

from typing import Dict, List, Optional
from datetime import datetime

from ..._compat import PYDANTIC_V1, ConfigDict
from ..._models import BaseModel
from .evaluator_aggregate import EvaluatorAggregate

__all__ = ["EvaluationResponse"]


class EvaluationResponse(BaseModel):
    """Evaluation as served by the API, including ClickHouse-hydrated rollups."""

    id: str

    dataset_name: str

    experiment_group_id: str
    """Deprecated single-experiment alias; the first of experiment_ids.

    Use experiment_ids.
    """

    experiment_ids: List[str]
    """Entity ids of the Experiments this Evaluation belongs to (>=1)."""

    name: str

    workspace: str

    agent_names: Optional[List[str]] = None
    """Distinct agent names observed across ingested sessions for this evaluation."""

    agent_versions: Optional[List[str]] = None
    """Distinct agent versions observed across ingested sessions for this evaluation."""

    aggregate_scores: Optional[Dict[str, EvaluatorAggregate]] = None

    cost_usd: Optional[EvaluatorAggregate] = None
    """Aggregate statistics over evaluator scores or session-level metric values."""

    created_at: Optional[datetime] = None

    dataset_version: Optional[str] = None

    description: Optional[str] = None

    evaluator_names: Optional[List[str]] = None

    latency_ms: Optional[EvaluatorAggregate] = None
    """Aggregate statistics over evaluator scores or session-level metric values."""

    metadata: Optional[Dict[str, str]] = None

    model_names: Optional[List[str]] = None
    """Distinct model names observed across ingested sessions for this evaluation."""

    parent_evaluation_id: Optional[str] = None

    pinned_at: Optional[datetime] = None
    """Timestamp at which the evaluation was pinned, or null if unpinned.

    Managed via POST/DELETE /evaluations/{name}/pin.
    """

    root_cause: Optional[str] = None

    run_count: Optional[int] = None
    """
    Number of distinct ingested evaluation sessions; one session is treated as one
    run.
    """

    source_link: Optional[str] = None

    status: Optional[str] = None

    test_case_count: Optional[int] = None
    """Number of distinct test cases in the evaluation, i.e.

    distinct test_case_name values (sessions with no test_case_name each count as
    their own). A test case run k times counts once; the rollup metrics are averaged
    per test case before pooling across test cases.
    """

    tokens: Optional[EvaluatorAggregate] = None
    """Aggregate statistics over evaluator scores or session-level metric values."""

    updated_at: Optional[datetime] = None

    if not PYDANTIC_V1:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
