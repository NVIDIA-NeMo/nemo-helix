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

from typing import Dict, Iterable
from typing_extensions import Literal, Required, TypedDict

from .atif_step_param import AtifStepParam
from .atif_agent_param import AtifAgentParam
from .atif_final_metrics_param import AtifFinalMetricsParam
from ..evaluation_context_param import EvaluationContextParam

__all__ = ["AtifCreateParams"]


class AtifCreateParams(TypedDict, total=False):
    workspace: str

    agent: Required[AtifAgentParam]

    schema_version: Required[
        Literal["ATIF-v1.0", "ATIF-v1.1", "ATIF-v1.2", "ATIF-v1.3", "ATIF-v1.4", "ATIF-v1.5", "ATIF-v1.6", "ATIF-v1.7"]
    ]

    continued_trajectory_ref: str

    evaluation_context: EvaluationContextParam
    """
    Identifies the Evaluation and optional test case associated with ingested
    telemetry.
    """

    extra: Dict[str, object]

    final_metrics: AtifFinalMetricsParam

    notes: str

    session_id: str

    steps: Iterable[AtifStepParam]

    subagent_trajectories: Iterable["AtifTrajectoryParam"]

    trajectory_id: str


from .atif_trajectory_param import AtifTrajectoryParam
