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

from typing import Dict, Union, Iterable
from datetime import datetime
from typing_extensions import Literal, Required, Annotated, TypedDict

from ...._utils import PropertyInfo
from .atif_metrics_param import AtifMetricsParam
from .atif_tool_call_param import AtifToolCallParam
from .atif_observation_param import AtifObservationParam
from .atif_content_part_param import AtifContentPartParam

__all__ = ["AtifStepAgentParam"]


class AtifStepAgentParam(TypedDict, total=False):
    source: Required[Literal["agent"]]

    step_id: Required[int]

    extra: Dict[str, object]

    is_copied_context: bool

    llm_call_count: int

    message: Union[str, Iterable[AtifContentPartParam]]

    metrics: AtifMetricsParam

    model_name: str

    observation: AtifObservationParam

    reasoning_content: str

    reasoning_effort: Union[str, float]

    timestamp: Annotated[Union[str, datetime], PropertyInfo(format="iso8601")]

    tool_calls: Iterable[AtifToolCallParam]
