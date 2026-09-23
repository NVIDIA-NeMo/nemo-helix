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

from ..._models import BaseModel
from .llm_call_info import LLMCallInfo

__all__ = ["ExecutedAction"]


class ExecutedAction(BaseModel):
    """Information about an action that was executed."""

    action_name: str
    """The name of the action that was executed."""

    action_params: Optional[Dict[str, object]] = None
    """The parameters for the action."""

    duration: Optional[float] = None
    """How long the action took to execute, in seconds."""

    finished_at: Optional[float] = None
    """Timestamp for when the action finished."""

    llm_calls: Optional[List[LLMCallInfo]] = None
    """Information about the LLM calls made by the action."""

    return_value: Optional[object] = None
    """The value returned by the action."""

    started_at: Optional[float] = None
    """Timestamp for when the action started."""
