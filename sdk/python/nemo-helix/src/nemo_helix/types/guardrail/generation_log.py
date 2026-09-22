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
from .activated_rail import ActivatedRail
from .generation_stats import GenerationStats

__all__ = ["GenerationLog"]


class GenerationLog(BaseModel):
    """Contains additional logging information associated with a generation call."""

    activated_rails: Optional[List[ActivatedRail]] = None
    """The list of rails that were activated during generation."""

    colang_history: Optional[str] = None
    """The Colang history associated with the generation."""

    internal_events: Optional[List[Dict[str, object]]] = None
    """The complete sequence of internal events generated."""

    llm_calls: Optional[List[LLMCallInfo]] = None
    """The list of LLM calls that have been made to fulfill the generation request."""

    stats: Optional[GenerationStats] = None
    """General stats about the generation."""
