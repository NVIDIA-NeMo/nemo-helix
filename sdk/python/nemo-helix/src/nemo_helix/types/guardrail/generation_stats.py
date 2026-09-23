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

from ..._models import BaseModel

__all__ = ["GenerationStats"]


class GenerationStats(BaseModel):
    """General stats about the generation."""

    dialog_rails_duration: Optional[float] = None
    """The time in seconds spent in processing the dialog rails."""

    generation_rails_duration: Optional[float] = None
    """The time in seconds spent in generation rails."""

    input_rails_duration: Optional[float] = None
    """The time in seconds spent in processing the input rails."""

    llm_calls_count: Optional[int] = None
    """The number of LLM calls in total."""

    llm_calls_duration: Optional[float] = None
    """The time in seconds spent in LLM calls."""

    llm_calls_total_completion_tokens: Optional[int] = None
    """The total number of completion tokens."""

    llm_calls_total_prompt_tokens: Optional[int] = None
    """The total number of prompt tokens."""

    llm_calls_total_tokens: Optional[int] = None
    """The total number of tokens."""

    output_rails_duration: Optional[float] = None
    """The time in seconds spent in processing the output rails."""

    total_duration: Optional[float] = None
    """The total time in seconds."""
