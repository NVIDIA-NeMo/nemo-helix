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

from typing_extensions import TypedDict

__all__ = ["GenerationLogOptionsParam"]


class GenerationLogOptionsParam(TypedDict, total=False):
    """Options for what should be included in the generation log."""

    activated_rails: bool
    """
    Include detailed information about the rails that were activated during
    generation.
    """

    colang_history: bool
    """Include the history of the conversation in Colang format."""

    internal_events: bool
    """Include the array of internal generated events."""

    llm_calls: bool
    """Include information about all the LLM calls that were made.

    This includes: prompt, completion, token usage, raw response, etc.
    """

    stats: bool
    """
    Include generation statistics — rail durations, LLM call counts, and token
    usage.
    """
