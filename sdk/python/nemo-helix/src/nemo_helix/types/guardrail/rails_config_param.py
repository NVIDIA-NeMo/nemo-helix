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
from typing_extensions import TypedDict

from .model_param import ModelParam
from .rails_param import RailsParam
from .instruction_param import InstructionParam
from .task_prompt_param import TaskPromptParam
from .tracing_config_param import TracingConfigParam

__all__ = ["RailsConfigParam"]


class RailsConfigParam(TypedDict, total=False):
    """Configuration object for the models and the rails."""

    actions_server_url: str
    """The URL of the actions server that should be used for the rails."""

    colang_version: str
    """The Colang version to use."""

    custom_data: Dict[str, object]
    """Any custom configuration data that might be needed."""

    enable_multi_step_generation: bool
    """Whether to enable multi-step generation for the LLM."""

    enable_rails_exceptions: bool
    """
    If set, the pre-defined guardrails raise exceptions instead of returning
    pre-defined messages.
    """

    instructions: Iterable[InstructionParam]
    """List of instructions in natural language that the LLM should use."""

    lowest_temperature: float
    """The lowest temperature that should be used for the LLM."""

    models: Iterable[ModelParam]
    """The list of models used by the rails configuration."""

    passthrough: bool
    """
    Whether the original prompt should pass through the guardrails configuration as
    is. This means it will not be altered in any way.
    """

    prompting_mode: str
    """Allows choosing between different prompting strategies."""

    prompts: Iterable[TaskPromptParam]
    """The prompts that should be used for the various LLM tasks."""

    rails: RailsParam
    """Configuration of specific rails."""

    sample_conversation: str
    """The sample conversation that should be used inside the prompts."""

    tracing: TracingConfigParam
    """Configuration for tracing."""
