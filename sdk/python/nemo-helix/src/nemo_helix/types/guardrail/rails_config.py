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

from .model import Model
from .rails import Rails
from ..._models import BaseModel
from .instruction import Instruction
from .task_prompt import TaskPrompt
from .tracing_config import TracingConfig

__all__ = ["RailsConfig"]


class RailsConfig(BaseModel):
    """Configuration object for the models and the rails."""

    actions_server_url: Optional[str] = None
    """The URL of the actions server that should be used for the rails."""

    colang_version: Optional[str] = None
    """The Colang version to use."""

    custom_data: Optional[Dict[str, object]] = None
    """Any custom configuration data that might be needed."""

    enable_multi_step_generation: Optional[bool] = None
    """Whether to enable multi-step generation for the LLM."""

    enable_rails_exceptions: Optional[bool] = None
    """
    If set, the pre-defined guardrails raise exceptions instead of returning
    pre-defined messages.
    """

    instructions: Optional[List[Instruction]] = None
    """List of instructions in natural language that the LLM should use."""

    lowest_temperature: Optional[float] = None
    """The lowest temperature that should be used for the LLM."""

    models: Optional[List[Model]] = None
    """The list of models used by the rails configuration."""

    passthrough: Optional[bool] = None
    """
    Whether the original prompt should pass through the guardrails configuration as
    is. This means it will not be altered in any way.
    """

    prompting_mode: Optional[str] = None
    """Allows choosing between different prompting strategies."""

    prompts: Optional[List[TaskPrompt]] = None
    """The prompts that should be used for the various LLM tasks."""

    rails: Optional[Rails] = None
    """Configuration of specific rails."""

    sample_conversation: Optional[str] = None
    """The sample conversation that should be used inside the prompts."""

    tracing: Optional[TracingConfig] = None
    """Configuration for tracing."""
