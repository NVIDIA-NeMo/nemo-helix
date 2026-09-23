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

from typing import Dict, List, Union, Optional
from datetime import datetime

from ..._models import BaseModel
from .prompt_message import PromptMessage
from .chat_completion_tool import ChatCompletionTool
from ..shared.inference_params import InferenceParams

__all__ = ["Prompt"]


class Prompt(BaseModel):
    """A reusable, stored chat prompt.

    A Prompt captures the messages, declared template variables, optional tool
    definitions, and default inference parameters needed to invoke a model
    through the Inference Gateway. The unique identifier is workspace/name.
    """

    created_at: datetime
    """The timestamp of model entity creation"""

    name: str
    """Name of the entity.

    Name/workspace combo must be unique across all entities. Name must start with a
    lowercase letter, be 2-63 characters, and use lowercase letters, digits,
    hyphens, and dots (no consecutive hyphens, cannot end with a hyphen).
    """

    updated_at: datetime
    """The timestamp of the last model entity update"""

    workspace: str
    """The workspace of the entity.

    Allowed characters: letters (a-z, A-Z), digits (0-9), underscores, hyphens, and
    dots.
    """

    id: Optional[str] = None
    """Unique identifier for the prompt."""

    description: Optional[str] = None
    """Optional description of the prompt."""

    inference_params: Optional[InferenceParams] = None
    """Parameters for model inference.

    Extra fields can be supplied for additional options applied to the inference
    request directly. Fields not supported by the model may cause inference errors
    during evaluation.
    """

    input_variables: Optional[List[str]] = None
    """Names of the Jinja2 template variables the prompt expects."""

    messages: Optional[List[PromptMessage]] = None
    """Ordered list of chat messages that make up the prompt."""

    project: Optional[str] = None
    """The URN of the project associated with this entity."""

    response_format: Optional[Dict[str, object]] = None
    """Optional OpenAI-compatible response_format, e.g.

    a json_schema structured-output spec.
    """

    tags: Optional[List[str]] = None
    """Optional free-form tags for organizing prompts."""

    tool_choice: Union[str, Dict[str, object], None] = None
    """
    Controls which (if any) tool is called: 'none', 'auto', 'required', or a
    named-tool object.
    """

    tools: Optional[List[ChatCompletionTool]] = None
    """Optional OpenAI-compatible tool definitions to send with the prompt."""
