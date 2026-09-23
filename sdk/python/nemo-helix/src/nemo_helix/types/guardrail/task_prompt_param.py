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

from typing import Union
from typing_extensions import Required, TypeAlias, TypedDict

from ..._types import SequenceNotStr
from .message_template_param import MessageTemplateParam

__all__ = ["TaskPromptParam", "Message"]

Message: TypeAlias = Union[MessageTemplateParam, str]


class TaskPromptParam(TypedDict, total=False):
    """Configuration for prompts that will be used for a specific task."""

    task: Required[str]
    """The id of the task associated with this prompt."""

    content: str
    """The content of the prompt, if it's a string."""

    max_length: int
    """The maximum length of the prompt in number of characters."""

    max_tokens: int
    """The maximum number of tokens that can be generated in the chat completion."""

    messages: SequenceNotStr[Message]
    """The list of messages included in the prompt. Used for chat models."""

    mode: str
    """Corresponds to the `prompting_mode` for which this prompt is fetched.

    Default is 'standard'.
    """

    models: SequenceNotStr[str]
    """If specified, the prompt will be used only for the given LLM engines/models.

    The format is a list of strings with the format: <engine> or <engine>/<model>.
    """

    output_parser: str
    """The name of the output parser to use for this prompt."""

    stop: SequenceNotStr[str]
    """If specified, will be configure stop tokens for models that support this."""
