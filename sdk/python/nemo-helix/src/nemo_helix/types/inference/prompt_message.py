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

from ..._models import BaseModel
from .prompt_message_role import PromptMessageRole

__all__ = ["PromptMessage"]


class PromptMessage(BaseModel):
    """A single templated message in a chat prompt.

    ``content`` is a Jinja2 template body that may reference the prompt's
    declared ``input_variables`` (e.g. ``{{ topic }}``).
    """

    content: str
    """Templated message content. May contain template variables."""

    role: PromptMessageRole
    """Role of a message author in a chat prompt.

    Follows the OpenAI chat schema the Inference Gateway speaks
    (`/v1/chat/completions`).
    """
