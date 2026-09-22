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

__all__ = ["ToolCallingMetadataContent"]


class ToolCallingMetadataContent(BaseModel):
    """Content for tool-calling configuration on model filesets.

    Stores chat template and tool calling settings that are merged into
    the ModelSpec during checkpoint analysis.
    """

    auto_tool_choice: Optional[bool] = None
    """Whether to enable automatic tool choice."""

    chat_template: Optional[str] = None
    """Jinja2 chat template for the model."""

    tool_call_parser: Optional[str] = None
    """
    Name of the tool call parser (e.g., 'openai', 'hermes', 'pythonic',
    'llama3_json', 'mistral').
    """

    tool_call_plugin: Optional[str] = None
    """Reference to a fileset containing a custom tool call plugin Python file.

    Expected format: '{workspace}/{fileset_name}'.
    """
