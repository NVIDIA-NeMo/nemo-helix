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
from .input_rails import InputRails
from .action_rails import ActionRails
from .dialog_rails import DialogRails
from .output_rails import OutputRails
from .retrieval_rails import RetrievalRails
from .tool_input_rails import ToolInputRails
from .rails_config_data import RailsConfigData
from .tool_output_rails import ToolOutputRails

__all__ = ["Rails"]


class Rails(BaseModel):
    """Configuration of specific rails."""

    actions: Optional[ActionRails] = None
    """Configuration of action rails.

    Action rails control various options related to the execution of actions.
    Currently, only

    In the future multiple options will be added, e.g., what input validation should
    be performed per action, output validation, throttling, disabling, etc.
    """

    config: Optional[RailsConfigData] = None
    """Configuration data for specific rails that are supported out-of-the-box."""

    dialog: Optional[DialogRails] = None
    """Configuration of topical rails."""

    input: Optional[InputRails] = None
    """Configuration of input rails."""

    output: Optional[OutputRails] = None
    """Configuration of output rails."""

    retrieval: Optional[RetrievalRails] = None
    """Configuration of retrieval rails."""

    tool_input: Optional[ToolInputRails] = None
    """
    Configuration of tool input rails. Tool input rails are applied to tool results
    before they are processed. They can validate, filter, or transform tool outputs
    for security and safety.
    """

    tool_output: Optional[ToolOutputRails] = None
    """
    Configuration of tool output rails. Tool output rails are applied to tool calls
    before they are executed. They can validate tool names, parameters, and context
    to ensure safe tool usage.
    """
