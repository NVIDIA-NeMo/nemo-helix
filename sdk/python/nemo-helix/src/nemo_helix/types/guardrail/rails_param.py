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

from .input_rails_param import InputRailsParam
from .action_rails_param import ActionRailsParam
from .dialog_rails_param import DialogRailsParam
from .output_rails_param import OutputRailsParam
from .retrieval_rails_param import RetrievalRailsParam
from .tool_input_rails_param import ToolInputRailsParam
from .rails_config_data_param import RailsConfigDataParam
from .tool_output_rails_param import ToolOutputRailsParam

__all__ = ["RailsParam"]


class RailsParam(TypedDict, total=False):
    """Configuration of specific rails."""

    actions: ActionRailsParam
    """Configuration of action rails.

    Action rails control various options related to the execution of actions.
    Currently, only

    In the future multiple options will be added, e.g., what input validation should
    be performed per action, output validation, throttling, disabling, etc.
    """

    config: RailsConfigDataParam
    """Configuration data for specific rails that are supported out-of-the-box."""

    dialog: DialogRailsParam
    """Configuration of topical rails."""

    input: InputRailsParam
    """Configuration of input rails."""

    output: OutputRailsParam
    """Configuration of output rails."""

    retrieval: RetrievalRailsParam
    """Configuration of retrieval rails."""

    tool_input: ToolInputRailsParam
    """
    Configuration of tool input rails. Tool input rails are applied to tool results
    before they are processed. They can validate, filter, or transform tool outputs
    for security and safety.
    """

    tool_output: ToolOutputRailsParam
    """
    Configuration of tool output rails. Tool output rails are applied to tool calls
    before they are executed. They can validate tool names, parameters, and context
    to ensure safe tool usage.
    """
