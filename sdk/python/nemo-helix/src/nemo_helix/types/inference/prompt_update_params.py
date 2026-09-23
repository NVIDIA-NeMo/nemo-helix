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

from typing import Dict, Union, Iterable
from typing_extensions import TypedDict

from ..._types import SequenceNotStr
from .prompt_message_param import PromptMessageParam
from .chat_completion_tool_param import ChatCompletionToolParam
from ..shared_params.inference_params import InferenceParams

__all__ = ["PromptUpdateParams"]


class PromptUpdateParams(TypedDict, total=False):
    workspace: str

    description: str

    inference_params: InferenceParams
    """Parameters for model inference.

    Extra fields can be supplied for additional options applied to the inference
    request directly. Fields not supported by the model may cause inference errors
    during evaluation.
    """

    input_variables: SequenceNotStr[str]

    messages: Iterable[PromptMessageParam]

    project: str
    """The URN of the project associated with this prompt."""

    response_format: Dict[str, object]

    tags: SequenceNotStr[str]

    tool_choice: Union[str, Dict[str, object]]

    tools: Iterable[ChatCompletionToolParam]
