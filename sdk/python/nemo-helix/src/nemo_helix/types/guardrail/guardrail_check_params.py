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
from typing_extensions import Required, TypeAlias, TypedDict

from ..._types import SequenceNotStr
from .guardrails_data_param import GuardrailsDataParam
from .chat_completion_tool_message_param import ChatCompletionToolMessageParam
from .chat_completion_user_message_param import ChatCompletionUserMessageParam
from .chat_completion_system_message_param import ChatCompletionSystemMessageParam
from .chat_completion_function_message_param import ChatCompletionFunctionMessageParam
from .chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam

__all__ = ["GuardrailCheckParams", "Message"]


class GuardrailCheckParams(TypedDict, total=False):
    workspace: str

    messages: Required[Iterable[Message]]
    """A list of messages comprising the conversation so far"""

    model: Required[str]
    """The model to use for completion. Must be one of the available models."""

    frequency_penalty: float
    """
    Positive values penalize new tokens based on their existing frequency in the
    text.
    """

    function_call: Union[str, Dict[str, object]]
    """Deprecated in favor of tool_choice.

    'none' means the model will not call a function and instead generates a message.
    'auto' means the model can pick between generating a message or calling a
    function. Specifying a particular function via {'name': 'my_function'} forces
    the model to call that function.
    """

    guardrails: GuardrailsDataParam
    """Guardrails specific options for the request."""

    ignore_eos: bool
    """Ignore the eos when running"""

    logit_bias: Dict[str, float]
    """Modify the likelihood of specified tokens appearing in the completion.

    Maps token IDs (as strings) to bias values from -100 to 100.
    """

    logprobs: bool
    """Whether to return log probabilities of the output tokens or not.

    If true, returns the log probabilities of each output token returned in the
    content of message
    """

    max_completion_tokens: int
    """
    An upper bound for the number of tokens that can be generated for a completion,
    including visible output tokens and reasoning tokens. Preferred over max_tokens
    for reasoning models.
    """

    max_tokens: int
    """The maximum number of tokens that can be generated in the chat completion."""

    n: int
    """How many chat completion choices to generate for each input message."""

    presence_penalty: float
    """
    Positive values penalize new tokens based on whether they appear in the text so
    far.
    """

    reasoning_effort: str
    """Constrains effort on reasoning for reasoning models.

    Reducing reasoning effort can result in faster responses and fewer tokens used
    on reasoning in a response.
    """

    response_format: Dict[str, object]
    """Format of the response.

    Use {'type': 'json_object'} for JSON mode or {'type': 'json_schema',
    'json_schema': {...}} for structured outputs.
    """

    seed: int
    """If specified, attempts to sample deterministically."""

    stop: Union[str, SequenceNotStr[str]]
    """Up to 4 sequences where the API will stop generating further tokens."""

    stream: bool
    """If set, partial message deltas will be sent, like in ChatGPT."""

    stream_options: Dict[str, bool]
    """Options for streaming response.

    Only set this when stream=True. Supports include_usage to receive token usage in
    the final stream chunk.
    """

    temperature: float
    """What sampling temperature to use, between 0 and 2."""

    tool_choice: Union[str, Dict[str, object]]
    """Controls which (if any) tool is called by the model.

    'none' means no tool is called, 'auto' lets the model decide, 'required' forces
    a tool call.
    """

    tools: Iterable[Dict[str, object]]
    """A list of tools the model may call.

    Each tool is an object with a 'type' field and a 'function' definition.
    """

    top_logprobs: int
    """The number of most likely tokens to return at each token position."""

    top_p: float
    """An alternative to sampling with temperature, called nucleus sampling."""

    user: str
    """
    A unique identifier representing your end-user, used by some providers for abuse
    monitoring.
    """

    vision: bool
    """Whether this is a vision-capable request with image inputs."""


Message: TypeAlias = Union[
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionFunctionMessageParam,
]
