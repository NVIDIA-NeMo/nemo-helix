# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible chat and completion schemas for the Guardrails service.

This module contains all request/response types for the OpenAI-compatible API endpoints.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import urlparse

from nhx.common.entities.values import Value
from pydantic import AliasChoices, BaseModel, Field, StrictBool, StrictFloat, StrictInt, StrictStr, field_validator

# =============================================================================
# Enums
# =============================================================================


class FinishReason(str, Enum):
    stop = "stop"
    max_tokens = "max_tokens"
    timeout = "timeout"
    other = "other"


class Role(str, Enum):
    user = "user"
    system = "system"
    function = "function"
    tool = "tool"
    assistant = "assistant"
    developer = "developer"


# =============================================================================
# Common Types
# =============================================================================


class LogProbs(Value):
    """Log probability information for regular (non-chat) completions."""

    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)


class TopLogprob(Value):
    """Represents an alternative token and its log probability."""

    token: str = Field(..., description="The token.")
    bytes: Optional[List[int]] = Field(default=None, description="UTF-8 bytes representation of the token.")
    logprob: float = Field(..., description="The log probability of this token.")


class ChatCompletionTokenLogprob(Value):
    """Log probability information for a single token in chat completions."""

    token: str = Field(..., description="The token.")
    bytes: Optional[List[int]] = Field(default=None, description="UTF-8 bytes representation of the token.")
    logprob: float = Field(..., description="The log probability of this token.")
    top_logprobs: List[TopLogprob] = Field(
        default_factory=list, description="List of the most likely tokens and their log probability at this position."
    )


class ChoiceLogprobs(Value):
    """Log probability information for a chat completion choice.

    This is used in both regular and streaming chat completions when
    logprobs=true is provided in the request.
    """

    content: Optional[List[ChatCompletionTokenLogprob]] = Field(
        default=None, description="A list of message content tokens with log probability information."
    )


class UsageInfo(Value):
    """Token usage information for completion requests."""

    prompt_tokens: int = Field(default=0, description="Number of tokens in the prompt.")
    total_tokens: int = Field(
        default=0, description="Total number of tokens used in the request (prompt + completion)."
    )
    completion_tokens: int = Field(default=0, description="Number of tokens in the generated completion.")


class BaseRequest(Value):
    """Base request schema for completion requests."""

    model: str = Field(
        ...,
        description="The model to use for completion. Must be one of the available models.",
    )
    response_format: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Format of the response. Use {'type': 'json_object'} for JSON mode or {'type': 'json_schema', 'json_schema': {...}} for structured outputs.",
    )
    max_tokens: Optional[StrictInt] = Field(
        default=None,
        ge=1,
        description="The maximum number of tokens that can be generated in the chat completion.",
    )
    n: Optional[StrictInt] = Field(
        default=None,
        ge=1,
        description="How many chat completion choices to generate for each input message.",
    )
    streaming: Optional[bool] = Field(
        default=False,
        alias="stream",
        description="If set, partial message deltas will be sent, like in ChatGPT.",
    )
    temperature: Optional[StrictFloat] = Field(
        default=None,
        description="What sampling temperature to use, between 0 and 2.",
        ge=0,
        le=2,
    )
    top_p: Optional[StrictFloat] = Field(
        default=None,
        ge=0,
        le=1,
        description="An alternative to sampling with temperature, called nucleus sampling.",
    )
    stop: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Up to 4 sequences where the API will stop generating further tokens.",
    )
    frequency_penalty: Optional[float] = Field(
        default=None,
        ge=-2,
        le=2,
        description="Positive values penalize new tokens based on their existing frequency in the text.",
    )
    presence_penalty: Optional[float] = Field(
        default=None,
        ge=-2,
        le=2,
        description="Positive values penalize new tokens based on whether they appear in the text so far.",
    )
    function_call: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description="Deprecated in favor of tool_choice. 'none' means the model will not call a function and instead generates a message. 'auto' means the model can pick between generating a message or calling a function. Specifying a particular function via {'name': 'my_function'} forces the model to call that function.",
        validation_alias=AliasChoices("function_call", "function_c"),
    )
    seed: Optional[int] = Field(default=None, description="If specified, attempts to sample deterministically.")
    logit_bias: Optional[Dict[str, float]] = Field(
        default=None,
        description="Modify the likelihood of specified tokens appearing in the completion. Maps token IDs (as strings) to bias values from -100 to 100.",
    )
    top_logprobs: Optional[int] = Field(
        default=None,
        description="The number of most likely tokens to return at each token position.",
        le=20,
        ge=0,
    )
    logprobs: Optional[StrictBool] = Field(
        default=None,
        description="Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message",
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description="Controls which (if any) tool is called by the model. 'none' means no tool is called, 'auto' lets the model decide, 'required' forces a tool call.",
    )
    user: Optional[str] = Field(
        default=None,
        description="A unique identifier representing your end-user, used by some providers for abuse monitoring.",
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="A list of tools the model may call. Each tool is an object with a 'type' field and a 'function' definition.",
    )
    ignore_eos: Optional[bool] = Field(
        default=None,
        description="Ignore the eos when running",
    )
    reasoning_effort: Optional[str] = Field(
        default=None,
        description="Constrains effort on reasoning for reasoning models. Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response.",
    )
    max_completion_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens. Preferred over max_tokens for reasoning models.",
    )
    stream_options: Optional[Dict[str, bool]] = Field(
        default=None,
        description="Options for streaming response. Only set this when stream=True. Supports include_usage to receive token usage in the final stream chunk.",
    )


# =============================================================================
# Chat Completion Types
# =============================================================================


class ImageURL(Value):
    """Image URL for vision requests."""

    url: str = Field(..., description="Either a URL of the image or the base64 encoded image data.")
    detail: Optional[Literal["auto", "low", "high"]] = Field(
        default=None, description="Specifies the detail level of the image."
    )

    class Config:
        extra = "forbid"

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if v.startswith("data:"):
            return v  # base64 data URI — no further validation needed

        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"Image URL scheme '{parsed.scheme or '<empty>'}' is not supported. "
                "The URL must be a HTTP, HTTPS, or base64 data URI (data:image/...)."
            )

        if parsed.scheme in ("http", "https") and not parsed.netloc:
            raise ValueError("Image URL must include a valid hostname.")
        return v


class ChatCompletionContentPartTextParam(Value):
    """Text content part for chat messages."""

    text: str = Field(..., description="The text content.")
    type: Literal["text"] = Field(..., description="The type of the content part.")

    class Config:
        extra = "forbid"


class ChatCompletionContentPartImageParam(Value):
    """Image content part for chat messages."""

    image_url: ImageURL = Field(..., description="The image URL information.")
    type: Literal["image_url"] = Field(..., description="The type of the content part.")

    class Config:
        extra = "forbid"


ChatCompletionContentPartParam = Union[ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam]


class Function(Value):
    """Function definition for tool calls."""

    arguments: str = Field(
        ..., description="The arguments to call the function with, as generated by the model in JSON format."
    )
    name: str = Field(..., description="The name of the function to call.")

    class Config:
        extra = "forbid"


class FunctionCall(Value):
    """Function call information."""

    arguments: str = Field(
        ..., description="The arguments to call the function with, as generated by the model in JSON format."
    )
    name: str = Field(..., description="The name of the function to call.")

    class Config:
        extra = "forbid"


class ChatCompletionMessageToolCallParam(Value):
    """Tool call parameter for chat completion messages."""

    id: str = Field(..., description="The ID of the tool call.")
    function: Function = Field(..., description="The function that the model called.")
    type: Literal["function"] = Field(..., description="The type of the tool. Currently, only `function` is supported.")

    class Config:
        extra = "forbid"


class ChatCompletionSystemMessageParam(Value):
    """System message parameter for chat completion."""

    content: str = Field(..., description="The contents of the system message.")
    role: Literal["system"] = Field(..., description="The role of the messages author, in this case `system`.")
    name: Optional[str] = Field(default=None, description="An optional name for the participant.")

    class Config:
        extra = "forbid"


class ChatCompletionUserMessageParam(Value):
    """User message parameter for chat completion."""

    content: Union[str, List[ChatCompletionContentPartParam]] = Field(
        ..., description="The contents of the user message."
    )
    role: Literal["user"] = Field(..., description="The role of the messages author, in this case `user`.")
    name: Optional[str] = Field(default=None, description="An optional name for the participant.")

    class Config:
        extra = "forbid"


class ChatCompletionAssistantMessageParam(Value):
    """Assistant message parameter for chat completion."""

    role: Literal["assistant"] = Field(..., description="The role of the messages author, in this case `assistant`.")
    content: Optional[str] = Field(default=None, description="The contents of the assistant message.")
    function_call: Optional[FunctionCall] = Field(default=None, description="Deprecated and replaced by `tool_calls`.")
    name: Optional[str] = Field(default=None, description="An optional name for the participant.")
    tool_calls: Optional[List[ChatCompletionMessageToolCallParam]] = Field(
        default=None, description="The tool calls generated by the model, such as function calls."
    )

    class Config:
        extra = "forbid"


class ChatCompletionToolMessageParam(Value):
    """Tool message parameter for chat completion."""

    content: str = Field(..., description="The contents of the tool message.")
    role: Literal["tool"] = Field(..., description="The role of the messages author, in this case `tool`.")
    tool_call_id: str = Field(..., description="Tool call that this message is responding to.")

    class Config:
        extra = "forbid"


class ChatCompletionFunctionMessageParam(Value):
    """Function message parameter for chat completion."""

    content: Optional[str] = Field(..., description="The contents of the function message.")
    name: str = Field(..., description="The name of the function to call.")
    role: Literal["function"] = Field(..., description="The role of the messages author, in this case `function`.")

    class Config:
        extra = "forbid"


ChatCompletionMessageParam = Union[
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionFunctionMessageParam,
]


class ChatMessage(Value):
    """Chat message with role and content."""

    content: Union[StrictStr, List[ChatCompletionContentPartParam]] = Field(
        ..., description="The content of the message. Can be text or a list of content parts."
    )
    role: Role = Field(..., description="The role of the message")
    name: Optional[str] = Field(default=None, description="An optional name for the participant.")


class ChoiceDeltaToolCallFunction(Value):
    """Function information in a streaming tool call delta."""

    name: Optional[str] = None
    arguments: Optional[str] = None


class ChoiceDeltaToolCall(Value):
    """Tool call information in a streaming delta."""

    index: int
    id: Optional[str] = None
    type: Optional[Literal["function"]] = None
    function: Optional[ChoiceDeltaToolCallFunction] = None


class ChoiceDeltaFunctionCall(Value):
    """Function call information in a streaming delta."""

    name: Optional[str] = None
    arguments: Optional[str] = None


class DeltaMessage(Value):
    """Delta message for streaming chat completions."""

    role: Optional[Literal["system", "user", "assistant", "tool"]] = None
    content: Optional[Union[str, List[ChatCompletionContentPartParam]]] = None
    function_call: Optional[ChoiceDeltaFunctionCall] = None
    tool_calls: Optional[List[ChoiceDeltaToolCall]] = None


class ChatCompletionMessageToolCall(Value):
    """Tool call in a chat completion message."""

    id: str = Field(..., description="The ID of the tool call.")
    function: Function = Field(..., description="The function that the model called.")
    type: Literal["function"] = Field(..., description="The type of the tool. Currently, only `function` is supported.")


class ChatCompletionMessage(Value):
    """Chat completion message generated by the model."""

    content: Optional[str] = Field(default=None, description="The contents of the message.")
    role: str = Field("assistant", description="The role of the author of this message.")
    function_call: Optional[FunctionCall] = Field(
        default=None,
        description="Deprecated and replaced by `tool_calls`. The name and arguments of a function that should be called, as generated by the model.",
    )
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = Field(
        default=None, description="The tool calls generated by the model, such as function calls."
    )


class ChatCompletionRequest(BaseRequest):
    """Chat completion request."""

    messages: List[ChatCompletionMessageParam] = Field(
        ..., description="A list of messages comprising the conversation so far"
    )
    vision: Optional[bool] = Field(
        default=None, description="Whether this is a vision-capable request with image inputs."
    )


class ChatCompletionResponseChoice(BaseModel):
    """A single choice in a chat completion response."""

    index: int = Field(..., description="The index of the choice in the list of choices.")
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls", "function_call"]] = Field(
        default=None, description="The reasons why the conversation ended."
    )
    logprobs: Optional[ChoiceLogprobs] = Field(
        default=None,
        description="The log probabilities of the output tokens. Only returned when logprobs=true is provided to the request.",
    )
    message: ChatCompletionMessage = Field(..., description="A chat completion message generated by the model.")


class ChatCompletionResponse(Value):
    """Chat completion response."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{str(uuid.uuid4())}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo
    system_fingerprint: Optional[str] = Field(
        default=None,
        description="Represents the backend configuration that the model runs with. Used with seed for determinism.",
    )


class ChatCompletionResponseStreamChoice(Value):
    """A single choice in a streaming chat completion response."""

    index: StrictInt
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls", "function_call"]] = None
    logprobs: Optional[ChoiceLogprobs] = Field(
        default=None,
        description="Log probability information for the choice. Only returned when logprobs=true is provided to the request.",
    )


class ChatCompletionStreamResponse(Value):
    """Streaming chat completion response chunk."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{str(uuid.uuid4())}")
    object: str = "chat.completion.chunk"
    created: StrictInt = Field(default_factory=lambda: StrictInt(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    system_fingerprint: Optional[str] = Field(
        default=None,
        description="Represents the backend configuration that the model runs with. Used with seed for determinism.",
    )


# =============================================================================
# Text Completion Types
# =============================================================================


class CompletionResponseChoice(Value):
    """A single choice in a completion response."""

    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = Field(
        default=None, description="The reasons why the conversation ended."
    )


class CompletionResponseStreamChoice(CompletionResponseChoice):
    """A single choice in a streaming completion response."""

    pass


class CompletionStreamResponse(Value):
    """Streaming completion response chunk."""

    id: str = Field(default_factory=lambda: f"cmpl-{str(uuid.uuid4())}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None


class CompletionResponse(Value):
    """Completion response."""

    id: str = Field(default_factory=lambda: f"cmpl-{str(uuid.uuid4())}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionRequest(BaseRequest):
    """Completion request for the completions endpoint."""

    prompt: Union[List[StrictInt], List[List[StrictInt]], str, List[str]] = Field(
        ...,
        min_length=1,
        description="User prompt or list of token ids.",
    )
    best_of: Optional[StrictInt] = Field(
        default=None,
        description='Generates best_of completions server-side and returns the "best" (the one with the highest log probability per token). Results cannot be streamed. When used with n, best_of controls the number of candidate completions and n specifies how many to return - best_of must be greater than n.',
    )
    echo: Optional[bool] = Field(
        default=None,
        description="If true, the response will include the prompt and optionally its token IDs and log probabilities.",
    )
    suffix: Optional[str] = Field(
        default=None,
        description="The suffix that comes after a completion of inserted text.",
    )
