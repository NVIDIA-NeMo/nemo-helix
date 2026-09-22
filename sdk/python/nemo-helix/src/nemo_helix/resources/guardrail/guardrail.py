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

import httpx

from .configs import (
    ConfigsResource,
    AsyncConfigsResource,
    ConfigsResourceWithRawResponse,
    AsyncConfigsResourceWithRawResponse,
    ConfigsResourceWithStreamingResponse,
    AsyncConfigsResourceWithStreamingResponse,
)
from ..._types import Body, Omit, Query, Headers, NotGiven, SequenceNotStr, omit, not_given
from ..._utils import path_template, maybe_transform, async_maybe_transform
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from ..._base_client import make_request_options
from ...types.guardrail import GuardrailsDataParam, guardrail_check_params
from ...types.guardrail.guardrails_data_param import GuardrailsDataParam
from ...types.guardrail.guardrail_check_response import GuardrailCheckResponse

__all__ = ["GuardrailResource", "AsyncGuardrailResource"]


class GuardrailResource(SyncAPIResource):
    @cached_property
    def configs(self) -> ConfigsResource:
        return ConfigsResource(self._client)

    @cached_property
    def with_raw_response(self) -> GuardrailResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return GuardrailResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> GuardrailResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return GuardrailResourceWithStreamingResponse(self)

    def check(
        self,
        *,
        workspace: str | None = None,
        messages: Iterable[guardrail_check_params.Message],
        model: str,
        frequency_penalty: float | Omit = omit,
        function_call: Union[str, Dict[str, object]] | Omit = omit,
        guardrails: GuardrailsDataParam | Omit = omit,
        ignore_eos: bool | Omit = omit,
        logit_bias: Dict[str, float] | Omit = omit,
        logprobs: bool | Omit = omit,
        max_completion_tokens: int | Omit = omit,
        max_tokens: int | Omit = omit,
        n: int | Omit = omit,
        presence_penalty: float | Omit = omit,
        reasoning_effort: str | Omit = omit,
        response_format: Dict[str, object] | Omit = omit,
        seed: int | Omit = omit,
        stop: Union[str, SequenceNotStr[str]] | Omit = omit,
        stream: bool | Omit = omit,
        stream_options: Dict[str, bool] | Omit = omit,
        temperature: float | Omit = omit,
        tool_choice: Union[str, Dict[str, object]] | Omit = omit,
        tools: Iterable[Dict[str, object]] | Omit = omit,
        top_logprobs: int | Omit = omit,
        top_p: float | Omit = omit,
        user: str | Omit = omit,
        vision: bool | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> GuardrailCheckResponse:
        """
        Chat completion for the provided conversation.

        Args:
          messages: A list of messages comprising the conversation so far

          model: The model to use for completion. Must be one of the available models.

          frequency_penalty: Positive values penalize new tokens based on their existing frequency in the
              text.

          function_call: Deprecated in favor of tool_choice. 'none' means the model will not call a
              function and instead generates a message. 'auto' means the model can pick
              between generating a message or calling a function. Specifying a particular
              function via {'name': 'my_function'} forces the model to call that function.

          guardrails: Guardrails specific options for the request.

          ignore_eos: Ignore the eos when running

          logit_bias: Modify the likelihood of specified tokens appearing in the completion. Maps
              token IDs (as strings) to bias values from -100 to 100.

          logprobs: Whether to return log probabilities of the output tokens or not. If true,
              returns the log probabilities of each output token returned in the content of
              message

          max_completion_tokens: An upper bound for the number of tokens that can be generated for a completion,
              including visible output tokens and reasoning tokens. Preferred over max_tokens
              for reasoning models.

          max_tokens: The maximum number of tokens that can be generated in the chat completion.

          n: How many chat completion choices to generate for each input message.

          presence_penalty: Positive values penalize new tokens based on whether they appear in the text so
              far.

          reasoning_effort: Constrains effort on reasoning for reasoning models. Reducing reasoning effort
              can result in faster responses and fewer tokens used on reasoning in a response.

          response_format:
              Format of the response. Use {'type': 'json_object'} for JSON mode or {'type':
              'json_schema', 'json_schema': {...}} for structured outputs.

          seed: If specified, attempts to sample deterministically.

          stop: Up to 4 sequences where the API will stop generating further tokens.

          stream: If set, partial message deltas will be sent, like in ChatGPT.

          stream_options: Options for streaming response. Only set this when stream=True. Supports
              include_usage to receive token usage in the final stream chunk.

          temperature: What sampling temperature to use, between 0 and 2.

          tool_choice: Controls which (if any) tool is called by the model. 'none' means no tool is
              called, 'auto' lets the model decide, 'required' forces a tool call.

          tools: A list of tools the model may call. Each tool is an object with a 'type' field
              and a 'function' definition.

          top_logprobs: The number of most likely tokens to return at each token position.

          top_p: An alternative to sampling with temperature, called nucleus sampling.

          user: A unique identifier representing your end-user, used by some providers for abuse
              monitoring.

          vision: Whether this is a vision-capable request with image inputs.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        return self._post(
            path_template("/apis/guardrails/v2/workspaces/{workspace}/checks", workspace=workspace),
            body=maybe_transform(
                {
                    "messages": messages,
                    "model": model,
                    "frequency_penalty": frequency_penalty,
                    "function_call": function_call,
                    "guardrails": guardrails,
                    "ignore_eos": ignore_eos,
                    "logit_bias": logit_bias,
                    "logprobs": logprobs,
                    "max_completion_tokens": max_completion_tokens,
                    "max_tokens": max_tokens,
                    "n": n,
                    "presence_penalty": presence_penalty,
                    "reasoning_effort": reasoning_effort,
                    "response_format": response_format,
                    "seed": seed,
                    "stop": stop,
                    "stream": stream,
                    "stream_options": stream_options,
                    "temperature": temperature,
                    "tool_choice": tool_choice,
                    "tools": tools,
                    "top_logprobs": top_logprobs,
                    "top_p": top_p,
                    "user": user,
                    "vision": vision,
                },
                guardrail_check_params.GuardrailCheckParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=GuardrailCheckResponse,
        )


class AsyncGuardrailResource(AsyncAPIResource):
    @cached_property
    def configs(self) -> AsyncConfigsResource:
        return AsyncConfigsResource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncGuardrailResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncGuardrailResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncGuardrailResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncGuardrailResourceWithStreamingResponse(self)

    async def check(
        self,
        *,
        workspace: str | None = None,
        messages: Iterable[guardrail_check_params.Message],
        model: str,
        frequency_penalty: float | Omit = omit,
        function_call: Union[str, Dict[str, object]] | Omit = omit,
        guardrails: GuardrailsDataParam | Omit = omit,
        ignore_eos: bool | Omit = omit,
        logit_bias: Dict[str, float] | Omit = omit,
        logprobs: bool | Omit = omit,
        max_completion_tokens: int | Omit = omit,
        max_tokens: int | Omit = omit,
        n: int | Omit = omit,
        presence_penalty: float | Omit = omit,
        reasoning_effort: str | Omit = omit,
        response_format: Dict[str, object] | Omit = omit,
        seed: int | Omit = omit,
        stop: Union[str, SequenceNotStr[str]] | Omit = omit,
        stream: bool | Omit = omit,
        stream_options: Dict[str, bool] | Omit = omit,
        temperature: float | Omit = omit,
        tool_choice: Union[str, Dict[str, object]] | Omit = omit,
        tools: Iterable[Dict[str, object]] | Omit = omit,
        top_logprobs: int | Omit = omit,
        top_p: float | Omit = omit,
        user: str | Omit = omit,
        vision: bool | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> GuardrailCheckResponse:
        """
        Chat completion for the provided conversation.

        Args:
          messages: A list of messages comprising the conversation so far

          model: The model to use for completion. Must be one of the available models.

          frequency_penalty: Positive values penalize new tokens based on their existing frequency in the
              text.

          function_call: Deprecated in favor of tool_choice. 'none' means the model will not call a
              function and instead generates a message. 'auto' means the model can pick
              between generating a message or calling a function. Specifying a particular
              function via {'name': 'my_function'} forces the model to call that function.

          guardrails: Guardrails specific options for the request.

          ignore_eos: Ignore the eos when running

          logit_bias: Modify the likelihood of specified tokens appearing in the completion. Maps
              token IDs (as strings) to bias values from -100 to 100.

          logprobs: Whether to return log probabilities of the output tokens or not. If true,
              returns the log probabilities of each output token returned in the content of
              message

          max_completion_tokens: An upper bound for the number of tokens that can be generated for a completion,
              including visible output tokens and reasoning tokens. Preferred over max_tokens
              for reasoning models.

          max_tokens: The maximum number of tokens that can be generated in the chat completion.

          n: How many chat completion choices to generate for each input message.

          presence_penalty: Positive values penalize new tokens based on whether they appear in the text so
              far.

          reasoning_effort: Constrains effort on reasoning for reasoning models. Reducing reasoning effort
              can result in faster responses and fewer tokens used on reasoning in a response.

          response_format:
              Format of the response. Use {'type': 'json_object'} for JSON mode or {'type':
              'json_schema', 'json_schema': {...}} for structured outputs.

          seed: If specified, attempts to sample deterministically.

          stop: Up to 4 sequences where the API will stop generating further tokens.

          stream: If set, partial message deltas will be sent, like in ChatGPT.

          stream_options: Options for streaming response. Only set this when stream=True. Supports
              include_usage to receive token usage in the final stream chunk.

          temperature: What sampling temperature to use, between 0 and 2.

          tool_choice: Controls which (if any) tool is called by the model. 'none' means no tool is
              called, 'auto' lets the model decide, 'required' forces a tool call.

          tools: A list of tools the model may call. Each tool is an object with a 'type' field
              and a 'function' definition.

          top_logprobs: The number of most likely tokens to return at each token position.

          top_p: An alternative to sampling with temperature, called nucleus sampling.

          user: A unique identifier representing your end-user, used by some providers for abuse
              monitoring.

          vision: Whether this is a vision-capable request with image inputs.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        return await self._post(
            path_template("/apis/guardrails/v2/workspaces/{workspace}/checks", workspace=workspace),
            body=await async_maybe_transform(
                {
                    "messages": messages,
                    "model": model,
                    "frequency_penalty": frequency_penalty,
                    "function_call": function_call,
                    "guardrails": guardrails,
                    "ignore_eos": ignore_eos,
                    "logit_bias": logit_bias,
                    "logprobs": logprobs,
                    "max_completion_tokens": max_completion_tokens,
                    "max_tokens": max_tokens,
                    "n": n,
                    "presence_penalty": presence_penalty,
                    "reasoning_effort": reasoning_effort,
                    "response_format": response_format,
                    "seed": seed,
                    "stop": stop,
                    "stream": stream,
                    "stream_options": stream_options,
                    "temperature": temperature,
                    "tool_choice": tool_choice,
                    "tools": tools,
                    "top_logprobs": top_logprobs,
                    "top_p": top_p,
                    "user": user,
                    "vision": vision,
                },
                guardrail_check_params.GuardrailCheckParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=GuardrailCheckResponse,
        )


class GuardrailResourceWithRawResponse:
    def __init__(self, guardrail: GuardrailResource) -> None:
        self._guardrail = guardrail

        self.check = to_raw_response_wrapper(
            guardrail.check,
        )

    @cached_property
    def configs(self) -> ConfigsResourceWithRawResponse:
        return ConfigsResourceWithRawResponse(self._guardrail.configs)


class AsyncGuardrailResourceWithRawResponse:
    def __init__(self, guardrail: AsyncGuardrailResource) -> None:
        self._guardrail = guardrail

        self.check = async_to_raw_response_wrapper(
            guardrail.check,
        )

    @cached_property
    def configs(self) -> AsyncConfigsResourceWithRawResponse:
        return AsyncConfigsResourceWithRawResponse(self._guardrail.configs)


class GuardrailResourceWithStreamingResponse:
    def __init__(self, guardrail: GuardrailResource) -> None:
        self._guardrail = guardrail

        self.check = to_streamed_response_wrapper(
            guardrail.check,
        )

    @cached_property
    def configs(self) -> ConfigsResourceWithStreamingResponse:
        return ConfigsResourceWithStreamingResponse(self._guardrail.configs)


class AsyncGuardrailResourceWithStreamingResponse:
    def __init__(self, guardrail: AsyncGuardrailResource) -> None:
        self._guardrail = guardrail

        self.check = async_to_streamed_response_wrapper(
            guardrail.check,
        )

    @cached_property
    def configs(self) -> AsyncConfigsResourceWithStreamingResponse:
        return AsyncConfigsResourceWithStreamingResponse(self._guardrail.configs)
