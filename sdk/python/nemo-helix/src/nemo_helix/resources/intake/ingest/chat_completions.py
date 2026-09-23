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

from typing import Dict

import httpx

from ...._types import Body, Omit, Query, Headers, NotGiven, omit, not_given
from ...._utils import path_template, maybe_transform, async_maybe_transform
from ...._compat import cached_property
from ...._resource import SyncAPIResource, AsyncAPIResource
from ...._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from ...._base_client import make_request_options
from ....types.intake.ingest import (
    chat_completion_create_params,
)
from ....types.intake.evaluation_context_param import EvaluationContextParam
from ....types.intake.ingest.chat_completions_ingest_response import ChatCompletionsIngestResponse
from ....types.intake.ingest.captured_chat_completions_request_param import CapturedChatCompletionsRequestParam
from ....types.intake.ingest.captured_chat_completions_response_param import CapturedChatCompletionsResponseParam

__all__ = ["ChatCompletionsResource", "AsyncChatCompletionsResource"]


class ChatCompletionsResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> ChatCompletionsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return ChatCompletionsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> ChatCompletionsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return ChatCompletionsResourceWithStreamingResponse(self)

    def create(
        self,
        *,
        workspace: str | None = None,
        request: CapturedChatCompletionsRequestParam,
        response: CapturedChatCompletionsResponseParam,
        cost_details: Dict[str, float] | Omit = omit,
        cost_input_usd: float | Omit = omit,
        cost_output_usd: float | Omit = omit,
        cost_usd: float | Omit = omit,
        evaluation_context: EvaluationContextParam | Omit = omit,
        provider: str | Omit = omit,
        session_id: str | Omit = omit,
        trace_id: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ChatCompletionsIngestResponse:
        """
        Ingest Chat Completion

        Args:
          request: Flexible captured chat-completions request.

          response: Flexible captured chat-completions response.

          cost_details: Additional estimated cost breakdown fields in USD.

          cost_input_usd: Estimated input-token cost of this model call in USD.

          cost_output_usd: Estimated output-token cost of this model call in USD.

          cost_usd: Total estimated cost of this model call in USD. This matches ATIF step metrics;
              Intake stores it as semantic cost_total_usd on spans.

          evaluation_context: Identifies the Evaluation and optional test case associated with ingested
              telemetry.

          session_id: Groups related chat-completions calls without forcing them into the same trace.

          trace_id: Opt into joining an existing trace built via OTel or ATIF. This is not a
              grouping mechanism for chat-completions calls; use session_id to group related
              calls.

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
            path_template("/apis/intake/v2/workspaces/{workspace}/ingest/chat-completions", workspace=workspace),
            body=maybe_transform(
                {
                    "request": request,
                    "response": response,
                    "cost_details": cost_details,
                    "cost_input_usd": cost_input_usd,
                    "cost_output_usd": cost_output_usd,
                    "cost_usd": cost_usd,
                    "evaluation_context": evaluation_context,
                    "provider": provider,
                    "session_id": session_id,
                    "trace_id": trace_id,
                },
                chat_completion_create_params.ChatCompletionCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ChatCompletionsIngestResponse,
        )


class AsyncChatCompletionsResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncChatCompletionsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncChatCompletionsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncChatCompletionsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncChatCompletionsResourceWithStreamingResponse(self)

    async def create(
        self,
        *,
        workspace: str | None = None,
        request: CapturedChatCompletionsRequestParam,
        response: CapturedChatCompletionsResponseParam,
        cost_details: Dict[str, float] | Omit = omit,
        cost_input_usd: float | Omit = omit,
        cost_output_usd: float | Omit = omit,
        cost_usd: float | Omit = omit,
        evaluation_context: EvaluationContextParam | Omit = omit,
        provider: str | Omit = omit,
        session_id: str | Omit = omit,
        trace_id: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ChatCompletionsIngestResponse:
        """
        Ingest Chat Completion

        Args:
          request: Flexible captured chat-completions request.

          response: Flexible captured chat-completions response.

          cost_details: Additional estimated cost breakdown fields in USD.

          cost_input_usd: Estimated input-token cost of this model call in USD.

          cost_output_usd: Estimated output-token cost of this model call in USD.

          cost_usd: Total estimated cost of this model call in USD. This matches ATIF step metrics;
              Intake stores it as semantic cost_total_usd on spans.

          evaluation_context: Identifies the Evaluation and optional test case associated with ingested
              telemetry.

          session_id: Groups related chat-completions calls without forcing them into the same trace.

          trace_id: Opt into joining an existing trace built via OTel or ATIF. This is not a
              grouping mechanism for chat-completions calls; use session_id to group related
              calls.

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
            path_template("/apis/intake/v2/workspaces/{workspace}/ingest/chat-completions", workspace=workspace),
            body=await async_maybe_transform(
                {
                    "request": request,
                    "response": response,
                    "cost_details": cost_details,
                    "cost_input_usd": cost_input_usd,
                    "cost_output_usd": cost_output_usd,
                    "cost_usd": cost_usd,
                    "evaluation_context": evaluation_context,
                    "provider": provider,
                    "session_id": session_id,
                    "trace_id": trace_id,
                },
                chat_completion_create_params.ChatCompletionCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ChatCompletionsIngestResponse,
        )


class ChatCompletionsResourceWithRawResponse:
    def __init__(self, chat_completions: ChatCompletionsResource) -> None:
        self._chat_completions = chat_completions

        self.create = to_raw_response_wrapper(
            chat_completions.create,
        )


class AsyncChatCompletionsResourceWithRawResponse:
    def __init__(self, chat_completions: AsyncChatCompletionsResource) -> None:
        self._chat_completions = chat_completions

        self.create = async_to_raw_response_wrapper(
            chat_completions.create,
        )


class ChatCompletionsResourceWithStreamingResponse:
    def __init__(self, chat_completions: ChatCompletionsResource) -> None:
        self._chat_completions = chat_completions

        self.create = to_streamed_response_wrapper(
            chat_completions.create,
        )


class AsyncChatCompletionsResourceWithStreamingResponse:
    def __init__(self, chat_completions: AsyncChatCompletionsResource) -> None:
        self._chat_completions = chat_completions

        self.create = async_to_streamed_response_wrapper(
            chat_completions.create,
        )
