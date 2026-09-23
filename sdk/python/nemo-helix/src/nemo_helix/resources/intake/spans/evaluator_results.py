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

import httpx

from ...._types import Body, Query, Headers, NotGiven, not_given
from ...._utils import path_template
from ...._compat import cached_property
from ...._resource import SyncAPIResource, AsyncAPIResource
from ...._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from ...._base_client import make_request_options
from ....types.intake.spans.evaluator_result_list_response import EvaluatorResultListResponse

__all__ = ["EvaluatorResultsResource", "AsyncEvaluatorResultsResource"]


class EvaluatorResultsResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> EvaluatorResultsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return EvaluatorResultsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> EvaluatorResultsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return EvaluatorResultsResourceWithStreamingResponse(self)

    def list(
        self,
        span_id: str,
        *,
        workspace: str | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> EvaluatorResultListResponse:
        """
        List Evaluator Results For Span

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not span_id:
            raise ValueError(f"Expected a non-empty value for `span_id` but received {span_id!r}")
        return self._get(
            path_template(
                "/apis/intake/v2/workspaces/{workspace}/spans/{span_id}/evaluator-results",
                workspace=workspace,
                span_id=span_id,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=EvaluatorResultListResponse,
        )


class AsyncEvaluatorResultsResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncEvaluatorResultsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncEvaluatorResultsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncEvaluatorResultsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncEvaluatorResultsResourceWithStreamingResponse(self)

    async def list(
        self,
        span_id: str,
        *,
        workspace: str | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> EvaluatorResultListResponse:
        """
        List Evaluator Results For Span

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not span_id:
            raise ValueError(f"Expected a non-empty value for `span_id` but received {span_id!r}")
        return await self._get(
            path_template(
                "/apis/intake/v2/workspaces/{workspace}/spans/{span_id}/evaluator-results",
                workspace=workspace,
                span_id=span_id,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=EvaluatorResultListResponse,
        )


class EvaluatorResultsResourceWithRawResponse:
    def __init__(self, evaluator_results: EvaluatorResultsResource) -> None:
        self._evaluator_results = evaluator_results

        self.list = to_raw_response_wrapper(
            evaluator_results.list,
        )


class AsyncEvaluatorResultsResourceWithRawResponse:
    def __init__(self, evaluator_results: AsyncEvaluatorResultsResource) -> None:
        self._evaluator_results = evaluator_results

        self.list = async_to_raw_response_wrapper(
            evaluator_results.list,
        )


class EvaluatorResultsResourceWithStreamingResponse:
    def __init__(self, evaluator_results: EvaluatorResultsResource) -> None:
        self._evaluator_results = evaluator_results

        self.list = to_streamed_response_wrapper(
            evaluator_results.list,
        )


class AsyncEvaluatorResultsResourceWithStreamingResponse:
    def __init__(self, evaluator_results: AsyncEvaluatorResultsResource) -> None:
        self._evaluator_results = evaluator_results

        self.list = async_to_streamed_response_wrapper(
            evaluator_results.list,
        )
