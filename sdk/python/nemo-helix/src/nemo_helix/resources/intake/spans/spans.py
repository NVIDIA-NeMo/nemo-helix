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

from typing_extensions import Literal

import httpx

from .groups import (
    GroupsResource,
    AsyncGroupsResource,
    GroupsResourceWithRawResponse,
    AsyncGroupsResourceWithRawResponse,
    GroupsResourceWithStreamingResponse,
    AsyncGroupsResourceWithStreamingResponse,
)
from ...._types import Body, Omit, Query, Headers, NotGiven, omit, not_given
from ...._utils import path_template, maybe_transform
from ...._compat import cached_property
from ...._resource import SyncAPIResource, AsyncAPIResource
from ...._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from ....pagination import SyncDefaultPagination, AsyncDefaultPagination
from ...._base_client import AsyncPaginator, make_request_options
from ....types.intake import SpanSortField, span_list_params
from .evaluator_results import (
    EvaluatorResultsResource,
    AsyncEvaluatorResultsResource,
    EvaluatorResultsResourceWithRawResponse,
    AsyncEvaluatorResultsResourceWithRawResponse,
    EvaluatorResultsResourceWithStreamingResponse,
    AsyncEvaluatorResultsResourceWithStreamingResponse,
)
from ....types.intake.span import Span
from ....types.intake.span_sort_field import SpanSortField
from ....types.intake.span_filter_param import SpanFilterParam

__all__ = ["SpansResource", "AsyncSpansResource"]


class SpansResource(SyncAPIResource):
    @cached_property
    def groups(self) -> GroupsResource:
        return GroupsResource(self._client)

    @cached_property
    def evaluator_results(self) -> EvaluatorResultsResource:
        return EvaluatorResultsResource(self._client)

    @cached_property
    def with_raw_response(self) -> SpansResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return SpansResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> SpansResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return SpansResourceWithStreamingResponse(self)

    def retrieve(
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
    ) -> Span:
        """
        Get Span

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
                "/apis/intake/v2/workspaces/{workspace}/spans/{span_id}", workspace=workspace, span_id=span_id
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Span,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: SpanFilterParam | Omit = omit,
        mode: Literal["summary", "preview", "detailed"] | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: SpanSortField | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> SyncDefaultPagination[Span]:
        """
        List Spans

        Args:
          filter: Filter spans by session_id, trace_id, parent_span_id, project, evaluation_name,
              test_case_name, source, kind, status, model, tool_name, provider, agent_id,
              agent_name, and started_at. Every field takes one exact value, except
              started_at, which takes gte and lte.

          mode: Response mode. summary omits payloads and raw attributes; preview includes input
              and output truncated to 300 characters; detailed returns full payloads and raw
              attributes.

          page: Page number.

          page_size: Page size.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        return self._get_api_list(
            path_template("/apis/intake/v2/workspaces/{workspace}/spans", workspace=workspace),
            page=SyncDefaultPagination[Span],
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "filter": filter,
                        "mode": mode,
                        "page": page,
                        "page_size": page_size,
                        "sort": sort,
                    },
                    span_list_params.SpanListParams,
                ),
            ),
            model=Span,
        )


class AsyncSpansResource(AsyncAPIResource):
    @cached_property
    def groups(self) -> AsyncGroupsResource:
        return AsyncGroupsResource(self._client)

    @cached_property
    def evaluator_results(self) -> AsyncEvaluatorResultsResource:
        return AsyncEvaluatorResultsResource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncSpansResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncSpansResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncSpansResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncSpansResourceWithStreamingResponse(self)

    async def retrieve(
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
    ) -> Span:
        """
        Get Span

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
                "/apis/intake/v2/workspaces/{workspace}/spans/{span_id}", workspace=workspace, span_id=span_id
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Span,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: SpanFilterParam | Omit = omit,
        mode: Literal["summary", "preview", "detailed"] | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: SpanSortField | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> AsyncPaginator[Span, AsyncDefaultPagination[Span]]:
        """
        List Spans

        Args:
          filter: Filter spans by session_id, trace_id, parent_span_id, project, evaluation_name,
              test_case_name, source, kind, status, model, tool_name, provider, agent_id,
              agent_name, and started_at. Every field takes one exact value, except
              started_at, which takes gte and lte.

          mode: Response mode. summary omits payloads and raw attributes; preview includes input
              and output truncated to 300 characters; detailed returns full payloads and raw
              attributes.

          page: Page number.

          page_size: Page size.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        return self._get_api_list(
            path_template("/apis/intake/v2/workspaces/{workspace}/spans", workspace=workspace),
            page=AsyncDefaultPagination[Span],
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "filter": filter,
                        "mode": mode,
                        "page": page,
                        "page_size": page_size,
                        "sort": sort,
                    },
                    span_list_params.SpanListParams,
                ),
            ),
            model=Span,
        )


class SpansResourceWithRawResponse:
    def __init__(self, spans: SpansResource) -> None:
        self._spans = spans

        self.retrieve = to_raw_response_wrapper(
            spans.retrieve,
        )
        self.list = to_raw_response_wrapper(
            spans.list,
        )

    @cached_property
    def groups(self) -> GroupsResourceWithRawResponse:
        return GroupsResourceWithRawResponse(self._spans.groups)

    @cached_property
    def evaluator_results(self) -> EvaluatorResultsResourceWithRawResponse:
        return EvaluatorResultsResourceWithRawResponse(self._spans.evaluator_results)


class AsyncSpansResourceWithRawResponse:
    def __init__(self, spans: AsyncSpansResource) -> None:
        self._spans = spans

        self.retrieve = async_to_raw_response_wrapper(
            spans.retrieve,
        )
        self.list = async_to_raw_response_wrapper(
            spans.list,
        )

    @cached_property
    def groups(self) -> AsyncGroupsResourceWithRawResponse:
        return AsyncGroupsResourceWithRawResponse(self._spans.groups)

    @cached_property
    def evaluator_results(self) -> AsyncEvaluatorResultsResourceWithRawResponse:
        return AsyncEvaluatorResultsResourceWithRawResponse(self._spans.evaluator_results)


class SpansResourceWithStreamingResponse:
    def __init__(self, spans: SpansResource) -> None:
        self._spans = spans

        self.retrieve = to_streamed_response_wrapper(
            spans.retrieve,
        )
        self.list = to_streamed_response_wrapper(
            spans.list,
        )

    @cached_property
    def groups(self) -> GroupsResourceWithStreamingResponse:
        return GroupsResourceWithStreamingResponse(self._spans.groups)

    @cached_property
    def evaluator_results(self) -> EvaluatorResultsResourceWithStreamingResponse:
        return EvaluatorResultsResourceWithStreamingResponse(self._spans.evaluator_results)


class AsyncSpansResourceWithStreamingResponse:
    def __init__(self, spans: AsyncSpansResource) -> None:
        self._spans = spans

        self.retrieve = async_to_streamed_response_wrapper(
            spans.retrieve,
        )
        self.list = async_to_streamed_response_wrapper(
            spans.list,
        )

    @cached_property
    def groups(self) -> AsyncGroupsResourceWithStreamingResponse:
        return AsyncGroupsResourceWithStreamingResponse(self._spans.groups)

    @cached_property
    def evaluator_results(self) -> AsyncEvaluatorResultsResourceWithStreamingResponse:
        return AsyncEvaluatorResultsResourceWithStreamingResponse(self._spans.evaluator_results)
