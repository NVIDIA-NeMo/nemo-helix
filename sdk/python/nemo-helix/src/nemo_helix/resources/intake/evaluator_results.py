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

from ..._types import Body, Omit, Query, Headers, NotGiven, omit, not_given
from ..._utils import path_template, maybe_transform, async_maybe_transform
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from ...pagination import SyncDefaultPagination, AsyncDefaultPagination
from ..._base_client import AsyncPaginator, make_request_options
from ...types.intake import (
    EvaluatorResultDataType,
    EvaluatorResultSortField,
    evaluator_result_list_params,
    evaluator_result_create_params,
)
from ...types.intake.evaluator_result import EvaluatorResult
from ...types.intake.evaluator_result_data_type import EvaluatorResultDataType
from ...types.intake.evaluator_result_sort_field import EvaluatorResultSortField
from ...types.intake.evaluator_result_filter_param import EvaluatorResultFilterParam
from ..._exceptions import ConflictError

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

    def create(
        self,
        *,
        workspace: str | None = None,
        data_type: EvaluatorResultDataType,
        name: str,
        session_id: str,
        span_id: str,
        comment: str | Omit = omit,
        string_value: str | Omit = omit,
        value: float | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> EvaluatorResult:
        """
        Create Evaluator Result

        Args:
          data_type: Discriminator for which of value / string_value carries the payload.

          name: Evaluator / metric identity (e.g. 'faithfulness/v1').

          session_id: Session id the target span belongs to. Denormalized so session-scoped reads stay
              fast.

          span_id: Target span id. Not validated against existing spans (loose target policy).

          comment: Free-text rationale or explanation.

          string_value: String value. Required when data_type is CATEGORICAL or TEXT.

          value: Numeric value. Required when data_type is NUMERIC or BOOLEAN (0|1).


          exist_ok: Do not raise an error if the resource already exists. Returns the existing resource.


          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        try:
            if workspace is None:
                workspace = self._client._get_workspace_path_param()
            if not workspace:
                raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
            return self._post(
                path_template("/apis/intake/v2/workspaces/{workspace}/evaluator-results", workspace=workspace),
                body=maybe_transform(
                    {
                        "data_type": data_type,
                        "name": name,
                        "session_id": session_id,
                        "span_id": span_id,
                        "comment": comment,
                        "string_value": string_value,
                        "value": value,
                    },
                    evaluator_result_create_params.EvaluatorResultCreateParams,
                ),
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=EvaluatorResult,
            )
        except ConflictError:
            if not exist_ok:
                raise
            return self.retrieve(name = name, workspace = workspace)

    def retrieve(
        self,
        evaluator_result_id: str,
        *,
        workspace: str | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> EvaluatorResult:
        """
        Get Evaluator Result

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
        if not evaluator_result_id:
            raise ValueError(
                f"Expected a non-empty value for `evaluator_result_id` but received {evaluator_result_id!r}"
            )
        return self._get(
            path_template(
                "/apis/intake/v2/workspaces/{workspace}/evaluator-results/{evaluator_result_id}",
                workspace=workspace,
                evaluator_result_id=evaluator_result_id,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=EvaluatorResult,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: EvaluatorResultFilterParam | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: EvaluatorResultSortField | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> SyncDefaultPagination[EvaluatorResult]:
        """
        List Evaluator Results

        Args:
          filter: Filter evaluator results by span_id, session_id, name, data_type, created_by,
              value range, and created_at range.

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
            path_template("/apis/intake/v2/workspaces/{workspace}/evaluator-results", workspace=workspace),
            page=SyncDefaultPagination[EvaluatorResult],
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "filter": filter,
                        "page": page,
                        "page_size": page_size,
                        "sort": sort,
                    },
                    evaluator_result_list_params.EvaluatorResultListParams,
                ),
            ),
            model=EvaluatorResult,
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

    async def create(
        self,
        *,
        workspace: str | None = None,
        data_type: EvaluatorResultDataType,
        name: str,
        session_id: str,
        span_id: str,
        comment: str | Omit = omit,
        string_value: str | Omit = omit,
        value: float | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> EvaluatorResult:
        """
        Create Evaluator Result

        Args:
          data_type: Discriminator for which of value / string_value carries the payload.

          name: Evaluator / metric identity (e.g. 'faithfulness/v1').

          session_id: Session id the target span belongs to. Denormalized so session-scoped reads stay
              fast.

          span_id: Target span id. Not validated against existing spans (loose target policy).

          comment: Free-text rationale or explanation.

          string_value: String value. Required when data_type is CATEGORICAL or TEXT.

          value: Numeric value. Required when data_type is NUMERIC or BOOLEAN (0|1).


          exist_ok: Do not raise an error if the resource already exists. Returns the existing resource.


          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        try:
            if workspace is None:
                workspace = self._client._get_workspace_path_param()
            if not workspace:
                raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
            return await self._post(
                path_template("/apis/intake/v2/workspaces/{workspace}/evaluator-results", workspace=workspace),
                body=await async_maybe_transform(
                    {
                        "data_type": data_type,
                        "name": name,
                        "session_id": session_id,
                        "span_id": span_id,
                        "comment": comment,
                        "string_value": string_value,
                        "value": value,
                    },
                    evaluator_result_create_params.EvaluatorResultCreateParams,
                ),
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=EvaluatorResult,
            )
        except ConflictError:
            if not exist_ok:
                raise
            return await self.retrieve(name = name, workspace = workspace)

    async def retrieve(
        self,
        evaluator_result_id: str,
        *,
        workspace: str | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> EvaluatorResult:
        """
        Get Evaluator Result

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
        if not evaluator_result_id:
            raise ValueError(
                f"Expected a non-empty value for `evaluator_result_id` but received {evaluator_result_id!r}"
            )
        return await self._get(
            path_template(
                "/apis/intake/v2/workspaces/{workspace}/evaluator-results/{evaluator_result_id}",
                workspace=workspace,
                evaluator_result_id=evaluator_result_id,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=EvaluatorResult,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: EvaluatorResultFilterParam | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: EvaluatorResultSortField | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> AsyncPaginator[EvaluatorResult, AsyncDefaultPagination[EvaluatorResult]]:
        """
        List Evaluator Results

        Args:
          filter: Filter evaluator results by span_id, session_id, name, data_type, created_by,
              value range, and created_at range.

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
            path_template("/apis/intake/v2/workspaces/{workspace}/evaluator-results", workspace=workspace),
            page=AsyncDefaultPagination[EvaluatorResult],
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "filter": filter,
                        "page": page,
                        "page_size": page_size,
                        "sort": sort,
                    },
                    evaluator_result_list_params.EvaluatorResultListParams,
                ),
            ),
            model=EvaluatorResult,
        )


class EvaluatorResultsResourceWithRawResponse:
    def __init__(self, evaluator_results: EvaluatorResultsResource) -> None:
        self._evaluator_results = evaluator_results

        self.create = to_raw_response_wrapper(
            evaluator_results.create,
        )
        self.retrieve = to_raw_response_wrapper(
            evaluator_results.retrieve,
        )
        self.list = to_raw_response_wrapper(
            evaluator_results.list,
        )


class AsyncEvaluatorResultsResourceWithRawResponse:
    def __init__(self, evaluator_results: AsyncEvaluatorResultsResource) -> None:
        self._evaluator_results = evaluator_results

        self.create = async_to_raw_response_wrapper(
            evaluator_results.create,
        )
        self.retrieve = async_to_raw_response_wrapper(
            evaluator_results.retrieve,
        )
        self.list = async_to_raw_response_wrapper(
            evaluator_results.list,
        )


class EvaluatorResultsResourceWithStreamingResponse:
    def __init__(self, evaluator_results: EvaluatorResultsResource) -> None:
        self._evaluator_results = evaluator_results

        self.create = to_streamed_response_wrapper(
            evaluator_results.create,
        )
        self.retrieve = to_streamed_response_wrapper(
            evaluator_results.retrieve,
        )
        self.list = to_streamed_response_wrapper(
            evaluator_results.list,
        )


class AsyncEvaluatorResultsResourceWithStreamingResponse:
    def __init__(self, evaluator_results: AsyncEvaluatorResultsResource) -> None:
        self._evaluator_results = evaluator_results

        self.create = async_to_streamed_response_wrapper(
            evaluator_results.create,
        )
        self.retrieve = async_to_streamed_response_wrapper(
            evaluator_results.retrieve,
        )
        self.list = async_to_streamed_response_wrapper(
            evaluator_results.list,
        )
