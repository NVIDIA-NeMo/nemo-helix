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

from ..._types import Body, Omit, Query, Headers, NotGiven, omit, not_given
from ..._utils import path_template, maybe_transform
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
from ...types.evaluations import session_list_params
from ...types.evaluations.evaluation_session_response import EvaluationSessionResponse
from ...types.evaluations.evaluation_session_filter_param import EvaluationSessionFilterParam

__all__ = ["SessionsResource", "AsyncSessionsResource"]


class SessionsResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> SessionsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return SessionsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> SessionsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return SessionsResourceWithStreamingResponse(self)

    def list(
        self,
        name: str,
        *,
        workspace: str | None = None,
        filter: EvaluationSessionFilterParam | Omit = omit,
        mode: Literal["summary", "preview", "detailed"] | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> SyncDefaultPagination[EvaluationSessionResponse]:
        """
        List Evaluation Sessions

        Args:
          filter: Filter sessions by test_case_name and status.

          mode: Response mode. summary omits root-span input and output; preview includes both
              truncated to 300 characters; detailed returns full root-span payloads.

          page: Page number.

          page_size: Page size.

          sort: Comma-separated list of fields to sort by, applied in order (the first field
              dominates); prefix a field with '-' for descending — e.g.
              '-cost_total_usd,latency_ms'. Fields: test_case_name, started_at, ended_at,
              latency_ms, status, cost_total_usd, tokens. When omitted, sessions are ordered
              by started_at ascending.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        return self._get_api_list(
            path_template(
                "/apis/intake/v2/workspaces/{workspace}/evaluations/{name}/sessions", workspace=workspace, name=name
            ),
            page=SyncDefaultPagination[EvaluationSessionResponse],
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
                    session_list_params.SessionListParams,
                ),
            ),
            model=EvaluationSessionResponse,
        )


class AsyncSessionsResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncSessionsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncSessionsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncSessionsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncSessionsResourceWithStreamingResponse(self)

    def list(
        self,
        name: str,
        *,
        workspace: str | None = None,
        filter: EvaluationSessionFilterParam | Omit = omit,
        mode: Literal["summary", "preview", "detailed"] | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> AsyncPaginator[EvaluationSessionResponse, AsyncDefaultPagination[EvaluationSessionResponse]]:
        """
        List Evaluation Sessions

        Args:
          filter: Filter sessions by test_case_name and status.

          mode: Response mode. summary omits root-span input and output; preview includes both
              truncated to 300 characters; detailed returns full root-span payloads.

          page: Page number.

          page_size: Page size.

          sort: Comma-separated list of fields to sort by, applied in order (the first field
              dominates); prefix a field with '-' for descending — e.g.
              '-cost_total_usd,latency_ms'. Fields: test_case_name, started_at, ended_at,
              latency_ms, status, cost_total_usd, tokens. When omitted, sessions are ordered
              by started_at ascending.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        return self._get_api_list(
            path_template(
                "/apis/intake/v2/workspaces/{workspace}/evaluations/{name}/sessions", workspace=workspace, name=name
            ),
            page=AsyncDefaultPagination[EvaluationSessionResponse],
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
                    session_list_params.SessionListParams,
                ),
            ),
            model=EvaluationSessionResponse,
        )


class SessionsResourceWithRawResponse:
    def __init__(self, sessions: SessionsResource) -> None:
        self._sessions = sessions

        self.list = to_raw_response_wrapper(
            sessions.list,
        )


class AsyncSessionsResourceWithRawResponse:
    def __init__(self, sessions: AsyncSessionsResource) -> None:
        self._sessions = sessions

        self.list = async_to_raw_response_wrapper(
            sessions.list,
        )


class SessionsResourceWithStreamingResponse:
    def __init__(self, sessions: SessionsResource) -> None:
        self._sessions = sessions

        self.list = to_streamed_response_wrapper(
            sessions.list,
        )


class AsyncSessionsResourceWithStreamingResponse:
    def __init__(self, sessions: AsyncSessionsResource) -> None:
        self._sessions = sessions

        self.list = async_to_streamed_response_wrapper(
            sessions.list,
        )
