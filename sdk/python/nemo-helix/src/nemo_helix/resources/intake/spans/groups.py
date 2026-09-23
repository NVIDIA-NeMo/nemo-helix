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
from ....types.intake.spans import SpanGroupSortField, group_list_params
from ....types.intake.spans.span_group import SpanGroup
from ....types.intake.span_filter_param import SpanFilterParam
from ....types.intake.spans.span_group_sort_field import SpanGroupSortField

__all__ = ["GroupsResource", "AsyncGroupsResource"]


class GroupsResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> GroupsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return GroupsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> GroupsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return GroupsResourceWithStreamingResponse(self)

    def list(
        self,
        *,
        workspace: str | None = None,
        by: str,
        filter: SpanFilterParam | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: SpanGroupSortField | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> SyncDefaultPagination[SpanGroup]:
        """List Span Groups

        Args:
          by: Comma-separated span fields to group by, e.g.

        trace_id or session_id,trace_id.

          filter: Filter spans by the same fields as the span list endpoint, then group matching
              spans by the comma-separated fields in the by query parameter.

          page: Page number.

          page_size: Page size.

          sort: Sort groups by size or by start time. Use -started_at for the traces or sessions
              that began most recently, which answers 'what ran lately' in one call instead of
              paging through spans. A group's time is its earliest matching span, so this
              orders by when work started and not by when it was last active.

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
            path_template("/apis/intake/v2/workspaces/{workspace}/spans/groups", workspace=workspace),
            page=SyncDefaultPagination[SpanGroup],
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "by": by,
                        "filter": filter,
                        "page": page,
                        "page_size": page_size,
                        "sort": sort,
                    },
                    group_list_params.GroupListParams,
                ),
            ),
            model=SpanGroup,
        )


class AsyncGroupsResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncGroupsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncGroupsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncGroupsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncGroupsResourceWithStreamingResponse(self)

    def list(
        self,
        *,
        workspace: str | None = None,
        by: str,
        filter: SpanFilterParam | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: SpanGroupSortField | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> AsyncPaginator[SpanGroup, AsyncDefaultPagination[SpanGroup]]:
        """List Span Groups

        Args:
          by: Comma-separated span fields to group by, e.g.

        trace_id or session_id,trace_id.

          filter: Filter spans by the same fields as the span list endpoint, then group matching
              spans by the comma-separated fields in the by query parameter.

          page: Page number.

          page_size: Page size.

          sort: Sort groups by size or by start time. Use -started_at for the traces or sessions
              that began most recently, which answers 'what ran lately' in one call instead of
              paging through spans. A group's time is its earliest matching span, so this
              orders by when work started and not by when it was last active.

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
            path_template("/apis/intake/v2/workspaces/{workspace}/spans/groups", workspace=workspace),
            page=AsyncDefaultPagination[SpanGroup],
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "by": by,
                        "filter": filter,
                        "page": page,
                        "page_size": page_size,
                        "sort": sort,
                    },
                    group_list_params.GroupListParams,
                ),
            ),
            model=SpanGroup,
        )


class GroupsResourceWithRawResponse:
    def __init__(self, groups: GroupsResource) -> None:
        self._groups = groups

        self.list = to_raw_response_wrapper(
            groups.list,
        )


class AsyncGroupsResourceWithRawResponse:
    def __init__(self, groups: AsyncGroupsResource) -> None:
        self._groups = groups

        self.list = async_to_raw_response_wrapper(
            groups.list,
        )


class GroupsResourceWithStreamingResponse:
    def __init__(self, groups: GroupsResource) -> None:
        self._groups = groups

        self.list = to_streamed_response_wrapper(
            groups.list,
        )


class AsyncGroupsResourceWithStreamingResponse:
    def __init__(self, groups: AsyncGroupsResource) -> None:
        self._groups = groups

        self.list = async_to_streamed_response_wrapper(
            groups.list,
        )
