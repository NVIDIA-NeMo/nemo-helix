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
    TraceSortField,
    TraceMetricBucketParam,
    trace_list_params,
    trace_retrieve_params,
    trace_get_metrics_params,
)
from ...types.intake.trace import Trace
from ...types.intake.trace_metrics import TraceMetrics
from ...types.intake.trace_sort_field import TraceSortField
from ...types.intake.trace_filter_param import TraceFilterParam
from ...types.intake.trace_metric_bucket_param import TraceMetricBucketParam

__all__ = ["TracesResource", "AsyncTracesResource"]


class TracesResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> TracesResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return TracesResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> TracesResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return TracesResourceWithStreamingResponse(self)

    def retrieve(
        self,
        id: str,
        *,
        workspace: str | None = None,
        mode: Literal["summary", "preview", "detailed"] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Trace:
        """Get Trace

        Args:
          mode: Response mode.

        summary returns root-span fields without payloads or rollups;
              preview adds token, cost, and span-count rollups plus 300-character input/output
              previews; detailed returns rollups and full payloads.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not id:
            raise ValueError(f"Expected a non-empty value for `id` but received {id!r}")
        return self._get(
            path_template("/apis/intake/v2/workspaces/{workspace}/traces/{id}", workspace=workspace, id=id),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform({"mode": mode}, trace_retrieve_params.TraceRetrieveParams),
            ),
            cast_to=Trace,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: TraceFilterParam | Omit = omit,
        mode: Literal["summary", "preview", "detailed"] | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: TraceSortField | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> SyncDefaultPagination[Trace]:
        """
        List Traces

        Args:
          filter: Filter root-span-backed traces by id, session_id, root status, root span
              started_at, evaluation_name, test_case_name, and agent_name.

          mode: Response mode. summary returns root-span fields without payloads or rollups;
              preview adds token, cost, and span-count rollups plus 300-character input/output
              previews; detailed returns rollups and full payloads.

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
            path_template("/apis/intake/v2/workspaces/{workspace}/traces", workspace=workspace),
            page=SyncDefaultPagination[Trace],
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
                    trace_list_params.TraceListParams,
                ),
            ),
            model=Trace,
        )

    def get_metrics(
        self,
        *,
        workspace: str | None = None,
        bucket: TraceMetricBucketParam | Omit = omit,
        filter: TraceFilterParam | Omit = omit,
        timezone: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> TraceMetrics:
        """Get Trace Metrics

        Args:
          bucket: Time bucket granularity.

        total collapses the filtered range into a single row.

          filter: Filter the traces the metrics are computed over. Accepts the same fields as the
              traces list, so agent_name scopes the rollup to one agent. Without a started_at
              lower bound the rollup covers the last 7 days.

          timezone: IANA timezone the buckets are aligned to, e.g. America/Los_Angeles.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        return self._get(
            path_template("/apis/intake/v2/workspaces/{workspace}/traces/metrics", workspace=workspace),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "bucket": bucket,
                        "filter": filter,
                        "timezone": timezone,
                    },
                    trace_get_metrics_params.TraceGetMetricsParams,
                ),
            ),
            cast_to=TraceMetrics,
        )


class AsyncTracesResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncTracesResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncTracesResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncTracesResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncTracesResourceWithStreamingResponse(self)

    async def retrieve(
        self,
        id: str,
        *,
        workspace: str | None = None,
        mode: Literal["summary", "preview", "detailed"] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Trace:
        """Get Trace

        Args:
          mode: Response mode.

        summary returns root-span fields without payloads or rollups;
              preview adds token, cost, and span-count rollups plus 300-character input/output
              previews; detailed returns rollups and full payloads.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not id:
            raise ValueError(f"Expected a non-empty value for `id` but received {id!r}")
        return await self._get(
            path_template("/apis/intake/v2/workspaces/{workspace}/traces/{id}", workspace=workspace, id=id),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform({"mode": mode}, trace_retrieve_params.TraceRetrieveParams),
            ),
            cast_to=Trace,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: TraceFilterParam | Omit = omit,
        mode: Literal["summary", "preview", "detailed"] | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: TraceSortField | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> AsyncPaginator[Trace, AsyncDefaultPagination[Trace]]:
        """
        List Traces

        Args:
          filter: Filter root-span-backed traces by id, session_id, root status, root span
              started_at, evaluation_name, test_case_name, and agent_name.

          mode: Response mode. summary returns root-span fields without payloads or rollups;
              preview adds token, cost, and span-count rollups plus 300-character input/output
              previews; detailed returns rollups and full payloads.

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
            path_template("/apis/intake/v2/workspaces/{workspace}/traces", workspace=workspace),
            page=AsyncDefaultPagination[Trace],
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
                    trace_list_params.TraceListParams,
                ),
            ),
            model=Trace,
        )

    async def get_metrics(
        self,
        *,
        workspace: str | None = None,
        bucket: TraceMetricBucketParam | Omit = omit,
        filter: TraceFilterParam | Omit = omit,
        timezone: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> TraceMetrics:
        """Get Trace Metrics

        Args:
          bucket: Time bucket granularity.

        total collapses the filtered range into a single row.

          filter: Filter the traces the metrics are computed over. Accepts the same fields as the
              traces list, so agent_name scopes the rollup to one agent. Without a started_at
              lower bound the rollup covers the last 7 days.

          timezone: IANA timezone the buckets are aligned to, e.g. America/Los_Angeles.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        return await self._get(
            path_template("/apis/intake/v2/workspaces/{workspace}/traces/metrics", workspace=workspace),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {
                        "bucket": bucket,
                        "filter": filter,
                        "timezone": timezone,
                    },
                    trace_get_metrics_params.TraceGetMetricsParams,
                ),
            ),
            cast_to=TraceMetrics,
        )


class TracesResourceWithRawResponse:
    def __init__(self, traces: TracesResource) -> None:
        self._traces = traces

        self.retrieve = to_raw_response_wrapper(
            traces.retrieve,
        )
        self.list = to_raw_response_wrapper(
            traces.list,
        )
        self.get_metrics = to_raw_response_wrapper(
            traces.get_metrics,
        )


class AsyncTracesResourceWithRawResponse:
    def __init__(self, traces: AsyncTracesResource) -> None:
        self._traces = traces

        self.retrieve = async_to_raw_response_wrapper(
            traces.retrieve,
        )
        self.list = async_to_raw_response_wrapper(
            traces.list,
        )
        self.get_metrics = async_to_raw_response_wrapper(
            traces.get_metrics,
        )


class TracesResourceWithStreamingResponse:
    def __init__(self, traces: TracesResource) -> None:
        self._traces = traces

        self.retrieve = to_streamed_response_wrapper(
            traces.retrieve,
        )
        self.list = to_streamed_response_wrapper(
            traces.list,
        )
        self.get_metrics = to_streamed_response_wrapper(
            traces.get_metrics,
        )


class AsyncTracesResourceWithStreamingResponse:
    def __init__(self, traces: AsyncTracesResource) -> None:
        self._traces = traces

        self.retrieve = async_to_streamed_response_wrapper(
            traces.retrieve,
        )
        self.list = async_to_streamed_response_wrapper(
            traces.list,
        )
        self.get_metrics = async_to_streamed_response_wrapper(
            traces.get_metrics,
        )
