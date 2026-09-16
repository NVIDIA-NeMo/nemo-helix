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
from ....types.files.otlp import log_query_params, log_create_params
from ....types.shared.platform_job_log_page import PlatformJobLogPage
from ....types.files.otlp.otel_export_logs_service_response import OtelExportLogsServiceResponse

__all__ = ["LogsResource", "AsyncLogsResource"]


class LogsResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> LogsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return LogsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> LogsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return LogsResourceWithStreamingResponse(self)

    def create(
        self,
        name: str,
        *,
        workspace: str | None = None,
        artifact_base_path: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> OtelExportLogsServiceResponse:
        """
        Upload OTLP logs to a specified fileset in JSON or Protobuf format.

        Supports both application/json and application/x-protobuf content types.

        Args:
          artifact_base_path: Folder inside the fileset to nest logs under

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
        return self._post(
            path_template(
                "/apis/files/v2/workspaces/{workspace}/filesets/{name}/otlp/v1/logs", workspace=workspace, name=name
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform({"artifact_base_path": artifact_base_path}, log_create_params.LogCreateParams),
            ),
            cast_to=OtelExportLogsServiceResponse,
        )

    def query(
        self,
        name: str,
        *,
        workspace: str | None = None,
        artifact_base_path: str | Omit = omit,
        filters: Dict[str, str] | Omit = omit,
        limit: int | Omit = omit,
        page_cursor: str | Omit = omit,
        tail: int | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> PlatformJobLogPage:
        """
        Query logs from parquet files in a fileset.

        This is an internal endpoint that runs DuckDB queries with direct storage
        access.

        Args:
          artifact_base_path: Folder inside the fileset the logs were nested under (must match the value used
              on write)

          filters: Key-value filters to apply to the query

          limit: Maximum number of results to return

          page_cursor: Cursor for pagination

          tail: Number of newest log lines to return

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
        return self._post(
            path_template(
                "/apis/files/v2/workspaces/{workspace}/filesets/{name}/otlp/v1/logs/query",
                workspace=workspace,
                name=name,
            ),
            body=maybe_transform(
                {
                    "artifact_base_path": artifact_base_path,
                    "filters": filters,
                    "limit": limit,
                    "page_cursor": page_cursor,
                    "tail": tail,
                },
                log_query_params.LogQueryParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=PlatformJobLogPage,
        )


class AsyncLogsResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncLogsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncLogsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncLogsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncLogsResourceWithStreamingResponse(self)

    async def create(
        self,
        name: str,
        *,
        workspace: str | None = None,
        artifact_base_path: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> OtelExportLogsServiceResponse:
        """
        Upload OTLP logs to a specified fileset in JSON or Protobuf format.

        Supports both application/json and application/x-protobuf content types.

        Args:
          artifact_base_path: Folder inside the fileset to nest logs under

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
        return await self._post(
            path_template(
                "/apis/files/v2/workspaces/{workspace}/filesets/{name}/otlp/v1/logs", workspace=workspace, name=name
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {"artifact_base_path": artifact_base_path}, log_create_params.LogCreateParams
                ),
            ),
            cast_to=OtelExportLogsServiceResponse,
        )

    async def query(
        self,
        name: str,
        *,
        workspace: str | None = None,
        artifact_base_path: str | Omit = omit,
        filters: Dict[str, str] | Omit = omit,
        limit: int | Omit = omit,
        page_cursor: str | Omit = omit,
        tail: int | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> PlatformJobLogPage:
        """
        Query logs from parquet files in a fileset.

        This is an internal endpoint that runs DuckDB queries with direct storage
        access.

        Args:
          artifact_base_path: Folder inside the fileset the logs were nested under (must match the value used
              on write)

          filters: Key-value filters to apply to the query

          limit: Maximum number of results to return

          page_cursor: Cursor for pagination

          tail: Number of newest log lines to return

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
        return await self._post(
            path_template(
                "/apis/files/v2/workspaces/{workspace}/filesets/{name}/otlp/v1/logs/query",
                workspace=workspace,
                name=name,
            ),
            body=await async_maybe_transform(
                {
                    "artifact_base_path": artifact_base_path,
                    "filters": filters,
                    "limit": limit,
                    "page_cursor": page_cursor,
                    "tail": tail,
                },
                log_query_params.LogQueryParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=PlatformJobLogPage,
        )


class LogsResourceWithRawResponse:
    def __init__(self, logs: LogsResource) -> None:
        self._logs = logs

        self.create = to_raw_response_wrapper(
            logs.create,
        )
        self.query = to_raw_response_wrapper(
            logs.query,
        )


class AsyncLogsResourceWithRawResponse:
    def __init__(self, logs: AsyncLogsResource) -> None:
        self._logs = logs

        self.create = async_to_raw_response_wrapper(
            logs.create,
        )
        self.query = async_to_raw_response_wrapper(
            logs.query,
        )


class LogsResourceWithStreamingResponse:
    def __init__(self, logs: LogsResource) -> None:
        self._logs = logs

        self.create = to_streamed_response_wrapper(
            logs.create,
        )
        self.query = to_streamed_response_wrapper(
            logs.query,
        )


class AsyncLogsResourceWithStreamingResponse:
    def __init__(self, logs: AsyncLogsResource) -> None:
        self._logs = logs

        self.create = async_to_streamed_response_wrapper(
            logs.create,
        )
        self.query = async_to_streamed_response_wrapper(
            logs.query,
        )
