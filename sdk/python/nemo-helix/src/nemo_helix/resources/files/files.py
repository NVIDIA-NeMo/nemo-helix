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

import os

import httpx

from ..._files import read_file_content, async_read_file_content
from ..._types import (
    Body,
    Omit,
    Query,
    Headers,
    NotGiven,
    BinaryTypes,
    FileContent,
    AsyncBinaryTypes,
    omit,
    not_given,
)
from ..._utils import path_template, maybe_transform, async_maybe_transform
from .filesets import (
    FilesetsResource,
    AsyncFilesetsResource,
    FilesetsResourceWithRawResponse,
    AsyncFilesetsResourceWithRawResponse,
    FilesetsResourceWithStreamingResponse,
    AsyncFilesetsResourceWithStreamingResponse,
)
from ..._compat import cached_property
from .otlp.otlp import (
    OtlpResource,
    AsyncOtlpResource,
    OtlpResourceWithRawResponse,
    AsyncOtlpResourceWithRawResponse,
    OtlpResourceWithStreamingResponse,
    AsyncOtlpResourceWithStreamingResponse,
)
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._response import (
    BinaryAPIResponse,
    AsyncBinaryAPIResponse,
    StreamedBinaryAPIResponse,
    AsyncStreamedBinaryAPIResponse,
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    to_custom_raw_response_wrapper,
    async_to_streamed_response_wrapper,
    to_custom_streamed_response_wrapper,
    async_to_custom_raw_response_wrapper,
    async_to_custom_streamed_response_wrapper,
)
from ...types.files import file_list_files_params
from ..._base_client import make_request_options
from ...types.files.fileset_file import FilesetFile
from ...types.files.list_fileset_files_response import ListFilesetFilesResponse

__all__ = ["FilesResource", "AsyncFilesResource"]


class FilesResource(SyncAPIResource):
    @cached_property
    def filesets(self) -> FilesetsResource:
        return FilesetsResource(self._client)

    @cached_property
    def otlp(self) -> OtlpResource:
        return OtlpResource(self._client)

    @cached_property
    def with_raw_response(self) -> FilesResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return FilesResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> FilesResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return FilesResourceWithStreamingResponse(self)

    def _delete_file(
        self,
        path: str,
        *,
        workspace: str | None = None,
        name: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> FilesetFile:
        """
        Delete a specific file from a fileset.

        Permanently deletes the file from the storage backend. Returns metadata about
        the deleted file.

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
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        if not path:
            raise ValueError(f"Expected a non-empty value for `path` but received {path!r}")
        return self._delete(
            path_template(
                "/apis/files/v2/workspaces/{workspace}/filesets/{name}/-/{path}",
                workspace=workspace,
                name=name,
                path=path,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=FilesetFile,
        )

    def _download_file(
        self,
        path: str,
        *,
        workspace: str | None = None,
        name: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> BinaryAPIResponse:
        """
        Download file content from a fileset.

        Supports HTTP Range requests for partial content retrieval (status 206). Returns
        the full file content (status 200) if no Range header is provided. For external
        resources (HuggingFace, NGC), content is cached locally on first access.

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
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        if not path:
            raise ValueError(f"Expected a non-empty value for `path` but received {path!r}")
        extra_headers = {"Accept": "application/octet-stream", **(extra_headers or {})}
        return self._get(
            path_template(
                "/apis/files/v2/workspaces/{workspace}/filesets/{name}/-/{path}",
                workspace=workspace,
                name=name,
                path=path,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=BinaryAPIResponse,
        )

    def _list_files(
        self,
        name: str,
        *,
        workspace: str | None = None,
        include_cache_status: bool | Omit = omit,
        path: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ListFilesetFilesResponse:
        """List Files in Fileset.

        Returns a list of files stored in the specified fileset.

        Optionally filter by
        path prefix to list files under a specific directory.

        Each file includes a cache_status field:

        - "not_cacheable": File is on default storage, caching not applicable
        - "cached": File exists in cache storage
        - "caching": File is currently being downloaded and cached
        - "not_cached": File not in cache, will be cached on next download
        - null: External storage, but cache status not checked (use
          include_cache_status=true)

        Args:
          include_cache_status: Check and return cache status for each file. When false, storage files return
              null for cache_status.

          path: Filter files by path prefix

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
        return self._get(
            path_template(
                "/apis/files/v2/workspaces/{workspace}/filesets/{name}/files", workspace=workspace, name=name
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "include_cache_status": include_cache_status,
                        "path": path,
                    },
                    file_list_files_params.FileListFilesParams,
                ),
            ),
            cast_to=ListFilesetFilesResponse,
        )

    def _upload_file(
        self,
        path: str,
        body: FileContent | BinaryTypes,
        *,
        workspace: str | None = None,
        name: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> FilesetFile:
        """
        Upload file content to a fileset.

        Args:
          body: Raw binary file content

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
        if not path:
            raise ValueError(f"Expected a non-empty value for `path` but received {path!r}")
        extra_headers = {"Content-Type": "application/octet-stream", **(extra_headers or {})}
        return self._put(
            path_template(
                "/apis/files/v2/workspaces/{workspace}/filesets/{name}/-/{path}",
                workspace=workspace,
                name=name,
                path=path,
            ),
            content=read_file_content(body) if isinstance(body, os.PathLike) else body,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=FilesetFile,
        )


class AsyncFilesResource(AsyncAPIResource):
    @cached_property
    def filesets(self) -> AsyncFilesetsResource:
        return AsyncFilesetsResource(self._client)

    @cached_property
    def otlp(self) -> AsyncOtlpResource:
        return AsyncOtlpResource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncFilesResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncFilesResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncFilesResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncFilesResourceWithStreamingResponse(self)

    async def _delete_file(
        self,
        path: str,
        *,
        workspace: str | None = None,
        name: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> FilesetFile:
        """
        Delete a specific file from a fileset.

        Permanently deletes the file from the storage backend. Returns metadata about
        the deleted file.

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
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        if not path:
            raise ValueError(f"Expected a non-empty value for `path` but received {path!r}")
        return await self._delete(
            path_template(
                "/apis/files/v2/workspaces/{workspace}/filesets/{name}/-/{path}",
                workspace=workspace,
                name=name,
                path=path,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=FilesetFile,
        )

    async def _download_file(
        self,
        path: str,
        *,
        workspace: str | None = None,
        name: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> AsyncBinaryAPIResponse:
        """
        Download file content from a fileset.

        Supports HTTP Range requests for partial content retrieval (status 206). Returns
        the full file content (status 200) if no Range header is provided. For external
        resources (HuggingFace, NGC), content is cached locally on first access.

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
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        if not path:
            raise ValueError(f"Expected a non-empty value for `path` but received {path!r}")
        extra_headers = {"Accept": "application/octet-stream", **(extra_headers or {})}
        return await self._get(
            path_template(
                "/apis/files/v2/workspaces/{workspace}/filesets/{name}/-/{path}",
                workspace=workspace,
                name=name,
                path=path,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=AsyncBinaryAPIResponse,
        )

    async def _list_files(
        self,
        name: str,
        *,
        workspace: str | None = None,
        include_cache_status: bool | Omit = omit,
        path: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ListFilesetFilesResponse:
        """List Files in Fileset.

        Returns a list of files stored in the specified fileset.

        Optionally filter by
        path prefix to list files under a specific directory.

        Each file includes a cache_status field:

        - "not_cacheable": File is on default storage, caching not applicable
        - "cached": File exists in cache storage
        - "caching": File is currently being downloaded and cached
        - "not_cached": File not in cache, will be cached on next download
        - null: External storage, but cache status not checked (use
          include_cache_status=true)

        Args:
          include_cache_status: Check and return cache status for each file. When false, storage files return
              null for cache_status.

          path: Filter files by path prefix

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
        return await self._get(
            path_template(
                "/apis/files/v2/workspaces/{workspace}/filesets/{name}/files", workspace=workspace, name=name
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {
                        "include_cache_status": include_cache_status,
                        "path": path,
                    },
                    file_list_files_params.FileListFilesParams,
                ),
            ),
            cast_to=ListFilesetFilesResponse,
        )

    async def _upload_file(
        self,
        path: str,
        body: FileContent | AsyncBinaryTypes,
        *,
        workspace: str | None = None,
        name: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> FilesetFile:
        """
        Upload file content to a fileset.

        Args:
          body: Raw binary file content

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
        if not path:
            raise ValueError(f"Expected a non-empty value for `path` but received {path!r}")
        extra_headers = {"Content-Type": "application/octet-stream", **(extra_headers or {})}
        return await self._put(
            path_template(
                "/apis/files/v2/workspaces/{workspace}/filesets/{name}/-/{path}",
                workspace=workspace,
                name=name,
                path=path,
            ),
            content=await async_read_file_content(body) if isinstance(body, os.PathLike) else body,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=FilesetFile,
        )


class FilesResourceWithRawResponse:
    def __init__(self, files: FilesResource) -> None:
        self._files = files

        self._delete_file = to_raw_response_wrapper(
            files._delete_file,
        )
        self._download_file = to_custom_raw_response_wrapper(
            files._download_file,
            BinaryAPIResponse,
        )
        self._list_files = to_raw_response_wrapper(
            files._list_files,
        )
        self._upload_file = to_raw_response_wrapper(
            files._upload_file,
        )

    @cached_property
    def filesets(self) -> FilesetsResourceWithRawResponse:
        return FilesetsResourceWithRawResponse(self._files.filesets)

    @cached_property
    def otlp(self) -> OtlpResourceWithRawResponse:
        return OtlpResourceWithRawResponse(self._files.otlp)


class AsyncFilesResourceWithRawResponse:
    def __init__(self, files: AsyncFilesResource) -> None:
        self._files = files

        self._delete_file = async_to_raw_response_wrapper(
            files._delete_file,
        )
        self._download_file = async_to_custom_raw_response_wrapper(
            files._download_file,
            AsyncBinaryAPIResponse,
        )
        self._list_files = async_to_raw_response_wrapper(
            files._list_files,
        )
        self._upload_file = async_to_raw_response_wrapper(
            files._upload_file,
        )

    @cached_property
    def filesets(self) -> AsyncFilesetsResourceWithRawResponse:
        return AsyncFilesetsResourceWithRawResponse(self._files.filesets)

    @cached_property
    def otlp(self) -> AsyncOtlpResourceWithRawResponse:
        return AsyncOtlpResourceWithRawResponse(self._files.otlp)


class FilesResourceWithStreamingResponse:
    def __init__(self, files: FilesResource) -> None:
        self._files = files

        self._delete_file = to_streamed_response_wrapper(
            files._delete_file,
        )
        self._download_file = to_custom_streamed_response_wrapper(
            files._download_file,
            StreamedBinaryAPIResponse,
        )
        self._list_files = to_streamed_response_wrapper(
            files._list_files,
        )
        self._upload_file = to_streamed_response_wrapper(
            files._upload_file,
        )

    @cached_property
    def filesets(self) -> FilesetsResourceWithStreamingResponse:
        return FilesetsResourceWithStreamingResponse(self._files.filesets)

    @cached_property
    def otlp(self) -> OtlpResourceWithStreamingResponse:
        return OtlpResourceWithStreamingResponse(self._files.otlp)


class AsyncFilesResourceWithStreamingResponse:
    def __init__(self, files: AsyncFilesResource) -> None:
        self._files = files

        self._delete_file = async_to_streamed_response_wrapper(
            files._delete_file,
        )
        self._download_file = async_to_custom_streamed_response_wrapper(
            files._download_file,
            AsyncStreamedBinaryAPIResponse,
        )
        self._list_files = async_to_streamed_response_wrapper(
            files._list_files,
        )
        self._upload_file = async_to_streamed_response_wrapper(
            files._upload_file,
        )

    @cached_property
    def filesets(self) -> AsyncFilesetsResourceWithStreamingResponse:
        return AsyncFilesetsResourceWithStreamingResponse(self._files.filesets)

    @cached_property
    def otlp(self) -> AsyncOtlpResourceWithStreamingResponse:
        return AsyncOtlpResourceWithStreamingResponse(self._files.otlp)
