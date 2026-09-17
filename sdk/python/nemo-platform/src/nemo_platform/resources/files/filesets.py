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
from ...types.files import (
    FilesetPurpose,
    fileset_list_params,
    fileset_create_params,
    fileset_update_params,
)
from ..._base_client import AsyncPaginator, make_request_options
from ...types.files.fileset import Fileset
from ...types.files.fileset_purpose import FilesetPurpose
from ...types.shared.generic_sort_field import GenericSortField
from ...types.files.fileset_filter_param import FilesetFilterParam
from ...types.shared_params.fileset_metadata import FilesetMetadata
from ..._exceptions import ConflictError

__all__ = ["FilesetsResource", "AsyncFilesetsResource"]


class FilesetsResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> FilesetsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return FilesetsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> FilesetsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return FilesetsResourceWithStreamingResponse(self)

    def create(
        self,
        *,
        workspace: str | None = None,
        name: str,
        cache: bool | Omit = omit,
        custom_fields: Dict[str, object] | Omit = omit,
        description: str | Omit = omit,
        metadata: FilesetMetadata | Omit = omit,
        project: str | Omit = omit,
        purpose: FilesetPurpose | Omit = omit,
        storage: fileset_create_params.Storage | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Fileset:
        """
        Create a new fileset.

        If no storage configuration is provided, the default storage backend will be
        used.

        Args:
          name: The name of the fileset. Name must start with a lowercase letter, be 2-63
              characters, and use lowercase letters, digits, hyphens, and dots (no consecutive
              hyphens, cannot end with a hyphen).

          cache: Cache all files after creation. Only applies to external storage.

          custom_fields: Custom fields for the fileset.

          description: The description of the fileset.

          metadata: Tagged metadata container - the key indicates the type.

              Example: metadata = FilesetMetadata( dataset=DatasetMetadataContent(
              schema={"columns": ["id", "name"]}, ) )

          project: The name of the project associated with this fileset.

          purpose: The purpose of the fileset.

          storage: The storage configuration for the fileset. If not provided, uses default
              storage.


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
                path_template("/apis/files/v2/workspaces/{workspace}/filesets", workspace=workspace),
                body=maybe_transform(
                    {
                        "name": name,
                        "cache": cache,
                        "custom_fields": custom_fields,
                        "description": description,
                        "metadata": metadata,
                        "project": project,
                        "purpose": purpose,
                        "storage": storage,
                    },
                    fileset_create_params.FilesetCreateParams,
                ),
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=Fileset,
            )
        except ConflictError:
            if not exist_ok:
                raise
            return self.retrieve(name = name, workspace = workspace)

    def retrieve(
        self,
        name: str,
        *,
        workspace: str | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Fileset:
        """
        Get Fileset by Workspace and Name.

        Returns the details of a specific fileset identified by its workspace and name.

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
        return self._get(
            path_template("/apis/files/v2/workspaces/{workspace}/filesets/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Fileset,
        )

    def update(
        self,
        name: str,
        *,
        workspace: str | None = None,
        custom_fields: Dict[str, object] | Omit = omit,
        description: str | Omit = omit,
        metadata: FilesetMetadata | Omit = omit,
        project: str | Omit = omit,
        purpose: FilesetPurpose | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Fileset:
        """
        Update Fileset Metadata.

        Args:
          custom_fields: Custom fields for the fileset.

          description: The description of the fileset.

          metadata: Tagged metadata container - the key indicates the type.

              Example: metadata = FilesetMetadata( dataset=DatasetMetadataContent(
              schema={"columns": ["id", "name"]}, ) )

          project: The name of the project associated with this fileset.

          purpose: The purpose of the fileset.

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
        return self._patch(
            path_template("/apis/files/v2/workspaces/{workspace}/filesets/{name}", workspace=workspace, name=name),
            body=maybe_transform(
                {
                    "custom_fields": custom_fields,
                    "description": description,
                    "metadata": metadata,
                    "project": project,
                    "purpose": purpose,
                },
                fileset_update_params.FilesetUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Fileset,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: FilesetFilterParam | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: GenericSortField | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> SyncDefaultPagination[Fileset]:
        """
        List Filesets endpoint with filtering and pagination.

        Supports filtering by name, description, purpose, storage_type, created_at, and
        updated_at via query parameters. Returns paginated results with sorting options.

        Args:
          filter: Filter filesets by name, description, purpose, storage_type, created_at, and
              updated_at.

          page: Page number.

          page_size: Page size.

          sort: The field to sort by. To sort in decreasing order, use `-` in front of the field
              name.

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
            path_template("/apis/files/v2/workspaces/{workspace}/filesets", workspace=workspace),
            page=SyncDefaultPagination[Fileset],
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
                    fileset_list_params.FilesetListParams,
                ),
            ),
            model=Fileset,
        )

    def delete(
        self,
        name: str,
        *,
        workspace: str | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Fileset:
        """Delete Fileset.

        Permanently deletes an unreferenced fileset from the platform.

        Referencing model
        or adapter entities must be relinked or deleted first. For local storage
        backends, this also deletes the underlying files.

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
        return self._delete(
            path_template("/apis/files/v2/workspaces/{workspace}/filesets/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Fileset,
        )

    def refresh(
        self,
        name: str,
        *,
        workspace: str | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Fileset:
        """
        Re-resolve a fileset's tracked revision against its source.

        A fileset created from a mutable ref is pinned to an immutable id so its
        contents cannot shift under a deployment. This re-resolves that same ref and
        repoints the fileset at whatever it names now. Everything else about the storage
        config, including the repository and directory, is left alone.

        A running deployment keeps serving the revision it staged. It moves to this one
        when the runner next stages the fileset, and records the revision it staged.

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
        return self._post(
            path_template(
                "/apis/files/v2/workspaces/{workspace}/filesets/{name}/refresh", workspace=workspace, name=name
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Fileset,
        )


class AsyncFilesetsResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncFilesetsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncFilesetsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncFilesetsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncFilesetsResourceWithStreamingResponse(self)

    async def create(
        self,
        *,
        workspace: str | None = None,
        name: str,
        cache: bool | Omit = omit,
        custom_fields: Dict[str, object] | Omit = omit,
        description: str | Omit = omit,
        metadata: FilesetMetadata | Omit = omit,
        project: str | Omit = omit,
        purpose: FilesetPurpose | Omit = omit,
        storage: fileset_create_params.Storage | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Fileset:
        """
        Create a new fileset.

        If no storage configuration is provided, the default storage backend will be
        used.

        Args:
          name: The name of the fileset. Name must start with a lowercase letter, be 2-63
              characters, and use lowercase letters, digits, hyphens, and dots (no consecutive
              hyphens, cannot end with a hyphen).

          cache: Cache all files after creation. Only applies to external storage.

          custom_fields: Custom fields for the fileset.

          description: The description of the fileset.

          metadata: Tagged metadata container - the key indicates the type.

              Example: metadata = FilesetMetadata( dataset=DatasetMetadataContent(
              schema={"columns": ["id", "name"]}, ) )

          project: The name of the project associated with this fileset.

          purpose: The purpose of the fileset.

          storage: The storage configuration for the fileset. If not provided, uses default
              storage.


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
                path_template("/apis/files/v2/workspaces/{workspace}/filesets", workspace=workspace),
                body=await async_maybe_transform(
                    {
                        "name": name,
                        "cache": cache,
                        "custom_fields": custom_fields,
                        "description": description,
                        "metadata": metadata,
                        "project": project,
                        "purpose": purpose,
                        "storage": storage,
                    },
                    fileset_create_params.FilesetCreateParams,
                ),
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=Fileset,
            )
        except ConflictError:
            if not exist_ok:
                raise
            return await self.retrieve(name = name, workspace = workspace)

    async def retrieve(
        self,
        name: str,
        *,
        workspace: str | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Fileset:
        """
        Get Fileset by Workspace and Name.

        Returns the details of a specific fileset identified by its workspace and name.

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
        return await self._get(
            path_template("/apis/files/v2/workspaces/{workspace}/filesets/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Fileset,
        )

    async def update(
        self,
        name: str,
        *,
        workspace: str | None = None,
        custom_fields: Dict[str, object] | Omit = omit,
        description: str | Omit = omit,
        metadata: FilesetMetadata | Omit = omit,
        project: str | Omit = omit,
        purpose: FilesetPurpose | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Fileset:
        """
        Update Fileset Metadata.

        Args:
          custom_fields: Custom fields for the fileset.

          description: The description of the fileset.

          metadata: Tagged metadata container - the key indicates the type.

              Example: metadata = FilesetMetadata( dataset=DatasetMetadataContent(
              schema={"columns": ["id", "name"]}, ) )

          project: The name of the project associated with this fileset.

          purpose: The purpose of the fileset.

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
        return await self._patch(
            path_template("/apis/files/v2/workspaces/{workspace}/filesets/{name}", workspace=workspace, name=name),
            body=await async_maybe_transform(
                {
                    "custom_fields": custom_fields,
                    "description": description,
                    "metadata": metadata,
                    "project": project,
                    "purpose": purpose,
                },
                fileset_update_params.FilesetUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Fileset,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: FilesetFilterParam | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: GenericSortField | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> AsyncPaginator[Fileset, AsyncDefaultPagination[Fileset]]:
        """
        List Filesets endpoint with filtering and pagination.

        Supports filtering by name, description, purpose, storage_type, created_at, and
        updated_at via query parameters. Returns paginated results with sorting options.

        Args:
          filter: Filter filesets by name, description, purpose, storage_type, created_at, and
              updated_at.

          page: Page number.

          page_size: Page size.

          sort: The field to sort by. To sort in decreasing order, use `-` in front of the field
              name.

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
            path_template("/apis/files/v2/workspaces/{workspace}/filesets", workspace=workspace),
            page=AsyncDefaultPagination[Fileset],
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
                    fileset_list_params.FilesetListParams,
                ),
            ),
            model=Fileset,
        )

    async def delete(
        self,
        name: str,
        *,
        workspace: str | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Fileset:
        """Delete Fileset.

        Permanently deletes an unreferenced fileset from the platform.

        Referencing model
        or adapter entities must be relinked or deleted first. For local storage
        backends, this also deletes the underlying files.

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
        return await self._delete(
            path_template("/apis/files/v2/workspaces/{workspace}/filesets/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Fileset,
        )

    async def refresh(
        self,
        name: str,
        *,
        workspace: str | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Fileset:
        """
        Re-resolve a fileset's tracked revision against its source.

        A fileset created from a mutable ref is pinned to an immutable id so its
        contents cannot shift under a deployment. This re-resolves that same ref and
        repoints the fileset at whatever it names now. Everything else about the storage
        config, including the repository and directory, is left alone.

        A running deployment keeps serving the revision it staged. It moves to this one
        when the runner next stages the fileset, and records the revision it staged.

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
        return await self._post(
            path_template(
                "/apis/files/v2/workspaces/{workspace}/filesets/{name}/refresh", workspace=workspace, name=name
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Fileset,
        )


class FilesetsResourceWithRawResponse:
    def __init__(self, filesets: FilesetsResource) -> None:
        self._filesets = filesets

        self.create = to_raw_response_wrapper(
            filesets.create,
        )
        self.retrieve = to_raw_response_wrapper(
            filesets.retrieve,
        )
        self.update = to_raw_response_wrapper(
            filesets.update,
        )
        self.list = to_raw_response_wrapper(
            filesets.list,
        )
        self.delete = to_raw_response_wrapper(
            filesets.delete,
        )
        self.refresh = to_raw_response_wrapper(
            filesets.refresh,
        )


class AsyncFilesetsResourceWithRawResponse:
    def __init__(self, filesets: AsyncFilesetsResource) -> None:
        self._filesets = filesets

        self.create = async_to_raw_response_wrapper(
            filesets.create,
        )
        self.retrieve = async_to_raw_response_wrapper(
            filesets.retrieve,
        )
        self.update = async_to_raw_response_wrapper(
            filesets.update,
        )
        self.list = async_to_raw_response_wrapper(
            filesets.list,
        )
        self.delete = async_to_raw_response_wrapper(
            filesets.delete,
        )
        self.refresh = async_to_raw_response_wrapper(
            filesets.refresh,
        )


class FilesetsResourceWithStreamingResponse:
    def __init__(self, filesets: FilesetsResource) -> None:
        self._filesets = filesets

        self.create = to_streamed_response_wrapper(
            filesets.create,
        )
        self.retrieve = to_streamed_response_wrapper(
            filesets.retrieve,
        )
        self.update = to_streamed_response_wrapper(
            filesets.update,
        )
        self.list = to_streamed_response_wrapper(
            filesets.list,
        )
        self.delete = to_streamed_response_wrapper(
            filesets.delete,
        )
        self.refresh = to_streamed_response_wrapper(
            filesets.refresh,
        )


class AsyncFilesetsResourceWithStreamingResponse:
    def __init__(self, filesets: AsyncFilesetsResource) -> None:
        self._filesets = filesets

        self.create = async_to_streamed_response_wrapper(
            filesets.create,
        )
        self.retrieve = async_to_streamed_response_wrapper(
            filesets.retrieve,
        )
        self.update = async_to_streamed_response_wrapper(
            filesets.update,
        )
        self.list = async_to_streamed_response_wrapper(
            filesets.list,
        )
        self.delete = async_to_streamed_response_wrapper(
            filesets.delete,
        )
        self.refresh = async_to_streamed_response_wrapper(
            filesets.refresh,
        )
