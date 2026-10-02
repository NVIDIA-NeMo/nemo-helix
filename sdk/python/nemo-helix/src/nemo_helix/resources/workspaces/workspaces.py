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

from .members import (
    MembersResource,
    AsyncMembersResource,
    MembersResourceWithRawResponse,
    AsyncMembersResourceWithRawResponse,
    MembersResourceWithStreamingResponse,
    AsyncMembersResourceWithStreamingResponse,
)
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
from ...types.workspaces import workspace_list_params, workspace_create_params, workspace_update_params
from ...types.workspaces.workspace import Workspace
from ...types.shared.delete_response import DeleteResponse
from ...types.shared.generic_sort_field import GenericSortField
from ..._exceptions import ConflictError

__all__ = ["WorkspacesResource", "AsyncWorkspacesResource"]


class WorkspacesResource(SyncAPIResource):
    @cached_property
    def members(self) -> MembersResource:
        return MembersResource(self._client)

    @cached_property
    def with_raw_response(self) -> WorkspacesResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return WorkspacesResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> WorkspacesResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return WorkspacesResourceWithStreamingResponse(self)

    def create(
        self,
        *,
        name: str,
        wait_role_propagation: bool | Omit = omit,
        description: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Workspace:
        """
        Create a new workspace.

        The creator is automatically granted Admin role on the workspace. By default,
        this endpoint waits for the Admin role to propagate before returning. Use
        `wait_role_propagation=false` to skip waiting (useful for bulk operations).

        Example:

        ```
        POST / apis / entities / v2 / workspaces
        {"name": "ml-team", "description": "Machine Learning Team workspace"}
        ```

        Args:
          name: Workspace name (unique identifier). Name must start with a lowercase letter, be
              2-63 characters, and use lowercase letters, digits, hyphens, and dots (no
              consecutive hyphens, cannot end with a hyphen).

          wait_role_propagation: If true, wait for Admin role to propagate before returning (default: true). Set
              to false for bulk operations.

          description: Optional description of the workspace


          exist_ok: Do not raise an error if the resource already exists. Returns the existing resource.


          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        try:
            return self._post(
                "/apis/entities/v2/workspaces",
                body=maybe_transform(
                    {
                        "name": name,
                        "description": description,
                    },
                    workspace_create_params.WorkspaceCreateParams,
                ),
                options=make_request_options(
                    extra_headers=extra_headers,
                    extra_query=extra_query,
                    extra_body=extra_body,
                    timeout=timeout,
                    query=maybe_transform(
                        {"wait_role_propagation": wait_role_propagation}, workspace_create_params.WorkspaceCreateParams
                    ),
                ),
                cast_to=Workspace,
            )
        except ConflictError:
            if not exist_ok:
                raise
            return self.retrieve(name = name)

    def retrieve(
        self,
        name: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Workspace:
        """
        Get a specific workspace by ID.

        Example:

        ```
        GET / apis / entities / v2 / workspaces / ml - team
        ```

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        return self._get(
            path_template("/apis/entities/v2/workspaces/{name}", name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Workspace,
        )

    def update(
        self,
        name: str,
        *,
        description: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Workspace:
        """
        Update a workspace's description.

        Example:

        ```
        PUT / apis / entities / v2 / workspaces / ml - team
        {"description": "Updated description for ML Team"}
        ```

        Args:
          description: Updated description

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        return self._put(
            path_template("/apis/entities/v2/workspaces/{name}", name=name),
            body=maybe_transform({"description": description}, workspace_update_params.WorkspaceUpdateParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Workspace,
        )

    def list(
        self,
        *,
        filter: str | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: GenericSortField | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> SyncDefaultPagination[Workspace]:
        """
        List all workspaces with pagination.

        Workspaces marked for deletion (non-null deletion_stage) are omitted so the list
        matches GET/DELETE, which treat those workspaces as not found.

        When authentication is enabled, only workspaces the principal has access to are
        returned. Service principals and platform admins have access to all workspaces.

        Query Parameters:

        - page, page_size: Pagination
        - sort: Sort field
        - filter: Advanced filters (JSON, text, or bracket notation)

        Example:

        ```
        GET /apis/entities/v2/workspaces?sort=-created_at&page=1&page_size=10
        ```

        Args:
          filter:
              Query filter expression. Supports text and JSON syntaxes:

              - Text: name:"value" AND status>500 with operators : ~ > >= < <= IN NOT IN AND
                OR and negation prefix -
              - Object (JSON): {"name":{"$like":"value"}} with operators $eq, $like, $lt,
                $lte, $gt, $gte, $in, $nin, $and, $or, $not
              - Bracket notation: ?filter[name][$like]=value
              - Relationship traversal: ?filter[relationship][$exists]=true or
                ?filter[relationship][field]=value

          page: Page number

          page_size: Items per page

          sort: Sort field

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return self._get_api_list(
            "/apis/entities/v2/workspaces",
            page=SyncDefaultPagination[Workspace],
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
                    workspace_list_params.WorkspaceListParams,
                ),
            ),
            model=Workspace,
        )

    def delete(
        self,
        name: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> DeleteResponse:
        """
        Delete a workspace.

        This marks the workspace for deletion and returns immediately. The workspace
        will no longer be accessible via the API. An asynchronous cleanup controller
        will handle deletion of all entities and external resources.

        Role bindings are immediately deleted to revoke access.

        Example:

        ```
        DELETE / apis / entities / v2 / workspaces / ml - team
        ```

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        return self._delete(
            path_template("/apis/entities/v2/workspaces/{name}", name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=DeleteResponse,
        )


class AsyncWorkspacesResource(AsyncAPIResource):
    @cached_property
    def members(self) -> AsyncMembersResource:
        return AsyncMembersResource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncWorkspacesResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncWorkspacesResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncWorkspacesResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncWorkspacesResourceWithStreamingResponse(self)

    async def create(
        self,
        *,
        name: str,
        wait_role_propagation: bool | Omit = omit,
        description: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Workspace:
        """
        Create a new workspace.

        The creator is automatically granted Admin role on the workspace. By default,
        this endpoint waits for the Admin role to propagate before returning. Use
        `wait_role_propagation=false` to skip waiting (useful for bulk operations).

        Example:

        ```
        POST / apis / entities / v2 / workspaces
        {"name": "ml-team", "description": "Machine Learning Team workspace"}
        ```

        Args:
          name: Workspace name (unique identifier). Name must start with a lowercase letter, be
              2-63 characters, and use lowercase letters, digits, hyphens, and dots (no
              consecutive hyphens, cannot end with a hyphen).

          wait_role_propagation: If true, wait for Admin role to propagate before returning (default: true). Set
              to false for bulk operations.

          description: Optional description of the workspace


          exist_ok: Do not raise an error if the resource already exists. Returns the existing resource.


          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        try:
            return await self._post(
                "/apis/entities/v2/workspaces",
                body=await async_maybe_transform(
                    {
                        "name": name,
                        "description": description,
                    },
                    workspace_create_params.WorkspaceCreateParams,
                ),
                options=make_request_options(
                    extra_headers=extra_headers,
                    extra_query=extra_query,
                    extra_body=extra_body,
                    timeout=timeout,
                    query=await async_maybe_transform(
                        {"wait_role_propagation": wait_role_propagation}, workspace_create_params.WorkspaceCreateParams
                    ),
                ),
                cast_to=Workspace,
            )
        except ConflictError:
            if not exist_ok:
                raise
            return await self.retrieve(name = name)

    async def retrieve(
        self,
        name: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Workspace:
        """
        Get a specific workspace by ID.

        Example:

        ```
        GET / apis / entities / v2 / workspaces / ml - team
        ```

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        return await self._get(
            path_template("/apis/entities/v2/workspaces/{name}", name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Workspace,
        )

    async def update(
        self,
        name: str,
        *,
        description: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Workspace:
        """
        Update a workspace's description.

        Example:

        ```
        PUT / apis / entities / v2 / workspaces / ml - team
        {"description": "Updated description for ML Team"}
        ```

        Args:
          description: Updated description

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        return await self._put(
            path_template("/apis/entities/v2/workspaces/{name}", name=name),
            body=await async_maybe_transform(
                {"description": description}, workspace_update_params.WorkspaceUpdateParams
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Workspace,
        )

    def list(
        self,
        *,
        filter: str | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: GenericSortField | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> AsyncPaginator[Workspace, AsyncDefaultPagination[Workspace]]:
        """
        List all workspaces with pagination.

        Workspaces marked for deletion (non-null deletion_stage) are omitted so the list
        matches GET/DELETE, which treat those workspaces as not found.

        When authentication is enabled, only workspaces the principal has access to are
        returned. Service principals and platform admins have access to all workspaces.

        Query Parameters:

        - page, page_size: Pagination
        - sort: Sort field
        - filter: Advanced filters (JSON, text, or bracket notation)

        Example:

        ```
        GET /apis/entities/v2/workspaces?sort=-created_at&page=1&page_size=10
        ```

        Args:
          filter:
              Query filter expression. Supports text and JSON syntaxes:

              - Text: name:"value" AND status>500 with operators : ~ > >= < <= IN NOT IN AND
                OR and negation prefix -
              - Object (JSON): {"name":{"$like":"value"}} with operators $eq, $like, $lt,
                $lte, $gt, $gte, $in, $nin, $and, $or, $not
              - Bracket notation: ?filter[name][$like]=value
              - Relationship traversal: ?filter[relationship][$exists]=true or
                ?filter[relationship][field]=value

          page: Page number

          page_size: Items per page

          sort: Sort field

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return self._get_api_list(
            "/apis/entities/v2/workspaces",
            page=AsyncDefaultPagination[Workspace],
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
                    workspace_list_params.WorkspaceListParams,
                ),
            ),
            model=Workspace,
        )

    async def delete(
        self,
        name: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> DeleteResponse:
        """
        Delete a workspace.

        This marks the workspace for deletion and returns immediately. The workspace
        will no longer be accessible via the API. An asynchronous cleanup controller
        will handle deletion of all entities and external resources.

        Role bindings are immediately deleted to revoke access.

        Example:

        ```
        DELETE / apis / entities / v2 / workspaces / ml - team
        ```

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        return await self._delete(
            path_template("/apis/entities/v2/workspaces/{name}", name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=DeleteResponse,
        )


class WorkspacesResourceWithRawResponse:
    def __init__(self, workspaces: WorkspacesResource) -> None:
        self._workspaces = workspaces

        self.create = to_raw_response_wrapper(
            workspaces.create,
        )
        self.retrieve = to_raw_response_wrapper(
            workspaces.retrieve,
        )
        self.update = to_raw_response_wrapper(
            workspaces.update,
        )
        self.list = to_raw_response_wrapper(
            workspaces.list,
        )
        self.delete = to_raw_response_wrapper(
            workspaces.delete,
        )

    @cached_property
    def members(self) -> MembersResourceWithRawResponse:
        return MembersResourceWithRawResponse(self._workspaces.members)


class AsyncWorkspacesResourceWithRawResponse:
    def __init__(self, workspaces: AsyncWorkspacesResource) -> None:
        self._workspaces = workspaces

        self.create = async_to_raw_response_wrapper(
            workspaces.create,
        )
        self.retrieve = async_to_raw_response_wrapper(
            workspaces.retrieve,
        )
        self.update = async_to_raw_response_wrapper(
            workspaces.update,
        )
        self.list = async_to_raw_response_wrapper(
            workspaces.list,
        )
        self.delete = async_to_raw_response_wrapper(
            workspaces.delete,
        )

    @cached_property
    def members(self) -> AsyncMembersResourceWithRawResponse:
        return AsyncMembersResourceWithRawResponse(self._workspaces.members)


class WorkspacesResourceWithStreamingResponse:
    def __init__(self, workspaces: WorkspacesResource) -> None:
        self._workspaces = workspaces

        self.create = to_streamed_response_wrapper(
            workspaces.create,
        )
        self.retrieve = to_streamed_response_wrapper(
            workspaces.retrieve,
        )
        self.update = to_streamed_response_wrapper(
            workspaces.update,
        )
        self.list = to_streamed_response_wrapper(
            workspaces.list,
        )
        self.delete = to_streamed_response_wrapper(
            workspaces.delete,
        )

    @cached_property
    def members(self) -> MembersResourceWithStreamingResponse:
        return MembersResourceWithStreamingResponse(self._workspaces.members)


class AsyncWorkspacesResourceWithStreamingResponse:
    def __init__(self, workspaces: AsyncWorkspacesResource) -> None:
        self._workspaces = workspaces

        self.create = async_to_streamed_response_wrapper(
            workspaces.create,
        )
        self.retrieve = async_to_streamed_response_wrapper(
            workspaces.retrieve,
        )
        self.update = async_to_streamed_response_wrapper(
            workspaces.update,
        )
        self.list = async_to_streamed_response_wrapper(
            workspaces.list,
        )
        self.delete = async_to_streamed_response_wrapper(
            workspaces.delete,
        )

    @cached_property
    def members(self) -> AsyncMembersResourceWithStreamingResponse:
        return AsyncMembersResourceWithStreamingResponse(self._workspaces.members)
