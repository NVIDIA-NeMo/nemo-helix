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

from ..._types import Body, Omit, Query, Headers, NotGiven, SequenceNotStr, omit, not_given
from ..._utils import path_template, maybe_transform, async_maybe_transform
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from ..._base_client import make_request_options
from ...types.workspaces import member_create_params, member_delete_params, member_update_params
from ...types.shared.delete_response import DeleteResponse
from ...types.workspaces.workspace_member import WorkspaceMember
from ...types.workspaces.workspace_member_list_response import WorkspaceMemberListResponse

__all__ = ["MembersResource", "AsyncMembersResource"]


class MembersResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> MembersResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return MembersResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> MembersResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return MembersResourceWithStreamingResponse(self)

    def create(
        self,
        *,
        workspace: str | None = None,
        principal: str,
        wait_role_propagation: bool | Omit = omit,
        roles: SequenceNotStr[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> WorkspaceMember:
        """
        Add a new member to the workspace with specified roles.

        This creates role bindings for the specified principal with the given roles. By
        default, this endpoint waits for the roles to propagate before returning. Use
        `wait_role_propagation=false` to skip waiting (useful for bulk operations).

        Example:

        ```
        POST / apis / entities / v2 / workspaces / ml - team / members
        {"principal": "user@example.com", "roles": ["Editor"]}
        ```

        Args:
          principal: The principal identifier (email, user ID, or group ID)

          wait_role_propagation: If true, wait for roles to propagate before returning (default: true). Set to
              false for bulk operations.

          roles: List of roles to grant to the principal

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        return self._post(
            path_template("/apis/entities/v2/workspaces/{workspace}/members", workspace=workspace),
            body=maybe_transform(
                {
                    "principal": principal,
                    "roles": roles,
                },
                member_create_params.MemberCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {"wait_role_propagation": wait_role_propagation}, member_create_params.MemberCreateParams
                ),
            ),
            cast_to=WorkspaceMember,
        )

    def update(
        self,
        principal_id: str,
        *,
        workspace: str | None = None,
        roles: SequenceNotStr[str],
        wait_role_propagation: bool | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> WorkspaceMember:
        """
        Update the roles for a workspace member.

        This will revoke existing roles not in the new list and add new roles. By
        default, this endpoint waits for the roles to propagate before returning. Use
        `wait_role_propagation=false` to skip waiting (useful for bulk operations).

        Example:

        ```
        PUT / apis / entities / v2 / workspaces / ml - team / members / user @ example.com
        {"roles": ["Viewer", "Editor"]}
        ```

        Args:
          roles: Updated list of roles for the principal

          wait_role_propagation: If true, wait for roles to propagate before returning (default: true). Set to
              false for bulk operations.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not principal_id:
            raise ValueError(f"Expected a non-empty value for `principal_id` but received {principal_id!r}")
        return self._put(
            path_template(
                "/apis/entities/v2/workspaces/{workspace}/members/{principal_id}",
                workspace=workspace,
                principal_id=principal_id,
            ),
            body=maybe_transform({"roles": roles}, member_update_params.MemberUpdateParams),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {"wait_role_propagation": wait_role_propagation}, member_update_params.MemberUpdateParams
                ),
            ),
            cast_to=WorkspaceMember,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> WorkspaceMemberListResponse:
        """
        List all members of a workspace with their roles.

        Returns a list of all principals with active role bindings in the workspace.

        Example:

        ```
        GET / apis / entities / v2 / workspaces / ml - team / members
        ```

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
        return self._get(
            path_template("/apis/entities/v2/workspaces/{workspace}/members", workspace=workspace),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=WorkspaceMemberListResponse,
        )

    def delete(
        self,
        principal_id: str,
        *,
        workspace: str | None = None,
        wait_role_propagation: bool | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> DeleteResponse:
        """
        Remove a member from the workspace by revoking all their roles.

        This revokes all active role bindings for the principal in the workspace. By
        default, this endpoint waits for all roles to be revoked before returning. Use
        `wait_role_propagation=false` to skip waiting (useful for bulk operations).

        Example:

        ```
        DELETE / apis / entities / v2 / workspaces / ml - team / members / user @ example.com
        ```

        Args:
          wait_role_propagation: If true, wait for roles to propagate before returning (default: true). Set to
              false for bulk operations.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not principal_id:
            raise ValueError(f"Expected a non-empty value for `principal_id` but received {principal_id!r}")
        return self._delete(
            path_template(
                "/apis/entities/v2/workspaces/{workspace}/members/{principal_id}",
                workspace=workspace,
                principal_id=principal_id,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {"wait_role_propagation": wait_role_propagation}, member_delete_params.MemberDeleteParams
                ),
            ),
            cast_to=DeleteResponse,
        )


class AsyncMembersResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncMembersResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncMembersResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncMembersResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncMembersResourceWithStreamingResponse(self)

    async def create(
        self,
        *,
        workspace: str | None = None,
        principal: str,
        wait_role_propagation: bool | Omit = omit,
        roles: SequenceNotStr[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> WorkspaceMember:
        """
        Add a new member to the workspace with specified roles.

        This creates role bindings for the specified principal with the given roles. By
        default, this endpoint waits for the roles to propagate before returning. Use
        `wait_role_propagation=false` to skip waiting (useful for bulk operations).

        Example:

        ```
        POST / apis / entities / v2 / workspaces / ml - team / members
        {"principal": "user@example.com", "roles": ["Editor"]}
        ```

        Args:
          principal: The principal identifier (email, user ID, or group ID)

          wait_role_propagation: If true, wait for roles to propagate before returning (default: true). Set to
              false for bulk operations.

          roles: List of roles to grant to the principal

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        return await self._post(
            path_template("/apis/entities/v2/workspaces/{workspace}/members", workspace=workspace),
            body=await async_maybe_transform(
                {
                    "principal": principal,
                    "roles": roles,
                },
                member_create_params.MemberCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {"wait_role_propagation": wait_role_propagation}, member_create_params.MemberCreateParams
                ),
            ),
            cast_to=WorkspaceMember,
        )

    async def update(
        self,
        principal_id: str,
        *,
        workspace: str | None = None,
        roles: SequenceNotStr[str],
        wait_role_propagation: bool | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> WorkspaceMember:
        """
        Update the roles for a workspace member.

        This will revoke existing roles not in the new list and add new roles. By
        default, this endpoint waits for the roles to propagate before returning. Use
        `wait_role_propagation=false` to skip waiting (useful for bulk operations).

        Example:

        ```
        PUT / apis / entities / v2 / workspaces / ml - team / members / user @ example.com
        {"roles": ["Viewer", "Editor"]}
        ```

        Args:
          roles: Updated list of roles for the principal

          wait_role_propagation: If true, wait for roles to propagate before returning (default: true). Set to
              false for bulk operations.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not principal_id:
            raise ValueError(f"Expected a non-empty value for `principal_id` but received {principal_id!r}")
        return await self._put(
            path_template(
                "/apis/entities/v2/workspaces/{workspace}/members/{principal_id}",
                workspace=workspace,
                principal_id=principal_id,
            ),
            body=await async_maybe_transform({"roles": roles}, member_update_params.MemberUpdateParams),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {"wait_role_propagation": wait_role_propagation}, member_update_params.MemberUpdateParams
                ),
            ),
            cast_to=WorkspaceMember,
        )

    async def list(
        self,
        *,
        workspace: str | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> WorkspaceMemberListResponse:
        """
        List all members of a workspace with their roles.

        Returns a list of all principals with active role bindings in the workspace.

        Example:

        ```
        GET / apis / entities / v2 / workspaces / ml - team / members
        ```

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
        return await self._get(
            path_template("/apis/entities/v2/workspaces/{workspace}/members", workspace=workspace),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=WorkspaceMemberListResponse,
        )

    async def delete(
        self,
        principal_id: str,
        *,
        workspace: str | None = None,
        wait_role_propagation: bool | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> DeleteResponse:
        """
        Remove a member from the workspace by revoking all their roles.

        This revokes all active role bindings for the principal in the workspace. By
        default, this endpoint waits for all roles to be revoked before returning. Use
        `wait_role_propagation=false` to skip waiting (useful for bulk operations).

        Example:

        ```
        DELETE / apis / entities / v2 / workspaces / ml - team / members / user @ example.com
        ```

        Args:
          wait_role_propagation: If true, wait for roles to propagate before returning (default: true). Set to
              false for bulk operations.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not principal_id:
            raise ValueError(f"Expected a non-empty value for `principal_id` but received {principal_id!r}")
        return await self._delete(
            path_template(
                "/apis/entities/v2/workspaces/{workspace}/members/{principal_id}",
                workspace=workspace,
                principal_id=principal_id,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {"wait_role_propagation": wait_role_propagation}, member_delete_params.MemberDeleteParams
                ),
            ),
            cast_to=DeleteResponse,
        )


class MembersResourceWithRawResponse:
    def __init__(self, members: MembersResource) -> None:
        self._members = members

        self.create = to_raw_response_wrapper(
            members.create,
        )
        self.update = to_raw_response_wrapper(
            members.update,
        )
        self.list = to_raw_response_wrapper(
            members.list,
        )
        self.delete = to_raw_response_wrapper(
            members.delete,
        )


class AsyncMembersResourceWithRawResponse:
    def __init__(self, members: AsyncMembersResource) -> None:
        self._members = members

        self.create = async_to_raw_response_wrapper(
            members.create,
        )
        self.update = async_to_raw_response_wrapper(
            members.update,
        )
        self.list = async_to_raw_response_wrapper(
            members.list,
        )
        self.delete = async_to_raw_response_wrapper(
            members.delete,
        )


class MembersResourceWithStreamingResponse:
    def __init__(self, members: MembersResource) -> None:
        self._members = members

        self.create = to_streamed_response_wrapper(
            members.create,
        )
        self.update = to_streamed_response_wrapper(
            members.update,
        )
        self.list = to_streamed_response_wrapper(
            members.list,
        )
        self.delete = to_streamed_response_wrapper(
            members.delete,
        )


class AsyncMembersResourceWithStreamingResponse:
    def __init__(self, members: AsyncMembersResource) -> None:
        self._members = members

        self.create = async_to_streamed_response_wrapper(
            members.create,
        )
        self.update = async_to_streamed_response_wrapper(
            members.update,
        )
        self.list = async_to_streamed_response_wrapper(
            members.list,
        )
        self.delete = async_to_streamed_response_wrapper(
            members.delete,
        )
