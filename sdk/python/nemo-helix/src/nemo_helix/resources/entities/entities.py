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
from ..._base_client import AsyncPaginator, make_request_options
from ...types.entities import (
    entity_list_params,
    entity_create_params,
    entity_get_entity_by_name_params,
    entity_delete_entity_by_name_params,
    entity_update_entity_by_name_params,
)
from ...types.entities.entity import Entity
from ...types.shared.delete_response import DeleteResponse

__all__ = ["EntitiesResource", "AsyncEntitiesResource"]


class EntitiesResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> EntitiesResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return EntitiesResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> EntitiesResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return EntitiesResourceWithStreamingResponse(self)

    def create(
        self,
        entity_type: str,
        *,
        workspace: str | None = None,
        data: Dict[str, object],
        name: str | Omit = omit,
        parent: str | Omit = omit,
        project: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Entity:
        """
        Create a new entity of the specified type in the given workspace.

        If name is not provided, it will be auto-generated based on the entity type.

        Example:

        ```
        POST / apis / entities / v2 / workspaces / default / entities / customization_config
        {"name": "my-config", "data": {"target_id": "llama-2-7b", "training_options": {"learning_rate": 0.01}}}
        ```

        Args:
          data: Entity-specific data (schema is opaque to entity store, validated by client SDK)

          name: Entity name (optional - auto-generated if not provided). Name must start with a
              lowercase letter, be 2-63 characters, and use lowercase letters, digits,
              hyphens, and dots (no consecutive hyphens, cannot end with a hyphen).

          parent: Parent entity ID for nested entities

          project: The name of the project associated with this entity

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not entity_type:
            raise ValueError(f"Expected a non-empty value for `entity_type` but received {entity_type!r}")
        return self._post(
            path_template(
                "/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}",
                workspace=workspace,
                entity_type=entity_type,
            ),
            body=maybe_transform(
                {
                    "data": data,
                    "name": name,
                    "parent": parent,
                    "project": project,
                },
                entity_create_params.EntityCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Entity,
        )

    def list(
        self,
        entity_type: str,
        *,
        workspace: str | None = None,
        count_by: str | Omit = omit,
        filter: str | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> SyncDefaultPagination[Entity]:
        """
        List all entities of a specific type in the given workspace.

        Use workspace="-" to list entities across all workspaces the principal has
        access to.

        Query Parameters:

        - sort: Sort field
        - page, page_size: Pagination
        - filter: Advanced filters (JSON, text, or bracket notation)

        Examples:

        ```
        GET /apis/entities/v2/workspaces/default/entities/customization_config?sort=-created_at
        GET /apis/entities/v2/workspaces/-/entities/customization_config  # Cross-workspace query
        ```

        Args:
          count_by: Optional direct string data field whose matching values should be counted.

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
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not entity_type:
            raise ValueError(f"Expected a non-empty value for `entity_type` but received {entity_type!r}")
        return self._get_api_list(
            path_template(
                "/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}",
                workspace=workspace,
                entity_type=entity_type,
            ),
            page=SyncDefaultPagination[Entity],
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "count_by": count_by,
                        "filter": filter,
                        "page": page,
                        "page_size": page_size,
                        "sort": sort,
                    },
                    entity_list_params.EntityListParams,
                ),
            ),
            model=Entity,
        )

    def delete_entity_by_name(
        self,
        name: str,
        *,
        workspace: str | None = None,
        entity_type: str,
        expected_db_version: int | Omit = omit,
        parent: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> DeleteResponse:
        """
        Delete an entity by its name.

        Example:

        ```
        DELETE / apis / entities / v2 / workspaces / default / entities / customization_config / my - config
        ```

        Args:
          expected_db_version: Optional database version for optimistic locking. Delete only succeeds if the
              entity still has this version.

          parent: Parent entity ID for nested entities

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not entity_type:
            raise ValueError(f"Expected a non-empty value for `entity_type` but received {entity_type!r}")
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        return self._delete(
            path_template(
                "/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}/{name}",
                workspace=workspace,
                entity_type=entity_type,
                name=name,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "expected_db_version": expected_db_version,
                        "parent": parent,
                    },
                    entity_delete_entity_by_name_params.EntityDeleteEntityByNameParams,
                ),
            ),
            cast_to=DeleteResponse,
        )

    def get_entity_by_id(
        self,
        id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Entity:
        """Get a specific entity by its unique identifier.

        This endpoint is primarily for
        debugging and internal use.

        Example:

        ```
        GET /apis/entities/v2/entities/customization-config-5Q2LoF8z8M9JZxZsHwJKNn
        ```

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not id:
            raise ValueError(f"Expected a non-empty value for `id` but received {id!r}")
        return self._get(
            path_template("/apis/entities/v2/entities/{id}", id=id),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Entity,
        )

    def get_entity_by_name(
        self,
        name: str,
        *,
        workspace: str | None = None,
        entity_type: str,
        parent: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Entity:
        """
        Get a specific entity by its workspace, type, and name.

        Example:

        ```
        GET / apis / entities / v2 / workspaces / default / entities / customization_config / my - config
        ```

        Args:
          parent: Parent entity ID for nested entities

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not entity_type:
            raise ValueError(f"Expected a non-empty value for `entity_type` but received {entity_type!r}")
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        return self._get(
            path_template(
                "/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}/{name}",
                workspace=workspace,
                entity_type=entity_type,
                name=name,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform({"parent": parent}, entity_get_entity_by_name_params.EntityGetEntityByNameParams),
            ),
            cast_to=Entity,
        )

    def update_entity_by_name(
        self,
        name: str,
        *,
        workspace: str | None = None,
        entity_type: str,
        data: Dict[str, object],
        parent: str | Omit = omit,
        expected_db_version: int | Omit = omit,
        new_name: str | Omit = omit,
        project: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Entity:
        """Update an entity by its name.

        Optionally change the entity's name.

        Example:

        ```
        PUT / apis / entities / v2 / workspaces / default / entities / customization_config / my - config
        {"data": {"target_id": "llama-2-7b", "training_options": {"learning_rate": 0.02}}}
        ```

        Args:
          data: Updated entity-specific data

          parent: Parent entity ID for nested entities

          expected_db_version: Optional database version for optimistic locking. Update only succeeds if
              current version matches.

          new_name: Updated entity name (optional). Name must start with a lowercase letter, be 2-63
              characters, and use lowercase letters, digits, hyphens, and dots (no consecutive
              hyphens, cannot end with a hyphen).

          project: The name of the project associated with this entity

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not entity_type:
            raise ValueError(f"Expected a non-empty value for `entity_type` but received {entity_type!r}")
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        return self._put(
            path_template(
                "/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}/{name}",
                workspace=workspace,
                entity_type=entity_type,
                name=name,
            ),
            body=maybe_transform(
                {
                    "data": data,
                    "expected_db_version": expected_db_version,
                    "new_name": new_name,
                    "project": project,
                },
                entity_update_entity_by_name_params.EntityUpdateEntityByNameParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {"parent": parent}, entity_update_entity_by_name_params.EntityUpdateEntityByNameParams
                ),
            ),
            cast_to=Entity,
        )


class AsyncEntitiesResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncEntitiesResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncEntitiesResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncEntitiesResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncEntitiesResourceWithStreamingResponse(self)

    async def create(
        self,
        entity_type: str,
        *,
        workspace: str | None = None,
        data: Dict[str, object],
        name: str | Omit = omit,
        parent: str | Omit = omit,
        project: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Entity:
        """
        Create a new entity of the specified type in the given workspace.

        If name is not provided, it will be auto-generated based on the entity type.

        Example:

        ```
        POST / apis / entities / v2 / workspaces / default / entities / customization_config
        {"name": "my-config", "data": {"target_id": "llama-2-7b", "training_options": {"learning_rate": 0.01}}}
        ```

        Args:
          data: Entity-specific data (schema is opaque to entity store, validated by client SDK)

          name: Entity name (optional - auto-generated if not provided). Name must start with a
              lowercase letter, be 2-63 characters, and use lowercase letters, digits,
              hyphens, and dots (no consecutive hyphens, cannot end with a hyphen).

          parent: Parent entity ID for nested entities

          project: The name of the project associated with this entity

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not entity_type:
            raise ValueError(f"Expected a non-empty value for `entity_type` but received {entity_type!r}")
        return await self._post(
            path_template(
                "/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}",
                workspace=workspace,
                entity_type=entity_type,
            ),
            body=await async_maybe_transform(
                {
                    "data": data,
                    "name": name,
                    "parent": parent,
                    "project": project,
                },
                entity_create_params.EntityCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Entity,
        )

    def list(
        self,
        entity_type: str,
        *,
        workspace: str | None = None,
        count_by: str | Omit = omit,
        filter: str | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> AsyncPaginator[Entity, AsyncDefaultPagination[Entity]]:
        """
        List all entities of a specific type in the given workspace.

        Use workspace="-" to list entities across all workspaces the principal has
        access to.

        Query Parameters:

        - sort: Sort field
        - page, page_size: Pagination
        - filter: Advanced filters (JSON, text, or bracket notation)

        Examples:

        ```
        GET /apis/entities/v2/workspaces/default/entities/customization_config?sort=-created_at
        GET /apis/entities/v2/workspaces/-/entities/customization_config  # Cross-workspace query
        ```

        Args:
          count_by: Optional direct string data field whose matching values should be counted.

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
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not entity_type:
            raise ValueError(f"Expected a non-empty value for `entity_type` but received {entity_type!r}")
        return self._get_api_list(
            path_template(
                "/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}",
                workspace=workspace,
                entity_type=entity_type,
            ),
            page=AsyncDefaultPagination[Entity],
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "count_by": count_by,
                        "filter": filter,
                        "page": page,
                        "page_size": page_size,
                        "sort": sort,
                    },
                    entity_list_params.EntityListParams,
                ),
            ),
            model=Entity,
        )

    async def delete_entity_by_name(
        self,
        name: str,
        *,
        workspace: str | None = None,
        entity_type: str,
        expected_db_version: int | Omit = omit,
        parent: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> DeleteResponse:
        """
        Delete an entity by its name.

        Example:

        ```
        DELETE / apis / entities / v2 / workspaces / default / entities / customization_config / my - config
        ```

        Args:
          expected_db_version: Optional database version for optimistic locking. Delete only succeeds if the
              entity still has this version.

          parent: Parent entity ID for nested entities

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not entity_type:
            raise ValueError(f"Expected a non-empty value for `entity_type` but received {entity_type!r}")
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        return await self._delete(
            path_template(
                "/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}/{name}",
                workspace=workspace,
                entity_type=entity_type,
                name=name,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {
                        "expected_db_version": expected_db_version,
                        "parent": parent,
                    },
                    entity_delete_entity_by_name_params.EntityDeleteEntityByNameParams,
                ),
            ),
            cast_to=DeleteResponse,
        )

    async def get_entity_by_id(
        self,
        id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Entity:
        """Get a specific entity by its unique identifier.

        This endpoint is primarily for
        debugging and internal use.

        Example:

        ```
        GET /apis/entities/v2/entities/customization-config-5Q2LoF8z8M9JZxZsHwJKNn
        ```

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not id:
            raise ValueError(f"Expected a non-empty value for `id` but received {id!r}")
        return await self._get(
            path_template("/apis/entities/v2/entities/{id}", id=id),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Entity,
        )

    async def get_entity_by_name(
        self,
        name: str,
        *,
        workspace: str | None = None,
        entity_type: str,
        parent: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Entity:
        """
        Get a specific entity by its workspace, type, and name.

        Example:

        ```
        GET / apis / entities / v2 / workspaces / default / entities / customization_config / my - config
        ```

        Args:
          parent: Parent entity ID for nested entities

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not entity_type:
            raise ValueError(f"Expected a non-empty value for `entity_type` but received {entity_type!r}")
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        return await self._get(
            path_template(
                "/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}/{name}",
                workspace=workspace,
                entity_type=entity_type,
                name=name,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {"parent": parent}, entity_get_entity_by_name_params.EntityGetEntityByNameParams
                ),
            ),
            cast_to=Entity,
        )

    async def update_entity_by_name(
        self,
        name: str,
        *,
        workspace: str | None = None,
        entity_type: str,
        data: Dict[str, object],
        parent: str | Omit = omit,
        expected_db_version: int | Omit = omit,
        new_name: str | Omit = omit,
        project: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Entity:
        """Update an entity by its name.

        Optionally change the entity's name.

        Example:

        ```
        PUT / apis / entities / v2 / workspaces / default / entities / customization_config / my - config
        {"data": {"target_id": "llama-2-7b", "training_options": {"learning_rate": 0.02}}}
        ```

        Args:
          data: Updated entity-specific data

          parent: Parent entity ID for nested entities

          expected_db_version: Optional database version for optimistic locking. Update only succeeds if
              current version matches.

          new_name: Updated entity name (optional). Name must start with a lowercase letter, be 2-63
              characters, and use lowercase letters, digits, hyphens, and dots (no consecutive
              hyphens, cannot end with a hyphen).

          project: The name of the project associated with this entity

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not entity_type:
            raise ValueError(f"Expected a non-empty value for `entity_type` but received {entity_type!r}")
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        return await self._put(
            path_template(
                "/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}/{name}",
                workspace=workspace,
                entity_type=entity_type,
                name=name,
            ),
            body=await async_maybe_transform(
                {
                    "data": data,
                    "expected_db_version": expected_db_version,
                    "new_name": new_name,
                    "project": project,
                },
                entity_update_entity_by_name_params.EntityUpdateEntityByNameParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {"parent": parent}, entity_update_entity_by_name_params.EntityUpdateEntityByNameParams
                ),
            ),
            cast_to=Entity,
        )


class EntitiesResourceWithRawResponse:
    def __init__(self, entities: EntitiesResource) -> None:
        self._entities = entities

        self.create = to_raw_response_wrapper(
            entities.create,
        )
        self.list = to_raw_response_wrapper(
            entities.list,
        )
        self.delete_entity_by_name = to_raw_response_wrapper(
            entities.delete_entity_by_name,
        )
        self.get_entity_by_id = to_raw_response_wrapper(
            entities.get_entity_by_id,
        )
        self.get_entity_by_name = to_raw_response_wrapper(
            entities.get_entity_by_name,
        )
        self.update_entity_by_name = to_raw_response_wrapper(
            entities.update_entity_by_name,
        )


class AsyncEntitiesResourceWithRawResponse:
    def __init__(self, entities: AsyncEntitiesResource) -> None:
        self._entities = entities

        self.create = async_to_raw_response_wrapper(
            entities.create,
        )
        self.list = async_to_raw_response_wrapper(
            entities.list,
        )
        self.delete_entity_by_name = async_to_raw_response_wrapper(
            entities.delete_entity_by_name,
        )
        self.get_entity_by_id = async_to_raw_response_wrapper(
            entities.get_entity_by_id,
        )
        self.get_entity_by_name = async_to_raw_response_wrapper(
            entities.get_entity_by_name,
        )
        self.update_entity_by_name = async_to_raw_response_wrapper(
            entities.update_entity_by_name,
        )


class EntitiesResourceWithStreamingResponse:
    def __init__(self, entities: EntitiesResource) -> None:
        self._entities = entities

        self.create = to_streamed_response_wrapper(
            entities.create,
        )
        self.list = to_streamed_response_wrapper(
            entities.list,
        )
        self.delete_entity_by_name = to_streamed_response_wrapper(
            entities.delete_entity_by_name,
        )
        self.get_entity_by_id = to_streamed_response_wrapper(
            entities.get_entity_by_id,
        )
        self.get_entity_by_name = to_streamed_response_wrapper(
            entities.get_entity_by_name,
        )
        self.update_entity_by_name = to_streamed_response_wrapper(
            entities.update_entity_by_name,
        )


class AsyncEntitiesResourceWithStreamingResponse:
    def __init__(self, entities: AsyncEntitiesResource) -> None:
        self._entities = entities

        self.create = async_to_streamed_response_wrapper(
            entities.create,
        )
        self.list = async_to_streamed_response_wrapper(
            entities.list,
        )
        self.delete_entity_by_name = async_to_streamed_response_wrapper(
            entities.delete_entity_by_name,
        )
        self.get_entity_by_id = async_to_streamed_response_wrapper(
            entities.get_entity_by_id,
        )
        self.get_entity_by_name = async_to_streamed_response_wrapper(
            entities.get_entity_by_name,
        )
        self.update_entity_by_name = async_to_streamed_response_wrapper(
            entities.update_entity_by_name,
        )
