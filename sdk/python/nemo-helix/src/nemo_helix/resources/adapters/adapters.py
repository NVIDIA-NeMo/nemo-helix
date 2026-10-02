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

from ..._types import Body, Omit, Query, Headers, NoneType, NotGiven, omit, not_given
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
from ...types.adapters import adapter_list_params, adapter_patch_params, adapter_create_params
from ...types.models.adapter import Adapter
from ...types.models.lora_param import LoraParam
from ...types.shared.finetuning_type import FinetuningType
from ...types.adapters.adapter_entity_filter_param import AdapterEntityFilterParam

__all__ = ["AdaptersResource", "AsyncAdaptersResource"]


class AdaptersResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AdaptersResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AdaptersResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AdaptersResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AdaptersResourceWithStreamingResponse(self)

    def create(
        self,
        *,
        workspace: str | None = None,
        fileset: str,
        finetuning_type: FinetuningType,
        model: str,
        name: str,
        description: str | Omit = omit,
        enabled: bool | Omit = omit,
        lora_config: LoraParam | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Adapter:
        """
        Create an adapter under a base model specified by the "model" field in the body.

        Args:
          fileset: Location where adapter files are stored - expected format
              {workspace}/{fileset_name}

          finetuning_type: Finetuning types.

          model: Base model entity. Use `{workspace}/{model_name}` to reference a model in any
              workspace, or a single `{model_name}` resolved in the path workspace. A single
              name (2-63 characters) or 'workspace/model*name' where each segment is a valid
              name (lowercase, digits, hyphens, and temporarily @ . + *; no leading/trailing
              or consecutive hyphens). If one slash, both sides must be non-empty.

          name: Name of the adapter. Name must be unique in the workspace. Name must start with
              a lowercase letter, be 2-63 characters, and use lowercase letters, digits,
              hyphens, and dots (no consecutive hyphens, cannot end with a hyphen).

          description: Optional description of the adapter

          enabled: Whether to make this adapter available for inference post training

          lora_config: Lora configuration specifics

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
            path_template("/apis/models/v2/workspaces/{workspace}/adapters", workspace=workspace),
            body=maybe_transform(
                {
                    "fileset": fileset,
                    "finetuning_type": finetuning_type,
                    "model": model,
                    "name": name,
                    "description": description,
                    "enabled": enabled,
                    "lora_config": lora_config,
                },
                adapter_create_params.AdapterCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Adapter,
        )

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
    ) -> Adapter:
        """
        Get Adapter

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
            path_template("/apis/models/v2/workspaces/{workspace}/adapters/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Adapter,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: AdapterEntityFilterParam | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> SyncDefaultPagination[Adapter]:
        """
        List Adapters

        Args:
          filter: Filter adapters by name, model (parent model ref string, stored on the adapter),
              description, fileset, finetuning_type, enabled, created_at, and updated_at.

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
            path_template("/apis/models/v2/workspaces/{workspace}/adapters", workspace=workspace),
            page=SyncDefaultPagination[Adapter],
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
                    adapter_list_params.AdapterListParams,
                ),
            ),
            model=Adapter,
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
    ) -> None:
        """
        Delete Adapter

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
        extra_headers = {"Accept": "*/*", **(extra_headers or {})}
        return self._delete(
            path_template("/apis/models/v2/workspaces/{workspace}/adapters/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
        )

    def patch(
        self,
        name: str,
        *,
        workspace: str | None = None,
        description: str | Omit = omit,
        enabled: bool | Omit = omit,
        fileset: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Adapter:
        """
        Update Adapter

        Args:
          description: Optional description of the adapter

          enabled: Whether to make this adapter available for inference post training

          fileset: Updated fileset for the adapter

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
            path_template("/apis/models/v2/workspaces/{workspace}/adapters/{name}", workspace=workspace, name=name),
            body=maybe_transform(
                {
                    "description": description,
                    "enabled": enabled,
                    "fileset": fileset,
                },
                adapter_patch_params.AdapterPatchParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Adapter,
        )


class AsyncAdaptersResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncAdaptersResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncAdaptersResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncAdaptersResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncAdaptersResourceWithStreamingResponse(self)

    async def create(
        self,
        *,
        workspace: str | None = None,
        fileset: str,
        finetuning_type: FinetuningType,
        model: str,
        name: str,
        description: str | Omit = omit,
        enabled: bool | Omit = omit,
        lora_config: LoraParam | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Adapter:
        """
        Create an adapter under a base model specified by the "model" field in the body.

        Args:
          fileset: Location where adapter files are stored - expected format
              {workspace}/{fileset_name}

          finetuning_type: Finetuning types.

          model: Base model entity. Use `{workspace}/{model_name}` to reference a model in any
              workspace, or a single `{model_name}` resolved in the path workspace. A single
              name (2-63 characters) or 'workspace/model*name' where each segment is a valid
              name (lowercase, digits, hyphens, and temporarily @ . + *; no leading/trailing
              or consecutive hyphens). If one slash, both sides must be non-empty.

          name: Name of the adapter. Name must be unique in the workspace. Name must start with
              a lowercase letter, be 2-63 characters, and use lowercase letters, digits,
              hyphens, and dots (no consecutive hyphens, cannot end with a hyphen).

          description: Optional description of the adapter

          enabled: Whether to make this adapter available for inference post training

          lora_config: Lora configuration specifics

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
            path_template("/apis/models/v2/workspaces/{workspace}/adapters", workspace=workspace),
            body=await async_maybe_transform(
                {
                    "fileset": fileset,
                    "finetuning_type": finetuning_type,
                    "model": model,
                    "name": name,
                    "description": description,
                    "enabled": enabled,
                    "lora_config": lora_config,
                },
                adapter_create_params.AdapterCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Adapter,
        )

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
    ) -> Adapter:
        """
        Get Adapter

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
            path_template("/apis/models/v2/workspaces/{workspace}/adapters/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Adapter,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: AdapterEntityFilterParam | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> AsyncPaginator[Adapter, AsyncDefaultPagination[Adapter]]:
        """
        List Adapters

        Args:
          filter: Filter adapters by name, model (parent model ref string, stored on the adapter),
              description, fileset, finetuning_type, enabled, created_at, and updated_at.

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
            path_template("/apis/models/v2/workspaces/{workspace}/adapters", workspace=workspace),
            page=AsyncDefaultPagination[Adapter],
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
                    adapter_list_params.AdapterListParams,
                ),
            ),
            model=Adapter,
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
    ) -> None:
        """
        Delete Adapter

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
        extra_headers = {"Accept": "*/*", **(extra_headers or {})}
        return await self._delete(
            path_template("/apis/models/v2/workspaces/{workspace}/adapters/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
        )

    async def patch(
        self,
        name: str,
        *,
        workspace: str | None = None,
        description: str | Omit = omit,
        enabled: bool | Omit = omit,
        fileset: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Adapter:
        """
        Update Adapter

        Args:
          description: Optional description of the adapter

          enabled: Whether to make this adapter available for inference post training

          fileset: Updated fileset for the adapter

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
            path_template("/apis/models/v2/workspaces/{workspace}/adapters/{name}", workspace=workspace, name=name),
            body=await async_maybe_transform(
                {
                    "description": description,
                    "enabled": enabled,
                    "fileset": fileset,
                },
                adapter_patch_params.AdapterPatchParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Adapter,
        )


class AdaptersResourceWithRawResponse:
    def __init__(self, adapters: AdaptersResource) -> None:
        self._adapters = adapters

        self.create = to_raw_response_wrapper(
            adapters.create,
        )
        self.retrieve = to_raw_response_wrapper(
            adapters.retrieve,
        )
        self.list = to_raw_response_wrapper(
            adapters.list,
        )
        self.delete = to_raw_response_wrapper(
            adapters.delete,
        )
        self.patch = to_raw_response_wrapper(
            adapters.patch,
        )


class AsyncAdaptersResourceWithRawResponse:
    def __init__(self, adapters: AsyncAdaptersResource) -> None:
        self._adapters = adapters

        self.create = async_to_raw_response_wrapper(
            adapters.create,
        )
        self.retrieve = async_to_raw_response_wrapper(
            adapters.retrieve,
        )
        self.list = async_to_raw_response_wrapper(
            adapters.list,
        )
        self.delete = async_to_raw_response_wrapper(
            adapters.delete,
        )
        self.patch = async_to_raw_response_wrapper(
            adapters.patch,
        )


class AdaptersResourceWithStreamingResponse:
    def __init__(self, adapters: AdaptersResource) -> None:
        self._adapters = adapters

        self.create = to_streamed_response_wrapper(
            adapters.create,
        )
        self.retrieve = to_streamed_response_wrapper(
            adapters.retrieve,
        )
        self.list = to_streamed_response_wrapper(
            adapters.list,
        )
        self.delete = to_streamed_response_wrapper(
            adapters.delete,
        )
        self.patch = to_streamed_response_wrapper(
            adapters.patch,
        )


class AsyncAdaptersResourceWithStreamingResponse:
    def __init__(self, adapters: AsyncAdaptersResource) -> None:
        self._adapters = adapters

        self.create = async_to_streamed_response_wrapper(
            adapters.create,
        )
        self.retrieve = async_to_streamed_response_wrapper(
            adapters.retrieve,
        )
        self.list = async_to_streamed_response_wrapper(
            adapters.list,
        )
        self.delete = async_to_streamed_response_wrapper(
            adapters.delete,
        )
        self.patch = async_to_streamed_response_wrapper(
            adapters.patch,
        )
