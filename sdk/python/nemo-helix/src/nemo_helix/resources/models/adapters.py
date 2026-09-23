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
from ..._base_client import make_request_options
from ...types.models import adapter_create_params, adapter_update_params
from ...types.models.adapter import Adapter
from ...types.models.lora_param import LoraParam
from ...types.shared.finetuning_type import FinetuningType

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
        model_name: str,
        *,
        workspace: str | None = None,
        fileset: str,
        finetuning_type: FinetuningType,
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
        Adds an Adapter to the Model

        Args:
          fileset: Location where adapter files are stored - expected format
              {workspace}/{fileset_name}

          finetuning_type: Finetuning types.

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
        if not model_name:
            raise ValueError(f"Expected a non-empty value for `model_name` but received {model_name!r}")
        return self._post(
            path_template(
                "/apis/models/v2/workspaces/{workspace}/models/{model_name}/adapters",
                workspace=workspace,
                model_name=model_name,
            ),
            body=maybe_transform(
                {
                    "fileset": fileset,
                    "finetuning_type": finetuning_type,
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

    def update(
        self,
        adapter: str,
        *,
        workspace: str | None = None,
        model_name: str,
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
        Update Adapter deployment or description.

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
        if not model_name:
            raise ValueError(f"Expected a non-empty value for `model_name` but received {model_name!r}")
        if not adapter:
            raise ValueError(f"Expected a non-empty value for `adapter` but received {adapter!r}")
        return self._patch(
            path_template(
                "/apis/models/v2/workspaces/{workspace}/models/{model_name}/adapters/{adapter}",
                workspace=workspace,
                model_name=model_name,
                adapter=adapter,
            ),
            body=maybe_transform(
                {
                    "description": description,
                    "enabled": enabled,
                    "fileset": fileset,
                },
                adapter_update_params.AdapterUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Adapter,
        )

    def delete(
        self,
        adapter: str,
        *,
        workspace: str | None = None,
        model_name: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> None:
        """
        Delete Adapter from Model entity.

        Permanently deletes an adapter from a model entity, if it was deployed, it will
        be cleaned up automatically.

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
        if not model_name:
            raise ValueError(f"Expected a non-empty value for `model_name` but received {model_name!r}")
        if not adapter:
            raise ValueError(f"Expected a non-empty value for `adapter` but received {adapter!r}")
        extra_headers = {"Accept": "*/*", **(extra_headers or {})}
        return self._delete(
            path_template(
                "/apis/models/v2/workspaces/{workspace}/models/{model_name}/adapters/{adapter}",
                workspace=workspace,
                model_name=model_name,
                adapter=adapter,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
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
        model_name: str,
        *,
        workspace: str | None = None,
        fileset: str,
        finetuning_type: FinetuningType,
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
        Adds an Adapter to the Model

        Args:
          fileset: Location where adapter files are stored - expected format
              {workspace}/{fileset_name}

          finetuning_type: Finetuning types.

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
        if not model_name:
            raise ValueError(f"Expected a non-empty value for `model_name` but received {model_name!r}")
        return await self._post(
            path_template(
                "/apis/models/v2/workspaces/{workspace}/models/{model_name}/adapters",
                workspace=workspace,
                model_name=model_name,
            ),
            body=await async_maybe_transform(
                {
                    "fileset": fileset,
                    "finetuning_type": finetuning_type,
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

    async def update(
        self,
        adapter: str,
        *,
        workspace: str | None = None,
        model_name: str,
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
        Update Adapter deployment or description.

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
        if not model_name:
            raise ValueError(f"Expected a non-empty value for `model_name` but received {model_name!r}")
        if not adapter:
            raise ValueError(f"Expected a non-empty value for `adapter` but received {adapter!r}")
        return await self._patch(
            path_template(
                "/apis/models/v2/workspaces/{workspace}/models/{model_name}/adapters/{adapter}",
                workspace=workspace,
                model_name=model_name,
                adapter=adapter,
            ),
            body=await async_maybe_transform(
                {
                    "description": description,
                    "enabled": enabled,
                    "fileset": fileset,
                },
                adapter_update_params.AdapterUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Adapter,
        )

    async def delete(
        self,
        adapter: str,
        *,
        workspace: str | None = None,
        model_name: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> None:
        """
        Delete Adapter from Model entity.

        Permanently deletes an adapter from a model entity, if it was deployed, it will
        be cleaned up automatically.

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
        if not model_name:
            raise ValueError(f"Expected a non-empty value for `model_name` but received {model_name!r}")
        if not adapter:
            raise ValueError(f"Expected a non-empty value for `adapter` but received {adapter!r}")
        extra_headers = {"Accept": "*/*", **(extra_headers or {})}
        return await self._delete(
            path_template(
                "/apis/models/v2/workspaces/{workspace}/models/{model_name}/adapters/{adapter}",
                workspace=workspace,
                model_name=model_name,
                adapter=adapter,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
        )


class AdaptersResourceWithRawResponse:
    def __init__(self, adapters: AdaptersResource) -> None:
        self._adapters = adapters

        self.create = to_raw_response_wrapper(
            adapters.create,
        )
        self.update = to_raw_response_wrapper(
            adapters.update,
        )
        self.delete = to_raw_response_wrapper(
            adapters.delete,
        )


class AsyncAdaptersResourceWithRawResponse:
    def __init__(self, adapters: AsyncAdaptersResource) -> None:
        self._adapters = adapters

        self.create = async_to_raw_response_wrapper(
            adapters.create,
        )
        self.update = async_to_raw_response_wrapper(
            adapters.update,
        )
        self.delete = async_to_raw_response_wrapper(
            adapters.delete,
        )


class AdaptersResourceWithStreamingResponse:
    def __init__(self, adapters: AdaptersResource) -> None:
        self._adapters = adapters

        self.create = to_streamed_response_wrapper(
            adapters.create,
        )
        self.update = to_streamed_response_wrapper(
            adapters.update,
        )
        self.delete = to_streamed_response_wrapper(
            adapters.delete,
        )


class AsyncAdaptersResourceWithStreamingResponse:
    def __init__(self, adapters: AsyncAdaptersResource) -> None:
        self._adapters = adapters

        self.create = async_to_streamed_response_wrapper(
            adapters.create,
        )
        self.update = async_to_streamed_response_wrapper(
            adapters.update,
        )
        self.delete = async_to_streamed_response_wrapper(
            adapters.delete,
        )
