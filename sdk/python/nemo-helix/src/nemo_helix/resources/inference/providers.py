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

from typing import Dict, Iterable

import httpx

from ..._types import Body, Omit, Query, Headers, NoneType, NotGiven, SequenceNotStr, omit, not_given
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
from ...types.inference import (
    ModelProviderSort,
    ModelProviderStatus,
    provider_list_params,
    provider_create_params,
    provider_update_params,
    provider_update_status_params,
)
from ...types.inference.model_provider import ModelProvider
from ...types.inference.model_provider_sort import ModelProviderSort
from ...types.inference.model_provider_status import ModelProviderStatus
from ...types.inference.served_model_mapping_param import ServedModelMappingParam
from ...types.inference.model_provider_filter_param import ModelProviderFilterParam
from ..._exceptions import ConflictError

__all__ = ["ProvidersResource", "AsyncProvidersResource"]


class ProvidersResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> ProvidersResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return ProvidersResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> ProvidersResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return ProvidersResourceWithStreamingResponse(self)

    def create(
        self,
        *,
        workspace: str | None = None,
        host_url: str,
        name: str,
        api_key_secret_name: str | Omit = omit,
        auth_header_format: str | Omit = omit,
        default_extra_body: Dict[str, object] | Omit = omit,
        default_extra_headers: Dict[str, str] | Omit = omit,
        description: str | Omit = omit,
        enabled_models: SequenceNotStr[str] | Omit = omit,
        model_deployment_id: str | Omit = omit,
        project: str | Omit = omit,
        required_extra_body: Dict[str, object] | Omit = omit,
        required_extra_headers: Dict[str, str] | Omit = omit,
        status: ModelProviderStatus | Omit = omit,
        status_message: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ModelProvider:
        """
        Create a new model provider.

        Args:
          host_url: The network endpoint URL for the model provider

          name: Name of the model provider. Name must start with a lowercase letter, be 2-63
              characters, and use lowercase letters, digits, hyphens, and dots (no consecutive
              hyphens, cannot end with a hyphen).

          api_key_secret_name: Reference to an API key secret stored in the Secrets service. Create the secret
              first via secrets API, then pass the secret name here.

          auth_header_format: Jinja2 template string controlling how the API key secret is sent to the
              upstream. Must contain exactly one variable named `auth_secret`, which is
              substituted with the resolved secret value at request time. Example:
              `'X-Api-Key: {{ auth_secret }}'`. If not set, defaults to
              `'Authorization: Bearer {{ auth_secret }}'`.

          default_extra_body: Default body parameters for inference requests. Can be overridden by user
              requests.

          default_extra_headers: Default headers for inference requests. Can be overridden by user requests.

          description: Optional description of the model provider

          enabled_models: Optional list of specific models to enable from this provider

          model_deployment_id: Optional reference to the ModelDeployment ID if this provider is being
              auto-created for a deployment

          project: The URN of the project associated with this model provider

          required_extra_body: Required body parameters for inference requests. Cannot be overridden by user
              requests.

          required_extra_headers: Required headers for inference requests. Cannot be overridden by user requests.

          status: Status enum for ModelProvider objects.

          status_message: Status message


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
                path_template("/apis/models/v2/workspaces/{workspace}/providers", workspace=workspace),
                body=maybe_transform(
                    {
                        "host_url": host_url,
                        "name": name,
                        "api_key_secret_name": api_key_secret_name,
                        "auth_header_format": auth_header_format,
                        "default_extra_body": default_extra_body,
                        "default_extra_headers": default_extra_headers,
                        "description": description,
                        "enabled_models": enabled_models,
                        "model_deployment_id": model_deployment_id,
                        "project": project,
                        "required_extra_body": required_extra_body,
                        "required_extra_headers": required_extra_headers,
                        "status": status,
                        "status_message": status_message,
                    },
                    provider_create_params.ProviderCreateParams,
                ),
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=ModelProvider,
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
    ) -> ModelProvider:
        """
        Get a model provider by workspace and name.

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
            path_template("/apis/models/v2/workspaces/{workspace}/providers/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ModelProvider,
        )

    def update(
        self,
        name: str,
        *,
        workspace: str | None = None,
        host_url: str,
        api_key_secret_name: str | Omit = omit,
        auth_header_format: str | Omit = omit,
        default_extra_body: Dict[str, object] | Omit = omit,
        default_extra_headers: Dict[str, str] | Omit = omit,
        description: str | Omit = omit,
        enabled_models: SequenceNotStr[str] | Omit = omit,
        model_deployment_id: str | Omit = omit,
        project: str | Omit = omit,
        required_extra_body: Dict[str, object] | Omit = omit,
        required_extra_headers: Dict[str, str] | Omit = omit,
        status: ModelProviderStatus | Omit = omit,
        status_message: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ModelProvider:
        """
        Create or update a model provider.

        Args:
          host_url: The network endpoint URL for the model provider

          api_key_secret_name: Reference to an API key secret stored in the Secrets service. Create the secret
              first via secrets API, then pass the secret name here.

          auth_header_format: Jinja2 template string controlling how the API key secret is sent to the
              upstream. Must contain exactly one variable named `auth_secret`, which is
              substituted with the resolved secret value at request time. Example:
              `'X-Api-Key: {{ auth_secret }}'`. If not set, defaults to
              `'Authorization: Bearer {{ auth_secret }}'`.

          default_extra_body: Default body parameters for inference requests. Can be overridden by user
              requests.

          default_extra_headers: Default headers for inference requests. Can be overridden by user requests.

          description: Optional description of the model provider

          enabled_models: Optional list of specific models to enable from this provider

          model_deployment_id: Optional reference to the ModelDeployment ID if this provider is associated with
              a deployment

          project: The URN of the project associated with this model provider

          required_extra_body: Required body parameters for inference requests. Cannot be overridden by user
              requests.

          required_extra_headers: Required headers for inference requests. Cannot be overridden by user requests.

          status: Status enum for ModelProvider objects.

          status_message: Status message

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
        return self._put(
            path_template("/apis/models/v2/workspaces/{workspace}/providers/{name}", workspace=workspace, name=name),
            body=maybe_transform(
                {
                    "host_url": host_url,
                    "api_key_secret_name": api_key_secret_name,
                    "auth_header_format": auth_header_format,
                    "default_extra_body": default_extra_body,
                    "default_extra_headers": default_extra_headers,
                    "description": description,
                    "enabled_models": enabled_models,
                    "model_deployment_id": model_deployment_id,
                    "project": project,
                    "required_extra_body": required_extra_body,
                    "required_extra_headers": required_extra_headers,
                    "status": status,
                    "status_message": status_message,
                },
                provider_update_params.ProviderUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ModelProvider,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: ModelProviderFilterParam | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: ModelProviderSort | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> SyncDefaultPagination[ModelProvider]:
        """
        List model providers for a specific workspace.

        Args:
          filter: Filter model providers by workspace, project, status, model_deployment_id, name,
              description, host_url, created_at, and updated_at.

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
            path_template("/apis/models/v2/workspaces/{workspace}/providers", workspace=workspace),
            page=SyncDefaultPagination[ModelProvider],
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
                    provider_list_params.ProviderListParams,
                ),
            ),
            model=ModelProvider,
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
        Delete a model provider by workspace and name.

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
            path_template("/apis/models/v2/workspaces/{workspace}/providers/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
        )

    def update_status(
        self,
        name: str,
        *,
        workspace: str | None = None,
        model_deployment_id: str | Omit = omit,
        served_models: Iterable[ServedModelMappingParam] | Omit = omit,
        status: ModelProviderStatus | Omit = omit,
        status_message: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ModelProvider:
        """
        Update status-related fields of a model provider.

        This endpoint supports partial updates for fields managed by Models Controller:

        - model_deployment_id
        - served_models
        - status
        - status_message

        If status is provided without status_message, status_message will be set to
        empty string.

        Args:
          model_deployment_id: Reference to the ModelDeployment ID if this provider is associated with a
              deployment

          served_models: List of models served by this provider with routing information for IGW

          status: Status enum for ModelProvider objects.

          status_message: Status message. If status is provided without status_message, defaults to empty
              string.

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
        return self._put(
            path_template(
                "/apis/models/v2/workspaces/{workspace}/providers/{name}/status", workspace=workspace, name=name
            ),
            body=maybe_transform(
                {
                    "model_deployment_id": model_deployment_id,
                    "served_models": served_models,
                    "status": status,
                    "status_message": status_message,
                },
                provider_update_status_params.ProviderUpdateStatusParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ModelProvider,
        )


class AsyncProvidersResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncProvidersResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncProvidersResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncProvidersResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncProvidersResourceWithStreamingResponse(self)

    async def create(
        self,
        *,
        workspace: str | None = None,
        host_url: str,
        name: str,
        api_key_secret_name: str | Omit = omit,
        auth_header_format: str | Omit = omit,
        default_extra_body: Dict[str, object] | Omit = omit,
        default_extra_headers: Dict[str, str] | Omit = omit,
        description: str | Omit = omit,
        enabled_models: SequenceNotStr[str] | Omit = omit,
        model_deployment_id: str | Omit = omit,
        project: str | Omit = omit,
        required_extra_body: Dict[str, object] | Omit = omit,
        required_extra_headers: Dict[str, str] | Omit = omit,
        status: ModelProviderStatus | Omit = omit,
        status_message: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ModelProvider:
        """
        Create a new model provider.

        Args:
          host_url: The network endpoint URL for the model provider

          name: Name of the model provider. Name must start with a lowercase letter, be 2-63
              characters, and use lowercase letters, digits, hyphens, and dots (no consecutive
              hyphens, cannot end with a hyphen).

          api_key_secret_name: Reference to an API key secret stored in the Secrets service. Create the secret
              first via secrets API, then pass the secret name here.

          auth_header_format: Jinja2 template string controlling how the API key secret is sent to the
              upstream. Must contain exactly one variable named `auth_secret`, which is
              substituted with the resolved secret value at request time. Example:
              `'X-Api-Key: {{ auth_secret }}'`. If not set, defaults to
              `'Authorization: Bearer {{ auth_secret }}'`.

          default_extra_body: Default body parameters for inference requests. Can be overridden by user
              requests.

          default_extra_headers: Default headers for inference requests. Can be overridden by user requests.

          description: Optional description of the model provider

          enabled_models: Optional list of specific models to enable from this provider

          model_deployment_id: Optional reference to the ModelDeployment ID if this provider is being
              auto-created for a deployment

          project: The URN of the project associated with this model provider

          required_extra_body: Required body parameters for inference requests. Cannot be overridden by user
              requests.

          required_extra_headers: Required headers for inference requests. Cannot be overridden by user requests.

          status: Status enum for ModelProvider objects.

          status_message: Status message


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
                path_template("/apis/models/v2/workspaces/{workspace}/providers", workspace=workspace),
                body=await async_maybe_transform(
                    {
                        "host_url": host_url,
                        "name": name,
                        "api_key_secret_name": api_key_secret_name,
                        "auth_header_format": auth_header_format,
                        "default_extra_body": default_extra_body,
                        "default_extra_headers": default_extra_headers,
                        "description": description,
                        "enabled_models": enabled_models,
                        "model_deployment_id": model_deployment_id,
                        "project": project,
                        "required_extra_body": required_extra_body,
                        "required_extra_headers": required_extra_headers,
                        "status": status,
                        "status_message": status_message,
                    },
                    provider_create_params.ProviderCreateParams,
                ),
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=ModelProvider,
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
    ) -> ModelProvider:
        """
        Get a model provider by workspace and name.

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
            path_template("/apis/models/v2/workspaces/{workspace}/providers/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ModelProvider,
        )

    async def update(
        self,
        name: str,
        *,
        workspace: str | None = None,
        host_url: str,
        api_key_secret_name: str | Omit = omit,
        auth_header_format: str | Omit = omit,
        default_extra_body: Dict[str, object] | Omit = omit,
        default_extra_headers: Dict[str, str] | Omit = omit,
        description: str | Omit = omit,
        enabled_models: SequenceNotStr[str] | Omit = omit,
        model_deployment_id: str | Omit = omit,
        project: str | Omit = omit,
        required_extra_body: Dict[str, object] | Omit = omit,
        required_extra_headers: Dict[str, str] | Omit = omit,
        status: ModelProviderStatus | Omit = omit,
        status_message: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ModelProvider:
        """
        Create or update a model provider.

        Args:
          host_url: The network endpoint URL for the model provider

          api_key_secret_name: Reference to an API key secret stored in the Secrets service. Create the secret
              first via secrets API, then pass the secret name here.

          auth_header_format: Jinja2 template string controlling how the API key secret is sent to the
              upstream. Must contain exactly one variable named `auth_secret`, which is
              substituted with the resolved secret value at request time. Example:
              `'X-Api-Key: {{ auth_secret }}'`. If not set, defaults to
              `'Authorization: Bearer {{ auth_secret }}'`.

          default_extra_body: Default body parameters for inference requests. Can be overridden by user
              requests.

          default_extra_headers: Default headers for inference requests. Can be overridden by user requests.

          description: Optional description of the model provider

          enabled_models: Optional list of specific models to enable from this provider

          model_deployment_id: Optional reference to the ModelDeployment ID if this provider is associated with
              a deployment

          project: The URN of the project associated with this model provider

          required_extra_body: Required body parameters for inference requests. Cannot be overridden by user
              requests.

          required_extra_headers: Required headers for inference requests. Cannot be overridden by user requests.

          status: Status enum for ModelProvider objects.

          status_message: Status message

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
        return await self._put(
            path_template("/apis/models/v2/workspaces/{workspace}/providers/{name}", workspace=workspace, name=name),
            body=await async_maybe_transform(
                {
                    "host_url": host_url,
                    "api_key_secret_name": api_key_secret_name,
                    "auth_header_format": auth_header_format,
                    "default_extra_body": default_extra_body,
                    "default_extra_headers": default_extra_headers,
                    "description": description,
                    "enabled_models": enabled_models,
                    "model_deployment_id": model_deployment_id,
                    "project": project,
                    "required_extra_body": required_extra_body,
                    "required_extra_headers": required_extra_headers,
                    "status": status,
                    "status_message": status_message,
                },
                provider_update_params.ProviderUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ModelProvider,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: ModelProviderFilterParam | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: ModelProviderSort | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> AsyncPaginator[ModelProvider, AsyncDefaultPagination[ModelProvider]]:
        """
        List model providers for a specific workspace.

        Args:
          filter: Filter model providers by workspace, project, status, model_deployment_id, name,
              description, host_url, created_at, and updated_at.

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
            path_template("/apis/models/v2/workspaces/{workspace}/providers", workspace=workspace),
            page=AsyncDefaultPagination[ModelProvider],
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
                    provider_list_params.ProviderListParams,
                ),
            ),
            model=ModelProvider,
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
        Delete a model provider by workspace and name.

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
            path_template("/apis/models/v2/workspaces/{workspace}/providers/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
        )

    async def update_status(
        self,
        name: str,
        *,
        workspace: str | None = None,
        model_deployment_id: str | Omit = omit,
        served_models: Iterable[ServedModelMappingParam] | Omit = omit,
        status: ModelProviderStatus | Omit = omit,
        status_message: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ModelProvider:
        """
        Update status-related fields of a model provider.

        This endpoint supports partial updates for fields managed by Models Controller:

        - model_deployment_id
        - served_models
        - status
        - status_message

        If status is provided without status_message, status_message will be set to
        empty string.

        Args:
          model_deployment_id: Reference to the ModelDeployment ID if this provider is associated with a
              deployment

          served_models: List of models served by this provider with routing information for IGW

          status: Status enum for ModelProvider objects.

          status_message: Status message. If status is provided without status_message, defaults to empty
              string.

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
        return await self._put(
            path_template(
                "/apis/models/v2/workspaces/{workspace}/providers/{name}/status", workspace=workspace, name=name
            ),
            body=await async_maybe_transform(
                {
                    "model_deployment_id": model_deployment_id,
                    "served_models": served_models,
                    "status": status,
                    "status_message": status_message,
                },
                provider_update_status_params.ProviderUpdateStatusParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ModelProvider,
        )


class ProvidersResourceWithRawResponse:
    def __init__(self, providers: ProvidersResource) -> None:
        self._providers = providers

        self.create = to_raw_response_wrapper(
            providers.create,
        )
        self.retrieve = to_raw_response_wrapper(
            providers.retrieve,
        )
        self.update = to_raw_response_wrapper(
            providers.update,
        )
        self.list = to_raw_response_wrapper(
            providers.list,
        )
        self.delete = to_raw_response_wrapper(
            providers.delete,
        )
        self.update_status = to_raw_response_wrapper(
            providers.update_status,
        )


class AsyncProvidersResourceWithRawResponse:
    def __init__(self, providers: AsyncProvidersResource) -> None:
        self._providers = providers

        self.create = async_to_raw_response_wrapper(
            providers.create,
        )
        self.retrieve = async_to_raw_response_wrapper(
            providers.retrieve,
        )
        self.update = async_to_raw_response_wrapper(
            providers.update,
        )
        self.list = async_to_raw_response_wrapper(
            providers.list,
        )
        self.delete = async_to_raw_response_wrapper(
            providers.delete,
        )
        self.update_status = async_to_raw_response_wrapper(
            providers.update_status,
        )


class ProvidersResourceWithStreamingResponse:
    def __init__(self, providers: ProvidersResource) -> None:
        self._providers = providers

        self.create = to_streamed_response_wrapper(
            providers.create,
        )
        self.retrieve = to_streamed_response_wrapper(
            providers.retrieve,
        )
        self.update = to_streamed_response_wrapper(
            providers.update,
        )
        self.list = to_streamed_response_wrapper(
            providers.list,
        )
        self.delete = to_streamed_response_wrapper(
            providers.delete,
        )
        self.update_status = to_streamed_response_wrapper(
            providers.update_status,
        )


class AsyncProvidersResourceWithStreamingResponse:
    def __init__(self, providers: AsyncProvidersResource) -> None:
        self._providers = providers

        self.create = async_to_streamed_response_wrapper(
            providers.create,
        )
        self.retrieve = async_to_streamed_response_wrapper(
            providers.retrieve,
        )
        self.update = async_to_streamed_response_wrapper(
            providers.update,
        )
        self.list = async_to_streamed_response_wrapper(
            providers.list,
        )
        self.delete = async_to_streamed_response_wrapper(
            providers.delete,
        )
        self.update_status = async_to_streamed_response_wrapper(
            providers.update_status,
        )
