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

from typing import Dict, Optional

import httpx

from ..._types import Body, Omit, Query, Headers, NoneType, NotGiven, SequenceNotStr, omit, not_given
from ..._utils import path_template, maybe_transform, async_maybe_transform
from .adapters import (
    AdaptersResource,
    AsyncAdaptersResource,
    AdaptersResourceWithRawResponse,
    AsyncAdaptersResourceWithRawResponse,
    AdaptersResourceWithStreamingResponse,
    AsyncAdaptersResourceWithStreamingResponse,
)
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
from ...types.models import (
    ModelEntitySortField,
    model_list_params,
    model_create_params,
    model_update_params,
    model_retrieve_params,
)
from ...types.models.model_entity import ModelEntity
from ...types.shared.backend_format import BackendFormat
from ...types.shared.finetuning_type import FinetuningType
from ...types.shared_params.model_spec import ModelSpec
from ...types.shared_params.prompt_data import PromptData
from ...types.models.model_entity_sort_field import ModelEntitySortField
from ...types.shared_params.api_endpoint_data import APIEndpointData
from ...types.models.model_entity_filter_param import ModelEntityFilterParam
from ..._exceptions import ConflictError

__all__ = ["ModelsResource", "AsyncModelsResource"]


class ModelsResource(SyncAPIResource):
    @cached_property
    def adapters(self) -> AdaptersResource:
        return AdaptersResource(self._client)

    @cached_property
    def with_raw_response(self) -> ModelsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return ModelsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> ModelsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return ModelsResourceWithStreamingResponse(self)

    def create(
        self,
        *,
        workspace: str | None = None,
        name: str,
        api_endpoint: APIEndpointData | Omit = omit,
        backend_format: Optional[BackendFormat] | Omit = omit,
        base_model: str | Omit = omit,
        custom_fields: Dict[str, object] | Omit = omit,
        description: str | Omit = omit,
        fileset: str | Omit = omit,
        finetuning_type: FinetuningType | Omit = omit,
        model_providers: SequenceNotStr[str] | Omit = omit,
        ownership: Dict[str, object] | Omit = omit,
        project: str | Omit = omit,
        prompt: PromptData | Omit = omit,
        spec: ModelSpec | Omit = omit,
        trust_remote_code: bool | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ModelEntity:
        """
        Create a new model entity.

        This endpoint creates a new Model Entity in the Models service database. The
        Model Entity will be registered for use within the platform.

        Args:
          name: Name of the model entity. Name must start with a lowercase letter, be 2-63
              characters, and use lowercase letters, digits, hyphens, and dots (no consecutive
              hyphens, cannot end with a hyphen).

          api_endpoint: Data about an inference endpoint.

          backend_format: Inference backend API wire formats understood by IGW and middleware plugins.

          base_model: Link to another model which is used as a base for the current model

          custom_fields: Custom fields for additional metadata

          description: Optional description of the model

          fileset: A set of checkpoint files, configs, and other auxiliary info associated with
              this model - expected format {workspace}/{fileset_name}

          finetuning_type: Finetuning types.

          model_providers: List of ModelProvider workspace/name resource names that provide inference for
              this Model Entity

          ownership: Ownership information for the model

          project: The URN of the project associated with this model entity

          prompt: Configuration for prompt engineering.

          spec: Detailed specification for a model.

          trust_remote_code: Whether to trust remote code for the checkpoint. Some models without support in
              certain libraries such as Transformers require additional custom Python code to
              execute. Due to security ramifications of running arbitrary code, this can only
              be set to true on one of the following conditions: (1) the model's fileset's
              source is pre-approved in the platform config, or (2) the user creating this
              model is an administrator.


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
                path_template("/apis/models/v2/workspaces/{workspace}/models", workspace=workspace),
                body=maybe_transform(
                    {
                        "name": name,
                        "api_endpoint": api_endpoint,
                        "backend_format": backend_format,
                        "base_model": base_model,
                        "custom_fields": custom_fields,
                        "description": description,
                        "fileset": fileset,
                        "finetuning_type": finetuning_type,
                        "model_providers": model_providers,
                        "ownership": ownership,
                        "project": project,
                        "prompt": prompt,
                        "spec": spec,
                        "trust_remote_code": trust_remote_code,
                    },
                    model_create_params.ModelCreateParams,
                ),
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=ModelEntity,
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
        verbose: bool | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ModelEntity:
        """
        Get Model by Workspace and Name.

        Returns the details of a specific model entity identified by its workspace and
        name.

        Args:
          verbose: Whether to include full spec details

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
            path_template("/apis/models/v2/workspaces/{workspace}/models/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform({"verbose": verbose}, model_retrieve_params.ModelRetrieveParams),
            ),
            cast_to=ModelEntity,
        )

    def update(
        self,
        name: str,
        *,
        workspace: str | None = None,
        verbose: bool | Omit = omit,
        api_endpoint: APIEndpointData | Omit = omit,
        backend_format: Optional[BackendFormat] | Omit = omit,
        base_model: str | Omit = omit,
        custom_fields: Dict[str, object] | Omit = omit,
        description: str | Omit = omit,
        fileset: str | Omit = omit,
        finetuning_type: FinetuningType | Omit = omit,
        model_providers: SequenceNotStr[str] | Omit = omit,
        ownership: Dict[str, object] | Omit = omit,
        prompt: PromptData | Omit = omit,
        spec: ModelSpec | Omit = omit,
        trust_remote_code: bool | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ModelEntity:
        """Update Model metadata.

        Updates the metadata of an existing model entity.

        If the request body has an
        empty field, the old value is kept.

        Args:
          verbose: Whether to include full spec details

          api_endpoint: Data about an inference endpoint.

          backend_format: Inference backend API wire formats understood by IGW and middleware plugins.

          base_model: Link to another model which is used as a base for the current model

          custom_fields: Custom fields for additional metadata

          description: Optional description of the model

          fileset: A set of checkpoint files, configs, and other auxiliary info associated with
              this model - expected format {workspace}/{fileset_name}

          finetuning_type: Finetuning types.

          model_providers: List of ModelProvider workspace/name resource names that provide inference for
              this Model Entity

          ownership: Ownership information for the model

          prompt: Configuration for prompt engineering.

          spec: Detailed specification for a model.

          trust_remote_code: Whether to trust remote code for the checkpoint. Some models without support in
              certain libraries such as Transformers require additional custom Python code to
              execute. Due to security ramifications of running arbitrary code, this can only
              be set to true on one of the following conditions: (1) the model's fileset's
              source is pre-approved in the platform config, or (2) the user creating this
              model is an administrator.

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
            path_template("/apis/models/v2/workspaces/{workspace}/models/{name}", workspace=workspace, name=name),
            body=maybe_transform(
                {
                    "api_endpoint": api_endpoint,
                    "backend_format": backend_format,
                    "base_model": base_model,
                    "custom_fields": custom_fields,
                    "description": description,
                    "fileset": fileset,
                    "finetuning_type": finetuning_type,
                    "model_providers": model_providers,
                    "ownership": ownership,
                    "prompt": prompt,
                    "spec": spec,
                    "trust_remote_code": trust_remote_code,
                },
                model_update_params.ModelUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform({"verbose": verbose}, model_update_params.ModelUpdateParams),
            ),
            cast_to=ModelEntity,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: ModelEntityFilterParam | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: ModelEntitySortField | Omit = omit,
        verbose: bool | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> SyncDefaultPagination[ModelEntity]:
        """
        List Models endpoint with filtering, pagination, and sorting.

        Supports filter parameters for various criteria (including peft, custom fields),
        pagination (page, page_size), sorting, and workspace filtering via query
        parameter.

        Args:
          filter: Filter models by name, project, workspace, base_model, adapters,
              finetuning_type, prompt, lora_enabled, description, created_at, and updated_at.

          page: Page number.

          page_size: Page size.

          sort: The field to sort by. To sort in decreasing order, use `-` in front of the field
              name.

          verbose: Whether to include full spec details

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
            path_template("/apis/models/v2/workspaces/{workspace}/models", workspace=workspace),
            page=SyncDefaultPagination[ModelEntity],
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
                        "verbose": verbose,
                    },
                    model_list_params.ModelListParams,
                ),
            ),
            model=ModelEntity,
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
        Delete Model entity.

        Permanently deletes a model entity from the platform.

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
            path_template("/apis/models/v2/workspaces/{workspace}/models/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
        )


class AsyncModelsResource(AsyncAPIResource):
    @cached_property
    def adapters(self) -> AsyncAdaptersResource:
        return AsyncAdaptersResource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncModelsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncModelsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncModelsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncModelsResourceWithStreamingResponse(self)

    async def create(
        self,
        *,
        workspace: str | None = None,
        name: str,
        api_endpoint: APIEndpointData | Omit = omit,
        backend_format: Optional[BackendFormat] | Omit = omit,
        base_model: str | Omit = omit,
        custom_fields: Dict[str, object] | Omit = omit,
        description: str | Omit = omit,
        fileset: str | Omit = omit,
        finetuning_type: FinetuningType | Omit = omit,
        model_providers: SequenceNotStr[str] | Omit = omit,
        ownership: Dict[str, object] | Omit = omit,
        project: str | Omit = omit,
        prompt: PromptData | Omit = omit,
        spec: ModelSpec | Omit = omit,
        trust_remote_code: bool | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ModelEntity:
        """
        Create a new model entity.

        This endpoint creates a new Model Entity in the Models service database. The
        Model Entity will be registered for use within the platform.

        Args:
          name: Name of the model entity. Name must start with a lowercase letter, be 2-63
              characters, and use lowercase letters, digits, hyphens, and dots (no consecutive
              hyphens, cannot end with a hyphen).

          api_endpoint: Data about an inference endpoint.

          backend_format: Inference backend API wire formats understood by IGW and middleware plugins.

          base_model: Link to another model which is used as a base for the current model

          custom_fields: Custom fields for additional metadata

          description: Optional description of the model

          fileset: A set of checkpoint files, configs, and other auxiliary info associated with
              this model - expected format {workspace}/{fileset_name}

          finetuning_type: Finetuning types.

          model_providers: List of ModelProvider workspace/name resource names that provide inference for
              this Model Entity

          ownership: Ownership information for the model

          project: The URN of the project associated with this model entity

          prompt: Configuration for prompt engineering.

          spec: Detailed specification for a model.

          trust_remote_code: Whether to trust remote code for the checkpoint. Some models without support in
              certain libraries such as Transformers require additional custom Python code to
              execute. Due to security ramifications of running arbitrary code, this can only
              be set to true on one of the following conditions: (1) the model's fileset's
              source is pre-approved in the platform config, or (2) the user creating this
              model is an administrator.


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
                path_template("/apis/models/v2/workspaces/{workspace}/models", workspace=workspace),
                body=await async_maybe_transform(
                    {
                        "name": name,
                        "api_endpoint": api_endpoint,
                        "backend_format": backend_format,
                        "base_model": base_model,
                        "custom_fields": custom_fields,
                        "description": description,
                        "fileset": fileset,
                        "finetuning_type": finetuning_type,
                        "model_providers": model_providers,
                        "ownership": ownership,
                        "project": project,
                        "prompt": prompt,
                        "spec": spec,
                        "trust_remote_code": trust_remote_code,
                    },
                    model_create_params.ModelCreateParams,
                ),
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=ModelEntity,
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
        verbose: bool | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ModelEntity:
        """
        Get Model by Workspace and Name.

        Returns the details of a specific model entity identified by its workspace and
        name.

        Args:
          verbose: Whether to include full spec details

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
            path_template("/apis/models/v2/workspaces/{workspace}/models/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform({"verbose": verbose}, model_retrieve_params.ModelRetrieveParams),
            ),
            cast_to=ModelEntity,
        )

    async def update(
        self,
        name: str,
        *,
        workspace: str | None = None,
        verbose: bool | Omit = omit,
        api_endpoint: APIEndpointData | Omit = omit,
        backend_format: Optional[BackendFormat] | Omit = omit,
        base_model: str | Omit = omit,
        custom_fields: Dict[str, object] | Omit = omit,
        description: str | Omit = omit,
        fileset: str | Omit = omit,
        finetuning_type: FinetuningType | Omit = omit,
        model_providers: SequenceNotStr[str] | Omit = omit,
        ownership: Dict[str, object] | Omit = omit,
        prompt: PromptData | Omit = omit,
        spec: ModelSpec | Omit = omit,
        trust_remote_code: bool | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ModelEntity:
        """Update Model metadata.

        Updates the metadata of an existing model entity.

        If the request body has an
        empty field, the old value is kept.

        Args:
          verbose: Whether to include full spec details

          api_endpoint: Data about an inference endpoint.

          backend_format: Inference backend API wire formats understood by IGW and middleware plugins.

          base_model: Link to another model which is used as a base for the current model

          custom_fields: Custom fields for additional metadata

          description: Optional description of the model

          fileset: A set of checkpoint files, configs, and other auxiliary info associated with
              this model - expected format {workspace}/{fileset_name}

          finetuning_type: Finetuning types.

          model_providers: List of ModelProvider workspace/name resource names that provide inference for
              this Model Entity

          ownership: Ownership information for the model

          prompt: Configuration for prompt engineering.

          spec: Detailed specification for a model.

          trust_remote_code: Whether to trust remote code for the checkpoint. Some models without support in
              certain libraries such as Transformers require additional custom Python code to
              execute. Due to security ramifications of running arbitrary code, this can only
              be set to true on one of the following conditions: (1) the model's fileset's
              source is pre-approved in the platform config, or (2) the user creating this
              model is an administrator.

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
            path_template("/apis/models/v2/workspaces/{workspace}/models/{name}", workspace=workspace, name=name),
            body=await async_maybe_transform(
                {
                    "api_endpoint": api_endpoint,
                    "backend_format": backend_format,
                    "base_model": base_model,
                    "custom_fields": custom_fields,
                    "description": description,
                    "fileset": fileset,
                    "finetuning_type": finetuning_type,
                    "model_providers": model_providers,
                    "ownership": ownership,
                    "prompt": prompt,
                    "spec": spec,
                    "trust_remote_code": trust_remote_code,
                },
                model_update_params.ModelUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform({"verbose": verbose}, model_update_params.ModelUpdateParams),
            ),
            cast_to=ModelEntity,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: ModelEntityFilterParam | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: ModelEntitySortField | Omit = omit,
        verbose: bool | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> AsyncPaginator[ModelEntity, AsyncDefaultPagination[ModelEntity]]:
        """
        List Models endpoint with filtering, pagination, and sorting.

        Supports filter parameters for various criteria (including peft, custom fields),
        pagination (page, page_size), sorting, and workspace filtering via query
        parameter.

        Args:
          filter: Filter models by name, project, workspace, base_model, adapters,
              finetuning_type, prompt, lora_enabled, description, created_at, and updated_at.

          page: Page number.

          page_size: Page size.

          sort: The field to sort by. To sort in decreasing order, use `-` in front of the field
              name.

          verbose: Whether to include full spec details

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
            path_template("/apis/models/v2/workspaces/{workspace}/models", workspace=workspace),
            page=AsyncDefaultPagination[ModelEntity],
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
                        "verbose": verbose,
                    },
                    model_list_params.ModelListParams,
                ),
            ),
            model=ModelEntity,
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
        Delete Model entity.

        Permanently deletes a model entity from the platform.

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
            path_template("/apis/models/v2/workspaces/{workspace}/models/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
        )


class ModelsResourceWithRawResponse:
    def __init__(self, models: ModelsResource) -> None:
        self._models = models

        self.create = to_raw_response_wrapper(
            models.create,
        )
        self.retrieve = to_raw_response_wrapper(
            models.retrieve,
        )
        self.update = to_raw_response_wrapper(
            models.update,
        )
        self.list = to_raw_response_wrapper(
            models.list,
        )
        self.delete = to_raw_response_wrapper(
            models.delete,
        )

    @cached_property
    def adapters(self) -> AdaptersResourceWithRawResponse:
        return AdaptersResourceWithRawResponse(self._models.adapters)


class AsyncModelsResourceWithRawResponse:
    def __init__(self, models: AsyncModelsResource) -> None:
        self._models = models

        self.create = async_to_raw_response_wrapper(
            models.create,
        )
        self.retrieve = async_to_raw_response_wrapper(
            models.retrieve,
        )
        self.update = async_to_raw_response_wrapper(
            models.update,
        )
        self.list = async_to_raw_response_wrapper(
            models.list,
        )
        self.delete = async_to_raw_response_wrapper(
            models.delete,
        )

    @cached_property
    def adapters(self) -> AsyncAdaptersResourceWithRawResponse:
        return AsyncAdaptersResourceWithRawResponse(self._models.adapters)


class ModelsResourceWithStreamingResponse:
    def __init__(self, models: ModelsResource) -> None:
        self._models = models

        self.create = to_streamed_response_wrapper(
            models.create,
        )
        self.retrieve = to_streamed_response_wrapper(
            models.retrieve,
        )
        self.update = to_streamed_response_wrapper(
            models.update,
        )
        self.list = to_streamed_response_wrapper(
            models.list,
        )
        self.delete = to_streamed_response_wrapper(
            models.delete,
        )

    @cached_property
    def adapters(self) -> AdaptersResourceWithStreamingResponse:
        return AdaptersResourceWithStreamingResponse(self._models.adapters)


class AsyncModelsResourceWithStreamingResponse:
    def __init__(self, models: AsyncModelsResource) -> None:
        self._models = models

        self.create = async_to_streamed_response_wrapper(
            models.create,
        )
        self.retrieve = async_to_streamed_response_wrapper(
            models.retrieve,
        )
        self.update = async_to_streamed_response_wrapper(
            models.update,
        )
        self.list = async_to_streamed_response_wrapper(
            models.list,
        )
        self.delete = async_to_streamed_response_wrapper(
            models.delete,
        )

    @cached_property
    def adapters(self) -> AsyncAdaptersResourceWithStreamingResponse:
        return AsyncAdaptersResourceWithStreamingResponse(self._models.adapters)
