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

from .versions import (
    VersionsResource,
    AsyncVersionsResource,
    VersionsResourceWithRawResponse,
    AsyncVersionsResourceWithRawResponse,
    VersionsResourceWithStreamingResponse,
    AsyncVersionsResourceWithStreamingResponse,
)
from ...._types import Body, Omit, Query, Headers, NoneType, NotGiven, omit, not_given
from ...._utils import path_template, maybe_transform, async_maybe_transform
from ...._compat import cached_property
from ...._resource import SyncAPIResource, AsyncAPIResource
from ...._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from ....pagination import SyncDefaultPagination, AsyncDefaultPagination
from ...._base_client import AsyncPaginator, make_request_options
from ....types.inference import (
    Engine,
    deployment_config_list_params,
    deployment_config_create_params,
    deployment_config_update_params,
)
from ....types.inference.engine import Engine
from ....types.inference.model_deployment_config import ModelDeploymentConfig
from ....types.inference.container_executor_config_param import ContainerExecutorConfigParam
from ....types.inference.model_deployment_config_filter_param import ModelDeploymentConfigFilterParam
from ....types.inference.model_deployment_config_model_spec_param import ModelDeploymentConfigModelSpecParam
from ...._exceptions import ConflictError

__all__ = ["DeploymentConfigsResource", "AsyncDeploymentConfigsResource"]


class DeploymentConfigsResource(SyncAPIResource):
    @cached_property
    def versions(self) -> VersionsResource:
        return VersionsResource(self._client)

    @cached_property
    def with_raw_response(self) -> DeploymentConfigsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return DeploymentConfigsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> DeploymentConfigsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return DeploymentConfigsResourceWithStreamingResponse(self)

    def create(
        self,
        *,
        workspace: str | None = None,
        engine: Engine,
        executor_config: ContainerExecutorConfigParam,
        model_spec: ModelDeploymentConfigModelSpecParam,
        name: str,
        description: str | Omit = omit,
        model_entity_id: str | Omit = omit,
        project: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ModelDeploymentConfig:
        """
        Create a new ModelDeploymentConfig (version 1).

        Args:
          engine: Inference engine selecting the compiler path for a deployment.

              The engine determines what command, image, and env a deployment compiles to. The
              fields a compiler consumes are not engine-specific; engines take the same inputs
              (model_spec + executor_config) and differ in what they do with them.

          executor_config: Compute + container settings shared by the docker and k8s executors.

              Both the docker and k8s executors run containers and share this shape. A future
              non-container executor (e.g. subprocess) would warrant turning `executor_config`
              into a discriminated union.

          model_spec: What model to serve and how -- independent of the executor it runs on.

              Executor-invariant facts about the model. The compiler resolves the weight
              source per engine; serving fields override the model entity spec when set.

          name: Name of the deployment configuration. Name must start with a lowercase letter,
              be 2-63 characters, and use lowercase letters, digits, hyphens, and dots (no
              consecutive hyphens, cannot end with a hyphen).

          description: Optional description of the deployment configuration

          model_entity_id: Optional reference to the base model entity ID for this deployment

          project: The URN of the project associated with this deployment configuration


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
                path_template("/apis/models/v2/workspaces/{workspace}/deployment-configs", workspace=workspace),
                body=maybe_transform(
                    {
                        "engine": engine,
                        "executor_config": executor_config,
                        "model_spec": model_spec,
                        "name": name,
                        "description": description,
                        "model_entity_id": model_entity_id,
                        "project": project,
                    },
                    deployment_config_create_params.DeploymentConfigCreateParams,
                ),
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=ModelDeploymentConfig,
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
    ) -> ModelDeploymentConfig:
        """
        Get the latest version of a ModelDeploymentConfig.

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
            path_template(
                "/apis/models/v2/workspaces/{workspace}/deployment-configs/{name}", workspace=workspace, name=name
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ModelDeploymentConfig,
        )

    def update(
        self,
        name: str,
        *,
        workspace: str | None = None,
        engine: Engine,
        executor_config: ContainerExecutorConfigParam,
        model_spec: ModelDeploymentConfigModelSpecParam,
        description: str | Omit = omit,
        model_entity_id: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ModelDeploymentConfig:
        """
        Update a ModelDeploymentConfig (creates a new immutable version).

        Args:
          engine: Inference engine selecting the compiler path for a deployment.

              The engine determines what command, image, and env a deployment compiles to. The
              fields a compiler consumes are not engine-specific; engines take the same inputs
              (model_spec + executor_config) and differ in what they do with them.

          executor_config: Compute + container settings shared by the docker and k8s executors.

              Both the docker and k8s executors run containers and share this shape. A future
              non-container executor (e.g. subprocess) would warrant turning `executor_config`
              into a discriminated union.

          model_spec: What model to serve and how -- independent of the executor it runs on.

              Executor-invariant facts about the model. The compiler resolves the weight
              source per engine; serving fields override the model entity spec when set.

          description: Optional description of the deployment configuration

          model_entity_id: Optional reference to the base model entity ID for this deployment

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
                "/apis/models/v2/workspaces/{workspace}/deployment-configs/{name}", workspace=workspace, name=name
            ),
            body=maybe_transform(
                {
                    "engine": engine,
                    "executor_config": executor_config,
                    "model_spec": model_spec,
                    "description": description,
                    "model_entity_id": model_entity_id,
                },
                deployment_config_update_params.DeploymentConfigUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ModelDeploymentConfig,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: ModelDeploymentConfigFilterParam | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> SyncDefaultPagination[ModelDeploymentConfig]:
        """List ModelDeploymentConfigs for a specific workspace.

        Returns only the latest
        version of each config.

        Args:
          filter: Filter deployment configs by workspace, project, model_entity_id, name,
              description, created_at, and updated_at.

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
            path_template("/apis/models/v2/workspaces/{workspace}/deployment-configs", workspace=workspace),
            page=SyncDefaultPagination[ModelDeploymentConfig],
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
                    deployment_config_list_params.DeploymentConfigListParams,
                ),
            ),
            model=ModelDeploymentConfig,
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
        Delete all versions of a ModelDeploymentConfig.

        This operation will fail with 409 Conflict if any ModelDeployments currently
        reference this config and are not in DELETED status. Delete or wait for
        dependent deployments to reach DELETED status before deleting the config.

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
            path_template(
                "/apis/models/v2/workspaces/{workspace}/deployment-configs/{name}", workspace=workspace, name=name
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
        )


class AsyncDeploymentConfigsResource(AsyncAPIResource):
    @cached_property
    def versions(self) -> AsyncVersionsResource:
        return AsyncVersionsResource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncDeploymentConfigsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncDeploymentConfigsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncDeploymentConfigsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncDeploymentConfigsResourceWithStreamingResponse(self)

    async def create(
        self,
        *,
        workspace: str | None = None,
        engine: Engine,
        executor_config: ContainerExecutorConfigParam,
        model_spec: ModelDeploymentConfigModelSpecParam,
        name: str,
        description: str | Omit = omit,
        model_entity_id: str | Omit = omit,
        project: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ModelDeploymentConfig:
        """
        Create a new ModelDeploymentConfig (version 1).

        Args:
          engine: Inference engine selecting the compiler path for a deployment.

              The engine determines what command, image, and env a deployment compiles to. The
              fields a compiler consumes are not engine-specific; engines take the same inputs
              (model_spec + executor_config) and differ in what they do with them.

          executor_config: Compute + container settings shared by the docker and k8s executors.

              Both the docker and k8s executors run containers and share this shape. A future
              non-container executor (e.g. subprocess) would warrant turning `executor_config`
              into a discriminated union.

          model_spec: What model to serve and how -- independent of the executor it runs on.

              Executor-invariant facts about the model. The compiler resolves the weight
              source per engine; serving fields override the model entity spec when set.

          name: Name of the deployment configuration. Name must start with a lowercase letter,
              be 2-63 characters, and use lowercase letters, digits, hyphens, and dots (no
              consecutive hyphens, cannot end with a hyphen).

          description: Optional description of the deployment configuration

          model_entity_id: Optional reference to the base model entity ID for this deployment

          project: The URN of the project associated with this deployment configuration


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
                path_template("/apis/models/v2/workspaces/{workspace}/deployment-configs", workspace=workspace),
                body=await async_maybe_transform(
                    {
                        "engine": engine,
                        "executor_config": executor_config,
                        "model_spec": model_spec,
                        "name": name,
                        "description": description,
                        "model_entity_id": model_entity_id,
                        "project": project,
                    },
                    deployment_config_create_params.DeploymentConfigCreateParams,
                ),
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=ModelDeploymentConfig,
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
    ) -> ModelDeploymentConfig:
        """
        Get the latest version of a ModelDeploymentConfig.

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
            path_template(
                "/apis/models/v2/workspaces/{workspace}/deployment-configs/{name}", workspace=workspace, name=name
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ModelDeploymentConfig,
        )

    async def update(
        self,
        name: str,
        *,
        workspace: str | None = None,
        engine: Engine,
        executor_config: ContainerExecutorConfigParam,
        model_spec: ModelDeploymentConfigModelSpecParam,
        description: str | Omit = omit,
        model_entity_id: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ModelDeploymentConfig:
        """
        Update a ModelDeploymentConfig (creates a new immutable version).

        Args:
          engine: Inference engine selecting the compiler path for a deployment.

              The engine determines what command, image, and env a deployment compiles to. The
              fields a compiler consumes are not engine-specific; engines take the same inputs
              (model_spec + executor_config) and differ in what they do with them.

          executor_config: Compute + container settings shared by the docker and k8s executors.

              Both the docker and k8s executors run containers and share this shape. A future
              non-container executor (e.g. subprocess) would warrant turning `executor_config`
              into a discriminated union.

          model_spec: What model to serve and how -- independent of the executor it runs on.

              Executor-invariant facts about the model. The compiler resolves the weight
              source per engine; serving fields override the model entity spec when set.

          description: Optional description of the deployment configuration

          model_entity_id: Optional reference to the base model entity ID for this deployment

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
                "/apis/models/v2/workspaces/{workspace}/deployment-configs/{name}", workspace=workspace, name=name
            ),
            body=await async_maybe_transform(
                {
                    "engine": engine,
                    "executor_config": executor_config,
                    "model_spec": model_spec,
                    "description": description,
                    "model_entity_id": model_entity_id,
                },
                deployment_config_update_params.DeploymentConfigUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ModelDeploymentConfig,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: ModelDeploymentConfigFilterParam | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> AsyncPaginator[ModelDeploymentConfig, AsyncDefaultPagination[ModelDeploymentConfig]]:
        """List ModelDeploymentConfigs for a specific workspace.

        Returns only the latest
        version of each config.

        Args:
          filter: Filter deployment configs by workspace, project, model_entity_id, name,
              description, created_at, and updated_at.

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
            path_template("/apis/models/v2/workspaces/{workspace}/deployment-configs", workspace=workspace),
            page=AsyncDefaultPagination[ModelDeploymentConfig],
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
                    deployment_config_list_params.DeploymentConfigListParams,
                ),
            ),
            model=ModelDeploymentConfig,
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
        Delete all versions of a ModelDeploymentConfig.

        This operation will fail with 409 Conflict if any ModelDeployments currently
        reference this config and are not in DELETED status. Delete or wait for
        dependent deployments to reach DELETED status before deleting the config.

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
            path_template(
                "/apis/models/v2/workspaces/{workspace}/deployment-configs/{name}", workspace=workspace, name=name
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
        )


class DeploymentConfigsResourceWithRawResponse:
    def __init__(self, deployment_configs: DeploymentConfigsResource) -> None:
        self._deployment_configs = deployment_configs

        self.create = to_raw_response_wrapper(
            deployment_configs.create,
        )
        self.retrieve = to_raw_response_wrapper(
            deployment_configs.retrieve,
        )
        self.update = to_raw_response_wrapper(
            deployment_configs.update,
        )
        self.list = to_raw_response_wrapper(
            deployment_configs.list,
        )
        self.delete = to_raw_response_wrapper(
            deployment_configs.delete,
        )

    @cached_property
    def versions(self) -> VersionsResourceWithRawResponse:
        return VersionsResourceWithRawResponse(self._deployment_configs.versions)


class AsyncDeploymentConfigsResourceWithRawResponse:
    def __init__(self, deployment_configs: AsyncDeploymentConfigsResource) -> None:
        self._deployment_configs = deployment_configs

        self.create = async_to_raw_response_wrapper(
            deployment_configs.create,
        )
        self.retrieve = async_to_raw_response_wrapper(
            deployment_configs.retrieve,
        )
        self.update = async_to_raw_response_wrapper(
            deployment_configs.update,
        )
        self.list = async_to_raw_response_wrapper(
            deployment_configs.list,
        )
        self.delete = async_to_raw_response_wrapper(
            deployment_configs.delete,
        )

    @cached_property
    def versions(self) -> AsyncVersionsResourceWithRawResponse:
        return AsyncVersionsResourceWithRawResponse(self._deployment_configs.versions)


class DeploymentConfigsResourceWithStreamingResponse:
    def __init__(self, deployment_configs: DeploymentConfigsResource) -> None:
        self._deployment_configs = deployment_configs

        self.create = to_streamed_response_wrapper(
            deployment_configs.create,
        )
        self.retrieve = to_streamed_response_wrapper(
            deployment_configs.retrieve,
        )
        self.update = to_streamed_response_wrapper(
            deployment_configs.update,
        )
        self.list = to_streamed_response_wrapper(
            deployment_configs.list,
        )
        self.delete = to_streamed_response_wrapper(
            deployment_configs.delete,
        )

    @cached_property
    def versions(self) -> VersionsResourceWithStreamingResponse:
        return VersionsResourceWithStreamingResponse(self._deployment_configs.versions)


class AsyncDeploymentConfigsResourceWithStreamingResponse:
    def __init__(self, deployment_configs: AsyncDeploymentConfigsResource) -> None:
        self._deployment_configs = deployment_configs

        self.create = async_to_streamed_response_wrapper(
            deployment_configs.create,
        )
        self.retrieve = async_to_streamed_response_wrapper(
            deployment_configs.retrieve,
        )
        self.update = async_to_streamed_response_wrapper(
            deployment_configs.update,
        )
        self.list = async_to_streamed_response_wrapper(
            deployment_configs.list,
        )
        self.delete = async_to_streamed_response_wrapper(
            deployment_configs.delete,
        )

    @cached_property
    def versions(self) -> AsyncVersionsResourceWithStreamingResponse:
        return AsyncVersionsResourceWithStreamingResponse(self._deployment_configs.versions)
