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

from .models import (
    ModelsResource,
    AsyncModelsResource,
    ModelsResourceWithRawResponse,
    AsyncModelsResourceWithRawResponse,
    ModelsResourceWithStreamingResponse,
    AsyncModelsResourceWithStreamingResponse,
)
from .prompts import (
    PromptsResource,
    AsyncPromptsResource,
    PromptsResourceWithRawResponse,
    AsyncPromptsResourceWithRawResponse,
    PromptsResourceWithStreamingResponse,
    AsyncPromptsResourceWithStreamingResponse,
)
from ..._compat import cached_property
from .providers import (
    ProvidersResource,
    AsyncProvidersResource,
    ProvidersResourceWithRawResponse,
    AsyncProvidersResourceWithRawResponse,
    ProvidersResourceWithStreamingResponse,
    AsyncProvidersResourceWithStreamingResponse,
)
from ..._resource import SyncAPIResource, AsyncAPIResource
from .virtual_models import (
    VirtualModelsResource,
    AsyncVirtualModelsResource,
    VirtualModelsResourceWithRawResponse,
    AsyncVirtualModelsResourceWithRawResponse,
    VirtualModelsResourceWithStreamingResponse,
    AsyncVirtualModelsResourceWithStreamingResponse,
)
from .gateway.gateway import (
    GatewayResource,
    AsyncGatewayResource,
    GatewayResourceWithRawResponse,
    AsyncGatewayResourceWithRawResponse,
    GatewayResourceWithStreamingResponse,
    AsyncGatewayResourceWithStreamingResponse,
)
from .deployments.deployments import (
    DeploymentsResource,
    AsyncDeploymentsResource,
    DeploymentsResourceWithRawResponse,
    AsyncDeploymentsResourceWithRawResponse,
    DeploymentsResourceWithStreamingResponse,
    AsyncDeploymentsResourceWithStreamingResponse,
)
from .deployment_configs.deployment_configs import (
    DeploymentConfigsResource,
    AsyncDeploymentConfigsResource,
    DeploymentConfigsResourceWithRawResponse,
    AsyncDeploymentConfigsResourceWithRawResponse,
    DeploymentConfigsResourceWithStreamingResponse,
    AsyncDeploymentConfigsResourceWithStreamingResponse,
)

__all__ = ["InferenceResource", "AsyncInferenceResource"]


class InferenceResource(SyncAPIResource):
    @cached_property
    def virtual_models(self) -> VirtualModelsResource:
        return VirtualModelsResource(self._client)

    @cached_property
    def models(self) -> ModelsResource:
        return ModelsResource(self._client)

    @cached_property
    def deployment_configs(self) -> DeploymentConfigsResource:
        return DeploymentConfigsResource(self._client)

    @cached_property
    def deployments(self) -> DeploymentsResource:
        return DeploymentsResource(self._client)

    @cached_property
    def providers(self) -> ProvidersResource:
        return ProvidersResource(self._client)

    @cached_property
    def prompts(self) -> PromptsResource:
        return PromptsResource(self._client)

    @cached_property
    def gateway(self) -> GatewayResource:
        return GatewayResource(self._client)

    @cached_property
    def with_raw_response(self) -> InferenceResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return InferenceResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> InferenceResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return InferenceResourceWithStreamingResponse(self)


class AsyncInferenceResource(AsyncAPIResource):
    @cached_property
    def virtual_models(self) -> AsyncVirtualModelsResource:
        return AsyncVirtualModelsResource(self._client)

    @cached_property
    def models(self) -> AsyncModelsResource:
        return AsyncModelsResource(self._client)

    @cached_property
    def deployment_configs(self) -> AsyncDeploymentConfigsResource:
        return AsyncDeploymentConfigsResource(self._client)

    @cached_property
    def deployments(self) -> AsyncDeploymentsResource:
        return AsyncDeploymentsResource(self._client)

    @cached_property
    def providers(self) -> AsyncProvidersResource:
        return AsyncProvidersResource(self._client)

    @cached_property
    def prompts(self) -> AsyncPromptsResource:
        return AsyncPromptsResource(self._client)

    @cached_property
    def gateway(self) -> AsyncGatewayResource:
        return AsyncGatewayResource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncInferenceResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncInferenceResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncInferenceResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncInferenceResourceWithStreamingResponse(self)


class InferenceResourceWithRawResponse:
    def __init__(self, inference: InferenceResource) -> None:
        self._inference = inference

    @cached_property
    def virtual_models(self) -> VirtualModelsResourceWithRawResponse:
        return VirtualModelsResourceWithRawResponse(self._inference.virtual_models)

    @cached_property
    def models(self) -> ModelsResourceWithRawResponse:
        return ModelsResourceWithRawResponse(self._inference.models)

    @cached_property
    def deployment_configs(self) -> DeploymentConfigsResourceWithRawResponse:
        return DeploymentConfigsResourceWithRawResponse(self._inference.deployment_configs)

    @cached_property
    def deployments(self) -> DeploymentsResourceWithRawResponse:
        return DeploymentsResourceWithRawResponse(self._inference.deployments)

    @cached_property
    def providers(self) -> ProvidersResourceWithRawResponse:
        return ProvidersResourceWithRawResponse(self._inference.providers)

    @cached_property
    def prompts(self) -> PromptsResourceWithRawResponse:
        return PromptsResourceWithRawResponse(self._inference.prompts)

    @cached_property
    def gateway(self) -> GatewayResourceWithRawResponse:
        return GatewayResourceWithRawResponse(self._inference.gateway)


class AsyncInferenceResourceWithRawResponse:
    def __init__(self, inference: AsyncInferenceResource) -> None:
        self._inference = inference

    @cached_property
    def virtual_models(self) -> AsyncVirtualModelsResourceWithRawResponse:
        return AsyncVirtualModelsResourceWithRawResponse(self._inference.virtual_models)

    @cached_property
    def models(self) -> AsyncModelsResourceWithRawResponse:
        return AsyncModelsResourceWithRawResponse(self._inference.models)

    @cached_property
    def deployment_configs(self) -> AsyncDeploymentConfigsResourceWithRawResponse:
        return AsyncDeploymentConfigsResourceWithRawResponse(self._inference.deployment_configs)

    @cached_property
    def deployments(self) -> AsyncDeploymentsResourceWithRawResponse:
        return AsyncDeploymentsResourceWithRawResponse(self._inference.deployments)

    @cached_property
    def providers(self) -> AsyncProvidersResourceWithRawResponse:
        return AsyncProvidersResourceWithRawResponse(self._inference.providers)

    @cached_property
    def prompts(self) -> AsyncPromptsResourceWithRawResponse:
        return AsyncPromptsResourceWithRawResponse(self._inference.prompts)

    @cached_property
    def gateway(self) -> AsyncGatewayResourceWithRawResponse:
        return AsyncGatewayResourceWithRawResponse(self._inference.gateway)


class InferenceResourceWithStreamingResponse:
    def __init__(self, inference: InferenceResource) -> None:
        self._inference = inference

    @cached_property
    def virtual_models(self) -> VirtualModelsResourceWithStreamingResponse:
        return VirtualModelsResourceWithStreamingResponse(self._inference.virtual_models)

    @cached_property
    def models(self) -> ModelsResourceWithStreamingResponse:
        return ModelsResourceWithStreamingResponse(self._inference.models)

    @cached_property
    def deployment_configs(self) -> DeploymentConfigsResourceWithStreamingResponse:
        return DeploymentConfigsResourceWithStreamingResponse(self._inference.deployment_configs)

    @cached_property
    def deployments(self) -> DeploymentsResourceWithStreamingResponse:
        return DeploymentsResourceWithStreamingResponse(self._inference.deployments)

    @cached_property
    def providers(self) -> ProvidersResourceWithStreamingResponse:
        return ProvidersResourceWithStreamingResponse(self._inference.providers)

    @cached_property
    def prompts(self) -> PromptsResourceWithStreamingResponse:
        return PromptsResourceWithStreamingResponse(self._inference.prompts)

    @cached_property
    def gateway(self) -> GatewayResourceWithStreamingResponse:
        return GatewayResourceWithStreamingResponse(self._inference.gateway)


class AsyncInferenceResourceWithStreamingResponse:
    def __init__(self, inference: AsyncInferenceResource) -> None:
        self._inference = inference

    @cached_property
    def virtual_models(self) -> AsyncVirtualModelsResourceWithStreamingResponse:
        return AsyncVirtualModelsResourceWithStreamingResponse(self._inference.virtual_models)

    @cached_property
    def models(self) -> AsyncModelsResourceWithStreamingResponse:
        return AsyncModelsResourceWithStreamingResponse(self._inference.models)

    @cached_property
    def deployment_configs(self) -> AsyncDeploymentConfigsResourceWithStreamingResponse:
        return AsyncDeploymentConfigsResourceWithStreamingResponse(self._inference.deployment_configs)

    @cached_property
    def deployments(self) -> AsyncDeploymentsResourceWithStreamingResponse:
        return AsyncDeploymentsResourceWithStreamingResponse(self._inference.deployments)

    @cached_property
    def providers(self) -> AsyncProvidersResourceWithStreamingResponse:
        return AsyncProvidersResourceWithStreamingResponse(self._inference.providers)

    @cached_property
    def prompts(self) -> AsyncPromptsResourceWithStreamingResponse:
        return AsyncPromptsResourceWithStreamingResponse(self._inference.prompts)

    @cached_property
    def gateway(self) -> AsyncGatewayResourceWithStreamingResponse:
        return AsyncGatewayResourceWithStreamingResponse(self._inference.gateway)
