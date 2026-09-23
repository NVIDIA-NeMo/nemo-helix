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

from .model import (
    ModelResource,
    AsyncModelResource,
    ModelResourceWithRawResponse,
    AsyncModelResourceWithRawResponse,
    ModelResourceWithStreamingResponse,
    AsyncModelResourceWithStreamingResponse,
)
from .provider import (
    ProviderResource,
    AsyncProviderResource,
    ProviderResourceWithRawResponse,
    AsyncProviderResourceWithRawResponse,
    ProviderResourceWithStreamingResponse,
    AsyncProviderResourceWithStreamingResponse,
)
from ...._compat import cached_property
from ...._resource import SyncAPIResource, AsyncAPIResource
from .openai.openai import (
    OpenAIResource,
    AsyncOpenAIResource,
    OpenAIResourceWithRawResponse,
    AsyncOpenAIResourceWithRawResponse,
    OpenAIResourceWithStreamingResponse,
    AsyncOpenAIResourceWithStreamingResponse,
)

__all__ = ["GatewayResource", "AsyncGatewayResource"]


class GatewayResource(SyncAPIResource):
    @cached_property
    def openai(self) -> OpenAIResource:
        return OpenAIResource(self._client)

    @cached_property
    def model(self) -> ModelResource:
        return ModelResource(self._client)

    @cached_property
    def provider(self) -> ProviderResource:
        return ProviderResource(self._client)

    @cached_property
    def with_raw_response(self) -> GatewayResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return GatewayResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> GatewayResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return GatewayResourceWithStreamingResponse(self)


class AsyncGatewayResource(AsyncAPIResource):
    @cached_property
    def openai(self) -> AsyncOpenAIResource:
        return AsyncOpenAIResource(self._client)

    @cached_property
    def model(self) -> AsyncModelResource:
        return AsyncModelResource(self._client)

    @cached_property
    def provider(self) -> AsyncProviderResource:
        return AsyncProviderResource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncGatewayResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncGatewayResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncGatewayResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncGatewayResourceWithStreamingResponse(self)


class GatewayResourceWithRawResponse:
    def __init__(self, gateway: GatewayResource) -> None:
        self._gateway = gateway

    @cached_property
    def openai(self) -> OpenAIResourceWithRawResponse:
        return OpenAIResourceWithRawResponse(self._gateway.openai)

    @cached_property
    def model(self) -> ModelResourceWithRawResponse:
        return ModelResourceWithRawResponse(self._gateway.model)

    @cached_property
    def provider(self) -> ProviderResourceWithRawResponse:
        return ProviderResourceWithRawResponse(self._gateway.provider)


class AsyncGatewayResourceWithRawResponse:
    def __init__(self, gateway: AsyncGatewayResource) -> None:
        self._gateway = gateway

    @cached_property
    def openai(self) -> AsyncOpenAIResourceWithRawResponse:
        return AsyncOpenAIResourceWithRawResponse(self._gateway.openai)

    @cached_property
    def model(self) -> AsyncModelResourceWithRawResponse:
        return AsyncModelResourceWithRawResponse(self._gateway.model)

    @cached_property
    def provider(self) -> AsyncProviderResourceWithRawResponse:
        return AsyncProviderResourceWithRawResponse(self._gateway.provider)


class GatewayResourceWithStreamingResponse:
    def __init__(self, gateway: GatewayResource) -> None:
        self._gateway = gateway

    @cached_property
    def openai(self) -> OpenAIResourceWithStreamingResponse:
        return OpenAIResourceWithStreamingResponse(self._gateway.openai)

    @cached_property
    def model(self) -> ModelResourceWithStreamingResponse:
        return ModelResourceWithStreamingResponse(self._gateway.model)

    @cached_property
    def provider(self) -> ProviderResourceWithStreamingResponse:
        return ProviderResourceWithStreamingResponse(self._gateway.provider)


class AsyncGatewayResourceWithStreamingResponse:
    def __init__(self, gateway: AsyncGatewayResource) -> None:
        self._gateway = gateway

    @cached_property
    def openai(self) -> AsyncOpenAIResourceWithStreamingResponse:
        return AsyncOpenAIResourceWithStreamingResponse(self._gateway.openai)

    @cached_property
    def model(self) -> AsyncModelResourceWithStreamingResponse:
        return AsyncModelResourceWithStreamingResponse(self._gateway.model)

    @cached_property
    def provider(self) -> AsyncProviderResourceWithStreamingResponse:
        return AsyncProviderResourceWithStreamingResponse(self._gateway.provider)
