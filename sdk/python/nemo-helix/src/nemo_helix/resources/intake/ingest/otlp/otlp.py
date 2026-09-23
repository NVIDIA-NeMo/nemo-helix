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

from .v1.v1 import (
    V1Resource,
    AsyncV1Resource,
    V1ResourceWithRawResponse,
    AsyncV1ResourceWithRawResponse,
    V1ResourceWithStreamingResponse,
    AsyncV1ResourceWithStreamingResponse,
)
from ....._compat import cached_property
from ....._resource import SyncAPIResource, AsyncAPIResource

__all__ = ["OtlpResource", "AsyncOtlpResource"]


class OtlpResource(SyncAPIResource):
    @cached_property
    def v1(self) -> V1Resource:
        return V1Resource(self._client)

    @cached_property
    def with_raw_response(self) -> OtlpResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return OtlpResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> OtlpResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return OtlpResourceWithStreamingResponse(self)


class AsyncOtlpResource(AsyncAPIResource):
    @cached_property
    def v1(self) -> AsyncV1Resource:
        return AsyncV1Resource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncOtlpResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncOtlpResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncOtlpResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncOtlpResourceWithStreamingResponse(self)


class OtlpResourceWithRawResponse:
    def __init__(self, otlp: OtlpResource) -> None:
        self._otlp = otlp

    @cached_property
    def v1(self) -> V1ResourceWithRawResponse:
        return V1ResourceWithRawResponse(self._otlp.v1)


class AsyncOtlpResourceWithRawResponse:
    def __init__(self, otlp: AsyncOtlpResource) -> None:
        self._otlp = otlp

    @cached_property
    def v1(self) -> AsyncV1ResourceWithRawResponse:
        return AsyncV1ResourceWithRawResponse(self._otlp.v1)


class OtlpResourceWithStreamingResponse:
    def __init__(self, otlp: OtlpResource) -> None:
        self._otlp = otlp

    @cached_property
    def v1(self) -> V1ResourceWithStreamingResponse:
        return V1ResourceWithStreamingResponse(self._otlp.v1)


class AsyncOtlpResourceWithStreamingResponse:
    def __init__(self, otlp: AsyncOtlpResource) -> None:
        self._otlp = otlp

    @cached_property
    def v1(self) -> AsyncV1ResourceWithStreamingResponse:
        return AsyncV1ResourceWithStreamingResponse(self._otlp.v1)
