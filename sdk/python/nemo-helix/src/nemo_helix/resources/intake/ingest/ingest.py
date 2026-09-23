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

from .atif import (
    AtifResource,
    AsyncAtifResource,
    AtifResourceWithRawResponse,
    AsyncAtifResourceWithRawResponse,
    AtifResourceWithStreamingResponse,
    AsyncAtifResourceWithStreamingResponse,
)
from .spans import (
    SpansResource,
    AsyncSpansResource,
    SpansResourceWithRawResponse,
    AsyncSpansResourceWithRawResponse,
    SpansResourceWithStreamingResponse,
    AsyncSpansResourceWithStreamingResponse,
)
from .otlp.otlp import (
    OtlpResource,
    AsyncOtlpResource,
    OtlpResourceWithRawResponse,
    AsyncOtlpResourceWithRawResponse,
    OtlpResourceWithStreamingResponse,
    AsyncOtlpResourceWithStreamingResponse,
)
from ...._compat import cached_property
from ...._resource import SyncAPIResource, AsyncAPIResource
from .chat_completions import (
    ChatCompletionsResource,
    AsyncChatCompletionsResource,
    ChatCompletionsResourceWithRawResponse,
    AsyncChatCompletionsResourceWithRawResponse,
    ChatCompletionsResourceWithStreamingResponse,
    AsyncChatCompletionsResourceWithStreamingResponse,
)

__all__ = ["IngestResource", "AsyncIngestResource"]


class IngestResource(SyncAPIResource):
    @cached_property
    def atif(self) -> AtifResource:
        return AtifResource(self._client)

    @cached_property
    def chat_completions(self) -> ChatCompletionsResource:
        return ChatCompletionsResource(self._client)

    @cached_property
    def spans(self) -> SpansResource:
        return SpansResource(self._client)

    @cached_property
    def otlp(self) -> OtlpResource:
        return OtlpResource(self._client)

    @cached_property
    def with_raw_response(self) -> IngestResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return IngestResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> IngestResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return IngestResourceWithStreamingResponse(self)


class AsyncIngestResource(AsyncAPIResource):
    @cached_property
    def atif(self) -> AsyncAtifResource:
        return AsyncAtifResource(self._client)

    @cached_property
    def chat_completions(self) -> AsyncChatCompletionsResource:
        return AsyncChatCompletionsResource(self._client)

    @cached_property
    def spans(self) -> AsyncSpansResource:
        return AsyncSpansResource(self._client)

    @cached_property
    def otlp(self) -> AsyncOtlpResource:
        return AsyncOtlpResource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncIngestResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncIngestResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncIngestResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncIngestResourceWithStreamingResponse(self)


class IngestResourceWithRawResponse:
    def __init__(self, ingest: IngestResource) -> None:
        self._ingest = ingest

    @cached_property
    def atif(self) -> AtifResourceWithRawResponse:
        return AtifResourceWithRawResponse(self._ingest.atif)

    @cached_property
    def chat_completions(self) -> ChatCompletionsResourceWithRawResponse:
        return ChatCompletionsResourceWithRawResponse(self._ingest.chat_completions)

    @cached_property
    def spans(self) -> SpansResourceWithRawResponse:
        return SpansResourceWithRawResponse(self._ingest.spans)

    @cached_property
    def otlp(self) -> OtlpResourceWithRawResponse:
        return OtlpResourceWithRawResponse(self._ingest.otlp)


class AsyncIngestResourceWithRawResponse:
    def __init__(self, ingest: AsyncIngestResource) -> None:
        self._ingest = ingest

    @cached_property
    def atif(self) -> AsyncAtifResourceWithRawResponse:
        return AsyncAtifResourceWithRawResponse(self._ingest.atif)

    @cached_property
    def chat_completions(self) -> AsyncChatCompletionsResourceWithRawResponse:
        return AsyncChatCompletionsResourceWithRawResponse(self._ingest.chat_completions)

    @cached_property
    def spans(self) -> AsyncSpansResourceWithRawResponse:
        return AsyncSpansResourceWithRawResponse(self._ingest.spans)

    @cached_property
    def otlp(self) -> AsyncOtlpResourceWithRawResponse:
        return AsyncOtlpResourceWithRawResponse(self._ingest.otlp)


class IngestResourceWithStreamingResponse:
    def __init__(self, ingest: IngestResource) -> None:
        self._ingest = ingest

    @cached_property
    def atif(self) -> AtifResourceWithStreamingResponse:
        return AtifResourceWithStreamingResponse(self._ingest.atif)

    @cached_property
    def chat_completions(self) -> ChatCompletionsResourceWithStreamingResponse:
        return ChatCompletionsResourceWithStreamingResponse(self._ingest.chat_completions)

    @cached_property
    def spans(self) -> SpansResourceWithStreamingResponse:
        return SpansResourceWithStreamingResponse(self._ingest.spans)

    @cached_property
    def otlp(self) -> OtlpResourceWithStreamingResponse:
        return OtlpResourceWithStreamingResponse(self._ingest.otlp)


class AsyncIngestResourceWithStreamingResponse:
    def __init__(self, ingest: AsyncIngestResource) -> None:
        self._ingest = ingest

    @cached_property
    def atif(self) -> AsyncAtifResourceWithStreamingResponse:
        return AsyncAtifResourceWithStreamingResponse(self._ingest.atif)

    @cached_property
    def chat_completions(self) -> AsyncChatCompletionsResourceWithStreamingResponse:
        return AsyncChatCompletionsResourceWithStreamingResponse(self._ingest.chat_completions)

    @cached_property
    def spans(self) -> AsyncSpansResourceWithStreamingResponse:
        return AsyncSpansResourceWithStreamingResponse(self._ingest.spans)

    @cached_property
    def otlp(self) -> AsyncOtlpResourceWithStreamingResponse:
        return AsyncOtlpResourceWithStreamingResponse(self._ingest.otlp)
