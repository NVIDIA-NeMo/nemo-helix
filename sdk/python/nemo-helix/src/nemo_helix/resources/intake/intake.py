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

from .traces import (
    TracesResource,
    AsyncTracesResource,
    TracesResourceWithRawResponse,
    AsyncTracesResourceWithRawResponse,
    TracesResourceWithStreamingResponse,
    AsyncTracesResourceWithStreamingResponse,
)
from .sessions import (
    SessionsResource,
    AsyncSessionsResource,
    SessionsResourceWithRawResponse,
    AsyncSessionsResourceWithRawResponse,
    SessionsResourceWithStreamingResponse,
    AsyncSessionsResourceWithStreamingResponse,
)
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from .annotations import (
    AnnotationsResource,
    AsyncAnnotationsResource,
    AnnotationsResourceWithRawResponse,
    AsyncAnnotationsResourceWithRawResponse,
    AnnotationsResourceWithStreamingResponse,
    AsyncAnnotationsResourceWithStreamingResponse,
)
from .spans.spans import (
    SpansResource,
    AsyncSpansResource,
    SpansResourceWithRawResponse,
    AsyncSpansResourceWithRawResponse,
    SpansResourceWithStreamingResponse,
    AsyncSpansResourceWithStreamingResponse,
)
from .ingest.ingest import (
    IngestResource,
    AsyncIngestResource,
    IngestResourceWithRawResponse,
    AsyncIngestResourceWithRawResponse,
    IngestResourceWithStreamingResponse,
    AsyncIngestResourceWithStreamingResponse,
)
from .evaluator_results import (
    EvaluatorResultsResource,
    AsyncEvaluatorResultsResource,
    EvaluatorResultsResourceWithRawResponse,
    AsyncEvaluatorResultsResourceWithRawResponse,
    EvaluatorResultsResourceWithStreamingResponse,
    AsyncEvaluatorResultsResourceWithStreamingResponse,
)

__all__ = ["IntakeResource", "AsyncIntakeResource"]


class IntakeResource(SyncAPIResource):
    @cached_property
    def evaluator_results(self) -> EvaluatorResultsResource:
        return EvaluatorResultsResource(self._client)

    @cached_property
    def annotations(self) -> AnnotationsResource:
        return AnnotationsResource(self._client)

    @cached_property
    def ingest(self) -> IngestResource:
        return IngestResource(self._client)

    @cached_property
    def spans(self) -> SpansResource:
        return SpansResource(self._client)

    @cached_property
    def sessions(self) -> SessionsResource:
        return SessionsResource(self._client)

    @cached_property
    def traces(self) -> TracesResource:
        return TracesResource(self._client)

    @cached_property
    def with_raw_response(self) -> IntakeResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return IntakeResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> IntakeResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return IntakeResourceWithStreamingResponse(self)


class AsyncIntakeResource(AsyncAPIResource):
    @cached_property
    def evaluator_results(self) -> AsyncEvaluatorResultsResource:
        return AsyncEvaluatorResultsResource(self._client)

    @cached_property
    def annotations(self) -> AsyncAnnotationsResource:
        return AsyncAnnotationsResource(self._client)

    @cached_property
    def ingest(self) -> AsyncIngestResource:
        return AsyncIngestResource(self._client)

    @cached_property
    def spans(self) -> AsyncSpansResource:
        return AsyncSpansResource(self._client)

    @cached_property
    def sessions(self) -> AsyncSessionsResource:
        return AsyncSessionsResource(self._client)

    @cached_property
    def traces(self) -> AsyncTracesResource:
        return AsyncTracesResource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncIntakeResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncIntakeResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncIntakeResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncIntakeResourceWithStreamingResponse(self)


class IntakeResourceWithRawResponse:
    def __init__(self, intake: IntakeResource) -> None:
        self._intake = intake

    @cached_property
    def evaluator_results(self) -> EvaluatorResultsResourceWithRawResponse:
        return EvaluatorResultsResourceWithRawResponse(self._intake.evaluator_results)

    @cached_property
    def annotations(self) -> AnnotationsResourceWithRawResponse:
        return AnnotationsResourceWithRawResponse(self._intake.annotations)

    @cached_property
    def ingest(self) -> IngestResourceWithRawResponse:
        return IngestResourceWithRawResponse(self._intake.ingest)

    @cached_property
    def spans(self) -> SpansResourceWithRawResponse:
        return SpansResourceWithRawResponse(self._intake.spans)

    @cached_property
    def sessions(self) -> SessionsResourceWithRawResponse:
        return SessionsResourceWithRawResponse(self._intake.sessions)

    @cached_property
    def traces(self) -> TracesResourceWithRawResponse:
        return TracesResourceWithRawResponse(self._intake.traces)


class AsyncIntakeResourceWithRawResponse:
    def __init__(self, intake: AsyncIntakeResource) -> None:
        self._intake = intake

    @cached_property
    def evaluator_results(self) -> AsyncEvaluatorResultsResourceWithRawResponse:
        return AsyncEvaluatorResultsResourceWithRawResponse(self._intake.evaluator_results)

    @cached_property
    def annotations(self) -> AsyncAnnotationsResourceWithRawResponse:
        return AsyncAnnotationsResourceWithRawResponse(self._intake.annotations)

    @cached_property
    def ingest(self) -> AsyncIngestResourceWithRawResponse:
        return AsyncIngestResourceWithRawResponse(self._intake.ingest)

    @cached_property
    def spans(self) -> AsyncSpansResourceWithRawResponse:
        return AsyncSpansResourceWithRawResponse(self._intake.spans)

    @cached_property
    def sessions(self) -> AsyncSessionsResourceWithRawResponse:
        return AsyncSessionsResourceWithRawResponse(self._intake.sessions)

    @cached_property
    def traces(self) -> AsyncTracesResourceWithRawResponse:
        return AsyncTracesResourceWithRawResponse(self._intake.traces)


class IntakeResourceWithStreamingResponse:
    def __init__(self, intake: IntakeResource) -> None:
        self._intake = intake

    @cached_property
    def evaluator_results(self) -> EvaluatorResultsResourceWithStreamingResponse:
        return EvaluatorResultsResourceWithStreamingResponse(self._intake.evaluator_results)

    @cached_property
    def annotations(self) -> AnnotationsResourceWithStreamingResponse:
        return AnnotationsResourceWithStreamingResponse(self._intake.annotations)

    @cached_property
    def ingest(self) -> IngestResourceWithStreamingResponse:
        return IngestResourceWithStreamingResponse(self._intake.ingest)

    @cached_property
    def spans(self) -> SpansResourceWithStreamingResponse:
        return SpansResourceWithStreamingResponse(self._intake.spans)

    @cached_property
    def sessions(self) -> SessionsResourceWithStreamingResponse:
        return SessionsResourceWithStreamingResponse(self._intake.sessions)

    @cached_property
    def traces(self) -> TracesResourceWithStreamingResponse:
        return TracesResourceWithStreamingResponse(self._intake.traces)


class AsyncIntakeResourceWithStreamingResponse:
    def __init__(self, intake: AsyncIntakeResource) -> None:
        self._intake = intake

    @cached_property
    def evaluator_results(self) -> AsyncEvaluatorResultsResourceWithStreamingResponse:
        return AsyncEvaluatorResultsResourceWithStreamingResponse(self._intake.evaluator_results)

    @cached_property
    def annotations(self) -> AsyncAnnotationsResourceWithStreamingResponse:
        return AsyncAnnotationsResourceWithStreamingResponse(self._intake.annotations)

    @cached_property
    def ingest(self) -> AsyncIngestResourceWithStreamingResponse:
        return AsyncIngestResourceWithStreamingResponse(self._intake.ingest)

    @cached_property
    def spans(self) -> AsyncSpansResourceWithStreamingResponse:
        return AsyncSpansResourceWithStreamingResponse(self._intake.spans)

    @cached_property
    def sessions(self) -> AsyncSessionsResourceWithStreamingResponse:
        return AsyncSessionsResourceWithStreamingResponse(self._intake.sessions)

    @cached_property
    def traces(self) -> AsyncTracesResourceWithStreamingResponse:
        return AsyncTracesResourceWithStreamingResponse(self._intake.traces)
