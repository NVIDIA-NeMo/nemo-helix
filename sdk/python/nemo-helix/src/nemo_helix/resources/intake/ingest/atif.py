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
from typing_extensions import Literal

import httpx

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
from ...._base_client import make_request_options
from ....types.intake.ingest import atif_create_params
from ....types.intake.ingest.atif_step_param import AtifStepParam
from ....types.intake.ingest.atif_agent_param import AtifAgentParam
from ....types.intake.evaluation_context_param import EvaluationContextParam
from ....types.intake.ingest.atif_trajectory_param import AtifTrajectoryParam
from ....types.intake.ingest.atif_final_metrics_param import AtifFinalMetricsParam

__all__ = ["AtifResource", "AsyncAtifResource"]


class AtifResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AtifResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AtifResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AtifResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AtifResourceWithStreamingResponse(self)

    def create(
        self,
        *,
        workspace: str | None = None,
        agent: AtifAgentParam,
        schema_version: Literal[
            "ATIF-v1.0", "ATIF-v1.1", "ATIF-v1.2", "ATIF-v1.3", "ATIF-v1.4", "ATIF-v1.5", "ATIF-v1.6", "ATIF-v1.7"
        ],
        continued_trajectory_ref: str | Omit = omit,
        evaluation_context: EvaluationContextParam | Omit = omit,
        extra: Dict[str, object] | Omit = omit,
        final_metrics: AtifFinalMetricsParam | Omit = omit,
        notes: str | Omit = omit,
        session_id: str | Omit = omit,
        steps: Iterable[AtifStepParam] | Omit = omit,
        subagent_trajectories: Iterable[AtifTrajectoryParam] | Omit = omit,
        trajectory_id: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> None:
        """
        Ingest Atif

        Args:
          evaluation_context: Identifies the Evaluation and optional test case associated with ingested
              telemetry.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        extra_headers = {"Accept": "*/*", **(extra_headers or {})}
        return self._post(
            path_template("/apis/intake/v2/workspaces/{workspace}/ingest/atif", workspace=workspace),
            body=maybe_transform(
                {
                    "agent": agent,
                    "schema_version": schema_version,
                    "continued_trajectory_ref": continued_trajectory_ref,
                    "evaluation_context": evaluation_context,
                    "extra": extra,
                    "final_metrics": final_metrics,
                    "notes": notes,
                    "session_id": session_id,
                    "steps": steps,
                    "subagent_trajectories": subagent_trajectories,
                    "trajectory_id": trajectory_id,
                },
                atif_create_params.AtifCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
        )


class AsyncAtifResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncAtifResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncAtifResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncAtifResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncAtifResourceWithStreamingResponse(self)

    async def create(
        self,
        *,
        workspace: str | None = None,
        agent: AtifAgentParam,
        schema_version: Literal[
            "ATIF-v1.0", "ATIF-v1.1", "ATIF-v1.2", "ATIF-v1.3", "ATIF-v1.4", "ATIF-v1.5", "ATIF-v1.6", "ATIF-v1.7"
        ],
        continued_trajectory_ref: str | Omit = omit,
        evaluation_context: EvaluationContextParam | Omit = omit,
        extra: Dict[str, object] | Omit = omit,
        final_metrics: AtifFinalMetricsParam | Omit = omit,
        notes: str | Omit = omit,
        session_id: str | Omit = omit,
        steps: Iterable[AtifStepParam] | Omit = omit,
        subagent_trajectories: Iterable[AtifTrajectoryParam] | Omit = omit,
        trajectory_id: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> None:
        """
        Ingest Atif

        Args:
          evaluation_context: Identifies the Evaluation and optional test case associated with ingested
              telemetry.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        extra_headers = {"Accept": "*/*", **(extra_headers or {})}
        return await self._post(
            path_template("/apis/intake/v2/workspaces/{workspace}/ingest/atif", workspace=workspace),
            body=await async_maybe_transform(
                {
                    "agent": agent,
                    "schema_version": schema_version,
                    "continued_trajectory_ref": continued_trajectory_ref,
                    "evaluation_context": evaluation_context,
                    "extra": extra,
                    "final_metrics": final_metrics,
                    "notes": notes,
                    "session_id": session_id,
                    "steps": steps,
                    "subagent_trajectories": subagent_trajectories,
                    "trajectory_id": trajectory_id,
                },
                atif_create_params.AtifCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
        )


class AtifResourceWithRawResponse:
    def __init__(self, atif: AtifResource) -> None:
        self._atif = atif

        self.create = to_raw_response_wrapper(
            atif.create,
        )


class AsyncAtifResourceWithRawResponse:
    def __init__(self, atif: AsyncAtifResource) -> None:
        self._atif = atif

        self.create = async_to_raw_response_wrapper(
            atif.create,
        )


class AtifResourceWithStreamingResponse:
    def __init__(self, atif: AtifResource) -> None:
        self._atif = atif

        self.create = to_streamed_response_wrapper(
            atif.create,
        )


class AsyncAtifResourceWithStreamingResponse:
    def __init__(self, atif: AsyncAtifResource) -> None:
        self._atif = atif

        self.create = async_to_streamed_response_wrapper(
            atif.create,
        )
