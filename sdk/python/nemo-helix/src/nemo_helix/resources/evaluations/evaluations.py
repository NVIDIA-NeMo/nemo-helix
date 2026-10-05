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

from typing import Dict

import httpx

from ..._types import Body, Omit, Query, Headers, NoneType, NotGiven, SequenceNotStr, omit, not_given
from ..._utils import path_template, maybe_transform, async_maybe_transform
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
from ..._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from ...pagination import SyncDefaultPagination, AsyncDefaultPagination
from ..._base_client import AsyncPaginator, make_request_options
from ...types.evaluations import (
    evaluation_list_params,
    evaluation_patch_params,
    evaluation_create_params,
    evaluation_update_params,
)
from ...types.evaluations.evaluation_response import EvaluationResponse
from ...types.evaluations.evaluation_filter_param import EvaluationFilterParam
from ..._exceptions import ConflictError

__all__ = ["EvaluationsResource", "AsyncEvaluationsResource"]


class EvaluationsResource(SyncAPIResource):
    @cached_property
    def sessions(self) -> SessionsResource:
        return SessionsResource(self._client)

    @cached_property
    def with_raw_response(self) -> EvaluationsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return EvaluationsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> EvaluationsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return EvaluationsResourceWithStreamingResponse(self)

    def create(
        self,
        *,
        workspace: str | None = None,
        dataset_name: str,
        name: str,
        dataset_version: str | Omit = omit,
        description: str | Omit = omit,
        experiment_group_id: str | Omit = omit,
        experiment_ids: SequenceNotStr[str] | Omit = omit,
        metadata: Dict[str, str] | Omit = omit,
        parent_evaluation_id: str | Omit = omit,
        root_cause: str | Omit = omit,
        source_link: str | Omit = omit,
        status: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> EvaluationResponse:
        """
        Create Evaluation

        Args:
          dataset_name: Producer-supplied dataset name.

          name: Producer-supplied, workspace-unique evaluation id.

          dataset_version: Producer-supplied dataset version.

          description: Human-readable description.

          experiment_group_id: Deprecated single-experiment field; provide experiment_ids instead. Coalesced
              into experiment_ids when experiment_ids is omitted.

          experiment_ids: Entity ids of the Experiments this Evaluation belongs to (>=1). Preferred; each
              experiment must already exist. When omitted, the deprecated experiment_group_id
              is used instead.

          metadata: Free-form producer metadata.

          parent_evaluation_id: Entity id of the evaluation this one was derived from (e.g. a variant of a
              baseline), if any.

          root_cause: Human- or agent-authored explanation of the evaluation's outcome (e.g. why it
              was killed).

          source_link: Optional URL for the source evaluation.

          status: Producer-defined lifecycle status of the evaluation.


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
                path_template("/apis/intake/v2/workspaces/{workspace}/evaluations", workspace=workspace),
                body=maybe_transform(
                    {
                        "dataset_name": dataset_name,
                        "name": name,
                        "dataset_version": dataset_version,
                        "description": description,
                        "experiment_group_id": experiment_group_id,
                        "experiment_ids": experiment_ids,
                        "metadata": metadata,
                        "parent_evaluation_id": parent_evaluation_id,
                        "root_cause": root_cause,
                        "source_link": source_link,
                        "status": status,
                    },
                    evaluation_create_params.EvaluationCreateParams,
                ),
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=EvaluationResponse,
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
    ) -> EvaluationResponse:
        """
        Get Evaluation

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
            path_template("/apis/intake/v2/workspaces/{workspace}/evaluations/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=EvaluationResponse,
        )

    def update(
        self,
        path_name: str,
        *,
        workspace: str | None = None,
        dataset_name: str,
        body_name: str,
        dataset_version: str | Omit = omit,
        description: str | Omit = omit,
        experiment_group_id: str | Omit = omit,
        experiment_ids: SequenceNotStr[str] | Omit = omit,
        metadata: Dict[str, str] | Omit = omit,
        parent_evaluation_id: str | Omit = omit,
        root_cause: str | Omit = omit,
        source_link: str | Omit = omit,
        status: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> EvaluationResponse:
        """
        Update Evaluation

        Args:
          dataset_name: Producer-supplied dataset name.

          body_name: Producer-supplied, workspace-unique evaluation id.

          dataset_version: Producer-supplied dataset version.

          description: Human-readable description.

          experiment_group_id: Deprecated single-experiment field; provide experiment_ids instead. Coalesced
              into experiment_ids when experiment_ids is omitted.

          experiment_ids: Entity ids of the Experiments this Evaluation belongs to (>=1). Preferred; each
              experiment must already exist. When omitted, the deprecated experiment_group_id
              is used instead.

          metadata: Free-form producer metadata.

          parent_evaluation_id: Entity id of the evaluation this one was derived from (e.g. a variant of a
              baseline), if any.

          root_cause: Human- or agent-authored explanation of the evaluation's outcome (e.g. why it
              was killed).

          source_link: Optional URL for the source evaluation.

          status: Producer-defined lifecycle status of the evaluation.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not path_name:
            raise ValueError(f"Expected a non-empty value for `path_name` but received {path_name!r}")
        return self._put(
            path_template(
                "/apis/intake/v2/workspaces/{workspace}/evaluations/{path_name}",
                workspace=workspace,
                path_name=path_name,
            ),
            body=maybe_transform(
                {
                    "dataset_name": dataset_name,
                    "body_name": body_name,
                    "dataset_version": dataset_version,
                    "description": description,
                    "experiment_group_id": experiment_group_id,
                    "experiment_ids": experiment_ids,
                    "metadata": metadata,
                    "parent_evaluation_id": parent_evaluation_id,
                    "root_cause": root_cause,
                    "source_link": source_link,
                    "status": status,
                },
                evaluation_update_params.EvaluationUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=EvaluationResponse,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: EvaluationFilterParam | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> SyncDefaultPagination[EvaluationResponse]:
        """
        List Evaluations

        Args:
          filter: Filter evaluations by name, experiment_id (experiment group membership;
              experiment_group_id is a deprecated alias), dataset_name, dataset_version,
              created_by, created_at, or updated_at. Pass is_deleted=true to return only
              soft-deleted evaluations; omit to see only live ones. Pass is_pinned=true (or
              false) to filter by pinned state; omit to return both. Filter by a metadata
              key/value: filter[metadata.<key>]=<value>. Filter by a rollup metric with
              numeric range operators ($gte/$lte/$gt/$lt/$eq): filter[run_count][$gte]=5,
              filter[cost_usd.mean][$lte]=0.5, filter[latency_ms.p95][$lte]=1000,
              filter[tokens.mean][$lte]=5000, or filter[evaluators.<name>.mean][$gte]=0.8.

          page: Page number.

          page_size: Page size.

          sort: Comma-separated list of fields to sort by, applied in order (the first field
              dominates); prefix any field with '-' for descending — e.g.
              '-evaluators.reward.mean,cost_usd.mean'. Each field is an evaluation attribute
              (name, created_at, updated_at, pinned_at) or an aggregate metric: run_count,
              test_case_count, cost_usd.<stat>, latency_ms.<stat>, tokens.<stat>, or
              evaluators.<name>.<stat>, where <stat> is one of mean, median, p90, p95, p99,
              sum, count. When omitted, defaults to -created_at with pinned evaluations first.

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
            path_template("/apis/intake/v2/workspaces/{workspace}/evaluations", workspace=workspace),
            page=SyncDefaultPagination[EvaluationResponse],
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
                    evaluation_list_params.EvaluationListParams,
                ),
            ),
            model=EvaluationResponse,
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
        Delete Evaluation

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
            path_template("/apis/intake/v2/workspaces/{workspace}/evaluations/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
        )

    def patch(
        self,
        name: str,
        *,
        workspace: str | None = None,
        description: str | Omit = omit,
        experiment_ids: SequenceNotStr[str] | Omit = omit,
        metadata: Dict[str, str] | Omit = omit,
        parent_evaluation_id: str | Omit = omit,
        root_cause: str | Omit = omit,
        source_link: str | Omit = omit,
        status: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> EvaluationResponse:
        """
        Partially update an evaluation: only fields present in the request are changed.

        The common case is curating an evaluation into another ExperimentGroup — PATCH
        with the merged `experiment_ids`. Membership is replaced (not appended), so send
        the full desired set; any new group must exist and the set must be non-empty (an
        evaluation always belongs to >=1 group). Omitted fields are left untouched
        (unlike the full-body PUT, which overwrites them).

        Args:
          description: Human-readable description.

          experiment_ids: Replace the Experiments this Evaluation belongs to. Must be non-empty when
              provided; each experiment must already exist. Omit to leave membership
              unchanged.

          metadata: Free-form producer metadata.

          parent_evaluation_id: Entity id of the evaluation this one was derived from (e.g. a variant of a
              baseline), if any.

          root_cause: Human- or agent-authored explanation of the evaluation's outcome (e.g. why it
              was killed).

          source_link: Optional URL for the source evaluation.

          status: Producer-defined lifecycle status of the evaluation.

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
            path_template("/apis/intake/v2/workspaces/{workspace}/evaluations/{name}", workspace=workspace, name=name),
            body=maybe_transform(
                {
                    "description": description,
                    "experiment_ids": experiment_ids,
                    "metadata": metadata,
                    "parent_evaluation_id": parent_evaluation_id,
                    "root_cause": root_cause,
                    "source_link": source_link,
                    "status": status,
                },
                evaluation_patch_params.EvaluationPatchParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=EvaluationResponse,
        )

    def pin(
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
    ) -> EvaluationResponse:
        """
        Pin an evaluation to the top of the list (workspace-shared).

        Re-pinning an already-pinned evaluation refreshes `pinned_at` to the current
        timestamp, which is intentional (most-recently-pinned sorts first).

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
        return self._post(
            path_template(
                "/apis/intake/v2/workspaces/{workspace}/evaluations/{name}/pin", workspace=workspace, name=name
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=EvaluationResponse,
        )

    def unpin(
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
    ) -> EvaluationResponse:
        """Unpin an evaluation.

        Idempotent: unpinning an already-unpinned evaluation is a
        no-op.

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
        return self._delete(
            path_template(
                "/apis/intake/v2/workspaces/{workspace}/evaluations/{name}/pin", workspace=workspace, name=name
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=EvaluationResponse,
        )


class AsyncEvaluationsResource(AsyncAPIResource):
    @cached_property
    def sessions(self) -> AsyncSessionsResource:
        return AsyncSessionsResource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncEvaluationsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncEvaluationsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncEvaluationsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncEvaluationsResourceWithStreamingResponse(self)

    async def create(
        self,
        *,
        workspace: str | None = None,
        dataset_name: str,
        name: str,
        dataset_version: str | Omit = omit,
        description: str | Omit = omit,
        experiment_group_id: str | Omit = omit,
        experiment_ids: SequenceNotStr[str] | Omit = omit,
        metadata: Dict[str, str] | Omit = omit,
        parent_evaluation_id: str | Omit = omit,
        root_cause: str | Omit = omit,
        source_link: str | Omit = omit,
        status: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> EvaluationResponse:
        """
        Create Evaluation

        Args:
          dataset_name: Producer-supplied dataset name.

          name: Producer-supplied, workspace-unique evaluation id.

          dataset_version: Producer-supplied dataset version.

          description: Human-readable description.

          experiment_group_id: Deprecated single-experiment field; provide experiment_ids instead. Coalesced
              into experiment_ids when experiment_ids is omitted.

          experiment_ids: Entity ids of the Experiments this Evaluation belongs to (>=1). Preferred; each
              experiment must already exist. When omitted, the deprecated experiment_group_id
              is used instead.

          metadata: Free-form producer metadata.

          parent_evaluation_id: Entity id of the evaluation this one was derived from (e.g. a variant of a
              baseline), if any.

          root_cause: Human- or agent-authored explanation of the evaluation's outcome (e.g. why it
              was killed).

          source_link: Optional URL for the source evaluation.

          status: Producer-defined lifecycle status of the evaluation.


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
                path_template("/apis/intake/v2/workspaces/{workspace}/evaluations", workspace=workspace),
                body=await async_maybe_transform(
                    {
                        "dataset_name": dataset_name,
                        "name": name,
                        "dataset_version": dataset_version,
                        "description": description,
                        "experiment_group_id": experiment_group_id,
                        "experiment_ids": experiment_ids,
                        "metadata": metadata,
                        "parent_evaluation_id": parent_evaluation_id,
                        "root_cause": root_cause,
                        "source_link": source_link,
                        "status": status,
                    },
                    evaluation_create_params.EvaluationCreateParams,
                ),
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=EvaluationResponse,
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
    ) -> EvaluationResponse:
        """
        Get Evaluation

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
            path_template("/apis/intake/v2/workspaces/{workspace}/evaluations/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=EvaluationResponse,
        )

    async def update(
        self,
        path_name: str,
        *,
        workspace: str | None = None,
        dataset_name: str,
        body_name: str,
        dataset_version: str | Omit = omit,
        description: str | Omit = omit,
        experiment_group_id: str | Omit = omit,
        experiment_ids: SequenceNotStr[str] | Omit = omit,
        metadata: Dict[str, str] | Omit = omit,
        parent_evaluation_id: str | Omit = omit,
        root_cause: str | Omit = omit,
        source_link: str | Omit = omit,
        status: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> EvaluationResponse:
        """
        Update Evaluation

        Args:
          dataset_name: Producer-supplied dataset name.

          body_name: Producer-supplied, workspace-unique evaluation id.

          dataset_version: Producer-supplied dataset version.

          description: Human-readable description.

          experiment_group_id: Deprecated single-experiment field; provide experiment_ids instead. Coalesced
              into experiment_ids when experiment_ids is omitted.

          experiment_ids: Entity ids of the Experiments this Evaluation belongs to (>=1). Preferred; each
              experiment must already exist. When omitted, the deprecated experiment_group_id
              is used instead.

          metadata: Free-form producer metadata.

          parent_evaluation_id: Entity id of the evaluation this one was derived from (e.g. a variant of a
              baseline), if any.

          root_cause: Human- or agent-authored explanation of the evaluation's outcome (e.g. why it
              was killed).

          source_link: Optional URL for the source evaluation.

          status: Producer-defined lifecycle status of the evaluation.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not path_name:
            raise ValueError(f"Expected a non-empty value for `path_name` but received {path_name!r}")
        return await self._put(
            path_template(
                "/apis/intake/v2/workspaces/{workspace}/evaluations/{path_name}",
                workspace=workspace,
                path_name=path_name,
            ),
            body=await async_maybe_transform(
                {
                    "dataset_name": dataset_name,
                    "body_name": body_name,
                    "dataset_version": dataset_version,
                    "description": description,
                    "experiment_group_id": experiment_group_id,
                    "experiment_ids": experiment_ids,
                    "metadata": metadata,
                    "parent_evaluation_id": parent_evaluation_id,
                    "root_cause": root_cause,
                    "source_link": source_link,
                    "status": status,
                },
                evaluation_update_params.EvaluationUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=EvaluationResponse,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: EvaluationFilterParam | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> AsyncPaginator[EvaluationResponse, AsyncDefaultPagination[EvaluationResponse]]:
        """
        List Evaluations

        Args:
          filter: Filter evaluations by name, experiment_id (experiment group membership;
              experiment_group_id is a deprecated alias), dataset_name, dataset_version,
              created_by, created_at, or updated_at. Pass is_deleted=true to return only
              soft-deleted evaluations; omit to see only live ones. Pass is_pinned=true (or
              false) to filter by pinned state; omit to return both. Filter by a metadata
              key/value: filter[metadata.<key>]=<value>. Filter by a rollup metric with
              numeric range operators ($gte/$lte/$gt/$lt/$eq): filter[run_count][$gte]=5,
              filter[cost_usd.mean][$lte]=0.5, filter[latency_ms.p95][$lte]=1000,
              filter[tokens.mean][$lte]=5000, or filter[evaluators.<name>.mean][$gte]=0.8.

          page: Page number.

          page_size: Page size.

          sort: Comma-separated list of fields to sort by, applied in order (the first field
              dominates); prefix any field with '-' for descending — e.g.
              '-evaluators.reward.mean,cost_usd.mean'. Each field is an evaluation attribute
              (name, created_at, updated_at, pinned_at) or an aggregate metric: run_count,
              test_case_count, cost_usd.<stat>, latency_ms.<stat>, tokens.<stat>, or
              evaluators.<name>.<stat>, where <stat> is one of mean, median, p90, p95, p99,
              sum, count. When omitted, defaults to -created_at with pinned evaluations first.

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
            path_template("/apis/intake/v2/workspaces/{workspace}/evaluations", workspace=workspace),
            page=AsyncDefaultPagination[EvaluationResponse],
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
                    evaluation_list_params.EvaluationListParams,
                ),
            ),
            model=EvaluationResponse,
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
        Delete Evaluation

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
            path_template("/apis/intake/v2/workspaces/{workspace}/evaluations/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
        )

    async def patch(
        self,
        name: str,
        *,
        workspace: str | None = None,
        description: str | Omit = omit,
        experiment_ids: SequenceNotStr[str] | Omit = omit,
        metadata: Dict[str, str] | Omit = omit,
        parent_evaluation_id: str | Omit = omit,
        root_cause: str | Omit = omit,
        source_link: str | Omit = omit,
        status: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> EvaluationResponse:
        """
        Partially update an evaluation: only fields present in the request are changed.

        The common case is curating an evaluation into another ExperimentGroup — PATCH
        with the merged `experiment_ids`. Membership is replaced (not appended), so send
        the full desired set; any new group must exist and the set must be non-empty (an
        evaluation always belongs to >=1 group). Omitted fields are left untouched
        (unlike the full-body PUT, which overwrites them).

        Args:
          description: Human-readable description.

          experiment_ids: Replace the Experiments this Evaluation belongs to. Must be non-empty when
              provided; each experiment must already exist. Omit to leave membership
              unchanged.

          metadata: Free-form producer metadata.

          parent_evaluation_id: Entity id of the evaluation this one was derived from (e.g. a variant of a
              baseline), if any.

          root_cause: Human- or agent-authored explanation of the evaluation's outcome (e.g. why it
              was killed).

          source_link: Optional URL for the source evaluation.

          status: Producer-defined lifecycle status of the evaluation.

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
            path_template("/apis/intake/v2/workspaces/{workspace}/evaluations/{name}", workspace=workspace, name=name),
            body=await async_maybe_transform(
                {
                    "description": description,
                    "experiment_ids": experiment_ids,
                    "metadata": metadata,
                    "parent_evaluation_id": parent_evaluation_id,
                    "root_cause": root_cause,
                    "source_link": source_link,
                    "status": status,
                },
                evaluation_patch_params.EvaluationPatchParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=EvaluationResponse,
        )

    async def pin(
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
    ) -> EvaluationResponse:
        """
        Pin an evaluation to the top of the list (workspace-shared).

        Re-pinning an already-pinned evaluation refreshes `pinned_at` to the current
        timestamp, which is intentional (most-recently-pinned sorts first).

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
        return await self._post(
            path_template(
                "/apis/intake/v2/workspaces/{workspace}/evaluations/{name}/pin", workspace=workspace, name=name
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=EvaluationResponse,
        )

    async def unpin(
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
    ) -> EvaluationResponse:
        """Unpin an evaluation.

        Idempotent: unpinning an already-unpinned evaluation is a
        no-op.

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
        return await self._delete(
            path_template(
                "/apis/intake/v2/workspaces/{workspace}/evaluations/{name}/pin", workspace=workspace, name=name
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=EvaluationResponse,
        )


class EvaluationsResourceWithRawResponse:
    def __init__(self, evaluations: EvaluationsResource) -> None:
        self._evaluations = evaluations

        self.create = to_raw_response_wrapper(
            evaluations.create,
        )
        self.retrieve = to_raw_response_wrapper(
            evaluations.retrieve,
        )
        self.update = to_raw_response_wrapper(
            evaluations.update,
        )
        self.list = to_raw_response_wrapper(
            evaluations.list,
        )
        self.delete = to_raw_response_wrapper(
            evaluations.delete,
        )
        self.patch = to_raw_response_wrapper(
            evaluations.patch,
        )
        self.pin = to_raw_response_wrapper(
            evaluations.pin,
        )
        self.unpin = to_raw_response_wrapper(
            evaluations.unpin,
        )

    @cached_property
    def sessions(self) -> SessionsResourceWithRawResponse:
        return SessionsResourceWithRawResponse(self._evaluations.sessions)


class AsyncEvaluationsResourceWithRawResponse:
    def __init__(self, evaluations: AsyncEvaluationsResource) -> None:
        self._evaluations = evaluations

        self.create = async_to_raw_response_wrapper(
            evaluations.create,
        )
        self.retrieve = async_to_raw_response_wrapper(
            evaluations.retrieve,
        )
        self.update = async_to_raw_response_wrapper(
            evaluations.update,
        )
        self.list = async_to_raw_response_wrapper(
            evaluations.list,
        )
        self.delete = async_to_raw_response_wrapper(
            evaluations.delete,
        )
        self.patch = async_to_raw_response_wrapper(
            evaluations.patch,
        )
        self.pin = async_to_raw_response_wrapper(
            evaluations.pin,
        )
        self.unpin = async_to_raw_response_wrapper(
            evaluations.unpin,
        )

    @cached_property
    def sessions(self) -> AsyncSessionsResourceWithRawResponse:
        return AsyncSessionsResourceWithRawResponse(self._evaluations.sessions)


class EvaluationsResourceWithStreamingResponse:
    def __init__(self, evaluations: EvaluationsResource) -> None:
        self._evaluations = evaluations

        self.create = to_streamed_response_wrapper(
            evaluations.create,
        )
        self.retrieve = to_streamed_response_wrapper(
            evaluations.retrieve,
        )
        self.update = to_streamed_response_wrapper(
            evaluations.update,
        )
        self.list = to_streamed_response_wrapper(
            evaluations.list,
        )
        self.delete = to_streamed_response_wrapper(
            evaluations.delete,
        )
        self.patch = to_streamed_response_wrapper(
            evaluations.patch,
        )
        self.pin = to_streamed_response_wrapper(
            evaluations.pin,
        )
        self.unpin = to_streamed_response_wrapper(
            evaluations.unpin,
        )

    @cached_property
    def sessions(self) -> SessionsResourceWithStreamingResponse:
        return SessionsResourceWithStreamingResponse(self._evaluations.sessions)


class AsyncEvaluationsResourceWithStreamingResponse:
    def __init__(self, evaluations: AsyncEvaluationsResource) -> None:
        self._evaluations = evaluations

        self.create = async_to_streamed_response_wrapper(
            evaluations.create,
        )
        self.retrieve = async_to_streamed_response_wrapper(
            evaluations.retrieve,
        )
        self.update = async_to_streamed_response_wrapper(
            evaluations.update,
        )
        self.list = async_to_streamed_response_wrapper(
            evaluations.list,
        )
        self.delete = async_to_streamed_response_wrapper(
            evaluations.delete,
        )
        self.patch = async_to_streamed_response_wrapper(
            evaluations.patch,
        )
        self.pin = async_to_streamed_response_wrapper(
            evaluations.pin,
        )
        self.unpin = async_to_streamed_response_wrapper(
            evaluations.unpin,
        )

    @cached_property
    def sessions(self) -> AsyncSessionsResourceWithStreamingResponse:
        return AsyncSessionsResourceWithStreamingResponse(self._evaluations.sessions)
