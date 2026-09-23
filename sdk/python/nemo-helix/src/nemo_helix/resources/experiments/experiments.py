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
from typing_extensions import Literal

import httpx

from ..._types import Body, Omit, Query, Headers, NoneType, NotGiven, omit, not_given
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
from ...types.experiments import (
    experiment_list_params,
    experiment_create_params,
    experiment_update_params,
)
from ...types.experiments.column_layout_param import ColumnLayoutParam
from ...types.experiments.experiment_response import ExperimentResponse
from ...types.experiments.pareto_config_param import ParetoConfigParam
from ...types.experiments.experiment_filter_param import ExperimentFilterParam
from ..._exceptions import ConflictError

__all__ = ["ExperimentsResource", "AsyncExperimentsResource"]


class ExperimentsResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> ExperimentsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return ExperimentsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> ExperimentsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return ExperimentsResourceWithStreamingResponse(self)

    def create(
        self,
        *,
        workspace: str | None = None,
        name: str,
        column_layout: ColumnLayoutParam | Omit = omit,
        default_sort: str | Omit = omit,
        description: str | Omit = omit,
        insight_id: str | Omit = omit,
        is_favorite: bool | Omit = omit,
        metadata: Dict[str, str] | Omit = omit,
        pareto: ParetoConfigParam | Omit = omit,
        show_evaluations_over_time: bool | Omit = omit,
        summary: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ExperimentResponse:
        """
        Create Experiment

        Args:
          name: Workspace-unique experiment name.

          column_layout: A saved table layout for a group's evaluations list: column order and which
              columns are hidden.

              Column ids are Studio's and cannot be enumerated here — the table builds a
              column per evaluator and metadata key found in the rows — so ids are stored and
              echoed back unvalidated.

              Visibility is stored as the _hidden_ ids rather than a map over every column, so
              a column that appears later (a new evaluator, a new metadata key) shows up by
              default.

          default_sort: Default sort for this experiment's evaluations list, as a `sort`-param string: a
              comma-separated, ordered list of fields where the first is the primary sort and
              the rest break ties (leading '-' on a field = descending), e.g.
              '-evaluators.reward.mean,cost_usd.mean'. Defaults to '-created_at'. Accepts any
              field the evaluations list `sort` param does; clients apply it as the list
              `sort` param.

          description: Human-readable purpose of the experiment.

          insight_id: Reference to an external insight that seeded this experiment, if any.

          is_favorite: Whether this Experiment is marked as a favorite. Defaults to false on create;
              omit on update to preserve the existing value.

          metadata: Free-form producer metadata for the experiment.

          pareto: Default X/Y metrics for a group's cost-vs-accuracy Pareto view.

              Metric ids use the same vocabulary as the evaluations list sort/filter fields —
              `cost_usd`, `latency_ms`, or `evaluators.<name>`. Defaults to cost (x) vs
              latency (y): both exist for every group, so the chart always has something to
              render before anyone customizes it.

          show_evaluations_over_time: Whether Studio should display this Experiment's Evaluation results over time.
              Defaults to false on create; omit on update to preserve the existing value.

          summary: Human- or agent-authored summary of the experiment's findings.


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
                path_template("/apis/intake/v2/workspaces/{workspace}/experiments", workspace=workspace),
                body=maybe_transform(
                    {
                        "name": name,
                        "column_layout": column_layout,
                        "default_sort": default_sort,
                        "description": description,
                        "insight_id": insight_id,
                        "is_favorite": is_favorite,
                        "metadata": metadata,
                        "pareto": pareto,
                        "show_evaluations_over_time": show_evaluations_over_time,
                        "summary": summary,
                    },
                    experiment_create_params.ExperimentCreateParams,
                ),
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=ExperimentResponse,
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
    ) -> ExperimentResponse:
        """
        Get Experiment

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
            path_template("/apis/intake/v2/workspaces/{workspace}/experiments/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ExperimentResponse,
        )

    def update(
        self,
        path_name: str,
        *,
        workspace: str | None = None,
        body_name: str,
        baseline_evaluation_name: Optional[str] | Omit = omit,
        column_layout: ColumnLayoutParam | Omit = omit,
        default_sort: str | Omit = omit,
        description: str | Omit = omit,
        insight_id: str | Omit = omit,
        is_favorite: bool | Omit = omit,
        metadata: Dict[str, str] | Omit = omit,
        pareto: ParetoConfigParam | Omit = omit,
        show_evaluations_over_time: bool | Omit = omit,
        summary: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ExperimentResponse:
        """
        Update Experiment

        Args:
          body_name: Workspace-unique experiment name.

          baseline_evaluation_name: Name of this Experiment's baseline Evaluation. The Evaluation must already be a
              live member of the Experiment. Set null to clear the selected baseline.

          column_layout: A saved table layout for a group's evaluations list: column order and which
              columns are hidden.

              Column ids are Studio's and cannot be enumerated here — the table builds a
              column per evaluator and metadata key found in the rows — so ids are stored and
              echoed back unvalidated.

              Visibility is stored as the _hidden_ ids rather than a map over every column, so
              a column that appears later (a new evaluator, a new metadata key) shows up by
              default.

          default_sort: Default sort for this experiment's evaluations list, as a `sort`-param string: a
              comma-separated, ordered list of fields where the first is the primary sort and
              the rest break ties (leading '-' on a field = descending), e.g.
              '-evaluators.reward.mean,cost_usd.mean'. Defaults to '-created_at'. Accepts any
              field the evaluations list `sort` param does; clients apply it as the list
              `sort` param.

          description: Human-readable purpose of the experiment.

          insight_id: Reference to an external insight that seeded this experiment, if any.

          is_favorite: Whether this Experiment is marked as a favorite. Defaults to false on create;
              omit on update to preserve the existing value.

          metadata: Free-form producer metadata for the experiment.

          pareto: Default X/Y metrics for a group's cost-vs-accuracy Pareto view.

              Metric ids use the same vocabulary as the evaluations list sort/filter fields —
              `cost_usd`, `latency_ms`, or `evaluators.<name>`. Defaults to cost (x) vs
              latency (y): both exist for every group, so the chart always has something to
              render before anyone customizes it.

          show_evaluations_over_time: Whether Studio should display this Experiment's Evaluation results over time.
              Defaults to false on create; omit on update to preserve the existing value.

          summary: Human- or agent-authored summary of the experiment's findings.

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
                "/apis/intake/v2/workspaces/{workspace}/experiments/{path_name}",
                workspace=workspace,
                path_name=path_name,
            ),
            body=maybe_transform(
                {
                    "body_name": body_name,
                    "baseline_evaluation_name": baseline_evaluation_name,
                    "column_layout": column_layout,
                    "default_sort": default_sort,
                    "description": description,
                    "insight_id": insight_id,
                    "is_favorite": is_favorite,
                    "metadata": metadata,
                    "pareto": pareto,
                    "show_evaluations_over_time": show_evaluations_over_time,
                    "summary": summary,
                },
                experiment_update_params.ExperimentUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ExperimentResponse,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: ExperimentFilterParam | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: Literal["-created_at", "created_at", "-updated_at", "updated_at", "-name", "name"] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> SyncDefaultPagination[ExperimentResponse]:
        """
        List Experiments

        Args:
          filter: Filter experiments by name, insight_id, is_favorite, show_evaluations_over_time,
              baseline_evaluation_name, is_deleted, or a metadata key/value
              (filter[metadata.<key>]=<value>). Pass is_deleted=true to return only
              soft-deleted experiments; omit to see only live ones.

          page: Page number.

          page_size: Page size.

          sort: Sort field; prefix with '-' for descending.

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
            path_template("/apis/intake/v2/workspaces/{workspace}/experiments", workspace=workspace),
            page=SyncDefaultPagination[ExperimentResponse],
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
                    experiment_list_params.ExperimentListParams,
                ),
            ),
            model=ExperimentResponse,
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
        Delete Experiment

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
            path_template("/apis/intake/v2/workspaces/{workspace}/experiments/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
        )


class AsyncExperimentsResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncExperimentsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncExperimentsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncExperimentsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncExperimentsResourceWithStreamingResponse(self)

    async def create(
        self,
        *,
        workspace: str | None = None,
        name: str,
        column_layout: ColumnLayoutParam | Omit = omit,
        default_sort: str | Omit = omit,
        description: str | Omit = omit,
        insight_id: str | Omit = omit,
        is_favorite: bool | Omit = omit,
        metadata: Dict[str, str] | Omit = omit,
        pareto: ParetoConfigParam | Omit = omit,
        show_evaluations_over_time: bool | Omit = omit,
        summary: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ExperimentResponse:
        """
        Create Experiment

        Args:
          name: Workspace-unique experiment name.

          column_layout: A saved table layout for a group's evaluations list: column order and which
              columns are hidden.

              Column ids are Studio's and cannot be enumerated here — the table builds a
              column per evaluator and metadata key found in the rows — so ids are stored and
              echoed back unvalidated.

              Visibility is stored as the _hidden_ ids rather than a map over every column, so
              a column that appears later (a new evaluator, a new metadata key) shows up by
              default.

          default_sort: Default sort for this experiment's evaluations list, as a `sort`-param string: a
              comma-separated, ordered list of fields where the first is the primary sort and
              the rest break ties (leading '-' on a field = descending), e.g.
              '-evaluators.reward.mean,cost_usd.mean'. Defaults to '-created_at'. Accepts any
              field the evaluations list `sort` param does; clients apply it as the list
              `sort` param.

          description: Human-readable purpose of the experiment.

          insight_id: Reference to an external insight that seeded this experiment, if any.

          is_favorite: Whether this Experiment is marked as a favorite. Defaults to false on create;
              omit on update to preserve the existing value.

          metadata: Free-form producer metadata for the experiment.

          pareto: Default X/Y metrics for a group's cost-vs-accuracy Pareto view.

              Metric ids use the same vocabulary as the evaluations list sort/filter fields —
              `cost_usd`, `latency_ms`, or `evaluators.<name>`. Defaults to cost (x) vs
              latency (y): both exist for every group, so the chart always has something to
              render before anyone customizes it.

          show_evaluations_over_time: Whether Studio should display this Experiment's Evaluation results over time.
              Defaults to false on create; omit on update to preserve the existing value.

          summary: Human- or agent-authored summary of the experiment's findings.


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
                path_template("/apis/intake/v2/workspaces/{workspace}/experiments", workspace=workspace),
                body=await async_maybe_transform(
                    {
                        "name": name,
                        "column_layout": column_layout,
                        "default_sort": default_sort,
                        "description": description,
                        "insight_id": insight_id,
                        "is_favorite": is_favorite,
                        "metadata": metadata,
                        "pareto": pareto,
                        "show_evaluations_over_time": show_evaluations_over_time,
                        "summary": summary,
                    },
                    experiment_create_params.ExperimentCreateParams,
                ),
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=ExperimentResponse,
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
    ) -> ExperimentResponse:
        """
        Get Experiment

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
            path_template("/apis/intake/v2/workspaces/{workspace}/experiments/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ExperimentResponse,
        )

    async def update(
        self,
        path_name: str,
        *,
        workspace: str | None = None,
        body_name: str,
        baseline_evaluation_name: Optional[str] | Omit = omit,
        column_layout: ColumnLayoutParam | Omit = omit,
        default_sort: str | Omit = omit,
        description: str | Omit = omit,
        insight_id: str | Omit = omit,
        is_favorite: bool | Omit = omit,
        metadata: Dict[str, str] | Omit = omit,
        pareto: ParetoConfigParam | Omit = omit,
        show_evaluations_over_time: bool | Omit = omit,
        summary: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ExperimentResponse:
        """
        Update Experiment

        Args:
          body_name: Workspace-unique experiment name.

          baseline_evaluation_name: Name of this Experiment's baseline Evaluation. The Evaluation must already be a
              live member of the Experiment. Set null to clear the selected baseline.

          column_layout: A saved table layout for a group's evaluations list: column order and which
              columns are hidden.

              Column ids are Studio's and cannot be enumerated here — the table builds a
              column per evaluator and metadata key found in the rows — so ids are stored and
              echoed back unvalidated.

              Visibility is stored as the _hidden_ ids rather than a map over every column, so
              a column that appears later (a new evaluator, a new metadata key) shows up by
              default.

          default_sort: Default sort for this experiment's evaluations list, as a `sort`-param string: a
              comma-separated, ordered list of fields where the first is the primary sort and
              the rest break ties (leading '-' on a field = descending), e.g.
              '-evaluators.reward.mean,cost_usd.mean'. Defaults to '-created_at'. Accepts any
              field the evaluations list `sort` param does; clients apply it as the list
              `sort` param.

          description: Human-readable purpose of the experiment.

          insight_id: Reference to an external insight that seeded this experiment, if any.

          is_favorite: Whether this Experiment is marked as a favorite. Defaults to false on create;
              omit on update to preserve the existing value.

          metadata: Free-form producer metadata for the experiment.

          pareto: Default X/Y metrics for a group's cost-vs-accuracy Pareto view.

              Metric ids use the same vocabulary as the evaluations list sort/filter fields —
              `cost_usd`, `latency_ms`, or `evaluators.<name>`. Defaults to cost (x) vs
              latency (y): both exist for every group, so the chart always has something to
              render before anyone customizes it.

          show_evaluations_over_time: Whether Studio should display this Experiment's Evaluation results over time.
              Defaults to false on create; omit on update to preserve the existing value.

          summary: Human- or agent-authored summary of the experiment's findings.

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
                "/apis/intake/v2/workspaces/{workspace}/experiments/{path_name}",
                workspace=workspace,
                path_name=path_name,
            ),
            body=await async_maybe_transform(
                {
                    "body_name": body_name,
                    "baseline_evaluation_name": baseline_evaluation_name,
                    "column_layout": column_layout,
                    "default_sort": default_sort,
                    "description": description,
                    "insight_id": insight_id,
                    "is_favorite": is_favorite,
                    "metadata": metadata,
                    "pareto": pareto,
                    "show_evaluations_over_time": show_evaluations_over_time,
                    "summary": summary,
                },
                experiment_update_params.ExperimentUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ExperimentResponse,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: ExperimentFilterParam | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: Literal["-created_at", "created_at", "-updated_at", "updated_at", "-name", "name"] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> AsyncPaginator[ExperimentResponse, AsyncDefaultPagination[ExperimentResponse]]:
        """
        List Experiments

        Args:
          filter: Filter experiments by name, insight_id, is_favorite, show_evaluations_over_time,
              baseline_evaluation_name, is_deleted, or a metadata key/value
              (filter[metadata.<key>]=<value>). Pass is_deleted=true to return only
              soft-deleted experiments; omit to see only live ones.

          page: Page number.

          page_size: Page size.

          sort: Sort field; prefix with '-' for descending.

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
            path_template("/apis/intake/v2/workspaces/{workspace}/experiments", workspace=workspace),
            page=AsyncDefaultPagination[ExperimentResponse],
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
                    experiment_list_params.ExperimentListParams,
                ),
            ),
            model=ExperimentResponse,
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
        Delete Experiment

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
            path_template("/apis/intake/v2/workspaces/{workspace}/experiments/{name}", workspace=workspace, name=name),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
        )


class ExperimentsResourceWithRawResponse:
    def __init__(self, experiments: ExperimentsResource) -> None:
        self._experiments = experiments

        self.create = to_raw_response_wrapper(
            experiments.create,
        )
        self.retrieve = to_raw_response_wrapper(
            experiments.retrieve,
        )
        self.update = to_raw_response_wrapper(
            experiments.update,
        )
        self.list = to_raw_response_wrapper(
            experiments.list,
        )
        self.delete = to_raw_response_wrapper(
            experiments.delete,
        )


class AsyncExperimentsResourceWithRawResponse:
    def __init__(self, experiments: AsyncExperimentsResource) -> None:
        self._experiments = experiments

        self.create = async_to_raw_response_wrapper(
            experiments.create,
        )
        self.retrieve = async_to_raw_response_wrapper(
            experiments.retrieve,
        )
        self.update = async_to_raw_response_wrapper(
            experiments.update,
        )
        self.list = async_to_raw_response_wrapper(
            experiments.list,
        )
        self.delete = async_to_raw_response_wrapper(
            experiments.delete,
        )


class ExperimentsResourceWithStreamingResponse:
    def __init__(self, experiments: ExperimentsResource) -> None:
        self._experiments = experiments

        self.create = to_streamed_response_wrapper(
            experiments.create,
        )
        self.retrieve = to_streamed_response_wrapper(
            experiments.retrieve,
        )
        self.update = to_streamed_response_wrapper(
            experiments.update,
        )
        self.list = to_streamed_response_wrapper(
            experiments.list,
        )
        self.delete = to_streamed_response_wrapper(
            experiments.delete,
        )


class AsyncExperimentsResourceWithStreamingResponse:
    def __init__(self, experiments: AsyncExperimentsResource) -> None:
        self._experiments = experiments

        self.create = async_to_streamed_response_wrapper(
            experiments.create,
        )
        self.retrieve = async_to_streamed_response_wrapper(
            experiments.retrieve,
        )
        self.update = async_to_streamed_response_wrapper(
            experiments.update,
        )
        self.list = async_to_streamed_response_wrapper(
            experiments.list,
        )
        self.delete = async_to_streamed_response_wrapper(
            experiments.delete,
        )
