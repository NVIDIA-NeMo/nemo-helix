# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the Evaluator service.

Wraps the endpoint functions from ``evaluator.endpoints`` as direct methods
using the ``method()`` descriptor, following the files/models pattern.

The evaluator's high-level ``submit()`` convenience method (overloaded for row
vs. taskset evaluation) stays in the SDK layer — it packages parameters into a
job spec and calls ``submit_evaluate_job`` / ``submit_agent_eval_job`` here.
"""

from dataclasses import replace

from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.method import method
from nemo_platform_plugin.client.response import AsyncNemoBinaryResponse, NemoBinaryResponse, NemoResponse
from nemo_platform_plugin.client.types import PreparedRequest
from nemo_platform_plugin.evaluator import endpoints
from nemo_platform_plugin.evaluator.types import CreateMetricRequest, FlatQueryParams, Metric, ProjectQueryParams

_AGGREGATE_SCORES_RESULT_NAME = "aggregate-scores"
_ROW_SCORES_RESULT_NAME = "row-scores"
_ARTIFACTS_RESULT_NAME = "artifacts"


def _create_metric_request(
    *,
    workspace: str | None,
    name: str,
    body: CreateMetricRequest,
    query_params: ProjectQueryParams | FlatQueryParams | None,
    exist_ok: bool,
) -> PreparedRequest[Metric]:
    request = endpoints.create_metric(workspace=workspace, name=name, body=body, query_params=query_params)
    if not exist_ok:
        return request
    return replace(
        request,
        client_options={"exist_ok": True},
        on_conflict_get=endpoints.get_metric(workspace=workspace, name=name),
    )


class _EvaluatorMethods:
    get_health = method(endpoints.get_health)
    hello = method(endpoints.hello)

    list_evaluate_jobs = method(endpoints.list_evaluate_jobs)
    submit_evaluate_job = method(endpoints.submit_evaluate_job)
    get_evaluate_job = method(endpoints.get_evaluate_job)
    get_evaluate_job_status = method(endpoints.get_evaluate_job_status)
    delete_evaluate_job = method(endpoints.delete_evaluate_job)
    cancel_evaluate_job = method(endpoints.cancel_evaluate_job)
    list_evaluate_job_logs = method(endpoints.list_evaluate_job_logs)
    list_evaluate_job_results = method(endpoints.list_evaluate_job_results)
    get_evaluate_job_result = method(endpoints.get_evaluate_job_result)
    download_evaluate_job_result = method(endpoints.download_evaluate_job_result)

    list_agent_eval_jobs = method(endpoints.list_agent_eval_jobs)
    submit_agent_eval_job = method(endpoints.submit_agent_eval_job)
    get_agent_eval_job = method(endpoints.get_agent_eval_job)
    get_agent_eval_job_status = method(endpoints.get_agent_eval_job_status)
    delete_agent_eval_job = method(endpoints.delete_agent_eval_job)
    cancel_agent_eval_job = method(endpoints.cancel_agent_eval_job)
    list_agent_eval_job_logs = method(endpoints.list_agent_eval_job_logs)
    list_agent_eval_job_results = method(endpoints.list_agent_eval_job_results)
    get_agent_eval_job_result = method(endpoints.get_agent_eval_job_result)
    download_agent_eval_job_result = method(endpoints.download_agent_eval_job_result)

    list_retrieve_eval_jobs = method(endpoints.list_retrieve_eval_jobs)
    submit_retrieve_eval_job = method(endpoints.submit_retrieve_eval_job)
    get_retrieve_eval_job = method(endpoints.get_retrieve_eval_job)
    get_retrieve_eval_job_status = method(endpoints.get_retrieve_eval_job_status)
    delete_retrieve_eval_job = method(endpoints.delete_retrieve_eval_job)
    cancel_retrieve_eval_job = method(endpoints.cancel_retrieve_eval_job)
    list_retrieve_eval_job_logs = method(endpoints.list_retrieve_eval_job_logs)
    list_retrieve_eval_job_results = method(endpoints.list_retrieve_eval_job_results)
    get_retrieve_eval_job_result = method(endpoints.get_retrieve_eval_job_result)
    download_retrieve_eval_job_result = method(endpoints.download_retrieve_eval_job_result)

    get_metric = method(endpoints.get_metric)
    list_metrics = method(endpoints.list_metrics)
    delete_metric = method(endpoints.delete_metric)

    get_task = method(endpoints.get_task)
    list_tasks = method(endpoints.list_tasks)
    create_task = method(endpoints.create_task)
    replace_task = method(endpoints.replace_task)
    list_task_revisions = method(endpoints.list_task_revisions)
    get_task_revision = method(endpoints.get_task_revision)
    tag_task_revision = method(endpoints.tag_task_revision)
    delete_task = method(endpoints.delete_task)

    get_taskset = method(endpoints.get_taskset)
    list_tasksets = method(endpoints.list_tasksets)
    create_taskset = method(endpoints.create_taskset)
    replace_taskset = method(endpoints.replace_taskset)
    list_taskset_revisions = method(endpoints.list_taskset_revisions)
    get_taskset_revision = method(endpoints.get_taskset_revision)
    tag_taskset_revision = method(endpoints.tag_taskset_revision)
    delete_taskset = method(endpoints.delete_taskset)

    get_eval_result = method(endpoints.get_eval_result)
    list_eval_results = method(endpoints.list_eval_results)
    delete_eval_result = method(endpoints.delete_eval_result)
    get_agent_eval_result = method(endpoints.get_agent_eval_result)
    list_agent_eval_results = method(endpoints.list_agent_eval_results)
    delete_agent_eval_result = method(endpoints.delete_agent_eval_result)


class EvaluatorClient(_EvaluatorMethods, NemoClient):
    """Sync client for the Evaluator service API."""

    def download_evaluate_job_aggregate_scores(self, *, workspace: str | None = None, name: str) -> NemoBinaryResponse:
        """Download the aggregate-scores artifact for an evaluate job."""
        return self.download_evaluate_job_result(
            workspace=workspace,
            job=name,
            name=_AGGREGATE_SCORES_RESULT_NAME,
        )

    def download_evaluate_job_row_scores(self, *, workspace: str | None = None, name: str) -> NemoBinaryResponse:
        """Download the row-scores artifact for an evaluate job."""
        return self.download_evaluate_job_result(
            workspace=workspace,
            job=name,
            name=_ROW_SCORES_RESULT_NAME,
        )

    def download_evaluate_job_artifacts(self, *, workspace: str | None = None, name: str) -> NemoBinaryResponse:
        """Download the artifacts bundle for an evaluate job."""
        return self.download_evaluate_job_result(
            workspace=workspace,
            job=name,
            name=_ARTIFACTS_RESULT_NAME,
        )

    def create_metric(
        self,
        *,
        workspace: str | None = None,
        name: str,
        body: CreateMetricRequest,
        query_params: ProjectQueryParams | FlatQueryParams | None = None,
        exist_ok: bool = False,
    ) -> NemoResponse[Metric]:
        return self.send(
            _create_metric_request(
                workspace=workspace,
                name=name,
                body=body,
                query_params=query_params,
                exist_ok=exist_ok,
            )
        )


class AsyncEvaluatorClient(_EvaluatorMethods, AsyncNemoClient):
    """Async client for the Evaluator service API."""

    async def download_evaluate_job_aggregate_scores(
        self, *, workspace: str | None = None, name: str
    ) -> AsyncNemoBinaryResponse:
        """Download the aggregate-scores artifact for an evaluate job."""
        return await self.download_evaluate_job_result(
            workspace=workspace,
            job=name,
            name=_AGGREGATE_SCORES_RESULT_NAME,
        )

    async def download_evaluate_job_row_scores(
        self, *, workspace: str | None = None, name: str
    ) -> AsyncNemoBinaryResponse:
        """Download the row-scores artifact for an evaluate job."""
        return await self.download_evaluate_job_result(
            workspace=workspace,
            job=name,
            name=_ROW_SCORES_RESULT_NAME,
        )

    async def download_evaluate_job_artifacts(
        self, *, workspace: str | None = None, name: str
    ) -> AsyncNemoBinaryResponse:
        """Download the artifacts bundle for an evaluate job."""
        return await self.download_evaluate_job_result(
            workspace=workspace,
            job=name,
            name=_ARTIFACTS_RESULT_NAME,
        )

    async def create_metric(
        self,
        *,
        workspace: str | None = None,
        name: str,
        body: CreateMetricRequest,
        query_params: ProjectQueryParams | FlatQueryParams | None = None,
        exist_ok: bool = False,
    ) -> NemoResponse[Metric]:
        return await self.send(
            _create_metric_request(
                workspace=workspace,
                name=name,
                body=body,
                query_params=query_params,
                exist_ok=exist_ok,
            )
        )
