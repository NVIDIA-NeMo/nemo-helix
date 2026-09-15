# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import get_args, get_origin

import pytest
from nemo_platform_plugin.client.types import BinaryContent, Paginated, PreparedRequest
from nemo_platform_plugin.evaluator import endpoints
from nemo_platform_plugin.evaluator.types import (
    AgentEvalJob,
    AgentEvalResult,
    BundledMetricOutputSpec,
    CreateMetricRequest,
    CreateTaskRequest,
    CreateTasksetRequest,
    EvalResult,
    EvaluateJob,
    EvaluatorHealth,
    HelloResponse,
    InlineMetricPayload,
    Metric,
    ReplaceTaskRequest,
    ReplaceTasksetRequest,
    RetrieveEvalFilesetRef,
    RetrieveEvalInputSpec,
    RetrieveEvalJob,
    RetrieveEvalModel,
    RetrieveEvalModelRef,
    RetrieveEvalRetrievalInputSpec,
    Revision,
    SubmitAgentEvalJobRequest,
    SubmitEvaluateJobRequest,
    SubmitRetrieveEvalJobRequest,
    Task,
    Taskset,
)
from nemo_platform_plugin.jobs.schemas import (
    PlatformJobLog,
    PlatformJobResultResponse,
    PlatformJobStatus,
    PlatformJobStatusResponse,
)
from nemo_platform_plugin.jobs.types import PlatformJobListResultResponse
from pydantic import JsonValue, ValidationError


def _assert_paginated_model(response_type: object, model_type: type[object]) -> None:
    assert get_origin(response_type) is Paginated
    assert get_args(response_type)[0] is model_type


def _metric_create_request() -> CreateMetricRequest:
    value_json_schema: dict[str, JsonValue] = {"type": "number"}
    metric: dict[str, JsonValue] = {
        "type": "exact-match",
        "reference": "{{item.expected}}",
        "candidate": "{{item.output}}",
    }
    return CreateMetricRequest(
        metric_type="exact-match",
        outputs=[BundledMetricOutputSpec(name="score", value_json_schema=value_json_schema)],
        payload=InlineMetricPayload(kind="inline", metric=metric),
    )


def test_submit_evaluate_job_endpoint_shape() -> None:
    body = SubmitEvaluateJobRequest(spec={"metrics": [], "dataset": []})
    prepared = endpoints.submit_evaluate_job(workspace="team-a", body=body)

    assert isinstance(prepared, PreparedRequest)
    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/evaluator/v2/workspaces/{workspace}/evaluate/jobs"
    assert prepared.path_params == {"workspace": "team-a"}
    assert isinstance(prepared.content, bytes)
    assert json.loads(prepared.content) == {"spec": {"metrics": [], "dataset": []}}
    assert prepared.content_type == "application/json"
    assert prepared.response_type is EvaluateJob


def test_submit_agent_eval_job_endpoint_shape() -> None:
    prepared = endpoints.submit_agent_eval_job(
        workspace="team-a",
        body=SubmitAgentEvalJobRequest(spec={"tasks": "default/suite", "target": {"kind": "gym"}}),
    )

    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/evaluator/v2/workspaces/{workspace}/agent-evaluate/jobs"
    assert prepared.path_params == {"workspace": "team-a"}
    assert prepared.response_type is AgentEvalJob


def test_health_endpoint_shape() -> None:
    prepared = endpoints.get_health()

    assert prepared.method == "GET"
    assert prepared.path_template == "/apis/evaluator/v1/healthz"
    assert prepared.path_params == {}
    assert prepared.response_type is EvaluatorHealth


def test_hello_endpoint_shape() -> None:
    prepared = endpoints.hello(name="ada")

    assert prepared.method == "GET"
    assert prepared.path_template == "/apis/evaluator/v1/hello/{name}"
    assert prepared.path_params == {"name": "ada"}
    assert prepared.response_type is HelloResponse


def test_evaluate_job_and_download_endpoint_shapes() -> None:
    job = endpoints.get_evaluate_job(workspace="team-a", name="job-1")
    jobs = endpoints.list_evaluate_jobs(
        workspace="team-a",
        query_params={"page": 2, "page_size": 25, "sort": "-created_at", "filter": '{"status":"completed"}'},
    )
    status = endpoints.get_evaluate_job_status(workspace="team-a", name="job-1")
    deleted = endpoints.delete_evaluate_job(workspace="team-a", name="job-1")
    cancelled = endpoints.cancel_evaluate_job(workspace="team-a", name="job-1")
    logs = endpoints.list_evaluate_job_logs(workspace="team-a", name="job-1", query_params={"tail": 10})
    results = endpoints.list_evaluate_job_results(workspace="team-a", name="job-1")
    result = endpoints.get_evaluate_job_result(workspace="team-a", job="job-1", name="aggregate-scores")
    download = endpoints.download_evaluate_job_result(workspace="team-a", job="job-1", name="aggregate-scores")

    assert job.method == "GET"
    assert job.path_template == "/apis/evaluator/v2/workspaces/{workspace}/evaluate/jobs/{name}"
    assert job.path_params == {"workspace": "team-a", "name": "job-1"}
    assert job.response_type is EvaluateJob
    assert jobs.query_params == {"page": 2, "page_size": 25, "sort": "-created_at", "filter": '{"status":"completed"}'}
    _assert_paginated_model(jobs.response_type, EvaluateJob)
    assert status.path_template.endswith("/evaluate/jobs/{name}/status")
    assert status.response_type is PlatformJobStatusResponse
    assert deleted.response_type is None
    assert cancelled.path_template.endswith("/evaluate/jobs/{name}/cancel")
    assert cancelled.response_type is EvaluateJob
    _assert_paginated_model(logs.response_type, PlatformJobLog)
    assert logs.query_params == {"tail": 10}
    assert results.path_template.endswith("/evaluate/jobs/{name}/results")
    assert results.response_type is PlatformJobListResultResponse
    assert result.path_template.endswith("/evaluate/jobs/{job}/results/{name}")
    assert result.path_params == {"workspace": "team-a", "job": "job-1", "name": "aggregate-scores"}
    assert result.response_type is PlatformJobResultResponse
    assert download.path_template.endswith("/evaluate/jobs/{job}/results/{name}/download")
    assert download.response_type is BinaryContent


def test_agent_eval_job_endpoint_shapes() -> None:
    job = endpoints.get_agent_eval_job(workspace="team-a", name="job-1")
    jobs = endpoints.list_agent_eval_jobs(
        workspace="team-a",
        query_params={"page": 2, "page_size": 25, "sort": "-created_at", "filter": '{"status":"completed"}'},
    )
    status = endpoints.get_agent_eval_job_status(workspace="team-a", name="job-1")
    deleted = endpoints.delete_agent_eval_job(workspace="team-a", name="job-1")
    cancelled = endpoints.cancel_agent_eval_job(workspace="team-a", name="job-1")
    logs = endpoints.list_agent_eval_job_logs(workspace="team-a", name="job-1", query_params={"limit": 10})
    results = endpoints.list_agent_eval_job_results(workspace="team-a", name="job-1")
    result = endpoints.get_agent_eval_job_result(workspace="team-a", job="job-1", name="summary")
    download = endpoints.download_agent_eval_job_result(workspace="team-a", job="job-1", name="summary")

    assert job.method == "GET"
    assert job.path_template == "/apis/evaluator/v2/workspaces/{workspace}/agent-evaluate/jobs/{name}"
    assert job.path_params == {"workspace": "team-a", "name": "job-1"}
    assert job.response_type is AgentEvalJob
    assert jobs.query_params == {"page": 2, "page_size": 25, "sort": "-created_at", "filter": '{"status":"completed"}'}
    _assert_paginated_model(jobs.response_type, AgentEvalJob)
    assert status.path_template.endswith("/agent-evaluate/jobs/{name}/status")
    assert status.response_type is PlatformJobStatusResponse
    assert deleted.response_type is None
    assert cancelled.response_type is AgentEvalJob
    _assert_paginated_model(logs.response_type, PlatformJobLog)
    assert logs.query_params == {"limit": 10}
    assert results.response_type is PlatformJobListResultResponse
    assert result.path_template.endswith("/agent-evaluate/jobs/{job}/results/{name}")
    assert result.response_type is PlatformJobResultResponse
    assert download.path_template.endswith("/agent-evaluate/jobs/{job}/results/{name}/download")
    assert download.response_type is BinaryContent


def test_retrieve_eval_job_endpoint_shapes() -> None:
    submitted = endpoints.submit_retrieve_eval_job(
        workspace="team-a",
        body=SubmitRetrieveEvalJobRequest(
            name="retrieval-smoke",
            description="BEIR smoke",
            project="search",
            spec=RetrieveEvalInputSpec(
                dataset=RetrieveEvalFilesetRef("team-a/beir"),
                target=RetrieveEvalModelRef("team-a/embedder"),
                k=[1, 10],
            ),
            profile="cpu-small",
            options={"priority": "low"},
            ownership={"principal_id": "principal-1"},
            custom_fields={"suite": "nightly"},
            output_location="retrieval-results",
        ),
    )
    job = endpoints.get_retrieve_eval_job(workspace="team-a", name="job-1")
    jobs = endpoints.list_retrieve_eval_jobs(workspace="team-a", query_params={"page": 2})
    status = endpoints.get_retrieve_eval_job_status(workspace="team-a", name="job-1")
    deleted = endpoints.delete_retrieve_eval_job(workspace="team-a", name="job-1")
    cancelled = endpoints.cancel_retrieve_eval_job(workspace="team-a", name="job-1")
    logs = endpoints.list_retrieve_eval_job_logs(workspace="team-a", name="job-1", query_params={"tail": 10})
    results = endpoints.list_retrieve_eval_job_results(workspace="team-a", name="job-1")
    result = endpoints.get_retrieve_eval_job_result(workspace="team-a", job="job-1", name="eval-results")
    download = endpoints.download_retrieve_eval_job_result(workspace="team-a", job="job-1", name="eval-results")

    assert submitted.method == "POST"
    assert submitted.path_template == "/apis/evaluator/v2/workspaces/{workspace}/retrieve-eval/jobs"
    assert isinstance(submitted.content, bytes)
    assert json.loads(submitted.content) == {
        "name": "retrieval-smoke",
        "description": "BEIR smoke",
        "project": "search",
        "spec": {"dataset": "team-a/beir", "target": "team-a/embedder", "k": [1, 10]},
        "profile": "cpu-small",
        "options": {"priority": "low"},
        "ownership": {"principal_id": "principal-1"},
        "custom_fields": {"suite": "nightly"},
        "output_location": "retrieval-results",
    }
    assert submitted.response_type is RetrieveEvalJob
    assert job.path_template.endswith("/retrieve-eval/jobs/{name}")
    assert job.response_type is RetrieveEvalJob
    _assert_paginated_model(jobs.response_type, RetrieveEvalJob)
    assert status.response_type is PlatformJobStatusResponse
    assert deleted.response_type is None
    assert cancelled.response_type is RetrieveEvalJob
    _assert_paginated_model(logs.response_type, PlatformJobLog)
    assert results.response_type is PlatformJobListResultResponse
    assert result.response_type is PlatformJobResultResponse
    assert download.path_template.endswith("/retrieve-eval/jobs/{job}/results/{name}/download")
    assert download.response_type is BinaryContent


def test_retrieve_eval_job_models_match_job_route_contract() -> None:
    job = RetrieveEvalJob.model_validate(
        {
            "id": "job-id",
            "name": "retrieval-smoke",
            "description": "BEIR smoke",
            "project": "search",
            "workspace": "team-a",
            "spec": {
                "dataset": "team-a/beir",
                "target": {
                    "embeddings": {
                        "url": "https://igw.example.test/v1/embeddings",
                        "name": "embedder",
                    },
                    "first_stage_k": 50,
                },
                "k": [10, 1],
            },
            "status": "completed",
            "status_details": {"progress": 1.0},
            "error_details": None,
            "ownership": {"principal_id": "principal-1"},
            "custom_fields": {"suite": "nightly"},
        }
    )

    assert job.name == "retrieval-smoke"
    assert job.spec.dataset.root == "team-a/beir"
    assert job.spec.target.embeddings.name == "embedder"
    assert job.spec.target.first_stage_k == 50
    assert job.spec.k == [1, 10]
    assert job.status is PlatformJobStatus.COMPLETED
    assert job.status_details == {"progress": 1.0}
    assert job.ownership == {"principal_id": "principal-1"}
    assert job.custom_fields == {"suite": "nightly"}

    with pytest.raises(ValidationError):
        RetrieveEvalJob.model_validate({"status": "completed", "spec": {}})

    with pytest.raises(ValidationError):
        SubmitRetrieveEvalJobRequest.model_validate({"spec": {}})

    with pytest.raises(ValidationError):
        SubmitRetrieveEvalJobRequest(
            spec=RetrieveEvalInputSpec(
                dataset=RetrieveEvalFilesetRef("team-a/beir"), target=RetrieveEvalModelRef("bad")
            ),
        )

    with pytest.raises(ValidationError):
        SubmitRetrieveEvalJobRequest(
            spec=RetrieveEvalInputSpec(
                dataset=RetrieveEvalFilesetRef("team-a/beir"),
                target=RetrieveEvalRetrievalInputSpec(
                    embeddings=RetrieveEvalModel(url="https://igw.example.test/v1/embeddings", name="embedder"),
                    first_stage_k=0,
                ),
            ),
        )

    with pytest.raises(ValidationError):
        SubmitRetrieveEvalJobRequest(
            spec=RetrieveEvalInputSpec(
                dataset=RetrieveEvalFilesetRef("team-a/beir"),
                target=RetrieveEvalModelRef("team-a/embedder"),
                k=[1, 1],
            )
        )

    with pytest.raises(ValidationError):
        SubmitRetrieveEvalJobRequest(
            spec=RetrieveEvalInputSpec(
                dataset=RetrieveEvalFilesetRef("team-a/beir"),
                target=RetrieveEvalModelRef("team-a/embedder"),
            ),
            output_location="workspace/fileset",
        )


def test_metric_endpoint_shapes() -> None:
    body = _metric_create_request()
    metric_create = endpoints.create_metric(
        workspace="team-a",
        name="accuracy",
        body=body,
        query_params={"project": "proj-a"},
    )

    assert metric_create.method == "POST"
    assert metric_create.path_template == "/apis/evaluator/v2/workspaces/{workspace}/metrics/{name}"
    assert metric_create.path_params == {"workspace": "team-a", "name": "accuracy"}
    assert metric_create.query_params == {"project": "proj-a"}
    assert isinstance(metric_create.content, bytes)
    assert json.loads(metric_create.content) == {
        "metric_type": "exact-match",
        "outputs": [{"name": "score", "value_json_schema": {"type": "number"}}],
        "payload": {
            "kind": "inline",
            "metric": {
                "type": "exact-match",
                "reference": "{{item.expected}}",
                "candidate": "{{item.output}}",
            },
        },
    }
    assert metric_create.response_type is Metric
    assert metric_create.client_options is None
    assert metric_create.on_conflict_get is None
    _assert_paginated_model(endpoints.list_metrics(workspace="team-a").response_type, Metric)
    assert endpoints.get_metric(workspace="team-a", name="accuracy").response_type is Metric
    assert endpoints.delete_metric(workspace="team-a", name="accuracy").response_type is None


def test_task_endpoint_shapes() -> None:
    created = endpoints.create_task(
        workspace="team-a",
        name="task-1",
        body=CreateTaskRequest(root={"metrics": []}),
        query_params={"project": "proj-a"},
    )
    replaced = endpoints.replace_task(
        workspace="team-a",
        name="task-1",
        body=ReplaceTaskRequest(root={"metrics": []}),
    )
    revisions = endpoints.list_task_revisions(workspace="team-a", name="task-1", query_params={"page": 2})
    revision = endpoints.get_task_revision(workspace="team-a", name="task-1", revision="sha")
    tagged = endpoints.tag_task_revision(
        workspace="team-a",
        name="task-1",
        tag="prod",
        query_params={"revision": "sha"},
    )

    assert created.method == "POST"
    assert created.path_template == "/apis/evaluator/v2/workspaces/{workspace}/tasks/{name}"
    assert created.query_params == {"project": "proj-a"}
    assert created.response_type is Task
    assert replaced.method == "PUT"
    assert replaced.response_type is Task
    assert endpoints.get_task(workspace="team-a", name="task-1").response_type is Task
    _assert_paginated_model(endpoints.list_tasks(workspace="team-a").response_type, Task)
    _assert_paginated_model(revisions.response_type, Revision)
    assert revision.path_template.endswith("/tasks/{name}/revisions/{revision}")
    assert revision.response_type is Task
    assert tagged.path_template.endswith("/tasks/{name}/tags/{tag}")
    assert tagged.query_params == {"revision": "sha"}
    assert endpoints.delete_task(workspace="team-a", name="task-1").response_type is None


def test_taskset_endpoint_shapes() -> None:
    created = endpoints.create_taskset(
        workspace="team-a",
        name="suite",
        body=CreateTasksetRequest(root={"tasks": []}),
    )
    replaced = endpoints.replace_taskset(
        workspace="team-a",
        name="suite",
        body=ReplaceTasksetRequest(root={"tasks": []}),
        query_params={"project": "proj-a"},
    )
    revisions = endpoints.list_taskset_revisions(workspace="team-a", name="suite")
    revision = endpoints.get_taskset_revision(workspace="team-a", name="suite", revision="sha")
    tagged = endpoints.tag_taskset_revision(
        workspace="team-a",
        name="suite",
        tag="prod",
        query_params={"revision": "sha"},
    )

    assert created.method == "POST"
    assert created.path_template == "/apis/evaluator/v2/workspaces/{workspace}/tasksets/{name}"
    assert created.response_type is Taskset
    assert replaced.method == "PUT"
    assert replaced.query_params == {"project": "proj-a"}
    assert replaced.response_type is Taskset
    assert endpoints.get_taskset(workspace="team-a", name="suite").response_type is Taskset
    _assert_paginated_model(endpoints.list_tasksets(workspace="team-a").response_type, Taskset)
    _assert_paginated_model(revisions.response_type, Revision)
    assert revision.path_template.endswith("/tasksets/{name}/revisions/{revision}")
    assert revision.response_type is Taskset
    assert tagged.path_template.endswith("/tasksets/{name}/tags/{tag}")
    assert endpoints.delete_taskset(workspace="team-a", name="suite").response_type is None


def test_result_endpoint_shapes() -> None:
    eval_result = endpoints.get_eval_result(workspace="team-a", name="result-1")
    eval_results = endpoints.list_eval_results(workspace="team-a", query_params={"page": 2})
    agent_result = endpoints.get_agent_eval_result(workspace="team-a", name="result-1")
    agent_results = endpoints.list_agent_eval_results(workspace="team-a", query_params={"sort": "-created_at"})

    assert eval_result.response_type is EvalResult
    _assert_paginated_model(eval_results.response_type, EvalResult)
    assert endpoints.delete_eval_result(workspace="team-a", name="result-1").response_type is None
    assert agent_result.response_type is AgentEvalResult
    _assert_paginated_model(agent_results.response_type, AgentEvalResult)
    assert endpoints.delete_agent_eval_result(workspace="team-a", name="result-1").response_type is None
