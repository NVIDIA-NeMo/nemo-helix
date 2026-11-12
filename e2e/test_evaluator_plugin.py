# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container E2E tests for the nemo-evals plugin.

The suite exercises the plugin through the public Platform SDK against an
external deployment configured through ``NHX_BASE_URL``. Durable evaluator
jobs execute in CPU task containers, so these tests intentionally do not mock
the evals service or job scheduler.
"""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlparse

import httpx
import pytest
from nemo_evals.api.schemas import MetricInline, MetricRef
from nemo_evals.filesets import FilesetRef
from nemo_evals.jobs.agent_spec import GymAgentSource, GymRunnerTarget
from nemo_evals.jobs.evaluate import EvaluateInputSpec
from nemo_evals.sdk.job_resources import EvaluatorJobResource
from nemo_evals.sdk.resources import Evaluator
from nemo_evals.shared.metric_bundles.bundles import bundle_metric
from nemo_evals.shared.metric_bundles.inline import InlineMetricBundlePackager
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NemoHTTPError, NemoTransportError
from nemo_helix_plugin.evals.client import EvaluatorClient
from nemo_helix_plugin.evals.types import SubmitAgentEvalJobRequest, SubmitEvaluateJobRequest
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetPurpose
from nemo_helix_plugin.inference_gateway.client import InferenceGatewayClient
from nemo_helix_plugin.inference_gateway.types import JsonBody
from nemo_helix_plugin.inference_middleware import BackendFormat
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import CreateModelEntityRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.testing import add_mock_provider, short_unique_name, wait_for_model_entity
from nhx.testing.e2e import wait_for_platform_job
from nhx.testing.utils import ensure_passthrough_virtual_model
from nhx_evals_sdk import (
    ExactMatchMetric,
    InferenceParams,
    Model,
    ModelRef,
    RunConfig,
    RunConfigOnlineModel,
)
from nhx_evals_sdk.agent_eval.results import AgentEvalSummary
from nhx_evals_sdk.agent_eval.runtimes.gym import GymRewardMetric, discover_gym_tasks
from nhx_evals_sdk.enums import ModelFormat
from nhx_evals_sdk.metrics.llm_judge import LLMJudgeMetric
from nhx_evals_sdk.metrics.string_check import StringCheckMetric
from nhx_evals_sdk.metrics.tool_calling import ToolCallingMetric
from nhx_evals_sdk.values.results import EvaluationResult
from nhx_evals_sdk.values.scores import JSONScoreParser, RangeScore

pytestmark = [
    pytest.mark.container_only,
    pytest.mark.timeout(1800),
]

EVALUATOR_JOB_TIMEOUT_SECONDS = 900.0
EVALUATOR_PENDING_TIMEOUT_SECONDS = 600.0
EVALUATOR_POLL_INTERVAL_SECONDS = 5.0
EVALUATOR_TRANSIENT_STATUS_CODES = frozenset({502, 503, 504})
DEFAULT_INTERNAL_PLATFORM_BASE_URL = "http://nemo-helix-api:8080"
IGW_ROUTE_TIMEOUT_SECONDS = 60.0
IGW_ROUTE_POLL_INTERVAL_SECONDS = 0.5
IGW_ROUTE_STABLE_SECONDS = 5.0
IGW_TRANSIENT_ROUTE_STATUSES = frozenset({404, 408, 425, 429, 500, 502, 503, 504})
EXACT_MATCH_AGGREGATE_SCORE_NAMES = frozenset({"exact-match", "exact-match.exact-match"})
EXACT_MATCH_ROW_SCORE_NAME = "exact-match"


def _exact_match_metric(*, candidate: str | None = "{{item.output}}") -> ExactMatchMetric:
    return ExactMatchMetric(reference="{{item.expected}}", candidate=candidate)


def _offline_rows() -> list[dict[str, object]]:
    return [
        {"input": "capital of France", "expected": "Paris", "output": "Paris"},
        {"input": "largest planet", "expected": "Jupiter", "output": "Saturn"},
    ]


def _chat_completion(content: str) -> dict[str, object]:
    return {
        "id": "chatcmpl-evaluator-e2e",
        "object": "chat.completion",
        "created": 0,
        "model": "mock-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


def _add_mock_provider_or_skip(
    client: NemoClient,
    *,
    workspace: str,
    name: str,
    mock_response_body: dict[str, object],
    should_autoprovision_virtual_model: bool = True,
) -> str:
    """Create an IGW mock provider (returning its name) or skip when the deployment does not support one."""
    try:
        return add_mock_provider(
            client,
            workspace=workspace,
            name=name,
            mock_response_body=mock_response_body,
            should_autoprovision_virtual_model=should_autoprovision_virtual_model,
        ).name
    except RuntimeError as exc:
        if "mock_provider_prefix is not configured" in str(exc):
            pytest.skip(
                "The running platform does not have mock-provider mode enabled. "
                "Configure NHX_INFERENCE_GATEWAY_MOCK_PROVIDER_PREFIX=igw-mock- to run this test."
            )
        raise
    except NemoHTTPError as exc:
        error_body = None if exc.body is None else json.dumps(exc.body, default=str)
        mock_mode_unavailable = (
            exc.status_code == 502 and error_body is not None and "Cannot connect to host mock.local" in error_body
        )
        if mock_mode_unavailable:
            pytest.skip(
                "IGW could not reach mock.local while configuring the mock provider. "
                "The running platform does not have mock-provider mode enabled."
            )
        raise


def _assert_http_status(exc: NemoHTTPError | httpx.HTTPStatusError, status_code: int) -> None:
    actual = exc.status_code if isinstance(exc, NemoHTTPError) else exc.response.status_code
    assert actual == status_code


def _aggregate_score(result: EvaluationResult) -> Any:
    for score in result.aggregate_scores.scores:
        if score.name in EXACT_MATCH_AGGREGATE_SCORE_NAMES:
            return score
    raise AssertionError(f"No exact-match aggregate score in {result.aggregate_scores.scores!r}")


def _rows_in_index_order(result: EvaluationResult) -> Sequence[Any]:
    """Order rows by explicit row index, preserving input order when it is absent."""
    return [
        row
        for _, row in sorted(
            enumerate(result.row_scores),
            key=lambda indexed_row: (
                indexed_row[1].row_index if indexed_row[1].row_index is not None else indexed_row[0]
            ),
        )
    ]


def _row_score_values(result: EvaluationResult) -> list[float]:
    values: list[float] = []
    seen_score_names: list[str] = []
    for row in _rows_in_index_order(result):
        for metric_scores in row.metrics.values():
            seen_score_names.extend(score.name for score in metric_scores)
            values.extend(float(score.value) for score in metric_scores if score.name == EXACT_MATCH_ROW_SCORE_NAME)
    if not values:
        raise AssertionError(f"No exact-match row scores found. Saw score names: {seen_score_names!r}")
    return values


def _internal_model_route(workspace: str, model_name: str) -> str:
    base_url = os.environ.get("NHX_E2E_INTERNAL_BASE_URL", DEFAULT_INTERNAL_PLATFORM_BASE_URL).rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    return f"{base_url}/apis/inference-gateway/v2/workspaces/{workspace}/model/{model_name}/-/v1"


def _evaluator(client: NemoClient) -> Evaluator:
    """High-level evaluator resource driven by the typed platform client."""
    return Evaluator(EvaluatorClient.from_client(client))


def _submit_evaluate_job(client: NemoClient, workspace: str, spec: Mapping[str, object]) -> str:
    """Submit a durable evaluate job spec and return the platform job name."""
    return (
        EvaluatorClient.from_client(client)
        .submit_evaluate_job(workspace=workspace, body=SubmitEvaluateJobRequest(spec=dict(spec)))
        .data()
        .name
    )


def _submit_agent_eval_job(client: NemoClient, workspace: str, spec: Mapping[str, object]) -> str:
    """Submit a durable agent-evaluate job spec and return the platform job name."""
    return (
        EvaluatorClient.from_client(client)
        .submit_agent_eval_job(workspace=workspace, body=SubmitAgentEvalJobRequest(spec=dict(spec)))
        .data()
        .name
    )


def _wait_for_stable_model_chat_route(client: NemoClient, workspace: str, model_name: str) -> None:
    """Require stable successful chat completions through the model-entity route."""
    gateway = InferenceGatewayClient.from_client(client)
    deadline = time.monotonic() + IGW_ROUTE_TIMEOUT_SECONDS
    stable_since: float | None = None
    last_status: int | None = None
    last_body = ""
    last_error: str | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            gateway.with_options(timeout=min(10.0, remaining)).model_post(
                trailing_uri="v1/chat/completions",
                name=model_name,
                workspace=workspace,
                body=JsonBody({"model": model_name, "messages": [{"role": "user", "content": "ping"}]}),
            )
        except NemoTransportError as exc:
            stable_since = None
            last_error = repr(exc)
        except NemoHTTPError as exc:
            stable_since = None
            last_status = exc.status_code
            last_body = str(exc)
            last_error = None
            if exc.status_code not in IGW_TRANSIENT_ROUTE_STATUSES:
                raise
        else:
            last_status = 200
            last_body = ""
            last_error = None
            if stable_since is None:
                stable_since = time.monotonic()
            if time.monotonic() - stable_since >= IGW_ROUTE_STABLE_SECONDS:
                return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(IGW_ROUTE_POLL_INTERVAL_SECONDS, remaining))
    if stable_since is not None:
        stable_for = time.monotonic() - stable_since
        detail = (
            f"{last_status} {last_body[:500]} (stable for {stable_for:.1f}s; required {IGW_ROUTE_STABLE_SECONDS:.1f}s)"
        )
    else:
        detail = last_error if last_error is not None else f"{last_status} {last_body[:500]}"
    raise TimeoutError(
        f"Model entity route {workspace}/{model_name} was not ready after "
        f"{IGW_ROUTE_TIMEOUT_SECONDS}s. Last response: {detail}"
    )


def _create_ready_mock_model(
    client: NemoClient,
    *,
    workspace: str,
    name: str,
    mock_response_body: dict[str, object],
) -> None:
    """Create a mock model and wait until its model-entity route is stable."""
    provider_name = _add_mock_provider_or_skip(
        client,
        workspace=workspace,
        name=name,
        mock_response_body=mock_response_body,
        should_autoprovision_virtual_model=False,
    )
    ModelsClient.from_client(client).create_model(
        workspace=workspace,
        body=CreateModelEntityRequest(
            name=name,
            backend_format=BackendFormat.OPENAI_CHAT,
            model_providers=[f"{workspace}/{provider_name}"],
        ),
        exist_ok=True,
    ).data()
    wait_for_model_entity(
        client,
        workspace,
        name,
        ensure_virtual_model=True,
        should_autoprovision_virtual_model=False,
    )
    ensure_passthrough_virtual_model(
        client,
        workspace,
        name,
        timeout=IGW_ROUTE_TIMEOUT_SECONDS,
        autoprovisioned=False,
    )
    _wait_for_stable_model_chat_route(client, workspace, name)


def _cleanup_evaluator_job(client: NemoClient, job_name: str) -> None:
    jobs = JobsClient.from_client(client)
    with suppress(Exception):
        jobs.cancel_job(name=job_name, workspace=client.workspace)
    with suppress(Exception):
        jobs.delete_job(name=job_name, workspace=client.workspace)


def _wait_for_evaluator_job(job: EvaluatorJobResource) -> None:
    started_at = time.monotonic()
    while True:
        remaining = EVALUATOR_JOB_TIMEOUT_SECONDS - (time.monotonic() - started_at)
        if remaining <= 0:
            raise TimeoutError(f"Evaluator job {job.name!r} did not complete after {EVALUATOR_JOB_TIMEOUT_SECONDS}s.")
        try:
            job.wait_until_done(
                poll_interval_seconds=EVALUATOR_POLL_INTERVAL_SECONDS,
                job_timeout_seconds=remaining,
                pending_timeout_seconds=min(EVALUATOR_PENDING_TIMEOUT_SECONDS, remaining),
            )
            return
        except NemoHTTPError as exc:
            if exc.status_code not in EVALUATOR_TRANSIENT_STATUS_CODES:
                raise
            remaining = EVALUATOR_JOB_TIMEOUT_SECONDS - (time.monotonic() - started_at)
            if remaining <= 0:
                raise
            time.sleep(min(EVALUATOR_POLL_INTERVAL_SECONDS, remaining))


def _submit_input_spec(client: NemoClient, spec: EvaluateInputSpec) -> EvaluatorJobResource:
    job_name = _submit_evaluate_job(client, client.require_workspace(), spec.model_dump(mode="json"))
    return _evaluator(client).get_job_resource(job_name)


def _metric_output_values(result: EvaluationResult, name: str) -> list[float]:
    values: list[float] = []
    for row in _rows_in_index_order(result):
        for outputs in row.metrics.values():
            for output in outputs:
                if output.name == name:
                    values.append(float(output.value))
    return values


@pytest.fixture(scope="module")
def evaluator_workspace(client: NemoClient) -> Iterator[str]:
    workspaces = WorkspacesClient.from_client(client)
    name = short_unique_name("e2e-eval")
    try:
        workspaces.create_workspace(body=CreateWorkspaceRequest(name=name)).data()
        yield name
    finally:
        with suppress(Exception):
            workspaces.delete_workspace(name=name).data()


@pytest.fixture(scope="module")
def evaluator_client(client: NemoClient, evaluator_workspace: str) -> Iterator[NemoClient]:
    yield client.with_workspace(evaluator_workspace).with_options(timeout=EVALUATOR_JOB_TIMEOUT_SECONDS)


@pytest.fixture(scope="module")
def completed_offline_job(evaluator_client: NemoClient) -> Iterator[EvaluatorJobResource]:
    job = _evaluator(evaluator_client).submit(
        metric=_exact_match_metric(),
        dataset=_offline_rows(),
        config=RunConfig(parallelism=1),
    )
    try:
        _wait_for_evaluator_job(job)
        yield job
    finally:
        _cleanup_evaluator_job(evaluator_client, job.name)


def test_health_check(client: NemoClient) -> None:
    status = _evaluator(client).plugin_status()

    assert status["plugin"] == "evals"
    assert status["status"] == "ok"
    jobs = status["jobs"]
    assert isinstance(jobs, list)
    assert "evals.evaluate" in jobs


def test_stored_metric_lifecycle(evaluator_client: NemoClient) -> None:
    name = short_unique_name("exact")
    try:
        created = _evaluator(evaluator_client).metrics.create(name, metric=_exact_match_metric())

        assert created.name == name
        assert created.workspace == evaluator_client.workspace
        assert created.metric_type == "exact-match"
        assert created.payload_kind == "inline"
        assert created.bundle_ref

        retrieved = _evaluator(evaluator_client).metrics.retrieve(name)
        assert retrieved.id == created.id
        assert retrieved.metric_type == "exact-match"
        assert retrieved.payload_digest == created.payload_digest

        listing = _evaluator(evaluator_client).metrics.list(sort="-created_at")
        assert any(metric.name == name for metric in listing.data)

        with pytest.raises((httpx.HTTPStatusError, NemoHTTPError)) as exc_info:
            _evaluator(evaluator_client).metrics.create(name, metric=_exact_match_metric())
        _assert_http_status(exc_info.value, 409)

        _evaluator(evaluator_client).metrics.delete(name)
        with pytest.raises((httpx.HTTPStatusError, NemoHTTPError)) as exc_info:
            _evaluator(evaluator_client).metrics.retrieve(name)
        _assert_http_status(exc_info.value, 404)
    finally:
        with suppress(Exception):
            _evaluator(evaluator_client).metrics.delete(name)


def test_offline_evaluate_job_lifecycle(
    completed_offline_job: EvaluatorJobResource,
    tmp_path: Path,
) -> None:
    status = completed_offline_job.get_job_status()
    assert status.status == "completed"

    result = completed_offline_job.get_result()
    aggregate = _aggregate_score(result)
    assert aggregate.count == 2
    assert aggregate.mean == 0.5
    assert _row_score_values(result) == [1.0, 0.0]

    artifact_dir = completed_offline_job.download_artifacts(tmp_path)
    aggregate_files = list(artifact_dir.rglob("aggregate-scores.json"))
    row_score_files = list(artifact_dir.rglob("row-scores.jsonl"))

    assert aggregate_files, f"aggregate-scores.json missing under {artifact_dir}"
    assert row_score_files, f"row-scores.jsonl missing under {artifact_dir}"

    aggregate_payload = json.loads(aggregate_files[0].read_text(encoding="utf-8"))
    row_lines = [line for line in row_score_files[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(score["name"] in EXACT_MATCH_AGGREGATE_SCORE_NAMES for score in aggregate_payload["scores"])
    assert len(row_lines) == 2


def test_offline_evaluate_job_persists_queryable_eval_result(
    evaluator_client: NemoClient,
    completed_offline_job: EvaluatorJobResource,
) -> None:
    result = _evaluator(evaluator_client).eval_results.retrieve(completed_offline_job.name)
    by_job = _evaluator(evaluator_client).eval_results.list(job_id=completed_offline_job.name)

    assert result.job_id == completed_offline_job.name
    assert result.metric_types == ["exact-match"]
    assert result.dataset_ref is None
    assert any(item.job_id == completed_offline_job.name for item in by_job.data)


def test_fileset_fragment_and_glob_datasets(evaluator_client: NemoClient) -> None:
    """Durable plugin jobs can read selected and globbed files from Files."""
    fileset_name = short_unique_name("eval-data")
    workspace = str(evaluator_client.workspace)
    submitted_jobs: list[tuple[str, list[float], EvaluatorJobResource]] = []
    files = FilesClient.from_client(evaluator_client)
    files.create_fileset(body=CreateFilesetRequest(name=fileset_name), workspace=workspace)
    try:
        files.upload_file(
            content=json.dumps(
                [
                    {"expected": "alpha", "output": "alpha"},
                    {"expected": "beta", "output": "wrong"},
                ]
            ).encode(),
            path="part-a.json",
            name=fileset_name,
            workspace=workspace,
        )
        files.upload_file(
            content=json.dumps([{"expected": "gamma", "output": "gamma"}]).encode(),
            path="part-b.json",
            name=fileset_name,
            workspace=workspace,
        )
        files.upload_file(
            content=json.dumps([{"expected": "ignored", "output": "ignored"}]).encode(),
            path="other.json",
            name=fileset_name,
            workspace=workspace,
        )

        cases = {
            "specific file": (f"{workspace}/{fileset_name}#part-a.json", [1.0, 0.0]),
            "glob": (f"{workspace}/{fileset_name}#part-*.json", [1.0, 0.0, 1.0]),
        }
        for label, (reference, expected_scores) in cases.items():
            job = _evaluator(evaluator_client).submit(
                metric=_exact_match_metric(),
                dataset=FilesetRef(root=reference),
                config=RunConfig(parallelism=1),
            )
            submitted_jobs.append((label, expected_scores, job))

        with ThreadPoolExecutor(max_workers=len(submitted_jobs)) as executor:
            wait_futures = [executor.submit(_wait_for_evaluator_job, job) for _, _, job in submitted_jobs]
            for (label, expected_scores, job), wait_future in zip(submitted_jobs, wait_futures, strict=True):
                wait_future.result()
                assert _row_score_values(job.get_result()) == expected_scores, label
    finally:
        for _, _, job in submitted_jobs:
            _cleanup_evaluator_job(evaluator_client, job.name)
        with suppress(Exception):
            files.delete_fileset(name=fileset_name, workspace=workspace)


def test_run_config_limits_samples(evaluator_client: NemoClient) -> None:
    rows = [{"expected": str(index), "output": str(index)} for index in range(8)]
    job = _evaluator(evaluator_client).submit(
        metric=_exact_match_metric(),
        dataset=rows,
        config=RunConfig(limit_samples=3, parallelism=2),
    )
    try:
        _wait_for_evaluator_job(job)
        result = job.get_result()

        assert len(result.row_scores) == 3
        assert _aggregate_score(result).count == 3
    finally:
        _cleanup_evaluator_job(evaluator_client, job.name)


def test_multi_metric_durable_evaluation(evaluator_client: NemoClient) -> None:
    """The unified plugin job supports benchmark-style metric sets."""
    metrics = [
        _exact_match_metric(),
        StringCheckMetric(
            operation="contains",
            left_template="{{item.output}}",
            right_template="{{item.required_phrase}}",
        ),
    ]
    packager = InlineMetricBundlePackager()
    spec = EvaluateInputSpec(
        metrics=[
            MetricInline.model_validate(bundle_metric(metric, packager).model_dump(mode="json")) for metric in metrics
        ],
        dataset=[
            {"expected": "Paris", "output": "Paris, France", "required_phrase": "Paris"},
            {"expected": "Jupiter", "output": "Saturn", "required_phrase": "Jupiter"},
        ],
        params=RunConfig(parallelism=2),
    )
    job = _submit_input_spec(evaluator_client, spec)
    try:
        _wait_for_evaluator_job(job)
        result = job.get_result()

        exact_match = _aggregate_score(result)
        assert exact_match.count == 2
        assert exact_match.mean == 0.0
        assert _metric_output_values(result, "exact-match") == [0.0, 0.0]
        assert _metric_output_values(result, "string-check") == [1.0, 0.0]
        string_check = next(
            score for score in result.aggregate_scores.scores if score.name == "string-check.string-check"
        )
        assert string_check.count == 2
        assert string_check.mean == 0.5
        assert len(result.row_scores) == 2
    finally:
        _cleanup_evaluator_job(evaluator_client, job.name)


def test_stored_metric_ref_executes_in_durable_job(evaluator_client: NemoClient) -> None:
    metric_name = short_unique_name("stored-exact")
    job: EvaluatorJobResource | None = None
    try:
        _evaluator(evaluator_client).metrics.create(metric_name, metric=_exact_match_metric())
        spec = EvaluateInputSpec(
            metrics=[MetricRef(root=metric_name)],
            dataset=_offline_rows(),
            params=RunConfig(parallelism=1),
        )
        job = _submit_input_spec(evaluator_client, spec)
        _wait_for_evaluator_job(job)

        assert _row_score_values(job.get_result()) == [1.0, 0.0]
    finally:
        if job is not None:
            _cleanup_evaluator_job(evaluator_client, job.name)
        with suppress(Exception):
            _evaluator(evaluator_client).metrics.delete(metric_name)


def test_tool_calling_metric_preserves_structured_references(evaluator_client: NemoClient) -> None:
    rows = [
        {
            "expected_tool_calls": [{"function": {"name": "weather", "arguments": {"city": "Paris"}}}],
            "response": {
                "choices": [
                    {"message": {"tool_calls": [{"function": {"name": "weather", "arguments": '{"city": "Paris"}'}}]}}
                ]
            },
        },
        {
            "expected_tool_calls": [{"function": {"name": "search", "arguments": {"query": "python"}}}],
            "response": {
                "choices": [
                    {"message": {"tool_calls": [{"function": {"name": "search", "arguments": '{"query": "rust"}'}}]}}
                ]
            },
        },
    ]
    job = _evaluator(evaluator_client).submit(
        metric=ToolCallingMetric(reference="{{item.expected_tool_calls}}"),
        dataset=rows,
        config=RunConfig(parallelism=2),
    )
    try:
        _wait_for_evaluator_job(job)
        result = job.get_result()

        assert _metric_output_values(result, "function_name_accuracy") == [1.0, 1.0]
        assert _metric_output_values(result, "function_name_and_args_accuracy") == [1.0, 0.0]
    finally:
        _cleanup_evaluator_job(evaluator_client, job.name)


@pytest.mark.parametrize("use_model_ref", [False, True], ids=["inline-model", "model-ref"])
def test_online_evaluate_job_uses_mock_provider(
    client: NemoClient,
    evaluator_client: NemoClient,
    evaluator_workspace: str,
    use_model_ref: bool,
) -> None:
    model_name = short_unique_name("eval-model")
    _create_ready_mock_model(
        client,
        workspace=evaluator_workspace,
        name=model_name,
        mock_response_body=_chat_completion("Paris"),
    )
    # A ref is resolved by the server to the in-cluster gateway, as Studio submits it; an inline model
    # names that route directly.
    target = (
        ModelRef(root=f"{evaluator_workspace}/{model_name}")
        if use_model_ref
        else Model(
            url=_internal_model_route(evaluator_workspace, model_name),
            name=model_name,
            format=ModelFormat.OPEN_AI,
        )
    )

    job = _evaluator(evaluator_client).submit(
        metric=_exact_match_metric(candidate=None),
        dataset=[{"question": "What is the capital of France?", "expected": "Paris"}],
        config=RunConfigOnlineModel(
            parallelism=1,
            request_timeout=60,
            max_retries=0,
            inference=InferenceParams(max_tokens=8),
        ),
        target=target,
        prompt_template={"messages": [{"role": "user", "content": "{{item.question}}"}]},
    )
    try:
        _wait_for_evaluator_job(job)

        result = job.get_result()

        aggregate = _aggregate_score(result)
        assert aggregate.count == 1
        assert aggregate.mean == 1.0
        assert result.row_scores[0].sample["output_text"] == "Paris"
    finally:
        _cleanup_evaluator_job(evaluator_client, job.name)


def test_llm_judge_metric_resolves_model_ref(
    client: NemoClient,
    evaluator_client: NemoClient,
    evaluator_workspace: str,
) -> None:
    model_name = short_unique_name("eval-judge")
    _create_ready_mock_model(
        client,
        workspace=evaluator_workspace,
        name=model_name,
        mock_response_body=_chat_completion('{"quality": 4}'),
    )
    metric = LLMJudgeMetric(
        model=ModelRef(root=f"{evaluator_workspace}/{model_name}"),
        scores=[
            RangeScore(
                name="quality",
                minimum=1,
                maximum=5,
                parser=JSONScoreParser(json_path="quality"),
            )
        ],
        prompt_template={
            "messages": [
                {"role": "system", "content": "Return a JSON quality score from 1 to 5."},
                {"role": "user", "content": "Evaluate this answer: {{item.answer}}"},
            ]
        },
    )
    job = _evaluator(evaluator_client).submit(
        metric=metric,
        dataset=[{"answer": "Paris"}],
        config=RunConfig(parallelism=1),
    )
    try:
        _wait_for_evaluator_job(job)

        assert _metric_output_values(job.get_result(), "quality") == [4.0]
    finally:
        _cleanup_evaluator_job(evaluator_client, job.name)


def _assert_runtime_input_failure(
    evaluator_client: NemoClient,
    metric: StringCheckMetric | ExactMatchMetric,
    dataset: list[dict[str, object]] | FilesetRef,
) -> None:
    job = _evaluator(evaluator_client).submit(
        metric=metric,
        dataset=dataset,
        config=RunConfig(parallelism=1),
    )
    try:
        with pytest.raises(RuntimeError):
            _wait_for_evaluator_job(job)
        assert job.get_job_status().status.value == "error"
    finally:
        _cleanup_evaluator_job(evaluator_client, job.name)


def test_invalid_template_reaches_terminal_error(evaluator_client: NemoClient) -> None:
    metric = StringCheckMetric(
        operation="equals",
        left_template="{{item.missing}}",
        right_template="{{item.expected}}",
    )
    _assert_runtime_input_failure(
        evaluator_client,
        metric,
        [{"expected": "value", "output": "value"}],
    )


def test_missing_fileset_reaches_terminal_error(evaluator_client: NemoClient) -> None:
    dataset = FilesetRef(root=f"{evaluator_client.workspace}/missing-fileset#dataset.json")
    _assert_runtime_input_failure(evaluator_client, _exact_match_metric(), dataset)


# --- Gym agent-evaluate job ------------------------------------------------------------------
#
# GymAgentTaskRunner runs inside the Gym task job, not this test process, so this is the only place
# exercising the real `gym env start`/`gym eval run` subprocess path (the SDK's unit tests use
# synthetic data). A dedicated Kind CI job runs the platform with the
# nhx-gym-tasks image. See docker/Dockerfile.nhx-gym-tasks.
#
# Checked-in copy of mcqa's example.jsonl: discover_gym_tasks runs client-side here, where
# nemo-gym isn't installed (the dedicated task image has it).
GYM_MCQA_FIXTURE = (
    Path(__file__).resolve().parents[1] / "packages/nhx_evals_sdk/tests/agent_eval/fixtures/gym_mcqa_example.jsonl"
)


def _gym_task_payloads(limit: int) -> list[dict[str, object]]:
    tasks = discover_gym_tasks(GYM_MCQA_FIXTURE)[:limit]
    bundled_reward = bundle_metric(GymRewardMetric(), InlineMetricBundlePackager()).model_dump(mode="json")
    return [
        {
            "id": task.id,
            "intent": task.intent,
            "inputs": task.inputs or {},
            "reference": task.reference or {},
            "metrics": [bundled_reward],
            "metadata": [{"key": key, "value": value} for key, value in (task.metadata or {}).items()],
        }
        for task in tasks
    ]


@pytest.mark.gym_e2e
@pytest.mark.timeout(1800)
def test_gym_agent_evaluate_job_completes(
    client: NemoClient,
    evaluator_client: NemoClient,
    evaluator_workspace: str,
    tmp_path: Path,
) -> None:
    """Submits mcqa as a real agent-evaluate job and verifies it completes with real rewards."""
    model_name = short_unique_name("gym-mcqa")
    _create_ready_mock_model(
        client,
        workspace=evaluator_workspace,
        name=model_name,
        mock_response_body=_chat_completion(r"\boxed{B}"),
    )

    task_dicts = _gym_task_payloads(2)  # keep the run cheap; this is a wiring test

    # A fresh task container installs each Gym server's dependencies at startup.
    startup_timeout_s = 600
    collection_timeout_s = 120
    target = GymRunnerTarget(
        source=GymAgentSource(
            component="simple_agent", config="responses_api_agents/simple_agent/configs/simple_agent.yaml"
        ),
        resources_server="mcqa",
        num_repeats=1,
        concurrency=2,
        startup_timeout_s=startup_timeout_s,
        collection_timeout_s=collection_timeout_s,
        hydra_params={
            "policy_base_url": _internal_model_route(evaluator_workspace, model_name),
            "policy_api_key": "not-used-mock-provider",
            "policy_model_name": model_name,
        },
    )

    job_name = _submit_agent_eval_job(
        evaluator_client, evaluator_workspace, {"tasks": task_dicts, "target": target.model_dump(mode="json")}
    )
    try:
        job = wait_for_platform_job(
            evaluator_client,
            job_name,
            evaluator_workspace,
            timeout=startup_timeout_s + collection_timeout_s + 120,
        )
        assert job.status.lower() == "completed", f"job {job_name!r} ended {job.status!r}"

        # Download the results archive to a tmp path.
        archive_path = tmp_path / "agent-eval-results.tar.gz"
        archive = JobsClient.from_client(evaluator_client).download_job_result(
            name="agent-eval-results", job=job_name, workspace=evaluator_workspace
        )
        archive_path.write_bytes(archive.read())

        # Extract the trials.jsonl file from the archive.
        extract_dir = tmp_path / "agent-eval-results"
        with tarfile.open(archive_path) as archive:
            archive.extractall(extract_dir, filter="data")
        trials_files = list(extract_dir.rglob("trials.jsonl"))
        assert trials_files, f"trials.jsonl missing under {extract_dir}"

        trial_lines = [line for line in trials_files[0].read_text(encoding="utf-8").splitlines() if line.strip()]
        assert trial_lines, "Gym job produced no trials"

        trials = [json.loads(line) for line in trial_lines]
        assert len(trials) == 2
        assert [trial.get("status") for trial in trials] == ["completed", "completed"]
        rewards = [float(trial["metadata"]["reward"]) for trial in trials]
        assert sorted(rewards) == [0.0, 1.0]
    finally:
        _cleanup_evaluator_job(evaluator_client, job_name)


# --- Sandboxed Gym: a FileSet environment in an OpenSandbox Gym host -----------------------------
#
# The path docs/evals/agent-eval/gym-sandbox-configuration.mdx describes: stage-environment copies
# the FileSet onto the Jobs PVC, then agent-evaluate asks OpenSandbox for a Gym host, which mounts
# that storage and runs the environment's own code with egress denied by default. These tests need a
# deployment configured for it, and kubectl access to its cluster: whether a host was removed is
# visible only there. The kind-gym-sandbox-e2e CI job provides both, and e2e/README.md has the
# settings and the manual invocation.

requires_sandboxed_gym = pytest.mark.skipif(
    os.environ.get("NHX_E2E_SANDBOXED_GYM") != "1",
    reason="needs a sandbox-capable deployment (sandboxed_gym_default + OpenSandbox); set NHX_E2E_SANDBOXED_GYM=1",
)

#: The label the OpenSandbox server puts on every sandbox pod it creates.
SANDBOX_POD_LABEL = "opensandbox.io/id"
#: Where a Gym host mounts the staged environment (read-only) and its workspace (read-write).
GYM_HOST_ENVIRONMENT_PATH = "/job/environment"
GYM_HOST_WORK_PATH = "/job/work"
GYM_HOST_POLL_INTERVAL_SECONDS = 2.0
#: A run destroys its host as it ends, and the host's pod then has a 30s termination grace period.
GYM_HOST_GONE_TIMEOUT_SECONDS = 180.0
#: A public address that no egress allowlist names and no allowed name resolves to.
UNLISTED_PUBLIC_ADDRESS = "1.1.1.1:443"
#: Long enough for a host to start on a small CI node, where the default 240s is tight.
GYM_HOST_STARTUP_TIMEOUT_SECONDS = 600
SANDBOXED_GYM_JOB_TIMEOUT_SECONDS = 1200.0

# The environment's resources server, which runs inside the Gym host. Besides scoring the greeting,
# it checks one fact about the sandbox it runs in when a row's ``probe`` names one. The reward is
# 1.0 only when the greeting matches and that fact holds, so each probe is a row and a task of its
# own, and a failing one is named by its task. ``hang`` never answers, for a run that must fail.
_GREETING_VERIFIER_SOURCE = """\
import asyncio
import os
import socket
from pathlib import Path

from greeting_extra import normalize
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)

ENVIRONMENT = Path(os.environ.get("NHX_ENVIRONMENT_PATH") or "/job/environment")
WORK = Path(os.environ.get("NHX_WORK_PATH") or "/job/work")


class GreetingConfig(BaseResourcesServerConfig):
    pass


class GreetingVerifyRequest(BaseVerifyRequest):
    expected_greeting: str
    probe: str = ""
    probe_target: str = ""


class GreetingVerifyResponse(BaseVerifyResponse):
    expected_greeting: str
    probe: str = ""
    probe_target: str = ""
    observed_text: str


def _assistant_text(response):
    parts = []
    for item in response.output:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []):
            if getattr(content, "type", None) == "output_text":
                parts.append(content.text)
    return "".join(parts)


def _environment_staged(_target):
    # The manifest this package was uploaded with, read from where the host mounted it.
    return "name: e2e-greeting" in (ENVIRONMENT / "nemo-environment.yaml").read_text()


def _environment_read_only(_target):
    try:
        (ENVIRONMENT / "written-by-the-environment").write_text("refused")
    except OSError:
        return True
    return False


def _workspace_writable(_target):
    marker = WORK / "written-by-the-environment"
    marker.write_text("kept")
    return marker.read_text() == "kept"


def _connects(target):
    host, _, port = target.rpartition(":")
    try:
        socket.create_connection((host, int(port)), timeout=5).close()
    except OSError:
        return False
    return True


def _does_not_connect(target):
    return not _connects(target)


PROBES = {
    "": lambda _target: True,
    "environment_staged": _environment_staged,
    "environment_read_only": _environment_read_only,
    "workspace_writable": _workspace_writable,
    "egress_allowed": _connects,
    "egress_denied": _does_not_connect,
}


class GreetingVerifier(SimpleResourcesServer):
    config: GreetingConfig

    async def verify(self, body: GreetingVerifyRequest) -> GreetingVerifyResponse:
        if body.probe == "hang":
            await asyncio.sleep(3600)
        observed = _assistant_text(body.response)
        greeted = normalize(body.expected_greeting) in normalize(observed)
        try:
            # Off the event loop: a denied connection takes its whole timeout to fail.
            held = await asyncio.to_thread(PROBES[body.probe], body.probe_target)
        except (KeyError, OSError, ValueError):
            held = False
        return GreetingVerifyResponse(**body.model_dump(), reward=float(greeted and held), observed_text=observed)


if __name__ == "__main__":
    GreetingVerifier.run_webserver()
"""


def _greeting_wheels_v1_package(root: Path) -> dict[str, bytes]:
    """A wheels-v1 environment: one custom resources server whose only extra dependency is a vendored wheel.

    The wheel is generated here (a pure-Python module), so the test needs no package index at all.
    """
    import io
    import zipfile

    package_name = "greeting_extra"
    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"{package_name}/__init__.py", "def normalize(text: str) -> str:\n    return text.casefold().strip()\n"
        )
        archive.writestr(
            f"{package_name}-1.0.dist-info/METADATA", f"Metadata-Version: 2.1\nName: {package_name}\nVersion: 1.0\n"
        )
        archive.writestr(
            f"{package_name}-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: e2e\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{package_name}-1.0.dist-info/RECORD", "")
    config = (
        "greeting_verifier:\n  resources_servers:\n    greeting_verifier:\n      entrypoint: app.py\n"
        "      domain: instruction_following\n      verified: false\n"
        "      description: e2e greeting verifier\n      value: Verify the expected greeting\n"
    )
    # The verifier's environment_staged probe looks for this name.
    manifest = (
        "format: wheels-v1\nconfig_paths:\n  - resources_servers/greeting_verifier/configs/greeting_verifier.yaml\n"
        "metadata:\n  name: e2e-greeting\n"
    )
    files = {
        "nemo-environment.yaml": manifest.encode(),
        "resources_servers/greeting_verifier/app.py": _GREETING_VERIFIER_SOURCE.encode(),
        "resources_servers/greeting_verifier/requirements.txt": b"greeting-extra==1.0\n",
        "resources_servers/greeting_verifier/configs/greeting_verifier.yaml": config.encode(),
        f"wheels/{package_name}-1.0-py3-none-any.whl": wheel.getvalue(),
    }
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return files


def _greeting_row(greeting: str, probe: str = "", probe_target: str = "") -> dict[str, object]:
    """One dataset row: the prompt, the greeting to expect, and optionally a fact about the sandbox to check."""
    row: dict[str, object] = {
        "responses_create_params": {
            "input": [{"role": "user", "content": f"Reply with exactly this greeting and nothing else: {greeting}"}]
        },
        "expected_greeting": greeting,
    }
    if probe:
        row.update(probe=probe, probe_target=probe_target)
    return row


def _greeting_task_payloads(
    rows: Sequence[Mapping[str, object]], rows_path: Path
) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Task payloads for ``rows``, and each task id's label: its probe and target, or ``greeting``."""
    bundled_reward = bundle_metric(GymRewardMetric(), InlineMetricBundlePackager()).model_dump(mode="json")
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    tasks = discover_gym_tasks(rows_path)
    # One distinct row is one task, so repeating a row would quietly run fewer trials than asked for.
    assert len(tasks) == len(rows), f"{len(rows)} rows made {len(tasks)} tasks; every row must differ"
    labels: dict[str, str] = {}
    for task in tasks:
        extras = (task.metadata or {}).get("gym_row_extras") or {}
        labels[task.id] = " ".join(filter(None, (extras.get("probe") or "greeting", extras.get("probe_target"))))
    payloads: list[dict[str, object]] = [
        {
            "id": task.id,
            "intent": task.intent,
            "inputs": task.inputs or {},
            "reference": task.reference or {},
            "metrics": [bundled_reward],
            "metadata": [{"key": key, "value": value} for key, value in (task.metadata or {}).items()],
        }
        for task in tasks
    ]
    return payloads, labels


@dataclass(frozen=True)
class _SandboxCluster:
    """The cluster the platform runs on, read with kubectl on the named context only."""

    context: str
    namespace: str

    def _get(self, *resource: str, namespace: str | None = None) -> dict[str, Any]:
        cmd = ["kubectl", "--context", self.context, "--namespace", namespace or self.namespace, "get", *resource]
        result = subprocess.run(
            [*cmd, "-o", "json"], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(f"`{' '.join(cmd)}` failed: {result.stderr.strip()}")
        return json.loads(result.stdout)

    def api_server_address(self) -> str:
        """The Kubernetes API's in-cluster ``address:port``, which any pod reaches unless something stops it."""
        spec = self._get("service", "kubernetes", namespace="default")["spec"]
        return f"{spec['clusterIP']}:{spec['ports'][0]['port']}"

    def gym_hosts(self, markers: frozenset[str]) -> dict[str, dict[str, Any]]:
        """A job's Gym host: the BatchSandboxes that name the job and their pods, by ``kind/name``.

        A sandbox carries its job's id in its metadata and its bootstrap environment, which is also how
        k8s/helm/examples/opensandbox/verify/lib.sh finds one. A pod is the host's when it has the
        label OpenSandbox gives the pods it creates and either belongs to one of those BatchSandboxes
        or names the job itself. The label matters: the job's own step pods name the job too.
        """

        def names_job(item: Mapping[str, Any]) -> bool:
            manifest = json.dumps(item)
            return any(marker in manifest for marker in markers)

        found: dict[str, dict[str, Any]] = {}
        sandboxes: set[str] = set()
        for sandbox in self._get("batchsandboxes").get("items") or []:
            if names_job(sandbox):
                sandboxes.add(sandbox["metadata"]["name"])
                found[f"batchsandboxes/{sandbox['metadata']['name']}"] = sandbox
        for pod in self._get("pods").get("items") or []:
            metadata = pod.get("metadata") or {}
            sandbox_id = (metadata.get("labels") or {}).get(SANDBOX_POD_LABEL)
            if sandbox_id is None:
                continue
            owners = {owner.get("name") for owner in metadata.get("ownerReferences") or []}
            if sandbox_id in sandboxes or owners & sandboxes or names_job(pod):
                found[f"pods/{metadata.get('name')}"] = pod
        return found


class _GymHostWatch:
    """Records a job's Gym host while the job runs, because the host is meant to be gone when it ends."""

    def __init__(self, cluster: _SandboxCluster, markers: frozenset[str]) -> None:
        self.markers = markers
        #: Every host object seen, and the sandbox pods seen running, each as last seen.
        self.seen: dict[str, dict[str, Any]] = {}
        self.running_pods: dict[str, dict[str, Any]] = {}
        self.failed_polls: list[str] = []
        self._cluster = cluster
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, name="gym-host-watch", daemon=True)

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                hosts = self._cluster.gym_hosts(self.markers)
            except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
                # One failed read must not end the watch: the next may still see the host.
                self.failed_polls.append(str(error))
            else:
                self.seen.update(hosts)
                self.running_pods.update(
                    {
                        name: host
                        for name, host in hosts.items()
                        if name.startswith("pods/") and (host.get("status") or {}).get("phase") == "Running"
                    }
                )
            self._stop.wait(GYM_HOST_POLL_INTERVAL_SECONDS)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=120)


@pytest.fixture(scope="module")
def sandbox_cluster() -> _SandboxCluster:
    """kubectl access to the cluster under test, on a context named for the purpose.

    Required, not skipped: without it these tests could not tell a removed Gym host from a leaked one.
    """
    context = os.environ.get("NHX_E2E_KUBE_CONTEXT")
    if not context:
        pytest.fail(
            "NHX_E2E_KUBE_CONTEXT is required by the sandboxed Gym tests: name the kubectl context of the "
            "cluster the platform runs on (they never use the current context); see e2e/README.md"
        )
    return _SandboxCluster(context=context, namespace=os.environ.get("NHX_E2E_KUBE_NAMESPACE") or "nemo-helix")


def _host_port(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.hostname}:{parsed.port or (443 if parsed.scheme == 'https' else 80)}"


def _upload_greeting_environment(client: NemoClient, workspace: str, root: Path) -> str:
    """Upload the greeting package as an environment FileSet and return the FileSet's name."""
    fileset_name = short_unique_name("gym-greeting-env")
    files = FilesClient.from_client(client)
    files.create_fileset(
        body=CreateFilesetRequest(name=fileset_name, purpose=FilesetPurpose.ENVIRONMENT), workspace=workspace
    )
    for relative, content in _greeting_wheels_v1_package(root).items():
        files.upload_file(content=content, path=relative, name=fileset_name, workspace=workspace)
    return fileset_name


def _greeting_policy(client: NemoClient, workspace: str, greeting: str) -> tuple[str, str]:
    """The model route and name the agent calls: a mock that replies with ``greeting``, unless one is configured."""
    policy_route = os.environ.get("NHX_E2E_GYM_POLICY_ROUTE")
    policy_model = os.environ.get("NHX_E2E_GYM_POLICY_MODEL")
    if policy_route and policy_model:
        return policy_route, policy_model
    policy_model = short_unique_name("gym-greeting")
    _create_ready_mock_model(
        client, workspace=workspace, name=policy_model, mock_response_body=_chat_completion(greeting)
    )
    return _internal_model_route(workspace, policy_model), policy_model


def _greeting_target(
    workspace: str, fileset_name: str, policy: tuple[str, str], *, collection_timeout_s: float | None = None
) -> GymRunnerTarget:
    policy_route, policy_model = policy
    return GymRunnerTarget(
        source=GymAgentSource(
            component="simple_agent",
            config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
        ),
        resources_server="greeting_verifier",
        environment=FilesetRef(root=f"{workspace}/{fileset_name}"),
        num_repeats=1,
        concurrency=2,
        startup_timeout_s=GYM_HOST_STARTUP_TIMEOUT_SECONDS,
        collection_timeout_s=collection_timeout_s,
        hydra_params={
            "policy_base_url": policy_route,
            "policy_api_key": "not-used",
            "policy_model_name": policy_model,
        },
    )


def _run_sandboxed_gym_job(
    client: NemoClient, workspace: str, job_name: str, cluster: _SandboxCluster
) -> tuple[Any, _GymHostWatch]:
    """Wait for the job to end, watching the cluster for its Gym host meanwhile."""
    submitted = JobsClient.from_client(client).get_job(name=job_name, workspace=workspace).data()
    markers = frozenset(marker for marker in (submitted.id, submitted.name) if marker)
    with _GymHostWatch(cluster, markers) as watch:
        job = wait_for_platform_job(client, job_name, workspace, timeout=SANDBOXED_GYM_JOB_TIMEOUT_SECONDS)
    return job, watch


def _job_steps(client: NemoClient, workspace: str, job_name: str) -> dict[str, str]:
    """Each step's status, by step name."""
    status = JobsClient.from_client(client).get_job_status(workspace=workspace, name=job_name).data()
    return {step.name: step.status.value for step in status.steps}


def _describe_job(client: NemoClient, workspace: str, job_name: str) -> str:
    """The job's and its steps' statuses and error details, for a failed assertion's message."""
    status = JobsClient.from_client(client).get_job_status(workspace=workspace, name=job_name).data()
    return json.dumps(
        {
            "status": status.status.value,
            "error_details": status.error_details,
            "steps": [
                {"name": step.name, "status": step.status.value, "error_details": step.error_details}
                for step in status.steps
            ],
        },
        default=str,
    )


def _assert_gym_host_removed(cluster: _SandboxCluster, watch: _GymHostWatch, job_name: str) -> None:
    """Fail unless the job's Gym host ran and is now gone.

    Call it while the job still exists, so the removal is the run's own doing and not a side effect
    of this test deleting the job.
    """
    assert watch.running_pods, (
        f"no Gym host pod was seen running for job {job_name!r}, so its removal cannot be checked; "
        f"host objects seen: {sorted(watch.seen)}; failed polls: {watch.failed_polls[-3:]}"
    )
    deadline = time.monotonic() + GYM_HOST_GONE_TIMEOUT_SECONDS
    while left := cluster.gym_hosts(watch.markers):
        if time.monotonic() >= deadline:
            pytest.fail(
                f"job {job_name!r} ended, but its Gym host was still on the cluster "
                f"{GYM_HOST_GONE_TIMEOUT_SECONDS:g}s later: {sorted(left)}"
            )
        time.sleep(GYM_HOST_POLL_INTERVAL_SECONDS)


def _assert_host_mounts_job_storage(pod: Mapping[str, Any]) -> None:
    """The host's environment and workspace are two directories of one PVC, the environment read-only."""
    spec = pod["spec"]
    volumes = {volume["name"]: volume for volume in spec.get("volumes") or []}
    mounts = {
        mount["mountPath"]: mount
        for container in spec.get("containers") or []
        for mount in container.get("volumeMounts") or []
    }
    environment, work = mounts.get(GYM_HOST_ENVIRONMENT_PATH), mounts.get(GYM_HOST_WORK_PATH)
    assert environment and work, (
        f"the Gym host mounts {sorted(mounts)}, not both {GYM_HOST_ENVIRONMENT_PATH} and {GYM_HOST_WORK_PATH}"
    )
    assert environment.get("readOnly") is True, f"the environment mount is not read-only: {environment}"
    assert not work.get("readOnly"), f"the workspace mount is read-only: {work}"
    claims = [
        ((volumes.get(mount["name"]) or {}).get("persistentVolumeClaim") or {}).get("claimName")
        for mount in (environment, work)
    ]
    assert claims[0] and claims[0] == claims[1], f"environment and workspace are not one PVC: claims {claims}"
    assert environment.get("subPath") != work.get("subPath"), (
        f"environment and workspace mount the same directory of the PVC: {environment.get('subPath')!r}"
    )


def _download_agent_eval_bundle(client: NemoClient, workspace: str, job_name: str, tmp_path: Path) -> Path:
    """Download and unpack the job's ``agent-eval-results`` archive, returning the directory its files are in."""
    archive_path = tmp_path / "agent-eval-results.tar.gz"
    archive = JobsClient.from_client(client).download_job_result(
        name="agent-eval-results", job=job_name, workspace=workspace
    )
    archive_path.write_bytes(archive.read())
    extract_dir = tmp_path / "agent-eval-results"
    with tarfile.open(archive_path) as bundle:
        bundle.extractall(extract_dir, filter="data")
    trials_files = list(extract_dir.rglob("trials.jsonl"))
    assert trials_files, f"trials.jsonl missing under {extract_dir}"
    return trials_files[0].parent


@pytest.mark.gym_e2e
@pytest.mark.gym_sandbox_e2e
@requires_sandboxed_gym
def test_gym_wheels_v1_environment_runs_with_only_its_extras_vendored(
    client: NemoClient,
    evaluator_client: NemoClient,
    evaluator_workspace: str,
    sandbox_cluster: _SandboxCluster,
    tmp_path: Path,
) -> None:
    """A wheels-v1 FileSet vendors only what its server adds; Gym and Gym's dependencies come from the image.

    The job must complete and score with the package's one vendored wheel and no package index, on a
    deployment whose sandbox egress allows only the policy endpoint. Its verifier reports from inside
    the Gym host: the staged package is mounted read-only from the Jobs PVC, the workspace is
    writable, the policy endpoint is reachable, and the Kubernetes API and a public address are not.
    The host must be gone once the job has ended. Set NHX_E2E_GYM_POLICY_ROUTE and
    NHX_E2E_GYM_POLICY_MODEL to use a real model instead of the mock.
    """
    greeting = "hello there"
    fileset_name = _upload_greeting_environment(evaluator_client, evaluator_workspace, tmp_path / "package")
    policy = _greeting_policy(client, evaluator_workspace, greeting)
    rows = [
        _greeting_row(greeting),
        _greeting_row(greeting, "environment_staged"),
        _greeting_row(greeting, "environment_read_only"),
        _greeting_row(greeting, "workspace_writable"),
        _greeting_row(greeting, "egress_allowed", _host_port(policy[0])),
        _greeting_row(greeting, "egress_denied", sandbox_cluster.api_server_address()),
        _greeting_row(greeting, "egress_denied", UNLISTED_PUBLIC_ADDRESS),
    ]
    payloads, labels = _greeting_task_payloads(rows, tmp_path / "greeting-rows.jsonl")
    target = _greeting_target(evaluator_workspace, fileset_name, policy)
    job_name = _submit_agent_eval_job(
        evaluator_client, evaluator_workspace, {"tasks": payloads, "target": target.model_dump(mode="json")}
    )
    try:
        job, watch = _run_sandboxed_gym_job(evaluator_client, evaluator_workspace, job_name, sandbox_cluster)
        report = _describe_job(evaluator_client, evaluator_workspace, job_name)
        assert job.status.lower() == "completed", f"job {job_name!r} ended {job.status!r}: {report}"
        steps = _job_steps(evaluator_client, evaluator_workspace, job_name)
        assert (steps.get("stage-environment"), steps.get("agent-evaluate")) == ("completed", "completed"), report

        _assert_gym_host_removed(sandbox_cluster, watch, job_name)
        _assert_host_mounts_job_storage(next(iter(watch.running_pods.values())))

        bundle = _download_agent_eval_bundle(evaluator_client, evaluator_workspace, job_name, tmp_path)
        trials = [
            json.loads(line)
            for line in (bundle / "trials.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert sorted(trial["task_id"] for trial in trials) == sorted(labels), trials
        assert {trial.get("status") for trial in trials} == {"completed"}, trials
        # A reward below 1.0 names the fact that did not hold; `greeting` alone means the agent's reply was wrong.
        rewards = {labels[trial["task_id"]]: trial["metadata"]["reward"] for trial in trials}
        assert rewards == dict.fromkeys(labels.values(), 1.0)

        summary_path = bundle / "summary.json"
        assert summary_path.is_file(), f"summary.json missing from {bundle}"
        summary = AgentEvalSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
        assert (summary.task_count, summary.trial_count, summary.error_count) == (len(rows), len(rows), 0), summary
        reward_scores = [score for score in summary.scores.scores if score.name.endswith(".reward")]
        assert reward_scores, f"summary.json aggregates no reward: {summary.scores}"
        assert all(score.mean == 1.0 and score.nan_count == 0 for score in reward_scores), reward_scores
    finally:
        _cleanup_evaluator_job(evaluator_client, job_name)


@pytest.mark.gym_e2e
@pytest.mark.gym_sandbox_e2e
@requires_sandboxed_gym
def test_gym_host_is_removed_when_a_sandboxed_evaluation_fails_after_provisioning(
    client: NemoClient,
    evaluator_client: NemoClient,
    evaluator_workspace: str,
    sandbox_cluster: _SandboxCluster,
    tmp_path: Path,
) -> None:
    """A run that fails with its Gym host already up still removes the host.

    The environment stages and the host starts as they do for the passing run, then its verifier
    never answers, so collection overruns ``collection_timeout_s`` and agent-evaluate fails.
    """
    greeting = "hello there"
    fileset_name = _upload_greeting_environment(evaluator_client, evaluator_workspace, tmp_path / "package")
    policy = _greeting_policy(client, evaluator_workspace, greeting)
    payloads, _ = _greeting_task_payloads([_greeting_row(greeting, "hang")], tmp_path / "greeting-rows.jsonl")
    target = _greeting_target(evaluator_workspace, fileset_name, policy, collection_timeout_s=30)
    job_name = _submit_agent_eval_job(
        evaluator_client, evaluator_workspace, {"tasks": payloads, "target": target.model_dump(mode="json")}
    )
    try:
        job, watch = _run_sandboxed_gym_job(evaluator_client, evaluator_workspace, job_name, sandbox_cluster)
        report = _describe_job(evaluator_client, evaluator_workspace, job_name)
        assert job.status.lower() == "error", f"job {job_name!r} ended {job.status!r}: {report}"
        steps = _job_steps(evaluator_client, evaluator_workspace, job_name)
        assert (steps.get("stage-environment"), steps.get("agent-evaluate")) == ("completed", "error"), report

        # Also what shows the failure came after provisioning: it needs the host's pod seen running.
        _assert_gym_host_removed(sandbox_cluster, watch, job_name)
    finally:
        _cleanup_evaluator_job(evaluator_client, job_name)


@pytest.mark.gym_e2e
def test_gym_agent_evaluate_job_invalid_config_fails(
    evaluator_client: NemoClient,
    evaluator_workspace: str,
) -> None:
    """Rejects an invalid Gym selection before starting its environment servers."""
    target = GymRunnerTarget(
        source=GymAgentSource(
            component="simple_agent", config="responses_api_agents/simple_agent/configs/simple_agent.yaml"
        ),
        resources_server="missing-e2e-resources-server",
        num_repeats=1,
        concurrency=1,
        hydra_params={
            "policy_base_url": "http://unused.invalid/v1",
            "policy_api_key": "not-used",
            "policy_model_name": "not-used",
        },
    )
    job_name = _submit_agent_eval_job(
        evaluator_client,
        evaluator_workspace,
        {"tasks": _gym_task_payloads(1), "target": target.model_dump(mode="json")},
    )
    try:
        job = wait_for_platform_job(evaluator_client, job_name, evaluator_workspace, timeout=240)
        assert job.status.lower() == "error", f"job {job_name!r} ended {job.status!r}"

        job_status = JobsClient.from_client(evaluator_client).get_job_status(
            workspace=evaluator_workspace, name=job_name
        )
        assert job_status.data().steps[0].status == "error"
    finally:
        _cleanup_evaluator_job(evaluator_client, job_name)
