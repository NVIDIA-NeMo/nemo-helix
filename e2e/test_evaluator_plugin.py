# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container E2E tests for the nemo-evaluator plugin.

The suite exercises the plugin through the public Platform SDK against an
external deployment configured through ``NHX_BASE_URL``. Durable evaluator
jobs execute in CPU task containers, so these tests intentionally do not mock
the evaluator service or job scheduler.
"""

from __future__ import annotations

import json
import os
import tarfile
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx
import pytest
from nemo_evaluator.api.schemas import MetricInline, MetricRef
from nemo_evaluator.filesets import FilesetRef
from nemo_evaluator.jobs.agent_spec import GymRunnerTarget
from nemo_evaluator.jobs.evaluate import EvaluateInputSpec
from nemo_evaluator.sdk.job_resources import EvaluatorJobResource
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator.shared.metric_bundles.inline import InlineMetricBundlePackager
from nemo_evaluator_sdk import (
    ExactMatchMetric,
    InferenceParams,
    Model,
    ModelRef,
    RunConfig,
    RunConfigOnlineModel,
)
from nemo_evaluator_sdk.agent_eval.runtimes.gym import GymRewardMetric, discover_gym_tasks
from nemo_evaluator_sdk.enums import ModelFormat
from nemo_evaluator_sdk.metrics.llm_judge import LLMJudgeMetric
from nemo_evaluator_sdk.metrics.string_check import StringCheckMetric
from nemo_evaluator_sdk.metrics.tool_calling import ToolCallingMetric
from nemo_evaluator_sdk.values.results import EvaluationResult
from nemo_evaluator_sdk.values.scores import JSONScoreParser, RangeScore
from nemo_helix import NeMoHelix
from nemo_helix.types.inference import ModelProvider
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import NemoHTTPError
from nemo_helix_plugin.client.errors import NemoTransportError as APIConnectionError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.inference_middleware import BackendFormat
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import CreateModelEntityRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.testing import add_mock_provider, short_unique_name, wait_for_model_entity
from nhx.testing.e2e import wait_for_platform_job
from nhx.testing.utils import ensure_passthrough_virtual_model

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
    sdk: NeMoHelix,
    *,
    workspace: str,
    name: str,
    mock_response_body: dict[str, object],
    should_autoprovision_virtual_model: bool = True,
) -> ModelProvider:
    """Create an IGW mock provider or skip when the deployment does not support one."""
    try:
        return add_mock_provider(
            sdk,
            workspace=workspace,
            name=name,
            mock_response_body=mock_response_body,
            should_autoprovision_virtual_model=should_autoprovision_virtual_model,
        )
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


def _post_evaluator_payload(
    sdk: NeMoHelix,
    workspace: str,
    path: str,
    payload: Mapping[str, object],
) -> object:
    return sdk.post(
        f"/apis/evaluator/v2/workspaces/{workspace}/{path.lstrip('/')}",
        cast_to=object,
        body=dict(payload),
    )


def _wait_for_stable_model_chat_route(sdk: NeMoHelix, workspace: str, model_name: str) -> None:
    """Require stable successful chat completions through the model-entity route."""
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
            sdk.inference.gateway.model.post(
                "v1/chat/completions",
                name=model_name,
                workspace=workspace,
                body={"model": model_name, "messages": [{"role": "user", "content": "ping"}]},
                timeout=min(10.0, remaining),
            )
        except APIConnectionError as exc:
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
    sdk: NeMoHelix,
    *,
    workspace: str,
    name: str,
    mock_response_body: dict[str, object],
) -> None:
    """Create a mock model and wait until its model-entity route is stable."""
    provider = _add_mock_provider_or_skip(
        sdk,
        workspace=workspace,
        name=name,
        mock_response_body=mock_response_body,
        should_autoprovision_virtual_model=False,
    )
    client_from_platform(sdk, ModelsClient).create_model(
        workspace=workspace,
        body=CreateModelEntityRequest(
            name=name,
            backend_format=BackendFormat.OPENAI_CHAT,
            model_providers=[f"{workspace}/{provider.name}"],
        ),
        exist_ok=True,
    ).data()
    wait_for_model_entity(
        sdk,
        workspace,
        name,
        ensure_virtual_model=True,
        should_autoprovision_virtual_model=False,
    )
    ensure_passthrough_virtual_model(
        sdk,
        workspace,
        name,
        timeout=IGW_ROUTE_TIMEOUT_SECONDS,
        autoprovisioned=False,
    )
    _wait_for_stable_model_chat_route(sdk, workspace, name)


def _cleanup_evaluator_job(sdk: NeMoHelix, job_name: str) -> None:
    with suppress(Exception):
        jobs = client_from_platform(sdk, JobsClient)
        jobs.cancel_job(name=job_name, workspace=sdk.workspace)
    with suppress(Exception):
        jobs.delete_job(name=job_name, workspace=sdk.workspace)


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
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in EVALUATOR_TRANSIENT_STATUS_CODES:
                raise
            remaining = EVALUATOR_JOB_TIMEOUT_SECONDS - (time.monotonic() - started_at)
            if remaining <= 0:
                raise
            time.sleep(min(EVALUATOR_POLL_INTERVAL_SECONDS, remaining))


def _require_job_name(payload: object) -> str:
    job_name = payload.get("name") if isinstance(payload, Mapping) else None
    if not isinstance(job_name, str):
        raise TypeError(f"Unexpected job response: {payload!r}")
    return job_name


def _submit_input_spec(sdk: NeMoHelix, spec: EvaluateInputSpec) -> EvaluatorJobResource:
    payload = _post_evaluator_payload(
        sdk,
        str(sdk.workspace),
        "evaluate/jobs",
        {"spec": spec.model_dump(mode="json")},
    )
    return sdk.evaluator.get_job_resource(_require_job_name(payload))


def _metric_output_values(result: EvaluationResult, name: str) -> list[float]:
    values: list[float] = []
    for row in _rows_in_index_order(result):
        for outputs in row.metrics.values():
            for output in outputs:
                if output.name == name:
                    values.append(float(output.value))
    return values


@pytest.fixture(scope="module")
def evaluator_workspace(sdk: NeMoHelix) -> Iterator[str]:
    workspaces = client_from_platform(sdk, WorkspacesClient)
    name = short_unique_name("e2e-eval")
    try:
        workspaces.create_workspace(body=CreateWorkspaceRequest(name=name)).data()
        yield name
    finally:
        with suppress(Exception):
            workspaces.delete_workspace(name=name).data()


@pytest.fixture(scope="module")
def evaluator_sdk(sdk: NeMoHelix, evaluator_workspace: str) -> Iterator[NeMoHelix]:
    yield sdk.copy(
        workspace=evaluator_workspace,
        max_retries=2,
        timeout=EVALUATOR_JOB_TIMEOUT_SECONDS,
    )


@pytest.fixture(scope="module")
def completed_offline_job(evaluator_sdk: NeMoHelix) -> Iterator[EvaluatorJobResource]:
    job = evaluator_sdk.evaluator.submit(
        metric=_exact_match_metric(),
        dataset=_offline_rows(),
        config=RunConfig(parallelism=1),
    )
    try:
        _wait_for_evaluator_job(job)
        yield job
    finally:
        _cleanup_evaluator_job(evaluator_sdk, job.name)


def test_health_check(sdk: NeMoHelix) -> None:
    status = sdk.evaluator.plugin_status()

    assert status["plugin"] == "evaluator"
    assert status["status"] == "ok"
    assert "evaluator.evaluate" in status["jobs"]


def test_stored_metric_lifecycle(evaluator_sdk: NeMoHelix) -> None:
    name = short_unique_name("exact")
    try:
        created = evaluator_sdk.evaluator.metrics.create(name, metric=_exact_match_metric())

        assert created.name == name
        assert created.workspace == evaluator_sdk.workspace
        assert created.metric_type == "exact-match"
        assert created.payload_kind == "inline"
        assert created.bundle_ref

        retrieved = evaluator_sdk.evaluator.metrics.retrieve(name)
        assert retrieved.id == created.id
        assert retrieved.metric_type == "exact-match"
        assert retrieved.payload_digest == created.payload_digest

        listing = evaluator_sdk.evaluator.metrics.list(sort="-created_at")
        assert any(metric.name == name for metric in listing.data)

        with pytest.raises((httpx.HTTPStatusError, NemoHTTPError)) as exc_info:
            evaluator_sdk.evaluator.metrics.create(name, metric=_exact_match_metric())
        _assert_http_status(exc_info.value, 409)

        evaluator_sdk.evaluator.metrics.delete(name)
        with pytest.raises((httpx.HTTPStatusError, NemoHTTPError)) as exc_info:
            evaluator_sdk.evaluator.metrics.retrieve(name)
        _assert_http_status(exc_info.value, 404)
    finally:
        with suppress(Exception):
            evaluator_sdk.evaluator.metrics.delete(name)


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
    evaluator_sdk: NeMoHelix,
    completed_offline_job: EvaluatorJobResource,
) -> None:
    result = evaluator_sdk.evaluator.eval_results.retrieve(completed_offline_job.name)
    by_job = evaluator_sdk.evaluator.eval_results.list(job_id=completed_offline_job.name)

    assert result.job_id == completed_offline_job.name
    assert result.metric_types == ["exact-match"]
    assert result.dataset_ref is None
    assert any(item.job_id == completed_offline_job.name for item in by_job.data)


def test_fileset_fragment_and_glob_datasets(evaluator_sdk: NeMoHelix) -> None:
    """Durable plugin jobs can read selected and globbed files from Files."""
    fileset_name = short_unique_name("eval-data")
    workspace = str(evaluator_sdk.workspace)
    submitted_jobs: list[tuple[str, list[float], EvaluatorJobResource]] = []
    files = client_from_platform(evaluator_sdk, FilesClient)
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
            job = evaluator_sdk.evaluator.submit(
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
            _cleanup_evaluator_job(evaluator_sdk, job.name)
        with suppress(Exception):
            files.delete_fileset(name=fileset_name, workspace=workspace)


def test_run_config_limits_samples(evaluator_sdk: NeMoHelix) -> None:
    rows = [{"expected": str(index), "output": str(index)} for index in range(8)]
    job = evaluator_sdk.evaluator.submit(
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
        _cleanup_evaluator_job(evaluator_sdk, job.name)


def test_multi_metric_durable_evaluation(evaluator_sdk: NeMoHelix) -> None:
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
    job = _submit_input_spec(evaluator_sdk, spec)
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
        _cleanup_evaluator_job(evaluator_sdk, job.name)


def test_stored_metric_ref_executes_in_durable_job(evaluator_sdk: NeMoHelix) -> None:
    metric_name = short_unique_name("stored-exact")
    job: EvaluatorJobResource | None = None
    try:
        evaluator_sdk.evaluator.metrics.create(metric_name, metric=_exact_match_metric())
        spec = EvaluateInputSpec(
            metrics=[MetricRef(root=metric_name)],
            dataset=_offline_rows(),
            params=RunConfig(parallelism=1),
        )
        job = _submit_input_spec(evaluator_sdk, spec)
        _wait_for_evaluator_job(job)

        assert _row_score_values(job.get_result()) == [1.0, 0.0]
    finally:
        if job is not None:
            _cleanup_evaluator_job(evaluator_sdk, job.name)
        with suppress(Exception):
            evaluator_sdk.evaluator.metrics.delete(metric_name)


def test_tool_calling_metric_preserves_structured_references(evaluator_sdk: NeMoHelix) -> None:
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
    job = evaluator_sdk.evaluator.submit(
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
        _cleanup_evaluator_job(evaluator_sdk, job.name)


def test_online_evaluate_job_uses_mock_provider(
    sdk: NeMoHelix,
    evaluator_sdk: NeMoHelix,
    evaluator_workspace: str,
) -> None:
    model_name = short_unique_name("eval-model")
    _create_ready_mock_model(
        sdk,
        workspace=evaluator_workspace,
        name=model_name,
        mock_response_body=_chat_completion("Paris"),
    )
    target = Model(
        url=_internal_model_route(evaluator_workspace, model_name),
        name=model_name,
        format=ModelFormat.OPEN_AI,
    )

    job = evaluator_sdk.evaluator.submit(
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
        _cleanup_evaluator_job(evaluator_sdk, job.name)


def test_llm_judge_metric_resolves_model_ref(
    sdk: NeMoHelix,
    evaluator_sdk: NeMoHelix,
    evaluator_workspace: str,
) -> None:
    model_name = short_unique_name("eval-judge")
    _create_ready_mock_model(
        sdk,
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
    job = evaluator_sdk.evaluator.submit(
        metric=metric,
        dataset=[{"answer": "Paris"}],
        config=RunConfig(parallelism=1),
    )
    try:
        _wait_for_evaluator_job(job)

        assert _metric_output_values(job.get_result(), "quality") == [4.0]
    finally:
        _cleanup_evaluator_job(evaluator_sdk, job.name)


def _assert_runtime_input_failure(
    evaluator_sdk: NeMoHelix,
    metric: StringCheckMetric | ExactMatchMetric,
    dataset: list[dict[str, object]] | FilesetRef,
) -> None:
    job = evaluator_sdk.evaluator.submit(
        metric=metric,
        dataset=dataset,
        config=RunConfig(parallelism=1),
    )
    try:
        with pytest.raises(RuntimeError):
            _wait_for_evaluator_job(job)
        assert job.get_job_status().status.value == "error"
    finally:
        _cleanup_evaluator_job(evaluator_sdk, job.name)


def test_invalid_template_reaches_terminal_error(evaluator_sdk: NeMoHelix) -> None:
    metric = StringCheckMetric(
        operation="equals",
        left_template="{{item.missing}}",
        right_template="{{item.expected}}",
    )
    _assert_runtime_input_failure(
        evaluator_sdk,
        metric,
        [{"expected": "value", "output": "value"}],
    )


def test_missing_fileset_reaches_terminal_error(evaluator_sdk: NeMoHelix) -> None:
    dataset = FilesetRef(root=f"{evaluator_sdk.workspace}/missing-fileset#dataset.json")
    _assert_runtime_input_failure(evaluator_sdk, _exact_match_metric(), dataset)


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
    Path(__file__).resolve().parents[1] / "packages/nemo_evaluator_sdk/tests/agent_eval/fixtures/gym_mcqa_example.jsonl"
)


def _gym_task_payloads(limit: int) -> list[dict[str, object]]:
    tasks = discover_gym_tasks(GYM_MCQA_FIXTURE)[:limit]
    bundled_reward = bundle_metric(GymRewardMetric(), CloudpickleMetricBundlePackager()).model_dump(mode="json")
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
def test_gym_agent_evaluate_job_completes(
    sdk: NeMoHelix,
    evaluator_sdk: NeMoHelix,
    evaluator_workspace: str,
    tmp_path: Path,
) -> None:
    """Submits mcqa as a real agent-evaluate job and verifies it completes with real rewards."""
    model_name = short_unique_name("gym-mcqa")
    _create_ready_mock_model(
        sdk,
        workspace=evaluator_workspace,
        name=model_name,
        mock_response_body=_chat_completion(r"\boxed{B}"),
    )

    task_dicts = _gym_task_payloads(2)  # keep the run cheap; this is a wiring test

    target = GymRunnerTarget(
        agent="simple_agent",
        agent_config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
        resources_server="mcqa",
        num_repeats=1,
        concurrency=2,
        hydra_params={
            "policy_base_url": _internal_model_route(evaluator_workspace, model_name),
            "policy_api_key": "not-used-mock-provider",
            "policy_model_name": model_name,
        },
    )

    payload = _post_evaluator_payload(
        evaluator_sdk,
        evaluator_workspace,
        "agent-evaluate/jobs",
        {"spec": {"tasks": task_dicts, "target": target.model_dump(mode="json")}},
    )
    job_name = _require_job_name(payload)
    try:
        job = wait_for_platform_job(evaluator_sdk, job_name, evaluator_workspace, timeout=480)
        assert job.status.lower() == "completed", f"job {job_name!r} ended {job.status!r}"

        # Download the results archive to a tmp path.
        archive_path = tmp_path / "agent-eval-results.tar.gz"
        archive = client_from_platform(evaluator_sdk, JobsClient).download_job_result(
            name="agent-eval-results", job=job_name, workspace=evaluator_workspace
        )
        archive_path.write_bytes(archive.read())

        # Extract the trials.jsonl file from the archive.
        extract_dir = tmp_path / "agent-eval-results"
        with tarfile.open(archive_path) as archive:
            archive.extractall(extract_dir)  # noqa: S202 - trusted first-party job artifact, not user input
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
        _cleanup_evaluator_job(evaluator_sdk, job_name)


@pytest.mark.gym_e2e
def test_gym_agent_evaluate_job_invalid_config_fails(
    evaluator_sdk: NeMoHelix,
    evaluator_workspace: str,
) -> None:
    """Rejects an invalid Gym selection before starting its environment servers."""
    target = GymRunnerTarget(
        agent="simple_agent",
        agent_config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
        resources_server="missing-e2e-resources-server",
        num_repeats=1,
        concurrency=1,
        hydra_params={
            "policy_base_url": "http://unused.invalid/v1",
            "policy_api_key": "not-used",
            "policy_model_name": "not-used",
        },
    )
    payload = _post_evaluator_payload(
        evaluator_sdk,
        evaluator_workspace,
        "agent-evaluate/jobs",
        {"spec": {"tasks": _gym_task_payloads(1), "target": target.model_dump(mode="json")}},
    )
    job_name = _require_job_name(payload)
    try:
        job = wait_for_platform_job(evaluator_sdk, job_name, evaluator_workspace, timeout=240)
        assert job.status.lower() == "error", f"job {job_name!r} ended {job.status!r}"

        job_status = client_from_platform(evaluator_sdk, JobsClient).get_job_status(
            workspace=evaluator_workspace, name=job_name
        )
        assert job_status.data().steps[0].status == "error"
    finally:
        _cleanup_evaluator_job(evaluator_sdk, job_name)
