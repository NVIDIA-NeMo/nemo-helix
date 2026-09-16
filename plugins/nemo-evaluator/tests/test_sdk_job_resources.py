# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluator plugin job resources."""

from __future__ import annotations

import json
import tarfile
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, Mock

import httpx
import pytest
from nemo_evaluator.sdk.job_resources import (
    AsyncEvaluatorJobResource,
    EvaluatorJob,
    EvaluatorJobResource,
    _coerce_aggregate_scores,
    _coerce_row_score,
    _poll_until_terminal,
    _raise_for_terminal_status,
    _status_is_complete,
    metric_job_status_details_value,
    metric_job_status_value,
)
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_evaluator_sdk.values.results import (
    AggregatedMetricResult,
    AggregateRangeScore,
    EvaluationResult,
    MetricOutput,
    RowScore,
)
from nemo_helix_plugin.client.errors import InternalServerError
from nemo_helix_plugin.evaluator.client import AsyncEvaluatorClient, EvaluatorClient
from nemo_helix_plugin.jobs.schemas import PlatformJobStatus, PlatformJobStatusResponse
from pydantic import BaseModel
from pytest_mock import MockerFixture

_JOB_PAYLOAD = {
    "name": "job-123",
    "status": "created",
    "spec": {
        "metrics": [
            bundle_metric(
                ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}"),
                CloudpickleMetricBundlePackager(),
            ).model_dump(mode="json")
        ],
        "dataset": [{"expected": "a", "output": "a"}],
    },
}
_STATUS_URL = "https://nhx.test/apis/evaluator/v2/workspaces/client-ws/evaluate/jobs/job-123/status"
_AGGREGATE_URL = (
    "https://nhx.test/apis/evaluator/v2/workspaces/client-ws/evaluate/jobs/job-123/results/aggregate-scores/download"
)
_ROW_SCORES_URL = (
    "https://nhx.test/apis/evaluator/v2/workspaces/client-ws/evaluate/jobs/job-123/results/row-scores/download"
)
_ARTIFACTS_URL = (
    "https://nhx.test/apis/evaluator/v2/workspaces/client-ws/evaluate/jobs/job-123/results/artifacts/download"
)


@pytest.fixture
def http_client(mocker: MockerFixture) -> Mock:
    """Return a sync HTTP client mock for evaluator job-resource calls."""
    return mocker.Mock(spec=httpx.Client)


@pytest.fixture
async def async_http_client() -> AsyncIterator[httpx.AsyncClient]:
    """Return an async HTTP client with network calls handled by tests."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def job() -> EvaluatorJob:
    """Return a typed evaluator job payload."""
    return EvaluatorJob.model_validate(_JOB_PAYLOAD)


@pytest.fixture
def job_resource(http_client: Mock, job: EvaluatorJob) -> EvaluatorJobResource:
    """Return a sync evaluator job resource."""
    client = EvaluatorClient(
        base_url="https://nhx.test",
        workspace="client-ws",
        default_headers={"Authorization": "Bearer platform-token"},
        http_client=cast(httpx.Client, http_client),
    )
    return EvaluatorJobResource(
        job=job,
        client=client,
        workspace="client-ws",
    )


@pytest.fixture
async def async_job_resource(async_http_client: httpx.AsyncClient, job: EvaluatorJob) -> AsyncEvaluatorJobResource:
    """Return an async evaluator job resource."""
    client = AsyncEvaluatorClient(
        base_url="https://nhx.test",
        workspace="client-ws",
        default_headers={"Authorization": "Bearer platform-token"},
        http_client=async_http_client,
    )
    return AsyncEvaluatorJobResource(
        job=job,
        client=client,
        workspace="client-ws",
    )


def _stream_response(response: httpx.Response) -> MagicMock:
    context = MagicMock()
    context.__enter__.return_value = response
    context.__exit__.return_value = None
    return context


def _status_response(
    status: PlatformJobStatus | str,
    *,
    status_details: dict[str, object] | None = None,
    error_details: dict[str, object] | None = None,
) -> PlatformJobStatusResponse:
    """Return a platform job status response for resource tests from an enum or wire value."""
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    return PlatformJobStatusResponse(
        id="job-id",
        created_at=now,
        error_details=error_details or {},
        name="job-123",
        status=PlatformJobStatus(status),
        status_details=status_details or {},
        steps=[],
        updated_at=now,
    )


def _evaluation_result_parts() -> tuple[AggregatedMetricResult, list[RowScore]]:
    """Return typed aggregate and row scores for result download tests."""
    aggregate_scores = AggregatedMetricResult(
        scores=[
            AggregateRangeScore(
                name="serializable.score",
                count=1,
                nan_count=0,
                sum=1.0,
                mean=1.0,
                min=1.0,
                max=1.0,
            )
        ]
    )
    row_scores = [
        RowScore(
            row_index=0,
            item={"expected": "a", "output": "a"},
            sample={},
            metrics={"serializable": [MetricOutput(name="score", value=1.0)]},
            requests=[],
        )
    ]
    return aggregate_scores, row_scores


def _artifact_tar_bytes(member_name: str = "artifacts/report.json", *, member_type: bytes | None = None) -> bytes:
    """Return tar.gz bytes containing one artifact member for download tests."""
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        if member_type is not None:
            info = tarfile.TarInfo(name=member_name)
            info.type = member_type
            if member_type in {tarfile.SYMTYPE, tarfile.LNKTYPE}:
                info.linkname = "target"
            tar.addfile(info)
        else:
            data = b'{"ok": true}'
            info = tarfile.TarInfo(name=member_name)
            info.size = len(data)
            tar.addfile(info, BytesIO(data))
    return buffer.getvalue()


def test_name_and_job_properties(job_resource: EvaluatorJobResource, job: EvaluatorJob) -> None:
    """The resource should expose the job name and raw job payload."""
    assert job_resource.name == "job-123"
    assert job_resource.job is job


def test_metric_job_status_helpers_handle_empty_and_detailed_payloads() -> None:
    """Status helpers should tolerate non-string status values and surface details only when present."""
    status = cast(
        PlatformJobStatusResponse,
        SimpleNamespace(status=None, status_details={"step": "compile"}),
    )

    assert metric_job_status_value(status) == ""
    assert metric_job_status_details_value(status) == {"step": "compile"}
    assert metric_job_status_details_value(_status_response("active")) is None


def test_score_coercion_accepts_existing_models_and_base_models() -> None:
    """Score coercion should accept SDK values directly and pydantic-compatible generated SDK values."""

    class RowScorePayload(BaseModel):
        row_index: int
        item: dict[str, object]
        sample: dict[str, object]
        metrics: dict[str, list[dict[str, object]]]
        requests: list[object]

    class AggregatePayload(BaseModel):
        scores: list[dict[str, object]]

    aggregate_scores, row_scores = _evaluation_result_parts()

    assert _coerce_row_score(row_scores[0]) is row_scores[0]
    assert _coerce_row_score(RowScorePayload.model_validate(row_scores[0].model_dump(mode="json"))) == row_scores[0]
    assert _coerce_aggregate_scores(aggregate_scores) is aggregate_scores
    assert (
        _coerce_aggregate_scores(AggregatePayload.model_validate(aggregate_scores.model_dump(mode="json")))
        == aggregate_scores
    )


def test_get_job_status_delegates_to_metric_jobs_resource(
    job_resource: EvaluatorJobResource,
    http_client: Mock,
) -> None:
    """Status checks should call the evaluator plugin status route over HTTP."""
    status = _status_response("active")
    http_client.request.return_value = httpx.Response(
        200,
        request=httpx.Request("GET", _STATUS_URL),
        json=status.model_dump(mode="json"),
    )

    assert job_resource.get_job_status() == status
    http_client.request.assert_called_once_with(
        "GET",
        _STATUS_URL,
        content=None,
        headers={"Authorization": "Bearer platform-token"},
        params=None,
    )


def test_get_job_status_accepts_nullable_detail_fields(
    job_resource: EvaluatorJobResource,
    http_client: Mock,
) -> None:
    """Status payloads may contain null error details while status details remain structured."""
    status = _status_response("active")
    payload = status.model_dump(mode="json")
    payload["error_details"] = None
    http_client.request.return_value = httpx.Response(
        200,
        request=httpx.Request("GET", _STATUS_URL),
        json=payload,
    )

    parsed = job_resource.get_job_status()

    assert parsed.error_details is None
    assert parsed.status_details == {}


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("completed", True),
        ("active", False),
        ("pending", False),
        ("created", False),
        ("error", False),
        ("cancelled", False),
    ],
)
def test_check_if_complete_returns_status_result(
    job_resource: EvaluatorJobResource,
    mocker: MockerFixture,
    status: PlatformJobStatus,
    expected: bool,
) -> None:
    """Completion checks should distinguish completed, running, and failed jobs."""
    mocker.patch.object(job_resource, "get_job_status", return_value=_status_response(status))

    assert job_resource.check_if_complete() is expected


@pytest.mark.parametrize("status", ["active", "pending", "created", "error", "cancelled"])
def test_check_if_complete_raises_when_requested(
    job_resource: EvaluatorJobResource,
    mocker: MockerFixture,
    status: PlatformJobStatus,
) -> None:
    """Completion checks should raise for non-completed statuses when requested."""
    mocker.patch.object(job_resource, "get_job_status", return_value=_status_response(status))

    with pytest.raises(RuntimeError, match=status):
        job_resource.check_if_complete(raise_if_not_complete=True)


def test_check_if_complete_handles_unknown_status(caplog: pytest.LogCaptureFixture) -> None:
    """Unknown statuses should log in soft mode and raise in strict mode."""
    status = cast(
        PlatformJobStatusResponse,
        SimpleNamespace(status="mystery", name="job-123", error_details={}),
    )

    assert _status_is_complete(status, raise_if_not_complete=False) is False
    assert "unknown state" in caplog.text
    with pytest.raises(RuntimeError, match="unknown state"):
        _status_is_complete(status, raise_if_not_complete=True)


def test_wait_until_done_returns_after_completed_status(
    job_resource: EvaluatorJobResource,
    mocker: MockerFixture,
) -> None:
    """Wait should poll until the evaluator job reaches completed status."""
    get_status = mocker.patch.object(
        job_resource,
        "get_job_status",
        side_effect=[_status_response("active"), _status_response("completed")],
    )
    pause = mocker.patch("nemo_evaluator.sdk.job_resources._pause")

    job_resource.wait_until_done(poll_interval_seconds=0)

    assert get_status.call_count == 2
    pause.assert_called_once_with(0)


def test_wait_until_done_raises_for_terminal_failure(
    job_resource: EvaluatorJobResource,
    mocker: MockerFixture,
) -> None:
    """Wait should raise terminal failure details without downloading results."""
    mocker.patch.object(
        job_resource,
        "get_job_status",
        return_value=_status_response("error", error_details={"message": "task failed"}),
    )

    with pytest.raises(RuntimeError, match=r"job-123.*error.*task failed"):
        job_resource.wait_until_done(poll_interval_seconds=0)


def test_wait_until_done_raises_for_unexpected_terminal_status(
    job_resource: EvaluatorJobResource,
    mocker: MockerFixture,
) -> None:
    """Unexpected terminal statuses should fail loudly."""
    unexpected = cast(
        PlatformJobStatusResponse,
        SimpleNamespace(status="done-ish", name="job-123", error_details={}),
    )

    mocker.patch.object(job_resource, "get_job_status", return_value=unexpected)
    with pytest.raises(RuntimeError, match="unexpected terminal status"):
        _raise_for_terminal_status(unexpected)


def test_wait_until_done_does_not_sleep_after_timeout(
    job_resource: EvaluatorJobResource,
    mocker: MockerFixture,
) -> None:
    """Sync timeout must fire before sleeping once the budget is already exhausted."""
    fake_now = [0.0]

    def slow_status() -> PlatformJobStatusResponse:
        """Advance the fake clock past the configured timeout on each poll."""
        fake_now[0] += 2.0
        return _status_response("active")

    mocker.patch("nemo_evaluator.sdk.job_resources.monotonic", side_effect=lambda: fake_now[0])
    pause = mocker.patch("nemo_evaluator.sdk.job_resources._pause")
    mocker.patch.object(job_resource, "get_job_status", side_effect=slow_status)

    with pytest.raises(TimeoutError, match=r"timed out after 1\.0s.*Status: active"):
        job_resource.wait_until_done(poll_interval_seconds=10, job_timeout_seconds=1.0)

    pause.assert_not_called()


def test_poll_until_terminal_tracks_pending_elapsed_and_details(mocker: MockerFixture) -> None:
    """Pending polls should account for poll and sleep time before returning a terminal status."""
    statuses = [
        _status_response("pending", status_details={"queue": "cpu"}),
        _status_response("completed"),
    ]
    get_status = mocker.Mock(side_effect=statuses)
    mocker.patch("nemo_evaluator.sdk.job_resources.monotonic", side_effect=[0.0, 0.25, 0.25, 0.75, 0.75])
    pause = mocker.patch("nemo_evaluator.sdk.job_resources._pause")

    assert (
        _poll_until_terminal(
            get_status,
            job_name="job-123",
            timeout=10.0,
            pending_timeout=2.0,
            poll_interval=0.5,
        )
        is statuses[-1]
    )
    pause.assert_called_once_with(0.5)


def test_poll_until_terminal_raises_for_pending_timeout(mocker: MockerFixture) -> None:
    """Pending timeout should fire before sleeping when the pending budget is exhausted."""
    mocker.patch("nemo_evaluator.sdk.job_resources.monotonic", side_effect=[0.0, 3.0])
    pause = mocker.patch("nemo_evaluator.sdk.job_resources._pause")

    with pytest.raises(TimeoutError, match="stuck in pending"):
        _poll_until_terminal(
            mocker.Mock(return_value=_status_response("pending")),
            job_name="job-123",
            timeout=10.0,
            pending_timeout=1.0,
            poll_interval=0.5,
        )

    pause.assert_not_called()


def test_get_result_returns_evaluation_result(
    job_resource: EvaluatorJobResource,
    http_client: Mock,
) -> None:
    """Result downloads should combine plugin aggregate JSON and row-score JSONL artifacts."""
    aggregate_scores, row_scores = _evaluation_result_parts()
    http_client.stream.side_effect = [
        _stream_response(
            httpx.Response(
                200,
                request=httpx.Request("GET", _AGGREGATE_URL),
                json=aggregate_scores.model_dump(mode="json"),
            )
        ),
        _stream_response(
            httpx.Response(
                200,
                request=httpx.Request("GET", _ROW_SCORES_URL),
                text="\n".join(json.dumps(row_score.model_dump(mode="json")) for row_score in row_scores) + "\n",
            )
        ),
    ]

    assert job_resource.get_result() == EvaluationResult(row_scores=row_scores, aggregate_scores=aggregate_scores)
    assert [call.args for call in http_client.stream.call_args_list] == [
        ("GET", _AGGREGATE_URL),
        ("GET", _ROW_SCORES_URL),
    ]
    assert [call.kwargs for call in http_client.stream.call_args_list] == [
        {"content": None, "headers": {"Authorization": "Bearer platform-token"}, "params": None},
        {"content": None, "headers": {"Authorization": "Bearer platform-token"}, "params": None},
    ]


def test_get_result_filters_aggregate_fields(
    http_client: Mock,
) -> None:
    """Requested aggregate_fields should shape downloaded aggregate scores."""
    job = EvaluatorJob.model_validate(_JOB_PAYLOAD)
    client = EvaluatorClient(
        base_url="https://nhx.test",
        workspace="client-ws",
        default_headers={"Authorization": "Bearer platform-token"},
        http_client=cast(httpx.Client, http_client),
    )
    job_resource = EvaluatorJobResource(
        job=job,
        client=client,
        workspace="client-ws",
    )
    aggregate_scores, row_scores = _evaluation_result_parts()
    http_client.stream.side_effect = [
        _stream_response(
            httpx.Response(
                200,
                request=httpx.Request("GET", _AGGREGATE_URL),
                json=aggregate_scores.model_dump(mode="json"),
            )
        ),
        _stream_response(
            httpx.Response(
                200,
                request=httpx.Request("GET", _ROW_SCORES_URL),
                text="\n".join(json.dumps(row_score.model_dump(mode="json")) for row_score in row_scores) + "\n",
            )
        ),
    ]

    result = job_resource.get_result(aggregate_fields=("mean",))

    assert result.aggregate_scores.model_dump(mode="json") == {
        "scores": [
            {
                "name": "serializable.score",
                "count": 1,
                "mean": 1.0,
            }
        ]
    }
    assert result.row_scores == row_scores


def test_download_artifacts_extracts_artifact_tarball(
    job_resource: EvaluatorJobResource,
    http_client: Mock,
    tmp_path: Path,
) -> None:
    """Artifact downloads should fetch and extract the full artifacts result tarball."""
    http_client.stream.return_value = _stream_response(
        httpx.Response(
            200,
            request=httpx.Request("GET", _ARTIFACTS_URL),
            content=_artifact_tar_bytes(),
        )
    )

    artifacts_path = job_resource.download_artifacts(tmp_path)

    assert artifacts_path == tmp_path / "job-123"
    assert (artifacts_path / "artifacts" / "report.json").read_text(encoding="utf-8") == '{"ok": true}'
    http_client.stream.assert_called_once_with(
        "GET",
        _ARTIFACTS_URL,
        content=None,
        headers={"Authorization": "Bearer platform-token"},
        params=None,
    )


def test_download_artifacts_defaults_to_job_name_directory(
    job_resource: EvaluatorJobResource,
    http_client: Mock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Artifact downloads should default to a directory named after the evaluator job."""
    monkeypatch.chdir(tmp_path)
    http_client.stream.return_value = _stream_response(
        httpx.Response(
            200,
            request=httpx.Request("GET", _ARTIFACTS_URL),
            content=_artifact_tar_bytes("other.txt"),
        )
    )

    output_path = job_resource.download_artifacts()

    assert output_path == Path("job-123")
    assert (tmp_path / "job-123" / "other.txt").read_text(encoding="utf-8") == '{"ok": true}'


def test_download_artifacts_raises_for_http_errors(
    job_resource: EvaluatorJobResource,
    http_client: Mock,
    tmp_path: Path,
) -> None:
    """Artifact downloads should surface non-successful HTTP responses."""
    http_client.stream.return_value = _stream_response(
        httpx.Response(
            500,
            request=httpx.Request("GET", _ARTIFACTS_URL),
        )
    )

    with pytest.raises(InternalServerError):
        job_resource.download_artifacts(tmp_path)


@pytest.mark.parametrize(
    ("member_name", "member_type"),
    [
        ("../escape.txt", None),
        ("@absolute_escape", None),
        ("artifacts/link", tarfile.SYMTYPE),
        ("artifacts/fifo", tarfile.FIFOTYPE),
    ],
)
def test_download_artifacts_rejects_unsafe_tar_members(
    job_resource: EvaluatorJobResource,
    http_client: Mock,
    tmp_path: Path,
    member_name: str,
    member_type: bytes | None,
) -> None:
    """Artifact extraction should reject traversal paths, tar links, and special files."""
    # Resolve the @absolute_escape marker into a unique absolute path that lives
    # *outside* tmp_path so a buggy extraction would create a real file we can
    # detect, without colliding with system files like /tmp/escape.txt.
    if member_name == "@absolute_escape":
        member_name = str(tmp_path.parent / f"abs_escape_{tmp_path.name}.txt")

    http_client.stream.return_value = _stream_response(
        httpx.Response(
            200,
            request=httpx.Request("GET", _ARTIFACTS_URL),
            content=_artifact_tar_bytes(member_name, member_type=member_type),
        )
    )

    with pytest.raises(ValueError, match="unsafe tar member|tar link member|special tar member"):
        job_resource.download_artifacts(tmp_path)

    if Path(member_name).is_absolute():
        assert not Path(member_name).exists()
    else:
        assert not (tmp_path / "escape.txt").exists()


@pytest.mark.asyncio
async def test_async_get_job_status_delegates_to_metric_jobs_resource(
    async_job_resource: AsyncEvaluatorJobResource,
    async_http_client: httpx.AsyncClient,
) -> None:
    """Async status checks should call the evaluator plugin status route over HTTP."""
    status = _status_response("active")

    async def handler(request: httpx.Request) -> httpx.Response:
        assert (
            str(request.url) == "https://nhx.test/apis/evaluator/v2/workspaces/client-ws/evaluate/jobs/job-123/status"
        )
        assert request.headers["authorization"] == "Bearer platform-token"
        return httpx.Response(200, json=status.model_dump(mode="json"))

    async_http_client._transport = httpx.MockTransport(handler)

    assert await async_job_resource.get_job_status() == status


@pytest.mark.asyncio
async def test_async_get_job_status_accepts_nullable_detail_fields(
    async_job_resource: AsyncEvaluatorJobResource,
    async_http_client: httpx.AsyncClient,
) -> None:
    """Async status parsing should accept nullable error details from the plugin status route."""
    status = _status_response("active")
    payload = status.model_dump(mode="json")
    payload["error_details"] = None

    async def handler(request: httpx.Request) -> httpx.Response:
        assert (
            str(request.url) == "https://nhx.test/apis/evaluator/v2/workspaces/client-ws/evaluate/jobs/job-123/status"
        )
        assert request.headers["authorization"] == "Bearer platform-token"
        return httpx.Response(200, json=payload)

    async_http_client._transport = httpx.MockTransport(handler)

    parsed = await async_job_resource.get_job_status()

    assert parsed.error_details is None
    assert parsed.status_details == {}


@pytest.mark.asyncio
async def test_async_get_result_collects_async_row_score_stream(
    async_job_resource: AsyncEvaluatorJobResource,
    async_http_client: httpx.AsyncClient,
) -> None:
    """Async result downloads should parse row-score JSONL from the plugin route."""
    aggregate_scores, row_scores = _evaluation_result_parts()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer platform-token"
        if str(request.url).endswith("/aggregate-scores/download"):
            return httpx.Response(200, json=aggregate_scores.model_dump(mode="json"))
        if str(request.url).endswith("/row-scores/download"):
            return httpx.Response(
                200,
                text="\n".join(json.dumps(row_score.model_dump(mode="json")) for row_score in row_scores) + "\n",
            )
        return httpx.Response(404)

    async_http_client._transport = httpx.MockTransport(handler)

    result = await async_job_resource.get_result()

    assert result == EvaluationResult(row_scores=row_scores, aggregate_scores=aggregate_scores)


@pytest.mark.asyncio
async def test_async_get_result_filters_aggregate_fields(
    async_http_client: httpx.AsyncClient,
) -> None:
    """Async requested aggregate_fields should shape downloaded aggregate scores."""
    job = EvaluatorJob.model_validate(_JOB_PAYLOAD)
    client = AsyncEvaluatorClient(
        base_url="https://nhx.test",
        workspace="client-ws",
        default_headers={"Authorization": "Bearer platform-token"},
        http_client=async_http_client,
    )
    async_job_resource = AsyncEvaluatorJobResource(
        job=job,
        client=client,
        workspace="client-ws",
    )
    aggregate_scores, row_scores = _evaluation_result_parts()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer platform-token"
        if str(request.url).endswith("/aggregate-scores/download"):
            return httpx.Response(200, json=aggregate_scores.model_dump(mode="json"))
        if str(request.url).endswith("/row-scores/download"):
            return httpx.Response(
                200,
                text="\n".join(json.dumps(row_score.model_dump(mode="json")) for row_score in row_scores) + "\n",
            )
        return httpx.Response(404)

    async_http_client._transport = httpx.MockTransport(handler)

    result = await async_job_resource.get_result(aggregate_fields=("mean",))

    assert result.aggregate_scores.model_dump(mode="json") == {
        "scores": [
            {
                "name": "serializable.score",
                "count": 1,
                "mean": 1.0,
            }
        ]
    }
    assert result.row_scores == row_scores


@pytest.mark.asyncio
async def test_async_download_artifacts_extracts_artifact_tarball(
    async_job_resource: AsyncEvaluatorJobResource,
    async_http_client: httpx.AsyncClient,
    tmp_path: Path,
) -> None:
    """Async artifact downloads should fetch and extract the full artifacts result tarball."""

    async def handler(request: httpx.Request) -> httpx.Response:
        assert (
            str(request.url)
            == "https://nhx.test/apis/evaluator/v2/workspaces/client-ws/evaluate/jobs/job-123/results/artifacts/download"
        )
        assert request.headers["authorization"] == "Bearer platform-token"
        return httpx.Response(200, content=_artifact_tar_bytes())

    async_http_client._transport = httpx.MockTransport(handler)

    artifacts_path = await async_job_resource.download_artifacts(tmp_path)

    assert artifacts_path == tmp_path / "job-123"
    assert (artifacts_path / "artifacts" / "report.json").read_text(encoding="utf-8") == '{"ok": true}'


@pytest.mark.asyncio
async def test_async_get_result_ignores_blank_jsonl_lines(
    async_job_resource: AsyncEvaluatorJobResource,
    async_http_client: httpx.AsyncClient,
) -> None:
    """Blank lines in streamed row-score JSONL should not create empty score entries."""
    aggregate_scores, row_scores = _evaluation_result_parts()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer platform-token"
        if str(request.url).endswith("/aggregate-scores/download"):
            return httpx.Response(200, json=aggregate_scores.model_dump(mode="json"))
        if str(request.url).endswith("/row-scores/download"):
            return httpx.Response(
                200,
                text="\n\n".join(json.dumps(row_score.model_dump(mode="json")) for row_score in row_scores) + "\n\n",
            )
        return httpx.Response(404)

    async_http_client._transport = httpx.MockTransport(handler)

    result = await async_job_resource.get_result()

    assert result == EvaluationResult(row_scores=row_scores, aggregate_scores=aggregate_scores)


@pytest.mark.asyncio
async def test_async_check_if_complete_returns_status_result(
    async_job_resource: AsyncEvaluatorJobResource,
    mocker: MockerFixture,
) -> None:
    """Async completion checks should delegate through async status retrieval."""
    mocker.patch.object(
        async_job_resource, "get_job_status", new=mocker.AsyncMock(return_value=_status_response("completed"))
    )

    assert await async_job_resource.check_if_complete() is True


@pytest.mark.asyncio
async def test_async_wait_until_done_returns_after_completed_status(
    async_job_resource: AsyncEvaluatorJobResource,
    mocker: MockerFixture,
) -> None:
    """Async wait should poll with the shared async polling helper until completion."""
    poll = mocker.patch(
        "nemo_evaluator.sdk.job_resources.async_poll_until_terminal",
        new=mocker.AsyncMock(return_value=_status_response("completed")),
    )

    await async_job_resource.wait_until_done(
        poll_interval_seconds=0.1, job_timeout_seconds=2.0, pending_timeout_seconds=1.0
    )

    poll.assert_awaited_once_with(
        async_job_resource.get_job_status,
        status_value=metric_job_status_value,
        details_value=metric_job_status_details_value,
        job_name="job-123",
        terminal=frozenset({"error", "cancelled", "failed", "completed"}),
        timeout=2.0,
        pending_timeout=1.0,
        poll_interval=0.1,
    )
