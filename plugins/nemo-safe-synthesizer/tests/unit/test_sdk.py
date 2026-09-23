# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pandas as pd
import pytest
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.errors import NemoTransportError
from nemo_helix_plugin.discovery import discover, discover_entry_points
from nemo_safe_synthesizer_plugin.sdk.job import SafeSynthesizerJob
from nemo_safe_synthesizer_plugin.sdk.job_builder import SafeSynthesizerJobBuilder
from nemo_safe_synthesizer_plugin.sdk.resources import (
    AsyncSafeSynthesizerJobsResource,
    SafeSynthesizerJobsResource,
    SafeSynthesizerResource,
)


def _resp(data):
    """Wrap a payload in a NemoResponse-like object whose ``.data()`` returns it.

    Production now consumes typed-client responses via ``client.<op>(...).data()``,
    so mocked jobs-client methods return an object with a ``.data()`` accessor.
    """
    m = MagicMock()
    m.data.return_value = data
    return m


def _binary_resp(data: bytes):
    """Wrap bytes in a binary NemoResponse-like object whose ``.read()`` returns them."""
    m = MagicMock()
    m.read.return_value = data
    return m


def _paginated_resp(items, *, total: int, next_page: str | None, prev_page: str | None = None):
    response = MagicMock()
    response.page.return_value = SimpleNamespace(
        items=items,
        metadata={"total": total, "next_page": next_page, "prev_page": prev_page},
    )
    response.items.return_value = iter(items)
    return response


def _mock_platform(requests: list[httpx.Request]) -> NeMoHelix:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            201,
            json={"name": "safe-synth-job", "status": "created", "spec": {"data_source": "default/data#input.csv"}},
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return NeMoHelix(base_url="http://nhx.test", http_client=http_client, workspace="default")


def test_safe_synthesizer_resource_creates_job_through_plugin_route() -> None:
    requests: list[httpx.Request] = []
    platform = _mock_platform(requests)
    resource = SafeSynthesizerResource(platform)

    response = resource.jobs.create(
        workspace="default",
        name="safe-synth-job",
        spec={"data_source": "default/data#input.csv", "config": {}},
    )

    assert response.name == "safe-synth-job"
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "http://nhx.test/apis/safe-synthesizer/v2/workspaces/default/jobs"
    assert json.loads(requests[0].read()) == {
        "spec": {"data_source": "default/data#input.csv", "config": {}},
        "name": "safe-synth-job",
    }


def test_safe_synthesizer_resource_mounts_on_platform_client() -> None:
    discover.cache_clear()
    discover_entry_points.cache_clear()
    requests: list[httpx.Request] = []
    platform = _mock_platform(requests)

    response = platform.safe_synthesizer.jobs.create(
        workspace="default",
        name="safe-synth-job",
        spec={"data_source": "default/data#input.csv", "config": {}},
    )

    assert response.name == "safe-synth-job"
    assert str(requests[0].url) == "http://nhx.test/apis/safe-synthesizer/v2/workspaces/default/jobs"


def test_safe_synthesizer_resource_includes_response_detail_in_errors() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "Failed to compile safe-synthesizer job spec"})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    platform = NeMoHelix(base_url="http://nhx.test", http_client=http_client, workspace="default")
    resource = SafeSynthesizerResource(platform)

    try:
        resource.jobs.create(workspace="default", spec={"data_source": "default/data#input.csv", "config": {}})
    except httpx.HTTPStatusError as e:
        assert "Response detail: Failed to compile safe-synthesizer job spec" in str(e)
    else:
        raise AssertionError("Expected HTTPStatusError")


@pytest.mark.asyncio
async def test_async_safe_synthesizer_resource_get_logs_forwards_query_params() -> None:
    platform = MagicMock()
    resource = AsyncSafeSynthesizerJobsResource(platform)

    mock_jobs = MagicMock()
    mock_jobs.list_job_logs = AsyncMock(return_value=_paginated_resp([], total=0, next_page=None))
    with patch("nemo_safe_synthesizer_plugin.sdk.resources.client_from_platform", return_value=mock_jobs):
        response = await resource.get_logs(
            "safe-synth-job",
            workspace="default",
            limit=10,
            page_cursor="next-page",
            step_id=None,
        )

    assert response.data == []
    mock_jobs.list_job_logs.assert_awaited_once_with(
        name="safe-synth-job",
        workspace="default",
        query_params={"limit": 10, "page_cursor": "next-page"},
    )


def test_safe_synthesizer_resource_get_logs_forwards_query_params() -> None:
    platform = MagicMock()
    resource = SafeSynthesizerJobsResource(platform)
    mock_jobs = MagicMock()
    mock_jobs.list_job_logs.return_value = _paginated_resp([], total=0, next_page=None)

    with patch("nemo_safe_synthesizer_plugin.sdk.resources.client_from_platform", return_value=mock_jobs):
        response = resource.get_logs(
            "safe-synth-job",
            workspace="default",
            attempt_id=2,
            step_id="step-1",
            task_id=None,
        )

    assert response.data == []
    mock_jobs.list_job_logs.assert_called_once_with(
        name="safe-synth-job",
        workspace="default",
        query_params={"attempt_id": 2, "step_id": "step-1"},
    )


def test_job_builder_uploads_dataframe_and_creates_job() -> None:
    client = MagicMock()
    client.files.upload = MagicMock()
    client.safe_synthesizer.jobs.create.return_value = SimpleNamespace(name="safe-synth-job")

    builder = (
        SafeSynthesizerJobBuilder(client, workspace="default")
        .with_data_source(pd.DataFrame({"value": [1]}))
        .with_classify_model_provider("nvidia-build")
        .with_replace_pii()
        .synthesize()
        .with_generate(num_records=10)
        .with_hf_token_secret("hf-token")
    )

    job = builder.create_job(name="safe-synth-job")

    assert job.job_name == "safe-synth-job"
    client.files.upload.assert_called_once()
    create_kwargs = client.safe_synthesizer.jobs.create.call_args.kwargs
    assert create_kwargs["workspace"] == "default"
    assert create_kwargs["name"] == "safe-synth-job"
    assert create_kwargs["spec"]["data_source"].startswith("default/safe-synthesizer-inputs#dataset")
    assert create_kwargs["spec"]["hf_token_secret"] == "hf-token"
    config = create_kwargs["spec"]["config"]
    assert config["enable_synthesis"] is True
    assert config["enable_replace_pii"] is True
    assert config["generation"] == {"num_records": 10}
    assert config["replace_pii"]["globals"]["classify"]["classify_model_provider"] == "default/nvidia-build"


def test_job_builder_creates_pretrained_model_job_for_adapter_reuse() -> None:
    client = MagicMock()
    client.files.upload = MagicMock()
    client.safe_synthesizer.jobs.create.return_value = SimpleNamespace(name="adapter-reuse-job")

    builder = (
        SafeSynthesizerJobBuilder(client, workspace="default")
        .with_data_source(pd.DataFrame({"value": [1]}))
        .with_pretrained_model_job("first-synth-job")
        .with_generate(num_records=25)
    )

    job = builder.create_job(name="adapter-reuse-job")

    assert job.job_name == "adapter-reuse-job"
    create_kwargs = client.safe_synthesizer.jobs.create.call_args.kwargs
    assert create_kwargs["spec"]["pretrained_model_job"] == "first-synth-job"
    assert create_kwargs["spec"]["config"]["generation"] == {"num_records": 25}
    assert "pretrained_model" not in create_kwargs["spec"]["config"]["training"]


def _make_job(mock_jobs: MagicMock, name: str = "safe-synth-job", workspace: str = "default") -> SafeSynthesizerJob:
    """Build a SafeSynthesizerJob whose typed jobs client is *mock_jobs*.

    ``SafeSynthesizerJob.__init__`` resolves ``self._jobs = client_from_platform(client, JobsClient)``,
    so we patch that lookup in the job module during construction.
    """
    with patch("nemo_safe_synthesizer_plugin.sdk.job.client_from_platform", return_value=mock_jobs):
        return SafeSynthesizerJob(name, MagicMock(), workspace=workspace)


@pytest.mark.parametrize("status", ["error", "cancelled"])
def test_safe_synthesizer_job_wait_for_completion_raises_on_terminal_failure(status: str) -> None:
    mock_jobs = MagicMock()
    mock_jobs.get_job_status.return_value = _resp(
        SimpleNamespace(
            status=status,
            status_details={"reason": "failed"},
            error_details={"message": "boom"},
        )
    )
    job = _make_job(mock_jobs)

    with pytest.raises(RuntimeError, match=f"ended with status '{status}'"):
        job.wait_for_completion(poll_interval=0, verbose=False)

    mock_jobs.get_job_status.assert_called_once_with(name="safe-synth-job", workspace="default")


def test_safe_synthesizer_job_wait_ignores_typed_client_log_failures() -> None:
    mock_jobs = MagicMock()
    mock_jobs.get_job_status.side_effect = [
        _resp(SimpleNamespace(status="active", status_details={}, error_details={})),
        _resp(SimpleNamespace(status="completed", status_details={}, error_details={})),
    ]
    job = _make_job(mock_jobs)
    request = httpx.Request("GET", "http://test/apis/jobs/v2/workspaces/default/jobs/safe-synth-job/logs")

    with patch.object(
        job,
        "_fetch_logs_incremental",
        side_effect=NemoTransportError(httpx.ConnectError("Connection refused", request=request)),
    ):
        job.wait_for_completion(poll_interval=0, verbose=True)

    assert mock_jobs.get_job_status.call_count == 2


def test_safe_synthesizer_job_fetch_data_reads_synthetic_csv() -> None:
    mock_jobs = MagicMock()
    mock_jobs.download_job_result.return_value = _binary_resp(b"name,value\nalice,1\nbob,2\n")
    job = _make_job(mock_jobs)

    result = job.fetch_data()

    mock_jobs.download_job_result.assert_called_once_with(
        name="synthetic-data", job="safe-synth-job", workspace="default"
    )
    assert list(result.columns) == ["name", "value"]
    assert result["value"].tolist() == [1, 2]


def test_safe_synthesizer_job_fetch_summary_parses_json() -> None:
    mock_jobs = MagicMock()
    mock_jobs.download_job_result.return_value = _binary_resp(
        json.dumps(
            {
                "synthetic_data_quality_score": 8.5,
                "data_privacy_score": 9.0,
                "num_valid_records": 10,
                "num_prompts": 10,
                "timing": {"total_time_sec": 12.5},
            }
        ).encode()
    )
    job = _make_job(mock_jobs)

    summary = job.fetch_summary()

    mock_jobs.download_job_result.assert_called_once_with(name="summary", job="safe-synth-job", workspace="default")
    assert summary.synthetic_data_quality_score == 8.5
    assert summary.data_privacy_score == 9.0
    assert summary.timing.total_time_sec == 12.5


def test_safe_synthesizer_job_fetch_logs_follows_pagination() -> None:
    mock_jobs = MagicMock()
    options_client = MagicMock()
    mock_jobs.with_options.return_value = options_client
    first_log = SimpleNamespace(
        job="safe-synth-job",
        job_step="safe-synthesizer",
        job_task="task-1",
        message="first",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    second_log = SimpleNamespace(
        job="safe-synth-job",
        job_step="safe-synthesizer",
        job_task="task-1",
        message="second",
        timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
    )
    paginated_response = _paginated_resp([first_log, second_log], total=2, next_page=None)
    options_client.list_job_logs.return_value = paginated_response
    job = _make_job(mock_jobs)

    logs = list(job.fetch_logs(timeout=5.0))

    assert [log.message for log in logs] == ["first", "second"]
    mock_jobs.with_options.assert_called_once_with(timeout=5.0)
    options_client.list_job_logs.assert_called_once_with(name="safe-synth-job", workspace="default")
    paginated_response.items.assert_called_once_with()
