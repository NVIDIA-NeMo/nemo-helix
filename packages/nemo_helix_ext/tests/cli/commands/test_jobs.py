# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from nemo_helix_ext.cli.commands.jobs import (
    _generate_jobs_python_code,
    download_results,
    get_logs_jobs,
    list_attempts,
    list_jobs,
    list_steps,
    retrieve_attempt,
    retrieve_results,
    tail_platform_job,
    update_jobs,
    update_status_steps,
)
from nemo_helix_plugin.client.client import NemoClient


class _Response:
    def __init__(self, body: object) -> None:
        self._body = body

    def data(self) -> object:
        return self._body


class _PaginatedResponse:
    def __init__(self, *pages: SimpleNamespace) -> None:
        self._pages = pages

    def page(self) -> SimpleNamespace:
        return self._pages[0]

    def pages(self) -> tuple[SimpleNamespace, ...]:
        return self._pages


class _BinaryResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.stream_calls = 0

    def read(self) -> bytes:
        raise AssertionError("download_results should stream chunks instead of reading the full artifact")

    @contextmanager
    def stream(self) -> Iterator[Iterator[bytes]]:
        self.stream_calls += 1
        yield iter(self._chunks)


def _ctx(client: object) -> SimpleNamespace:
    state = MagicMock()
    state.get_client.return_value = client
    state.get_output_format.return_value = None
    state.get_no_truncate.return_value = False
    state.get_timestamp_format.return_value = None
    return SimpleNamespace(obj=state)


def _job_page_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        request=request,
        json={
            "data": [
                {
                    "id": "job-id",
                    "attempt_id": "attempt-1",
                    "fileset": "fileset-1",
                    "name": "job-a",
                    "platform_spec": {
                        "steps": [
                            {
                                "name": "step-a",
                                "executor": {"provider": "cpu", "container": {"image": "test-image"}},
                            }
                        ]
                    },
                    "source": "customizer",
                    "spec": {},
                    "status": "active",
                    "workspace": "test-workspace",
                }
            ],
            "pagination": {
                "page": 2,
                "page_size": 10,
                "current_page_size": 1,
                "total_pages": 3,
                "total_results": 21,
            },
        },
    )


def test_list_jobs_uses_source_owned_client_and_keeps_first_page_semantics() -> None:
    platform_client = SimpleNamespace()
    jobs_client = MagicMock()
    item = SimpleNamespace(name="job-a")
    jobs_client.list_jobs.return_value = _PaginatedResponse(
        SimpleNamespace(
            items=[item],
            metadata={
                "page": 2,
                "page_size": 10,
                "current_page_size": 1,
                "total_pages": 3,
                "total_results": 21,
            },
        )
    )

    with (
        patch("nemo_helix_ext.cli.commands.jobs.JobsClient.from_client", return_value=jobs_client),
        patch("nemo_helix_ext.cli.commands.jobs.format_output") as format_output,
        patch("nemo_helix_ext.cli.commands.jobs.warn_if_more_pages") as warn_if_more_pages,
    ):
        list_jobs(
            _ctx(platform_client),
            workspace="test-workspace",
            filter_name="job-a",
            filter_status=["active"],
            page=2,
            page_size=10,
            sort="-created_at",
        )

    jobs_client.list_jobs.assert_called_once()
    call_kwargs = jobs_client.list_jobs.call_args.kwargs
    assert call_kwargs["workspace"] == "test-workspace"
    assert json.loads(call_kwargs["query_params"]["filter"]) == {"name": "job-a", "status": ["active"]}
    assert call_kwargs["query_params"]["page"] == 2
    assert call_kwargs["query_params"]["page_size"] == 10
    assert call_kwargs["query_params"]["sort"] == "-created_at"
    output_page = format_output.call_args.args[0]
    assert output_page.data == [item]
    assert output_page.pagination.total_pages == 3
    warn_if_more_pages.assert_called_once()


def test_list_jobs_sends_serialized_filter_query_to_source_owned_client() -> None:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return _job_page_response(request)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    platform = NemoClient(base_url="http://nemo.test", workspace="test-workspace", http_client=http_client)
    if True:
        with (
            patch("nemo_helix_ext.cli.commands.jobs.format_output"),
            patch("nemo_helix_ext.cli.commands.jobs.warn_if_more_pages"),
        ):
            list_jobs(
                _ctx(platform),
                workspace="test-workspace",
                filter_name="job-a",
                filter_status=["active"],
                page=2,
                page_size=10,
                sort="-created_at",
            )

    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.url.path == "/apis/jobs/v2/workspaces/test-workspace/jobs"
    assert json.loads(request.url.params["filter"]) == {"name": "job-a", "status": ["active"]}
    assert request.url.params["page"] == "2"
    assert request.url.params["page_size"] == "10"
    assert request.url.params["sort"] == "-created_at"


def test_list_steps_serializes_json_filter_for_source_owned_client() -> None:
    platform_client = SimpleNamespace()
    jobs_client = MagicMock()
    item = SimpleNamespace(name="step-a")
    jobs_client.list_steps.return_value = _PaginatedResponse(
        SimpleNamespace(
            items=[item],
            metadata={
                "page": 1,
                "page_size": 5,
                "current_page_size": 1,
                "total_pages": 1,
                "total_results": 1,
            },
        )
    )

    with (
        patch("nemo_helix_ext.cli.commands.jobs.JobsClient.from_client", return_value=jobs_client),
        patch("nemo_helix_ext.cli.commands.jobs.format_output") as format_output,
        patch("nemo_helix_ext.cli.commands.jobs.warn_if_more_pages") as warn_if_more_pages,
    ):
        list_steps(
            _ctx(platform_client),
            name="job-a",
            workspace="test-workspace",
            filter='{"status": "active"}',
            filter_source="customizer",
            page=1,
            page_size=5,
            sort="created_at",
        )

    jobs_client.list_steps.assert_called_once()
    call_kwargs = jobs_client.list_steps.call_args.kwargs
    assert call_kwargs["workspace"] == "test-workspace"
    assert call_kwargs["name"] == "job-a"
    assert json.loads(call_kwargs["query_params"]["filter"]) == {"status": "active", "source": "customizer"}
    assert call_kwargs["query_params"]["page"] == 1
    assert call_kwargs["query_params"]["page_size"] == 5
    assert call_kwargs["query_params"]["sort"] == "created_at"
    output_page = format_output.call_args.args[0]
    assert output_page.data == [item]
    warn_if_more_pages.assert_called_once()


def test_download_results_streams_chunks_to_output_file(tmp_path: Path) -> None:
    platform_client = SimpleNamespace()
    jobs_client = MagicMock()
    binary_response = _BinaryResponse([b"metric-", b"bytes"])
    output_file = tmp_path / "metrics.bin"
    jobs_client.download_job_result.return_value = binary_response

    with (
        patch("nemo_helix_ext.cli.commands.jobs.JobsClient.from_client", return_value=jobs_client),
        patch("nemo_helix_ext.cli.commands.jobs.typer.echo") as echo,
    ):
        download_results(
            _ctx(platform_client),
            name="metrics",
            workspace="test-workspace",
            job="job-a",
            output_file=output_file,
        )

    jobs_client.download_job_result.assert_called_once_with(
        workspace="test-workspace",
        job="job-a",
        name="metrics",
    )
    assert output_file.read_bytes() == b"metric-bytes"
    assert binary_response.stream_calls == 1
    echo.assert_called_once()
    assert echo.call_args.args[0].endswith(f"Downloaded to {output_file!r}")


def test_get_logs_uses_source_owned_cursor_page() -> None:
    platform_client = SimpleNamespace()
    jobs_client = MagicMock()
    log = SimpleNamespace(message="started")
    jobs_client.list_job_logs.return_value = _PaginatedResponse(
        SimpleNamespace(items=[log], metadata={"total": 5, "next_page": "cursor-2", "prev_page": None})
    )

    with (
        patch("nemo_helix_ext.cli.commands.jobs.JobsClient.from_client", return_value=jobs_client),
        patch("nemo_helix_ext.cli.commands.jobs.format_output") as format_output,
        patch("nemo_helix_ext.cli.commands.jobs.warn_if_more_pages") as warn_if_more_pages,
    ):
        get_logs_jobs(
            _ctx(platform_client),
            name="job-a",
            workspace="test-workspace",
            attempt_id="current",
            limit=25,
            page_cursor="cursor-1",
            step_id="step-a",
            tail=100,
            task_id="task-a",
        )

    jobs_client.list_job_logs.assert_called_once_with(
        workspace="test-workspace",
        name="job-a",
        query_params={
            "attempt_id": "current",
            "limit": 25,
            "page_cursor": "cursor-1",
            "step_id": "step-a",
            "tail": 100,
            "task_id": "task-a",
        },
    )
    output_page = format_output.call_args.args[0]
    assert output_page.data == [log]
    assert output_page.next_page == "cursor-2"
    warn_if_more_pages.assert_called_once()


def test_tail_platform_job_renders_newest_log_lines() -> None:
    platform_client = SimpleNamespace()
    jobs_client = MagicMock()
    log = SimpleNamespace(message="newest")
    jobs_client.list_job_logs.return_value = _PaginatedResponse(SimpleNamespace(items=[log], metadata={}))

    with (
        patch("nemo_helix_ext.cli.commands.jobs.JobsClient.from_client", return_value=jobs_client),
        patch("nemo_helix_ext.cli.commands.jobs.render_job_logs") as render_job_logs,
    ):
        tail_platform_job(
            _ctx(platform_client),
            name="job-a",
            lines=50,
            workspace="test-workspace",
            attempt_id="current",
            step_id="step-a",
            task_id="task-a",
        )

    jobs_client.list_job_logs.assert_called_once_with(
        workspace="test-workspace",
        name="job-a",
        query_params={
            "tail": 50,
            "attempt_id": "current",
            "step_id": "step-a",
            "task_id": "task-a",
        },
    )
    render_job_logs.assert_called_once_with([log])


def test_update_status_steps_maps_body_to_source_owned_request() -> None:
    platform_client = SimpleNamespace()
    jobs_client = MagicMock()
    updated_step = SimpleNamespace(name="step-a")
    jobs_client.update_job_step_status.return_value = _Response(updated_step)

    with (
        patch("nemo_helix_ext.cli.commands.jobs.JobsClient.from_client", return_value=jobs_client),
        patch("nemo_helix_ext.cli.commands.jobs.format_output") as format_output,
    ):
        update_status_steps(
            _ctx(platform_client),
            name="step-a",
            workspace="test-workspace",
            job="job-a",
            status="active",
            status_details='{"progress": 25}',
        )

    jobs_client.update_job_step_status.assert_called_once()
    call_kwargs = jobs_client.update_job_step_status.call_args.kwargs
    assert call_kwargs["workspace"] == "test-workspace"
    assert call_kwargs["job"] == "job-a"
    assert call_kwargs["name"] == "step-a"
    assert call_kwargs["body"].status == "active"
    assert call_kwargs["body"].status_details == {"progress": 25}
    format_output.assert_called_once_with(
        updated_step,
        is_list=False,
        output_format=None,
        no_truncate=False,
        timestamp_format=None,
    )


def test_list_attempts_uses_paginated_source_owned_route() -> None:
    platform_client = SimpleNamespace()
    jobs_client = MagicMock()
    attempt = SimpleNamespace(id="att-1")
    jobs_client.list_job_attempts.return_value = _PaginatedResponse(
        SimpleNamespace(
            items=[attempt],
            metadata={
                "page": 1,
                "page_size": 25,
                "current_page_size": 1,
                "total_pages": 1,
                "total_results": 1,
            },
        )
    )

    with (
        patch("nemo_helix_ext.cli.commands.jobs.JobsClient.from_client", return_value=jobs_client),
        patch("nemo_helix_ext.cli.commands.jobs.format_output") as format_output,
        patch("nemo_helix_ext.cli.commands.jobs.warn_if_more_pages"),
    ):
        list_attempts(
            _ctx(platform_client),
            job="job-a",
            workspace="test-workspace",
            page=1,
            page_size=25,
            sort="-seq",
        )

    jobs_client.list_job_attempts.assert_called_once_with(
        workspace="test-workspace",
        job="job-a",
        query_params={"page": 1, "page_size": 25, "sort": "-seq"},
    )
    assert format_output.call_args.args[0].data == [attempt]


def test_retrieve_current_attempt_uses_selector() -> None:
    platform_client = SimpleNamespace()
    jobs_client = MagicMock()
    attempt = SimpleNamespace(id="att-1")
    jobs_client.get_job_attempt.return_value = _Response(attempt)

    with (
        patch("nemo_helix_ext.cli.commands.jobs.JobsClient.from_client", return_value=jobs_client),
        patch("nemo_helix_ext.cli.commands.jobs.format_output"),
    ):
        retrieve_attempt(
            _ctx(platform_client),
            attempt="current",
            job="job-a",
            workspace="test-workspace",
        )

    jobs_client.get_job_attempt.assert_called_once_with(
        workspace="test-workspace",
        job="job-a",
        attempt="current",
    )


def test_update_job_preserves_explicit_null_description() -> None:
    platform_client = SimpleNamespace()
    jobs_client = MagicMock()
    updated_job = SimpleNamespace(name="job-a")
    jobs_client.update_job.return_value = _Response(updated_job)

    with (
        patch("nemo_helix_ext.cli.commands.jobs.JobsClient.from_client", return_value=jobs_client),
        patch("nemo_helix_ext.cli.commands.jobs.format_output"),
    ):
        update_jobs(
            _ctx(platform_client),
            name="job-a",
            workspace="test-workspace",
            input_data='{"description": null}',
        )

    call_kwargs = jobs_client.update_job.call_args.kwargs
    assert call_kwargs["workspace"] == "test-workspace"
    assert call_kwargs["body"].description is None
    assert "description" in call_kwargs["body"].model_fields_set


def test_download_result_with_attempt_uses_canonical_route(tmp_path: Path) -> None:
    platform_client = SimpleNamespace()
    jobs_client = MagicMock()
    binary_response = _BinaryResponse([b"bytes"])
    jobs_client.download_attempt_result.return_value = binary_response
    output_file = tmp_path / "metrics.bin"

    with (
        patch("nemo_helix_ext.cli.commands.jobs.JobsClient.from_client", return_value=jobs_client),
        patch("nemo_helix_ext.cli.commands.jobs.typer.echo"),
    ):
        download_results(
            _ctx(platform_client),
            name="metrics",
            workspace="test-workspace",
            job="job-a",
            attempt_id="att-1",
            output_file=output_file,
        )

    jobs_client.download_attempt_result.assert_called_once_with(
        workspace="test-workspace",
        job="job-a",
        attempt="att-1",
        name="metrics",
    )
    assert output_file.read_bytes() == b"bytes"


def test_retrieve_result_requires_attempt_route() -> None:
    platform_client = SimpleNamespace()
    jobs_client = MagicMock()
    result = SimpleNamespace(name="metrics")
    jobs_client.get_attempt_result.return_value = _Response(result)

    with (
        patch("nemo_helix_ext.cli.commands.jobs.JobsClient.from_client", return_value=jobs_client),
        patch("nemo_helix_ext.cli.commands.jobs.format_output"),
    ):
        retrieve_results(
            _ctx(platform_client),
            name="metrics",
            job="job-a",
            attempt_id="att-1",
            workspace="test-workspace",
        )

    jobs_client.get_attempt_result.assert_called_once_with(
        workspace="test-workspace",
        job="job-a",
        attempt="att-1",
        name="metrics",
    )


def test_retrieve_result_defaults_to_current_attempt() -> None:
    platform_client = SimpleNamespace()
    jobs_client = MagicMock()
    jobs_client.get_attempt_result.return_value = _Response(SimpleNamespace(name="metrics"))

    with (
        patch("nemo_helix_ext.cli.commands.jobs.JobsClient.from_client", return_value=jobs_client),
        patch("nemo_helix_ext.cli.commands.jobs.format_output"),
    ):
        retrieve_results(
            _ctx(platform_client),
            name="metrics",
            job="job-a",
            workspace="test-workspace",
        )

    jobs_client.get_attempt_result.assert_called_once_with(
        workspace="test-workspace",
        job="job-a",
        attempt="current",
        name="metrics",
    )


def test_jobs_code_generation_uses_source_owned_client_without_path_args_in_query_params() -> None:
    code = _generate_jobs_python_code(
        resource_path=["jobs", "results"],
        method="list",
        args={"workspace": "test-workspace", "name": "job-a", "sort": "-created_at"},
        base_url="http://example.test",
        wait_config=None,
        wait_options=None,
        watch_config=None,
        watch_options=None,
    )

    assert "client.jobs" not in code
    assert "JobsClient" in code
    assert "if key not in {'workspace', 'name'}" in code


def test_jobs_download_code_generation_streams_result() -> None:
    code = _generate_jobs_python_code(
        resource_path=["jobs", "results"],
        method="download",
        args={
            "workspace": "test-workspace",
            "job": "job-a",
            "name": "metrics",
            "output_file": "metrics.bin",
        },
        base_url="http://example.test",
        wait_config=None,
        wait_options=None,
        watch_config=None,
        watch_options=None,
    )

    assert ".read()" not in code
    assert "with Path(args['output_file']).open('wb') as output:" in code
    assert "with response.stream() as chunks:" in code
    assert "output.write(chunk)" in code


def test_jobs_result_get_code_generation_uses_attempt_route() -> None:
    code = _generate_jobs_python_code(
        resource_path=["jobs", "results"],
        method="retrieve",
        args={
            "workspace": "test-workspace",
            "job": "job-a",
            "attempt_id": "att-1",
            "name": "metrics",
        },
        base_url="http://example.test",
        wait_config=None,
        wait_options=None,
        watch_config=None,
        watch_options=None,
    )

    assert "get_attempt_result" in code
    assert "get_job_result" not in code
    assert "attempt=args['attempt_id']" in code
