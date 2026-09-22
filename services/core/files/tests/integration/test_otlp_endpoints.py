# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for OTLP log endpoints.

These tests use the SDK client fixture to make real HTTP requests
to the OTLP ingest and query endpoints with actual storage.

Tests are parametrized to run with both JSON and protobuf formats
since real OTLP clients (like Go services) send protobuf over HTTP.
"""

from dataclasses import dataclass

import httpx
import pytest
from nemo_platform_plugin.files.types import FilesetOutput
from opentelemetry.proto.collector.logs.v1 import logs_service_pb2


@dataclass
class OTLPRequest:
    """Container for OTLP request data with format-specific encoding."""

    content: bytes | dict
    content_type: str

    def post_kwargs(self) -> dict:
        """Return kwargs for httpx client.post()."""
        if self.content_type == "application/json":
            return {"json": self.content}
        else:
            return {
                "content": self.content,
                "headers": {"Content-Type": self.content_type},
            }


def _create_json_request(
    job: str,
    job_attempt: str,
    job_step: str,
    job_task: str,
    messages: list[str],
    base_timestamp_ns: int,
) -> OTLPRequest:
    """Create OTLP request in JSON format."""
    log_records = []
    for i, message in enumerate(messages):
        log_records.append(
            {
                "timeUnixNano": str(base_timestamp_ns + i * 1000000000),
                "severityNumber": 9,
                "severityText": "INFO",
                "body": {"stringValue": message},
            }
        )

    return OTLPRequest(
        content={
            "resourceLogs": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "job", "value": {"stringValue": job}},
                            {
                                "key": "job_attempt",
                                "value": {"stringValue": job_attempt},
                            },
                            {"key": "job_step", "value": {"stringValue": job_step}},
                            {"key": "job_task", "value": {"stringValue": job_task}},
                        ]
                    },
                    "scopeLogs": [
                        {
                            "scope": {"name": "test-scope"},
                            "logRecords": log_records,
                        }
                    ],
                }
            ]
        },
        content_type="application/json",
    )


def _create_protobuf_request(
    job: str,
    job_attempt: str,
    job_step: str,
    job_task: str,
    messages: list[str],
    base_timestamp_ns: int,
) -> OTLPRequest:
    """Create OTLP request in protobuf format."""
    proto_request = logs_service_pb2.ExportLogsServiceRequest()
    resource_logs = proto_request.resource_logs.add()

    # Add resource-level attributes
    for key, value in [
        ("job", job),
        ("job_attempt", job_attempt),
        ("job_step", job_step),
        ("job_task", job_task),
    ]:
        attr = resource_logs.resource.attributes.add()
        attr.key = key
        attr.value.string_value = value

    scope_logs = resource_logs.scope_logs.add()
    scope_logs.scope.name = "test-scope"

    # Add log records
    for i, message in enumerate(messages):
        log_record = scope_logs.log_records.add()
        log_record.time_unix_nano = base_timestamp_ns + i * 1000000000
        log_record.severity_number = 9  # INFO
        log_record.severity_text = "INFO"
        log_record.body.string_value = message

    return OTLPRequest(
        content=proto_request.SerializeToString(),
        content_type="application/x-protobuf",
    )


@pytest.fixture(params=["json", "protobuf"])
def otlp_format(request):
    """Parametrized fixture for OTLP format (json or protobuf)."""
    return request.param


@pytest.fixture
def otlp_request_factory(otlp_format):
    """Factory for creating OTLP log request payloads in the parametrized format."""

    def create_request(
        job: str = "test-job",
        job_attempt: str = "attempt-1",
        job_step: str = "step-1",
        job_task: str = "task-1",
        messages: list[str] | None = None,
        base_timestamp_ns: int = 1704110400000000000,  # 2024-01-01 12:00:00 UTC
    ) -> OTLPRequest:
        if messages is None:
            messages = ["Test log message"]

        if otlp_format == "json":
            return _create_json_request(job, job_attempt, job_step, job_task, messages, base_timestamp_ns)
        else:
            return _create_protobuf_request(job, job_attempt, job_step, job_task, messages, base_timestamp_ns)

    return create_request


def test_upload_and_query_logs_roundtrip(
    client: httpx.Client,
    fileset: FilesetOutput,
    otlp_request_factory,
    otlp_format: str,
):
    """Test uploading logs via OTLP and querying them back."""
    workspace = fileset.workspace
    fileset_name = fileset.name

    # Upload logs
    request = otlp_request_factory(
        job=f"integration-test-job-{otlp_format}",
        messages=["Log message 1", "Log message 2", "Log message 3"],
    )

    upload_response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs",
        **request.post_kwargs(),
    )
    assert upload_response.status_code == 200
    assert upload_response.json().get("partialSuccess") is None

    # Query logs back
    query_response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs/query",
        json={
            "filters": {"job": f"integration-test-job-{otlp_format}"},
            "limit": 100,
        },
    )
    assert query_response.status_code == 200

    result = query_response.json()
    assert result["total"] == 3
    assert len(result["data"]) == 3

    # Verify log content
    messages = {log["message"] for log in result["data"]}
    assert messages == {"Log message 1", "Log message 2", "Log message 3"}


def test_query_logs_with_filters(
    client: httpx.Client,
    fileset: FilesetOutput,
    otlp_request_factory,
    otlp_format: str,
):
    """Test querying logs with different filter combinations."""
    workspace = fileset.workspace
    fileset_name = fileset.name

    # Upload logs for two different jobs
    for job_suffix, message in [("alpha", "Alpha log"), ("beta", "Beta log")]:
        job_name = f"job-{job_suffix}-{otlp_format}"
        request = otlp_request_factory(
            job=job_name,
            messages=[message],
        )
        response = client.post(
            f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs",
            **request.post_kwargs(),
        )
        assert response.status_code == 200

    # Query for job-alpha only
    response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs/query",
        json={"filters": {"job": f"job-alpha-{otlp_format}"}},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["total"] == 1
    assert result["data"][0]["message"] == "Alpha log"

    # Query for job-beta only
    response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs/query",
        json={"filters": {"job": f"job-beta-{otlp_format}"}},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["total"] == 1
    assert result["data"][0]["message"] == "Beta log"


def test_query_logs_pagination(
    client: httpx.Client,
    fileset: FilesetOutput,
    otlp_request_factory,
    otlp_format: str,
):
    """Test pagination of log queries."""
    workspace = fileset.workspace
    fileset_name = fileset.name

    # Upload 25 logs
    messages = [f"Log message {i}" for i in range(25)]
    request = otlp_request_factory(
        job=f"pagination-test-job-{otlp_format}",
        messages=messages,
    )
    response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs",
        **request.post_kwargs(),
    )
    assert response.status_code == 200

    # Get first page (10 items)
    page1_response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs/query",
        json={
            "filters": {"job": f"pagination-test-job-{otlp_format}"},
            "limit": 10,
        },
    )
    assert page1_response.status_code == 200
    page1 = page1_response.json()
    assert len(page1["data"]) == 10
    assert page1["total"] == 25
    assert page1["next_page"] is not None
    assert page1["prev_page"] is None

    # Get second page
    page2_response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs/query",
        json={
            "filters": {"job": f"pagination-test-job-{otlp_format}"},
            "limit": 10,
            "page_cursor": page1["next_page"],
        },
    )
    assert page2_response.status_code == 200
    page2 = page2_response.json()
    assert len(page2["data"]) == 10
    assert page2["next_page"] is not None
    assert page2["prev_page"] is not None

    # Get third page (remaining 5)
    page3_response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs/query",
        json={
            "filters": {"job": f"pagination-test-job-{otlp_format}"},
            "limit": 10,
            "page_cursor": page2["next_page"],
        },
    )
    assert page3_response.status_code == 200
    page3 = page3_response.json()
    assert len(page3["data"]) == 5
    assert page3["next_page"] is None


def test_multiple_batches_same_partition(
    client: httpx.Client,
    fileset: FilesetOutput,
    otlp_request_factory,
    otlp_format: str,
):
    """Test uploading multiple batches to the same partition."""
    workspace = fileset.workspace
    fileset_name = fileset.name
    job_name = f"batch-test-job-{otlp_format}"

    # Upload first batch
    request1 = otlp_request_factory(
        job=job_name,
        messages=["Batch 1 message"],
        base_timestamp_ns=1704110400000000000,
    )
    response1 = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs",
        **request1.post_kwargs(),
    )
    assert response1.status_code == 200

    # Upload second batch to same job
    request2 = otlp_request_factory(
        job=job_name,
        messages=["Batch 2 message"],
        base_timestamp_ns=1704110410000000000,  # 10 seconds later
    )
    response2 = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs",
        **request2.post_kwargs(),
    )
    assert response2.status_code == 200

    # Query should return both
    query_response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs/query",
        json={"filters": {"job": job_name}},
    )
    assert query_response.status_code == 200
    result = query_response.json()
    assert result["total"] == 2
    messages = {log["message"] for log in result["data"]}
    assert messages == {"Batch 1 message", "Batch 2 message"}


# =============================================================================
# Non-parametrized tests (format-agnostic or error cases)
# =============================================================================


def test_query_empty_fileset(
    client: httpx.Client,
    fileset: FilesetOutput,
):
    """Test querying a fileset with no logs returns empty result."""
    workspace = fileset.workspace
    fileset_name = fileset.name

    response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs/query",
        json={"filters": {"job": "nonexistent-job"}},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["total"] == 0
    assert result["data"] == []
    assert result["next_page"] is None
    assert result["prev_page"] is None


def test_query_logs_accepts_tail(
    client: httpx.Client,
    fileset: FilesetOutput,
):
    """The Files log query endpoint accepts tail without limit."""
    workspace = fileset.workspace
    fileset_name = fileset.name

    response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs/query",
        json={"filters": {"job": "tail-test-job"}, "tail": 3},
    )

    assert response.status_code == 200


def test_query_logs_rejects_tail_with_limit(
    client: httpx.Client,
    fileset: FilesetOutput,
):
    """Tail controls window size and cannot be combined with limit."""
    workspace = fileset.workspace
    fileset_name = fileset.name

    response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs/query",
        json={"filters": {"job": "tail-test-job"}, "tail": 3, "limit": 3},
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"] == "tail cannot be combined with limit; tail controls the returned log window size."
    )


def test_query_logs_rejects_tail_with_page_cursor(
    client: httpx.Client,
    fileset: FilesetOutput,
):
    """Tail starts at the end, so previous-page cursors must be requested without tail."""
    workspace = fileset.workspace
    fileset_name = fileset.name

    response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs/query",
        json={"filters": {"job": "tail-test-job"}, "tail": 3, "page_cursor": "abc"},
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "tail cannot be combined with page_cursor; pass the returned prev_page as page_cursor without tail."
    )


def test_query_logs_rejects_invalid_page_cursor(
    client: httpx.Client,
    fileset: FilesetOutput,
):
    """Malformed pagination cursors are client validation errors."""
    workspace = fileset.workspace
    fileset_name = fileset.name

    response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs/query",
        json={"filters": {"job": "tail-test-job"}, "page_cursor": "garbage"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid page cursor"


def test_upload_logs_missing_attributes_partial_success(
    client: httpx.Client,
    fileset: FilesetOutput,
):
    """Test that logs with missing required attributes are rejected (JSON format)."""
    workspace = fileset.workspace
    fileset_name = fileset.name

    # Request with one valid resource batch and one invalid (missing job_task)
    request_data = {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {"key": "job", "value": {"stringValue": "valid-job"}},
                        {"key": "job_attempt", "value": {"stringValue": "attempt-1"}},
                        {"key": "job_step", "value": {"stringValue": "step-1"}},
                        {"key": "job_task", "value": {"stringValue": "task-1"}},
                    ]
                },
                "scopeLogs": [
                    {
                        "logRecords": [
                            {
                                "timeUnixNano": "1704110400000000000",
                                "body": {"stringValue": "Valid log"},
                            }
                        ]
                    }
                ],
            },
            {
                "resource": {
                    "attributes": [
                        {"key": "job", "value": {"stringValue": "invalid-job"}},
                        # Missing job_attempt, job_step, job_task
                    ]
                },
                "scopeLogs": [
                    {
                        "logRecords": [
                            {
                                "timeUnixNano": "1704110405000000000",
                                "body": {"stringValue": "Invalid log"},
                            }
                        ]
                    }
                ],
            },
        ]
    }

    response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs",
        json=request_data,
    )
    assert response.status_code == 200
    result = response.json()

    # Should have partial success (1 rejected)
    assert result["partialSuccess"] is not None
    assert result["partialSuccess"]["rejectedLogRecords"] == 1

    # Valid log should be queryable
    query_response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs/query",
        json={"filters": {"job": "valid-job"}},
    )
    assert query_response.status_code == 200
    query_result = query_response.json()
    assert query_result["total"] == 1
    assert query_result["data"][0]["message"] == "Valid log"


def test_upload_logs_invalid_json(
    client: httpx.Client,
    fileset: FilesetOutput,
):
    """Test that invalid JSON returns 400 error."""
    workspace = fileset.workspace
    fileset_name = fileset.name

    response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs",
        content="not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert "Invalid JSON format" in response.json()["detail"]


def test_upload_logs_invalid_protobuf(
    client: httpx.Client,
    fileset: FilesetOutput,
):
    """Test that invalid protobuf returns 400 error."""
    workspace = fileset.workspace
    fileset_name = fileset.name

    response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs",
        content=b"not valid protobuf",
        headers={"Content-Type": "application/x-protobuf"},
    )
    assert response.status_code == 400
    assert "Invalid protobuf format" in response.json()["detail"]


def test_upload_logs_nonexistent_fileset(
    client: httpx.Client,
):
    """Test that uploading to nonexistent fileset returns 404."""
    response = client.post(
        "/apis/files/v2/workspaces/nonexistent-workspace/filesets/nonexistent/otlp/v1/logs",
        json={
            "resourceLogs": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "job", "value": {"stringValue": "test"}},
                            {"key": "job_attempt", "value": {"stringValue": "1"}},
                            {"key": "job_step", "value": {"stringValue": "1"}},
                            {"key": "job_task", "value": {"stringValue": "1"}},
                        ]
                    },
                    "scopeLogs": [
                        {
                            "logRecords": [
                                {
                                    "timeUnixNano": "1704110400000000000",
                                    "body": {"stringValue": "Test"},
                                }
                            ]
                        }
                    ],
                }
            ]
        },
    )
    assert response.status_code == 404


def test_query_logs_invalid_filter_key_returns_400(
    client: httpx.Client,
    fileset: FilesetOutput,
):
    """Test invalid filter key is rejected with a 400 response."""
    workspace = fileset.workspace
    fileset_name = fileset.name

    response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs/query",
        json={"filters": {"not_a_real_column": "x"}},
    )

    assert response.status_code == 400
    assert "Invalid filter" in response.json()["detail"]


def test_query_logs_invalid_partition_value_does_not_leak_internal_details(
    client: httpx.Client,
    fileset: FilesetOutput,
):
    """Test invalid partition filter input does not leak internal details."""
    workspace = fileset.workspace
    fileset_name = fileset.name

    response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs/query",
        json={"filters": {"job": "bad'; SELECT 1; --"}},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Invalid filter" in detail
    assert "duckdb" not in detail.lower()
    assert "parser" not in detail.lower()


def test_query_logs_invalid_partition_value_returns_400(
    client: httpx.Client,
    fileset: FilesetOutput,
):
    """Test unsafe partition filter values are rejected before query execution."""
    workspace = fileset.workspace
    fileset_name = fileset.name

    response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs/query",
        json={"filters": {"job": "bad'; SELECT 1; --"}, "limit": 10},
    )

    assert response.status_code == 400
    assert "Invalid filter" in response.json()["detail"]


def test_query_logs_log_message_allows_apostrophe(
    client: httpx.Client,
    fileset: FilesetOutput,
):
    """Test log_message filter remains usable for normal text with apostrophes."""
    workspace = fileset.workspace
    fileset_name = fileset.name

    upload_response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs",
        json={
            "resourceLogs": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "job", "value": {"stringValue": "apostrophe-job"}},
                            {"key": "job_attempt", "value": {"stringValue": "attempt-1"}},
                            {"key": "job_step", "value": {"stringValue": "step-1"}},
                            {"key": "job_task", "value": {"stringValue": "task-1"}},
                        ]
                    },
                    "scopeLogs": [
                        {
                            "logRecords": [
                                {
                                    "timeUnixNano": "1704110500000000000",
                                    "body": {"stringValue": "I'm testing apostrophes"},
                                }
                            ]
                        }
                    ],
                }
            ]
        },
    )
    assert upload_response.status_code == 200

    query_response = client.post(
        f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset_name}/otlp/v1/logs/query",
        json={"filters": {"log_message": "I'm testing apostrophes"}, "limit": 10},
    )
    assert query_response.status_code == 200
    result = query_response.json()
    assert result["total"] == 1
    assert result["data"][0]["message"] == "I'm testing apostrophes"
