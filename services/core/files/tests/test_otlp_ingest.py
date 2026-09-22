# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for OTLP log ingestion endpoints.

Note: The OTLP log ingestion implementation expects job-related attributes
(job, job_attempt, job_step, job_task) to be provided at the resource level
rather than individual log record level. Main fixtures and key tests have been
updated to reflect this. Some older test cases may still use the previous pattern
but are being gradually updated.
"""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nhx.common.auth import AuthClient, get_auth_client
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from nhx.common.entities.client import EntityClient, EntityNotFoundError
from nhx.common.service.dependencies import get_entity_client, get_sdk_client
from nhx.core.files.api.v2.otlp.endpoints import router
from nhx.core.files.app.log_storage import LogEntry, LogStorage, dep_log_storage
from nhx.core.files.entities import Fileset
from opentelemetry.proto.collector.logs.v1 import logs_service_pb2
from opentelemetry.proto.logs.v1 import logs_pb2

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_entity_client():
    """Create a mock EntityClient for testing."""
    mock_client = AsyncMock(spec=EntityClient)
    return mock_client


@pytest.fixture
def mock_log_storage():
    """Create a mock LogStorage for testing."""
    mock_storage = AsyncMock(spec=LogStorage)
    mock_storage.insert_logs = AsyncMock(return_value=1)
    return mock_storage


@pytest.fixture
def mock_fileset():
    """Create a mock Fileset for testing."""
    return Fileset(
        name="test-logs-fileset",
        workspace="test-workspace",
        storage={"type": "local", "path": "/tmp/test-logs"},
        purpose="generic",
    )


@pytest.fixture
def mock_storage():
    """Create a mock storage implementation."""
    mock_storage = AsyncMock()
    mock_storage.upload = AsyncMock()
    return mock_storage


@pytest.fixture
def mock_sdk():
    """Create a mock SDK for testing."""
    mock_sdk = AsyncMock()
    return mock_sdk


@pytest.fixture
def override_auth_client():
    """Shared auth dependency override for endpoint tests."""
    return AuthClient(
        principal=Principal(id="test@example.com"),
        config=AuthConfig(enabled=False),
    )


@pytest.fixture
def test_client(
    mock_entity_client,
    mock_log_storage,
    mock_fileset,
    mock_storage,
    mock_sdk,
    override_auth_client,
):
    """Create a test client with mocked dependencies."""

    async def override_entity_client():
        # Mock get to return our test fileset (used by get_fileset helper)
        mock_entity_client.get = AsyncMock(return_value=mock_fileset)
        return mock_entity_client

    def override_log_storage():
        return mock_log_storage

    def override_sdk_client():
        return mock_sdk

    # Patch storage_impl_factory and resolve_storage_secrets before creating the app
    with (
        patch(
            "nhx.core.files.api.v2.otlp.endpoints.storage_impl_factory",
            return_value=mock_storage,
        ),
        patch(
            "nhx.core.files.api.v2.otlp.endpoints.resolve_storage_secrets_for_user",
            return_value={},
        ),
    ):
        app = FastAPI()
        app.dependency_overrides[get_entity_client] = override_entity_client
        app.dependency_overrides[dep_log_storage] = override_log_storage
        app.dependency_overrides[get_auth_client] = lambda: override_auth_client
        app.dependency_overrides[get_sdk_client] = override_sdk_client
        app.include_router(router)

        with TestClient(app) as client:
            yield client


@pytest.fixture
def valid_otlp_request():
    """Create a valid OTLP log request payload with resource-level attributes."""
    return {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {"key": "job", "value": {"stringValue": "test-job-123"}},
                        {"key": "job_attempt", "value": {"stringValue": "attempt-1"}},
                        {"key": "job_step", "value": {"stringValue": "step-1"}},
                        {"key": "job_task", "value": {"stringValue": "task-1"}},
                    ]
                },
                "scopeLogs": [
                    {
                        "scope": {"name": "test-scope"},
                        "logRecords": [
                            {
                                "timeUnixNano": "1704110400000000000",  # 2024-01-01 12:00:00
                                "severityNumber": 9,
                                "severityText": "INFO",
                                "body": {"stringValue": "Test log message"},
                            }
                        ],
                    }
                ],
            }
        ]
    }


@pytest.fixture
def otlp_request_multiple_logs():
    """Create an OTLP request with multiple log records and resource-level attributes."""
    return {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {"key": "job", "value": {"stringValue": "test-job"}},
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
                                "body": {"stringValue": "First log"},
                            },
                            {
                                "timeUnixNano": "1704110405000000000",
                                "body": {"stringValue": "Second log"},
                            },
                            {
                                "timeUnixNano": "1704110410000000000",
                                "body": {"stringValue": "Third log"},
                            },
                        ]
                    }
                ],
            }
        ]
    }


# ============================================================================
# Success Cases
# ============================================================================


async def test_upload_otlp_logs_success(test_client, mock_log_storage, valid_otlp_request):
    """Test successful upload of OTLP logs."""
    # Configure mock to return successful insert count
    mock_log_storage.insert_logs.return_value = 1

    # Make request
    response = test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        json=valid_otlp_request,
    )

    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data.get("partialSuccess") is None  # Full success

    # Verify insert_logs was called
    mock_log_storage.insert_logs.assert_called_once()
    call_args = mock_log_storage.insert_logs.call_args
    log_entries = call_args.kwargs["log_entries"]

    # Verify log entry structure
    assert len(log_entries) == 1
    log_entry = log_entries[0]
    assert isinstance(log_entry, LogEntry)
    assert log_entry.workspace == "test-workspace"
    assert log_entry.job == "test-job-123"
    assert log_entry.job_attempt == "attempt-1"
    assert log_entry.job_step == "step-1"
    assert log_entry.job_task == "task-1"
    assert log_entry.log_message == "Test log message"
    # Timestamp is converted from nanoseconds since epoch
    assert isinstance(log_entry.timestamp, datetime)


async def test_upload_multiple_logs(test_client, mock_log_storage, otlp_request_multiple_logs):
    """Test uploading multiple log records in a single request."""
    mock_log_storage.insert_logs.return_value = 3

    response = test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        json=otlp_request_multiple_logs,
    )

    assert response.status_code == 200
    data = response.json()
    assert data.get("partialSuccess") is None

    # Verify all logs were processed
    mock_log_storage.insert_logs.assert_called_once()
    call_args = mock_log_storage.insert_logs.call_args
    log_entries = call_args.kwargs["log_entries"]
    assert len(log_entries) == 3


async def test_upload_logs_with_different_body_types(test_client, mock_log_storage):
    """Test that different AnyValue body types are handled correctly."""
    request_data = {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {"key": "job", "value": {"stringValue": "test-job"}},
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
                                "body": {"intValue": 42},  # Integer value
                            },
                            {
                                "timeUnixNano": "1704110405000000000",
                                "body": {"doubleValue": 3.14},  # Double value
                            },
                            {
                                "timeUnixNano": "1704110410000000000",
                                "body": {"boolValue": True},  # Boolean value
                            },
                        ]
                    }
                ],
            }
        ]
    }

    mock_log_storage.insert_logs.return_value = 3

    response = test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        json=request_data,
    )

    assert response.status_code == 200

    # Verify log messages were converted to strings
    call_args = mock_log_storage.insert_logs.call_args
    log_entries = call_args.kwargs["log_entries"]
    assert len(log_entries) == 3
    assert log_entries[0].log_message == "42"
    assert log_entries[1].log_message == "3.14"
    assert log_entries[2].log_message == "True"


async def test_upload_logs_empty_body(test_client, mock_log_storage):
    """Test uploading logs with no body (empty message)."""
    request_data = {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {"key": "job", "value": {"stringValue": "test-job"}},
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
                                # No body field
                            }
                        ]
                    }
                ],
            }
        ]
    }

    mock_log_storage.insert_logs.return_value = 1

    response = test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        json=request_data,
    )

    assert response.status_code == 200

    # Verify log was created with empty message
    call_args = mock_log_storage.insert_logs.call_args
    log_entries = call_args.kwargs["log_entries"]
    assert len(log_entries) == 1
    assert log_entries[0].log_message == ""


# ============================================================================
# Resource-Level Attribute Tests
# ============================================================================


async def test_resource_attributes_multiple_batches(test_client, mock_log_storage):
    """Test handling multiple resource batches with different job attributes."""
    request_data = {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {"key": "job", "value": {"stringValue": "job-1"}},
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
                                "body": {"stringValue": "Log from job 1"},
                            }
                        ]
                    }
                ],
            },
            {
                "resource": {
                    "attributes": [
                        {"key": "job", "value": {"stringValue": "job-2"}},
                        {"key": "job_attempt", "value": {"stringValue": "attempt-1"}},
                        {"key": "job_step", "value": {"stringValue": "step-2"}},
                        {"key": "job_task", "value": {"stringValue": "task-2"}},
                    ]
                },
                "scopeLogs": [
                    {
                        "logRecords": [
                            {
                                "timeUnixNano": "1704110405000000000",
                                "body": {"stringValue": "Log from job 2"},
                            }
                        ]
                    }
                ],
            },
        ]
    }

    mock_log_storage.insert_logs.return_value = 2

    response = test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        json=request_data,
    )

    assert response.status_code == 200

    call_args = mock_log_storage.insert_logs.call_args
    log_entries = call_args.kwargs["log_entries"]
    assert len(log_entries) == 2

    # Verify each log has its corresponding resource attributes
    assert log_entries[0].job == "job-1"
    assert log_entries[0].job_task == "task-1"
    assert log_entries[1].job == "job-2"
    assert log_entries[1].job_task == "task-2"


async def test_resource_attributes_with_additional_metadata(test_client, mock_log_storage):
    """Test that non-job resource attributes coexist with job attributes."""
    request_data = {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "service.name",
                            "value": {"stringValue": "job-executor"},
                        },
                        {"key": "service.version", "value": {"stringValue": "1.0.0"}},
                        {"key": "job", "value": {"stringValue": "pipeline-job"}},
                        {"key": "job_attempt", "value": {"stringValue": "attempt-1"}},
                        {"key": "job_step", "value": {"stringValue": "transform"}},
                        {"key": "job_task", "value": {"stringValue": "task-0"}},
                    ]
                },
                "scopeLogs": [
                    {
                        "logRecords": [
                            {
                                "timeUnixNano": "1704110400000000000",
                                "body": {"stringValue": "Processing data"},
                            }
                        ]
                    }
                ],
            }
        ]
    }

    mock_log_storage.insert_logs.return_value = 1

    response = test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        json=request_data,
    )

    assert response.status_code == 200

    call_args = mock_log_storage.insert_logs.call_args
    log_entries = call_args.kwargs["log_entries"]
    assert len(log_entries) == 1
    assert log_entries[0].job == "pipeline-job"


# ============================================================================
# Partial Success Cases
# ============================================================================


async def test_upload_logs_missing_required_attributes(test_client, mock_log_storage):
    """Test that resource batches missing required attributes are rejected."""
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
                        {"key": "job_attempt", "value": {"stringValue": "attempt-1"}},
                        # Missing job_step and job_task
                    ]
                },
                "scopeLogs": [
                    {
                        "logRecords": [
                            {
                                "timeUnixNano": "1704110405000000000",
                                "body": {"stringValue": "Invalid log 1"},
                            },
                            {
                                "timeUnixNano": "1704110410000000000",
                                "body": {"stringValue": "Invalid log 2"},
                            },
                        ]
                    }
                ],
            },
        ]
    }

    mock_log_storage.insert_logs.return_value = 1

    response = test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        json=request_data,
    )

    assert response.status_code == 200
    data = response.json()

    # Verify partial success response
    assert data["partialSuccess"] is not None
    assert data["partialSuccess"]["rejectedLogRecords"] == 2
    assert "Rejected 2 log records" in data["partialSuccess"]["errorMessage"]

    # Verify only valid log was inserted
    call_args = mock_log_storage.insert_logs.call_args
    log_entries = call_args.kwargs["log_entries"]
    assert len(log_entries) == 1
    assert log_entries[0].log_message == "Valid log"


async def test_upload_logs_all_rejected(test_client, mock_log_storage):
    """Test when all log records are rejected."""
    request_data = {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {"key": "job", "value": {"stringValue": "incomplete-job"}},
                        # Missing job_attempt, step, and task
                    ]
                },
                "scopeLogs": [
                    {
                        "logRecords": [
                            {
                                "timeUnixNano": "1704110400000000000",
                                "body": {"stringValue": "Missing attributes"},
                            },
                            {
                                "timeUnixNano": "1704110405000000000",
                                "body": {"stringValue": "Also missing attributes"},
                            },
                        ]
                    }
                ],
            }
        ]
    }

    response = test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        json=request_data,
    )

    assert response.status_code == 200
    data = response.json()

    # Verify all logs rejected
    assert data["partialSuccess"]["rejectedLogRecords"] == 2

    # Verify insert_logs was not called (no valid logs)
    mock_log_storage.insert_logs.assert_not_called()


async def test_upload_logs_processing_error(test_client, mock_log_storage):
    """Test handling of processing errors during log parsing."""
    request_data = {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {"key": "job", "value": {"stringValue": "test-job"}},
                        {"key": "job_attempt", "value": {"stringValue": "attempt-1"}},
                        {"key": "job_step", "value": {"stringValue": "step-1"}},
                        {"key": "job_task", "value": {"stringValue": "task-1"}},
                    ]
                },
                "scopeLogs": [
                    {
                        "logRecords": [
                            {
                                "timeUnixNano": "invalid_timestamp",  # Invalid timestamp
                                "body": {"stringValue": "Test log"},
                            }
                        ]
                    }
                ],
            }
        ]
    }

    response = test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        json=request_data,
    )

    assert response.status_code == 200
    data = response.json()

    # Verify processing error resulted in rejection
    assert data["partialSuccess"]["rejectedLogRecords"] == 1


# ============================================================================
# Error Cases
# ============================================================================


async def test_upload_fileset_not_found(
    mock_entity_client, mock_log_storage, mock_storage, mock_sdk, override_auth_client
):
    """Test upload when fileset doesn't exist."""
    # Mock get to raise EntityNotFoundError (used by get_fileset helper)
    mock_entity_client.get = AsyncMock(side_effect=EntityNotFoundError("Fileset not found"))

    async def override_entity_client():
        return mock_entity_client

    def override_log_storage():
        return mock_log_storage

    def override_sdk_client():
        return mock_sdk

    # Create a new test client with the mocked entity client
    with patch(
        "nhx.core.files.api.v2.otlp.endpoints.storage_impl_factory",
        return_value=mock_storage,
    ):
        app = FastAPI()
        app.dependency_overrides[get_entity_client] = override_entity_client
        app.dependency_overrides[dep_log_storage] = override_log_storage
        app.dependency_overrides[get_auth_client] = lambda: override_auth_client
        app.dependency_overrides[get_sdk_client] = override_sdk_client
        app.include_router(router)

        with TestClient(app) as client:
            request_data = {
                "resourceLogs": [
                    {
                        "scopeLogs": [
                            {
                                "logRecords": [
                                    {
                                        "timeUnixNano": "1704110400000000000",
                                        "body": {"stringValue": "Test"},
                                        "attributes": [
                                            {
                                                "key": "job",
                                                "value": {"stringValue": "test-job"},
                                            },
                                            {
                                                "key": "job_attempt",
                                                "value": {"stringValue": "attempt-1"},
                                            },
                                            {
                                                "key": "job_step",
                                                "value": {"stringValue": "step-1"},
                                            },
                                            {
                                                "key": "job_task",
                                                "value": {"stringValue": "task-1"},
                                            },
                                        ],
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }

            response = client.post(
                "/v2/workspaces/test-workspace/filesets/nonexistent/otlp/v1/logs",
                json=request_data,
            )

            # Should return 404 for fileset not found
            assert response.status_code == 404


async def test_upload_logs_empty_request(test_client):
    """Test uploading with empty resourceLogs."""
    request_data = {"resourceLogs": []}

    response = test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        json=request_data,
    )

    # Should return 200 with no logs processed
    assert response.status_code == 200
    data = response.json()
    assert data.get("partialSuccess") is None


async def test_upload_logs_invalid_json(test_client):
    """Test upload with invalid JSON structure."""
    response = test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        data="invalid json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400  # Bad Request
    assert "Invalid JSON format" in response.json()["detail"]


async def test_upload_logs_database_error(test_client, mock_log_storage, valid_otlp_request):
    """Test handling of database errors during insertion."""
    # Mock insert_logs to raise an exception
    mock_log_storage.insert_logs.side_effect = Exception("Database connection error")

    response = test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        json=valid_otlp_request,
    )

    # Should return 500 for internal errors
    assert response.status_code == 500
    data = response.json()
    assert "Error ingesting logs" in data["detail"]


# ============================================================================
# Attribute Extraction Tests
# ============================================================================


async def test_attribute_extraction_with_different_value_types(test_client, mock_log_storage):
    """Test that attribute values are correctly extracted from different AnyValue types."""
    request_data = {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "job",
                            "value": {"stringValue": "test-job"},
                        },
                        {
                            "key": "job_attempt",
                            "value": {"intValue": 1},
                        },  # Integer value type
                        {
                            "key": "job_step",
                            "value": {"stringValue": "step-1"},
                        },
                        {
                            "key": "job_task",
                            "value": {"stringValue": "task-1"},
                        },
                    ]
                },
                "scopeLogs": [
                    {
                        "logRecords": [
                            {
                                "timeUnixNano": "1704110400000000000",
                                "body": {"stringValue": "Test log"},
                            }
                        ]
                    }
                ],
            }
        ]
    }

    mock_log_storage.insert_logs.return_value = 1

    response = test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        json=request_data,
    )

    assert response.status_code == 200

    # Verify attribute was converted to string
    call_args = mock_log_storage.insert_logs.call_args
    log_entries = call_args.kwargs["log_entries"]
    assert log_entries[0].job_attempt == "1"


# ============================================================================
# Integration-style Tests
# ============================================================================


async def test_upload_logs_realistic_scenario(test_client, mock_log_storage):
    """Test a realistic scenario with multiple batches and scopes."""
    request_data = {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "service.name",
                            "value": {"stringValue": "job-executor"},
                        },
                        {"key": "service.version", "value": {"stringValue": "1.0.0"}},
                        {"key": "job", "value": {"stringValue": "data-pipeline-001"}},
                        {"key": "job_attempt", "value": {"stringValue": "attempt-1"}},
                        {"key": "job_step", "value": {"stringValue": "preprocessing"}},
                        {"key": "job_task", "value": {"stringValue": "task-0"}},
                    ]
                },
                "scopeLogs": [
                    {
                        "scope": {"name": "main", "version": "1.0"},
                        "logRecords": [
                            {
                                "timeUnixNano": "1704110400000000000",
                                "severityNumber": 9,
                                "severityText": "INFO",
                                "body": {"stringValue": "Job started"},
                                "traceId": "0123456789abcdef0123456789abcdef",
                                "spanId": "0123456789abcdef",
                            },
                            {
                                "timeUnixNano": "1704110405000000000",
                                "severityNumber": 13,
                                "severityText": "WARN",
                                "body": {"stringValue": "Memory usage high"},
                                "attributes": [
                                    {
                                        "key": "memory.used",
                                        "value": {"intValue": 8589934592},
                                    },
                                ],
                            },
                        ],
                    },
                    {
                        "scope": {"name": "monitoring", "version": "1.0"},
                        "logRecords": [
                            {
                                "timeUnixNano": "1704110410000000000",
                                "severityNumber": 9,
                                "severityText": "INFO",
                                "body": {"stringValue": "Job completed successfully"},
                                "attributes": [
                                    {
                                        "key": "duration.ms",
                                        "value": {"intValue": 10000},
                                    },
                                ],
                            }
                        ],
                    },
                ],
            }
        ]
    }

    mock_log_storage.insert_logs.return_value = 3

    response = test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        json=request_data,
    )

    assert response.status_code == 200
    data = response.json()
    assert data.get("partialSuccess") is None

    # Verify all logs from both scopes were processed
    call_args = mock_log_storage.insert_logs.call_args
    log_entries = call_args.kwargs["log_entries"]
    assert len(log_entries) == 3
    assert log_entries[0].log_message == "Job started"
    assert log_entries[1].log_message == "Memory usage high"
    assert log_entries[2].log_message == "Job completed successfully"


# ============================================================================
# Benchmark Tests
# ============================================================================


@pytest.mark.skip("To be replaced by pytest-benchmark")
async def test_benchmark_single_log_ingestion(test_client, mock_log_storage):
    """Benchmark ingestion of a single log record."""
    import time

    request_data = {
        "resourceLogs": [
            {
                "scopeLogs": [
                    {
                        "logRecords": [
                            {
                                "timeUnixNano": "1704110400000000000",
                                "body": {"stringValue": "Benchmark log message"},
                                "attributes": [
                                    {
                                        "key": "job",
                                        "value": {"stringValue": "bench-job"},
                                    },
                                    {
                                        "key": "job_attempt",
                                        "value": {"stringValue": "attempt-1"},
                                    },
                                    {
                                        "key": "job_step",
                                        "value": {"stringValue": "step-1"},
                                    },
                                    {
                                        "key": "job_task",
                                        "value": {"stringValue": "task-1"},
                                    },
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }

    mock_log_storage.insert_logs.return_value = 1

    # Warm up
    test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        json=request_data,
    )

    # Benchmark 100 requests
    iterations = 100
    start_time = time.perf_counter()

    for _ in range(iterations):
        response = test_client.post(
            "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
            json=request_data,
        )
        assert response.status_code == 200

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    # Calculate metrics
    avg_latency_ms = (elapsed / iterations) * 1000
    requests_per_sec = iterations / elapsed

    print(f"\n{'=' * 60}")
    print("Single Log Ingestion Benchmark")
    print(f"{'=' * 60}")
    print(f"Iterations: {iterations}")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Avg latency: {avg_latency_ms:.2f}ms")
    print(f"Throughput: {requests_per_sec:.2f} req/s")
    print(f"{'=' * 60}\n")

    # Basic performance assertions (adjust thresholds as needed)
    assert avg_latency_ms < 100, f"Average latency too high: {avg_latency_ms:.2f}ms"


@pytest.mark.skip("To be replaced by pytest-benchmark")
async def test_benchmark_batch_log_ingestion(test_client, mock_log_storage):
    """Benchmark ingestion of batches with varying sizes."""
    import time

    def create_batch_request(log_count: int):
        """Create a request with specified number of logs."""
        log_records = []
        for i in range(log_count):
            log_records.append(
                {
                    "timeUnixNano": str(1704110400000000000 + i * 1000000),
                    "body": {"stringValue": f"Log message {i}"},
                    "attributes": [
                        {"key": "job", "value": {"stringValue": "bench-job"}},
                        {"key": "job_attempt", "value": {"stringValue": "attempt-1"}},
                        {"key": "job_step", "value": {"stringValue": "step-1"}},
                        {"key": "job_task", "value": {"stringValue": "task-1"}},
                    ],
                }
            )

        return {"resourceLogs": [{"scopeLogs": [{"logRecords": log_records}]}]}

    batch_sizes = [10, 50, 100, 500]
    iterations = 10

    print(f"\n{'=' * 60}")
    print("Batch Log Ingestion Benchmark")
    print(f"{'=' * 60}")

    for batch_size in batch_sizes:
        mock_log_storage.insert_logs.return_value = batch_size
        request_data = create_batch_request(batch_size)

        # Warm up
        test_client.post(
            "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
            json=request_data,
        )

        start_time = time.perf_counter()

        for _ in range(iterations):
            response = test_client.post(
                "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
                json=request_data,
            )
            assert response.status_code == 200

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        avg_latency_ms = (elapsed / iterations) * 1000
        logs_per_sec = (batch_size * iterations) / elapsed

        print(f"Batch size: {batch_size:4d} logs")
        print(f"  Avg latency: {avg_latency_ms:7.2f}ms")
        print(f"  Throughput: {logs_per_sec:8.0f} logs/s")

    print(f"{'=' * 60}\n")


@pytest.mark.skip("To be replaced by pytest-benchmark")
async def test_benchmark_concurrent_requests(mock_entity_client, mock_log_storage, mock_fileset, mock_storage):
    """Benchmark concurrent OTLP log ingestion requests."""
    import asyncio
    import time

    from httpx import ASGITransport, AsyncClient

    async def override_entity_client():
        mock_entity_client.get = AsyncMock(return_value=mock_fileset)
        return mock_entity_client

    def override_log_storage():
        return mock_log_storage

    mock_log_storage.insert_logs.return_value = 10

    with (
        patch(
            "nhx.core.files.api.v2.otlp.endpoints.storage_impl_factory",
            return_value=mock_storage,
        ),
        patch(
            "nhx.core.files.api.v2.otlp.endpoints.resolve_storage_secrets",
            return_value={},
        ),
    ):
        app = FastAPI()
        app.dependency_overrides[get_entity_client] = override_entity_client
        app.dependency_overrides[dep_log_storage] = override_log_storage
        app.include_router(router)

        request_data = {
            "resourceLogs": [
                {
                    "scopeLogs": [
                        {
                            "logRecords": [
                                {
                                    "timeUnixNano": str(1704110400000000000 + i * 1000000),
                                    "body": {"stringValue": f"Concurrent log {i}"},
                                    "attributes": [
                                        {
                                            "key": "job",
                                            "value": {"stringValue": "bench-job"},
                                        },
                                        {
                                            "key": "job_attempt",
                                            "value": {"stringValue": "attempt-1"},
                                        },
                                        {
                                            "key": "job_step",
                                            "value": {"stringValue": "step-1"},
                                        },
                                        {
                                            "key": "job_task",
                                            "value": {"stringValue": "task-1"},
                                        },
                                    ],
                                }
                                for i in range(10)
                            ]
                        }
                    ]
                }
            ]
        }

        async def make_request(client, request_num):
            """Make a single request and return timing."""
            start = time.perf_counter()
            response = await client.post(
                "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
                json=request_data,
            )
            elapsed = time.perf_counter() - start
            return response.status_code, elapsed

        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        requests_per_level = 50

        print(f"\n{'=' * 60}")
        print("Concurrent Request Benchmark")
        print(f"{'=' * 60}")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            for concurrency in concurrency_levels:
                start_time = time.perf_counter()

                # Run requests in batches to control concurrency
                all_results = []
                for batch_start in range(0, requests_per_level, concurrency):
                    batch_size = min(concurrency, requests_per_level - batch_start)
                    tasks = [make_request(client, batch_start + i) for i in range(batch_size)]
                    results = await asyncio.gather(*tasks)
                    all_results.extend(results)

                end_time = time.perf_counter()
                elapsed = end_time - start_time

                # Calculate metrics
                status_codes = [r[0] for r in all_results]
                latencies = [r[1] * 1000 for r in all_results]  # Convert to ms
                success_count = sum(1 for code in status_codes if code == 200)

                avg_latency = sum(latencies) / len(latencies)
                p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
                throughput = requests_per_level / elapsed

                print(f"Concurrency: {concurrency:2d}")
                print(f"  Success rate: {success_count}/{requests_per_level}")
                print(f"  Avg latency: {avg_latency:7.2f}ms")
                print(f"  P95 latency: {p95_latency:7.2f}ms")
                print(f"  Throughput: {throughput:7.2f} req/s")

        print(f"{'=' * 60}\n")


@pytest.mark.skip("To be replaced by pytest-benchmark")
async def test_benchmark_large_payload(test_client, mock_log_storage):
    """Benchmark ingestion with large payloads (many attributes and complex messages)."""
    import time

    def create_large_log_record(index: int):
        """Create a log record with many attributes."""
        attributes = [
            {"key": "job", "value": {"stringValue": "bench-job"}},
            {"key": "job_attempt", "value": {"stringValue": "attempt-1"}},
            {"key": "job_step", "value": {"stringValue": "step-1"}},
            {"key": "job_task", "value": {"stringValue": "task-1"}},
        ]

        # Add 50 additional custom attributes
        for i in range(50):
            attributes.append(
                {
                    "key": f"custom_attr_{i}",
                    "value": {"stringValue": f"value_{i}_{index}"},
                }
            )

        return {
            "timeUnixNano": str(1704110400000000000 + index * 1000000),
            "severityNumber": 9,
            "severityText": "INFO",
            "body": {"stringValue": f"Large log message {index} " + "x" * 500},  # 500+ char message
            "attributes": attributes,
            "traceId": f"{index:032x}",
            "spanId": f"{index:016x}",
        }

    # Create request with 100 large log records
    log_count = 100
    request_data = {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "service.name",
                            "value": {"stringValue": "benchmark-service"},
                        },
                        {"key": "service.version", "value": {"stringValue": "1.0.0"}},
                    ]
                },
                "scopeLogs": [
                    {
                        "scope": {"name": "benchmark", "version": "1.0"},
                        "logRecords": [create_large_log_record(i) for i in range(log_count)],
                    }
                ],
            }
        ]
    }

    mock_log_storage.insert_logs.return_value = log_count

    # Warm up
    test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        json=request_data,
    )

    iterations = 10
    start_time = time.perf_counter()

    for _ in range(iterations):
        response = test_client.post(
            "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
            json=request_data,
        )
        assert response.status_code == 200

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    avg_latency_ms = (elapsed / iterations) * 1000
    logs_per_sec = (log_count * iterations) / elapsed

    # Calculate payload size
    import json

    payload_size_kb = len(json.dumps(request_data).encode("utf-8")) / 1024

    print(f"\n{'=' * 60}")
    print("Large Payload Benchmark")
    print(f"{'=' * 60}")
    print(f"Logs per request: {log_count}")
    print(f"Payload size: {payload_size_kb:.1f} KB")
    print(f"Iterations: {iterations}")
    print(f"Avg latency: {avg_latency_ms:.2f}ms")
    print(f"Throughput: {logs_per_sec:.0f} logs/s")
    print(f"{'=' * 60}\n")


@pytest.mark.skip("To be replaced by pytest-benchmark")
async def test_benchmark_attribute_extraction(test_client, mock_log_storage):
    """Benchmark attribute extraction performance with various value types."""
    import time

    def create_request_with_mixed_attributes(count: int):
        """Create logs with various attribute types."""
        log_records = []
        for i in range(count):
            log_records.append(
                {
                    "timeUnixNano": str(1704110400000000000 + i * 1000000),
                    "body": {"stringValue": f"Log {i}"},
                    "attributes": [
                        {"key": "job", "value": {"stringValue": "bench-job"}},
                        {"key": "job_attempt", "value": {"intValue": 1}},
                        {"key": "job_step", "value": {"stringValue": "step-1"}},
                        {"key": "job_task", "value": {"stringValue": "task-1"}},
                        {"key": "counter", "value": {"intValue": i}},
                        {"key": "ratio", "value": {"doubleValue": i * 0.1}},
                        {"key": "enabled", "value": {"boolValue": i % 2 == 0}},
                    ],
                }
            )

        return {"resourceLogs": [{"scopeLogs": [{"logRecords": log_records}]}]}

    log_count = 200
    mock_log_storage.insert_logs.return_value = log_count
    request_data = create_request_with_mixed_attributes(log_count)

    iterations = 10
    start_time = time.perf_counter()

    for _ in range(iterations):
        response = test_client.post(
            "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
            json=request_data,
        )
        assert response.status_code == 200

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    avg_latency_ms = (elapsed / iterations) * 1000
    logs_per_sec = (log_count * iterations) / elapsed
    attributes_per_sec = (log_count * 7 * iterations) / elapsed  # 7 attributes per log

    print(f"\n{'=' * 60}")
    print("Attribute Extraction Benchmark")
    print(f"{'=' * 60}")
    print(f"Logs per request: {log_count}")
    print("Attributes per log: 7 (mixed types)")
    print(f"Iterations: {iterations}")
    print(f"Avg latency: {avg_latency_ms:.2f}ms")
    print(f"Throughput: {logs_per_sec:.0f} logs/s")
    print(f"Attribute extraction: {attributes_per_sec:.0f} attrs/s")
    print(f"{'=' * 60}\n")


# ============================================================================
# Protobuf Format Tests
# ============================================================================


async def test_upload_otlp_logs_protobuf_format(test_client, mock_log_storage):
    """Test successful upload of OTLP logs in protobuf format with resource attributes."""
    # Create protobuf request
    proto_request = logs_service_pb2.ExportLogsServiceRequest()
    resource_logs = proto_request.resource_logs.add()

    # Add resource-level attributes
    attr = resource_logs.resource.attributes.add()
    attr.key = "job"
    attr.value.string_value = "test-job-123"

    attr = resource_logs.resource.attributes.add()
    attr.key = "job_attempt"
    attr.value.string_value = "attempt-1"

    attr = resource_logs.resource.attributes.add()
    attr.key = "job_step"
    attr.value.string_value = "step-1"

    attr = resource_logs.resource.attributes.add()
    attr.key = "job_task"
    attr.value.string_value = "task-1"

    scope_logs = resource_logs.scope_logs.add()
    scope_logs.scope.name = "test-scope"

    # Add a log record (without job attributes)
    log_record = scope_logs.log_records.add()
    log_record.time_unix_nano = 1704110400000000000  # 2024-01-01 12:00:00
    log_record.severity_number = logs_pb2.SeverityNumber.SEVERITY_NUMBER_INFO
    log_record.severity_text = "INFO"
    log_record.body.string_value = "Test protobuf log message"

    # Serialize to bytes
    proto_bytes = proto_request.SerializeToString()

    # Configure mock
    mock_log_storage.insert_logs.return_value = 1

    # Make request with protobuf content type
    response = test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        content=proto_bytes,
        headers={"Content-Type": "application/x-protobuf"},
    )

    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data.get("partialSuccess") is None

    # Verify insert_logs was called with correct data
    mock_log_storage.insert_logs.assert_called_once()
    call_args = mock_log_storage.insert_logs.call_args
    log_entries = call_args.kwargs["log_entries"]

    assert len(log_entries) == 1
    log_entry = log_entries[0]
    assert log_entry.workspace == "test-workspace"
    assert log_entry.job == "test-job-123"
    assert log_entry.job_attempt == "attempt-1"
    assert log_entry.job_step == "step-1"
    assert log_entry.job_task == "task-1"
    assert log_entry.log_message == "Test protobuf log message"


async def test_upload_protobuf_multiple_logs(test_client, mock_log_storage):
    """Test uploading multiple log records in protobuf format."""
    # Create protobuf request with multiple logs
    proto_request = logs_service_pb2.ExportLogsServiceRequest()
    resource_logs = proto_request.resource_logs.add()

    # Add resource-level attributes
    for key, value in [
        ("job", "test-job"),
        ("job_attempt", "attempt-1"),
        ("job_step", "step-1"),
        ("job_task", "task-1"),
    ]:
        attr = resource_logs.resource.attributes.add()
        attr.key = key
        attr.value.string_value = value

    scope_logs = resource_logs.scope_logs.add()

    # Add three log records
    for i in range(3):
        log_record = scope_logs.log_records.add()
        log_record.time_unix_nano = 1704110400000000000 + (i * 5000000000)
        log_record.body.string_value = f"Log message {i + 1}"

    proto_bytes = proto_request.SerializeToString()
    mock_log_storage.insert_logs.return_value = 3

    response = test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        content=proto_bytes,
        headers={"Content-Type": "application/x-protobuf"},
    )

    assert response.status_code == 200

    call_args = mock_log_storage.insert_logs.call_args
    log_entries = call_args.kwargs["log_entries"]
    assert len(log_entries) == 3
    assert log_entries[0].log_message == "Log message 1"
    assert log_entries[1].log_message == "Log message 2"
    assert log_entries[2].log_message == "Log message 3"


async def test_upload_protobuf_different_value_types(test_client, mock_log_storage):
    """Test protobuf format handles different AnyValue types correctly."""
    proto_request = logs_service_pb2.ExportLogsServiceRequest()
    resource_logs = proto_request.resource_logs.add()

    # Add resource-level attributes
    for key, value in [
        ("job", "test-job"),
        ("job_attempt", "attempt-1"),
        ("job_step", "step-1"),
        ("job_task", "task-1"),
    ]:
        attr = resource_logs.resource.attributes.add()
        attr.key = key
        attr.value.string_value = value

    scope_logs = resource_logs.scope_logs.add()

    # Log with int value body
    log_record = scope_logs.log_records.add()
    log_record.time_unix_nano = 1704110400000000000
    log_record.body.int_value = 42

    # Log with double value body
    log_record = scope_logs.log_records.add()
    log_record.time_unix_nano = 1704110405000000000
    log_record.body.double_value = 3.14

    # Log with bool value body
    log_record = scope_logs.log_records.add()
    log_record.time_unix_nano = 1704110410000000000
    log_record.body.bool_value = True

    proto_bytes = proto_request.SerializeToString()
    mock_log_storage.insert_logs.return_value = 3

    response = test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        content=proto_bytes,
        headers={"Content-Type": "application/x-protobuf"},
    )

    assert response.status_code == 200

    call_args = mock_log_storage.insert_logs.call_args
    log_entries = call_args.kwargs["log_entries"]
    assert len(log_entries) == 3
    assert log_entries[0].log_message == "42"
    assert log_entries[1].log_message == "3.14"
    assert log_entries[2].log_message == "True"


async def test_upload_protobuf_with_trace_context(test_client, mock_log_storage):
    """Test protobuf format preserves trace and span IDs."""
    proto_request = logs_service_pb2.ExportLogsServiceRequest()
    resource_logs = proto_request.resource_logs.add()

    # Add resource-level attributes
    for key, value in [
        ("job", "test-job"),
        ("job_attempt", "attempt-1"),
        ("job_step", "step-1"),
        ("job_task", "task-1"),
    ]:
        attr = resource_logs.resource.attributes.add()
        attr.key = key
        attr.value.string_value = value

    scope_logs = resource_logs.scope_logs.add()

    log_record = scope_logs.log_records.add()
    log_record.time_unix_nano = 1704110400000000000
    log_record.body.string_value = "Traced log"
    log_record.trace_id = bytes.fromhex("0102030405060708090a0b0c0d0e0f10")
    log_record.span_id = bytes.fromhex("0102030405060708")
    log_record.flags = 1  # TRACE_FLAGS_SAMPLED

    proto_bytes = proto_request.SerializeToString()
    mock_log_storage.insert_logs.return_value = 1

    response = test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        content=proto_bytes,
        headers={"Content-Type": "application/x-protobuf"},
    )

    assert response.status_code == 200
    mock_log_storage.insert_logs.assert_called_once()


async def test_upload_protobuf_invalid_format(test_client):
    """Test that invalid protobuf data returns 400 error."""

    # Send invalid protobuf bytes
    invalid_bytes = b"not valid protobuf data"

    response = test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        content=invalid_bytes,
        headers={"Content-Type": "application/x-protobuf"},
    )

    assert response.status_code == 400
    assert "Invalid protobuf format" in response.json()["detail"]


async def test_upload_protobuf_missing_attributes(test_client, mock_log_storage):
    """Test protobuf format handles missing required resource attributes."""

    proto_request = logs_service_pb2.ExportLogsServiceRequest()

    # First resource with incomplete attributes - both logs rejected
    resource_logs = proto_request.resource_logs.add()
    # Add only partial resource attributes (missing job_task)
    for key, value in [
        ("job", "incomplete-job"),
        ("job_attempt", "attempt-1"),
        ("job_step", "step-1"),
    ]:
        attr = resource_logs.resource.attributes.add()
        attr.key = key
        attr.value.string_value = value

    scope_logs = resource_logs.scope_logs.add()
    log_record = scope_logs.log_records.add()
    log_record.time_unix_nano = 1704110400000000000
    log_record.body.string_value = "Log 1"

    log_record = scope_logs.log_records.add()
    log_record.time_unix_nano = 1704110405000000000
    log_record.body.string_value = "Log 2"

    proto_bytes = proto_request.SerializeToString()

    response = test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        content=proto_bytes,
        headers={"Content-Type": "application/x-protobuf"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data.get("partialSuccess") is not None
    assert data["partialSuccess"]["rejectedLogRecords"] == 2
    assert "job_task" in data["partialSuccess"]["errorMessage"]

    # No logs should be inserted
    mock_log_storage.insert_logs.assert_not_called()


async def test_json_and_protobuf_produce_same_result(test_client, mock_log_storage):
    """Test that JSON and protobuf formats produce equivalent results."""

    # Create JSON request
    json_request = {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {"key": "job", "value": {"stringValue": "test-job"}},
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
                                "severityNumber": 9,
                                "severityText": "INFO",
                                "body": {"stringValue": "Test message"},
                            }
                        ]
                    }
                ],
            }
        ]
    }

    # Create equivalent protobuf request
    proto_request = logs_service_pb2.ExportLogsServiceRequest()
    resource_logs = proto_request.resource_logs.add()

    # Add resource-level attributes
    for key, value in [
        ("job", "test-job"),
        ("job_attempt", "attempt-1"),
        ("job_step", "step-1"),
        ("job_task", "task-1"),
    ]:
        attr = resource_logs.resource.attributes.add()
        attr.key = key
        attr.value.string_value = value

    scope_logs = resource_logs.scope_logs.add()
    log_record = scope_logs.log_records.add()
    log_record.time_unix_nano = 1704110400000000000
    log_record.severity_number = 9  # type: ignore
    log_record.severity_text = "INFO"
    log_record.body.string_value = "Test message"

    proto_bytes = proto_request.SerializeToString()
    mock_log_storage.insert_logs.return_value = 1

    # Send JSON request
    response_json = test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        json=json_request,
    )
    assert response_json.status_code == 200
    json_log_entries = mock_log_storage.insert_logs.call_args.kwargs["log_entries"]

    # Reset mock
    mock_log_storage.reset_mock()
    mock_log_storage.insert_logs.return_value = 1

    # Send protobuf request
    response_proto = test_client.post(
        "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
        content=proto_bytes,
        headers={"Content-Type": "application/x-protobuf"},
    )
    assert response_proto.status_code == 200
    proto_log_entries = mock_log_storage.insert_logs.call_args.kwargs["log_entries"]

    # Compare results
    assert len(json_log_entries) == len(proto_log_entries) == 1
    json_entry = json_log_entries[0]
    proto_entry = proto_log_entries[0]

    assert json_entry.workspace == proto_entry.workspace
    assert json_entry.job == proto_entry.job
    assert json_entry.job_attempt == proto_entry.job_attempt
    assert json_entry.job_step == proto_entry.job_step
    assert json_entry.job_task == proto_entry.job_task
    assert json_entry.log_message == proto_entry.log_message
    assert json_entry.timestamp == proto_entry.timestamp


async def test_content_type_variations(test_client, mock_log_storage):
    """Test that various protobuf content-type headers are recognized."""
    # Create simple protobuf request
    proto_request = logs_service_pb2.ExportLogsServiceRequest()
    resource_logs = proto_request.resource_logs.add()

    # Add resource-level attributes
    for key, value in [
        ("job", "test-job"),
        ("job_attempt", "attempt-1"),
        ("job_step", "step-1"),
        ("job_task", "task-1"),
    ]:
        attr = resource_logs.resource.attributes.add()
        attr.key = key
        attr.value.string_value = value

    scope_logs = resource_logs.scope_logs.add()
    log_record = scope_logs.log_records.add()
    log_record.time_unix_nano = 1704110400000000000
    log_record.body.string_value = "Test"

    proto_bytes = proto_request.SerializeToString()
    mock_log_storage.insert_logs.return_value = 1

    # Test various content type headers
    content_types = [
        "application/x-protobuf",
        "application/protobuf",
        "application/x-protobuf; charset=utf-8",
    ]

    for content_type in content_types:
        mock_log_storage.reset_mock()
        mock_log_storage.insert_logs.return_value = 1

        response = test_client.post(
            "/v2/workspaces/test-workspace/filesets/test-logs-fileset/otlp/v1/logs",
            content=proto_bytes,
            headers={"Content-Type": content_type},
        )

        assert response.status_code == 200, f"Failed for content-type: {content_type}"
        mock_log_storage.insert_logs.assert_called_once()
