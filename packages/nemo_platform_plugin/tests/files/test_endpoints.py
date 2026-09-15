# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Files service endpoint definitions."""

from __future__ import annotations

import json
from typing import get_origin

from nemo_platform_plugin.client.types import BinaryContent, Paginated, PreparedRequest
from nemo_platform_plugin.files import endpoints
from nemo_platform_plugin.files.types import (
    CreateFilesetRequest,
    FilesetFileOutput,
    FilesetOutput,
    ListFilesetFilesResponse,
    OtlpExportLogsResponse,
    OtlpLogQueryRequest,
    UpdateFilesetRequest,
)


def test_create_fileset() -> None:
    body = CreateFilesetRequest(name="my-fileset")
    prepared = endpoints.create_fileset(workspace="default", body=body)

    assert isinstance(prepared, PreparedRequest)
    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/files/v2/workspaces/{workspace}/filesets"
    assert prepared.path_params == {"workspace": "default"}
    assert prepared.content == body.model_dump_json(exclude_unset=True).encode()
    assert prepared.content_type == "application/json"
    assert prepared.response_type is FilesetOutput


def test_create_fileset_workspace_optional() -> None:
    body = CreateFilesetRequest(name="my-fileset")
    prepared = endpoints.create_fileset(body=body)

    assert prepared.path_params == {}


def test_list_filesets() -> None:
    prepared = endpoints.list_filesets(workspace="default")

    assert prepared.method == "GET"
    assert prepared.path_params == {"workspace": "default"}
    assert prepared.content is None
    assert get_origin(prepared.response_type) is Paginated


def test_list_filesets_with_query_params() -> None:
    prepared = endpoints.list_filesets(workspace="default", query_params={"page": 2, "page_size": 10})

    assert prepared.query_params == {"page": 2, "page_size": 10}


def test_get_fileset() -> None:
    prepared = endpoints.get_fileset(workspace="default", name="my-fileset")

    assert prepared.method == "GET"
    assert prepared.path_params == {"workspace": "default", "name": "my-fileset"}
    assert prepared.response_type is FilesetOutput


def test_update_fileset() -> None:
    body = UpdateFilesetRequest(description="updated desc")
    prepared = endpoints.update_fileset(workspace="default", name="my-fileset", body=body)

    assert prepared.method == "PATCH"
    assert prepared.path_params == {"workspace": "default", "name": "my-fileset"}
    assert prepared.content == body.model_dump_json(exclude_unset=True).encode()
    assert prepared.response_type is FilesetOutput


def test_delete_fileset() -> None:
    prepared = endpoints.delete_fileset(workspace="default", name="my-fileset")

    assert prepared.method == "DELETE"
    assert prepared.path_params == {"workspace": "default", "name": "my-fileset"}
    assert prepared.content is None
    assert prepared.response_type is FilesetOutput


def test_list_files() -> None:
    prepared = endpoints.list_files(workspace="default", name="my-fileset")

    assert prepared.method == "GET"
    assert prepared.path_params == {"workspace": "default", "name": "my-fileset"}
    assert prepared.response_type is ListFilesetFilesResponse


def test_list_files_with_query_params() -> None:
    prepared = endpoints.list_files(
        workspace="default", name="my-fileset", query_params={"path": "data/", "include_cache_status": True}
    )

    assert prepared.query_params == {"path": "data/", "include_cache_status": True}


def test_upload_file() -> None:
    prepared = endpoints.upload_file(workspace="default", name="my-fileset", path="data/file.txt", content=b"hello")

    assert prepared.method == "PUT"
    assert prepared.path_params == {"workspace": "default", "name": "my-fileset", "path": "data/file.txt"}
    assert prepared.content == b"hello"
    assert prepared.content_type == "application/octet-stream"
    assert prepared.response_type is FilesetFileOutput


def test_download_file() -> None:
    prepared = endpoints.download_file(workspace="default", name="my-fileset", path="data/file.txt")

    assert prepared.method == "GET"
    assert prepared.path_params == {"workspace": "default", "name": "my-fileset", "path": "data/file.txt"}
    assert prepared.content is None
    assert prepared.response_type is BinaryContent


def test_delete_file() -> None:
    prepared = endpoints.delete_file(workspace="default", name="my-fileset", path="data/file.txt")

    assert prepared.method == "DELETE"
    assert prepared.path_params == {"workspace": "default", "name": "my-fileset", "path": "data/file.txt"}
    assert prepared.content is None
    assert prepared.response_type is FilesetFileOutput


def test_create_fileset_with_project() -> None:
    """project field must be preserved in the request body."""
    body = CreateFilesetRequest(name="my-fileset", project="my-project")
    prepared = endpoints.create_fileset(workspace="default", body=body)

    import json

    content = json.loads(prepared.content)
    assert content["project"] == "my-project"


def test_update_fileset_excludes_unset_fields() -> None:
    """Only explicitly set fields should be in the request body (exclude_unset)."""
    body = UpdateFilesetRequest(description="updated")
    prepared = endpoints.update_fileset(workspace="default", name="my-fileset", body=body)

    import json

    content = json.loads(prepared.content)
    assert content == {"description": "updated"}
    assert "purpose" not in content
    assert "metadata" not in content


def test_upload_otlp_logs() -> None:
    prepared = endpoints.upload_otlp_logs(workspace="default", name="my-fileset", content=b'{"resourceLogs": []}')

    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/files/v2/workspaces/{workspace}/filesets/{name}/otlp/v1/logs"
    assert prepared.path_params == {"workspace": "default", "name": "my-fileset"}
    assert prepared.content == b'{"resourceLogs": []}'
    assert prepared.query_params is None
    assert prepared.response_type is OtlpExportLogsResponse


def test_upload_otlp_logs_with_artifact_base_path() -> None:
    prepared = endpoints.upload_otlp_logs(
        workspace="default", name="my-fileset", content=b"{}", query_params={"artifact_base_path": "logs/run-1"}
    )

    assert prepared.query_params == {"artifact_base_path": "logs/run-1"}


def test_query_otlp_logs() -> None:
    body = OtlpLogQueryRequest(limit=10, artifact_base_path="logs/run-1")
    prepared = endpoints.query_otlp_logs(workspace="default", name="my-fileset", body=body)

    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/files/v2/workspaces/{workspace}/filesets/{name}/otlp/v1/logs/query"
    assert prepared.path_params == {"workspace": "default", "name": "my-fileset"}
    assert json.loads(prepared.content) == {"limit": 10, "artifact_base_path": "logs/run-1"}
    assert prepared.content_type == "application/json"
