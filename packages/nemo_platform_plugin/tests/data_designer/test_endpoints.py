# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Data Designer service endpoint definitions."""

from __future__ import annotations

import json
from typing import get_origin

from nemo_platform_plugin.client.types import BinaryContent, Paginated, PreparedRequest, Stream
from nemo_platform_plugin.data_designer import endpoints
from nemo_platform_plugin.data_designer.types import (
    DataDesignerJobRequest,
    DataDesignerJobResponse,
    PreviewFrameData,
    PreviewRequest,
    RetrievalPreviewRequest,
)
from nemo_platform_plugin.jobs.schemas import (
    PlatformJobListResultResponse,
    PlatformJobLogPage,
    PlatformJobResultResponse,
    PlatformJobStatusResponse,
)


def _job_request() -> DataDesignerJobRequest:
    return DataDesignerJobRequest(
        name="job-a",
        spec={"config": {"columns": []}, "num_records": 10},
        profile="gpu-large",
        options={"priority": "high"},
        custom_fields={"purpose": "test"},
    )


def _json_content(prepared: PreparedRequest) -> object:
    assert isinstance(prepared.content, bytes)
    return json.loads(prepared.content)


def test_preview_endpoint_shape() -> None:
    body = PreviewRequest(config={"columns": []}, num_records=3)

    prepared = endpoints.preview(workspace="default", body=body)

    assert isinstance(prepared, PreparedRequest)
    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/data-designer/v2/workspaces/{workspace}/preview"
    assert prepared.path_params == {"workspace": "default"}
    assert prepared.content == body.model_dump_json(exclude_unset=True).encode()
    assert prepared.content_type == "application/json"
    assert get_origin(prepared.response_type) is Stream


def test_retrieval_preview_endpoint_shape() -> None:
    body = RetrievalPreviewRequest(generate={"corpus": "system/corpus", "provider": "default/llm"}, num_records=2)

    prepared = endpoints.retrieval_preview(workspace="default", body=body)

    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/data-designer/v2/workspaces/{workspace}/retrieval-preview"
    assert prepared.path_params == {"workspace": "default"}
    assert prepared.content == body.model_dump_json(exclude_unset=True).encode()
    assert get_origin(prepared.response_type) is Stream


def test_retrieval_preview_num_records_defaults_to_service_contract() -> None:
    body = RetrievalPreviewRequest(generate={"corpus": "system/corpus", "provider": "default/llm"})

    prepared = endpoints.retrieval_preview(workspace="default", body=body)

    assert _json_content(prepared) == {
        "generate": {"corpus": "system/corpus", "provider": "default/llm"},
    }


def test_create_job_defaults_to_create_collection() -> None:
    body = _job_request()

    prepared = endpoints.create_job(workspace="default", body=body)

    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/data-designer/v2/workspaces/{workspace}/jobs/{job_collection}"
    assert prepared.path_params == {"workspace": "default", "job_collection": "create"}
    assert _json_content(prepared) == {
        "name": "job-a",
        "spec": {"config": {"columns": []}, "num_records": 10},
        "profile": "gpu-large",
        "options": {"priority": "high"},
        "custom_fields": {"purpose": "test"},
    }
    assert prepared.response_type is DataDesignerJobResponse


def test_create_job_supports_retrieval_collection() -> None:
    prepared = endpoints.create_job(
        workspace="default",
        job_collection="retrieval-generate",
        body=DataDesignerJobRequest(spec={"corpus": "system/corpus"}),
    )

    assert prepared.path_params == {"workspace": "default", "job_collection": "retrieval-generate"}


def test_list_jobs_endpoint_shape() -> None:
    prepared = endpoints.list_jobs(
        workspace="default",
        job_collection="retrieval-prepare",
        query_params={"page": 2, "page_size": 25, "sort": "-created_at"},
    )

    assert prepared.method == "GET"
    assert prepared.path_params == {"workspace": "default", "job_collection": "retrieval-prepare"}
    assert prepared.query_params == {"page": 2, "page_size": 25, "sort": "-created_at"}
    assert get_origin(prepared.response_type) is Paginated


def test_get_job_endpoint_shape() -> None:
    prepared = endpoints.get_job(workspace="default", job_collection="retrieval-run", name="job-a")

    assert prepared.method == "GET"
    assert prepared.path_template.endswith("/jobs/{job_collection}/{name}")
    assert prepared.path_params == {"workspace": "default", "job_collection": "retrieval-run", "name": "job-a"}
    assert prepared.response_type is DataDesignerJobResponse


def test_delete_job_endpoint_shape() -> None:
    prepared = endpoints.delete_job(workspace="default", name="job-a")

    assert prepared.method == "DELETE"
    assert prepared.path_params == {"workspace": "default", "job_collection": "create", "name": "job-a"}
    assert prepared.response_type is None


def test_cancel_job_endpoint_shape() -> None:
    prepared = endpoints.cancel_job(workspace="default", job_collection="retrieval-run", name="job-a")

    assert prepared.method == "POST"
    assert prepared.path_template.endswith("/jobs/{job_collection}/{name}/cancel")
    assert prepared.response_type is DataDesignerJobResponse


def test_get_job_status_endpoint_shape() -> None:
    prepared = endpoints.get_job_status(workspace="default", name="job-a")

    assert prepared.method == "GET"
    assert prepared.path_template.endswith("/jobs/{job_collection}/{name}/status")
    assert prepared.response_type is PlatformJobStatusResponse


def test_get_job_logs_endpoint_shape() -> None:
    prepared = endpoints.get_job_logs(
        workspace="default",
        name="job-a",
        query_params={"limit": 100, "page_cursor": "abc", "tail": 20},
    )

    assert prepared.method == "GET"
    assert prepared.path_template.endswith("/jobs/{job_collection}/{name}/logs")
    assert prepared.query_params == {"limit": 100, "page_cursor": "abc", "tail": 20}
    assert prepared.response_type is PlatformJobLogPage


def test_list_job_results_endpoint_shape() -> None:
    prepared = endpoints.list_job_results(workspace="default", name="job-a")

    assert prepared.method == "GET"
    assert prepared.path_template.endswith("/jobs/{job_collection}/{name}/results")
    assert prepared.response_type is PlatformJobListResultResponse


def test_get_job_result_endpoint_shape() -> None:
    prepared = endpoints.get_job_result(workspace="default", job="job-a", name="artifacts")

    assert prepared.method == "GET"
    assert prepared.path_template.endswith("/jobs/{job_collection}/{job}/results/{name}")
    assert prepared.path_params == {
        "workspace": "default",
        "job_collection": "create",
        "job": "job-a",
        "name": "artifacts",
    }
    assert prepared.response_type is PlatformJobResultResponse


def test_download_job_result_endpoint_shape() -> None:
    prepared = endpoints.download_job_result(workspace="default", job="job-a", name="artifacts")

    assert prepared.method == "GET"
    assert prepared.path_template.endswith("/jobs/{job_collection}/{job}/results/{name}/download")
    assert prepared.content is None
    assert prepared.response_type is BinaryContent


def test_workspace_can_be_omitted_for_client_default_resolution() -> None:
    prepared = endpoints.get_job(name="job-a")

    assert prepared.path_params == {"job_collection": "create", "name": "job-a"}


def test_preview_frame_type_is_importable_for_stream_validation() -> None:
    frame = PreviewFrameData(kind="heartbeat")

    assert frame.kind == "heartbeat"
