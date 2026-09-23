# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from nemo_helix_plugin.client.errors import (
    ConflictError as ClientConflictError,
)
from nemo_helix_plugin.client.errors import (
    NotFoundError as ClientNotFoundError,
)
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    FileResultSerializer,
    HelixJobResultRoute,
    HelixJobSpec,
    HelixJobStep,
    JobRouteOption,
    PydanticJSONLResultSerializer,
    PydanticResultSerializer,
    _accepts_entity_client,
    _compile_platform_spec,
    _is_basemodel_union,
    _resolve_job_name,
    _transform_input_to_output,
    _unwrap_annotated_schema,
    _validate_and_resolve_job_output,
    _validate_basemodel_or_union,
    _validate_job_spec,
    job_route_factory,
)
from nemo_helix_plugin.jobs.exceptions import HelixJobDependencyUnavailableError
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_helix_plugin.jobs.types import HelixJobResponse as HelixJob
from nhx.common.errors.sdk_exception_handlers import register_sdk_exception_handlers
from nhx.common.jobs.exceptions import HelixJobCompilationError
from nhx.common.jobs.schemas import (
    FileStorageType,
    HelixJobListResultResponse,
    HelixJobLog,
    HelixJobLogPage,
    HelixJobResultResponse,
)
from nhx.common.service.dependencies import get_entity_client, get_sdk_client
from pydantic import BaseModel


class FooJobConfig(BaseModel):
    foo: str
    bar: int


def foo_job_config_compiler(
    workspace: str,
    input_spec: FooJobConfig,
    output_spec: FooJobConfig,
    entity_client: EntityClient,
    job_name: str | None,
    sdk,
) -> HelixJobSpec:
    return HelixJobSpec(
        steps=[
            HelixJobStep(
                name="foo_step",
                executor=CPUExecutionProviderSpec(
                    provider="cpu",
                    profile="default",
                    container=ContainerSpec(
                        image="foo_image",
                    ),
                ),
                config={"foo": output_spec.foo, "bar": output_spec.bar},
            )
        ]
    )


def test_api_factory_routes():
    router = job_route_factory(
        service_name="foo",
        job_type="Foo",
        job_input=FooJobConfig,
        route_options=[JobRouteOption.CORE, JobRouteOption.PAUSE_RESUME],
        platform_job_config_compiler=foo_job_config_compiler,
    )

    assert len(router.routes) > 0, "No routes were created by the job_route_factory"

    # Check that each expected route is present and matches the expected method and path
    expected_routes = [
        ("get", "/jobs"),
        ("post", "/jobs"),
        ("get", "/jobs/{name}"),
        ("get", "/jobs/{name}/status"),
        ("delete", "/jobs/{name}"),
        ("post", "/jobs/{name}/cancel"),
        ("post", "/jobs/{name}/pause"),
        ("post", "/jobs/{name}/resume"),
        ("get", "/jobs/{name}/logs"),
        ("get", "/jobs/{name}/results"),
        ("get", "/jobs/{job}/results/{name}"),
        ("get", "/jobs/{job}/results/{name}/download"),
    ]

    assert len(router.routes) == len(expected_routes), (
        f"Expected {len(expected_routes)} routes, but found {len(router.routes)}"
    )

    api_routes = [r for r in router.routes if isinstance(r, APIRoute)]
    for method, path in expected_routes:
        assert any(method.upper() in route.methods and route.path == path for route in api_routes), (
            f"Expected route {method.upper()} {path} not found in the API factory"
        )


def test_validate_job_spec():
    executor = CPUExecutionProviderSpec(provider="cpu", profile="default", container=ContainerSpec(image="foo_image"))
    valid_job = HelixJobSpec(
        steps=[
            HelixJobStep(
                name="foo_step",
                environment={},
                executor=executor,
                config={"foo": "test", "bar": 1},
            )
        ]
    )
    assert _validate_job_spec(valid_job) == valid_job

    # Invalid job spec, because the config provided into the step's config is not json serializeable
    invalid_job = HelixJobSpec(
        steps=[
            HelixJobStep(
                name="foo_step",
                environment={},
                executor=executor,
                config={"time_now": datetime.now(), "bar": "not_a_number"},
            )
        ]
    )
    with pytest.raises(HelixJobCompilationError) as exc_info:
        _validate_job_spec(invalid_job)
    assert (
        str(exc_info.value) == "step config is not json serializable: Object of type datetime is not JSON serializable"
    )


def test_platform_compiler_should_raise_compilation_error():
    # This test should check that the platform job config compiler raises a compilation error
    # when given an invalid job config.

    def job_compiler_always_raises(config: FooJobConfig) -> HelixJobSpec:
        raise HelixJobCompilationError("Compilation failed")

    with pytest.raises(HelixJobCompilationError) as exc_info:
        job_compiler_always_raises(FooJobConfig(foo="foo", bar=1))
    assert str(exc_info.value) == "Compilation failed"


def test_api_factory_routes_results():
    class TestModel(BaseModel):
        name: str
        age: int

    job_result_routes = [
        HelixJobResultRoute(name="testfile", serializer=FileResultSerializer()),
        HelixJobResultRoute(name="testjson", serializer=FileResultSerializer()),
        HelixJobResultRoute(name="testpydanticjsonl", serializer=PydanticJSONLResultSerializer(model=TestModel)),
        HelixJobResultRoute(name="testpydantic", serializer=PydanticResultSerializer(model=TestModel)),
    ]
    router = job_route_factory(
        service_name="foo",
        job_type="Foo",
        job_input=FooJobConfig,
        job_result_routes=job_result_routes,
        platform_job_config_compiler=foo_job_config_compiler,
    )

    app = FastAPI()
    app.include_router(router)

    openapi_schema = get_openapi(
        title="Your API Title",
        version="1.0.0",
        routes=app.routes,
    )
    download_schemas = {k: v for k, v in openapi_schema["paths"].items() if k.endswith("/download")}

    # ensure each custom result route and the fallback route exist
    assert len(download_schemas) == len(job_result_routes) + 1

    schema = download_schemas["/jobs/{job}/results/testfile/download"]
    assert "application/octet-stream" in str(schema)

    schema = download_schemas["/jobs/{job}/results/testjson/download"]
    assert "application/json" in str(schema)

    schema = download_schemas["/jobs/{job}/results/testpydantic/download"]
    assert "#/components/schemas/TestModel" in str(schema)

    schema = download_schemas["/jobs/{job}/results/testpydanticjsonl/download"]
    assert "application/jsonl" in str(schema)
    assert "#/components/schemas/TestModel" in str(schema)


def test_pydantic_result_serializer(tmp_path: Path):
    class TestModel(BaseModel):
        name: str
        age: int
        created_at: datetime
        something_unset: str | None = None

    def _test_json(json_in: dict, serialize_kwargs: dict) -> dict:
        json_path = tmp_path / "test.json"
        with json_path.open("w") as f:
            f.write(json.dumps(json_in))

        serializer = PydanticResultSerializer(model=TestModel, serialize_kwargs=serialize_kwargs)
        body = serializer.serialize(json_path).body
        assert isinstance(body, bytes)
        return json.loads(body.decode())

    created_at = "2022-09-20 10:27:21.240752"
    json_out = _test_json({"name": "jane doe", "age": 42, "created_at": created_at}, {})
    assert isinstance(json_out, dict)
    assert "something_unset" in json_out

    json_out = _test_json({"name": "jane doe", "age": 42, "created_at": created_at}, {"exclude_unset": True})
    assert isinstance(json_out, dict)
    assert "something_unset" not in json_out

    # json that doesn't match the schema should still return
    json_out = _test_json({"other": "other key", "field": "value"}, {})
    assert isinstance(json_out, dict)
    assert "other" in json_out


@pytest.mark.asyncio
async def test_pydantic_jsonl_result_serializer(tmp_path: Path):
    class TestModel(BaseModel):
        name: str
        age: int

    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text("\n".join([json.dumps({"name": "jane", "age": 42}), json.dumps({"name": "john", "age": 21})]))

    serializer = PydanticJSONLResultSerializer(model=TestModel)
    response = serializer.serialize(jsonl_path)
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, str) else bytes(chunk).decode())

    lines = [line for line in "".join(chunks).splitlines() if line.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"name": "jane", "age": 42}
    assert json.loads(lines[1]) == {"name": "john", "age": 21}


@pytest.mark.asyncio
async def test_pydantic_jsonl_result_serializer_validation_error(tmp_path: Path):
    class TestModel(BaseModel):
        name: str
        age: int

    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text(json.dumps({"name": "jane", "age": "bad"}) + "\n")

    serializer = PydanticJSONLResultSerializer(model=TestModel)
    response = serializer.serialize(jsonl_path)

    chunks: list[str] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, str) else bytes(chunk).decode())

    lines = [line for line in "".join(chunks).splitlines() if line.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["error"]["line"] == 1


def _resp(data):
    """Wrap a payload in a NemoResponse-like object whose ``.data()`` returns it.

    Production now consumes typed-client responses via ``(await client.<op>(...)).data()``,
    so mocked jobs-client methods must return an object with a ``.data()`` accessor
    rather than the payload directly.
    """
    m = MagicMock()
    m.data.return_value = data
    return m


def _page_resp(items, *, page=1, page_size=10, total_pages=1, total_results=None):
    """Wrap a list of jobs in a response whose ``.page()`` returns a PageResult-like object.

    The list handler calls ``.page()`` and reads its items and offset metadata.
    """
    page_result = MagicMock()
    page_result.items = items
    page_result.metadata = {
        "page": page,
        "page_size": page_size,
        "current_page_size": len(items),
        "total_pages": total_pages,
        "total_results": total_results if total_results is not None else len(items),
    }
    m = MagicMock()
    m.page.return_value = page_result
    return m


def _cursor_page_resp(page: HelixJobLogPage):
    """Wrap a Jobs log envelope as a cursor-paginated typed-client response."""
    page_result = MagicMock()
    page_result.items = page.data
    page_result.metadata = {
        "total": page.total,
        "next_page": page.next_page,
        "prev_page": page.prev_page,
    }
    response = MagicMock()
    response.page.return_value = page_result
    return response


def _client_error(error_cls, status_code: int, detail: str):
    """Build a NemoHTTPError subclass from an httpx.Response, as the typed client raises."""
    request = httpx.Request("POST", "http://test")
    response = httpx.Response(status_code=status_code, json={"detail": detail}, request=request)
    return error_cls(response)


@contextmanager
def _job_routes_app():
    """Create a test FastAPI app with job routes, patching the typed jobs client.

    The generated handlers call ``client_from_platform(sdk, AsyncJobsClient)`` and then
    invoke methods on the returned client. We patch ``client_from_platform`` in the
    api_factory module to return a single ``MagicMock`` jobs client that tests configure
    (each method as an ``AsyncMock`` returning a ``_resp(...)`` / ``_page_resp(...)``).
    """
    from nhx.common.service.dependencies import get_sdk_client

    mock_sdk = MagicMock()
    mock_jobs = MagicMock()
    mock_jobs.with_retry.return_value = mock_jobs
    router = job_route_factory(
        service_name="test_service",
        job_type="TestJob",
        job_input=FooJobConfig,
        platform_job_config_compiler=foo_job_config_compiler,
        route_options=[JobRouteOption.CORE, JobRouteOption.PAUSE_RESUME],
    )
    app = FastAPI()
    register_sdk_exception_handlers(app)

    # Setup dependency overrides for SDK and entity client injection
    app.dependency_overrides[get_sdk_client] = lambda: mock_sdk
    app.dependency_overrides[get_entity_client] = lambda: MagicMock()

    # Include a prefix to indicate the location of the jobs router
    app.include_router(router, prefix="/v2/workspaces/{workspace}/test")

    with patch("nemo_helix_plugin.jobs.api_factory.client_from_platform", return_value=mock_jobs):
        yield app, mock_jobs


@pytest.fixture
def mock_service_with_job_routes():
    """Yield a test FastAPI app plus the mocked typed jobs client for job-route tests."""
    with _job_routes_app() as (app, mock_jobs):
        yield app, mock_jobs


def create_mock_platform_job(
    job_id: str,
    status: HelixJobStatus | str = HelixJobStatus.COMPLETED,
    created_at: str = "2023-01-01T10:00:00Z",
) -> HelixJob:
    """Helper function to create a mock HelixJob for testing."""
    return HelixJob(
        id=job_id,
        workspace="default",
        name=f"Job {job_id}",
        source="test_service",
        project="test-project",
        status=HelixJobStatus(status),
        created_at=created_at,  # ty: ignore[invalid-argument-type] # type is coerced
        updated_at=created_at,  # ty: ignore[invalid-argument-type] # type is coerced
        spec={"foo": "test", "bar": 1},
        platform_spec=HelixJobSpec(
            steps=[
                HelixJobStep(
                    name="foo_step",
                    executor=CPUExecutionProviderSpec(
                        provider="cpu",
                        profile="default",
                        container=ContainerSpec(image="foo_image"),
                    ),
                    config={},
                )
            ]
        ),
        ownership={},
        attempt_id=f"attempt-{job_id}",
        fileset=f"fileset-{job_id}",
    )


def create_mock_log_page(
    num_logs: int = 3,
    next_page: str | None = None,
    prev_page: str | None = None,
    total: int | None = None,
) -> HelixJobLogPage:
    """Helper function to create a mock HelixJobLogPage for testing."""
    logs = []
    base_timestamp = datetime(2023, 1, 1, 10, 0, 0)

    for i in range(num_logs):
        # Create timestamps by adding seconds - this avoids overflow issues
        timestamp = base_timestamp + timedelta(seconds=i)

        log = HelixJobLog(
            timestamp=timestamp,
            job="test-job-123",
            job_step=f"step_{i}",
            job_task=f"task_{i}",
            message=f"Log message {i}",
        )
        logs.append(log)

    return HelixJobLogPage(
        data=logs,
        total=total if total is not None else num_logs,
        next_page=next_page,
        prev_page=prev_page,
    )


def test_get_job_logs_default_parameters(mock_service_with_job_routes):
    """Test get_job_logs with default parameters (no limit or page_cursor)."""
    app, mock_jobs = mock_service_with_job_routes
    client = TestClient(app)

    # Mock the jobs-client response
    mock_log_page = create_mock_log_page(num_logs=5, total=10)
    mock_jobs.list_job_logs = AsyncMock(return_value=_cursor_page_resp(mock_log_page))

    # Make the request
    response = client.get("/v2/workspaces/default/test/jobs/test-job-123/logs")

    # Verify the response
    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data["data"]) == 5
    assert response_data["total"] == 10

    # None-valued limit/page_cursor are omitted from query_params.
    mock_jobs.list_job_logs.assert_called_once_with(
        workspace="default",
        name="test-job-123",
        query_params={},
    )


def test_get_job_logs_with_limit(mock_service_with_job_routes):
    """Test get_job_logs with limit parameter."""
    app, mock_jobs = mock_service_with_job_routes
    client = TestClient(app)

    # Mock the jobs-client response
    mock_log_page = create_mock_log_page(num_logs=2, total=10, next_page="next_cursor_123")
    mock_jobs.list_job_logs = AsyncMock(return_value=_cursor_page_resp(mock_log_page))

    # Make the request with limit
    response = client.get("/v2/workspaces/default/test/jobs/test-job-123/logs?limit=2")

    # Verify the response
    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data["data"]) == 2
    assert response_data["total"] == 10
    assert response_data["next_page"] == "next_cursor_123"

    mock_jobs.list_job_logs.assert_called_once_with(
        workspace="default",
        name="test-job-123",
        query_params={"limit": 2},
    )


def test_get_job_logs_with_page_cursor(mock_service_with_job_routes):
    """Test get_job_logs with page_cursor parameter."""
    app, mock_jobs = mock_service_with_job_routes
    client = TestClient(app)

    # Mock the jobs-client response
    mock_log_page = create_mock_log_page(num_logs=3, total=10, next_page="next_cursor_456", prev_page="prev_cursor_789")
    mock_jobs.list_job_logs = AsyncMock(return_value=_cursor_page_resp(mock_log_page))

    # Make the request with page_cursor
    response = client.get("/v2/workspaces/default/test/jobs/test-job-123/logs?page_cursor=cursor_abc_123")

    # Verify the response
    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data["data"]) == 3
    assert response_data["total"] == 10
    assert response_data["next_page"] == "next_cursor_456"
    assert response_data["prev_page"] == "prev_cursor_789"

    mock_jobs.list_job_logs.assert_called_once_with(
        workspace="default",
        name="test-job-123",
        query_params={"page_cursor": "cursor_abc_123"},
    )


def test_get_job_logs_with_tail(mock_service_with_job_routes):
    """Test get_job_logs with tail parameter."""
    app, mock_jobs = mock_service_with_job_routes
    client = TestClient(app)

    mock_log_page = create_mock_log_page(num_logs=2, total=10, prev_page="prev_cursor_123")
    mock_jobs.list_job_logs = AsyncMock(return_value=_cursor_page_resp(mock_log_page))

    response = client.get("/v2/workspaces/default/test/jobs/test-job-123/logs?tail=2")

    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data["data"]) == 2
    assert response_data["total"] == 10
    assert response_data["prev_page"] == "prev_cursor_123"

    mock_jobs.list_job_logs.assert_called_once_with(
        workspace="default",
        name="test-job-123",
        query_params={"tail": 2},
    )


def test_get_job_logs_with_both_parameters(mock_service_with_job_routes):
    """Test get_job_logs with both limit and page_cursor parameters."""
    app, mock_jobs = mock_service_with_job_routes
    client = TestClient(app)

    # Mock the jobs-client response
    mock_log_page = create_mock_log_page(
        num_logs=5, total=100, next_page="next_cursor_combined", prev_page="prev_cursor_combined"
    )
    mock_jobs.list_job_logs = AsyncMock(return_value=_cursor_page_resp(mock_log_page))

    # Make the request with both parameters
    response = client.get("/v2/workspaces/default/test/jobs/test-job-123/logs?limit=5&page_cursor=combined_cursor_xyz")

    # Verify the response
    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data["data"]) == 5
    assert response_data["total"] == 100
    assert response_data["next_page"] == "next_cursor_combined"
    assert response_data["prev_page"] == "prev_cursor_combined"

    mock_jobs.list_job_logs.assert_called_once_with(
        workspace="default",
        name="test-job-123",
        query_params={"limit": 5, "page_cursor": "combined_cursor_xyz"},
    )


def test_get_job_logs_with_zero_limit(mock_service_with_job_routes):
    """Test get_job_logs with limit=0."""
    app, mock_jobs = mock_service_with_job_routes
    client = TestClient(app)

    # Mock the jobs-client response
    mock_log_page = create_mock_log_page(num_logs=0, total=10, next_page="next_cursor_zero")
    mock_jobs.list_job_logs = AsyncMock(return_value=_cursor_page_resp(mock_log_page))

    # Make the request with limit=0
    response = client.get("/v2/workspaces/default/test/jobs/test-job-123/logs?limit=0")

    # Verify the response
    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data["data"]) == 0
    assert response_data["total"] == 10
    assert response_data["next_page"] == "next_cursor_zero"

    # limit=0 is not None, so it is included in query_params.
    mock_jobs.list_job_logs.assert_called_once_with(
        workspace="default",
        name="test-job-123",
        query_params={"limit": 0},
    )


def test_get_job_logs_empty_page_cursor(mock_service_with_job_routes):
    """Test get_job_logs with empty string page_cursor."""
    app, mock_jobs = mock_service_with_job_routes
    client = TestClient(app)

    # Mock the jobs-client response
    mock_log_page = create_mock_log_page(num_logs=3, total=10)
    mock_jobs.list_job_logs = AsyncMock(return_value=_cursor_page_resp(mock_log_page))

    # Make the request with empty page_cursor
    response = client.get("/v2/workspaces/default/test/jobs/test-job-123/logs?page_cursor=")

    # Verify the response
    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data["data"]) == 3

    # Empty string is not None, so it is included in query_params.
    mock_jobs.list_job_logs.assert_called_once_with(
        workspace="default",
        name="test-job-123",
        query_params={"page_cursor": ""},
    )


def test_get_job_logs_job_not_found(mock_service_with_job_routes):
    """Test get_job_logs when job is not found."""
    app, mock_jobs = mock_service_with_job_routes
    client = TestClient(app)

    # The typed client raises a NemoHTTPError subclass on non-2xx responses.
    mock_jobs.list_job_logs = AsyncMock(side_effect=_client_error(ClientNotFoundError, 404, "Job not found"))

    # Make the request
    response = client.get("/v2/workspaces/default/test/jobs/nonexistent-job/logs")

    # Verify the response
    assert response.status_code == 404
    response_data = response.json()
    assert "Job not found" in response_data["detail"]

    # Verify the jobs client was called
    mock_jobs.list_job_logs.assert_called_once_with(
        workspace="default",
        name="nonexistent-job",
        query_params={},
    )


def test_get_job_logs_large_limit(mock_service_with_job_routes):
    """Test get_job_logs with a large limit value."""
    app, mock_jobs = mock_service_with_job_routes
    client = TestClient(app)

    # Mock the jobs-client response
    mock_log_page = create_mock_log_page(num_logs=1000, total=1000)
    mock_jobs.list_job_logs = AsyncMock(return_value=_cursor_page_resp(mock_log_page))

    # Make the request with large limit
    response = client.get("/v2/workspaces/default/test/jobs/test-job-123/logs?limit=1000")

    # Verify the response
    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data["data"]) == 1000
    assert response_data["total"] == 1000

    mock_jobs.list_job_logs.assert_called_once_with(
        workspace="default",
        name="test-job-123",
        query_params={"limit": 1000},
    )


def test_get_job_logs_special_characters_in_cursor(mock_service_with_job_routes):
    """Test get_job_logs with special characters in page_cursor."""
    import urllib.parse

    app, mock_jobs = mock_service_with_job_routes
    client = TestClient(app)

    # Mock the jobs-client response
    mock_log_page = create_mock_log_page(num_logs=2, total=10)
    mock_jobs.list_job_logs = AsyncMock(return_value=_cursor_page_resp(mock_log_page))

    # Make the request with special characters in cursor (URL encoded)
    special_cursor = "cursor_with_special_chars_!@#$%^&*()_+-="
    encoded_cursor = urllib.parse.quote(special_cursor, safe="")
    response = client.get(f"/v2/workspaces/default/test/jobs/test-job-123/logs?page_cursor={encoded_cursor}")

    # Verify the response
    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data["data"]) == 2

    # Verify the jobs client was called with correct parameters (decoded)
    mock_jobs.list_job_logs.assert_called_once_with(
        workspace="default",
        name="test-job-123",
        query_params={"page_cursor": special_cursor},
    )


def test_get_job_logs_response_structure(mock_service_with_job_routes):
    """Test that get_job_logs returns proper HelixJobLogPage structure."""
    app, mock_jobs = mock_service_with_job_routes
    client = TestClient(app)

    # Create a detailed mock log page
    mock_logs = [
        HelixJobLog(
            timestamp=datetime(2023, 6, 15, 14, 30, 45),
            job="test-job-456",
            job_step="preprocessing",
            job_task="data_validation",
            message="Starting data validation process",
        ),
        HelixJobLog(
            timestamp=datetime(2023, 6, 15, 14, 31, 0),
            job="test-job-456",
            job_step="preprocessing",
            job_task="data_validation",
            message="Data validation completed successfully",
        ),
    ]

    mock_log_page = HelixJobLogPage(
        data=mock_logs,
        total=25,
        next_page="cursor_next_page_789",
        prev_page="cursor_prev_page_123",
    )

    mock_jobs.list_job_logs = AsyncMock(return_value=_cursor_page_resp(mock_log_page))

    # Make the request
    response = client.get("/v2/workspaces/default/test/jobs/test-job-456/logs?limit=2&page_cursor=middle_cursor")

    # Verify the response structure
    assert response.status_code == 200
    response_data = response.json()

    # Verify top-level structure
    assert response_data["total"] == 25
    assert response_data["next_page"] == "cursor_next_page_789"
    assert response_data["prev_page"] == "cursor_prev_page_123"
    assert len(response_data["data"]) == 2

    # Verify log entry structure
    log_entry = response_data["data"][0]
    assert "timestamp" in log_entry
    assert log_entry["job"] == "test-job-456"
    assert log_entry["job_step"] == "preprocessing"
    assert log_entry["job_task"] == "data_validation"
    assert log_entry["message"] == "Starting data validation process"

    # Verify the jobs client was called with correct parameters
    mock_jobs.list_job_logs.assert_called_once_with(
        workspace="default",
        name="test-job-456",
        query_params={"limit": 2, "page_cursor": "middle_cursor"},
    )


def test_list_jobs_no_results(mock_service_with_job_routes):
    app, mock_jobs = mock_service_with_job_routes

    client = TestClient(app)
    mock_jobs.list_jobs = AsyncMock(return_value=_page_resp([], page=1, page_size=5, total_pages=0, total_results=0))

    response = client.get("/v2/workspaces/default/test/jobs?page=1&page_size=5&filter[status]=completed")

    response_data = response.json()
    assert response_data["data"] == []
    assert response_data["pagination"]["page"] == 1
    assert response_data["pagination"]["page_size"] == 5

    # The user filter is AND-composed with the service-source predicate so
    # logical roots ($or/$and/$not) stay scoped — a flat dict merge would
    # silently drop the source clause under a logical root. The composed tree
    # is forwarded as a JSON string in ``query_params`` so the querystring
    # serializer doesn't mangle list-of-dict values.
    mock_jobs.list_jobs.assert_called_once_with(
        workspace="default",
        query_params={
            "page": 1,
            "page_size": 5,
            "sort": "-created_at",
            "filter": json.dumps(
                {
                    "$and": [
                        {"status": {"$eq": "completed"}},
                        {"source": {"$eq": "test_service"}},
                    ]
                }
            ),
        },
    )


def test_list_jobs_with_results(mock_service_with_job_routes):
    app, mock_jobs = mock_service_with_job_routes

    client = TestClient(app)
    mock_jobs.list_jobs = AsyncMock(
        return_value=_page_resp(
            [
                create_mock_platform_job("job-1", "completed", "2023-01-01T10:00:00Z"),
                create_mock_platform_job("job-2", "completed", "2023-01-01T11:00:00Z"),
                create_mock_platform_job("job-3", "completed", "2023-01-01T12:00:00Z"),
            ],
            page=1,
            page_size=5,
            total_pages=1,
            total_results=3,
        )
    )

    response = client.get("/v2/workspaces/default/test/jobs?page=1&page_size=5")

    response_data = response.json()
    assert len(response_data["data"]) == 3
    assert response_data["data"][0]["id"] == "job-1"
    assert response_data["data"][1]["id"] == "job-2"
    assert response_data["data"][2]["id"] == "job-3"
    assert response_data["pagination"]["page"] == 1
    assert response_data["pagination"]["page_size"] == 5
    assert response_data["pagination"]["current_page_size"] == 3
    assert response_data["pagination"]["total_results"] == 3

    # No user filter — source predicate stands alone (no $and wrap needed).
    mock_jobs.list_jobs.assert_called_once_with(
        workspace="default",
        query_params={
            "page": 1,
            "page_size": 5,
            "sort": "-created_at",
            "filter": json.dumps({"source": {"$eq": "test_service"}}),
        },
    )


def test_list_jobs_with_multiple_pages(mock_service_with_job_routes):
    app, mock_jobs = mock_service_with_job_routes

    client = TestClient(app)
    mock_jobs.list_jobs = AsyncMock(
        side_effect=[
            _page_resp(
                [
                    create_mock_platform_job("job-1", "active", "2023-01-01T10:00:00Z"),
                    create_mock_platform_job("job-2", "active", "2023-01-01T11:00:00Z"),
                ],
                page=1,
                page_size=2,
                total_pages=3,
                total_results=6,
            ),
            _page_resp(
                [
                    create_mock_platform_job("job-3", "active", "2023-01-01T10:00:00Z"),
                    create_mock_platform_job("job-4", "active", "2023-01-01T11:00:00Z"),
                ],
                page=2,
                page_size=2,
                total_pages=3,
                total_results=6,
            ),
        ]
    )

    # Fetch the first page
    response = client.get("/v2/workspaces/default/test/jobs?page=1&page_size=2&filter[status]=active")

    response_data = response.json()
    assert len(response_data["data"]) == 2
    assert response_data["pagination"]["page"] == 1
    assert response_data["pagination"]["total_pages"] == 3
    assert response_data["pagination"]["total_results"] == 6

    expected_filter = json.dumps(
        {
            "$and": [
                {"status": {"$eq": "active"}},
                {"source": {"$eq": "test_service"}},
            ]
        }
    )
    mock_jobs.list_jobs.assert_called_once_with(
        workspace="default",
        query_params={
            "page": 1,
            "page_size": 2,
            "sort": "-created_at",
            "filter": expected_filter,
        },
    )

    # Fetch the second page
    mock_jobs.list_jobs.reset_mock()
    response = client.get("/v2/workspaces/default/test/jobs?page=2&page_size=2&filter[status]=active")

    response_data = response.json()
    assert len(response_data["data"]) == 2
    assert response_data["pagination"]["page"] == 2
    assert response_data["pagination"]["total_pages"] == 3
    assert response_data["pagination"]["total_results"] == 6

    mock_jobs.list_jobs.assert_called_once_with(
        workspace="default",
        query_params={
            "page": 2,
            "page_size": 2,
            "sort": "-created_at",
            "filter": expected_filter,
        },
    )


def test_list_jobs_with_custom_sort(mock_service_with_job_routes):
    app, mock_jobs = mock_service_with_job_routes

    client = TestClient(app)
    mock_jobs.list_jobs = AsyncMock(
        return_value=_page_resp(
            [
                create_mock_platform_job("job-3", "completed", "2023-01-03T10:00:00Z"),
                create_mock_platform_job("job-2", "completed", "2023-01-02T10:00:00Z"),
                create_mock_platform_job("job-1", "completed", "2023-01-01T10:00:00Z"),
            ],
            page=1,
            page_size=10,
            total_pages=1,
            total_results=3,
        )
    )

    response = client.get("/v2/workspaces/default/test/jobs?page=1&page_size=10&sort=created_at")

    response_data = response.json()
    assert len(response_data["data"]) == 3
    assert response_data["data"][0]["id"] == "job-3"
    assert response_data["data"][1]["id"] == "job-2"
    assert response_data["data"][2]["id"] == "job-1"

    mock_jobs.list_jobs.assert_called_once_with(
        workspace="default",
        query_params={
            "page": 1,
            "page_size": 10,
            "sort": "created_at",
            "filter": json.dumps({"source": {"$eq": "test_service"}}),
        },
    )


def test_list_jobs_invalid_filter(mock_service_with_job_routes):
    app, mock_jobs = mock_service_with_job_routes

    client = TestClient(app)
    # ``source`` is not in BaseJobsListFilter — make_filter_dep's allowlist
    # rejects unknown fields with a 400 before any jobs-client call is made.
    response = client.get("/v2/workspaces/default/test/jobs?page=1&page_size=5&filter[source]=foo")

    assert response.status_code == 400
    response_data = response.json()
    assert "source" in response_data["detail"]

    mock_jobs.list_jobs.assert_not_called()


def test_list_jobs_invalid_sort(mock_service_with_job_routes):
    app, mock_jobs = mock_service_with_job_routes

    client = TestClient(app)
    response = client.get("/v2/workspaces/default/test/jobs?page=1&page_size=5&sort=foo")

    assert response.status_code == 422
    response_data = response.json()
    assert (
        response_data["detail"][0]["msg"]
        == "Input should be 'created_at', '-created_at', 'updated_at' or '-updated_at'"
    )

    mock_jobs.list_jobs.assert_not_called()


def test_list_job_results_includes_download_url(mock_service_with_job_routes):
    """Test that list_job_results includes download_url for each result."""
    app, mock_jobs = mock_service_with_job_routes
    client = TestClient(app)

    mock_results = [
        HelixJobResultResponse(
            name="result1",
            job="test-job-123",
            workspace="default",
            artifact_storage_type=FileStorageType.FILESET,
            artifact_url="default/test-fileset#result1",
        ),
        HelixJobResultResponse(
            name="result2",
            job="test-job-123",
            workspace="default",
            artifact_storage_type=FileStorageType.FILESET,
            artifact_url="default/test-fileset#result2",
        ),
    ]

    mock_jobs.list_job_results = AsyncMock(return_value=_resp(HelixJobListResultResponse(data=mock_results)))

    # Make the request
    response = client.get("/v2/workspaces/default/test/jobs/test-job-123/results")

    # Verify the response
    assert response.status_code == 200
    response_data = response.json()

    # Verify structure
    assert "data" in response_data
    assert len(response_data["data"]) == 2

    # Verify download_url is populated for each result
    result1 = response_data["data"][0]
    assert result1["name"] == "result1"
    assert "download_url" in result1
    assert (
        result1["download_url"]
        == "http://testserver/v2/workspaces/default/test/jobs/test-job-123/results/result1/download"
    )

    result2 = response_data["data"][1]
    assert result2["name"] == "result2"
    assert "download_url" in result2
    assert (
        result2["download_url"]
        == "http://testserver/v2/workspaces/default/test/jobs/test-job-123/results/result2/download"
    )

    # Verify the jobs client was called correctly
    mock_jobs.list_job_results.assert_called_once_with(name="test-job-123", workspace="default")


def test_list_job_results_empty_list(mock_service_with_job_routes):
    """Test that list_job_results works correctly with empty results."""
    app, mock_jobs = mock_service_with_job_routes
    client = TestClient(app)

    mock_jobs.list_job_results = AsyncMock(return_value=_resp(HelixJobListResultResponse(data=[])))

    # Make the request
    response = client.get("/v2/workspaces/default/test/jobs/test-job-456/results")

    # Verify the response
    assert response.status_code == 200
    response_data = response.json()
    assert "data" in response_data
    assert len(response_data["data"]) == 0

    # Verify the jobs client was called correctly
    mock_jobs.list_job_results.assert_called_once_with(name="test-job-456", workspace="default")


def test_get_job_result_includes_download_url(mock_service_with_job_routes):
    """Test that get_job_result includes download_url in the response."""
    app, mock_jobs = mock_service_with_job_routes
    client = TestClient(app)

    # Create mock result without download_url
    mock_result = HelixJobResultResponse(
        name="test_result",
        job="test-job-789",
        workspace="default",
        artifact_storage_type=FileStorageType.FILESET,
        artifact_url="default/test-fileset#test_result",
    )

    mock_jobs.get_job_result = AsyncMock(return_value=_resp(mock_result))

    # Make the request
    response = client.get("/v2/workspaces/default/test/jobs/test-job-789/results/test_result")

    # Verify the response
    assert response.status_code == 200
    response_data = response.json()

    # Verify download_url is populated
    assert response_data["name"] == "test_result"
    assert "download_url" in response_data
    assert (
        response_data["download_url"]
        == "http://testserver/v2/workspaces/default/test/jobs/test-job-789/results/test_result/download"
    )

    # Verify other fields are preserved
    assert response_data["job"] == "test-job-789"
    assert response_data["workspace"] == "default"
    assert response_data["artifact_storage_type"] == "fileset"
    assert response_data["artifact_url"] == "default/test-fileset#test_result"

    # Verify the jobs client was called correctly
    mock_jobs.get_job_result.assert_called_once_with(name="test_result", job="test-job-789", workspace="default")


def test_get_job_result_download_url_format(mock_service_with_job_routes):
    """Test that download_url has the correct format with special characters in result name."""
    app, mock_jobs = mock_service_with_job_routes
    client = TestClient(app)

    # Create mock result with a result name that has underscores
    mock_result = HelixJobResultResponse(
        name="evaluation_report_v1",
        job="test-job-abc",
        workspace="default",
        artifact_storage_type=FileStorageType.FILESET,
        artifact_url="default/test-fileset#evaluation_report_v1",
    )

    mock_jobs.get_job_result = AsyncMock(return_value=_resp(mock_result))

    # Make the request
    response = client.get("/v2/workspaces/default/test/jobs/test-job-abc/results/evaluation_report_v1")

    # Verify the response
    assert response.status_code == 200
    response_data = response.json()

    # Verify download_url format
    assert response_data["name"] == "evaluation_report_v1"
    assert "download_url" in response_data
    assert (
        response_data["download_url"]
        == "http://testserver/v2/workspaces/default/test/jobs/test-job-abc/results/evaluation_report_v1/download"
    )

    # Verify the jobs client was called correctly
    mock_jobs.get_job_result.assert_called_once_with(
        name="evaluation_report_v1", job="test-job-abc", workspace="default"
    )


def test_list_job_results_download_url_different_job_ids(mock_service_with_job_routes):
    """Test that download_url correctly reflects different job IDs."""
    app, mock_jobs = mock_service_with_job_routes
    client = TestClient(app)

    # Test with different job IDs
    job_ids = ["job-001", "job-002", "job-with-dashes"]

    for job_id in job_ids:
        mock_result = HelixJobResultResponse(
            name="output",
            job=job_id,
            workspace="default",
            artifact_storage_type=FileStorageType.FILESET,
            artifact_url=f"default/test-fileset#{job_id}/output",
        )

        mock_jobs.list_job_results = AsyncMock(return_value=_resp(HelixJobListResultResponse(data=[mock_result])))

        # Make the request
        response = client.get(f"/v2/workspaces/default/test/jobs/{job_id}/results")

        # Verify the response
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data["data"]) == 1
        assert (
            response_data["data"][0]["download_url"]
            == f"http://testserver/v2/workspaces/default/test/jobs/{job_id}/results/output/download"
        )

        # Reset mock for next iteration
        mock_jobs.reset_mock()


def test_get_job_result_not_found(mock_service_with_job_routes):
    """Test that get_job_result handles not found errors correctly."""
    app, mock_jobs = mock_service_with_job_routes
    client = TestClient(app)

    # The typed client raises a NemoHTTPError subclass on non-2xx responses.
    mock_jobs.get_job_result = AsyncMock(side_effect=_client_error(ClientNotFoundError, 404, "Result not found"))

    # Make the request
    response = client.get("/v2/workspaces/default/test/jobs/nonexistent-job/results/nonexistent-result")

    # Verify the response
    assert response.status_code == 404
    response_data = response.json()
    assert "Result not found" in response_data["detail"]

    # Verify the jobs client was called
    mock_jobs.get_job_result.assert_called_once_with(
        name="nonexistent-result", job="nonexistent-job", workspace="default"
    )


# Tests for entity_client injection


def test_accepts_entity_client_detects_parameter():
    """Test _accepts_entity_client correctly detects entity_client/entities_client parameters."""

    def with_entity_client(config, entity_client):
        pass

    def with_entities_client(config, entities_client):
        pass

    def without_entity_client(config):
        pass

    assert _accepts_entity_client(with_entity_client) is True
    assert _accepts_entity_client(with_entities_client) is True
    assert _accepts_entity_client(without_entity_client) is False


def test_validate_basemodel_or_union_valid_basemodel():
    """Test that _validate_basemodel_or_union accepts a single BaseModel."""
    # Should not raise
    _validate_basemodel_or_union(FooJobConfig, "test_param")


def test_validate_basemodel_or_union_valid_union():
    """Test that _validate_basemodel_or_union accepts Union of BaseModel subclasses."""

    class ConfigA(BaseModel):
        value: str

    class ConfigB(BaseModel):
        count: int

    union_type = ConfigA | ConfigB

    # Should not raise
    _validate_basemodel_or_union(union_type, "test_param")


def test_validate_basemodel_or_union_invalid_union_with_primitives():
    """Test that _validate_basemodel_or_union rejects Union with non-BaseModel types."""
    invalid_union = str | int

    with pytest.raises(ValueError) as exc_info:
        _validate_basemodel_or_union(invalid_union, "test_param")

    assert "test_param must be a BaseModel or a Union of BaseModel subclasses" in str(exc_info.value)


def test_validate_basemodel_or_union_invalid_mixed_union():
    """Test that _validate_basemodel_or_union rejects Union mixing BaseModel and primitives."""
    invalid_union = FooJobConfig | str

    with pytest.raises(ValueError) as exc_info:
        _validate_basemodel_or_union(invalid_union, "test_param")

    assert "test_param must be a BaseModel or a Union of BaseModel subclasses" in str(exc_info.value)


def test_validate_basemodel_or_union_error_message_includes_param_name():
    """Test that error messages include the parameter name for clarity."""
    invalid_union = str | int

    with pytest.raises(ValueError) as exc_info:
        _validate_basemodel_or_union(invalid_union, "custom_parameter")

    assert "custom_parameter" in str(exc_info.value)


def test_is_basemodel_union_returns_false_for_non_union_type():
    """Test that _is_basemodel_union returns False when given a non-union type."""
    assert _is_basemodel_union(str) is False
    assert _is_basemodel_union(int) is False
    assert _is_basemodel_union(FooJobConfig) is False


def test_validate_basemodel_or_union_rejects_plain_non_basemodel():
    """Test that _validate_basemodel_or_union raises ValueError for a plain non-BaseModel, non-union type."""
    with pytest.raises(ValueError) as exc_info:
        _validate_basemodel_or_union(str, "test_param")

    assert "test_param must be a BaseModel or a Union of BaseModel subclasses" in str(exc_info.value)
    assert "got <class 'str'>" in str(exc_info.value)


def test_validate_and_resolve_job_output_none_defaults_to_job_config():
    """Test that job_output=None defaults to job_input (backward compatible)."""
    resolved_output, resolved_transformer = _validate_and_resolve_job_output(
        job_output=None,
        job_input=FooJobConfig,
        input_to_output=None,
    )

    assert resolved_output is FooJobConfig
    assert resolved_transformer is None


def test_validate_and_resolve_job_output_with_valid_output():
    """Test that valid job_output with input_to_output returns correctly."""

    class OutputConfig(BaseModel):
        result: str

    def transformer(input_spec, workspace, entity_client, job_name, sdk):
        return OutputConfig(result="transformed")

    resolved_output, resolved_transformer = _validate_and_resolve_job_output(
        job_output=OutputConfig,
        job_input=FooJobConfig,
        input_to_output=transformer,
    )

    assert resolved_output is OutputConfig
    assert resolved_transformer is transformer


def test_validate_and_resolve_job_output_union_type():
    """Test that job_output as Union of BaseModels is accepted."""

    class ConfigA(BaseModel):
        value: str

    class ConfigB(BaseModel):
        count: int

    union_output = ConfigA | ConfigB

    def transformer(input_spec, workspace, entity_client, job_name, sdk):
        return ConfigA(value="test")

    resolved_output, resolved_transformer = _validate_and_resolve_job_output(
        job_output=union_output,
        job_input=FooJobConfig,
        input_to_output=transformer,
    )

    assert resolved_output is union_output
    assert resolved_transformer is transformer


def test_validate_and_resolve_job_output_missing_transformer():
    """Test that job_output without input_to_output raises ValueError."""

    class OutputConfig(BaseModel):
        result: str

    with pytest.raises(ValueError) as exc_info:
        _validate_and_resolve_job_output(
            job_output=OutputConfig,
            job_input=FooJobConfig,
            input_to_output=None,
        )

    assert "input_to_output parameter is required when job_output is provided" in str(exc_info.value)
    # Verify typo is fixed
    assert "parameter" in str(exc_info.value)
    assert "paraemeter" not in str(exc_info.value)


def test_validate_and_resolve_job_output_invalid_union():
    """Test that invalid job_output Union raises ValueError."""
    invalid_union = str | int

    def transformer(input_spec, workspace, entity_client, job_name, sdk):
        return "invalid"

    with pytest.raises(ValueError) as exc_info:
        _validate_and_resolve_job_output(
            job_output=invalid_union,
            job_input=FooJobConfig,
            input_to_output=transformer,
        )

    assert "job_output must be a BaseModel or a Union of BaseModel subclasses" in str(exc_info.value)


@pytest.mark.parametrize(
    "job_output,input_to_output,expected_output,expect_transformer_none",
    [
        # None defaults to job_config
        (None, None, FooJobConfig, True),
        # Valid output with transformer
        (FooJobConfig, lambda *args: FooJobConfig(foo="x", bar=1), FooJobConfig, False),
    ],
    ids=["default_none", "valid_with_transformer"],
)
def test_validate_and_resolve_job_output_parametrized(
    job_output, input_to_output, expected_output, expect_transformer_none
):
    """Parametrized test for common job_output scenarios."""
    resolved_output, resolved_transformer = _validate_and_resolve_job_output(
        job_output=job_output,
        job_input=FooJobConfig,
        input_to_output=input_to_output,
    )

    assert resolved_output is expected_output
    if expect_transformer_none:
        assert resolved_transformer is None
    else:
        assert resolved_transformer is input_to_output


def test_validate_and_resolve_job_output_async_transformer():
    """Test that async input_to_output transformer is accepted."""

    class OutputConfig(BaseModel):
        result: str

    async def async_transformer(input_spec, workspace, entity_client, job_name, sdk):
        return OutputConfig(result="async_transformed")

    # Should not raise (validation happens at call time, not here)
    resolved_output, resolved_transformer = _validate_and_resolve_job_output(
        job_output=OutputConfig,
        job_input=FooJobConfig,
        input_to_output=async_transformer,
    )

    assert resolved_output is OutputConfig
    assert resolved_transformer is async_transformer


def test_validate_and_resolve_job_output_transformer_without_output():
    """Test that input_to_output without job_output raises ValueError."""

    def transformer(input_spec, workspace, entity_client, job_name, sdk):
        return input_spec

    with pytest.raises(ValueError, match="input_to_output parameter must not be provided"):
        _validate_and_resolve_job_output(
            job_output=None,
            job_input=FooJobConfig,
            input_to_output=transformer,
        )


def test_create_job_injects_workspace_and_entity_client():
    """Test that compiler receives workspace and entity_client."""
    from nhx.common.service.dependencies import get_sdk_client

    mock_sdk = MagicMock()
    mock_entity_client = MagicMock()
    received_workspace = None
    received_entity_client = None

    def compiler(
        workspace: str, input_spec: FooJobConfig, output_spec: FooJobConfig, entity_client, job_name: str | None, sdk
    ) -> HelixJobSpec:
        nonlocal received_workspace
        received_workspace = workspace
        nonlocal received_entity_client
        received_entity_client = entity_client
        return HelixJobSpec(
            steps=[
                HelixJobStep(
                    name="test_step",
                    executor=CPUExecutionProviderSpec(
                        provider="cpu",
                        profile="default",
                        container=ContainerSpec(image="test_image"),
                    ),
                    config={"foo": output_spec.foo, "bar": output_spec.bar},
                )
            ]
        )

    router = job_route_factory(
        service_name="test_service",
        job_type="TestJob",
        job_input=FooJobConfig,
        platform_job_config_compiler=compiler,
    )

    app = FastAPI()
    app.include_router(router, prefix="/v2/workspaces/{workspace}/test")
    app.dependency_overrides[get_sdk_client] = lambda: mock_sdk
    app.dependency_overrides[get_entity_client] = lambda: mock_entity_client

    mock_jobs = MagicMock()
    mock_jobs.create_job = AsyncMock(return_value=_resp(create_mock_platform_job("test-job-123", "pending")))

    client = TestClient(app)
    with patch("nemo_helix_plugin.jobs.api_factory.client_from_platform", return_value=mock_jobs):
        response = client.post("/v2/workspaces/default/test/jobs", json={"spec": {"foo": "test", "bar": 42}})

    assert response.status_code == 201
    assert received_workspace == "default"
    assert received_entity_client is mock_entity_client


def test_sync_compiler_is_called_correctly():
    """Test that sync compilers are called and produce correct results."""

    mock_entity_client = MagicMock()
    compiler_called = False

    def sync_compiler(
        workspace: str, input_spec: FooJobConfig, output_spec: FooJobConfig, entity_client, job_name: str | None, sdk
    ) -> HelixJobSpec:
        nonlocal compiler_called
        compiler_called = True
        return HelixJobSpec(
            steps=[
                HelixJobStep(
                    name="test_step",
                    executor=CPUExecutionProviderSpec(
                        provider="cpu",
                        profile="default",
                        container=ContainerSpec(image="test_image"),
                    ),
                    config={"foo": output_spec.foo, "bar": output_spec.bar},
                )
            ]
        )

    router = job_route_factory(
        service_name="test_service",
        job_type="TestJob",
        job_input=FooJobConfig,
        platform_job_config_compiler=sync_compiler,
    )

    app = FastAPI()
    app.include_router(router, prefix="/v2/workspaces/{workspace}/test")

    mock_sdk = MagicMock()
    mock_jobs = MagicMock()
    mock_jobs.create_job = AsyncMock(return_value=_resp(create_mock_platform_job("test-job-123", "pending")))

    app.dependency_overrides[get_sdk_client] = lambda: mock_sdk
    app.dependency_overrides[get_entity_client] = lambda: mock_entity_client

    client = TestClient(app)
    with patch("nemo_helix_plugin.jobs.api_factory.client_from_platform", return_value=mock_jobs):
        response = client.post("/v2/workspaces/default/test/jobs", json={"spec": {"foo": "test", "bar": 42}})

    assert response.status_code == 201
    assert compiler_called, "Sync compiler should have been called"


class TestUnwrapAnnotatedSchema:
    """Tests for the _unwrap_annotated_schema helper function."""

    def test_plain_basemodel(self):
        """Test that a plain BaseModel class is returned unchanged."""
        result = _unwrap_annotated_schema(FooJobConfig)
        assert result is FooJobConfig

    def test_annotated_basemodel(self):
        """Test that an Annotated BaseModel is unwrapped to the underlying type."""

        class SomeMetadata:
            pass

        annotated_type = Annotated[FooJobConfig, SomeMetadata()]
        result = _unwrap_annotated_schema(annotated_type)
        assert result is FooJobConfig

    def test_annotated_with_multiple_metadata(self):
        """Test that Annotated with multiple metadata items returns the first type argument."""

        class Meta1:
            pass

        class Meta2:
            pass

        annotated_type = Annotated[FooJobConfig, Meta1(), Meta2()]
        result = _unwrap_annotated_schema(annotated_type)
        assert result is FooJobConfig

    def test_primitive_type(self):
        """Test that primitive types are returned unchanged."""
        assert _unwrap_annotated_schema(str) is str
        assert _unwrap_annotated_schema(int) is int
        assert _unwrap_annotated_schema(float) is float

    def test_annotated_primitive(self):
        """Test that Annotated primitives are unwrapped correctly."""
        annotated_str = Annotated[str, "some_metadata"]
        result = _unwrap_annotated_schema(annotated_str)
        assert result is str

    def test_union_type(self):
        """Test that Union types are returned unchanged (no __metadata__)."""

        class ConfigA(BaseModel):
            value: str

        class ConfigB(BaseModel):
            count: int

        union_type = ConfigA | ConfigB
        result = _unwrap_annotated_schema(union_type)
        assert result is union_type

    def test_annotated_union(self):
        """Test that Annotated Union types are unwrapped correctly."""

        class ConfigA(BaseModel):
            value: str

        class ConfigB(BaseModel):
            count: int

        union_type = ConfigA | ConfigB
        annotated_union = Annotated[union_type, "metadata"]
        result = _unwrap_annotated_schema(annotated_union)
        assert result == union_type

    def test_none_value(self):
        """Test that None is returned unchanged."""
        result = _unwrap_annotated_schema(None)
        assert result is None

    def test_instance(self):
        """Test that object instances (not types) are returned unchanged."""
        instance = FooJobConfig(foo="test", bar=42)
        result = _unwrap_annotated_schema(instance)
        assert result is instance

    @pytest.mark.parametrize(
        "input_obj,expected_output",
        [
            (str, str),
            (int, int),
            (list, list),
            (dict, dict),
            (FooJobConfig, FooJobConfig),
        ],
        ids=["str", "int", "list", "dict", "basemodel"],
    )
    def test_non_annotated_types(self, input_obj, expected_output):
        """Parametrized test for various non-annotated types."""
        result = _unwrap_annotated_schema(input_obj)
        assert result is expected_output


class TestResolveJobName:
    """Tests for _resolve_job_name helper."""

    def test_user_provided_name(self):
        """User-provided name takes priority over generator."""
        result = _resolve_job_name("my-job", lambda: "generated-name")
        assert result == "my-job"

    def test_generated_name(self):
        """Generator is called when no user-provided name."""
        result = _resolve_job_name(None, lambda: "generated-name")
        assert result == "generated-name"

    def test_no_name_no_generator(self):
        """Returns None when neither name nor generator is available."""
        result = _resolve_job_name(None, None)
        assert result is None

    def test_user_name_wins_over_generator(self):
        """User name is returned even when generator is provided."""
        generator = MagicMock(return_value="should-not-be-used")
        result = _resolve_job_name("user-name", generator)
        assert result == "user-name"
        generator.assert_not_called()


class TestTransformInputToOutput:
    """Tests for _transform_input_to_output helper."""

    @pytest.mark.anyio
    async def test_no_transformer_returns_spec_unchanged(self):
        """When input_to_output is None, returns spec as-is."""
        spec = FooJobConfig(foo="hello", bar=1)
        result = await _transform_input_to_output(None, spec, "ws", MagicMock(), "name", "svc", MagicMock())
        assert result is spec

    @pytest.mark.anyio
    async def test_sync_transformer(self):
        """Sync transformer is called and result returned."""
        spec = FooJobConfig(foo="hello", bar=1)
        output = FooJobConfig(foo="transformed", bar=2)

        def transformer(s, workspace, entity_client, job_name, sdk):
            return output

        result = await _transform_input_to_output(transformer, spec, "ws", MagicMock(), "name", "svc", MagicMock())
        assert result is output

    @pytest.mark.anyio
    async def test_async_transformer(self):
        """Async transformer is awaited and result returned."""
        spec = FooJobConfig(foo="hello", bar=1)
        output = FooJobConfig(foo="transformed", bar=2)

        async def transformer(s, workspace, entity_client, job_name, sdk):
            return output

        result = await _transform_input_to_output(transformer, spec, "ws", MagicMock(), "name", "svc", MagicMock())
        assert result is output

    @pytest.mark.anyio
    async def test_transformer_exception_becomes_422(self):
        """Transformer exception is wrapped in HTTPException 422."""
        from fastapi import HTTPException

        def bad_transformer(s, workspace, entity_client, job_name, sdk):
            raise ValueError("bad input")

        with pytest.raises(HTTPException) as exc_info:
            await _transform_input_to_output(
                bad_transformer, FooJobConfig(foo="a", bar=1), "ws", MagicMock(), "name", "my_svc", MagicMock()
            )
        assert exc_info.value.status_code == 422
        assert "my_svc" in exc_info.value.detail
        assert "bad input" in exc_info.value.detail


class TestCompileHelixSpec:
    """Tests for _compile_platform_spec helper."""

    def _make_platform_spec(self, output_spec: FooJobConfig) -> HelixJobSpec:
        return HelixJobSpec(
            steps=[
                HelixJobStep(
                    name="step",
                    executor=CPUExecutionProviderSpec(
                        provider="cpu",
                        profile="default",
                        container=ContainerSpec(image="img"),
                    ),
                    config={"foo": output_spec.foo, "bar": output_spec.bar},
                )
            ]
        )

    @pytest.mark.anyio
    async def test_sync_compiler(self):
        """Sync compiler is called and validated result returned."""
        spec = FooJobConfig(foo="a", bar=1)
        expected = self._make_platform_spec(spec)

        def compiler(workspace, input_spec, output_spec, entity_client, job_name, sdk):
            return expected

        result = await _compile_platform_spec(
            compiler, "ws", spec, spec, MagicMock(), "name", "svc", MagicMock(), None, None
        )
        assert result is expected

    @pytest.mark.anyio
    async def test_async_compiler(self):
        """Async compiler is awaited and validated result returned."""
        spec = FooJobConfig(foo="a", bar=1)
        expected = self._make_platform_spec(spec)

        async def compiler(workspace, input_spec, output_spec, entity_client, job_name, sdk):
            return expected

        result = await _compile_platform_spec(
            compiler, "ws", spec, spec, MagicMock(), "name", "svc", MagicMock(), None, None
        )
        assert result is expected

    @pytest.mark.anyio
    async def test_compilation_error_becomes_422(self):
        """HelixJobCompilationError is wrapped in HTTPException 422."""
        from fastapi import HTTPException

        def bad_compiler(workspace, input_spec, output_spec, entity_client, job_name, sdk):
            raise HelixJobCompilationError("missing field")

        spec = FooJobConfig(foo="a", bar=1)
        with pytest.raises(HTTPException) as exc_info:
            await _compile_platform_spec(
                bad_compiler, "ws", spec, spec, MagicMock(), "name", "my_svc", MagicMock(), None, None
            )
        assert exc_info.value.status_code == 422
        assert "my_svc" in exc_info.value.detail
        assert "missing field" in exc_info.value.detail

    @pytest.mark.anyio
    async def test_dependency_unavailable_error_becomes_503(self):
        """A compiler dependency outage is reported as a service outage."""
        from fastapi import HTTPException

        def unavailable_compiler(workspace, input_spec, output_spec, entity_client, job_name, sdk):
            raise HelixJobDependencyUnavailableError(
                "The Jobs service is temporarily unavailable. Retry the submission."
            )

        spec = FooJobConfig(foo="a", bar=1)
        with pytest.raises(HTTPException) as exc_info:
            await _compile_platform_spec(
                unavailable_compiler,
                "ws",
                spec,
                spec,
                MagicMock(),
                "name",
                "my_svc",
                MagicMock(),
                None,
                None,
            )
        assert exc_info.value.status_code == 503
        assert "my_svc" in exc_info.value.detail
        assert "temporarily unavailable" in exc_info.value.detail
        assert "Retry the submission" in exc_info.value.detail

    @pytest.mark.anyio
    async def test_validate_job_spec_is_called(self):
        """_validate_job_spec is invoked on the compiled result (catches non-serializable config)."""
        from fastapi import HTTPException

        def compiler_bad_config(workspace, input_spec, output_spec, entity_client, job_name, sdk):
            # Return a spec whose step config is not JSON serializable
            return HelixJobSpec(
                steps=[
                    HelixJobStep(
                        name="step",
                        executor=CPUExecutionProviderSpec(
                            provider="cpu",
                            profile="default",
                            container=ContainerSpec(image="img"),
                        ),
                        config={"bad": object()},  # not JSON serializable
                    )
                ]
            )

        spec = FooJobConfig(foo="a", bar=1)
        with pytest.raises(HTTPException) as exc_info:
            await _compile_platform_spec(
                compiler_bad_config, "ws", spec, spec, MagicMock(), "name", "svc", MagicMock(), None, None
            )
        assert exc_info.value.status_code == 422
        assert "not json serializable" in exc_info.value.detail.lower()


class TestSDKExceptionHandling:
    """Tests that NemoHTTPError from typed-client calls is handled by the global exception handler.

    The api_factory endpoints should NOT catch the client errors themselves — the global
    nemo_client_error_handler registered on the app extracts the detail cleanly from
    the exception, avoiding an ugly stringified format like:
        {"detail": "Error code: 409 - {'detail': 'Job already exists'}"}
    """

    def test_create_job_duplicate_name_returns_clean_409(self, mock_service_with_job_routes):
        app, mock_jobs = mock_service_with_job_routes
        client = TestClient(app, raise_server_exceptions=False)

        error_detail = "Unable to create job: Job with name 'my-job' already exists in workspace 'default'."
        mock_jobs.create_job = AsyncMock(side_effect=_client_error(ClientConflictError, 409, error_detail))

        response = client.post(
            "/v2/workspaces/default/test/jobs",
            json={"spec": {"foo": "test", "bar": 1}},
        )

        assert response.status_code == 409
        assert response.json()["detail"] == error_detail

    def test_get_job_not_found_returns_clean_404(self, mock_service_with_job_routes):
        app, mock_jobs = mock_service_with_job_routes
        client = TestClient(app, raise_server_exceptions=False)

        error_detail = "Job 'nonexistent' not found in workspace 'default'."
        mock_jobs.get_job = AsyncMock(side_effect=_client_error(ClientNotFoundError, 404, error_detail))

        response = client.get("/v2/workspaces/default/test/jobs/nonexistent")

        assert response.status_code == 404
        assert response.json()["detail"] == error_detail

    def test_delete_job_not_found_returns_clean_404(self, mock_service_with_job_routes):
        app, mock_jobs = mock_service_with_job_routes
        client = TestClient(app, raise_server_exceptions=False)

        error_detail = "Job 'nonexistent' not found."
        mock_jobs.delete_job = AsyncMock(side_effect=_client_error(ClientNotFoundError, 404, error_detail))

        response = client.delete("/v2/workspaces/default/test/jobs/nonexistent")

        assert response.status_code == 404
        assert response.json()["detail"] == error_detail

    def test_cancel_job_bad_state_returns_clean_409(self, mock_service_with_job_routes):
        app, mock_jobs = mock_service_with_job_routes
        client = TestClient(app, raise_server_exceptions=False)

        error_detail = "Job 'my-job' cannot be cancelled in its current state."
        mock_jobs.cancel_job = AsyncMock(side_effect=_client_error(ClientConflictError, 409, error_detail))

        response = client.post("/v2/workspaces/default/test/jobs/my-job/cancel")

        assert response.status_code == 409
        assert response.json()["detail"] == error_detail

    def test_pause_job_returns_clean_error(self, mock_service_with_job_routes):
        app, mock_jobs = mock_service_with_job_routes
        client = TestClient(app, raise_server_exceptions=False)

        error_detail = "Job 'my-job' cannot be paused."
        mock_jobs.pause_job = AsyncMock(side_effect=_client_error(ClientConflictError, 409, error_detail))

        response = client.post("/v2/workspaces/default/test/jobs/my-job/pause")

        assert response.status_code == 409
        assert response.json()["detail"] == error_detail

    def test_resume_job_returns_clean_error(self, mock_service_with_job_routes):
        app, mock_jobs = mock_service_with_job_routes
        client = TestClient(app, raise_server_exceptions=False)

        error_detail = "Job 'my-job' cannot be resumed."
        mock_jobs.resume_job = AsyncMock(side_effect=_client_error(ClientConflictError, 409, error_detail))

        response = client.post("/v2/workspaces/default/test/jobs/my-job/resume")

        assert response.status_code == 409
        assert response.json()["detail"] == error_detail

    def test_get_logs_returns_clean_error(self, mock_service_with_job_routes):
        app, mock_jobs = mock_service_with_job_routes
        client = TestClient(app, raise_server_exceptions=False)

        error_detail = "Job 'nonexistent' not found."
        mock_jobs.list_job_logs = AsyncMock(side_effect=_client_error(ClientNotFoundError, 404, error_detail))

        response = client.get("/v2/workspaces/default/test/jobs/nonexistent/logs")

        assert response.status_code == 404
        assert response.json()["detail"] == error_detail

    def test_list_results_returns_clean_error(self, mock_service_with_job_routes):
        app, mock_jobs = mock_service_with_job_routes
        client = TestClient(app, raise_server_exceptions=False)

        error_detail = "Job 'nonexistent' not found."
        mock_jobs.list_job_results = AsyncMock(side_effect=_client_error(ClientNotFoundError, 404, error_detail))

        response = client.get("/v2/workspaces/default/test/jobs/nonexistent/results")

        assert response.status_code == 404
        assert response.json()["detail"] == error_detail

    def test_error_detail_is_not_stringified(self, mock_service_with_job_routes):
        """Verify the response detail is the clean string, not a stringified repr like
        "Error code: 409 - {'detail': '...'}" which was the old behavior."""
        app, mock_jobs = mock_service_with_job_routes
        client = TestClient(app, raise_server_exceptions=False)

        error_detail = "Job already exists."
        mock_jobs.create_job = AsyncMock(side_effect=_client_error(ClientConflictError, 409, error_detail))

        response = client.post(
            "/v2/workspaces/default/test/jobs",
            json={"spec": {"foo": "test", "bar": 1}},
        )

        detail = response.json()["detail"]
        assert detail == error_detail
        assert "Error code:" not in detail
