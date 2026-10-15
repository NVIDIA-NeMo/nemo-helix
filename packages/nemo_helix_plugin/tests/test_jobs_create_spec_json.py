# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression: plugin create_job forwards transformed specs in JSON mode."""

from __future__ import annotations

import base64
import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from nemo_helix_plugin.dependencies import get_entity_client, get_nemo_client
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    EnvironmentVariable,
    GPUExecutionProviderSpec,
    HelixJobSpec,
    HelixJobStep,
    JobSecret,
    job_route_factory,
)
from nemo_helix_plugin.jobs.docker import spec_has_gpu_step
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest, UpdateHelixJobRequest
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.testclient import TestClient

# Non-UTF-8 bytes like a cloudpickle metric bundle blob.
_PICKLE_BYTES = b"\x80\x04\x95\x00\x00\x00"


class _InputSpec(BaseModel):
    label: str = "metric"


class _OutputSpec(BaseModel):
    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")

    blob: bytes


def _to_output(
    input_spec: _InputSpec,
    workspace: str,
    entity_client: object,
    job_name: str | None,
    sdk: object,
) -> _OutputSpec:
    del input_spec, workspace, entity_client, job_name, sdk
    return _OutputSpec(blob=_PICKLE_BYTES)


def _compiler(
    workspace: str,
    original_spec: _InputSpec,
    transformed_spec: _OutputSpec,
    entity_client: object,
    job_name: str | None,
    sdk: object,
) -> HelixJobSpec:
    del workspace, original_spec, transformed_spec, entity_client, job_name, sdk
    return HelixJobSpec(
        steps=[
            HelixJobStep(
                name="step",
                executor=CPUExecutionProviderSpec(
                    provider="cpu",
                    profile="default",
                    container=ContainerSpec(image="test"),
                ),
                config={},
            )
        ]
    )


def _mock_create_response(spec: dict[str, object]) -> MagicMock:
    job = SimpleNamespace(
        id="job-1",
        name="test-job",
        description=None,
        project="project-a",
        workspace="default",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
        spec=spec,
        status="created",
        status_details=None,
        error_details=None,
        ownership=None,
        custom_fields=None,
        attempt_id="attempt-1",
    )
    response = MagicMock()
    response.data.return_value = job
    return response


def _mock_attempt_response(spec: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        id="attempt-1",
        name="attempt-1",
        job="job-1",
        workspace="default",
        seq=0,
        spec=spec,
        status="created",
        status_details={},
        error_details=None,
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
        is_current=True,
    )


class _RecordingJobsClient:
    def __init__(self) -> None:
        self.created_body: CreateHelixJobRequest | None = None
        self.updated_body: UpdateHelixJobRequest | None = None
        self.list_attempts_call: dict[str, object] | None = None
        self.get_attempt_call: dict[str, object] | None = None

    async def create_job(self, *, workspace: str, body: CreateHelixJobRequest) -> MagicMock:
        del workspace
        self.created_body = body
        expected_spec = {"blob": base64.b64encode(_PICKLE_BYTES).decode("ascii")}
        return _mock_create_response(expected_spec)

    async def update_job(self, *, name: str, workspace: str, body: UpdateHelixJobRequest) -> MagicMock:
        del name, workspace
        self.updated_body = body
        return _mock_create_response(body.spec or {"blob": base64.b64encode(_PICKLE_BYTES).decode("ascii")})

    async def list_job_attempts(self, **kwargs: object) -> MagicMock:
        self.list_attempts_call = kwargs
        response = MagicMock()
        response.page.return_value = SimpleNamespace(
            items=[_mock_attempt_response({"blob": base64.b64encode(_PICKLE_BYTES).decode("ascii")})],
            metadata={
                "page": 1,
                "page_size": 25,
                "current_page_size": 1,
                "total_pages": 1,
                "total_results": 1,
            },
        )
        return response

    async def get_job_attempt(self, **kwargs: object) -> MagicMock:
        self.get_attempt_call = kwargs
        response = MagicMock()
        response.data.return_value = _mock_attempt_response({"blob": base64.b64encode(_PICKLE_BYTES).decode("ascii")})
        return response


def test_job_spec_models_support_mapping_style_access() -> None:
    env = EnvironmentVariable(name="ENV_VAR", value="value")
    executor = CPUExecutionProviderSpec(
        provider="cpu",
        profile="default",
        container=ContainerSpec(image="test"),
    )
    step = HelixJobStep(
        name="step",
        executor=executor,
        config={},
        environment=[env],
    )

    assert "executor" in step
    assert "resources" not in executor
    assert step["environment"] == [{"name": "ENV_VAR", "value": "value"}]
    step["executor"]["profile"] = "custom"
    assert step.executor.profile == "custom"


def test_job_spec_mapping_assignment_validates_executor_models() -> None:
    step = HelixJobStep(
        name="step",
        executor=CPUExecutionProviderSpec(
            provider="cpu",
            profile="default",
            container=ContainerSpec(image="test"),
        ),
        config={},
    )

    with pytest.raises(ValidationError):
        step["executor"] = {}

    step["executor"] = {
        "provider": "gpu",
        "profile": "default",
        "container": {"image": "gpu-image"},
    }

    assert isinstance(step.executor, GPUExecutionProviderSpec)
    assert spec_has_gpu_step(HelixJobSpec(steps=[step])) is True


def test_job_spec_includes_inline_secrets() -> None:
    spec = HelixJobSpec(
        steps=[
            HelixJobStep(
                name="step",
                executor=CPUExecutionProviderSpec(
                    provider="cpu",
                    profile="default",
                    container=ContainerSpec(image="test"),
                ),
            )
        ],
        secrets=[JobSecret(name="secret-name", value="secret-value")],
    )

    assert spec["secrets"] == [{"name": "secret-name", "value": "secret-value"}]
    assert spec.model_dump(mode="json", exclude_none=True)["secrets"] == [
        {"name": "secret-name", "value": "secret-value"}
    ]


def test_create_job_forwards_transformed_spec_in_json_mode() -> None:
    """Binary fields in transformed specs must be base64 strings, not raw bytes."""
    router = job_route_factory(
        service_name="widgets",
        job_type="Widget",
        job_input=_InputSpec,
        job_output=_OutputSpec,
        input_to_output=_to_output,
        platform_job_config_compiler=_compiler,
    )
    app = FastAPI()
    app.include_router(router, prefix="/apis/widgets/v2/workspaces/{workspace}")
    app.dependency_overrides[get_nemo_client] = lambda: MagicMock()
    app.dependency_overrides[get_entity_client] = lambda: MagicMock()

    mock_jobs = _RecordingJobsClient()

    client = TestClient(app)
    with patch(
        "nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client",
        return_value=mock_jobs,
    ):
        response = client.post("/apis/widgets/v2/workspaces/default/jobs", json={"spec": {"label": "metric"}})

    assert response.status_code == 201, response.text
    body = mock_jobs.created_body
    assert body is not None
    spec = body.spec
    blob = spec["blob"] if isinstance(spec, dict) else spec.blob
    assert isinstance(blob, str), f"expected base64 string, got {type(blob)}"
    assert blob == base64.b64encode(_PICKLE_BYTES).decode("ascii")
    # Must be JSON-serializable for the typed jobs client (raw bytes would 500).
    json.dumps({"spec": spec})
    assert response.json()["project"] == "project-a"
    assert response.json()["attempt_id"] == "attempt-1"
    assert "db_version" not in response.json()


def test_update_job_recompiles_typed_spec_and_sends_paired_core_specs() -> None:
    router = job_route_factory(
        service_name="widgets",
        job_type="Widget",
        job_input=_InputSpec,
        job_output=_OutputSpec,
        input_to_output=_to_output,
        platform_job_config_compiler=_compiler,
    )
    app = FastAPI()
    app.include_router(router, prefix="/apis/widgets/v2/workspaces/{workspace}")
    app.dependency_overrides[get_nemo_client] = lambda: MagicMock()
    app.dependency_overrides[get_entity_client] = lambda: MagicMock()
    mock_jobs = _RecordingJobsClient()

    client = TestClient(app)
    with patch(
        "nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client",
        return_value=mock_jobs,
    ):
        response = client.patch(
            "/apis/widgets/v2/workspaces/default/jobs/test-job",
            json={"spec": {"label": "updated"}},
        )

    assert response.status_code == 200, response.text
    body = mock_jobs.updated_body
    assert body is not None
    assert body.spec == {"blob": base64.b64encode(_PICKLE_BYTES).decode("ascii")}
    assert body.platform_spec is not None


def test_factory_attempt_routes_preserve_typed_specs_and_current_selector() -> None:
    router = job_route_factory(
        service_name="widgets",
        job_type="Widget",
        job_input=_InputSpec,
        job_output=_OutputSpec,
        input_to_output=_to_output,
        platform_job_config_compiler=_compiler,
    )
    app = FastAPI()
    app.include_router(router, prefix="/apis/widgets/v2/workspaces/{workspace}")
    app.dependency_overrides[get_nemo_client] = lambda: MagicMock()
    app.dependency_overrides[get_entity_client] = lambda: MagicMock()
    mock_jobs = _RecordingJobsClient()

    client = TestClient(app)
    with patch(
        "nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client",
        return_value=mock_jobs,
    ):
        listed = client.get("/apis/widgets/v2/workspaces/default/jobs/test-job/attempts")
        current = client.get("/apis/widgets/v2/workspaces/default/jobs/test-job/attempts/current")

    assert listed.status_code == 200, listed.text
    assert listed.json()["data"][0]["spec"]["blob"] == base64.b64encode(_PICKLE_BYTES).decode("ascii")
    assert mock_jobs.list_attempts_call == {
        "workspace": "default",
        "job": "test-job",
        "query_params": {"page": 1, "page_size": 25, "sort": "-seq"},
    }
    assert current.status_code == 200, current.text
    assert current.json()["is_current"] is True
    assert mock_jobs.get_attempt_call == {"workspace": "default", "job": "test-job", "attempt": "current"}


def test_factory_update_rejects_internal_version_without_mutation() -> None:
    router = job_route_factory(
        service_name="widgets",
        job_type="Widget",
        job_input=_InputSpec,
        job_output=_OutputSpec,
        input_to_output=_to_output,
        platform_job_config_compiler=_compiler,
    )
    app = FastAPI()
    app.include_router(router, prefix="/apis/widgets/v2/workspaces/{workspace}")
    app.dependency_overrides[get_nemo_client] = lambda: MagicMock()
    app.dependency_overrides[get_entity_client] = lambda: MagicMock()

    response = TestClient(app).patch(
        "/apis/widgets/v2/workspaces/default/jobs/test-job",
        json={"expected_db_version": 1},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project", "project-a"),
        ("ownership", {"user": "owner"}),
        ("custom_fields", {"key": "value"}),
        ("profile", "default"),
        ("options", {"backend": {}}),
    ],
)
def test_factory_update_rejects_removed_fields(field: str, value: object) -> None:
    router = job_route_factory(
        service_name="widgets",
        job_type="Widget",
        job_input=_InputSpec,
        job_output=_OutputSpec,
        input_to_output=_to_output,
        platform_job_config_compiler=_compiler,
    )
    app = FastAPI()
    app.include_router(router, prefix="/apis/widgets/v2/workspaces/{workspace}")
    app.dependency_overrides[get_nemo_client] = lambda: MagicMock()
    app.dependency_overrides[get_entity_client] = lambda: MagicMock()

    response = TestClient(app).patch(
        "/apis/widgets/v2/workspaces/default/jobs/test-job",
        json={"description": "ignored", field: value},
    )

    assert response.status_code == 422


def test_create_job_stamps_plugin_telemetry_custom_fields() -> None:
    router = job_route_factory(
        service_name="nemo-data-designer-plugin",
        job_type="Widget",
        job_input=_InputSpec,
        job_output=_OutputSpec,
        input_to_output=_to_output,
        platform_job_config_compiler=_compiler,
    )
    app = FastAPI()
    app.include_router(router, prefix="/apis/data-designer/v2/workspaces/{workspace}")
    app.dependency_overrides[get_nemo_client] = lambda: MagicMock()
    app.dependency_overrides[get_entity_client] = lambda: MagicMock()

    mock_jobs = _RecordingJobsClient()

    client = TestClient(app)
    with patch(
        "nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client",
        return_value=mock_jobs,
    ):
        response = client.post(
            "/apis/data-designer/v2/workspaces/default/jobs",
            json={
                "spec": {"label": "metric"},
                "custom_fields": {
                    "kept": "value",
                    "_nemo_telemetry": {
                        "session_id": "session-123",
                        "plugins": ["caller-controlled"],
                    },
                },
            },
        )

    assert response.status_code == 201, response.text
    body = mock_jobs.created_body
    assert body is not None
    assert body.custom_fields == {
        "kept": "value",
        "_nemo_telemetry": {
            "session_id": "session-123",
            "plugins": ["data-designer"],
        },
    }


def test_create_job_forwards_profile_and_options_to_compiler() -> None:
    """Submitter controls belong to the compiler, not Pydantic's ignored extras."""
    seen: dict[str, object] = {}

    async def _capturing_compiler(
        workspace: str,
        original_spec: _InputSpec,
        transformed_spec: _InputSpec,
        entity_client: object,
        job_name: str | None,
        sdk: object,
        profile: str | None = None,
        options: dict | None = None,
    ) -> HelixJobSpec:
        del workspace, original_spec, transformed_spec, entity_client, job_name, sdk
        seen.update(profile=profile, options=options)
        return HelixJobSpec(
            steps=[
                HelixJobStep(
                    name="step",
                    executor=CPUExecutionProviderSpec(
                        provider="cpu",
                        profile=profile or "default",
                        container=ContainerSpec(image="test"),
                    ),
                    config={},
                )
            ]
        )

    router = job_route_factory(
        service_name="widgets",
        job_type="Widget",
        job_input=_InputSpec,
        platform_job_config_compiler=_capturing_compiler,
    )
    app = FastAPI()
    app.include_router(router, prefix="/apis/widgets/v2/workspaces/{workspace}")
    app.dependency_overrides[get_nemo_client] = lambda: MagicMock()
    app.dependency_overrides[get_entity_client] = lambda: MagicMock()

    async def _create_job(*, workspace: str, body: object) -> MagicMock:
        del workspace, body
        return _mock_create_response({"label": "metric"})

    mock_jobs = SimpleNamespace(create_job=_create_job)

    client = TestClient(app)
    with patch(
        "nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client",
        return_value=mock_jobs,
    ):
        response = client.post(
            "/apis/widgets/v2/workspaces/default/jobs",
            json={
                "spec": {"label": "metric"},
                "profile": "research",
                "options": {"slurm": {"nodes": 4}},
            },
        )

    assert response.status_code == 201, response.text
    assert seen == {"profile": "research", "options": {"slurm": {"nodes": 4}}}


def test_create_job_rejects_unsupported_profile_and_options() -> None:
    router = job_route_factory(
        service_name="widgets",
        job_type="Widget",
        job_input=_InputSpec,
        platform_job_config_compiler=_compiler,
    )
    app = FastAPI()
    app.include_router(router, prefix="/apis/widgets/v2/workspaces/{workspace}")
    app.dependency_overrides[get_nemo_client] = lambda: MagicMock()
    app.dependency_overrides[get_entity_client] = lambda: MagicMock()

    client = TestClient(app)
    response = client.post(
        "/apis/widgets/v2/workspaces/default/jobs",
        json={
            "spec": {"label": "metric"},
            "profile": "research",
            "options": {"slurm": {"nodes": 4}},
        },
    )

    assert response.status_code == 422, response.text
    assert "does not support submit field(s): profile, options" in response.json()["detail"]
