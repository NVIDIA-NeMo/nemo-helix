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
from nemo_platform_plugin.dependencies import get_entity_client, get_sdk_client
from nemo_platform_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    EnvironmentVariable,
    GPUExecutionProviderSpec,
    JobSecret,
    PlatformJobSpec,
    PlatformJobStep,
    job_route_factory,
)
from nemo_platform_plugin.jobs.docker import spec_has_gpu_step
from nemo_platform_plugin.jobs.types import CreatePlatformJobRequest
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
) -> PlatformJobSpec:
    del workspace, original_spec, transformed_spec, entity_client, job_name, sdk
    return PlatformJobSpec(
        steps=[
            PlatformJobStep(
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
        workspace="default",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
        spec=spec,
        status="created",
        status_details=None,
        error_details=None,
        ownership=None,
        custom_fields=None,
    )
    response = MagicMock()
    response.data.return_value = job
    return response


class _RecordingJobsClient:
    def __init__(self) -> None:
        self.created_body: CreatePlatformJobRequest | None = None

    async def create_job(self, *, workspace: str, body: CreatePlatformJobRequest) -> MagicMock:
        del workspace
        self.created_body = body
        expected_spec = {"blob": base64.b64encode(_PICKLE_BYTES).decode("ascii")}
        return _mock_create_response(expected_spec)


def test_job_spec_models_support_mapping_style_access() -> None:
    env = EnvironmentVariable(name="ENV_VAR", value="value")
    executor = CPUExecutionProviderSpec(
        provider="cpu",
        profile="default",
        container=ContainerSpec(image="test"),
    )
    step = PlatformJobStep(
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
    step = PlatformJobStep(
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
    assert spec_has_gpu_step(PlatformJobSpec(steps=[step])) is True


def test_job_spec_includes_inline_secrets() -> None:
    spec = PlatformJobSpec(
        steps=[
            PlatformJobStep(
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
    app.dependency_overrides[get_sdk_client] = lambda: MagicMock()
    app.dependency_overrides[get_entity_client] = lambda: MagicMock()

    mock_jobs = _RecordingJobsClient()

    client = TestClient(app)
    with patch(
        "nemo_platform_plugin.jobs.api_factory.client_from_platform",
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
    app.dependency_overrides[get_sdk_client] = lambda: MagicMock()
    app.dependency_overrides[get_entity_client] = lambda: MagicMock()

    mock_jobs = _RecordingJobsClient()

    client = TestClient(app)
    with patch(
        "nemo_platform_plugin.jobs.api_factory.client_from_platform",
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
    ) -> PlatformJobSpec:
        del workspace, original_spec, transformed_spec, entity_client, job_name, sdk
        seen.update(profile=profile, options=options)
        return PlatformJobSpec(
            steps=[
                PlatformJobStep(
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
    app.dependency_overrides[get_sdk_client] = lambda: MagicMock()
    app.dependency_overrides[get_entity_client] = lambda: MagicMock()

    async def _create_job(*, workspace: str, body: object) -> MagicMock:
        del workspace, body
        return _mock_create_response({"label": "metric"})

    mock_jobs = SimpleNamespace(create_job=_create_job)

    client = TestClient(app)
    with patch(
        "nemo_platform_plugin.jobs.api_factory.client_from_platform",
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
    app.dependency_overrides[get_sdk_client] = lambda: MagicMock()
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
