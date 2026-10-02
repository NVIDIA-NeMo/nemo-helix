# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kind smoke e2e for models → deployments_plugin → plugin k8s backend.

Runs against a real cluster when ``NHX_BASE_URL`` / ``NHX_E2E_CLUSTER_URL`` is set
(the ``kind-cpu-smoke`` CI job). Uses a CPU-only generic container image with no
NGC credentials.
"""

from __future__ import annotations

import time
import uuid

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import (
    ContainerExecutorConfig,
    CreateModelDeploymentConfigRequest,
    CreateModelDeploymentRequest,
    Engine,
    ModelDeploymentConfigModelSpec,
)

# Kind smoke generic deployment image (python -m http.server). Keep in sync with
# .github/actions/setup-kind-cluster/action.yaml GENERIC_HTTP_* prepull vars.
GENERIC_HTTP_IMAGE = "docker.io/library/python"
GENERIC_HTTP_TAG = "3.12-alpine"

pytestmark = [
    pytest.mark.container_only,
    pytest.mark.timeout(900),
]


def _unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _deployment_diagnostic(models: ModelsClient, *, workspace: str, name: str, prefix: str) -> str:
    try:
        deployment = models.get_deployment(name=name, workspace=workspace).data()
    except NotFoundError:
        return f"{prefix}\nDeployment {name!r} not found."
    return (
        f"{prefix}\n"
        f"status={deployment.status!r}\n"
        f"status_message={deployment.status_message!r}\n"
        f"model_provider_id={deployment.model_provider_id!r}"
    )


def _wait_for_deployment_ready(
    models: ModelsClient,
    *,
    workspace: str,
    name: str,
    timeout_seconds: float = 600,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_status: str | None = None
    last_message: str | None = None

    while time.monotonic() < deadline:
        deployment = models.get_deployment(name=name, workspace=workspace).data()
        last_status = deployment.status
        last_message = deployment.status_message
        if deployment.status == "READY":
            assert deployment.model_provider_id is not None, _deployment_diagnostic(
                models,
                workspace=workspace,
                name=name,
                prefix="Deployment reached READY without model_provider_id",
            )
            return
        if deployment.status == "ERROR":
            pytest.fail(
                _deployment_diagnostic(
                    models,
                    workspace=workspace,
                    name=name,
                    prefix=f"Deployment {name!r} entered ERROR",
                )
            )
        time.sleep(2)

    pytest.fail(
        f"Deployment {name!r} did not reach READY within {timeout_seconds}s; "
        f"last status={last_status!r}, status_message={last_message!r}"
    )


def _wait_for_deployment_deleted(
    models: ModelsClient,
    *,
    workspace: str,
    name: str,
    timeout_seconds: float = 300,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_status: str | None = None

    while time.monotonic() < deadline:
        try:
            deployment = models.get_deployment(name=name, workspace=workspace).data()
            last_status = deployment.status
            if deployment.status == "ERROR":
                pytest.fail(
                    _deployment_diagnostic(
                        models,
                        workspace=workspace,
                        name=name,
                        prefix=f"Deployment {name!r} entered ERROR while waiting for deletion",
                    )
                )
        except NotFoundError:
            return
        time.sleep(2)

    pytest.fail(f"Deployment {name!r} was not deleted within {timeout_seconds}s; last status={last_status!r}")


def test_generic_model_deployment_lifecycle(client: NemoClient, workspace: str) -> None:
    """Create → READY → delete a generic CPU deployment on the plugin k8s backend."""
    models = ModelsClient.from_client(client)
    config_name = _unique_name("kind-generic-cfg")
    deployment_name = _unique_name("kind-generic-dep")

    models.create_deployment_config(
        workspace=workspace,
        body=CreateModelDeploymentConfigRequest(
            name=config_name,
            engine=Engine.GENERIC,
            model_spec=ModelDeploymentConfigModelSpec(),
            executor_config=ContainerExecutorConfig(
                gpu=0,
                image_name=GENERIC_HTTP_IMAGE,
                image_tag=GENERIC_HTTP_TAG,
                additional_args=["python3", "-m", "http.server", "8000"],
                health_check_path="/",
            ),
        ),
    )
    models.create_deployment(
        workspace=workspace,
        body=CreateModelDeploymentRequest(name=deployment_name, config=config_name),
    )

    try:
        _wait_for_deployment_ready(models, workspace=workspace, name=deployment_name)
    finally:
        try:
            models.delete_deployment(name=deployment_name, workspace=workspace)
        except NotFoundError:
            pass

    _wait_for_deployment_deleted(models, workspace=workspace, name=deployment_name)
