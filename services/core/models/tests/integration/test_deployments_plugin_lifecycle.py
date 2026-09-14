# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for models controller via deployments_plugin on real Docker."""

from __future__ import annotations

import uuid

import pytest
from docker.errors import NotFound
from nemo_deployments_plugin.backends.labels import container_name
from nemo_platform_plugin.client.errors import NotFoundError
from nmp.core.models.controllers.backends.deployments_plugin.naming import entity_names
from tenacity import retry, stop_after_delay, wait_fixed

import docker

from .conftest import (
    create_deployment,
    create_deployment_config,
    delete_deployment,
    get_deployment,
    get_provider,
)

try:
    docker.from_env().ping()
    _DOCKER_AVAILABLE = True
except Exception:
    _DOCKER_AVAILABLE = False

skip_without_docker = pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="Docker daemon not available")

pytestmark = [
    skip_without_docker,
    pytest.mark.xdist_group("nemo_deployments_docker_integration"),
]


def test_deployments_plugin_docker_lifecycle(controller_with_deployments_plugin, docker_client: docker.DockerClient):
    """Full stack: models → deployments_plugin entities → plugin docker backend.

    Tests: create → PENDING → READY → delete → cleanup
    Also verifies ModelProvider creation and deletion.
    """
    controller, _, models_client, mock_nim_image, ctx, reconcile = controller_with_deployments_plugin
    test_uuid = uuid.uuid4().hex[:8]
    config_name = f"test-plugin-lifecycle-{test_uuid}"
    deployment_name = f"test-plugin-lifecycle-{test_uuid}"
    workspace = "default"
    server_entity = entity_names(deployment_name).server
    server_container_name = container_name(workspace, server_entity)

    ctx.register_container(server_container_name)

    image_name, image_tag = mock_nim_image.rsplit(":", 1)
    create_deployment_config(
        models_client,
        name=config_name,
        workspace=workspace,
        engine="nim",
        model_spec={},
        executor_config={
            "gpu": 0,
            "image_name": image_name,
            "image_tag": image_tag,
        },
    )
    create_deployment(
        models_client,
        name=deployment_name,
        workspace=workspace,
        config=config_name,
    )

    @retry(stop=stop_after_delay(30), wait=wait_fixed(0.2), reraise=True)
    def wait_for_container_created():
        reconcile(controller)
        container = docker_client.containers.get(server_container_name)
        assert container.status in ["created", "running"], f"Unexpected status: {container.status}"
        return container

    container = wait_for_container_created()

    @retry(stop=stop_after_delay(15), wait=wait_fixed(0.2), reraise=True)
    def wait_for_container_running():
        container.reload()
        assert container.status == "running", f"Container not running: {container.status}"

    wait_for_container_running()

    @retry(stop=stop_after_delay(45), wait=wait_fixed(0.2), reraise=True)
    def wait_for_deployment_ready():
        reconcile(controller)
        dep = get_deployment(models_client, name=deployment_name, workspace=workspace)
        assert dep.status == "READY", f"Deployment not READY: {dep.status} ({dep.status_message})"
        return dep

    deployment = wait_for_deployment_ready()

    provider_id = deployment.model_provider_id
    assert provider_id is not None, "ModelProvider should be created when deployment becomes READY"
    provider_workspace, provider_name = provider_id.split("/")
    provider = get_provider(models_client, name=provider_name, workspace=provider_workspace)
    assert provider.host_url is not None
    assert provider.status == "READY"

    reconcile(controller)
    get_provider(models_client, name=provider_name, workspace=provider_workspace)

    delete_deployment(models_client, name=deployment_name, workspace=workspace)
    reconcile(controller)

    @retry(stop=stop_after_delay(30), wait=wait_fixed(0.2), reraise=True)
    def wait_for_delete_complete():
        reconcile(controller)
        try:
            container = docker_client.containers.get(server_container_name)
            container.reload()
            if container.status not in ["exited", "removing", "dead"]:
                raise AssertionError(f"Container still running: {container.status}")
        except NotFound:
            pass
        try:
            get_provider(models_client, name=provider_name, workspace=provider_workspace)
        except NotFoundError:
            return
        raise AssertionError("ModelProvider still exists after deployment delete")

    wait_for_delete_complete()

    with pytest.raises(NotFoundError):
        get_provider(models_client, name=provider_name, workspace=provider_workspace)
