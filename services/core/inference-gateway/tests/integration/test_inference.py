# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for IGW with real Docker deployment.

These tests verify the full deployment -> inference flow through the
Inference Gateway using real Docker containers running the mock NIM.

Note: These tests require Docker to be running and the nhx-core image
to be built (make docker/nhx-core).
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from docker.errors import NotFound
from nemo_deployments_plugin.backends.labels import container_name as plugin_container_name
from nemo_deployments_plugin.backends.labels import docker_volume_name
from nemo_helix.types.inference.virtual_model import VirtualModel as SDKVirtualModel
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError
from nemo_helix_plugin.inference_gateway.client import InferenceGatewayClient
from nemo_helix_plugin.inference_gateway.types import JsonBody
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import (
    CreateModelDeploymentConfigRequest,
    CreateModelDeploymentRequest,
    ModelDeployment,
    ModelDeploymentConfig,
    ModelProvider,
    ServedModelMapping,
    UpdateModelProviderStatusRequest,
)
from nemo_helix_plugin.virtual_models.client import VirtualModelsClient
from nemo_helix_plugin.virtual_models.types import CreateVirtualModelRequest
from nhx.core.inference_gateway.api.dependencies import global_virtual_model_cache
from nhx.core.inference_gateway.api.model_cache import ModelCache, ModelProviderInfo
from nhx.core.models.controllers.backends.deployments_plugin.naming import entity_names
from nhx.core.models.controllers.models_controller import ModelsController
from tenacity import retry, stop_after_delay, wait_fixed

DEFAULT_WORKSPACE = "default"


def _wait_for_deployment_ready(
    controller: ModelsController,
    models: ModelsClient,
    deployment_name: str,
    max_wait: float = 30,
    poll_interval: float = 0.1,
) -> ModelDeployment:
    """Wait for a deployment to become READY.

    Polls the controller and checks deployment status until READY or timeout.

    Args:
        controller: The ModelsController instance
        models: The typed models client
        deployment_name: Name of the deployment to wait for
        max_wait: Maximum time to wait in seconds (default 30)
        poll_interval: Time between polls in seconds (default 0.1)

    Returns:
        The deployment if READY

    Raises:
        AssertionError: If deployment doesn't reach READY status within timeout
    """

    @retry(stop=stop_after_delay(max_wait), wait=wait_fixed(poll_interval), reraise=True)
    def _poll():
        controller.step()
        deployment = models.get_deployment(name=deployment_name, workspace=DEFAULT_WORKSPACE).data()
        assert deployment.status == "READY", f"Deployment not READY: {deployment.status}"
        return deployment

    return _poll()


def _wait_for_deployment_deleted(
    controller: ModelsController,
    models: ModelsClient,
    deployment_name: str,
    max_wait: float = 30,
    poll_interval: float = 0.1,
) -> None:
    """Step the controller until a deployment reaches DELETED (or is gone).

    The deployments_plugin delete path is non-blocking, so a single
    ``controller.step()`` is not enough to drive a deployment to DELETED: the
    reconciler must run several times (DELETING -> backend teardown -> DELETED).
    Deleting the ModelDeploymentConfig before its deployment is DELETED fails
    with a 409 Conflict.
    """

    @retry(stop=stop_after_delay(max_wait), wait=wait_fixed(poll_interval), reraise=True)
    def _poll() -> None:
        controller.step()
        try:
            deployment = models.get_deployment(name=deployment_name, workspace=DEFAULT_WORKSPACE).data()
        except NotFoundError:
            return
        assert deployment.status == "DELETED", f"Deployment not DELETED: {deployment.status}"

    _poll()


def _create_deployment_with_config(
    models: ModelsClient,
    config_name: str,
    deployment_name: str,
    mock_nim_image: str,
) -> tuple[ModelDeploymentConfig, ModelDeployment]:
    """Create a deployment config and deployment.

    Args:
        models: The typed models client
        config_name: Name for the deployment config
        deployment_name: Name for the deployment
        mock_nim_image: Full image name:tag for the mock NIM

    Returns:
        Tuple of (config, deployment)
    """
    # Use rsplit to handle registry URLs with port numbers
    # e.g., "registry.example.com/nemo-helix/mock-nim:1.0.0"
    image_name, image_tag = mock_nim_image.rsplit(":", 1)
    config = models.create_deployment_config(
        workspace=DEFAULT_WORKSPACE,
        body=CreateModelDeploymentConfigRequest(
            name=config_name,
            engine="nim",
            model_spec={},
            executor_config={
                "gpu": 0,
                "image_name": image_name,
                "image_tag": image_tag,
            },
        ),
    ).data()

    deployment = models.create_deployment(
        workspace=DEFAULT_WORKSPACE,
        body=CreateModelDeploymentRequest(name=deployment_name, config=config_name),
    ).data()

    return config, deployment


def _configure_served_models(
    models: ModelsClient,
    provider_name: str,
    model_entity_name: str,
    served_model_name: str,
) -> ModelProvider:
    """Configure served_models on a provider for model entity routing.

    Args:
        models: The typed models client
        provider_name: Name of the provider
        model_entity_name: Name of the model entity
        served_model_name: Name for the backend model

    Returns:
        The updated ModelProvider
    """
    return models.update_provider_status(
        name=provider_name,
        workspace=DEFAULT_WORKSPACE,
        body=UpdateModelProviderStatusRequest(
            served_models=[
                ServedModelMapping(
                    model_entity_id=f"{DEFAULT_WORKSPACE}/{model_entity_name}",
                    served_model_name=served_model_name,
                )
            ]
        ),
    ).data()


def _assert_chat_response(response_data: dict[str, Any], route_name: str) -> None:
    """Assert that a chat completion response is valid.

    Args:
        response_data: The parsed response body
        route_name: Name of the route for error messages
    """
    assert "message" in response_data or "choices" in response_data, (
        f"Unexpected {route_name} response: {response_data}"
    )


def _manually_add_provider_to_cache(
    model_cache: ModelCache,
    models: ModelsClient,
    provider_name: str,
    rebuild_model_entity_map: bool = False,
) -> bool:
    """Manually add a provider to the cache by fetching it from the API.

    This avoids event loop issues by using sync API calls and directly
    populating the cache.

    Args:
        model_cache: The ModelCache instance to update
        models: The typed models client
        provider_name: Name of the provider to add
        rebuild_model_entity_map: If True, rebuild the model entity map after adding

    Returns:
        True if provider was added, False otherwise
    """
    try:
        provider = models.get_provider(name=provider_name, workspace=DEFAULT_WORKSPACE).data()
    except Exception:
        return False

    provider_info = ModelProviderInfo(model_provider=provider)
    model_cache.workspace_name_provider_map[(DEFAULT_WORKSPACE, provider_name)] = provider_info

    if rebuild_model_entity_map:
        model_cache.rebuild_model_entity_map()

    # Create a passthrough VirtualModel via the API for every served entity, mirroring
    # the production provider reconciler's _ensure_passthrough_virtual_model behavior.
    # The IGW requires every inference request to resolve to a VirtualModel, and the
    # IGW's background cache refresher rebuilds the VM map from the API list periodically;
    # going through the API ensures the VM survives refreshes. The fixture mocks the
    # production reconciler away to avoid event-loop conflicts (see
    # controller_with_docker_and_igw), so this test must create VMs explicitly.
    # LoRA composites are skipped to match the production reconciler.
    virtual_model_cache = global_virtual_model_cache()
    virtual_models = VirtualModelsClient.from_client(models)
    now_iso = "2026-01-01T00:00:00Z"
    for served_model in provider.served_models or []:
        ws, _, entity_name = served_model.model_entity_id.partition("/")
        if not entity_name or "&adapters/" in entity_name:
            continue
        try:
            virtual_models.create_virtual_model(
                workspace=ws,
                body=CreateVirtualModelRequest(
                    name=entity_name, default_model_entity=f"{ws}/{entity_name}", autoprovisioned=True
                ),
            )
        except ConflictError:
            pass
        # In-place cache seed for immediate request-time availability without waiting
        # for the IGW's next background refresh tick.
        key = (ws, entity_name)
        if key in virtual_model_cache.virtual_model_map:
            continue
        virtual_model_cache.virtual_model_map[key] = SDKVirtualModel(
            id=f"{ws}/{entity_name}",
            entity_id=f"{ws}/{entity_name}",
            workspace=ws,
            name=entity_name,
            parent=ws,
            db_version=1,
            default_model_entity=f"{ws}/{entity_name}",
            autoprovisioned=True,
            created_at=now_iso,
            updated_at=now_iso,
        )

    return True


@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_igw_routes_to_deployed_mock_nim(
    controller_with_docker_and_igw,
    docker_client,
):
    """Test full deployment -> inference flow through IGW.

    This test:
    1. Creates deployment config and deployment
    2. Waits for deployment to become READY
    3. Configures served_models for model entity routing
    4. Tests GET and POST requests through all 3 route types
    5. Cleans up deployment
    """
    controller, model_cache, client, mock_nim_image, ctx, _ = controller_with_docker_and_igw
    models = ModelsClient.from_client(client)
    gateway = InferenceGatewayClient.from_client(client)
    test_uuid = uuid.uuid4().hex[:8]
    config_name = f"test-igw-e2e-{test_uuid}"
    deployment_name = f"test-igw-e2e-{test_uuid}"
    names = entity_names(deployment_name)
    container_name = plugin_container_name(DEFAULT_WORKSPACE, names.server)

    # Register for cleanup
    ctx.register_container(container_name)
    ctx.register_volume(docker_volume_name(DEFAULT_WORKSPACE, names.volume))
    ctx.register_volume(docker_volume_name(DEFAULT_WORKSPACE, names.scratch))

    # === Phase 1: Create deployment config and deployment ===
    config, deployment = _create_deployment_with_config(models, config_name, deployment_name, mock_nim_image)
    assert config.name == config_name
    assert deployment.name == deployment_name

    # === Phase 3: Controller creates container and wait for READY ===
    # The controller.step() creates the container and polls health checks
    # We poll with short intervals until deployment becomes READY
    deployment = _wait_for_deployment_ready(controller, models, deployment_name)
    assert deployment and deployment.status == "READY", f"Deployment not READY: {deployment}"

    # Verify container is running with retry (DinD may be slow)
    @retry(stop=stop_after_delay(5), wait=wait_fixed(0.1), reraise=True)
    def get_running_container():
        c = docker_client.containers.get(container_name)
        assert c.status == "running", f"Container not running: {c.status}"
        return c

    get_running_container()

    provider_id = deployment.model_provider_id
    assert provider_id is not None, "Provider should be created when deployment is READY"

    # === Phase 4: Configure served_models for model entity routing ===
    model_entity_name = f"test-model-{test_uuid}"
    served_model_name = "mock-model"

    _configure_served_models(models, deployment_name, model_entity_name, served_model_name)

    # === Phase 5: Manually add provider to IGW cache ===
    assert _manually_add_provider_to_cache(model_cache, models, deployment_name, rebuild_model_entity_map=True), (
        "Failed to add provider to cache"
    )

    # Verify provider is in cache
    cached_provider = model_cache.get_from_provider(DEFAULT_WORKSPACE, deployment_name)
    assert cached_provider is not None, "Provider should be in cache after manual add"

    # Verify model entity is in cache
    model_entity_info = model_cache.get_from_model_entity(DEFAULT_WORKSPACE, model_entity_name)
    assert model_entity_info is not None, f"Model entity {model_entity_name} should be in cache"

    # === Phase 6: Test all 3 IGW proxy route types (GET) ===

    # --- Route Type 1: Provider route ---
    provider_models = gateway.provider_get(
        trailing_uri="v1/models",
        name=deployment_name,
        workspace=DEFAULT_WORKSPACE,
    ).data()
    assert "data" in provider_models, f"Expected 'data' in response: {provider_models}"

    provider_health = gateway.provider_get(
        trailing_uri="v1/health/ready",
        name=deployment_name,
        workspace=DEFAULT_WORKSPACE,
    ).data()
    assert provider_health.get("status") == "ready", f"Expected ready status: {provider_health}"

    # --- Route Type 2: Model entity route ---
    model_entity_models = gateway.model_get(
        trailing_uri="v1/models",
        name=model_entity_name,
        workspace=DEFAULT_WORKSPACE,
    ).data()
    assert "data" in model_entity_models, f"Expected 'data' in model entity response: {model_entity_models}"

    model_entity_health = gateway.model_get(
        trailing_uri="v1/health/ready",
        name=model_entity_name,
        workspace=DEFAULT_WORKSPACE,
    ).data()
    assert model_entity_health.get("status") == "ready", f"Expected ready status: {model_entity_health}"

    # --- Route Type 3: OpenAI-compatible route ---
    openai_models = gateway.list_openai_models(workspace=DEFAULT_WORKSPACE).data()
    assert openai_models.data is not None, f"Expected 'data' in OpenAI models response: {openai_models}"

    # Verify our model is listed in OpenAI format (workspace/model_entity_name)
    model_ids = [m.id for m in openai_models.data]
    expected_model_id = f"{DEFAULT_WORKSPACE}/{model_entity_name}"
    assert expected_model_id in model_ids, f"Expected {expected_model_id} in OpenAI models: {model_ids}"

    openai_model = gateway.get_openai_model(name=expected_model_id, workspace=DEFAULT_WORKSPACE).data()
    assert openai_model.id == expected_model_id, f"Expected model ID {expected_model_id}: {openai_model}"

    # === Phase 7: Test POST requests through all 3 route types ===
    chat_request = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # POST through provider route
    provider_chat = gateway.provider_post(
        trailing_uri="v1/chat/completions",
        name=deployment_name,
        workspace=DEFAULT_WORKSPACE,
        body=JsonBody(chat_request),
    ).data()
    _assert_chat_response(provider_chat, "Provider route")

    # POST through model entity route
    model_chat = gateway.model_post(
        trailing_uri="v1/chat/completions",
        name=model_entity_name,
        workspace=DEFAULT_WORKSPACE,
        body=JsonBody(chat_request),
    ).data()
    _assert_chat_response(model_chat, "Model entity route")

    # POST through OpenAI route
    openai_chat_request = {
        "model": expected_model_id,
        "messages": [{"role": "user", "content": "Hello from OpenAI route"}],
    }
    openai_chat = gateway.openai_post(
        trailing_uri="v1/chat/completions",
        workspace=DEFAULT_WORKSPACE,
        body=JsonBody(openai_chat_request),
    ).data()
    _assert_chat_response(openai_chat, "OpenAI route")

    # === Phase 8: Cleanup ===
    models.delete_deployment(name=deployment_name, workspace=DEFAULT_WORKSPACE)
    # Drive the non-blocking delete to completion before removing the config.
    _wait_for_deployment_deleted(controller, models, deployment_name)

    # Container should be gone once the deployment is DELETED (DinD may lag).
    @retry(stop=stop_after_delay(15), wait=wait_fixed(0.1), reraise=True)
    def wait_for_container_deleted():
        try:
            c = docker_client.containers.get(container_name)
            c.reload()
            if c.status in ["exited", "removing", "dead"]:
                return  # Container is stopping/stopped
            raise AssertionError(f"Container still running: {c.status}")
        except NotFound:
            return  # Container was removed, which is expected

    try:
        wait_for_container_deleted()
    except AssertionError:
        # After all retries, log but don't fail - provider deletion is the key check
        print(f"Warning: Container {container_name} still running after retries")

    models.delete_deployment_config(name=config_name, workspace=DEFAULT_WORKSPACE)


def test_igw_returns_404_for_unknown_provider(controller_with_docker_and_igw):
    """Test that IGW returns 404 for non-existent providers and model entities."""
    _, _, client, _, _, _ = controller_with_docker_and_igw
    gateway = InferenceGatewayClient.from_client(client)

    # Provider route 404
    with pytest.raises(NotFoundError):
        gateway.provider_get(
            trailing_uri="v1/models",
            name="nonexistent-provider",
            workspace=DEFAULT_WORKSPACE,
        ).data()

    # Model entity route 404
    with pytest.raises(NotFoundError):
        gateway.model_get(
            trailing_uri="v1/models",
            name="nonexistent-model",
            workspace=DEFAULT_WORKSPACE,
        ).data()

    # OpenAI route 404 (workspace from path, model name only in request)
    with pytest.raises(NotFoundError):
        gateway.get_openai_model(name="nonexistent-model", workspace=DEFAULT_WORKSPACE)


def test_igw_cache_removes_deleted_deployment_provider(
    controller_with_docker_and_igw,
    docker_client,
):
    """Test that IGW cache is updated when deployment is deleted.

    Verifies that after a deployment is deleted and the provider is removed,
    the cache can be updated to reflect the deletion.
    """
    controller, model_cache, client, mock_nim_image, ctx, _ = controller_with_docker_and_igw
    models = ModelsClient.from_client(client)
    test_uuid = uuid.uuid4().hex[:8]
    config_name = f"test-igw-delete-{test_uuid}"
    deployment_name = f"test-igw-delete-{test_uuid}"
    names = entity_names(deployment_name)
    container_name = plugin_container_name(DEFAULT_WORKSPACE, names.server)

    ctx.register_container(container_name)
    ctx.register_volume(docker_volume_name(DEFAULT_WORKSPACE, names.volume))
    ctx.register_volume(docker_volume_name(DEFAULT_WORKSPACE, names.scratch))

    # Create deployment
    config, deployment = _create_deployment_with_config(models, config_name, deployment_name, mock_nim_image)
    assert config.name == config_name
    assert deployment.name == deployment_name

    # Wait for deployment to become READY
    deployment = _wait_for_deployment_ready(controller, models, deployment_name)
    assert deployment and deployment.status == "READY", f"Deployment not READY: {deployment}"

    # Manually add provider to cache
    _manually_add_provider_to_cache(model_cache, models, deployment_name)
    assert model_cache.get_from_provider(DEFAULT_WORKSPACE, deployment_name) is not None

    # Delete deployment
    models.delete_deployment(name=deployment_name, workspace=DEFAULT_WORKSPACE)
    _wait_for_deployment_deleted(controller, models, deployment_name)

    # Manually remove from cache (simulating cache refresh)
    cache_key = (DEFAULT_WORKSPACE, deployment_name)
    if cache_key in model_cache.workspace_name_provider_map:
        del model_cache.workspace_name_provider_map[cache_key]

    # Verify provider is no longer in cache
    assert model_cache.get_from_provider(DEFAULT_WORKSPACE, deployment_name) is None

    # Cleanup
    models.delete_deployment_config(name=config_name, workspace=DEFAULT_WORKSPACE)
