# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for Inference Gateway integration tests."""

from __future__ import annotations

from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_deployments_plugin.config import ControllerConfig, DeploymentsConfig, ExecutorConfigEntry
from nemo_deployments_plugin.controller import DeploymentsController
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nmp.common.config import Runtime
from nmp.core.inference_gateway.api.dependencies import global_model_cache
from nmp.core.inference_gateway.api.model_cache import ModelCache, model_provider_getter_from_sdk, refresh_model_cache
from nmp.core.inference_gateway.config import InferenceGatewayConfig
from nmp.core.inference_gateway.service import InferenceGatewayService
from nmp.core.models.controllers.backends.backends import DeploymentStatusUpdate, ServiceBackend
from nmp.core.models.controllers.backends.deployments_plugin.backend import DeploymentsPluginServiceBackend
from nmp.core.models.controllers.backends.registry import BackendRegistry
from nmp.core.models.controllers.models_controller import ModelsController
from nmp.core.models.service import ModelsService
from nmp.testing import ClientContext, create_test_client
from nmp.testing.docker import (
    MOCK_NIM_IMAGE_TAG,
    MOCK_SIDECAR_IMAGE_TAG,
    DockerTestContext,
    cleanup_model_deployment_containers,
    create_docker_client,
    ensure_mock_nim_image,
    ensure_mock_sidecar_image,
    get_worker_port_range,
)

import docker

# =============================================================================
# Constants
# =============================================================================

DEFAULT_WORKSPACE = "default"
MOCK_NIM_IMAGE_NAME = f"mock-nim-inference-gateway:{MOCK_NIM_IMAGE_TAG}"
MOCK_SIDECAR_IMAGE_NAME = f"mock-sidecar-inference-gateway:{MOCK_SIDECAR_IMAGE_TAG}"


# =============================================================================
# Mock Backend for Controller Tests
# =============================================================================


class MockServiceBackend(ServiceBackend):
    """Mock backend that returns controlled responses for testing."""

    def __init__(
        self,
        nmp_sdk: AsyncNeMoPlatform,
        config: dict[str, Any],
    ) -> None:
        """Initialize mock backend without calling parent init."""
        self._nmp_sdk = nmp_sdk
        self._config = config

        # Track method calls for assertions
        self.create_calls: list[tuple[Any, Any, Any]] = []
        self.update_calls: list[tuple[Any, Any, Any]] = []
        self.status_calls: list[Any] = []
        self.delete_calls: list[Any] = []

        # Configure responses (can be overridden per-test)
        # Note: host_url is None by default to avoid port conflicts in parallel tests.
        # Tests that need a specific host_url should set it explicitly.
        self.create_response = DeploymentStatusUpdate(
            status="PENDING",
            status_message="Container created and starting",
            host_url=None,
        )
        self.status_response = DeploymentStatusUpdate(
            status="READY",
            status_message="Container is ready",
            host_url=None,
        )
        self.delete_response = DeploymentStatusUpdate(
            status="DELETED",
            status_message="Container deleted",
        )

    def init(self) -> None:
        """No-op init for mock backend."""
        pass

    async def create_model_deployment(self, ctx: Any) -> DeploymentStatusUpdate:
        """Record call and return configured response."""
        self.create_calls.append((ctx.model_deployment, ctx.model_deployment_config, ctx.model_entity))
        return self.create_response

    async def update_model_deployment(self, ctx: Any) -> DeploymentStatusUpdate:
        """Record call and return configured response."""
        self.update_calls.append((ctx.model_deployment, ctx.model_deployment_config, ctx.model_entity))
        return self.create_response

    async def get_model_deployment_status(self, ctx: Any) -> DeploymentStatusUpdate:
        """Record call and return configured response."""
        self.status_calls.append(ctx.model_deployment)
        return self.status_response

    async def delete_model_deployment(self, deployment: Any) -> DeploymentStatusUpdate:
        """Record call and return configured response."""
        self.delete_calls.append(deployment)
        return self.delete_response


# =============================================================================
# Multi-Service Test Fixtures
# =============================================================================


@pytest.fixture
def test_clients() -> Generator[ClientContext, None, None]:
    """Create all client types sharing the same app for IGW + controller tests.

    Creates Models Service, Entities Service, and Inference Gateway Service
    with background cache refresh disabled to avoid event loop conflicts.
    """
    with create_test_client(
        ModelsService,
        InferenceGatewayService,
        client_type=ClientContext,
        service_configs={
            # Disable background cache refresh to avoid event loop conflicts in tests
            InferenceGatewayService: InferenceGatewayConfig(refresh_model_cache_interval_sec=0),
        },
    ) as clients:
        yield clients


@pytest.fixture
def controller_with_mock_backend(
    test_clients: ClientContext,
) -> Generator[tuple[ModelsController, MockServiceBackend, NeMoPlatform, ModelCache, AsyncNeMoPlatform], None, None]:
    """Create ModelsController with mock backend and access to IGW cache.

    Note: The ProviderReconciler's autodiscovery is mocked to avoid event loop
    conflicts when calling through the IGW proxy.

    Yields:
        Tuple of (controller, mock_backend, sync_sdk, model_cache, async_sdk)
    """
    mock_backend = MockServiceBackend(nmp_sdk=test_clients.async_sdk, config={})
    backend_registry = BackendRegistry(registry={"mock": mock_backend})

    mock_platform_config = MagicMock()
    mock_platform_config.models_url = "http://testserver"
    mock_platform_config.get_service_url.return_value = "http://testserver"
    with (
        patch("nmp.core.models.config.get_platform_config", return_value=mock_platform_config),
        patch("nmp.core.models.controllers.main.get_platform_config", return_value=mock_platform_config),
        patch("nmp.core.models.controllers.models_controller.get_async_platform_sdk") as mock_sdk_factory,
    ):
        mock_sdk_factory.return_value = test_clients.async_sdk

        controller = ModelsController(
            backend_registry=backend_registry,
            stop_signal=None,
        )

        # Mock the provider reconciler to avoid event loop conflicts
        # when it tries to call through the IGW proxy for autodiscovery
        controller._provider_reconciler.reconcile_model_providers = AsyncMock(return_value=None)

        # Access the global model cache
        model_cache = global_model_cache()

        yield controller, mock_backend, test_clients.sdk, model_cache, test_clients.async_sdk

        # Clean up controller resources (event loop, backend registry, etc.)
        controller.shutdown()


# =============================================================================
# Docker Test Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def docker_client() -> Generator[docker.DockerClient, None, None]:
    """Create a Docker client for tests."""
    client = create_docker_client()
    yield client


@pytest.fixture(scope="module")
def mock_nim_image(docker_client: docker.DockerClient) -> Generator[str, None, None]:
    """Build or retrieve the nginx-based mock NIM image."""
    yield ensure_mock_nim_image(docker_client, MOCK_NIM_IMAGE_NAME)


@pytest.fixture(scope="module")
def mock_sidecar_image(docker_client: docker.DockerClient) -> Generator[str, None, None]:
    """Build or retrieve the mock sidecar image with a unique name per service.

    Avoids concurrent-build races across workers. The backend's get_qualified_image
    is patched in controller_with_docker_and_igw to return this name for 'nmp-core'.
    """
    yield ensure_mock_sidecar_image(docker_client, MOCK_SIDECAR_IMAGE_NAME)


@pytest.fixture
def docker_test_context(
    docker_client: docker.DockerClient, request: pytest.FixtureRequest
) -> Generator[DockerTestContext, None, None]:
    """Create a Docker test context with automatic cleanup and diagnostics."""
    ctx = DockerTestContext(docker_client=docker_client)

    yield ctx

    # Check if test failed and print diagnostics
    if hasattr(request.node, "rep_call") and request.node.rep_call.failed:
        ctx.print_diagnostics()

    ctx.cleanup()


@pytest.fixture
def docker_owner_labels(worker_id: str, testrun_uid: str) -> dict[str, str]:
    """Labels used to scope Docker resources to this pytest worker/run."""
    return {
        "nmp.nvidia.com/test-run": testrun_uid,
        "nmp.nvidia.com/test-worker": worker_id,
    }


@pytest.fixture
def models_controller_container_cleanup(
    docker_client: docker.DockerClient,
    docker_owner_labels: dict[str, str],
) -> Generator[None, None, None]:
    """Teardown: remove all containers with label nmp.nvidia.com/managed-by=models-controller.

    Ensures failed tests (e.g. stuck in PENDING) don't leave NIM/sidecar containers.
    Request this via controller_with_docker_and_igw; no per-test try/finally needed.
    """
    yield
    cleanup_model_deployment_containers(docker_client, labels=docker_owner_labels)


@pytest.fixture
def docker_backend_config(
    worker_id: str,
    mock_sidecar_image: str,
    docker_owner_labels: dict[str, str],
) -> dict[str, Any]:
    """Configuration for Docker backend in tests.

    Uses worker_id from pytest-xdist to allocate unique port ranges per worker.
    Depends on mock_sidecar_image so the image get_qualified_image('nmp-core')
    exists before any test runs.
    """
    start_port, end_port = get_worker_port_range(worker_id)
    return {
        "models_docker_port_range_start": start_port,
        "models_docker_port_range_end": end_port,
        "docker_timeout": 60,
        "models_docker_host_service_name": "localhost",
        "model_labels": docker_owner_labels,
    }


@pytest.fixture
def deployments_plugin_backend_config() -> dict[str, Any]:
    return {
        "docker_executor": "local-docker",
        "default_executor": "local-docker",
        "deleting_timeout_seconds": 300,
    }


@pytest.fixture
def controller_with_docker_and_igw(
    test_clients: ClientContext,
    docker_client,
    mock_nim_image,
    mock_sidecar_image,
    docker_backend_config,
    docker_test_context,
    models_controller_container_cleanup,
    deployments_plugin_backend_config,
    worker_id: str,
) -> Generator[
    tuple[ModelsController, ModelCache, NeMoPlatform, str, DockerTestContext, AsyncNeMoPlatform], None, None
]:
    """Create ModelsController with Docker backend and IGW with shared SDK.

    Creates:
    - In-memory test client with Models + IGW services
    - Docker backend with mock NIM entrypoint
    - ModelsController for deployment reconciliation
    - IGW model cache for manual refresh

    Note: The ProviderReconciler's autodiscovery is mocked to avoid event loop
    conflicts when calling through the IGW proxy.

    Yields:
        Tuple of (controller, model_cache, sdk, mock_nim_image, docker_test_context, async_sdk)
    """
    from nemo_platform_plugin.jobs.image import get_qualified_image as real_get_qualified_image

    def patched_get_qualified_image(name: str, tag=None, registry=None):
        if name in ["nmp-core", "nmp-api"]:
            return mock_sidecar_image
        return real_get_qualified_image(name, tag=tag, registry=registry)

    start_port, end_port = get_worker_port_range(worker_id)
    deployments_config = DeploymentsConfig(
        executors=[
            ExecutorConfigEntry(
                name="local-docker",
                backend="docker",
                config={
                    "pull_images": False,
                    "port_range_start": start_port,
                    "port_range_end": end_port,
                },
            )
        ],
        default_executor="local-docker",
        controller=ControllerConfig(interval_seconds=1, orphan_cleanup_interval_seconds=0),
    )

    plugin_backend = DeploymentsPluginServiceBackend(
        nmp_sdk=test_clients.async_sdk,
        config=deployments_plugin_backend_config,
        huggingface_model_puller="alpine:3.20",
    )
    plugin_backend.init()
    backend_registry = BackendRegistry(registry={"deployments_plugin": plugin_backend})
    deployments_controller = DeploymentsController()

    mock_platform_config = MagicMock()
    mock_platform_config.models_url = "http://testserver"
    mock_platform_config.base_url = "http://testserver"
    mock_platform_config.runtime = Runtime.DOCKER
    mock_platform_config.get_service_url.return_value = "http://testserver"
    mock_platform_config.service_discovery = {"files": "http://testserver"}

    with (
        patch("nmp.core.models.config.get_platform_config", return_value=mock_platform_config),
        patch("nmp.core.models.controllers.main.get_platform_config", return_value=mock_platform_config),
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.resolve.get_platform_config",
            return_value=mock_platform_config,
        ),
        patch("nmp.core.models.controllers.models_controller.get_async_platform_sdk") as mock_sdk_factory,
        patch("nemo_deployments_plugin.controller.get_async_platform_sdk") as mock_deployments_controller_sdk,
        patch("nemo_platform_plugin.sdk_provider.get_async_platform_sdk") as mock_sdk,
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.backend.get_async_platform_sdk"
        ) as mock_deployments_backend_sdk,
        patch("nemo_deployments_plugin.config.DeploymentsConfig.get", return_value=deployments_config),
        patch(
            "nemo_platform_plugin.jobs.image.get_qualified_image",
            side_effect=patched_get_qualified_image,
        ),
    ):
        mock_sdk_factory.return_value = test_clients.async_sdk
        mock_deployments_controller_sdk.return_value = test_clients.async_sdk
        mock_sdk.return_value = test_clients.async_sdk
        mock_deployments_backend_sdk.return_value = test_clients.async_sdk

        controller = ModelsController(
            backend_registry=backend_registry,
            stop_signal=None,
        )
        controller._provider_reconciler.reconcile_model_providers = AsyncMock(return_value=None)
        controller._loop.run_until_complete(deployments_controller.on_startup())

        original_step = controller.step

        def step_with_deployments_plugin() -> None:
            original_step()
            controller._loop.run_until_complete(deployments_controller.reconcile())

        controller.step = step_with_deployments_plugin

        model_cache = global_model_cache()

        yield controller, model_cache, test_clients.sdk, mock_nim_image, docker_test_context, test_clients.async_sdk

        try:
            controller._loop.run_until_complete(deployments_controller.on_shutdown())
        finally:
            controller.shutdown()


# =============================================================================
# Cache Refresh Helpers
# =============================================================================


async def trigger_cache_refresh(
    model_cache: ModelCache,
    sdk: AsyncNeMoPlatform,
) -> None:
    """Trigger a manual cache refresh from the Models Service.

    Args:
        model_cache: The IGW model cache to refresh
        sdk: Async SDK for fetching providers
    """
    model_provider_getter = model_provider_getter_from_sdk(sdk)
    await refresh_model_cache(
        model_cache=model_cache,
        model_provider_getter=model_provider_getter,
        secrets_sdk=sdk,
    )


# =============================================================================
# Pytest Hooks
# =============================================================================


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Generator[None, None, None]:
    """Store test results on the item for fixture access."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)
