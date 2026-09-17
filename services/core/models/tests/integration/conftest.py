# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for Models service integration tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generator, Optional, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_deployments_plugin.config import ControllerConfig, DeploymentsConfig, ExecutorConfigEntry
from nemo_deployments_plugin.controller import DeploymentsController
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.models.client import ModelsClient
from nemo_platform_plugin.models.types import (
    ContainerExecutorConfig,
    CreateModelDeploymentConfigRequest,
    CreateModelDeploymentRequest,
    CreateModelProviderRequest,
    Engine,
    ModelDeployment,
    ModelDeploymentConfig,
    ModelDeploymentConfigModelSpec,
    ModelDeploymentStatus,
    ModelEntity,
    ModelProvider,
    UpdateModelDeploymentConfigRequest,
    UpdateModelDeploymentRequest,
    UpsertModelProviderRequest,
)
from nmp.common.config import Runtime
from nmp.common.secrets.encryption import get_base64_encoded_random_bytes
from nmp.core.files.app.backends.base import FileInfo
from nmp.core.files.app.backends.huggingface import HuggingfaceStorageImpl
from nmp.core.models.controllers.backends.backends import DeploymentStatusUpdate, ServiceBackend
from nmp.core.models.controllers.backends.deployments_plugin.backend import DeploymentsPluginServiceBackend
from nmp.core.models.controllers.backends.registry import BackendRegistry
from nmp.core.models.controllers.context import ModelContext
from nmp.core.models.controllers.models_controller import ModelsController
from nmp.core.models.service import ModelsService
from nmp.core.secrets.config import SecretsServiceConfig
from nmp.testing import ClientContext, create_test_client
from nmp.testing.blockbuster import blockbuster_fixture
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

blockbuster = blockbuster_fixture(autouse=True)


@pytest.fixture
def no_hf_network(monkeypatch):
    """Disable live HuggingFace API calls; keep fileset create/update paths local."""

    async def _validate_noop(self):
        return None

    async def _resolve_passthrough(self):
        return self.config

    async def _list_files_stub(_self: HuggingfaceStorageImpl, _path: str | None = None) -> list[FileInfo]:
        return [FileInfo(path="config.json", size=2)]

    monkeypatch.setattr(HuggingfaceStorageImpl, "validate_storage", _validate_noop)
    monkeypatch.setattr(HuggingfaceStorageImpl, "resolve_config", _resolve_passthrough)
    monkeypatch.setattr(HuggingfaceStorageImpl, "list_files", _list_files_stub)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_WORKSPACE = "default"
MOCK_NIM_IMAGE_NAME = f"mock-nim-models:{MOCK_NIM_IMAGE_TAG}"
MOCK_SIDECAR_IMAGE_NAME = f"mock-sidecar-models:{MOCK_SIDECAR_IMAGE_TAG}"


# =============================================================================
# Shared Test Helpers
# =============================================================================


@pytest.fixture
def create_secret() -> Callable:
    """Factory fixture for creating secrets in the test server."""

    def _create(client_context: ClientContext, name: str, workspace: str = "default") -> None:
        response = client_context.test_client.post(
            f"/apis/secrets/v2/workspaces/{workspace}/secrets",
            json={"name": name, "value": "test-token"},
        )
        assert response.status_code == 201, f"Failed to create secret: {response.status_code} {response.text}"

    return _create


@pytest.fixture
def secrets_service_config() -> SecretsServiceConfig:
    """Create encryption config required by SecretsService."""
    return SecretsServiceConfig(
        encryption={
            "current_provider": "test",
            "providers": {
                "secret_key": {
                    "test": {"value": get_base64_encoded_random_bytes(32)},
                },
            },
        },
    )


def models_client_from_sdk(sdk: NeMoPlatform) -> ModelsClient:
    """Create a typed Models client sharing the test platform transport."""
    return client_from_platform(sdk, ModelsClient)


def create_provider(
    client: ModelsClient,
    *,
    workspace: str = DEFAULT_WORKSPACE,
    name: str,
    host_url: str,
    **kwargs: Any,
) -> ModelProvider:
    return client.create_provider(
        workspace=workspace,
        body=CreateModelProviderRequest(name=name, host_url=host_url, **kwargs),
    ).data()


def upsert_provider(
    client: ModelsClient,
    *,
    workspace: str = DEFAULT_WORKSPACE,
    name: str,
    host_url: str,
    **kwargs: Any,
) -> ModelProvider:
    return client.upsert_provider(
        workspace=workspace,
        name=name,
        body=UpsertModelProviderRequest(host_url=host_url, **kwargs),
    ).data()


def get_provider(client: ModelsClient, *, workspace: str = DEFAULT_WORKSPACE, name: str) -> ModelProvider:
    return client.get_provider(workspace=workspace, name=name).data()


def list_providers(client: ModelsClient, *, workspace: str = DEFAULT_WORKSPACE) -> list[ModelProvider]:
    return list(client.list_providers(workspace=workspace).items())


def delete_provider(client: ModelsClient, *, workspace: str = DEFAULT_WORKSPACE, name: str) -> None:
    client.delete_provider(workspace=workspace, name=name).data()


def _engine(value: Engine | str) -> Engine:
    return value if isinstance(value, Engine) else Engine(value)


def _model_spec(
    value: ModelDeploymentConfigModelSpec | dict[str, Any] | None,
) -> ModelDeploymentConfigModelSpec:
    if isinstance(value, ModelDeploymentConfigModelSpec):
        return value
    return ModelDeploymentConfigModelSpec.model_validate(value or {})


def _executor_config(
    value: ContainerExecutorConfig | dict[str, Any] | None,
) -> ContainerExecutorConfig:
    if isinstance(value, ContainerExecutorConfig):
        return value
    return ContainerExecutorConfig.model_validate(value or {"gpu": 0})


def create_deployment_config(
    client: ModelsClient,
    *,
    workspace: str = DEFAULT_WORKSPACE,
    name: str,
    engine: Engine | str,
    model_spec: ModelDeploymentConfigModelSpec | dict[str, Any] | None = None,
    executor_config: ContainerExecutorConfig | dict[str, Any] | None = None,
    model_entity_id: str | None = None,
) -> ModelDeploymentConfig:
    body = CreateModelDeploymentConfigRequest(
        name=name,
        engine=_engine(engine),
        model_spec=_model_spec(model_spec),
        executor_config=_executor_config(executor_config),
        model_entity_id=model_entity_id,
    )
    return client.create_deployment_config(workspace=workspace, body=body).data()


def update_deployment_config(
    client: ModelsClient,
    *,
    workspace: str = DEFAULT_WORKSPACE,
    name: str,
    engine: Engine | str,
    model_spec: ModelDeploymentConfigModelSpec | dict[str, Any],
    executor_config: ContainerExecutorConfig | dict[str, Any],
    model_entity_id: str | None = None,
) -> ModelDeploymentConfig:
    body = UpdateModelDeploymentConfigRequest(
        engine=_engine(engine),
        model_spec=_model_spec(model_spec),
        executor_config=_executor_config(executor_config),
        model_entity_id=model_entity_id,
    )
    return client.update_deployment_config(workspace=workspace, name=name, body=body).data()


def get_deployment_config(
    client: ModelsClient, *, workspace: str = DEFAULT_WORKSPACE, name: str
) -> ModelDeploymentConfig:
    return client.get_deployment_config(workspace=workspace, name=name).data()


def list_deployment_configs(client: ModelsClient, *, workspace: str = DEFAULT_WORKSPACE) -> list[ModelDeploymentConfig]:
    return list(client.list_deployment_configs(workspace=workspace).items())


def delete_deployment_config(client: ModelsClient, *, workspace: str = DEFAULT_WORKSPACE, name: str) -> None:
    client.delete_deployment_config(workspace=workspace, name=name).data()


def create_deployment(
    client: ModelsClient,
    *,
    workspace: str = DEFAULT_WORKSPACE,
    name: str,
    config: str,
    config_version: int | None = None,
) -> ModelDeployment:
    return client.create_deployment(
        workspace=workspace,
        body=CreateModelDeploymentRequest(name=name, config=config, config_version=config_version),
    ).data()


def update_deployment(
    client: ModelsClient,
    *,
    workspace: str = DEFAULT_WORKSPACE,
    name: str,
    config: str,
    config_version: int | None = None,
) -> ModelDeployment:
    return client.update_deployment(
        workspace=workspace,
        name=name,
        body=UpdateModelDeploymentRequest(config=config, config_version=config_version),
    ).data()


def get_deployment(client: ModelsClient, *, workspace: str = DEFAULT_WORKSPACE, name: str) -> ModelDeployment:
    return client.get_deployment(workspace=workspace, name=name).data()


def list_deployments(client: ModelsClient, *, workspace: str = DEFAULT_WORKSPACE) -> list[ModelDeployment]:
    return list(client.list_deployments(workspace=workspace).items())


def delete_deployment(client: ModelsClient, *, workspace: str = DEFAULT_WORKSPACE, name: str) -> None:
    client.delete_deployment(workspace=workspace, name=name).data()


# =============================================================================
# Mock Backend for Backend-Agnostic Tests
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
        self.create_calls: list[tuple[ModelDeployment, ModelDeploymentConfig, Optional[ModelEntity]]] = []
        self.update_calls: list[tuple[ModelDeployment, ModelDeploymentConfig, Optional[ModelEntity]]] = []
        self.status_calls: list[ModelDeployment] = []
        self.delete_calls: list[tuple[str, str]] = []  # (workspace, name)

        # Configure responses (can be overridden per-test)
        # Note: host_url is None by default to avoid port conflicts in parallel tests.
        # Tests that need a specific host_url should set it explicitly in status_responses.
        self.create_response = DeploymentStatusUpdate(
            status=ModelDeploymentStatus.PENDING,
            status_message="Container created and starting",
            host_url=None,
        )
        # Per-deployment status responses - keyed by deployment name
        # If a deployment name is not in this dict, falls back to default_status_response
        self.status_responses: dict[str, DeploymentStatusUpdate] = {}
        self.default_status_response = DeploymentStatusUpdate(
            status=ModelDeploymentStatus.READY,
            status_message="Container is ready",
            host_url=None,
        )
        self.delete_response = DeploymentStatusUpdate(
            status=ModelDeploymentStatus.DELETED,
            status_message="Container deleted",
        )

    def init(self) -> None:
        """No-op init for mock backend."""
        pass

    def shutdown(self) -> None:
        """No-op shutdown for mock backend."""
        pass

    async def create_model_deployment(self, ctx: ModelContext) -> DeploymentStatusUpdate:
        """Record call and return configured response."""
        deployment = ctx.model_deployment
        config = ctx.model_deployment_config
        assert deployment is not None
        assert config is not None
        self.create_calls.append((deployment, config, ctx.model_entity))
        return self.create_response

    async def update_model_deployment(self, ctx: ModelContext) -> DeploymentStatusUpdate:
        """Record call and return configured response."""
        deployment = ctx.model_deployment
        config = ctx.model_deployment_config
        assert deployment is not None
        assert config is not None
        self.update_calls.append((deployment, config, ctx.model_entity))
        return self.create_response  # Update returns same as create

    async def get_model_deployment_status(self, ctx: ModelContext) -> DeploymentStatusUpdate:
        """Record call and return configured response.

        Takes the reconcile ``ctx`` (bundling deployment + config + entity) like the
        real ``ServiceBackend``; records the deployment so existing assertions that
        inspect ``status_calls`` by ``.name`` keep working.

        Uses per-deployment responses from status_responses dict if available,
        otherwise falls back to default_status_response.
        """
        deployment = ctx.model_deployment
        assert deployment is not None
        self.status_calls.append(deployment)
        return self.status_responses.get(deployment.name, self.default_status_response)

    async def delete_model_deployment(
        self,
        workspace: str,
        name: str,
        *,
        deleting_elapsed_seconds: float | None = None,
    ) -> DeploymentStatusUpdate:
        """Record call and return configured response."""
        self.delete_calls.append((workspace, name))
        return self.delete_response

    async def list_managed_deployment_names(self) -> list[str]:
        """Return empty list; mock has no managed deployments."""
        return []


# =============================================================================
# Integration Test Fixtures
# =============================================================================


@pytest.fixture
def test_clients() -> Generator[ClientContext, None, None]:
    """Create all client types sharing the same app for controller tests.

    The controller needs an async SDK, but we also need the sync SDK/test client
    to create test data. ClientContext provides all of these sharing one app.
    """
    with create_test_client(ModelsService, client_type=ClientContext) as clients:
        yield clients


@pytest.fixture
def mock_backend(test_clients: ClientContext) -> MockServiceBackend:
    """Create a mock backend for testing."""
    return MockServiceBackend(
        nmp_sdk=test_clients.async_sdk,
        config={},
    )


@pytest.fixture
def mock_backend_registry(mock_backend: MockServiceBackend) -> BackendRegistry:
    """Create a backend registry with the mock backend."""
    return BackendRegistry(registry={"mock": mock_backend})


@pytest.fixture
def controller_with_mock_backend(
    test_clients: ClientContext, mock_backend_registry: BackendRegistry
) -> Generator[tuple[ModelsController, MockServiceBackend, ModelsClient], None, None]:
    """Create a ModelsController wired to use the test SDK and mock backend.

    Note: The ProviderReconciler's autodiscovery is mocked to avoid issues when
    running tests in parallel. Without this mock, the reconciler would try to
    call through the IGW proxy (which isn't available in models-only tests) and
    would also iterate over providers from other tests running in the same worker.

    Yields:
        Tuple of (controller, mock_backend, models_client) for testing
    """
    mock_backend = cast(MockServiceBackend, mock_backend_registry.get_backend())

    # Create controller with mock backend registry
    # We need to patch the SDK factory and platform config (used in config and main modules)
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
            backend_registry=mock_backend_registry,
            stop_signal=None,
        )

        # Mock the provider reconciler to avoid issues in parallel test execution.
        # The reconciler tries to call through the IGW proxy for autodiscovery,
        # which isn't available in models-only tests.
        controller._provider_reconciler.reconcile_model_providers = AsyncMock(return_value=None)

        yield controller, mock_backend, models_client_from_sdk(test_clients.sdk)

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
    is patched in controller fixtures to return this name for 'nmp-core'.
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

    # Always cleanup
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
    Request this via controller_with_docker; no per-test try/finally needed.
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


# =============================================================================
# Deployments-plugin Docker Integration Fixtures
# =============================================================================


@pytest.fixture
def deployments_plugin_backend_config(worker_id: str) -> dict[str, Any]:
    """Flat deployments_plugin backend config for integration tests."""
    return {
        "docker_executor": "local-docker",
        "default_executor": "local-docker",
        "deleting_timeout_seconds": 300,
    }


@pytest.fixture
def deployments_plugin_platform_config(worker_id: str) -> MagicMock:
    """Platform config with docker runtime and worker-scoped executor ports."""
    start_port, end_port = get_worker_port_range(worker_id)
    mock_platform_config = MagicMock()
    mock_platform_config.models_url = "http://testserver"
    mock_platform_config.base_url = "http://testserver"
    mock_platform_config.runtime = Runtime.DOCKER
    mock_platform_config.get_service_url.return_value = "http://testserver"
    mock_platform_config.service_discovery = {"files": "http://testserver"}
    mock_platform_config._deployments_config = DeploymentsConfig(
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
        controller=ControllerConfig(
            interval_seconds=1,
            orphan_cleanup_interval_seconds=0,
        ),
    )
    return mock_platform_config


@pytest.fixture
def controller_with_deployments_plugin(
    test_clients,
    docker_client,  # noqa: ARG001 - ensures docker is available
    mock_nim_image,
    docker_test_context,
    models_controller_container_cleanup,
    deployments_plugin_backend_config,
    deployments_plugin_platform_config,
):
    """Models controller + deployments plugin controller on real Docker."""
    mock_platform_config = deployments_plugin_platform_config
    deployments_config = mock_platform_config._deployments_config

    plugin_backend = DeploymentsPluginServiceBackend(
        nmp_sdk=test_clients.async_sdk,
        config=deployments_plugin_backend_config,
        huggingface_model_puller="alpine:3.20",
    )
    plugin_backend.init()
    backend_registry = BackendRegistry(registry={"deployments_plugin": plugin_backend})

    deployments_controller = DeploymentsController()

    def reconcile_stack(models_controller: ModelsController) -> None:
        models_controller.step()
        models_controller._loop.run_until_complete(deployments_controller.reconcile())

    with (
        patch("nmp.core.models.config.get_platform_config", return_value=mock_platform_config),
        patch("nmp.core.models.controllers.main.get_platform_config", return_value=mock_platform_config),
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.resolve.get_platform_config",
            return_value=mock_platform_config,
        ),
        patch(
            "nmp.core.models.controllers.backends.deployments_plugin.backend.get_async_platform_sdk"
        ) as mock_deployments_backend_sdk,
        patch("nmp.core.models.controllers.models_controller.get_async_platform_sdk") as mock_models_sdk,
        patch("nemo_deployments_plugin.controller.get_async_platform_sdk") as mock_deployments_controller_sdk,
        patch("nemo_platform_plugin.sdk_provider.get_async_platform_sdk") as mock_sdk,
        patch("nemo_deployments_plugin.config.DeploymentsConfig.get", return_value=deployments_config),
    ):
        mock_deployments_backend_sdk.return_value = test_clients.async_sdk
        mock_deployments_controller_sdk.return_value = test_clients.async_sdk
        mock_models_sdk.return_value = test_clients.async_sdk
        mock_sdk.return_value = test_clients.async_sdk

        models_controller = ModelsController(
            backend_registry=backend_registry,
            stop_signal=None,
        )
        models_controller._provider_reconciler.reconcile_model_providers = AsyncMock(return_value=None)
        models_controller._loop.run_until_complete(deployments_controller.on_startup())

        yield (
            models_controller,
            deployments_controller,
            models_client_from_sdk(test_clients.sdk),
            mock_nim_image,
            docker_test_context,
            reconcile_stack,
        )

        try:
            models_controller._loop.run_until_complete(deployments_controller.on_shutdown())
        finally:
            models_controller.shutdown()


# =============================================================================
# Pytest Hooks
# =============================================================================


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Generator[None, None, None]:
    """Store test results on the item for fixture access."""
    outcome: Any = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)
