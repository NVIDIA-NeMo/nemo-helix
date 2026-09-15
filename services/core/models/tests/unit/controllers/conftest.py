# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test fixtures for Models Controller tests."""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from nemo_platform_plugin.client.errors import ConflictError, NemoHTTPError, NotFoundError
from nmp.common.config import PlatformConfig


def platform_config(
    *,
    base_url: str = "http://localhost:8080",
    service_discovery: dict[str, str] | None = None,
    **kwargs,
) -> PlatformConfig:
    """Create a real PlatformConfig for testing."""
    if service_discovery is None:
        service_discovery = {
            "files": "http://files-service:8000",
            "models": "http://models-api:8000",
        }
    return PlatformConfig(  # type: ignore[abstract]
        base_url=base_url,
        service_discovery=service_discovery,
        **kwargs,
    )


@pytest.fixture
def mock_platform_config():
    """Create a platform config for testing (real PlatformConfig instance)."""
    return platform_config()


@pytest.fixture
def mock_get_config_patch(mock_platform_config):
    """Patch get_platform_config to return mock config.

    Patches in nmp.core.models.config (used when loading backends) and
    nmp.core.models.controllers.main (used in run() for server ready check).
    """
    with (
        patch(
            "nmp.core.models.config.get_platform_config",
            return_value=mock_platform_config,
        ),
        patch(
            "nmp.core.models.controllers.main.get_platform_config",
            return_value=mock_platform_config,
        ),
    ):
        yield


@pytest.fixture
def mock_sdk_class_patch():
    """Patch get_async_platform_sdk factory function."""
    with patch("nmp.core.models.controllers.models_controller.get_async_platform_sdk") as mock:
        mock.return_value.close = AsyncMock()
        yield mock


@pytest.fixture
def mock_asyncio_run_patch():
    """Patch event loop run_until_complete for controller step tests."""
    # Create a mock event loop with run_until_complete method
    mock_loop = MagicMock()

    def _run_until_complete(awaitable):
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        return mock_loop.run_until_complete.return_value

    mock_loop.run_until_complete.side_effect = _run_until_complete
    with patch("nmp.core.models.controllers.models_controller.asyncio.new_event_loop", return_value=mock_loop):
        yield mock_loop.run_until_complete


@pytest.fixture
def mock_models_sdk():
    """Create a mock AsyncNeMoPlatform SDK for testing."""
    mock_sdk = MagicMock()
    return mock_sdk


@pytest.fixture
def mock_backend_registry():
    """Create a mock BackendRegistry for testing."""
    mock_registry = MagicMock()
    mock_backend = MagicMock()
    mock_registry.get_backend.return_value = mock_backend
    mock_registry.list_backends.return_value = ["docker"]
    return mock_registry


@pytest.fixture
def sample_deployment():
    """Create a sample ModelDeployment for testing."""
    deployment = MagicMock()
    deployment.workspace = "default"
    deployment.name = "test-deployment"
    deployment.entity_version = "v1"
    deployment.status = "PENDING"
    deployment.config = None
    deployment.config_version = None
    deployment.model_provider_id = None
    return deployment


@pytest.fixture
def sample_deployment_ready():
    """Create a sample ModelDeployment in READY state for testing."""
    deployment = MagicMock()
    deployment.workspace = "default"
    deployment.name = "ready-deployment"
    deployment.entity_version = "v1"
    deployment.status = "READY"
    deployment.config = None
    deployment.config_version = None
    deployment.model_provider_id = None
    return deployment


@pytest.fixture
def sample_deployment_unknown():
    """Create a sample ModelDeployment in UNKNOWN state for testing."""
    deployment = MagicMock()
    deployment.workspace = "test-ns"
    deployment.name = "unknown-deployment"
    deployment.entity_version = "v2"
    deployment.status = "UNKNOWN"
    return deployment


def _assert_controller_initialized(controller, mock_backend_registry):
    """Assert that controller is properly initialized with expected state."""
    assert controller._is_healthy is False
    assert controller._backend_registry == mock_backend_registry


def _assert_controller_has_required_attributes(controller):
    """Assert that controller has all required attributes."""
    assert hasattr(controller, "_is_healthy")
    assert hasattr(controller, "_backend_registry")
    assert hasattr(controller, "_models_sdk")


def _assert_controller_healthy(controller, is_healthy=True):
    """Assert controller health status."""
    assert controller.is_healthy is is_healthy


def _assert_sdk_initialized_correctly(mock_sdk_class_patch):
    """Assert that get_async_platform_sdk was called with correct args for Models API."""
    # Controller initializes ONE SDK for Models API (base_url is resolved from config inside the factory)
    assert mock_sdk_class_patch.call_count == 1
    call_kwargs = mock_sdk_class_patch.call_args.kwargs
    assert call_kwargs["as_service"] == "models"
    assert call_kwargs["internal"] is True
    assert "http_client" in call_kwargs


def _assert_asyncio_run_called_once(mock_asyncio_run_patch):
    """Assert that asyncio.run was called exactly once."""
    assert mock_asyncio_run_patch.call_count == 1


def _assert_deployments_count(deployments, expected_count):
    """Assert the number of deployments returned."""
    assert len(deployments) == expected_count


class AssertHelpers:
    """Container for assertion helper functions, accessible via fixture."""

    assert_controller_initialized = staticmethod(_assert_controller_initialized)
    assert_controller_has_required_attributes = staticmethod(_assert_controller_has_required_attributes)
    assert_controller_healthy = staticmethod(_assert_controller_healthy)
    assert_sdk_initialized_correctly = staticmethod(_assert_sdk_initialized_correctly)
    assert_asyncio_run_called_once = staticmethod(_assert_asyncio_run_called_once)
    assert_deployments_count = staticmethod(_assert_deployments_count)


@pytest.fixture
def assert_helpers():
    """Fixture providing access to assertion helper functions.

    Usage:
        def test_something(assert_helpers):
            assert_helpers.assert_controller_healthy(controller)
    """
    return AssertHelpers


class _AsyncPage:
    """Stand-in for ``AsyncNemoPaginatedResponse``: exposes an async ``items()``."""

    def __init__(self, items):
        self._items = list(items)

    async def items(self):
        for item in self._items:
            yield item


class _ModelResponse:
    """Stand-in for ``NemoResponse[ModelEntity]``: exposes ``data()``."""

    def __init__(self, value=None, exc: Exception | None = None):
        self._value = value
        self._exc = exc

    def data(self):
        if self._exc is not None:
            raise self._exc
        return self._value


def _status_error(status: int, detail: str):
    """Build a plugin client HTTP error for a given status."""
    request = httpx.Request("POST", "http://test")
    response = httpx.Response(status, request=request, json={"detail": detail})
    if status == 404:
        return NotFoundError(response)
    if status == 409:
        return ConflictError(response)
    return NemoHTTPError(response)


def make_async_models_client() -> MagicMock:
    """Build a mock ``AsyncModelsClient`` exposing the typed surface the
    controllers/reconcilers consume via ``client_from_platform``."""
    client = MagicMock()
    client.list_models = AsyncMock(return_value=_AsyncPage([]))
    client.create_model = AsyncMock(return_value=_ModelResponse())
    client.get_model = AsyncMock(return_value=_ModelResponse())
    client.update_model = AsyncMock(return_value=_ModelResponse())
    client.list_deployments = AsyncMock(return_value=_AsyncPage([]))
    client.get_deployment = AsyncMock(return_value=_ModelResponse())
    client.get_deployment_config_version = AsyncMock(return_value=_ModelResponse())
    client.update_deployment_status = AsyncMock(return_value=_ModelResponse())
    client.delete_deployment_version = AsyncMock(return_value=_ModelResponse())
    client.list_providers = AsyncMock(return_value=_AsyncPage([]))
    client.get_provider = AsyncMock(return_value=_ModelResponse())
    client.create_provider = AsyncMock(return_value=_ModelResponse())
    client.upsert_provider = AsyncMock(return_value=_ModelResponse())
    client.update_provider_status = AsyncMock(return_value=_ModelResponse())
    client.delete_provider = AsyncMock(return_value=_ModelResponse())
    return client


@pytest.fixture
async def mock_models_client() -> MagicMock:
    """A mock ``AsyncModelsClient`` returned by ``client_from_platform`` in the
    controller/reconciler modules under test."""
    return make_async_models_client()


@pytest.fixture
async def patch_models_client(mock_models_client):
    """Route ``client_from_platform(sdk, AsyncModelsClient)`` in the models
    controller modules back to :data:`mock_models_client`."""
    with (
        patch(
            "nmp.core.models.controllers.entity_cache.client_from_platform",
            return_value=mock_models_client,
        ),
        patch(
            "nmp.core.models.sidecars.adapters.main.client_from_platform",
            return_value=mock_models_client,
        ),
    ):
        yield mock_models_client


_ENTITY_FIELDS = ("model_providers", "fileset", "api_endpoint", "backend_format")


def make_entity(workspace: str, name: str, **attrs):
    """Model Entity stand-in addressable by ModelEntityCache.

    ``model_copy`` mirrors the real model in carrying every field across, which the
    cache relies on when overlaying staged changes onto a snapshot.
    """
    fields = {field: attrs.get(field) for field in _ENTITY_FIELDS}
    entity = MagicMock()
    entity.workspace = workspace
    entity.name = name
    for field, value in fields.items():
        setattr(entity, field, value)

    def _copy(update):
        return make_entity(workspace, name, **{**fields, **update})

    entity.model_copy = MagicMock(side_effect=_copy)
    return entity


async def seed_entity_cache(
    mock_models_sdk,
    entity_cache,
    entities=(),
    *,
    models_client: MagicMock | None = None,
):
    """Load the cache from the mock SDK so lookups resolve to ``entities``.

    ``models_client`` is the mock ``AsyncModelsClient`` returned by the patched
    ``client_from_platform``; defaulting to ``mock_models_sdk.models_client``.
    """
    models_client = models_client or mock_models_sdk.models_client
    models_client.list_models.return_value = _AsyncPage(list(entities))
    await entity_cache.refresh()
