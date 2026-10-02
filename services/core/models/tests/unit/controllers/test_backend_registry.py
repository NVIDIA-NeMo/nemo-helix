# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BackendRegistry."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nhx.core.models.controllers.backends.deployments_plugin.config import DeploymentsPluginBackendConfigModel
from nhx.core.models.controllers.backends.registry import BackendRegistry


@pytest.fixture
def mock_nhx_sdk():
    """Create a mock AsyncNeMoHelix SDK."""
    return AsyncMock()


@pytest.fixture
def sample_backend_configs():
    """Create sample backend configurations."""
    return {
        "deployments_plugin": DeploymentsPluginBackendConfigModel(enabled=True),
    }


def test_backend_registry_from_config(mock_nhx_sdk, sample_backend_configs):
    """Test creating BackendRegistry from configuration."""
    with patch(
        "nhx.core.models.controllers.backends.deployments_plugin.backend.NemoEntitiesClient",
    ):
        registry = BackendRegistry.from_config(
            nhx_sdk=mock_nhx_sdk,
            backend_configs=sample_backend_configs,
            huggingface_model_puller="nvcr.io/nvidia/nemo-microservices/nds-v2-huggingface-cli:25.10",
        )

    assert isinstance(registry, BackendRegistry)
    assert registry.get_backend("deployments_plugin") is not None


def test_backend_registry_get_default_backend(mock_nhx_sdk, sample_backend_configs):
    """Test getting default backend (the single enabled one)."""
    with patch(
        "nhx.core.models.controllers.backends.deployments_plugin.backend.NemoEntitiesClient",
    ):
        registry = BackendRegistry.from_config(
            nhx_sdk=mock_nhx_sdk,
            backend_configs=sample_backend_configs,
            huggingface_model_puller="nvcr.io/nvidia/nemo-microservices/nds-v2-huggingface-cli:25.10",
        )

    assert registry.get_backend() is not None


def test_backend_registry_get_backend_not_found(mock_nhx_sdk, sample_backend_configs):
    """Test that KeyError is raised for unknown backend."""
    with patch(
        "nhx.core.models.controllers.backends.deployments_plugin.backend.NemoEntitiesClient",
    ):
        registry = BackendRegistry.from_config(
            nhx_sdk=mock_nhx_sdk,
            backend_configs=sample_backend_configs,
            huggingface_model_puller="nvcr.io/nvidia/nemo-microservices/nds-v2-huggingface-cli:25.10",
        )

    with pytest.raises(KeyError, match="Backend 'unknown' not found"):
        registry.get_backend("unknown")


def test_backend_registry_list_backends(mock_nhx_sdk, sample_backend_configs):
    """Test listing all registered backends."""
    with patch(
        "nhx.core.models.controllers.backends.deployments_plugin.backend.NemoEntitiesClient",
    ):
        registry = BackendRegistry.from_config(
            nhx_sdk=mock_nhx_sdk,
            backend_configs=sample_backend_configs,
            huggingface_model_puller="nvcr.io/nvidia/nemo-microservices/nds-v2-huggingface-cli:25.10",
        )

    backends = registry.list_backends()
    assert backends == ["deployments_plugin"]


def test_backend_registry_empty_config_raises_error(mock_nhx_sdk):
    """Test that empty backend config raises ValueError."""
    with pytest.raises(ValueError, match="At least one backend must be configured"):
        BackendRegistry.from_config(
            nhx_sdk=mock_nhx_sdk,
            backend_configs={},
            huggingface_model_puller="nvcr.io/nvidia/nemo-microservices/nds-v2-huggingface-cli:25.10",
        )


def test_backend_registry_init_with_empty_dict_raises_error():
    """Test that initializing BackendRegistry with empty dict raises ValueError."""
    with pytest.raises(ValueError, match="Backend registry cannot be empty"):
        BackendRegistry(registry={})


def test_backend_registry_unknown_backend_type(mock_nhx_sdk):
    """Test that unknown backend type raises KeyError when backend class not in registry."""
    bad_config = {"legacy-docker": DeploymentsPluginBackendConfigModel(enabled=True)}

    with pytest.raises(KeyError, match="Unknown backend 'legacy-docker'"):
        BackendRegistry.from_config(
            nhx_sdk=mock_nhx_sdk,
            backend_configs=bad_config,
            huggingface_model_puller="nvcr.io/nvidia/nemo-microservices/nds-v2-huggingface-cli:25.10",
            available_backends={},
        )


def test_backend_config_from_yaml_to_registry(mock_nhx_sdk):
    """Test end-to-end: parse backend configs from dicts (like YAML) and use them with registry."""
    from nhx.core.models.config import ControllerConfig

    yaml_config = {
        "deployments_plugin": {"enabled": True, "default_pvc_size": "100Gi"},
    }

    controller_config = ControllerConfig(backends=yaml_config)
    parsed_configs = controller_config.backends

    assert isinstance(parsed_configs["deployments_plugin"], DeploymentsPluginBackendConfigModel)
    assert parsed_configs["deployments_plugin"].default_pvc_size == "100Gi"

    with patch(
        "nhx.core.models.controllers.backends.deployments_plugin.backend.NemoEntitiesClient",
    ):
        registry = BackendRegistry.from_config(
            nhx_sdk=mock_nhx_sdk,
            backend_configs=parsed_configs,
            huggingface_model_puller="nvcr.io/nvidia/nemo-microservices/nds-v2-huggingface-cli:25.10",
        )

    assert registry.list_backends() == ["deployments_plugin"]


def test_backend_registry_no_enabled_backends_raises_error(mock_nhx_sdk):
    """Test that having no enabled backends raises ValueError."""
    config_with_no_enabled = {
        "deployments_plugin": DeploymentsPluginBackendConfigModel(enabled=False),
    }

    with pytest.raises(ValueError, match="No backends are enabled"):
        BackendRegistry.from_config(
            nhx_sdk=mock_nhx_sdk,
            backend_configs=config_with_no_enabled,
            huggingface_model_puller="nvcr.io/nvidia/nemo-microservices/nds-v2-huggingface-cli:25.10",
        )


def test_backend_registry_multiple_enabled_backends_raises_error(mock_nhx_sdk):
    """Test that having multiple enabled backends raises ValueError."""
    config_with_multiple_enabled = {
        "deployments_plugin": DeploymentsPluginBackendConfigModel(enabled=True),
        "deployments_plugin_shadow": DeploymentsPluginBackendConfigModel(enabled=True),
    }

    with pytest.raises(ValueError, match="Multiple backends are enabled"):
        BackendRegistry.from_config(
            nhx_sdk=mock_nhx_sdk,
            backend_configs=config_with_multiple_enabled,
            huggingface_model_puller="nvcr.io/nvidia/nemo-microservices/nds-v2-huggingface-cli:25.10",
        )


def test_deployments_plugin_missing_package_raises_guidance(mock_nhx_sdk):
    """Missing nemo-deployments-plugin should surface install guidance."""
    import builtins

    from nhx.core.models.controllers.backends.registry import _resolve_backend_class, backend_classes

    real_import = builtins.__import__

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "nhx.core.models.controllers.backends.deployments_plugin.backend":
            raise ImportError("No module named 'nemo_deployments_plugin'")
        return real_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=mock_import),
        pytest.raises(ImportError, match=r'pip install "nemo-helix\[all\]"'),
    ):
        _resolve_backend_class("deployments_plugin", backend_classes)

    with (
        patch("builtins.__import__", side_effect=mock_import),
        pytest.raises(ImportError, match=r'pip install "nemo-helix\[all\]"'),
    ):
        BackendRegistry.from_config(
            nhx_sdk=mock_nhx_sdk,
            backend_configs={"deployments_plugin": DeploymentsPluginBackendConfigModel(enabled=True)},
            huggingface_model_puller="puller:latest",
        )


def test_backend_registry_shutdown_calls_backend_shutdown(mock_nhx_sdk, sample_backend_configs):
    """Test that shutdown_all_backends calls shutdown on each backend."""
    with patch(
        "nhx.core.models.controllers.backends.deployments_plugin.backend.NemoEntitiesClient",
    ):
        registry = BackendRegistry.from_config(
            nhx_sdk=mock_nhx_sdk,
            backend_configs=sample_backend_configs,
            huggingface_model_puller="puller:latest",
        )

    backend = registry.get_backend()
    backend.shutdown = MagicMock()
    registry.shutdown_all_backends()
    backend.shutdown.assert_called_once()
