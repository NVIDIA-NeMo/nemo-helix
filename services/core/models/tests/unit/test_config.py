# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Models service configuration."""

from unittest.mock import MagicMock, patch

import pytest
from nhx.common.config import Runtime
from nhx.core.models.config import (
    ControllerConfig,
    backends,
    config,
    get_default_backends_for_runtime,
    merge_backends,
)
from nhx.core.models.controllers.backends.deployments_plugin.config import DeploymentsPluginBackendConfigModel
from pydantic import ValidationError


def test_models_controller_config_defaults():
    """Test that Models Controller config has correct default values."""
    assert config.controller.interval_seconds == 5
    assert isinstance(config.controller.backends, dict)


def test_models_controller_config_types():
    """Test that Models Controller config fields have correct types."""
    assert isinstance(config.controller.interval_seconds, int)


def test_models_controller_config_positive_values():
    """Test that Models Controller config has positive values."""
    assert config.controller.interval_seconds > 0


def test_config_structure():
    """Test that config has the correct structure."""
    # Test that we have the expected structure
    assert hasattr(config, "controller")
    assert hasattr(config, "parallelism")

    # Test that Controller config has expected fields
    assert hasattr(config.controller, "interval_seconds")
    assert hasattr(config.controller, "backends")

    # Parallelism config (Pydantic models with defaults)
    assert config.parallelism.gpus_per_node_default == 8
    assert config.parallelism.gpu_memory_gb_default == 80
    assert config.parallelism.memory.pressure_threshold == 0.60
    assert config.parallelism.model_size_thresholds.very_large == 300.0


def test_get_default_backends_for_docker_runtime():
    """Test that deployments_plugin backend is selected and enabled for DOCKER runtime."""
    backends = get_default_backends_for_runtime(Runtime.DOCKER)
    assert "deployments_plugin" in backends
    assert isinstance(backends["deployments_plugin"], DeploymentsPluginBackendConfigModel)
    assert backends["deployments_plugin"].enabled is True
    assert backends["deployments_plugin"].docker_executor == "local-docker"
    assert backends["deployments_plugin"].default_executor == "local-docker"
    assert "docker" not in backends
    assert "nim_operator" not in backends


def test_get_default_backends_for_kubernetes_runtime():
    """Test that deployments_plugin backend is selected and enabled for KUBERNETES runtime."""
    backends = get_default_backends_for_runtime(Runtime.KUBERNETES)
    assert "deployments_plugin" in backends
    assert isinstance(backends["deployments_plugin"], DeploymentsPluginBackendConfigModel)
    assert backends["deployments_plugin"].enabled is True
    assert backends["deployments_plugin"].k8s_executor == "local-k8s"
    assert backends["deployments_plugin"].default_executor == "local-k8s"
    assert "nim_operator" not in backends
    assert "docker" not in backends


def test_get_default_backends_for_none_runtime():
    """Test that deployments_plugin backend is selected for NONE runtime."""
    backends = get_default_backends_for_runtime(Runtime.NONE)
    assert "deployments_plugin" in backends
    assert isinstance(backends["deployments_plugin"], DeploymentsPluginBackendConfigModel)
    assert backends["deployments_plugin"].enabled is True
    assert backends["deployments_plugin"].default_executor is None


def test_merge_backends_with_no_custom_backends():
    """Test that merge returns only default backends when no custom backends provided."""
    default_backends = {"deployments_plugin": DeploymentsPluginBackendConfigModel()}
    custom_backends = {}

    merged = merge_backends(custom_backends, default_backends)

    assert "deployments_plugin" in merged
    assert len(merged) == 1


def test_merge_backends_with_no_default_backends():
    """Test that merge returns only custom backends when no defaults provided."""
    default_backends = {}
    custom_backends = {"deployments_plugin": DeploymentsPluginBackendConfigModel()}

    merged = merge_backends(custom_backends, default_backends)

    assert "deployments_plugin" in merged
    assert len(merged) == 1


def test_merge_backends_custom_overrides_default():
    """Test that custom backend config overrides default backend config."""
    default_backends = {
        "deployments_plugin": DeploymentsPluginBackendConfigModel(enabled=False),
    }
    custom_backends = {
        "deployments_plugin": DeploymentsPluginBackendConfigModel(enabled=True),
    }

    merged = merge_backends(custom_backends, default_backends)

    assert merged["deployments_plugin"].enabled is True


def test_merge_backends_preserves_enabled_flag():
    """Test that merge correctly handles enabled flag overrides."""
    default_backends = {
        "deployments_plugin": DeploymentsPluginBackendConfigModel(enabled=False),
    }
    custom_backends = {
        "deployments_plugin": DeploymentsPluginBackendConfigModel(enabled=True),
    }

    merged = merge_backends(custom_backends, default_backends)

    assert merged["deployments_plugin"].enabled is True


@patch("nhx.core.models.config.get_platform_config")
def test_merge_backends_disables_conflicting_when_runtime_demoted_to_none(mock_platform_config):
    """When runtime is NONE, force-enable deployments_plugin and disable other enabled backends."""
    mock_platform_config.return_value = MagicMock(runtime=Runtime.NONE)
    default_backends = {"deployments_plugin": DeploymentsPluginBackendConfigModel(enabled=True)}
    custom_backends = {
        "deployments_plugin": DeploymentsPluginBackendConfigModel(
            enabled=True,
            docker_executor="local-docker",
        ),
    }

    merged = merge_backends(custom_backends, default_backends)

    assert merged["deployments_plugin"].enabled is True


@patch("nhx.core.models.config.get_platform_config")
def test_merge_backends_leaves_deployments_plugin_enabled_when_runtime_is_docker(mock_platform_config):
    """Sanity-check demotion logic does not disable a valid docker-runtime backend."""
    mock_platform_config.return_value = MagicMock(runtime=Runtime.DOCKER)
    default_backends = {"deployments_plugin": DeploymentsPluginBackendConfigModel(enabled=True)}
    custom_backends = {
        "deployments_plugin": DeploymentsPluginBackendConfigModel(
            enabled=True,
            docker_executor="local-docker",
        ),
    }

    merged = merge_backends(custom_backends, default_backends)

    assert merged["deployments_plugin"].enabled is True


@patch("nhx.core.models.config.get_platform_config")
def test_merge_backends_force_enables_deployments_plugin_when_user_disabled_it_during_demotion(
    mock_platform_config,
):
    """Runtime NONE must still leave exactly one enabled backend."""
    mock_platform_config.return_value = MagicMock(runtime=Runtime.NONE)
    default_backends = {"deployments_plugin": DeploymentsPluginBackendConfigModel(enabled=True)}
    custom_backends = {
        "deployments_plugin": DeploymentsPluginBackendConfigModel(enabled=False),
    }

    merged = merge_backends(custom_backends, default_backends)

    assert merged["deployments_plugin"].enabled is True


def test_module_level_backends_variable_exists():
    """Test that the module-level backends variable exists and is a dict."""
    assert backends is not None
    assert isinstance(backends, dict)


def test_merge_backends_with_flat_config_partial_override():
    """Test that merge handles flat config overrides correctly.

    When a custom backend has partial values set, those values should override
    the default, but unset values should be preserved from the default.
    """
    # Default backend with flat config
    default_backends = {
        "deployments_plugin": DeploymentsPluginBackendConfigModel(
            enabled=True,
            default_storage_class="standard",
            default_pvc_size="100Gi",
        ),
    }

    custom_backends = {
        "deployments_plugin": DeploymentsPluginBackendConfigModel(
            default_storage_class="fast-ssd",
        ),
    }

    merged = merge_backends(custom_backends, default_backends)

    assert merged["deployments_plugin"].default_storage_class == "fast-ssd"
    assert merged["deployments_plugin"].default_pvc_size == "100Gi"


# ============================================================================
# ERROR Deployment GC TTL Config Tests
# ============================================================================


def test_error_deployment_ttl_default():
    """Test that error_deployment_ttl_seconds defaults to 3 hours (10800s)."""
    controller_config = ControllerConfig()
    assert controller_config.error_deployment_ttl_seconds == 10800


def test_error_deployment_ttl_custom_override():
    """Test that error_deployment_ttl_seconds can be overridden."""
    controller_config = ControllerConfig(error_deployment_ttl_seconds=3600)
    assert controller_config.error_deployment_ttl_seconds == 3600


def test_error_deployment_ttl_in_module_config():
    """Test that the module-level config includes error_deployment_ttl_seconds."""
    assert hasattr(config.controller, "error_deployment_ttl_seconds")
    assert isinstance(config.controller.error_deployment_ttl_seconds, int)
    assert config.controller.error_deployment_ttl_seconds > 0


# ============================================================================
# Provider Discovery Config Tests
# ============================================================================


def test_provider_discovery_timeout_default():
    """Provider discovery timeout defaults to 180 seconds for slow external catalogs."""
    controller_config = ControllerConfig()
    assert controller_config.provider_discovery_timeout_seconds == 180


def test_provider_discovery_timeout_custom_override():
    """Provider discovery timeout can be overridden."""
    controller_config = ControllerConfig(provider_discovery_timeout_seconds=240)
    assert controller_config.provider_discovery_timeout_seconds == 240


def test_provider_discovery_max_retries_default():
    """Provider discovery disables SDK retries by default."""
    controller_config = ControllerConfig()
    assert controller_config.provider_discovery_max_retries == 0


def test_provider_discovery_config_in_module_config():
    """Module-level config exposes provider discovery settings."""
    assert config.controller.provider_discovery_timeout_seconds == 180
    assert config.controller.provider_discovery_max_retries == 0


def test_provider_discovery_timeout_rejects_zero():
    """Provider discovery timeout must be at least one second."""
    with pytest.raises(ValidationError):
        ControllerConfig(provider_discovery_timeout_seconds=0)


def test_provider_discovery_max_retries_rejects_negative():
    """Provider discovery max retries must be non-negative."""
    with pytest.raises(ValidationError):
        ControllerConfig(provider_discovery_max_retries=-1)
