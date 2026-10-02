# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from nemo_helix_plugin.models.client import ModelsClient
from nhx.guardrails.app.utils.config_utils import (
    _load_and_execute_py_config,
    configure_rails_config,
    enrich_config_with_data,
    extract_guardrails_models,
    get_path_to_py_configs,
    invalidate_and_reload_config_cache,
)
from nhx.guardrails.config import settings
from nhx.guardrails.entities import GuardrailConfig
from nhx.guardrails.entities.values._private import Model, RailsConfig


@patch("nhx.guardrails.app.utils.config_utils.importlib.util.spec_from_file_location")
@patch("nhx.guardrails.app.utils.config_utils.importlib.util.module_from_spec")
@patch("nhx.guardrails.app.utils.config_utils.fsspec.open")
def test_load_and_execute_py_config(mock_open, mock_module_from_spec, mock_spec_from_file_location):
    mock_spec = MagicMock()
    mock_loader = MagicMock()
    mock_module = MagicMock()
    mock_spec.loader = mock_loader
    mock_spec_from_file_location.return_value = mock_spec
    mock_module_from_spec.return_value = mock_module

    _load_and_execute_py_config("test_config.py")

    mock_open.assert_called_once_with("test_config.py")
    mock_spec_from_file_location.assert_called_once_with("test_config", "test_config.py")
    mock_module_from_spec.assert_called_once_with(mock_spec)
    mock_loader.exec_module.assert_called_once_with(mock_module)


@patch("nhx.guardrails.app.utils.config_utils.fsspec.core.url_to_fs")
@patch("nhx.guardrails.app.utils.config_utils._load_and_execute_py_config")
def test_invalidate_and_reload_config_cache(mock_load_and_execute_py_config, mock_url_to_fs):
    mock_fs = MagicMock()
    mock_url_to_fs.return_value = (mock_fs, None)
    mock_fs.isdir.return_value = True
    mock_fs.glob.return_value = ["test_config.py"]

    rails_registry = MagicMock()
    rails_registry.invalidate_config_cache = MagicMock()

    # TODO we must fix this for 25.04
    invalidate_and_reload_config_cache(rails_registry, "test_namespace/test_config_id", "test_files_url")

    rails_registry.invalidate_config_cache.assert_called_once_with(["test_namespace/test_config_id"])
    mock_url_to_fs.assert_called_once_with("test_files_url", **settings.storage_options)
    mock_fs.isdir.assert_called_once_with("test_files_url")
    mock_fs.glob.assert_called_once_with("test_files_url/**/config.py")
    mock_load_and_execute_py_config.assert_called_once_with("test_config.py")


@patch("nhx.guardrails.app.utils.config_utils.fsspec.core.url_to_fs")
def test_get_path_to_py_configs_directory(mock_url_to_fs):
    mock_fs = MagicMock()
    mock_fs.isdir.return_value = True
    mock_fs.glob.return_value = ["path/to/config.py"]
    mock_url_to_fs.return_value = (mock_fs, None)

    result = get_path_to_py_configs("mock_url")
    assert result == ["path/to/config.py"]


@patch("nhx.guardrails.app.utils.config_utils.fsspec.core.url_to_fs")
def test_get_path_to_py_configs_not_directory(mock_url_to_fs):
    mock_fs = MagicMock()
    mock_fs.isdir.return_value = False
    mock_url_to_fs.return_value = (mock_fs, None)

    result = get_path_to_py_configs("mock_url")
    assert result == []


def test_extract_guardrails_models_with_models():
    """Test that extract_guardrails_models returns list of model URNs when models exist."""
    # Create actual GuardrailConfig with models
    guardrail_config = GuardrailConfig(
        name="test-config",
        workspace="default",
        description="test config",
        data=RailsConfig(
            models=[
                Model(model="default/model1", type="main", engine="nim"),
                Model(model="nvidia/model2", type="main", engine="nim"),
            ],
        ),
    )

    result = extract_guardrails_models(guardrail_config)

    assert result == ["default/model1", "nvidia/model2"]


def test_extract_guardrails_models_no_models():
    """Test that extract_guardrails_models returns empty list when no models exist."""
    guardrail_config = GuardrailConfig(
        name="test-config",
        workspace="default",
        description="test config",
        data=RailsConfig(models=[]),
    )

    result = extract_guardrails_models(guardrail_config)

    assert result == []


def test_extract_guardrails_models_no_data():
    """Test that extract_guardrails_models returns empty list when no data exists."""
    guardrail_config = GuardrailConfig(
        name="test-config",
        workspace="default",
        description="test config",
        data=None,
    )

    result = extract_guardrails_models(guardrail_config)

    assert result == []


class TestEnrichConfigWithData:
    """Tests for the enrich_config_with_data function."""

    def test_enrich_config_without_files_url_returns_original(self):
        """Test that enrich_config_with_data returns original config when no files_url."""
        guardrail_config = GuardrailConfig(
            name="test-config",
            workspace="default",
            description="test config",
            data=None,
        )

        result = enrich_config_with_data(guardrail_config)

        assert result is guardrail_config
        assert result.data is None

    def test_enrich_config_with_data_returns_original(self):
        """Test that enrich_config_with_data returns original config when data is already present."""
        rails_config = RailsConfig(
            models=[Model(model="default/test-model", type="main", engine="nim")],
        )
        guardrail_config = GuardrailConfig(
            name="test-config",
            workspace="default",
            data=rails_config,
        )

        result = enrich_config_with_data(guardrail_config)

        assert result is guardrail_config
        assert result.data is rails_config


class TestConfigureRailsConfig:
    @pytest.fixture
    def mock_platform_sdk(self):
        with (
            patch("nhx.guardrails.app.utils.model_routing.get_platform_sdk") as get_sdk,
            patch("nhx.guardrails.app.utils.model_routing.client_from_platform") as make_client,
        ):
            sdk = MagicMock()
            models = MagicMock()
            models.get_openai_route_base_url.return_value = (
                "http://localhost:8000/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
            )
            get_sdk.return_value = sdk
            make_client.return_value = models
            yield sdk, models, make_client

    def test_resolves_model_entity_references(self, mock_platform_sdk):
        """Test that configure_rails_config resolves Model Entity references."""
        rails_config = RailsConfig(
            models=[
                Model(type="main", engine="nim", model="default/my-model"),
            ]
        )
        model = MagicMock()
        model.model = "default/my-model"

        result = configure_rails_config(rails_config, model)

        sdk, models, make_client = mock_platform_sdk
        make_client.assert_called_once_with(sdk, ModelsClient)
        models.get_openai_route_base_url.assert_called_once_with(workspace="default")
        main_model = result.models[0]
        assert (
            main_model.parameters["base_url"]
            == "http://localhost:8000/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
        )

    def test_resolves_multiple_model_entity_references(self, mock_platform_sdk):
        """Test that configure_rails_config resolves all Model Entity references."""
        rails_config = RailsConfig(
            models=[
                Model(type="main", engine="nim", model="default/llama-model"),
                Model(type="content_safety", engine="nim", model="default/safety-model"),
            ]
        )
        model = MagicMock()
        model.model = "default/llama-model"

        result = configure_rails_config(rails_config, model)

        assert (
            result.models[0].parameters["base_url"]
            == "http://localhost:8000/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
        )
        assert (
            result.models[1].parameters["base_url"]
            == "http://localhost:8000/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
        )

    def test_preserves_explicit_base_url(self, mock_platform_sdk):
        """Test that configure_rails_config preserves explicit base_url."""
        rails_config = RailsConfig(
            models=[
                Model(
                    type="main",
                    engine="nim",
                    model="default/my-model",
                    parameters={"base_url": "http://custom-endpoint/v1"},
                ),
            ]
        )
        model = MagicMock()
        model.model = "default/my-model"

        result = configure_rails_config(rails_config, model)

        assert result.models[0].parameters["base_url"] == "http://custom-endpoint/v1"

    def test_skips_non_model_entity_references(self, mock_platform_sdk):
        """Test that configure_rails_config skips non-Model Entity references."""
        rails_config = RailsConfig(
            models=[
                Model(
                    type="main",
                    engine="openai",
                    model="gpt-4",
                    parameters={"base_url": "https://api.openai.com/v1"},
                ),
            ]
        )
        model = MagicMock()
        model.model = "gpt-4"

        result = configure_rails_config(rails_config, model)

        assert result.models[0].parameters["base_url"] == "https://api.openai.com/v1"
