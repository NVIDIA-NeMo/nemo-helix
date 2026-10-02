# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO(v2): Fix missing module 'guardrails.schemas.global_model_configs'
# This test file is currently disabled because the module structure has changed.
# The GlobalModelConfig schema needs to be restored or the imports need to be updated.

import pytest

pytest.skip("Module structure changed - needs to be fixed in v2", allow_module_level=True)

# ruff: noqa: E402
# Imports after skip to avoid import errors
import os
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List
from unittest.mock import patch

import yaml
from nhx.guardrails.app.constants import NIM_CHAT, NIM_LLM
from nhx.guardrails.app.schemas.global_model_configs import GlobalModelConfig
from nhx.guardrails.app.services.configs.global_model_config_registry import (
    DEFAULT_PROVIDER_NAME,
    GlobalModelConfigRegistry,
    _standardize_provider_name,
)

# Fetched on 20250512, picked a few entries
NIM_MODELS_RESPONSE = {
    "object": "list",
    "data": [
        {"id": "nvidia/nemotron-4-340b-instruct", "object": "model", "created": 735790403, "owned_by": "nvidia"},
        {"id": "nvidia/nemotron-4-340b-reward", "object": "model", "created": 735790403, "owned_by": "nvidia"},
    ],
}

# Use first couple of models from the global config-store file to test loading models
# When `demo` == True the config.yml is copied (with 5 models) so use this to distinguish
# between the two
APP_MODEL_CONFIG_STRING = dedent("""\
models:
  - model_id: nvcf/meta/llama-3.1-70b-instruct
    engine: nim
    model: meta/llama-3.1-70b-instruct
    base_url: https://integrate.api.nvidia.com/v1

  - model_id: openai/davinci-002
    engine: openai
    model: davinci-002
    base_url: https://api.openai.com/v1
""")

# Create a Python dict equivalent to validate Yaml loaded correctly
APP_MODEL_CONFIG = yaml.safe_load(APP_MODEL_CONFIG_STRING)

# This is a hack because the location of the guardrails/config-store is brittle depending on where the test executes.
# If the guardrails/config-store/config.yml changes, tests using this will break.
CONFIG_STORE_MODEL_CONFIG_STRING = dedent("""\
models:
  - model_id: meta/llama-3.1-70b-instruct
    engine: nim
    model: meta/llama-3.1-70b-instruct
    base_url: https://integrate.api.nvidia.com/v1

  - model_id: openai/davinci-002
    engine: openai
    model: davinci-002
    base_url: https://api.openai.com/v1

  - model_id: openai/gpt-4o-mini
    engine: openai
    model: gpt-4o-mini
    base_url: https://api.openai.com/v1

  - model_id: openai/gpt-4o
    engine: openai
    model: gpt-4o
    base_url: https://api.openai.com/v1

  - model_id: openai/gpt-3.5-turbo-instruct
    engine: openai
    model: gpt-3.5-turbo-instruct
    base_url: https://api.openai.com/v1
""")
# Create a Python dict equivalent to validate Yaml loaded correctly
CONFIG_STORE_MODEL_CONFIG = yaml.safe_load(CONFIG_STORE_MODEL_CONFIG_STRING)

NIM_ENGINES = {NIM_LLM, NIM_CHAT}


@pytest.fixture
def registry():
    """Initialize a new empty registry"""
    return GlobalModelConfigRegistry(config_path="")


@pytest.fixture
def nvidia_model1():
    return GlobalModelConfig(
        model_id="nvidia_model1",
        engine="nim",
        model="meta/llama-3.3-70b-instruct",
        base_url="https://integrate.api.nvidia.com/v1",
        parameters={
            "temperature": 0.6,
            "max_tokens": 10,
            "top_p": 0.8,
        },
    )


@pytest.fixture
def nvidia_model2():
    return GlobalModelConfig(
        model_id="nvidia_model2",
        engine="nim",
        model="meta/llama-3.3-70b-instruct",
        base_url="https://integrate.api.nvidia.com/v1",
        parameters={
            "temperature": 0.01,
            "max_tokens": 2000,
            "top_p": 0.5,
        },
    )


def _check_registry_models(source_models: List[Dict[str, Any]], registry_models: List[Dict[str, Any]]) -> None:
    """Check the registry-loaded models against source model definitions. The Registry makes several
    substitutions if the `engine` field is set to `nim`:
     - Replaces `nim` engine to `nimchat` or `nimllm` depending on parameters.mode
     - Copies model name into `parameters` even if that doesn't exist

    The Registry also adds a `created` field with the UTC time of creation of the model
    """
    assert len(registry_models) == len(source_models)
    for idx, _ in enumerate(registry_models):
        assert source_models[idx]["model_id"] == registry_models[idx]["model_id"]
        assert source_models[idx]["model"] == registry_models[idx]["model"]
        assert source_models[idx]["base_url"] == registry_models[idx]["base_url"]

        # Registry expands `nim` engine to `nimchat` or `nimllm` depending on parameters.mode
        # And also copies the model name into parameters
        if source_models[idx]["engine"] == "nim":
            assert source_models[idx]["model"] == registry_models[idx]["parameters"]["model"]
            assert registry_models[idx]["engine"] in NIM_ENGINES
        else:
            assert source_models[idx]["engine"] == registry_models[idx]["engine"]

        assert "created" in registry_models[idx]


def _load_yaml_file(filepath):
    """Safely load a YAML file given the path"""
    with open(filepath, "r") as infile:
        data = yaml.safe_load(infile)
    return data


def _key(model: GlobalModelConfig) -> str:
    """Helper to return key used to store configs"""
    return model.model_id


def test_add_new_model(registry, nvidia_model1):
    """Add a new model to registry, read it back and check it matches"""
    registry.add(nvidia_model1)
    assert registry.get(_key(nvidia_model1)) == nvidia_model1


def test_add_existing_model(registry, nvidia_model1, nvidia_model2):
    """Add a model and then a second with the same (engine, model). Make sure the
    second one gets returned (i.e. we overwrote the first one)
    """
    registry.add(nvidia_model1)
    registry.add(nvidia_model2)
    assert registry.get(_key(nvidia_model1)) == nvidia_model1
    assert registry.get(_key(nvidia_model2)) == nvidia_model2


def test_get_nonexistent_model(registry, nvidia_model1):
    """Get a model that doesn't exist, should return None and not throw Exception"""
    assert registry.get(_key(nvidia_model1)) is None


def test_get_existing_model(registry, nvidia_model1):
    """Add a model and check we can retrieve it"""
    registry.add(nvidia_model1)
    assert registry.get(_key(nvidia_model1)) == nvidia_model1


def test_list_empty_registry_models(registry):
    """Listing empty registry should return an empty list"""
    assert registry.list() == []


def test_list_existing_models(registry, nvidia_model1):
    """Test listing models when some exist"""
    registry.add(nvidia_model1)
    assert registry.list() == [nvidia_model1]


def test_load_config_argument_demo_false():
    """Test when we use the `config_path` argument and demo is false we only get
    the models from APP_MODEL_CONFIG"""

    with tempfile.TemporaryDirectory() as config_dir:
        # Save the config YAML to a file in the config dir
        config_file = os.path.join(config_dir, "config.yml")
        with open(config_file, "w") as outfile:
            outfile.write(APP_MODEL_CONFIG_STRING)

        with patch("nhx.guardrails.app.services.configs.global_model_config_registry.settings") as mock_settings:
            mock_settings.demo = False
            mock_settings.config_store_path = None  #  Force the registry to load from argument
            mock_settings.fetch_nim_app_models = False

            reg = GlobalModelConfigRegistry(config_dir)

    reg_models = reg.list()
    reg_models_json = [m.model_dump() for m in reg_models]

    exp_models = APP_MODEL_CONFIG["models"]

    _check_registry_models(exp_models, reg_models_json)


@pytest.mark.skip()
def test_load_config_argument_demo_true():
    """Test when we set demo mode the full configs in config-store are loaded"""

    # Use this temp directory to copy global configs and also load via settings config_store_path
    with tempfile.TemporaryDirectory() as config_dir:
        # Save the config YAML to a file in the config dir

        with patch("nhx.guardrails.app.services.configs.global_model_config_registry.settings") as mock_settings:
            mock_settings.demo = True
            mock_settings.config_store_path = Path(config_dir)  # Force the registry to load from argument
            mock_settings.fetch_nim_app_models = False

            reg = GlobalModelConfigRegistry(config_dir)

    # Load the file we expect the registry to have loaded
    exp_models = CONFIG_STORE_MODEL_CONFIG["models"]
    reg_models = reg.list()
    reg_models_json = [m.model_dump() for m in reg_models]

    _check_registry_models(exp_models, reg_models_json)


def test_load_config_settings_demo_false():
    """Use Settings to point GlobalModelConfigRegistry at the config directory, don't pass argument"""

    with tempfile.TemporaryDirectory() as config_dir:
        # Save the config YAML to a file in the config dir
        config_file = os.path.join(config_dir, "config.yml")
        with open(config_file, "w") as outfile:
            outfile.write(APP_MODEL_CONFIG_STRING)

        with patch("nhx.guardrails.app.services.configs.global_model_config_registry.settings") as mock_settings:
            mock_settings.demo = False
            mock_settings.config_store_path = config_dir  #  Force the registry to load from argument
            mock_settings.fetch_nim_app_models = False

            reg = GlobalModelConfigRegistry(config_path=None)

    reg_models = reg.list()
    reg_models_json = [m.model_dump() for m in reg_models]
    exp_models = APP_MODEL_CONFIG["models"]

    _check_registry_models(exp_models, reg_models_json)


@pytest.mark.skip()
def test_load_config_settings_demo_true():
    """Load Model Configs using Demo mode, provide location in Settings singleton"""

    # Create a temporary directory, config-store configs will be copied into here as we
    # enabled Demo mode
    with tempfile.TemporaryDirectory() as config_dir:
        with patch("nhx.guardrails.app.services.configs.global_model_config_registry.settings") as mock_settings:
            mock_settings.demo = True
            mock_settings.config_store_path = Path(config_dir)  #  Force the registry to load from argument
            mock_settings.fetch_nim_app_models = False
            reg = GlobalModelConfigRegistry(config_path=None)

    reg_models = reg.list()
    reg_models_json = [m.model_dump() for m in reg_models]

    # Expect models to be loaded from YAML file copied from this location
    exp_models = CONFIG_STORE_MODEL_CONFIG["models"]
    _check_registry_models(exp_models, reg_models_json)


def test_check_model_exists_no_model(registry, nvidia_model1):
    """Check we get False back when no model exists in the registry"""
    assert not registry._check_model_exists(nvidia_model1.model_id)


def test_check_model_exists_with_model(registry, nvidia_model1):
    """Check we get True back when a model exists in the registry"""
    registry.add(nvidia_model1)
    assert registry._check_model_exists(nvidia_model1.model_id)


def test_get_endpoint_no_model(registry, nvidia_model1):
    """Cover the case where the model isn't in the registry and check we get None back"""
    assert registry.get_endpoint(nvidia_model1.model_id) is None


def test_get_endpoint_with_model(registry, nvidia_model1):
    """Cover the case where the model is in the registry"""
    registry.add(nvidia_model1)
    assert registry.get_endpoint(nvidia_model1.model_id) == nvidia_model1.base_url


def test_get_provider_no_model(registry, nvidia_model1):
    """Cover the case where the model isn't in the registry and check we get None back"""
    assert registry.get_provider(nvidia_model1.model_id) == DEFAULT_PROVIDER_NAME


def test_get_provider_with_model(registry, nvidia_model1):
    """Cover the case where the model is in the registry"""
    registry.add(nvidia_model1)

    # The Registry changes engines from nim to NIM_CHAT or NIM_LLM variables internally
    assert registry.get_provider(nvidia_model1.model_id) == DEFAULT_PROVIDER_NAME


def test_get_env_var_name_no_model(registry, nvidia_model1):
    """Cover the case where the model isn't in the registry and check we get None back"""
    assert registry.get_env_var_name(nvidia_model1.model_id) is None


def test_get_env_var_name_with_model(registry, nvidia_model1):
    """Cover the case where the model is in the registry"""
    registry.add(nvidia_model1)
    assert registry.get_env_var_name(nvidia_model1.model_id) is None


def test_get_api_key_name_no_model(registry, nvidia_model1):
    """Cover the case where the model isn't in the registry and check we get None back"""
    assert registry.get_api_key(nvidia_model1.model_id) is None


def test_get_api_key_name_with_model(registry, nvidia_model1, monkeypatch):
    """Cover the case where the model is in the registry"""
    # Clear environment API keys to test registry behavior in isolation
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    registry.add(nvidia_model1)
    assert registry.get_api_key(nvidia_model1.model_id) is None


def test_get_models_no_model(registry):
    """Get list of models when none are added to registry"""
    models = registry.get_models()
    assert models == []


def test_get_models_single_model(registry, nvidia_model1):
    """Get list of models when one is added to registry"""
    registry.add(nvidia_model1)
    models = registry.get_models()
    assert models[0]["id"] == nvidia_model1.model_id


def test_get_models_two_models(registry, nvidia_model1, nvidia_model2):
    """Get list of models when two are added to registry"""
    registry.add(nvidia_model1)
    registry.add(nvidia_model2)
    models = registry.get_models()
    assert models[0]["id"] == nvidia_model1.model_id
    assert models[1]["id"] == nvidia_model2.model_id


def test_standardize_provider_name_nimchat():
    """Check the NIM_CHAT is converted to just `nim`"""
    provider = _standardize_provider_name(NIM_CHAT)
    assert provider == "nim"


def test_standardize_provider_name_nimllm():
    """Check the NIM_CHAT is converted to just `nim`"""
    provider = _standardize_provider_name(NIM_LLM)
    assert provider == "nim"


def test_standardize_provider_name_nim():
    """Check the NIM_CHAT is converted to just `nim`"""
    provider = _standardize_provider_name("nim")
    assert provider == "nim"


def test_standardize_provider_name_not_nim():
    """Check the NIM_CHAT is converted to just `nim`"""
    provider = _standardize_provider_name("openai")
    assert provider == "openai"
