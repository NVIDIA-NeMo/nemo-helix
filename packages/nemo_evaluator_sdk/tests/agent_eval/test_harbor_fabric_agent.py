# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

pytest.importorskip("harbor", reason="NemoFabricAgent subclasses Harbor's BaseAgent")

from nemo_evaluator_sdk.agent_eval.runtimes.harbor_fabric_agent import NVIDIA_MODEL_BASE_URL, NemoFabricAgent

_DEEPAGENTS = "nvidia.fabric.langchain.deepagents"


def _agent(tmp_path: Path, **kwargs: str) -> NemoFabricAgent:
    return NemoFabricAgent(logs_dir=tmp_path, fabric_adapter_id=_DEEPAGENTS, fabric_workspace="/app", **kwargs)


def test_nvidia_model_gets_its_credential_name_endpoint_and_full_id(tmp_path: Path) -> None:
    """The deepagents adapter refuses a non-OpenAI provider without `api_key_env` and `base_url`.

    build.nvidia.com ids carry the `nvidia/` namespace, so unlike `openai/` the prefix must survive.
    """
    model = _agent(tmp_path, model_name="nvidia/nemotron-3-super-120b-a12b")._build_config().models["default"]

    assert model.provider == "nvidia"
    assert model.model == "nvidia/nemotron-3-super-120b-a12b"
    assert model.api_key_env == "NVIDIA_API_KEY"
    assert model.base_url == NVIDIA_MODEL_BASE_URL


def test_openai_model_drops_the_provider_prefix_like_the_codex_adapter(tmp_path: Path) -> None:
    """The deepagents adapter sends `model` verbatim, so `openai/gpt-5.4` would reach the API as-is."""
    model = _agent(tmp_path, model_name="openai/gpt-5.4")._build_config().models["default"]

    assert (model.provider, model.model, model.api_key_env, model.base_url) == (
        "openai",
        "gpt-5.4",
        "OPENAI_API_KEY",
        None,
    )


def test_explicit_credential_name_and_endpoint_win(tmp_path: Path) -> None:
    agent = _agent(
        tmp_path,
        model_name="nvidia/nemotron-3-super-120b-a12b",
        fabric_model_api_key_env="MY_NIM_KEY",
        fabric_model_base_url="https://nim.internal/v1",
    )

    model = agent._build_config().models["default"]

    assert model.api_key_env == "MY_NIM_KEY"
    assert model.base_url == "https://nim.internal/v1"


def test_unknown_provider_requires_a_credential_name(tmp_path: Path) -> None:
    """Guessing a variable name would send a key to the wrong endpoint; refuse instead."""
    with pytest.raises(ValueError, match="fabric_model_api_key_env is required for model provider 'anthropic'"):
        _agent(tmp_path, model_name="anthropic/claude-sonnet-5")._build_config()


def test_blank_credential_name_is_rejected_at_construction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fabric_model_api_key_env"):
        _agent(tmp_path, model_name="nvidia/nemotron-3-super-120b-a12b", fabric_model_api_key_env="  ")


def test_no_model_leaves_the_config_untouched(tmp_path: Path) -> None:
    assert "default" not in _agent(tmp_path)._build_config().models


def test_reports_its_own_name_to_harbor(tmp_path: Path) -> None:
    assert NemoFabricAgent.name() == "nemo-fabric"
