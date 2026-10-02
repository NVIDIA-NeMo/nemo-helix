# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Gym component that runs a registered agent: what it changes in the config and what it leaves alone.

``nemo_gym`` and ``nemo_fabric`` are not installed here (Gym cannot be a repo dependency), so the
module is loaded with stand-ins for the names it imports; only its pure functions are exercised.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

_APP = Path(__file__).parents[1] / "src" / "nemo_evaluator" / "gym_registered_agent" / "app.py"


def _stub(name: str, **attrs: Any) -> None:
    module = sys.modules.get(name) or types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


class _Item(BaseModel, extra="allow"):
    ray_enabled: ClassVar[bool] = True


@pytest.fixture(scope="module")
def app():
    for name in ("nemo_gym", "nemo_fabric", "fastapi"):
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                pass
    stand_ins = {
        "fastapi": {"Body": lambda *a, **k: None, "Request": object},
        "nemo_fabric": {"Fabric": object, "FabricConfig": object, "RunRequest": object},
        "nemo_gym": {},
        "nemo_gym.base_resources_server": {
            "NEMO_GYM_MCP_METADATA_KEY": "mcp",
            "BaseRunRequest": _Item,
            "BaseVerifyResponse": _Item,
        },
        "nemo_gym.base_responses_api_agent": {"BaseResponsesAPIAgentConfig": _Item, "SimpleResponsesAPIAgent": _Item},
        "nemo_gym.config_types": {"ModelServerRef": _Item, "ResourcesServerRef": _Item},
        "nemo_gym.global_config": {"SKILLS_REF_KEY_NAME": "skills_ref", "get_first_server_config_dict": lambda *a: {}},
        "nemo_gym.openai_utils": {
            name: type(name, (_Item,), {})
            for name in (
                "NeMoGymResponse",
                "NeMoGymResponseCreateParamsNonStreaming",
                "NeMoGymResponseFunctionToolCall",
                "NeMoGymResponseInputTokensDetails",
                "NeMoGymResponseOutputMessage",
                "NeMoGymResponseOutputText",
                "NeMoGymResponseOutputTokensDetails",
                "NeMoGymResponseReasoningItem",
                "NeMoGymResponseUsage",
                "NeMoGymSummary",
                "NeMoGymFunctionCallOutput",
            )
        },
        "nemo_gym.server_utils": {"get_response_json": None, "raise_for_status": None},
    }
    for name, attrs in stand_ins.items():
        if name not in sys.modules or not all(hasattr(sys.modules[name], k) for k in attrs):
            _stub(name, **attrs)
    spec = importlib.util.spec_from_file_location("nemo_registered_agent_app_under_test", _APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_REGISTERED = {
    "metadata": {"name": "calculator-agent"},
    "harness": {"adapter_id": "nvidia.fabric.langchain.deepagents", "settings": {"x": 1}},
    "instructions": {"system": {"content": "You add numbers.", "mode": "replace"}},
    "models": {
        "default": {
            "provider": "openai-compatible",
            "model": "nvidia/nemotron",
            "base_url": "http://igw/v1",
            "api_key_env": "NEMO_AGENTS_IGW_API_KEY",
            "temperature": 0.2,
        },
        "judge": {"provider": "openai-compatible", "model": "other", "base_url": "http://igw/v1", "api_key_env": "K"},
    },
    "environment": {"provider": "local", "env": {"NEMO_AGENTS_IGW_API_KEY": "not-used"}},
    "mcp": {
        "servers": {
            "calc": {
                "transport": "stdio",
                "url": "python",
                "args": ["-m", "calc"],
                "env": {"CALC_TOKEN": "${CALC_TOKEN}", "MODE": "strict"},
            }
        }
    },
    "skills": {"paths": ["skills/arithmetic"]},
}


def _compose(app, **overrides):
    kwargs = dict(
        model_name="gym-policy-model",
        model_base_url="http://policy:5000/v1",
        model_api_key="local",
        workspace="/tmp/rollout",
        system_prompt=None,
        mcp_servers={},
        skills=[],
        environ={},
    )
    kwargs.update(overrides)
    return app.compose_fabric_config(_REGISTERED, **kwargs)


def test_only_the_default_model_is_rebound_to_the_policy_server(app) -> None:
    composed = _compose(app)
    default = composed["models"]["default"]
    assert (default["model"], default["base_url"], default["api_key_env"]) == (
        "gym-policy-model",
        "http://policy:5000/v1",
        app.POLICY_MODEL_API_KEY_ENV,
    )
    assert default["temperature"] == 0.2  # the agent's own sampling settings survive
    assert composed["models"]["judge"] == _REGISTERED["models"]["judge"]  # secondary models still go where they did
    assert composed["environment"]["env"][app.POLICY_MODEL_API_KEY_ENV] == "local"
    assert composed["environment"]["workspace"] == "/tmp/rollout"


def test_identity_harness_and_instructions_are_the_registered_agents(app) -> None:
    composed = _compose(app)
    assert composed["metadata"] == {"name": "calculator-agent"}
    assert composed["harness"] == _REGISTERED["harness"]
    assert composed["instructions"]["system"]["content"] == "You add numbers."
    with_gym_prompt = _compose(app, system_prompt="Answer tersely.")
    assert with_gym_prompt["instructions"]["system"] == {
        "content": "You add numbers.\n\nAnswer tersely.",
        "mode": "replace",
    }


def test_stdio_secret_templates_are_filled_from_the_process_env_and_gym_servers_and_skills_are_added(app) -> None:
    composed = _compose(
        app,
        environ={"CALC_TOKEN": "tok-42"},
        mcp_servers={"mcqa": {"transport": "streamable-http", "url": "http://rs:7000/mcp"}},
        skills=["/skills/from-gym"],
    )
    assert composed["mcp"]["servers"]["calc"]["env"] == {"CALC_TOKEN": "tok-42", "MODE": "strict"}
    assert composed["mcp"]["servers"]["mcqa"]["url"] == "http://rs:7000/mcp"
    assert composed["skills"]["paths"] == ["skills/arithmetic", "/skills/from-gym"]
    untouched = _compose(app)  # a template with no value stays a template rather than becoming empty
    assert untouched["mcp"]["servers"]["calc"]["env"]["CALC_TOKEN"] == "${CALC_TOKEN}"
    assert _REGISTERED["mcp"]["servers"]["calc"]["env"]["CALC_TOKEN"] == "${CALC_TOKEN}"  # the input is not mutated


def test_fabric_messages_become_responses_items_ending_with_the_answer(app) -> None:
    output = {
        "messages": [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "add", "args": {"a": 2, "b": 2}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "4"},
            {"role": "assistant", "content": "4"},
        ]
    }
    items = app.fabric_output_items(output, "4")
    kinds = [type(item).__name__ for item in items]
    assert kinds == ["NeMoGymResponseFunctionToolCall", "NeMoGymFunctionCallOutput", "NeMoGymResponseOutputMessage"]
    assert items[0].arguments == '{"a": 2, "b": 2}' and items[1].call_id == "c1"


def test_a_request_without_input_runs_the_agent_on_an_empty_prompt(app) -> None:
    assert app._extract_request_input(None) == ("", None)
    assert app._extract_request_input("2+2?") == ("2+2?", None)
