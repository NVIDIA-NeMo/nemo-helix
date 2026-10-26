# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import pytest
import yaml
from nemo_agents_plugin.fabric.translator import FabricTranslationError
from nemo_optimization.agents import _to_fabric_agent_package


@pytest.fixture
def stored_agent() -> dict[str, Any]:
    example = Path(__file__).parents[1] / "examples/hermes-optimize/agents/chatonly/agent.yaml"
    return yaml.safe_load(example.read_text())


@pytest.mark.parametrize(
    "extensions",
    [None, {}, {"api": "openai-completions", "context_window": 32768}],
    ids=["omitted", "empty", "metadata"],
)
def test_named_model_extensions_are_flattened(stored_agent: dict[str, Any], extensions: dict[str, Any] | None) -> None:
    judge = stored_agent["models"]["judge"]
    if extensions is not None:
        judge["extensions"] = extensions
    del judge["api_key_env"]

    package = _to_fabric_agent_package(stored_agent, label="stored-agent")

    translated = package["models"]["judge"]
    assert "extensions" not in translated
    assert translated["model"] == judge["model"]
    assert translated["settings"] == judge["settings"]
    assert translated["api_key_env"] == "NEMO_AGENTS_IGW_API_KEY"
    assert package["environment"]["env"]["NEMO_AGENTS_IGW_API_KEY"] == "not-used"
    for key, value in (extensions or {}).items():
        assert translated[key] == value
    assert package["models"]["default"]["model"] == stored_agent["harnesses"]["hermes"]["model"]["model"]


def test_named_model_legacy_base_url_is_normalized(stored_agent: dict[str, Any]) -> None:
    judge = stored_agent["models"]["judge"]
    base_url = judge.pop("base_url")
    judge["settings"]["base_url"] = base_url

    package = _to_fabric_agent_package(stored_agent, label="stored-agent")

    assert package["models"]["judge"]["base_url"] == base_url
    assert "base_url" not in package["models"]["judge"]["settings"]


@pytest.mark.parametrize("field", ["max_tokens", "extensions"])
def test_named_model_extensions_cannot_shadow_declared_fields(stored_agent: dict[str, Any], field: str) -> None:
    stored_agent["models"]["judge"]["extensions"] = {field: 512}

    with pytest.raises(FabricTranslationError, match=field):
        _to_fabric_agent_package(stored_agent, label="stored-agent")
