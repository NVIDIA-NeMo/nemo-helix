# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Fabric-to-Strands bridge, piece by piece; no model is ever called."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("strands_harness_optimizer")

from strands.models.openai import OpenAIModel  # noqa: E402
from strands_harness_optimizer.datamodels import Reward, Rollout  # noqa: E402
from strands_harness_optimizer.formulas import SystemPromptFormula  # noqa: E402
from strands_harness_optimizer_plugin.strands_bridge import (  # noqa: E402
    ContainsExpectedReward,
    GatewayReflectionOptimizer,
    load_dataset,
    strands_agent_from_fabric,
    with_system_prompt,
)


@pytest.mark.parametrize("model_on_harness", [False, True])
def test_agent_from_fabric_takes_the_default_model_and_prompt(
    source_agent: dict[str, Any], model_on_harness: bool
) -> None:
    if model_on_harness:
        source_agent["harnesses"]["deepagents"]["model"] = source_agent.pop("models")["default"]

    agent = strands_agent_from_fabric(source_agent, base_url="http://gateway/v1", api_key="k")

    model = agent.model
    assert isinstance(model, OpenAIModel)
    assert agent.system_prompt == "old prompt"
    assert model.config == {"model_id": "calculator-model", "params": {"temperature": 0.0}}
    assert model.client_args == {"api_key": "k", "base_url": "http://gateway/v1"}


def test_with_system_prompt_copies(source_agent: dict[str, Any]) -> None:
    optimized = with_system_prompt(source_agent, "new prompt")

    assert optimized["instructions"]["system"]["content"] == "new prompt"
    assert source_agent["instructions"]["system"]["content"] == "old prompt"


def test_load_dataset_reads_jsonl_rows(tmp_path: Path) -> None:
    rows = [{"id": "a", "input": "1+1", "expected_output": "2"}, {"input": "2+2", "expected_output": "4"}]
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n\n", encoding="utf-8")

    assert load_dataset(path) == rows


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"input": "ok", "expected_output": "1"}\n{"input": "no answer"}\n', "line 2"),
        ('{"input": "ok", "expected_output": ""}\n', "line 1"),
        ("\n", "no rows"),
    ],
)
def test_load_dataset_rejects_bad_files(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_dataset(path)


@pytest.mark.parametrize(
    ("expected", "metadata", "reward"),
    [
        ("42", {"response_text": "The answer is 42."}, 1.0),
        ("hello world", {"response_text": "HELLO\n  WORLD!"}, 1.0),
        ("42", {"response_text": "forty-two"}, 0.0),
        ("42", {"error": "timeout"}, 0.0),
    ],
)
def test_contains_expected_reward(expected: str, metadata: dict[str, Any], reward: float) -> None:
    rollout = Rollout(data_sample={"input": "q", "expected_output": expected}, messages=[], metadata=metadata)

    assert ContainsExpectedReward()(rollout).reward == reward


@pytest.mark.parametrize(("proposal", "final"), [("  better prompt  ", "better prompt"), ("", "old prompt")])
def test_reflection_step_proposes_from_the_rollouts(proposal: str, final: str) -> None:
    formula = SystemPromptFormula(system_prompt="old prompt")
    seen: list[str] = []

    def propose(prompt: str) -> str:
        seen.append(prompt)
        return proposal

    optimizer = GatewayReflectionOptimizer(formula, propose)
    optimizer.add_rollouts(
        [
            Rollout(data_sample={"input": "1+1", "expected_output": "2"}, messages=[], metadata={"response_text": "2"}),
            Rollout(data_sample={"input": "2+2", "expected_output": "4"}, messages=[], metadata={"response_text": "5"}),
        ]
    )
    optimizer.add_rewards([Reward(reward=1.0), Reward(reward=0.0)])
    optimizer.step()

    assert formula.get_tunable_params() == {"system_prompt": final}
    (prompt,) = seen
    assert "Current system prompt:\nold prompt" in prompt
    assert "Input: 2+2\nExpected: 4\nResponse: 5\nReward: 0.0" in prompt
    assert prompt.endswith("Reply with ONLY the new system prompt.")
