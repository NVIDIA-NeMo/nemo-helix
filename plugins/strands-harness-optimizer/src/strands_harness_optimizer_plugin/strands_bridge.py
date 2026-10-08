# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Strands Harness Optimizer's training loop over a ``nemo-agents-spec-v1`` config.

Nothing here knows about the platform: the config is a plain dict, the model is reached as an
OpenAI-compatible endpoint at *base_url*, and the only thing optimized is the system prompt.
The job imports this module lazily so the plugin's job entry imports without Strands installed.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from strands import Agent
from strands.models.openai import OpenAIModel
from strands_harness_optimizer.adapters import apply_formulas_on_strands_agent
from strands_harness_optimizer.data import DataLoader
from strands_harness_optimizer.datamodels import Reward, Rollout
from strands_harness_optimizer.formulas import SystemPromptFormula
from strands_harness_optimizer.optimizers import FormulaOptimizer
from strands_harness_optimizer.rewards import RewardFunction
from strands_harness_optimizer.rollout_engines import LocalRolloutEngine
from strands_harness_optimizer.trainer import Trainer
from strands_harness_optimizer_plugin.config import StrandsHarnessConfig

#: ``ModelConfig`` fields that are OpenAI chat-completion parameters.
_MODEL_PARAMS = ("temperature", "top_p", "max_tokens")


@dataclass(frozen=True, slots=True)
class Outcome:
    original_prompt: str
    optimized_prompt: str
    #: ``Trainer.fit()``'s per-epoch ``{"epoch", "avg_reward"}`` rows.
    epoch_stats: list[dict[str, Any]]


def _model_block(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("models", {}).get("default") or config["harnesses"][config["default_harness"]]["model"]


def _openai_model(block: dict[str, Any], *, base_url: str, api_key: str, model_id: str | None = None) -> OpenAIModel:
    return OpenAIModel(
        client_args={"api_key": api_key, "base_url": base_url},
        model_id=model_id or block["model"],
        params={key: block[key] for key in _MODEL_PARAMS if block.get(key) is not None},
    )


def strands_agent_from_fabric(config: dict[str, Any], *, base_url: str, api_key: str) -> Agent:
    """A Strands agent on the config's default model and system prompt; tools and MCP servers are not carried over."""
    return Agent(
        model=_openai_model(_model_block(config), base_url=base_url, api_key=api_key),
        system_prompt=config["instructions"]["system"]["content"],
        callback_handler=None,
    )


def with_system_prompt(config: dict[str, Any], prompt: str) -> dict[str, Any]:
    optimized = copy.deepcopy(config)
    optimized["instructions"]["system"]["content"] = prompt
    return optimized


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path.name} has no rows")
    for number, row in enumerate(rows, 1):
        if not (row.get("input") and row.get("expected_output")):
            raise ValueError(f"{path.name} line {number}: every row needs a non-empty 'input' and 'expected_output'")
    return rows


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


class ContainsExpectedReward(RewardFunction):
    """1.0 when the expected output appears in the response, ignoring case and whitespace runs."""

    def __call__(self, rollout: Rollout) -> Reward:
        expected = _normalize(rollout.data_sample["expected_output"])
        response = _normalize(rollout.metadata.get("response_text", ""))
        return Reward(reward=1.0 if expected in response else 0.0)


class GatewayReflectionOptimizer(FormulaOptimizer):
    """Propose the next system prompt from one contrastive look at the epoch's rollouts.

    *propose* turns the reflection prompt into the model's reply; it is a callable rather
    than a Bedrock client so the reflection runs on whatever OpenAI-compatible model the
    caller wires up.
    """

    def __init__(self, formula: SystemPromptFormula, propose: Callable[[str], str]) -> None:
        super().__init__(formula)
        self._propose = propose

    def step(self) -> None:
        current = self.formula.get_tunable_params()["system_prompt"]
        cases = "\n\n".join(
            f"Input: {rollout.data_sample['input']}\n"
            f"Expected: {rollout.data_sample['expected_output']}\n"
            f"Response: {rollout.metadata.get('response_text', rollout.metadata.get('error', ''))}\n"
            f"Reward: {reward.reward}"
            for rollout, reward in zip(self._rollouts, self._rewards, strict=True)
        )
        proposal = self._propose(
            "You are improving the system prompt of an LLM agent.\n\n"
            f"Current system prompt:\n{current}\n\n"
            "Rollouts under that prompt (reward 1.0 means the response contained the expected output):\n\n"
            f"{cases}\n\n"
            "Contrast the rewarded rollouts with the unrewarded ones, then write an improved system prompt that "
            "keeps the agent's intent and fixes the failures.  Reply with ONLY the new system prompt."
        ).strip()
        if proposal:
            self.formula.update_params({"system_prompt": proposal})


def optimize_system_prompt(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    base_url: str,
    api_key: str,
    settings: StrandsHarnessConfig,
) -> Outcome:
    """Train ``SystemPromptFormula`` on *rows* and return the prompt it ends on."""
    original = config["instructions"]["system"]["content"]
    formula = SystemPromptFormula(system_prompt=original)

    def create_agent() -> Agent:
        return apply_formulas_on_strands_agent(
            strands_agent_from_fabric(config, base_url=base_url, api_key=api_key), [formula]
        )

    def invoke_agent(agent: Agent, data_sample: dict[str, Any]) -> Rollout:
        agent.messages.clear()
        text = str(agent(data_sample["input"]))
        return Rollout(
            data_sample=data_sample,
            messages=[dict(message) for message in agent.messages],
            metadata={"response_text": text},
        )

    reflector = Agent(
        model=_openai_model(
            _model_block(config), base_url=base_url, api_key=api_key, model_id=settings.optimizer_model
        ),
        callback_handler=None,
    )

    def propose(prompt: str) -> str:
        reflector.messages.clear()
        return str(reflector(prompt))

    stats = Trainer(
        formula=formula,
        optimizer=GatewayReflectionOptimizer(formula, propose),
        reward_fn=ContainsExpectedReward(),
        engine=LocalRolloutEngine(
            formula=formula,
            agent_creator=create_agent,
            agent_invoker=invoke_agent,
            num_workers=settings.num_workers,
        ),
        dataloader=DataLoader(rows, batch_size=settings.batch_size),  # ty: ignore[invalid-argument-type]  (any indexable sequence works)
        n_epochs=settings.epochs,
    ).fit()
    return Outcome(
        original_prompt=original,
        optimized_prompt=formula.get_tunable_params()["system_prompt"],
        epoch_stats=stats,
    )
