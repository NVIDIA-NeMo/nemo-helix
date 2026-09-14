# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import entry_points
from pathlib import Path

import pytest
import yaml
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import LocalJobResults
from prompt_master_plugin.strategy import PromptMasterStrategy


@pytest.fixture
def ctx(tmp_path: Path) -> JobContext:
    persistent = tmp_path / "persistent"
    ephemeral = tmp_path / "ephemeral"
    persistent.mkdir()
    ephemeral.mkdir()
    return JobContext(
        workspace="default",
        storage=StoragePaths(ephemeral=ephemeral, persistent=persistent),
        results=LocalJobResults(root=persistent / "results"),
    )


def _agent_config() -> dict:
    return {
        "schema_version": "fabric.agent/v1alpha1",
        "metadata": {"name": "calculator-agent"},
        "harness": {"adapter_id": "nvidia.fabric.langchain.deepagents"},
        "models": {"default": {"provider": "nvidia", "model": "calculator-model"}},
        "instructions": {"system": {"content": "Return only the numeric answer."}},
    }


def _source_agent_config() -> dict:
    return {
        "config_format": "nemo-agents-spec-v1",
        "name": "calculator-agent",
        "default_harness": "deepagents",
        "harnesses": {"deepagents": {"kind": "deepagents", "settings": {"deepagents": {}}}},
        "models": {"default": {"provider": "nvidia", "model": "calculator-model"}},
        "instructions": {"system": {"content": "Return only the numeric answer."}},
    }


def test_strategy_is_registered_as_an_optimization_entrypoint() -> None:
    entry = next(entry for entry in entry_points(group="nemo.optimization-strategy") if entry.name == "prompt-master")

    assert entry.load() is PromptMasterStrategy


def test_satisfies_optimization_strategy_protocol() -> None:
    from nemo_agent_optimization_plugin.strategies import OptimizationStrategy

    assert isinstance(PromptMasterStrategy(), OptimizationStrategy)


def test_strategy_requires_a_platform_agent() -> None:
    with pytest.raises(ValueError, match="--agent"):
        PromptMasterStrategy().validate_config(
            {"model": {"provider": "nvidia", "model": "optimizer-model"}},
            agent=None,
        )


def test_strategy_writes_an_optimized_fabric_config(
    ctx: JobContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "prompt_master_plugin.strategy.run_prompt_master",
        lambda config, agent_config, base_dir: "Return the exact numeric answer with no explanation.",
    )
    agent_config = _agent_config()
    strategy = PromptMasterStrategy()
    config = {
        "model": {
            "provider": "nvidia",
            "model": "optimizer-model",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "api_key_env": "NVIDIA_API_KEY",
        }
    }

    result = strategy.run(
        agent_config=agent_config,
        source_agent_config=_source_agent_config(),
        config=config,
        ctx=ctx,
        workspace="default",
    )

    artifact = ctx.storage.persistent / "results" / "prompt_master_results" / "optimized_config.yml"
    optimized = yaml.safe_load(artifact.read_text(encoding="utf-8"))
    assert optimized["config_format"] == "nemo-agents-spec-v1"
    assert optimized["instructions"]["system"]["content"] == "Return the exact numeric answer with no explanation."
    assert agent_config["instructions"]["system"]["content"] == "Return only the numeric answer."
    assert result["status"] == "completed"
    assert result["strategy"] == "prompt-master"
    assert result["agent"] == "calculator-agent"
    assert result["result"]["name"] == "prompt_master_results"
    assert result["_primary_artifact"] == str(artifact)
