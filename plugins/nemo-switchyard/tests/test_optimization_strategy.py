# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``switchyard`` ``nemo agents optimize`` strategy."""

from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import pytest
import yaml
from nemo_agent_optimization_plugin.strategies import PRIMARY_ARTIFACT_KEY, OptimizationStrategy
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import LocalJobResults
from nemo_switchyard.optimization_strategy import SwitchyardOptimizationStrategy


class _FakeVirtualModel:
    def __init__(self, name: str | None) -> None:
        self.name = name


class _FakeVirtualModels:
    def __init__(self, returned_name: str | None = None, use_requested_name: bool = True) -> None:
        self.calls: list[dict[str, Any]] = []
        self._returned_name = returned_name
        self._use_requested_name = use_requested_name

    def create(self, **kwargs: Any) -> _FakeVirtualModel:
        self.calls.append(kwargs)
        return _FakeVirtualModel(kwargs["name"] if self._use_requested_name else self._returned_name)


class _FakeInference:
    def __init__(self, virtual_models: _FakeVirtualModels | None = None) -> None:
        self.virtual_models = virtual_models or _FakeVirtualModels()


class _FakeSdk:
    def __init__(self, virtual_models: _FakeVirtualModels | None = None) -> None:
        self.inference = _FakeInference(virtual_models)


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


def _agent_config() -> dict[str, Any]:
    return {
        "schema_version": "fabric.agent/v1alpha1",
        "name": "my-agent",
        "models": {"default": {"provider": "nvidia", "model": "my-ws/llama-strong"}},
    }


def _source_agent_config() -> dict[str, Any]:
    return {
        "config_format": "nemo-agents-spec-v1",
        "name": "my-agent",
        "default_harness": "deepagents",
        "harnesses": {
            "deepagents": {
                "kind": "deepagents",
                "model": {"provider": "nvidia", "model": "my-ws/llama-strong"},
            }
        },
        "models": {"default": {"provider": "nvidia", "model": "my-ws/llama-strong"}},
    }


def _config() -> dict[str, Any]:
    return {
        "virtual_model": "my-agent-router",
        "strong": {"model": "my-ws/llama-strong"},
        "weak": {"model": "my-ws/llama-weak"},
        "strong_probability": 0.3,
    }


def test_strategy_is_registered_as_an_optimization_entrypoint() -> None:
    entry = next(entry for entry in entry_points(group="nemo.optimization-strategy") if entry.name == "switchyard")

    assert entry.load() is SwitchyardOptimizationStrategy


def test_satisfies_optimization_strategy_protocol() -> None:
    assert isinstance(SwitchyardOptimizationStrategy(), OptimizationStrategy)


def test_name_is_switchyard() -> None:
    assert SwitchyardOptimizationStrategy().name == "switchyard"


def test_validate_config_rejects_missing_agent() -> None:
    with pytest.raises(ValueError, match="--agent"):
        SwitchyardOptimizationStrategy().validate_config(_config(), agent=None)


def test_validate_config_rejects_missing_weak_tier() -> None:
    strategy = SwitchyardOptimizationStrategy()
    with pytest.raises(ValueError):
        strategy.validate_config({"virtual_model": "r", "strong": {"model": "a"}}, agent="my-agent")


def test_validate_config_rejects_probability_out_of_range() -> None:
    strategy = SwitchyardOptimizationStrategy()
    bad = {**_config(), "strong_probability": 1.5}
    with pytest.raises(ValueError):
        strategy.validate_config(bad, agent="my-agent")


def test_validate_config_accepts_a_complete_config() -> None:
    SwitchyardOptimizationStrategy().validate_config(_config(), agent="my-agent")


def test_run_creates_virtual_model_with_switchyard_middleware(ctx: JobContext) -> None:
    sdk = _FakeSdk()
    strategy = SwitchyardOptimizationStrategy()
    result = strategy.run(
        agent_config=_agent_config(),
        source_agent_config=None,
        config=_config(),
        ctx=ctx,
        workspace="my-ws",
        sdk=sdk,
    )

    (call,) = sdk.inference.virtual_models.calls
    assert call["workspace"] == "my-ws"
    assert call["name"] == "my-agent-router"
    assert call["exist_ok"] is True
    assert {entry["model"] for entry in call["models"]} == {"my-ws/llama-strong", "my-ws/llama-weak"}
    (middleware,) = call["request_middleware"]
    assert middleware["name"] == "nemo-switchyard"
    assert middleware["config_type"] == "random_routing"
    assert middleware["config"]["strong"]["model"] == "my-ws/llama-strong"
    assert middleware["config"]["weak"]["model"] == "my-ws/llama-weak"
    assert middleware["config"]["strong_probability"] == 0.3
    assert result["status"] == "completed"
    assert result["strategy"] == "switchyard"
    assert result["virtual_model"] == "my-ws/my-agent-router"
    assert result["result"]["name"] == "switchyard_results"


def test_run_uses_each_backend_format_vocabulary_where_it_belongs(ctx: JobContext) -> None:
    """IGW's ``models`` list is uppercase; switchyard's tier config is lowercase."""
    sdk = _FakeSdk()
    config = {**_config(), "weak": {"model": "my-ws/claude-weak", "backend_format": "ANTHROPIC_MESSAGES"}}
    SwitchyardOptimizationStrategy().run(
        agent_config=_agent_config(),
        source_agent_config=None,
        config=config,
        ctx=ctx,
        workspace="my-ws",
        sdk=sdk,
    )

    (call,) = sdk.inference.virtual_models.calls
    assert {entry["model"]: entry["backend_format"] for entry in call["models"]} == {
        "my-ws/llama-strong": "OPENAI_CHAT",
        "my-ws/claude-weak": "ANTHROPIC_MESSAGES",
    }
    (middleware,) = call["request_middleware"]
    assert middleware["config"]["strong"]["backend_format"] == "openai"
    assert middleware["config"]["weak"]["backend_format"] == "anthropic"


def test_middleware_config_validates_against_the_real_switchyard_factory(ctx: JobContext) -> None:
    """The payload must survive ``RandomRoutingFactory.validate`` or IGW rejects the VM upsert."""
    from switchyard.lib.factories.random_routing.factory import RandomRoutingFactory

    sdk = _FakeSdk()
    SwitchyardOptimizationStrategy().run(
        agent_config=_agent_config(),
        source_agent_config=None,
        config={**_config(), "rng_seed": 7},
        ctx=ctx,
        workspace="my-ws",
        sdk=sdk,
    )

    (call,) = sdk.inference.virtual_models.calls
    (middleware,) = call["request_middleware"]
    validated = RandomRoutingFactory().validate(dict(middleware["config"]))

    assert validated.strong.model == "my-ws/llama-strong"
    assert validated.weak.model == "my-ws/llama-weak"
    assert validated.strong_probability == 0.3
    assert validated.rng_seed == 7


def test_run_rewrites_agent_model_to_the_virtual_model(ctx: JobContext) -> None:
    strategy = SwitchyardOptimizationStrategy()
    result = strategy.run(
        agent_config=_agent_config(),
        source_agent_config=None,
        config=_config(),
        ctx=ctx,
        workspace="my-ws",
        sdk=_FakeSdk(),
    )

    written = yaml.safe_load(Path(result[PRIMARY_ARTIFACT_KEY]).read_text(encoding="utf-8"))
    assert written["models"]["default"]["model"] == "my-ws/my-agent-router"
    assert result[PRIMARY_ARTIFACT_KEY] == str(
        ctx.storage.persistent / "results" / "switchyard_results" / "optimized_config.yml"
    )


def test_run_rewrites_the_source_config_including_its_harness_model(ctx: JobContext) -> None:
    agent_config = _agent_config()
    source_agent_config = _source_agent_config()
    result = SwitchyardOptimizationStrategy().run(
        agent_config=agent_config,
        source_agent_config=source_agent_config,
        config=_config(),
        ctx=ctx,
        workspace="my-ws",
        sdk=_FakeSdk(),
    )

    written = yaml.safe_load(Path(result[PRIMARY_ARTIFACT_KEY]).read_text(encoding="utf-8"))
    assert written["config_format"] == "nemo-agents-spec-v1"
    assert written["models"]["default"]["model"] == "my-ws/my-agent-router"
    assert written["harnesses"]["deepagents"]["model"]["model"] == "my-ws/my-agent-router"
    # The caller's mappings are inputs, not scratch space.
    assert source_agent_config["models"]["default"]["model"] == "my-ws/llama-strong"
    assert agent_config["models"]["default"]["model"] == "my-ws/llama-strong"


def test_run_falls_back_to_the_requested_name_when_the_sdk_returns_none(ctx: JobContext) -> None:
    """``VirtualModel.name`` is ``Optional[str]``; a ``None`` must not become "my-ws/None"."""
    sdk = _FakeSdk(_FakeVirtualModels(returned_name=None, use_requested_name=False))
    result = SwitchyardOptimizationStrategy().run(
        agent_config=_agent_config(),
        source_agent_config=None,
        config=_config(),
        ctx=ctx,
        workspace="my-ws",
        sdk=sdk,
    )

    assert result["virtual_model"] == "my-ws/my-agent-router"


def test_run_without_an_agent_is_a_clear_error(ctx: JobContext) -> None:
    with pytest.raises(Exception, match="--agent"):
        SwitchyardOptimizationStrategy().run(
            agent_config=None,
            source_agent_config=None,
            config=_config(),
            ctx=ctx,
            workspace="my-ws",
            sdk=_FakeSdk(),
        )


def test_run_without_sdk_is_a_clear_error(ctx: JobContext) -> None:
    strategy = SwitchyardOptimizationStrategy()
    with pytest.raises(Exception, match="SDK"):
        strategy.run(
            agent_config=_agent_config(),
            source_agent_config=None,
            config=_config(),
            ctx=ctx,
            workspace="my-ws",
            sdk=None,
        )
