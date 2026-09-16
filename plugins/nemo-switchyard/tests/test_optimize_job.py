# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``switchyard`` agent-optimize job."""

from __future__ import annotations

from pathlib import Path
from typing import Any, get_args

import pytest
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import LocalJobResults
from nemo_platform_plugin.run_dependencies import LocalRunError
from nemo_switchyard.jobs.optimize import SwitchyardOptimizeJob
from nemo_switchyard.routing import (
    _PLATFORM_TO_SWITCHYARD_FORMAT,
    PlatformBackendFormat,
    SwitchyardConfig,
)
from pydantic import ValidationError


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


def test_the_job_declares_its_strategy() -> None:
    assert SwitchyardOptimizeJob.strategy == "switchyard"


def test_the_job_declares_its_task_module() -> None:
    assert SwitchyardOptimizeJob.task_module == "nemo_switchyard.tasks.agent_optimize"


def test_switchyard_config_rejects_missing_weak_tier() -> None:
    with pytest.raises(ValidationError):
        SwitchyardConfig.model_validate({"virtual_model": "r", "strong": {"model": "a"}})


def test_switchyard_config_rejects_probability_out_of_range() -> None:
    with pytest.raises(ValidationError):
        SwitchyardConfig.model_validate({**_config(), "strong_probability": 1.5})


def test_switchyard_config_accepts_a_complete_config() -> None:
    SwitchyardConfig.model_validate(_config())


def test_optimize_creates_virtual_model_with_switchyard_middleware(ctx: JobContext) -> None:
    sdk = _FakeSdk()
    optimized = SwitchyardOptimizeJob().optimize(
        source_agent_config=_source_agent_config(),
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
    assert optimized["models"]["default"]["model"] == "my-ws/my-agent-router"


def test_optimize_uses_each_backend_format_vocabulary_where_it_belongs(ctx: JobContext) -> None:
    """IGW's ``models`` list is uppercase; switchyard's tier config is lowercase."""
    sdk = _FakeSdk()
    config = {**_config(), "weak": {"model": "my-ws/claude-weak", "backend_format": "ANTHROPIC_MESSAGES"}}
    SwitchyardOptimizeJob().optimize(
        source_agent_config=_source_agent_config(),
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
    SwitchyardOptimizeJob().optimize(
        source_agent_config=_source_agent_config(),
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


def test_optimize_rewrites_the_default_model_when_no_harnesses_are_present(ctx: JobContext) -> None:
    source = {
        "config_format": "nemo-agents-spec-v1",
        "name": "my-agent",
        "models": {"default": {"provider": "nvidia", "model": "my-ws/llama-strong"}},
    }
    optimized = SwitchyardOptimizeJob().optimize(
        source_agent_config=source,
        config=_config(),
        ctx=ctx,
        workspace="my-ws",
        sdk=_FakeSdk(),
    )

    assert optimized["models"]["default"]["model"] == "my-ws/my-agent-router"


def test_optimize_rewrites_the_source_config_including_its_harness_model(ctx: JobContext) -> None:
    source = _source_agent_config()
    optimized = SwitchyardOptimizeJob().optimize(
        source_agent_config=source,
        config=_config(),
        ctx=ctx,
        workspace="my-ws",
        sdk=_FakeSdk(),
    )

    assert optimized["models"]["default"]["model"] == "my-ws/my-agent-router"
    assert optimized["harnesses"]["deepagents"]["model"]["model"] == "my-ws/my-agent-router"
    # The caller's mapping is an input, not scratch space.
    assert source["models"]["default"]["model"] == "my-ws/llama-strong"


def test_optimize_rewrites_a_harness_only_config(ctx: JobContext) -> None:
    """A harness model block alone is a rewrite target; no ``models.default`` is fine."""
    source = {
        "config_format": "nemo-agents-spec-v1",
        "name": "my-agent",
        "default_harness": "deepagents",
        "harnesses": {"deepagents": {"kind": "deepagents", "model": {"model": "my-ws/llama-strong"}}},
    }
    optimized = SwitchyardOptimizeJob().optimize(
        source_agent_config=source,
        config=_config(),
        ctx=ctx,
        workspace="my-ws",
        sdk=_FakeSdk(),
    )

    assert optimized["harnesses"]["deepagents"]["model"]["model"] == "my-ws/my-agent-router"


def test_optimize_refuses_to_report_success_when_no_model_parameter_was_rewritten(ctx: JobContext) -> None:
    """Returning the input unchanged as a "completed" optimization would be a silent failure."""
    unrecognized = {"config_format": "nemo-agents-spec-v1", "name": "my-agent", "models": {"judge": {"model": "x"}}}
    with pytest.raises(LocalRunError, match=r"models\.default\.model"):
        SwitchyardOptimizeJob().optimize(
            source_agent_config=unrecognized,
            config=_config(),
            ctx=ctx,
            workspace="my-ws",
            sdk=_FakeSdk(),
        )


def test_every_platform_backend_format_has_a_switchyard_mapping() -> None:
    assert set(_PLATFORM_TO_SWITCHYARD_FORMAT) == set(get_args(PlatformBackendFormat))


def test_optimize_falls_back_to_the_requested_name_when_the_sdk_returns_none(ctx: JobContext) -> None:
    """``VirtualModel.name`` is ``Optional[str]``; a ``None`` must not become "my-ws/None"."""
    sdk = _FakeSdk(_FakeVirtualModels(returned_name=None, use_requested_name=False))
    optimized = SwitchyardOptimizeJob().optimize(
        source_agent_config=_source_agent_config(),
        config=_config(),
        ctx=ctx,
        workspace="my-ws",
        sdk=sdk,
    )

    assert optimized["models"]["default"]["model"] == "my-ws/my-agent-router"
