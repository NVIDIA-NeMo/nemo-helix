# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from pathlib import Path
from typing import Any

import pytest
from nemo_agent_optimization_plugin import job_base
from nemo_agent_optimization_plugin.job_base import AgentOptimizeJob
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import LocalJobResults
from nemo_platform_plugin.run_dependencies import LocalRunError


@pytest.fixture
def ctx(tmp_path: Path) -> JobContext:
    ephemeral = tmp_path / "ephemeral"
    persistent = tmp_path / "persistent"
    ephemeral.mkdir()
    persistent.mkdir()
    return JobContext(
        workspace="my-ws",
        storage=StoragePaths(ephemeral=ephemeral, persistent=persistent),
        results=LocalJobResults(root=persistent / "results"),
    )


SOURCE = {
    "config_format": "nemo-agents-spec-v1",
    "name": "my-agent",
    "default_harness": "hermes",
    "harnesses": {"hermes": {"kind": "hermes", "model": {"provider": "openai", "model": "m"}}},
}


class _RecordingJob(AgentOptimizeJob):
    name = "agent_optimize"
    strategy = "fake"

    def __init__(self) -> None:
        self.seen: dict[str, Any] = {}

    def optimize(self, *, source_agent_config, config, ctx, workspace, sdk):
        self.seen = {
            "source_agent_config": source_agent_config,
            "config": config,
            "workspace": workspace,
            "cwd": Path.cwd(),
        }
        return {**source_agent_config, "description": "optimized"}


def _spec(**overrides: Any) -> dict[str, Any]:
    return {
        "agent": "my-ws/my-agent",
        "optimize_config_fileset": "my-ws/bundle",
        "optimize_config": "configs/optimize.yaml",
        "output_agent": "my-agent-opt",
        "workspace": "my-ws",
        **overrides,
    }


@pytest.fixture
def staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bundle = tmp_path / "bundle"
    (bundle / "configs").mkdir(parents=True)
    (bundle / "configs" / "optimize.yaml").write_text("tuning: {a: 1}\n", encoding="utf-8")

    @contextlib.contextmanager
    def fake_staged_bundle(spec, *, ctx, sdk):
        yield bundle / "configs" / "optimize.yaml", bundle

    monkeypatch.setattr(job_base, "_staged_bundle", fake_staged_bundle)
    return bundle


@pytest.fixture
def registered(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_register(optimized, *, name, source_agent, source_workspace, workspace, sdk):
        calls.append(
            {
                "optimized": optimized,
                "name": name,
                "workspace": workspace,
                "source_agent": source_agent,
                "source_workspace": source_workspace,
            }
        )
        return {"agent": f"{workspace}/{name}"}

    monkeypatch.setattr(job_base, "register_optimized_agent", fake_register)
    monkeypatch.setattr(job_base, "fetch_agent_config", lambda agent, *, workspace, sdk: dict(SOURCE))
    return calls


def test_run_hands_the_subclass_the_source_config_and_loaded_yaml(ctx, staged, registered) -> None:
    job = _RecordingJob()
    job.run(_spec(), ctx=ctx, sdk=object())

    assert job.seen["source_agent_config"] == SOURCE
    assert job.seen["config"] == {"tuning": {"a": 1}}
    assert job.seen["workspace"] == "my-ws"


def test_run_executes_the_subclass_inside_the_bundle_root(ctx, staged, registered) -> None:
    job = _RecordingJob()
    before = Path.cwd()
    job.run(_spec(), ctx=ctx, sdk=object())

    assert job.seen["cwd"] == staged.resolve()
    assert Path.cwd() == before


def test_run_registers_the_subclass_output_as_the_named_agent(ctx, staged, registered) -> None:
    result = _RecordingJob().run(_spec(), ctx=ctx, sdk=object())

    assert result == {"agent": "my-ws/my-agent-opt"}
    assert registered[0]["name"] == "my-agent-opt"
    assert registered[0]["optimized"]["description"] == "optimized"


def test_run_passes_the_source_agent_identity_to_registration(ctx, staged, registered) -> None:
    """Registration stages the *source* agent's ETHOS.md, so it needs its name+workspace."""
    _RecordingJob().run(_spec(agent="other-ws/my-agent"), ctx=ctx, sdk=object())

    assert registered[0]["source_agent"] == "my-agent"
    assert registered[0]["source_workspace"] == "other-ws"


def test_run_defaults_the_source_workspace_to_the_run_workspace(ctx, staged, registered) -> None:
    _RecordingJob().run(_spec(agent="my-agent"), ctx=ctx, sdk=object())

    assert registered[0]["source_agent"] == "my-agent"
    assert registered[0]["source_workspace"] == "my-ws"


def test_run_requires_an_sdk(ctx, staged, registered) -> None:
    with pytest.raises(LocalRunError, match="requires a platform SDK"):
        _RecordingJob().run(_spec(), ctx=ctx, sdk=None)


def test_the_base_class_has_no_default_optimize(ctx) -> None:
    class Bare(AgentOptimizeJob):
        name = "agent_optimize"
        strategy = "bare"

    with pytest.raises(NotImplementedError):
        Bare().optimize(source_agent_config={}, config={}, ctx=ctx, workspace="my-ws", sdk=object())
