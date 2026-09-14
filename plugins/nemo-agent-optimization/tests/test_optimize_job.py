# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from nemo_agent_optimization_plugin.jobs.optimize import OptimizeJob
from nemo_agent_optimization_plugin.strategies import PRIMARY_ARTIFACT_KEY, discover_optimization_strategies
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import LocalJobResults

AGENT_CONFIG = {"schema_version": "fabric.agent/v1alpha1"}


class _FakeStrategy:
    name = "fake"

    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.result = result if result is not None else {"status": "completed", "strategy": "fake"}
        self.validated: list[tuple[dict[str, Any], str | None]] = []
        self.run_kwargs: dict[str, Any] = {}

    def validate_config(self, config: dict[str, Any], *, agent: str | None) -> None:
        self.validated.append((config, agent))

    def run(self, **kwargs: Any) -> dict[str, Any]:
        self.run_kwargs = kwargs
        return dict(self.result)


def _ctx(root: Path) -> JobContext:
    """A real JobContext rooted at *root* (no mocks)."""
    persistent = root / "persistent"
    ephemeral = root / "ephemeral"
    persistent.mkdir(exist_ok=True)
    ephemeral.mkdir(exist_ok=True)
    return JobContext(
        workspace="default",
        storage=StoragePaths(ephemeral=ephemeral, persistent=persistent),
        results=LocalJobResults(root=persistent / "results"),
    )


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"any": "config"}), encoding="utf-8")
    return config_path


@pytest.fixture
def strategy() -> _FakeStrategy:
    return _FakeStrategy()


@pytest.fixture(autouse=True)
def _register_fake_strategy(monkeypatch: pytest.MonkeyPatch, strategy: _FakeStrategy) -> None:
    discover_optimization_strategies.cache_clear()
    monkeypatch.setattr(
        "nemo_agent_optimization_plugin.jobs.optimize.discover_optimization_strategies",
        lambda: {"fake": strategy},
    )
    monkeypatch.setattr(
        "nemo_agent_optimization_plugin.jobs.optimize.resolve_agent_config",
        lambda agent, *, workspace, sdk: dict(AGENT_CONFIG),
    )


def test_run_dispatches_to_discovered_strategy(tmp_path: Path, strategy: _FakeStrategy) -> None:
    config_path = _write_config(tmp_path)

    job = OptimizeJob()
    ctx = _ctx(tmp_path)
    result = job.run(
        {
            "strategy": "fake",
            "optimize_config": str(config_path),
            "agent": "some-agent",
            "workspace": "default",
        },
        ctx=ctx,
        sdk=None,
    )

    assert result["status"] == "completed"
    assert result["strategy"] == "fake"
    assert strategy.validated == [({"any": "config"}, "some-agent")]
    assert strategy.run_kwargs == {
        "agent_config": AGENT_CONFIG,
        "source_agent_config": None,
        "config": {"any": "config"},
        "ctx": ctx,
        "workspace": "default",
        "sdk": None,
    }


def test_run_publishes_primary_artifact(tmp_path: Path, strategy: _FakeStrategy) -> None:
    artifact = tmp_path / "optimized.yaml"
    artifact.write_text(yaml.safe_dump({"schema_version": "fabric.agent/v1alpha1"}), encoding="utf-8")
    strategy.result = {"status": "completed", PRIMARY_ARTIFACT_KEY: str(artifact)}
    output = tmp_path / "published" / "agent.yaml"

    result = OptimizeJob().run(
        {
            "strategy": "fake",
            "optimize_config": str(_write_config(tmp_path)),
            "agent": "some-agent",
            "workspace": "default",
            "output": str(output),
        },
        ctx=_ctx(tmp_path),
        sdk=None,
    )

    assert PRIMARY_ARTIFACT_KEY not in result
    assert result["output"] == {"type": "local_file", "path": str(output.resolve())}
    assert output.read_text(encoding="utf-8") == artifact.read_text(encoding="utf-8")


def test_run_raises_for_unknown_strategy(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    job = OptimizeJob()
    ctx = _ctx(tmp_path)
    with pytest.raises(Exception, match="not installed"):
        job.run(
            {
                "strategy": "does-not-exist",
                "optimize_config": str(config_path),
                "agent": "some-agent",
                "workspace": "default",
            },
            ctx=ctx,
            sdk=None,
        )


def test_task_entrypoint_imports() -> None:
    from nemo_agent_optimization_plugin.tasks import optimize as task_module

    assert callable(task_module.main)
