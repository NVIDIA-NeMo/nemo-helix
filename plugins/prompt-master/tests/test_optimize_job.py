# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import pytest
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import LocalJobResults
from prompt_master_plugin.jobs.optimize import PromptMasterOptimizeJob


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


def _source() -> dict[str, Any]:
    return {
        "config_format": "nemo-agents-spec-v1",
        "name": "my-agent",
        "default_harness": "hermes",
        "harnesses": {"hermes": {"kind": "hermes", "model": {"provider": "openai", "model": "m"}}},
        "instructions": {"system": {"content": "old prompt"}},
    }


def test_optimize_replaces_the_system_prompt(ctx: JobContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "prompt_master_plugin.jobs.optimize.run_prompt_master",
        lambda parsed, agent_config, runtime_dir: "new prompt",
    )
    source = _source()
    optimized = PromptMasterOptimizeJob().optimize(
        source_agent_config=source,
        config={"model": {"provider": "openai", "model": "optimizer-model"}},
        ctx=ctx,
        workspace="my-ws",
        sdk=object(),
    )

    assert optimized["instructions"]["system"]["content"] == "new prompt"
    # The caller's mapping is an input, not scratch space.
    assert source["instructions"]["system"]["content"] == "old prompt"


def test_the_job_declares_its_strategy() -> None:
    assert PromptMasterOptimizeJob.strategy == "prompt-master"


def test_the_job_declares_its_task_module() -> None:
    assert PromptMasterOptimizeJob.task_module == "prompt_master_plugin.tasks.agent_optimize"
