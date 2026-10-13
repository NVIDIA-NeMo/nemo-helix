# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures, and the import path for an uninstalled checkout.

``plugins/prompt-master`` is deliberately not a root uv-workspace member (see README,
"Install"), so a plain ``make test-unit`` from the repo root -- which collects
``plugins/*/tests`` -- would otherwise fail at import.  An installed copy wins: the path is
only added when the package cannot be found.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

if importlib.util.find_spec("prompt_master_plugin") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402
from nemo_helix_plugin.job_context import JobContext, StoragePaths  # noqa: E402
from nemo_helix_plugin.job_results import LocalJobResults  # noqa: E402
from prompt_master_plugin.jobs import optimize as optimize_module  # noqa: E402
from prompt_master_plugin.runner import PromptMasterOutcome  # noqa: E402


@pytest.fixture
def source_agent() -> dict[str, Any]:
    """A stored ``nemo-agents-spec-v1`` agent whose prompt the strategy rewrites."""
    return {
        "config_format": "nemo-agents-spec-v1",
        "name": "calculator-agent",
        "default_harness": "deepagents",
        "harnesses": {"deepagents": {"kind": "deepagents"}},
        "models": {
            "default": {
                "provider": "nvidia",
                "model": "calculator-model",
                "base_url": "https://integrate.api.nvidia.com/v1",
                "api_key_env": "NVIDIA_API_KEY",
            }
        },
        "instructions": {"system": {"content": "old prompt"}},
    }


@pytest.fixture
def ctx(tmp_path: Path) -> JobContext:
    """A local job context: ephemeral scratch plus a results sink under ``tmp_path/job-results``."""
    ephemeral = tmp_path / "ephemeral"
    ephemeral.mkdir()
    return JobContext(
        workspace="default",
        storage=StoragePaths(ephemeral=ephemeral),
        results=LocalJobResults(root=tmp_path / "job-results"),
    )


@pytest.fixture
def fake_prompt_master(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, Any, Path]]:
    """Replace the Fabric run with a canned outcome; returns the recorded ``(optimizer, agent_config, base_dir)`` calls."""
    calls: list[tuple[Any, Any, Path]] = []

    def fake(optimizer: Any, agent_config: Any, base_dir: Path) -> PromptMasterOutcome:
        calls.append((optimizer, agent_config, base_dir))
        return PromptMasterOutcome(
            optimized_prompt="new prompt",
            response="```\nnew prompt\n```\n🎯 Target: Fabric agent, 💡 Tightened scope.",
        )

    monkeypatch.setattr(optimize_module, "run_prompt_master", fake)
    return calls


@pytest.fixture
def stored_agent() -> Callable[[dict[str, Any] | BaseException], contextlib.AbstractContextManager[MagicMock]]:
    """Patch the Agents client so ``get_agent`` returns *config* (or raises it when given an exception)."""

    @contextlib.contextmanager
    def _stored(config: dict[str, Any] | BaseException) -> Iterator[MagicMock]:
        agents = MagicMock()
        if isinstance(config, BaseException):
            agents.get_agent.side_effect = config
        else:
            agents.get_agent.return_value.data.return_value = SimpleNamespace(config=config)
        with patch.object(optimize_module, "client_from_platform", return_value=agents):
            yield agents

    return _stored
