# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from nemo_eval_author_plugin.jobs import first_eval as job_module
from nemo_helix_plugin.agents.client import AgentsClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults

SOURCE_AGENT: dict[str, Any] = {
    "config_format": "nemo-agents-spec-v1",
    "name": "calculator-agent",
    "default_harness": "deepagents",
    "harnesses": {"deepagents": {"kind": "deepagents"}},
    "models": {"default": {"provider": "nvidia", "model": "calculator-model"}},
    "instructions": {"system": {"content": "add numbers"}},
}


def not_found() -> NotFoundError:
    return NotFoundError(httpx.Response(404, json={"detail": "not found"}, request=httpx.Request("GET", "http://x")))


@pytest.fixture
def ctx(tmp_path: Path) -> JobContext:
    ephemeral = tmp_path / "ephemeral"
    ephemeral.mkdir()
    return JobContext(
        workspace="default",
        storage=StoragePaths(ephemeral=ephemeral),
        results=LocalJobResults(root=tmp_path / "job-results"),
    )


@pytest.fixture
def platform(monkeypatch: pytest.MonkeyPatch) -> Callable[..., SimpleNamespace]:
    """Patch the platform clients and fileset I/O; returns the recorded downloads and uploads."""

    def _install(agent: Any = SOURCE_AGENT, ethos: Any = None) -> SimpleNamespace:
        agents = MagicMock()
        if isinstance(agent, BaseException):
            agents.get_agent.side_effect = agent
        else:
            agents.get_agent.return_value.data.return_value = SimpleNamespace(config=agent)
        files = MagicMock()
        if ethos is None:
            files.get_fileset.side_effect = not_found()
        clients = {AgentsClient: agents, FilesClient: files}
        monkeypatch.setattr(job_module, "client_from_platform", lambda sdk, cls: clients[cls])
        seen = SimpleNamespace(config=agent, agents=agents, downloads=[], uploads=[])
        monkeypatch.setattr(
            job_module, "_download_fileset", lambda sdk, ws, name, *, into: seen.downloads.append((ws, name, into))
        )
        monkeypatch.setattr(
            job_module, "upload_to_fileset", lambda local_dir, **kw: seen.uploads.append((local_dir, kw))
        )
        return seen

    return _install


@pytest.fixture
def fake_author(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the Fabric run with one that writes a plan and an Ethos; returns the recorded calls."""
    calls: list[dict[str, Any]] = []

    def fake(author: Any, *, input: str, base_dir: Path) -> str:
        calls.append({"author": author, "input": input, "base_dir": base_dir})
        workspace = base_dir / "workspace"
        (workspace / ".eval-author" / "task-drafts").mkdir(parents=True)
        (workspace / ".eval-author" / "first-eval.md").write_text("# plan\n", encoding="utf-8")
        (workspace / ".eval-author" / "task-drafts" / "add.md").write_text("# case\n", encoding="utf-8")
        (workspace / "ETHOS.md").write_text("# ethos\n", encoding="utf-8")
        return "Authored one case."

    monkeypatch.setattr(job_module, "run_author", fake)
    return calls
