# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compass post-run actions for generic execute-agent jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from nemo_compass_plugin.post_run_actions import REPORT_RESULT_NAME, CompassAnalysisAction
from nemo_helix_plugin.agents.post_run_actions import (
    FabricRuntimeResult,
    PostRunActionContext,
)
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults


@pytest.fixture
def ctx(tmp_path: Path) -> JobContext:
    persistent = tmp_path / "persistent"
    ephemeral = tmp_path / "ephemeral"
    persistent.mkdir()
    ephemeral.mkdir()
    return JobContext(
        workspace="default",
        storage=StoragePaths(ephemeral=ephemeral, persistent=persistent),
        results=LocalJobResults(root=ephemeral / "results"),
    )


class FakeClient:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def persist_result(self, *, workspace: str, agent: str, result: object) -> str:
        self.calls.append({"workspace": workspace, "agent": agent, "result": result})
        return "Persisted analysis report"


def test_compass_action_persists_compass_result_and_saves_report(
    ctx: JobContext,
    monkeypatch,
) -> None:
    client = FakeClient()
    backend = FakeBackend()

    monkeypatch.setattr("nemo_compass_plugin.post_run_actions.get_async_task_nemo_client", lambda service: client)
    monkeypatch.setattr("nemo_compass_plugin.post_run_actions.make_backend", lambda **_kwargs: backend)

    action = CompassAnalysisAction()
    action.after_invoke(
        PostRunActionContext(
            ctx=ctx,
            config={"agent": "research-agent"},
            agent_name="compass",
            agent_config={},
            fabric_result=FabricRuntimeResult(
                status="succeeded",
                output={
                    "response": "summary",
                    "compass_result": {
                        "summary": "summary",
                        "new_insights": [],
                        "updated_insights": [],
                    },
                },
            ),
        )
    )

    assert backend.calls[0]["agent"] == "research-agent"
    assert backend.calls[0]["result"].summary == "summary"
    assert client.closed
    report_path = Path(ctx.storage.ephemeral / "results" / REPORT_RESULT_NAME)
    assert report_path.read_text() == "Persisted analysis report"


def test_compass_action_names_the_missing_compass_result_key(ctx: JobContext) -> None:
    """An output without the key is a wiring problem, not a malformed CompassResult."""
    action = CompassAnalysisAction()

    with pytest.raises(ValueError) as excinfo:
        action.after_invoke(
            PostRunActionContext(
                ctx=ctx,
                config={"agent": "research-agent"},
                agent_name="compass",
                agent_config={},
                fabric_result=FabricRuntimeResult(status="succeeded", output={"response": "summary"}),
            )
        )

    message = str(excinfo.value)
    assert "compass_result" in message
    assert "'response'" in message, "the message should show what the output actually carried"
