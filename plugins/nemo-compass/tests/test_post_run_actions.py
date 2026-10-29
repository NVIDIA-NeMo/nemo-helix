# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compass post-run actions for generic execute-agent jobs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nemo_compass_plugin.analysis.agent_config import build_compass_agent_config
from nemo_compass_plugin.analysis.backend import InsightNotFoundError
from nemo_compass_plugin.analysis.result import CompassResult
from nemo_compass_plugin.post_run_actions import (
    REPORT_RESULT_NAME,
    CompassPersistAction,
    CompassPreviewAction,
    ExistingInsight,
    render_preview,
    target_agent,
)
from nemo_helix_plugin.agents.post_run_actions import (
    FabricRuntimeResult,
    PostRunActionContext,
)
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults

CHANGE_SET: dict[str, Any] = {
    "summary": "Retrieval drops context near the token limit.",
    "new_insights": [
        {
            "title": "Retrieval truncates long documents",
            "description": "The retriever cuts documents at 4k tokens.\nAnswers then miss the conclusion.",
            "severity": "high",
            "confidence": "med",
            "evidence": [{"trace_id": "trace-1", "spans": [{"span_id": "span-1"}, {"span_id": "span-2"}]}],
        }
    ],
    "updated_insights": [
        {"id": "insight-known", "evidence": [{"trace_id": "trace-2"}]},
        {"id": "insight-gone", "evidence": [{"trace_id": "trace-3"}]},
        {"id": "insight-foreign", "evidence": [{"trace_id": "trace-4"}]},
    ],
    "notes": ["- below threshold: one flaky tool call"],
}


@pytest.fixture
def ctx(tmp_path: Path) -> JobContext:
    persistent = tmp_path / "persistent"
    ephemeral = tmp_path / "ephemeral"
    persistent.mkdir()
    ephemeral.mkdir()
    return JobContext(
        workspace="job-workspace",
        storage=StoragePaths(ephemeral=ephemeral, persistent=persistent),
        results=LocalJobResults(root=ephemeral / "results"),
    )


def _agent_config() -> dict[str, Any]:
    return build_compass_agent_config(
        agent="research-agent",
        default_model="team-a/default",
        fast_model="team-a/fast",
    )


def _context(ctx: JobContext, *, output: Any = None, config: dict[str, Any] | None = None) -> PostRunActionContext:
    return PostRunActionContext(
        ctx=ctx,
        config=config or {},
        agent_name="compass",
        agent_config=_agent_config(),
        fabric_result=FabricRuntimeResult(
            status="succeeded",
            output={"response": "summary", "compass_result": CHANGE_SET} if output is None else output,
        ),
    )


class FakeClient:
    def __init__(self) -> None:
        self.closed = False

    async def __aenter__(self) -> FakeClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self.closed = True


class FakeBackend:
    """Stands in for ``RemoteCompassBackend``, recording every call it gets."""

    instances: list[FakeBackend] = []

    def __init__(self, client: FakeClient) -> None:
        self.client = client
        self.persisted: list[dict[str, Any]] = []
        self.looked_up: list[tuple[str, str]] = []
        FakeBackend.instances.append(self)

    async def persist_result(self, *, workspace: str, agent: str, result: CompassResult) -> str:
        self.persisted.append({"workspace": workspace, "agent": agent, "result": result})
        return "Persisted analysis report"

    async def get_insight(self, *, workspace: str, insight_id: str) -> SimpleNamespace:
        self.looked_up.append((workspace, insight_id))
        if insight_id == "insight-gone":
            raise InsightNotFoundError(insight_id)
        if insight_id == "insight-foreign":
            return SimpleNamespace(title="Another agent's insight", workspace=workspace, agent="other-agent")
        return SimpleNamespace(title="Known insight title", workspace=workspace, agent="research-agent")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    client = FakeClient()
    FakeBackend.instances = []
    monkeypatch.setattr("nemo_compass_plugin.post_run_actions.get_async_task_nemo_client", lambda service: client)
    monkeypatch.setattr("nemo_compass_plugin.post_run_actions.RemoteCompassBackend", FakeBackend)
    return client


def _report(ctx: JobContext) -> str:
    return Path(ctx.storage.ephemeral / "results" / REPORT_RESULT_NAME).read_text()


def test_persist_writes_the_change_set_for_the_agent_in_the_job_workspace(ctx: JobContext, client: FakeClient) -> None:
    CompassPersistAction().after_invoke(_context(ctx))

    [backend] = FakeBackend.instances
    [call] = backend.persisted
    assert (call["agent"], call["workspace"]) == ("research-agent", "job-workspace")
    assert call["result"] == CompassResult.model_validate(CHANGE_SET)
    assert client.closed
    assert _report(ctx).endswith("\n\nPersisted analysis report")


def test_preview_writes_nothing_and_reports_what_persisting_would_do(ctx: JobContext, client: FakeClient) -> None:
    CompassPreviewAction().after_invoke(_context(ctx))

    [backend] = FakeBackend.instances
    assert backend.persisted == []
    assert backend.looked_up == [
        ("job-workspace", "insight-known"),
        ("job-workspace", "insight-gone"),
        ("job-workspace", "insight-foreign"),
    ]
    assert client.closed
    report = _report(ctx)
    assert "Preview: no insights were saved." in report
    assert "Retrieval truncates long documents" in report
    assert "- Known insight title [insight-known]: 1 trace, 0 spans" in report
    assert "- would skip (insight not found): insight-gone" in report
    assert "- would fail (insight belongs to another agent): Another agent's insight [insight-foreign]" in report


@pytest.mark.parametrize("action", [CompassPersistAction, CompassPreviewAction])
def test_the_report_names_the_analyst_implementation(
    action: type[CompassPersistAction | CompassPreviewAction], ctx: JobContext, client: FakeClient
) -> None:
    action().after_invoke(_context(ctx))

    first_line = _report(ctx).splitlines()[0]
    assert first_line.startswith("Produced by nemo-compass-plugin ")
    assert "nemo-compass " in first_line


@pytest.mark.parametrize("action", [CompassPersistAction, CompassPreviewAction])
def test_the_actions_take_no_config(action: type[CompassPersistAction | CompassPreviewAction]) -> None:
    """``agent`` moved to the agent config and ``workspace`` is the job's; a stale request fails at create time."""
    with pytest.raises(ValueError):
        action.config_model.model_validate({"agent": "research-agent", "workspace": "team-a"})
    action.config_model.model_validate({})


@pytest.mark.parametrize("action", [CompassPersistAction, CompassPreviewAction])
def test_the_actions_name_the_missing_compass_result_key(
    action: type[CompassPersistAction | CompassPreviewAction], ctx: JobContext
) -> None:
    """An output without the key is a wiring problem, not a malformed CompassResult."""
    with pytest.raises(ValueError) as excinfo:
        action().after_invoke(_context(ctx, output={"response": "summary"}))

    message = str(excinfo.value)
    assert "compass_result" in message
    assert "'response'" in message, "the message should show what the output actually carried"


def test_an_agent_that_is_not_compass_is_rejected() -> None:
    config = {"default_harness": "main", "harnesses": {"main": {"kind": "deepagents", "settings": {"agent": "x"}}}}

    with pytest.raises(ValueError, match="Compass adapter"):
        target_agent(config)


def test_a_compass_harness_without_an_agent_is_rejected() -> None:
    config = _agent_config()
    del config["harnesses"]["compass"]["settings"]["agent"]

    with pytest.raises(ValueError, match=r"settings\.agent"):
        target_agent(config)


def test_an_empty_preview_says_so() -> None:
    report = render_preview(CompassResult(summary="Nothing new."), existing={})

    assert report == "\n".join(
        [
            "Preview: no insights were saved.",
            "",
            "Nothing new.",
            "",
            "Proposed new insights (0)",
            "- none",
            "",
            "Existing insights that would gain evidence (0)",
            "- none",
        ]
    )


def test_a_preview_lays_out_each_proposed_insight() -> None:
    result = CompassResult.model_validate(CHANGE_SET)

    report = render_preview(
        result,
        existing={
            "insight-known": ExistingInsight(title="Known insight title", owned=True),
            "insight-gone": None,
            "insight-foreign": ExistingInsight(title="Another agent's insight", owned=True),
        },
    )

    assert report.splitlines()[4:13] == [
        "Proposed new insights (1)",
        "",
        "1. Retrieval truncates long documents",
        "   status open, severity high, confidence med",
        "   The retriever cuts documents at 4k tokens.",
        "   Answers then miss the conclusion.",
        "   Evidence: 1 trace, 2 spans",
        "     - trace trace-1",
        "       - span span-1",
    ]
    assert report.splitlines()[-2:] == ["Notes", "- below threshold: one flaky tool call"]


def test_a_preview_warns_when_persisting_would_fail() -> None:
    """Persisting raises on evidence for another agent's insight; the preview says so up front."""
    result = CompassResult.model_validate(CHANGE_SET)

    report = render_preview(
        result,
        existing={
            "insight-known": ExistingInsight(title="Known insight title", owned=True),
            "insight-gone": None,
            "insight-foreign": ExistingInsight(title="Another agent's insight", owned=False),
        },
    )

    assert report.splitlines()[:2] == [
        "Preview: no insights were saved.",
        "Persisting this change-set would fail: it adds evidence to another agent's insights.",
    ]
    assert "Existing insights that would gain evidence (1)" in report
    assert "- would fail (insight belongs to another agent): Another agent's insight [insight-foreign]" in report


def test_an_unowned_insight_without_new_evidence_does_not_fail() -> None:
    """Persisting checks ownership only when it adds evidence."""
    result = CompassResult.model_validate({"summary": "s", "updated_insights": [{"id": "insight-foreign"}]})

    report = render_preview(result, existing={"insight-foreign": ExistingInsight(title="Theirs", owned=False)})

    assert "would fail" not in report
    assert "- Theirs [insight-foreign]: 0 traces, 0 spans" in report
