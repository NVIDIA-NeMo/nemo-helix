# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from nemo_agents_plugin.jobs.execute import FABRIC_RUN_RESULT_NAME
from nemo_agents_plugin.jobs.pipeline import (
    PIPELINE_RESULT_NAME,
    PIPELINE_STEPS,
    PipelineJob,
    PipelineJobConfig,
    PipelineStepFailedError,
    build_step_input,
)
from nemo_helix_plugin.job_context import JobContext
from pydantic import ValidationError

TASK = "Add retry logic to the uploader"


class _FakeExecuteJobs:
    """Stands in for ``_ExecuteJobsResource``: records submissions and serves canned results."""

    def __init__(self, *, statuses: dict[str, str] | None = None, responses: dict[str, Any] | None = None) -> None:
        self.created: list[dict[str, Any]] = []
        self._statuses = statuses or {}
        self._responses = responses or {}

    def create(self, *, spec: dict[str, Any], workspace: str | None = None, **_: Any) -> dict[str, Any]:
        self.created.append(spec)
        return {"name": f"job-{len(self.created)}"}

    def get(self, name: str, workspace: str | None = None) -> dict[str, Any]:
        return {"name": name, "status": self._statuses.get(name, "completed")}

    def download_result(self, name: str, *, job: str, workspace: str | None = None) -> bytes:
        assert name == FABRIC_RUN_RESULT_NAME
        return json.dumps(self._responses.get(job, {"response": f"output of {job}"})).encode()


def _ctx(tmp_path: Path) -> JobContext:
    return cast(
        JobContext,
        SimpleNamespace(workspace="default", storage=SimpleNamespace(ephemeral=tmp_path), results=MagicMock()),
    )


def _run(fake: _FakeExecuteJobs, tmp_path: Path, config: dict[str, Any] | None = None) -> tuple[dict, JobContext]:
    ctx = _ctx(tmp_path)
    with (
        patch("nemo_agents_plugin.jobs.pipeline._ExecuteJobsResource", return_value=fake),
        patch("nemo_agents_plugin.jobs.pipeline.AgentsClient"),
    ):
        result = PipelineJob().run(config or {"task": TASK}, ctx=ctx, sdk=MagicMock(), sleep=lambda _s: None)
    return result, ctx


def test_config_requires_task() -> None:
    with pytest.raises(ValidationError):
        PipelineJobConfig.model_validate({})
    with pytest.raises(ValidationError):
        PipelineJobConfig(task="   ")


def test_pipeline_has_three_steps_in_order() -> None:
    assert [(step.role, step.agent) for step in PIPELINE_STEPS] == [
        ("plan", "pr-planner"),
        ("write", "pr-writer"),
        ("review", "pr-reviewer"),
    ]


def test_step_inputs_carry_the_previous_output() -> None:
    assert build_step_input("plan", task=TASK, previous="") == TASK
    write = build_step_input("write", task=TASK, previous="THE PLAN")
    assert TASK in write
    assert "THE PLAN" in write
    assert "THE CHANGE" in build_step_input("review", task=TASK, previous="THE CHANGE")


@pytest.mark.asyncio
async def test_compile_returns_one_pipeline_step() -> None:
    spec = await PipelineJob.compile(
        workspace="default",
        spec=PipelineJobConfig(task=TASK),
        entity_client=None,
        job_name=None,
        async_sdk=None,
    )

    assert [step.name for step in spec.steps] == ["pipeline"]
    assert spec.steps[0].config["task"] == TASK


def test_run_executes_agents_in_order_and_chains_responses(tmp_path: Path) -> None:
    fake = _FakeExecuteJobs(
        responses={
            "job-1": {"response": "THE PLAN"},
            "job-2": {"response": "THE CHANGE"},
            "job-3": {"response": "LGTM"},
        }
    )

    result, ctx = _run(fake, tmp_path, {"task": TASK, "environment": "default/env"})

    assert [spec["agent"] for spec in fake.created] == ["pr-planner", "pr-writer", "pr-reviewer"]
    assert fake.created[0]["input"] == TASK
    assert "THE PLAN" in fake.created[1]["input"]
    assert "THE CHANGE" in fake.created[2]["input"]
    assert all(spec["environment"] == "default/env" for spec in fake.created)
    assert result["status"] == "completed"
    assert [(step["role"], step["job"]) for step in result["steps"]] == [
        ("plan", "job-1"),
        ("write", "job-2"),
        ("review", "job-3"),
    ]
    results = cast(MagicMock, ctx.results)
    results.save.assert_called_once()
    assert results.save.call_args.args[0] == PIPELINE_RESULT_NAME


def test_run_stops_after_a_failed_step(tmp_path: Path) -> None:
    fake = _FakeExecuteJobs(statuses={"job-2": "error"})
    ctx = _ctx(tmp_path)

    with (
        patch("nemo_agents_plugin.jobs.pipeline._ExecuteJobsResource", return_value=fake),
        patch("nemo_agents_plugin.jobs.pipeline.AgentsClient"),
        pytest.raises(PipelineStepFailedError, match="Step 'write'"),
    ):
        PipelineJob().run({"task": TASK}, ctx=ctx, sdk=MagicMock(), sleep=lambda _s: None)

    assert len(fake.created) == 2  # the reviewer was never submitted
    saved = json.loads((tmp_path / "pipeline_result.json").read_text())
    assert saved["status"] == "failed"
    assert [step["status"] for step in saved["steps"]] == ["completed", "error"]


def test_run_fails_when_a_step_returns_no_text(tmp_path: Path) -> None:
    fake = _FakeExecuteJobs(responses={"job-1": {"output": {}}})
    ctx = _ctx(tmp_path)

    with (
        patch("nemo_agents_plugin.jobs.pipeline._ExecuteJobsResource", return_value=fake),
        patch("nemo_agents_plugin.jobs.pipeline.AgentsClient"),
        pytest.raises(PipelineStepFailedError, match="without a text response"),
    ):
        PipelineJob().run({"task": TASK}, ctx=ctx, sdk=MagicMock(), sleep=lambda _s: None)

    assert len(fake.created) == 1


def test_run_requires_a_platform_client(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="platform client"):
        PipelineJob().run({"task": TASK}, ctx=_ctx(tmp_path), sdk=None)
