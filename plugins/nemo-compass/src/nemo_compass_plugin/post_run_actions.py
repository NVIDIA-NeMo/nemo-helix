# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compass post-run actions for generic ``agents.execute`` jobs."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from nemo_compass_plugin.analysis.backend import make_backend
from nemo_compass_plugin.analysis.result import CompassResult
from nemo_compass_plugin.analyst import SAVE_INSIGHTS_ACTION_KIND
from nemo_helix_plugin.agents.post_run_actions import PostRunActionContext
from nemo_helix_plugin.client_provider import get_async_task_nemo_client
from pydantic import BaseModel, ConfigDict

REPORT_RESULT_NAME = "analysis-report"
REPORT_FILE_NAME = "analysis-report.txt"


class SaveInsightsActionConfig(BaseModel):
    """The action takes no configuration: the run's context names the analyzed agent."""

    model_config = ConfigDict(extra="forbid")


class SaveInsightsAction:
    """Persist the storage-agnostic Compass change-set returned by Fabric."""

    kind: ClassVar[str] = SAVE_INSIGHTS_ACTION_KIND
    title: ClassVar[str] = "Save Compass insights"
    description: ClassVar[str] = "Apply the insight changes a Compass run found and save its analysis report."

    # ``after_invoke`` parses through this same attribute, so a config the
    # create request accepted is one this action can actually run. Annotated
    # narrowly rather than as ``type[BaseModel]`` so the parsed config keeps its
    # concrete type at the use sites below.
    config_model: ClassVar[type[SaveInsightsActionConfig]] = SaveInsightsActionConfig

    def after_invoke(self, context: PostRunActionContext) -> None:
        self.config_model.model_validate(context.config)
        agent = _analyzed_agent(context.run_context)
        result = _compass_result_from_fabric_output(context.fabric_result.output)
        report = asyncio.run(_persist_result(workspace=context.ctx.workspace, agent=agent, result=result))

        report_path = context.ctx.storage.ephemeral / REPORT_FILE_NAME
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        context.ctx.results.save(REPORT_RESULT_NAME, report_path)


def _analyzed_agent(run_context: dict[str, Any]) -> str:
    """Return the agent a Compass run analyzed, from its prepared context."""
    agent = run_context.get("agent")
    if not isinstance(agent, str) or not agent.strip():
        raise ValueError(
            "Compass save-insights action expected the run context to name the analyzed agent; "
            f"got keys: {list(run_context)}."
        )
    return agent


def _compass_result_from_fabric_output(output: Any) -> CompassResult:
    if not isinstance(output, dict):
        raise ValueError("Compass save-insights action expected Fabric output to be a mapping.")
    if "compass_result" not in output:
        # Validating the whole output instead would report missing CompassResult
        # fields, describing the shape we wanted rather than the key we lacked.
        raise ValueError(
            f"Compass save-insights action expected Fabric output to contain 'compass_result'; got keys: {list(output)}."
        )
    return CompassResult.model_validate(output["compass_result"])


async def _persist_result(*, workspace: str, agent: str, result: CompassResult) -> str:
    client = get_async_task_nemo_client("compass")
    try:
        backend = make_backend(client=client, insights_output=None)
        return await backend.persist_result(workspace=workspace, agent=agent, result=result)
    finally:
        await client.close()
