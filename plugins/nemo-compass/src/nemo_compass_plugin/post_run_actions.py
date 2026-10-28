# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compass post-run actions for generic ``agents.execute`` jobs."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from nemo_compass_plugin.analysis.backend import make_backend
from nemo_compass_plugin.analysis.result import CompassResult
from nemo_helix_plugin.agents.post_run_actions import PostRunActionContext
from nemo_helix_plugin.client_provider import get_async_task_nemo_client
from pydantic import BaseModel, ConfigDict, Field

REPORT_RESULT_NAME = "analysis-report"
REPORT_FILE_NAME = "analysis-report.txt"


class CompassAnalysisActionConfig(BaseModel):
    """Configuration for persisting a ``CompassResult`` emitted by Fabric."""

    model_config = ConfigDict(extra="forbid")

    agent: str | None = Field(default=None, description="Agent under test. Defaults to the invoked Agent entity name.")
    workspace: str | None = Field(
        default=None, description="Workspace to persist Insights in. Defaults to the job workspace."
    )
    insights_output: str | None = Field(default=None, description="Optional local YAML mirror path.")
    local_only: bool = Field(default=False, description="Persist only to the local YAML mirror.")


class CompassAnalysisAction:
    """Persist the storage-agnostic Compass change-set returned by Fabric."""

    title: ClassVar[str] = "Compass analysis"
    description: ClassVar[str] = "Persist the insights a Compass analysis run found and save an analysis report."

    # ``after_invoke`` parses through this same attribute, so a config the
    # create request accepted is one this action can actually run. Annotated
    # narrowly rather than as ``type[BaseModel]`` so the parsed config keeps its
    # concrete type at the use sites below.
    config_model: ClassVar[type[CompassAnalysisActionConfig]] = CompassAnalysisActionConfig

    def after_invoke(self, context: PostRunActionContext) -> None:
        config = self.config_model.model_validate(context.config)
        result = _compass_result_from_fabric_output(context.fabric_result.output)
        agent = config.agent or context.agent_name
        workspace = config.workspace or context.ctx.workspace
        report = asyncio.run(
            _persist_result(
                workspace=workspace,
                agent=agent,
                result=result,
                insights_output=config.insights_output,
                local_only=config.local_only,
            )
        )

        report_path = context.ctx.storage.ephemeral / REPORT_FILE_NAME
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        context.ctx.results.save(REPORT_RESULT_NAME, report_path)


def _compass_result_from_fabric_output(output: Any) -> CompassResult:
    if not isinstance(output, dict):
        raise ValueError("Compass analysis action expected Fabric output to be a mapping.")
    if "compass_result" not in output:
        # Validating the whole output instead would report missing CompassResult
        # fields, describing the shape we wanted rather than the key we lacked.
        raise ValueError(
            f"Compass analysis action expected Fabric output to contain 'compass_result'; got keys: {list(output)}."
        )
    return CompassResult.model_validate(output["compass_result"])


async def _persist_result(
    *,
    workspace: str,
    agent: str,
    result: CompassResult,
    insights_output: str | None,
    local_only: bool,
) -> str:
    client = get_async_task_nemo_client("compass")
    try:
        backend = make_backend(client=client, insights_output=insights_output, local_only=local_only)
        return await backend.persist_result(workspace=workspace, agent=agent, result=result)
    finally:
        await client.close()
