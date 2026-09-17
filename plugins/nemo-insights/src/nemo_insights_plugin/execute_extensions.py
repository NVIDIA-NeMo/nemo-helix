# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Insights extensions for generic ``agents.execute`` jobs."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from nemo_insights_plugin.analyst.analyst_backend import make_analyst_backend
from nemo_insights_plugin.analyst.result import AnalystResult
from nemo_insights_plugin.jobs.analyze import REPORT_FILE_NAME, REPORT_RESULT_NAME
from nemo_platform_plugin.agents.execute_extensions import ExecuteAgentAfterInvokeContext
from nemo_platform_plugin.sdk_provider import get_async_task_sdk
from pydantic import BaseModel, ConfigDict, Field


class InsightsAnalysisExtensionConfig(BaseModel):
    """Configuration for persisting an ``AnalystResult`` emitted by Fabric."""

    model_config = ConfigDict(extra="forbid")

    agent: str | None = Field(default=None, description="Agent under test. Defaults to the invoked Agent entity name.")
    workspace: str | None = Field(
        default=None, description="Workspace to persist Insights in. Defaults to the job workspace."
    )
    insights_output: str | None = Field(default=None, description="Optional local YAML mirror path.")
    local_only: bool = Field(default=False, description="Persist only to the local YAML mirror.")


class InsightsAnalysisExtension:
    """Persist the storage-agnostic Analyst change-set returned by Fabric."""

    # ``after_invoke`` parses through this same attribute, so a config the
    # create request accepted is one this extension can actually run. Annotated
    # narrowly rather than as ``type[BaseModel]`` so the parsed config keeps its
    # concrete type at the use sites below.
    config_model: ClassVar[type[InsightsAnalysisExtensionConfig]] = InsightsAnalysisExtensionConfig

    def after_invoke(self, context: ExecuteAgentAfterInvokeContext) -> None:
        config = self.config_model.model_validate(context.config)
        result = _analyst_result_from_fabric_output(context.fabric_result.output)
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


def _analyst_result_from_fabric_output(output: Any) -> AnalystResult:
    if not isinstance(output, dict):
        raise ValueError("Insights analysis extension expected Fabric output to be a mapping.")
    if "analyst_result" not in output:
        # Validating the whole output instead would report missing AnalystResult
        # fields, describing the shape we wanted rather than the key we lacked.
        raise ValueError(
            f"Insights analysis extension expected Fabric output to contain 'analyst_result'; got keys: {list(output)}."
        )
    return AnalystResult.model_validate(output["analyst_result"])


async def _persist_result(
    *,
    workspace: str,
    agent: str,
    result: AnalystResult,
    insights_output: str | None,
    local_only: bool,
) -> str:
    client = get_async_task_sdk("insights")
    try:
        backend = make_analyst_backend(client=client, insights_output=insights_output, local_only=local_only)
        return await backend.persist_result(workspace=workspace, agent=agent, result=result)
    finally:
        await client.close()
