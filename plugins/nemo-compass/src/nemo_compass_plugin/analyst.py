# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``compass.analyst`` plugin-managed agent type and its associated models."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, ClassVar

from nemo_compass_plugin.analysis.models import resolve_model_ref
from nemo_compass_plugin.config import CompassConfig
from nemo_helix_plugin.agents.ethos import ETHOS_FILENAME, ethos_fileset_name
from nemo_helix_plugin.agents.plugin_managed_agents import (
    CONFIGURED_DEFAULT_MODEL,
    CONFIGURED_FAST_MODEL,
    PluginManagedAgent,
    PreparedRun,
    helix_default,
)
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.config import get_nemo_config
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.models.client import AsyncModelsClient
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

COMPASS_ANALYST_TYPE_ID = "compass.analyst"
COMPASS_ADAPTER_ID = "nvidia.helix.compass.analyst"
SAVE_INSIGHTS_ACTION_KIND = "compass.save-insights"

_NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CompassContext(BaseModel):
    """What a Compass run is asked to do: an execute job's context."""

    model_config = ConfigDict(extra="forbid")

    agent: _NonBlank = Field(description=("Target agent whose telemetry is being analyzed."))
    default_model: _NonBlank = Field(
        description=("Model Entity Compass uses for analysis."),
        json_schema_extra=helix_default(CONFIGURED_DEFAULT_MODEL),
    )
    fast_model: _NonBlank | None = Field(
        default=None,
        description="A second, typically faster Model Entity Compass uses for part of its analysis. Defaults to default_model.",
        json_schema_extra=helix_default(CONFIGURED_FAST_MODEL),
    )
    since: datetime | None = Field(
        default=None,
        description="Analyze only telemetry from this time on. When unset, all traces from the target agent are in scope.",
    )
    evaluation_id: _NonBlank | None = Field(default=None, description="Analyze only the spans of this evaluation run.")


class CompassPreparedContext(BaseModel):
    """What the Compass adapter receives: the run's context, resolved by CompassAnalyst.prepare."""

    model_config = ConfigDict(extra="forbid")

    agent: str
    default_model: str = Field(description="Qualified Model Entity ref.")
    fast_model: str = Field(description="Qualified Model Entity ref.")
    # The ethos content is inlined here because we can't reach out to the Files service from the Fabric adapter
    ethos: str | None = Field(
        default=None,
        description=("The analyzed agent's Ethos Markdown, or None when it has none."),
    )
    since: datetime | None = None
    evaluation_id: str | None = None


class CompassAnalyst(PluginManagedAgent[CompassContext, CompassPreparedContext]):
    """Compass: analyzes an agent's telemetry and maintains durable insights about it."""

    type_id: ClassVar[str] = COMPASS_ANALYST_TYPE_ID
    title: ClassVar[str] = "Compass analyst"
    description: ClassVar[str] = "Analyzes an agent's telemetry and maintains durable insights about it."
    adapter_id: ClassVar[str] = COMPASS_ADAPTER_ID
    post_run_actions: ClassVar[frozenset[str]] = frozenset({SAVE_INSIGHTS_ACTION_KIND})
    default_name: ClassVar[str] = "compass-analyst"
    subject_label: ClassVar[str | None] = "Analyzed agent"
    # The adapter builds Compass's prompts itself, from the context; this is set only to satisfy the job contract.
    default_input: ClassVar[str | None] = "Analyze the agent's recent telemetry and maintain durable insights."

    @classmethod
    async def prepare(
        cls, context: CompassContext, *, workspace: str, client: AsyncNemoClient
    ) -> PreparedRun[CompassPreparedContext]:
        """Qualify the model refs and read the analyzed agent's Ethos, as the caller.

        Every run uses the configured ``analysis.job_profile`` unless its job request names a profile,
        whether the scheduler, Studio or a CLI user started it.
        """
        models_client = AsyncModelsClient.from_client(client)
        default_model = await resolve_model_ref(
            models_client, context.default_model, field="default_model", workspace=workspace
        )
        fast_model = (
            default_model
            if context.fast_model is None
            else await resolve_model_ref(models_client, context.fast_model, field="fast_model", workspace=workspace)
        )
        ethos = await read_ethos(AsyncFilesClient.from_client(client), workspace=workspace, agent=context.agent)
        return PreparedRun(
            context=CompassPreparedContext(
                agent=context.agent,
                default_model=default_model,
                fast_model=fast_model,
                ethos=ethos,
                since=context.since,
                evaluation_id=context.evaluation_id,
            ),
            subject=context.agent,
            job_profile=get_nemo_config(CompassConfig).analysis.job_profile,
        )


async def read_ethos(files_client: AsyncFilesClient, *, workspace: str, agent: str) -> str | None:
    """Return *agent*'s Ethos Markdown from its conventional fileset, or None when it has none.

    A denied read is not swallowed: it fails the run's create request, rather
    than silently analyzing without an Ethos the caller expected to apply.
    """
    response = await files_client.download_file(
        workspace=workspace, name=ethos_fileset_name(agent), path=ETHOS_FILENAME
    )
    try:
        # The download is lazy: a missing file raises here, when the body is read.
        content = await response.read()
    except NotFoundError:
        return None
    text = content.decode("utf-8", errors="replace")
    return text if text.strip() else None
