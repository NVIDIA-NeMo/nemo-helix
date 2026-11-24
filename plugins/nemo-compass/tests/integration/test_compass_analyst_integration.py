# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``compass.analyst`` through the real Compass, Agents, Jobs, Models and Files services.

Unit tests cover ``prepare`` against a mock transport; these check that the
installed type is what the platform loads, that saving an analysis config
creates the workspace's agent, and that a run's create request resolves
models and the Ethos as the caller.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from nemo_agents_plugin.sdk import AgentsResource
from nemo_agents_plugin.service import AgentsService
from nemo_compass_plugin.sdk import CompassPluginResource
from nemo_compass_plugin.service import CompassService
from nemo_helix_plugin.agents.client import AgentsClient
from nemo_helix_plugin.client.errors import NemoHTTPError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.inference_middleware import BackendFormat
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import CreateModelEntityRequest
from nhx.core.entities.service import EntitiesService
from nhx.core.files.service import FilesService
from nhx.core.jobs.service import JobsService
from nhx.core.models.service import ModelsService
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter
from nhx.testing import ClientContext, create_test_client, subprocess_job_executor_patch

pytestmark = pytest.mark.integration

WORKSPACE = "default"
ETHOS = "# Ethos\n\nAnswer only from retrieved context."


class _TestCompassService(NemoServiceAdapter):
    def __init__(self) -> None:
        super().__init__(CompassService())


class _TestAgentsService(NemoServiceAdapter):
    def __init__(self) -> None:
        super().__init__(AgentsService())


@pytest.fixture
def ctx() -> Iterator[ClientContext]:
    # Pin the CPU job profile, which Jobs otherwise derives from whether Docker is running.
    with (
        subprocess_job_executor_patch(),
        create_test_client(
            _TestCompassService,
            _TestAgentsService,
            EntitiesService,
            JobsService,
            FilesService,
            ModelsService,
            client_type=ClientContext,
            workspace=WORKSPACE,
        ) as ctx,
    ):
        models = ModelsClient.from_client(ctx.client)
        for name in ("compass-default", "compass-fast"):
            models.create_model(
                workspace=WORKSPACE, body=CreateModelEntityRequest(name=name, backend_format=BackendFormat.OPENAI_CHAT)
            )
        yield ctx


def _enable_analysis(ctx: ClientContext, agent: str = "support-bot") -> None:
    CompassPluginResource(ctx.client).analysis_configs.enable(
        workspace=WORKSPACE, agent=agent, default_model="compass-default", fast_model="compass-fast"
    )


def _compass_agents(ctx: ClientContext) -> list[Any]:
    page = AgentsClient.from_client(ctx.client).list_agents(
        workspace=WORKSPACE, query_params={"filter": json.dumps({"plugin_managed_agent.type": "compass.analyst"})}
    )
    return page.page().items


def _run(ctx: ClientContext, name: str, context: dict[str, Any]) -> dict[str, Any]:
    return AgentsResource(ctx.client).jobs.execute.create(
        name=name, spec={"agent": "compass-analyst", "context": context}
    )


def test_enabling_analysis_creates_the_workspaces_compass_agent_once(ctx: ClientContext) -> None:
    _enable_analysis(ctx, agent="support-bot")
    _enable_analysis(ctx, agent="billing-bot")

    [agent] = _compass_agents(ctx)
    assert agent.name == "compass-analyst"
    assert agent.config["harnesses"]["default"]["kind"] == "nvidia.helix.compass.analyst"
    assert agent.post_run_actions == {"compass.save-insights"}


def test_a_run_resolves_its_models_and_the_agents_ethos_as_the_caller(ctx: ClientContext) -> None:
    _enable_analysis(ctx)
    files = FilesClient.from_client(ctx.client)
    files.create_fileset(workspace=WORKSPACE, body=CreateFilesetRequest(name="support-bot-ethos"))
    files.upload_file(workspace=WORKSPACE, name="support-bot-ethos", path="ETHOS.md", content=ETHOS.encode())

    job = _run(ctx, "analyze-support-bot", {"agent": "support-bot", "default_model": "compass-default"})

    spec = job["spec"]
    assert spec["context"] == {
        "agent": "support-bot",
        "default_model": f"{WORKSPACE}/compass-default",
        "fast_model": f"{WORKSPACE}/compass-default",
        "ethos": ETHOS,
        "since": None,
        "evaluation_id": None,
    }
    assert spec["subject"] == "support-bot"
    assert spec["plugin_managed_agent_type"] == "compass.analyst"
    # The type's only action is chosen when the job names none.
    assert spec["post_run_action"]["kind"] == "compass.save-insights"


def test_a_run_naming_an_unknown_model_is_rejected_at_create(ctx: ClientContext) -> None:
    _enable_analysis(ctx)

    with pytest.raises(NemoHTTPError) as excinfo:
        _run(ctx, "analyze-bad-model", {"agent": "support-bot", "default_model": "nope"})

    assert excinfo.value.status_code == 422
    assert "default_model 'default/nope' is not a known Model Entity" in str(excinfo.value)
