# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent labels and listing an agent's execute runs, through the real services and entity store."""

from typing import Any

import pytest
from nemo_agents_plugin.sdk import AgentsResource
from nemo_agents_plugin.service import AgentsService
from nemo_helix_plugin.jobs.client import JobsClient
from nhx.core.files.service import FilesService
from nhx.core.jobs.service import JobsService
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter
from nhx.testing import ClientContext, create_test_client

pytestmark = pytest.mark.integration


class _TestAgentsService(NemoServiceAdapter):
    def __init__(self) -> None:
        super().__init__(AgentsService())


def _agent_config(name: str) -> dict[str, Any]:
    return {
        "config_format": "nemo-agents-spec-v1",
        "name": name,
        "default_harness": "hermes",
        "harnesses": {"hermes": {"kind": "hermes"}},
        "models": {"default": {"provider": "openai", "model": "openai/gpt-5.4"}},
    }


def _names(page: dict[str, Any]) -> set[str]:
    return {item["name"] for item in page["data"]}


def test_list_agents_by_label() -> None:
    with create_test_client(_TestAgentsService, client_type=ClientContext, workspace="default") as ctx:
        agents = AgentsResource(ctx.client)
        for name, labels in [
            ("support-analyst", {"team": "support", "helix-compass/target-agent": "default/bot"}),
            ("support-triage", {"team": "support"}),
            ("billing-analyst", {"team": "billing"}),
            ("unlabeled", None),
        ]:
            agents.register(name=name, config=_agent_config(name), labels=labels)

        assert _names(agents.list(labels={"team": "support"})) == {"support-analyst", "support-triage"}
        assert _names(agents.list(labels={"team": "support", "helix-compass/target-agent": "default/bot"})) == {
            "support-analyst"
        }
        assert _names(agents.list(labels={"team": "nobody"})) == set()
        assert agents.get("unlabeled")["labels"] == {}


def test_list_execute_runs_by_agent() -> None:
    with create_test_client(
        _TestAgentsService,
        FilesService,
        JobsService,
        client_type=ClientContext,
        workspaces=["default", "research"],
        workspace="default",
    ) as ctx:
        agents = AgentsResource(ctx.client)
        agents.register(name="calc", config=_agent_config("calc"))
        agents.register(name="other", config=_agent_config("other"))
        agents.register(name="calc", config=_agent_config("calc"), workspace="research")

        execute_jobs = agents.jobs.execute

        def execute(name: str, agent: str | dict[str, Any]) -> dict[str, Any]:
            return execute_jobs.create(name=name, spec={"agent": agent, "input": "hello"})

        assert execute("run-calc", "calc")["spec"]["agent_ref"] == "default/calc"
        execute("run-calc-qualified", "default/calc")
        execute("run-other", "other")
        assert execute("run-research-calc", "research/calc")["spec"]["agent_ref"] == "research/calc"
        inline = execute("run-inline", {"config_format": "nemo-agents-spec-v1", "config": _agent_config("calc")})
        assert inline["spec"].get("agent_ref") is None

        assert _names(execute_jobs.list(agents=["default/calc"])) == {"run-calc", "run-calc-qualified"}
        assert _names(execute_jobs.list(agents=["research/calc"])) == {"run-research-calc"}
        assert _names(execute_jobs.list(agents=["default/calc", "default/other"])) == {
            "run-calc",
            "run-calc-qualified",
            "run-other",
        }
        assert _names(execute_jobs.list()) == {
            "run-calc",
            "run-calc-qualified",
            "run-other",
            "run-research-calc",
            "run-inline",
        }

        # The documented CLI route: core jobs, scoped to execute jobs by source.
        core_jobs = JobsClient.from_client(ctx.client).list_jobs(
            query_params={
                "filter": {"spec.agent_ref": {"$in": ["default/calc"]}, "source": "nemo-agents-plugin-execute"}
            }
        )
        assert {job.name for job in core_jobs.page().items} == {"run-calc", "run-calc-qualified"}
