# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent labels and listing an agent's execute runs, through the real services and entity store."""

import json
from dataclasses import dataclass
from typing import Any, TypeVar

import pytest
from nemo_agents_plugin.cli import AgentsCLI
from nemo_agents_plugin.sdk import AgentsResource
from nemo_agents_plugin.service import AgentsService
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceQueryParams, CreateWorkspaceRequest
from nhx.core.files.service import FilesService
from nhx.core.jobs.service import JobsService
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter
from nhx.testing import (
    TEST_ADMIN_EMAIL,
    ClientContext,
    as_user,
    create_test_client,
    grant_workspace_role,
    short_unique_name,
    unique_email,
)
from typer.testing import CliRunner

pytestmark = pytest.mark.integration

ClientT = TypeVar("ClientT", bound=NemoClient)


@dataclass
class _CLIState:
    """The slice of the ``nemo`` CLI state ``list-jobs`` uses, over the in-process platform."""

    client: NemoClient

    def typed_client(self, client_cls: type[ClientT], timeout: float = 60.0) -> ClientT:
        return client_cls.from_client(self.client)

    def get_workspace(self) -> str | None:
        return "default"

    def get_base_url(self, default: str | None = None) -> str | None:
        return default

    def get_output_format(self, override: str | None = None, *, apply_non_tty_default: bool = True) -> str:
        return override or "json"

    def get_no_truncate(self, override: bool | None = None) -> bool:
        return True

    def get_timestamp_format(self, override: str | None = None) -> str:
        return "iso8601"


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

        # A bare name means the agent in the listing workspace.
        assert _names(execute_jobs.list(agents=["calc"])) == {"run-calc", "run-calc-qualified"}
        assert _names(execute_jobs.list("research", agents=["calc"])) == set()

        # The documented CLI command lists through the execute job route.
        result = CliRunner().invoke(
            AgentsCLI().get_cli(), ["list-jobs", "--agent", "calc", "-a", "research/calc"], obj=_CLIState(ctx.client)
        )
        assert result.exit_code == 0, result.output
        listed = json.loads(result.stdout)["data"]
        assert {job["name"] for job in listed} == {"run-calc", "run-calc-qualified", "run-research-calc"}
        assert {job["type"] for job in listed} == {"agents.execute"}

        # Without --agent it lists every agents job type; only execute jobs exist here.
        result = CliRunner().invoke(AgentsCLI().get_cli(), ["list-jobs", "--all-pages"], obj=_CLIState(ctx.client))
        assert result.exit_code == 0, result.output
        assert {job["name"] for job in json.loads(result.stdout)["data"]} == _names(execute_jobs.list())

        # The core jobs route still works, scoped to execute jobs by source.
        core_jobs = JobsClient.from_client(ctx.client).list_jobs(
            query_params={
                "filter": {"spec.agent_ref": {"$in": ["default/calc"]}, "source": "nemo-agents-plugin-execute"}
            }
        )
        assert {job.name for job in core_jobs.page().items} == {"run-calc", "run-calc-qualified"}


def test_listing_execute_runs_across_workspaces_returns_only_readable_workspaces() -> None:
    """Workspace ``-`` lists every workspace the caller can read, and no others, through the typed route.

    The agents route forwards ``-`` to core Jobs as the caller, and the entity store narrows it to
    the caller's readable workspaces. Every user can read ``default``, so both workspaces here are
    new ones: the reader is granted Viewer on one, and the other belongs to someone else.
    """
    with create_test_client(
        _TestAgentsService, FilesService, JobsService, auth_enabled=True, client_type=ClientContext
    ) as ctx:
        readable, hidden = short_unique_name("agents-readable"), short_unique_name("agents-hidden")
        admin = as_user(ctx.client, TEST_ADMIN_EMAIL)
        for workspace in (readable, hidden):
            WorkspacesClient.from_client(admin).create_workspace(
                query_params=CreateWorkspaceQueryParams(wait_role_propagation=True),
                body=CreateWorkspaceRequest(name=workspace),
            ).data()
            admin_agents = AgentsResource(admin)
            admin_agents.register(name="calc", config=_agent_config("calc"), workspace=workspace)
            admin_agents.jobs.execute.create(
                name=f"run-{workspace}", spec={"agent": f"{workspace}/calc", "input": "hello"}, workspace=workspace
            )

        reader_email = unique_email("agents-reader")
        grant_workspace_role(admin, workspace=readable, principal=reader_email, roles=["Viewer"])
        reader = as_user(ctx.client, reader_email)
        reader_jobs = AgentsResource(reader).jobs.execute
        both_agents = [f"{readable}/calc", f"{hidden}/calc"]

        # The admin sees both runs, so the reader's narrower view below is the filter, not missing data.
        assert _names(AgentsResource(admin).jobs.execute.list("-", agents=both_agents)) == {
            f"run-{readable}",
            f"run-{hidden}",
        }

        assert _names(reader_jobs.list("-", agents=both_agents)) == {f"run-{readable}"}
        assert _names(reader_jobs.list("-")) == {f"run-{readable}"}
        with pytest.raises(PermissionDeniedError):
            reader_jobs.list(hidden)

        # The documented CLI command, as the reader.
        result = CliRunner().invoke(
            AgentsCLI().get_cli(),
            ["list-jobs", "--workspace", "-", "--agent", both_agents[0], "--agent", both_agents[1]],
            obj=_CLIState(reader),
        )
        assert result.exit_code == 0, result.output
        assert {job["name"] for job in json.loads(result.stdout)["data"]} == {f"run-{readable}"}
