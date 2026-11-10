# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo agents list-jobs`` — wire-level tests.

Agents replaces the generated ``list-jobs`` to add ``--agent``, which lists an agent's execute
runs. The command runs against a stand-in ``nemo`` CLI state whose typed client talks to a
recording ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TypeVar
from urllib.parse import parse_qs

import httpx
import pytest
import typer
from click.testing import Result
from nemo_agents_plugin.cli import AgentsCLI
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.commands import add_job_commands
from nemo_helix_plugin.discovery import discover_entry_points
from typer.testing import CliRunner

BASE_URL = "http://nhx.test"
JOBS = "/apis/agents/v2/workspaces/team-a/jobs"

ClientT = TypeVar("ClientT", bound=NemoClient)


@dataclass
class _Api:
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        name = request.url.path.rsplit("/", 1)[-1]
        job = {"name": f"{name}-1", "workspace": "team-a", "created_at": "2026-01-01T00:00:00Z", "spec": {}}
        return httpx.Response(
            200,
            json={
                "data": [job],
                "pagination": {
                    "page": 1,
                    "page_size": 10,
                    "current_page_size": 1,
                    "total_pages": 1,
                    "total_results": 1,
                },
            },
        )

    def filter(self, index: int = 0) -> Any:
        return json.loads(parse_qs(self.requests[index].url.query.decode())["filter"][0])


@dataclass
class _State:
    """The slice of the ``nemo`` CLI state (``CLIState``) these commands use."""

    api: _Api

    def typed_client(self, client_cls: type[ClientT], timeout: float = 60.0) -> ClientT:
        http_client = httpx.Client(transport=httpx.MockTransport(self.api))
        return client_cls.from_client(NemoClient(base_url=BASE_URL, http_client=http_client))

    def get_workspace(self) -> str | None:
        return "team-a"

    def get_base_url(self, default: str | None = None) -> str | None:
        return BASE_URL

    def get_output_format(self, override: str | None = None, *, apply_non_tty_default: bool = True) -> str:
        return override or "json"

    def get_no_truncate(self, override: bool | None = None) -> bool:
        return True

    def get_timestamp_format(self, override: str | None = None) -> str:
        return "iso8601"


@pytest.fixture
def app() -> typer.Typer:
    return AgentsCLI().get_cli()


@pytest.fixture(autouse=True)
def _no_workspace_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NHX_WORKSPACE", raising=False)


def _invoke(app: typer.Typer, api: _Api, *args: str) -> Result:
    return CliRunner().invoke(app, ["list-jobs", *args], obj=_State(api))


def test_generated_list_jobs_does_not_replace_the_agents_command(app: typer.Typer) -> None:
    jobs = {key: entry.load() for key, entry in discover_entry_points("nemo.jobs").items() if key.startswith("agents.")}
    add_job_commands(app, jobs)

    result = CliRunner().invoke(app, ["list-jobs", "--help"], obj=_State(_Api()))

    assert result.exit_code == 0, result.output
    assert "--agent" in result.stdout
    assert [c.name for c in app.registered_commands].count("list-jobs") == 1


def test_lists_every_agents_job_type(app: typer.Typer) -> None:
    api = _Api()

    result = _invoke(app, api)

    assert result.exit_code == 0, result.output
    assert [r.url.path for r in api.requests] == [f"{JOBS}/evaluate", f"{JOBS}/execute", f"{JOBS}/package"]
    assert {job["type"] for job in json.loads(result.stdout)["data"]} == {
        "agents.evaluate",
        "agents.execute",
        "agents.package-agent",
    }


def test_agent_lists_execute_runs_and_qualifies_bare_name(app: typer.Typer) -> None:
    api = _Api()

    result = _invoke(app, api, "--agent", "calc", "-a", "research/calc")

    assert result.exit_code == 0, result.output
    assert [r.url.path for r in api.requests] == [f"{JOBS}/execute"]
    assert api.filter() == {"spec.agent_ref": {"$in": ["team-a/calc", "research/calc"]}}


def test_agent_qualifies_with_the_workspace_flag(app: typer.Typer) -> None:
    api = _Api()

    result = _invoke(app, api, "--agent", "calc", "--workspace", "research")

    assert result.exit_code == 0, result.output
    assert api.requests[0].url.path == "/apis/agents/v2/workspaces/research/jobs/execute"
    assert api.filter() == {"spec.agent_ref": {"$in": ["research/calc"]}}


def test_agent_combines_with_other_filters_and_execute_type(app: typer.Typer) -> None:
    api = _Api()

    result = _invoke(app, api, "--type", "execute", "--agent", "calc", "--filter.status", "completed")

    assert result.exit_code == 0, result.output
    assert api.filter() == {"status": "completed", "spec.agent_ref": {"$in": ["team-a/calc"]}}


def test_agent_across_workspaces_needs_qualified_refs(app: typer.Typer) -> None:
    api = _Api()

    result = _invoke(app, api, "--agent", "research/calc", "--workspace", "-")

    assert result.exit_code == 0, result.output
    assert api.requests[0].url.path == "/apis/agents/v2/workspaces/-/jobs/execute"
    assert api.filter() == {"spec.agent_ref": {"$in": ["research/calc"]}}


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--agent", "calc", "--workspace", "-"], "pass it as 'workspace/calc'"),
        (["--agent", "a/b/c"], "invalid entity reference"),
        (["--agent", "calc", "--type", "evaluate"], "only execute jobs record their agent"),
        (["--agent", "calc", "--filter", 'name~"x"'], "pass --filter as JSON"),
    ],
)
def test_agent_usage_errors_send_nothing(app: typer.Typer, args: list[str], message: str) -> None:
    api = _Api()

    result = _invoke(app, api, *args)

    assert result.exit_code == 2, result.output
    assert message in " ".join(result.output.split())
    assert api.requests == []
