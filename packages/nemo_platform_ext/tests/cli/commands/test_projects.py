# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire-level tests for ``nemo projects`` against a scripted typed client.

Each test drives the real Typer command with a ``NemoClient`` whose transport
records the HTTP requests, so the assertions pin the exact method, path, query
string, and JSON body the command sends.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from nemo_platform_ext.cli.app import app
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_plugin.client.client import NemoClient
from typer.testing import CliRunner

PROJECT = {
    "id": "project-8a4d8f1e",
    "name": "ml-project",
    "workspace": "default",
    "description": "Machine Learning project",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
}

PROJECTS_PATH = "/apis/entities/v2/workspaces/default/projects"


@dataclass
class Recorder:
    """Records requests and answers each with the next scripted response."""

    responses: list[httpx.Response]
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(f"unexpected request {request.method} {request.url}")
        return self.responses.pop(0)

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]


def make_runner(recorder: Recorder, *, workspace: str | None = "default") -> tuple[CliRunner, CLIContext]:
    client = NemoClient(
        base_url="http://test",
        workspace=workspace,
        http_client=httpx.Client(transport=httpx.MockTransport(recorder)),
    )
    state = CLIContext(overrides={"base_url": "http://test", "output_format": "json"}, _client=client)
    return CliRunner(), state


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.touch()
    monkeypatch.setenv("NMP_CONFIG_FILE", str(config_file))
    monkeypatch.delenv("NMP_ACCESS_TOKEN", raising=False)


def _page(items: list[dict], page: int, total_pages: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": items,
            "pagination": {
                "page": page,
                "page_size": 1,
                "current_page_size": len(items),
                "total_pages": total_pages,
                "total_results": total_pages,
            },
        },
    )


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


def test_get_uses_client_default_workspace() -> None:
    recorder = Recorder([httpx.Response(200, json=PROJECT)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["projects", "get", "ml-project"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "GET"
    assert recorder.last.url.path == f"{PROJECTS_PATH}/ml-project"
    assert dict(recorder.last.url.params) == {}
    assert json.loads(result.stdout)["name"] == "ml-project"


def test_get_explicit_workspace_overrides_default() -> None:
    recorder = Recorder([httpx.Response(200, json={**PROJECT, "workspace": "other"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["projects", "get", "ml-project", "--workspace", "other"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/other/projects/ml-project"


def test_get_without_any_workspace_is_usage_error() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder, workspace=None)

    result = runner.invoke(app, ["projects", "get", "ml-project"], obj=state)

    assert result.exit_code == 2
    assert "Missing workspace" in result.stderr
    assert recorder.requests == []


def test_get_not_found_maps_to_remote_error_exit_code() -> None:
    recorder = Recorder([httpx.Response(404, json={"detail": "Project 'nope' not found"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["projects", "get", "nope"], obj=state)

    assert result.exit_code == 3
    assert "Not found: (404) Project 'nope' not found" in result.stderr
    assert "nemo projects list" in result.stderr


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def test_create_sends_only_provided_fields() -> None:
    recorder = Recorder([httpx.Response(201, json=PROJECT)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app, ["projects", "create", "ml-project", "--description", "Machine Learning project"], obj=state
    )

    assert result.exit_code == 0, result.output
    assert len(recorder.requests) == 1
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == PROJECTS_PATH
    assert dict(recorder.last.url.params) == {}
    assert json.loads(recorder.last.content) == {"name": "ml-project", "description": "Machine Learning project"}
    assert json.loads(result.stdout)["id"] == "project-8a4d8f1e"


def test_create_without_description_omits_it_from_body() -> None:
    recorder = Recorder([httpx.Response(201, json=PROJECT)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["projects", "create", "ml-project"], obj=state)

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"name": "ml-project"}


def test_create_explicit_workspace_is_used_in_path() -> None:
    recorder = Recorder([httpx.Response(201, json={**PROJECT, "workspace": "other"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["projects", "create", "ml-project", "--workspace", "other"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/other/projects"


def test_create_requires_name() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["projects", "create", "--description", "d"], obj=state)

    assert result.exit_code == 2
    assert "name" in result.stderr
    assert recorder.requests == []


def test_create_conflict_without_exist_ok_is_remote_error() -> None:
    recorder = Recorder([httpx.Response(409, json={"detail": "Project already exists"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["projects", "create", "ml-project"], obj=state)

    assert result.exit_code == 3
    assert "Conflict" in result.stderr
    assert len(recorder.requests) == 1


def test_create_exist_ok_replays_get_on_conflict() -> None:
    recorder = Recorder(
        [httpx.Response(409, json={"detail": "Project already exists"}), httpx.Response(200, json=PROJECT)]
    )
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["projects", "create", "ml-project", "--exist-ok"], obj=state)

    assert result.exit_code == 0, result.output
    assert [(r.method, r.url.path) for r in recorder.requests] == [
        ("POST", PROJECTS_PATH),
        ("GET", f"{PROJECTS_PATH}/ml-project"),
    ]
    assert json.loads(result.stdout)["name"] == "ml-project"


def test_create_from_input_data_with_flag_override() -> None:
    recorder = Recorder([httpx.Response(201, json=PROJECT)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "projects",
            "create",
            "--input-data",
            '{"name": "from-json", "description": "json desc", "workspace": "json-ws"}',
            "--description",
            "flag desc",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/json-ws/projects"
    assert json.loads(recorder.last.content) == {"name": "from-json", "description": "flag desc"}


def test_create_from_stdin_file(tmp_path: Path) -> None:
    recorder = Recorder([httpx.Response(201, json=PROJECT)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["projects", "create", "--input-file", "-"],
        obj=state,
        input=json.dumps({"name": "piped", "description": "from stdin"}),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"name": "piped", "description": "from stdin"}


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


def test_update_only_sends_provided_fields() -> None:
    recorder = Recorder([httpx.Response(200, json={**PROJECT, "description": "new"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["projects", "update", "ml-project", "--description", "new"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "PUT"
    assert recorder.last.url.path == f"{PROJECTS_PATH}/ml-project"
    assert json.loads(recorder.last.content) == {"description": "new"}
    assert json.loads(result.stdout)["description"] == "new"


def test_update_without_fields_sends_empty_body() -> None:
    recorder = Recorder([httpx.Response(200, json=PROJECT)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["projects", "update", "ml-project"], obj=state)

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {}


def test_update_workspace_from_input_data() -> None:
    recorder = Recorder([httpx.Response(200, json=PROJECT)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["projects", "update", "ml-project", "--input-data", '{"workspace": "json-ws", "description": "d"}'],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/json-ws/projects/ml-project"
    assert json.loads(recorder.last.content) == {"description": "d"}


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def test_delete() -> None:
    recorder = Recorder([httpx.Response(200, json={"message": "Resource deleted successfully.", "id": "p-1"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["projects", "delete", "ml-project"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "DELETE"
    assert recorder.last.url.path == f"{PROJECTS_PATH}/ml-project"
    assert "Deleted successfully" in result.stdout


def test_delete_explicit_workspace() -> None:
    recorder = Recorder([httpx.Response(200, json={"message": "Resource deleted successfully.", "id": "p-1"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["projects", "delete", "ml-project", "--workspace", "other"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/other/projects/ml-project"


def test_delete_not_found_is_remote_error() -> None:
    recorder = Recorder([httpx.Response(404, json={"detail": "Project 'nope' not found"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["projects", "delete", "nope"], obj=state)

    assert result.exit_code == 3
    assert "Not found" in result.stderr


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_without_options_sends_no_query_params() -> None:
    recorder = Recorder([_page([PROJECT], 1, 1)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["projects", "list"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "GET"
    assert recorder.last.url.path == PROJECTS_PATH
    assert dict(recorder.last.url.params) == {}
    body = json.loads(result.stdout)
    assert [item["name"] for item in body["data"]] == ["ml-project"]
    assert body["pagination"]["total_pages"] == 1
    assert "More pages" not in result.stderr


def test_list_passes_query_params_and_warns_on_more_pages() -> None:
    recorder = Recorder([_page([PROJECT], 1, 2)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "projects",
            "list",
            "--workspace",
            "other",
            "--filter",
            'name~"ml"',
            "--page",
            "1",
            "--page-size",
            "1",
            "--sort",
            "-created_at",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/other/projects"
    assert dict(recorder.last.url.params) == {
        "filter": 'name~"ml"',
        "page": "1",
        "page_size": "1",
        "sort": "-created_at",
    }
    assert json.loads(result.stdout)["pagination"]["total_pages"] == 2
    assert "More pages" in result.stderr


def test_list_rejects_unknown_sort_field() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["projects", "list", "--sort", "bogus"], obj=state)

    assert result.exit_code == 2
    assert recorder.requests == []


def test_list_all_pages_follows_every_page() -> None:
    recorder = Recorder([_page([PROJECT], 1, 2), _page([{**PROJECT, "name": "second"}], 2, 2)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["projects", "list", "--page-size", "1", "--all-pages"], obj=state)

    assert result.exit_code == 0, result.output
    assert [dict(r.url.params).get("page") for r in recorder.requests] == [None, "2"]
    body = json.loads(result.stdout)
    assert [item["name"] for item in body["data"]] == ["ml-project", "second"]
    assert body["pagination"]["total_results"] == 2
    assert body["pagination"]["total_pages"] == 1
    assert "More pages" not in result.stderr


def test_list_table_output_uses_default_columns() -> None:
    recorder = Recorder([_page([PROJECT], 1, 1)])
    runner, state = make_runner(recorder)
    state.overrides["output_format"] = "table"

    result = runner.invoke(app, ["projects", "list", "-f", "table"], obj=state)

    assert result.exit_code == 0, result.output
    output = result.stdout.lower()
    assert "name" in output and "description" in output and "created_at" in output
    assert "workspace" not in output


def test_list_stream_emits_one_record_per_line() -> None:
    recorder = Recorder([_page([PROJECT, {**PROJECT, "name": "second"}], 1, 1)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["projects", "list", "-f", "json", "--stream"], obj=state)

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert [record["name"] for record in records] == ["ml-project", "second"]


def test_list_without_any_workspace_is_usage_error() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder, workspace=None)

    result = runner.invoke(app, ["projects", "list"], obj=state)

    assert result.exit_code == 2
    assert "Missing workspace" in result.stderr
    assert recorder.requests == []


# ---------------------------------------------------------------------------
# -f code
# ---------------------------------------------------------------------------


def test_code_output_does_not_send_request() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["projects", "create", "ml-project", "--description", "d", "--workspace", "other", "--exist-ok", "-f", "code"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "from nemo_platform_plugin.projects.client import ProjectsClient" in result.stdout
    assert "from nemo_platform_plugin.projects.types import CreateProjectRequest" in result.stdout
    assert 'client = ProjectsClient(base_url="http://test/")' in result.stdout
    assert 'workspace="other"' in result.stdout
    assert 'CreateProjectRequest(name="ml-project", description="d")' in result.stdout
    assert "exist_ok=True" in result.stdout
    assert "NeMoPlatform" not in result.stdout


def test_code_output_for_list_renders_query_params() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["projects", "list", "--page-size", "5", "--sort", "name", "-f", "code"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "client.list_projects(" in result.stdout
    assert 'query_params={"page_size": 5, "sort": "name"}' in result.stdout
    assert "for item in response.page().items:" in result.stdout
    assert "NeMoPlatform" not in result.stdout
