# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire-level tests for ``nemo workspaces`` against a scripted typed client.

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

WORKSPACE = {
    "id": "8a4d8f1e-0f7f-4b8a-9d5f-2f4b6c8e1a2b",
    "name": "ml-team",
    "description": "Machine Learning Team workspace",
    "created_at": "2026-01-01T00:00:00Z",
    "created_by": "alice",
    "updated_at": "2026-01-01T00:00:00Z",
    "updated_by": None,
}

MEMBER = {
    "principal": "user@example.com",
    "roles": ["Editor"],
    "granted_at": "2026-01-01T00:00:00Z",
    "granted_by": "alice",
}


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
# workspaces get / delete
# ---------------------------------------------------------------------------


def test_get_by_name() -> None:
    recorder = Recorder([httpx.Response(200, json=WORKSPACE)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "get", "ml-team"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "GET"
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/ml-team"
    assert dict(recorder.last.url.params) == {}
    assert json.loads(result.stdout)["name"] == "ml-team"


def test_get_does_not_need_a_default_workspace() -> None:
    recorder = Recorder([httpx.Response(200, json=WORKSPACE)])
    runner, state = make_runner(recorder, workspace=None)

    result = runner.invoke(app, ["workspaces", "get", "ml-team"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/ml-team"


def test_get_not_found_maps_to_remote_error_exit_code() -> None:
    recorder = Recorder([httpx.Response(404, json={"detail": "Workspace 'nope' not found"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "get", "nope"], obj=state)

    assert result.exit_code == 3
    assert "Not found: (404) Workspace 'nope' not found" in result.stderr
    assert "nemo workspaces list" in result.stderr


def test_delete() -> None:
    recorder = Recorder([httpx.Response(200, json={"message": "deleted", "id": "ml-team", "deleted_at": None})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "delete", "ml-team"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "DELETE"
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/ml-team"
    assert "Deleted successfully" in result.stdout


def test_delete_not_found_maps_to_remote_error_exit_code() -> None:
    recorder = Recorder([httpx.Response(404, json={"detail": "Workspace 'nope' not found"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "delete", "nope"], obj=state)

    assert result.exit_code == 3
    assert "Not found" in result.stderr


# ---------------------------------------------------------------------------
# workspaces create
# ---------------------------------------------------------------------------


def test_create_sends_name_and_description_in_body() -> None:
    recorder = Recorder([httpx.Response(201, json=WORKSPACE)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app, ["workspaces", "create", "ml-team", "--description", "Machine Learning Team workspace"], obj=state
    )

    assert result.exit_code == 0, result.output
    assert len(recorder.requests) == 1
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == "/apis/entities/v2/workspaces"
    assert dict(recorder.last.url.params) == {}
    assert json.loads(recorder.last.content) == {"name": "ml-team", "description": "Machine Learning Team workspace"}
    assert json.loads(result.stdout)["name"] == "ml-team"


def test_create_without_description_omits_it_from_body() -> None:
    recorder = Recorder([httpx.Response(201, json=WORKSPACE)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "create", "ml-team"], obj=state)

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"name": "ml-team"}


def test_create_wait_role_propagation_flag_is_a_query_param() -> None:
    recorder = Recorder([httpx.Response(201, json=WORKSPACE)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "create", "ml-team", "--wait-role-propagation"], obj=state)

    assert result.exit_code == 0, result.output
    assert dict(recorder.last.url.params) == {"wait_role_propagation": "true"}
    assert json.loads(recorder.last.content) == {"name": "ml-team"}


def test_create_wait_role_propagation_false_from_input_data() -> None:
    recorder = Recorder([httpx.Response(201, json=WORKSPACE)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["workspaces", "create", "ml-team", "--input-data", '{"wait_role_propagation": false}'],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert dict(recorder.last.url.params) == {"wait_role_propagation": "false"}
    assert json.loads(recorder.last.content) == {"name": "ml-team"}


def test_create_reads_payload_from_stdin_and_flags_override() -> None:
    recorder = Recorder([httpx.Response(201, json=WORKSPACE)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["workspaces", "create", "ml-team", "--input-file", "-", "--description", "From CLI"],
        obj=state,
        input=json.dumps({"name": "from-stdin", "description": "From stdin"}),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"name": "ml-team", "description": "From CLI"}


def test_create_name_can_come_from_stdin_payload() -> None:
    recorder = Recorder([httpx.Response(201, json=WORKSPACE)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["workspaces", "create", "--input-file", "-"],
        obj=state,
        input=json.dumps({"name": "ml-team", "description": "From stdin"}),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"name": "ml-team", "description": "From stdin"}


def test_create_reads_payload_from_input_file(tmp_path: Path) -> None:
    recorder = Recorder([httpx.Response(201, json=WORKSPACE)])
    runner, state = make_runner(recorder)
    config = tmp_path / "ws.json"
    config.write_text(json.dumps({"name": "ml-team", "description": "From file"}))

    result = runner.invoke(app, ["workspaces", "create", "--input-file", str(config)], obj=state)

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"name": "ml-team", "description": "From file"}


def test_create_requires_name() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "create", "--description", "d"], obj=state)

    assert result.exit_code == 2
    assert "Missing required options" in result.stderr
    assert "--name" in result.stderr
    assert recorder.requests == []


def test_create_rejects_both_input_file_and_input_data() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app, ["workspaces", "create", "ml-team", "--input-file", "x.json", "--input-data", "{}"], obj=state
    )

    assert result.exit_code == 1
    assert "Cannot use both --input-file and --input-data" in result.stderr
    assert recorder.requests == []


def test_create_invalid_stdin_payload_is_a_data_error() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "create", "--input-file", "-"], obj=state, input="{{{{{")

    assert result.exit_code == 1
    assert "Invalid data" in result.stderr
    assert recorder.requests == []


def test_create_conflict_maps_to_remote_error_exit_code() -> None:
    recorder = Recorder([httpx.Response(409, json={"detail": "Workspace 'ml-team' already exists"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "create", "ml-team"], obj=state)

    assert result.exit_code == 3
    assert "Conflict: (409)" in result.stderr


def test_create_exist_ok_returns_existing_workspace_on_conflict() -> None:
    recorder = Recorder(
        [
            httpx.Response(409, json={"detail": "Workspace 'ml-team' already exists"}),
            httpx.Response(200, json=WORKSPACE),
        ]
    )
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "create", "ml-team", "--exist-ok"], obj=state)

    assert result.exit_code == 0, result.output
    assert [(r.method, r.url.path) for r in recorder.requests] == [
        ("POST", "/apis/entities/v2/workspaces"),
        ("GET", "/apis/entities/v2/workspaces/ml-team"),
    ]
    assert json.loads(result.stdout)["name"] == "ml-team"


def test_create_exist_ok_is_not_sent_on_the_wire() -> None:
    recorder = Recorder([httpx.Response(201, json=WORKSPACE)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "create", "ml-team", "--exist-ok"], obj=state)

    assert result.exit_code == 0, result.output
    assert len(recorder.requests) == 1
    assert dict(recorder.last.url.params) == {}
    assert json.loads(recorder.last.content) == {"name": "ml-team"}


# ---------------------------------------------------------------------------
# workspaces update
# ---------------------------------------------------------------------------


def test_update_sends_description() -> None:
    recorder = Recorder([httpx.Response(200, json={**WORKSPACE, "description": "new"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "update", "ml-team", "--description", "new"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "PUT"
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/ml-team"
    assert json.loads(recorder.last.content) == {"description": "new"}
    assert json.loads(result.stdout)["description"] == "new"


def test_update_without_fields_sends_empty_body() -> None:
    recorder = Recorder([httpx.Response(200, json=WORKSPACE)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "update", "ml-team"], obj=state)

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {}


def test_update_stdin_payload_with_flag_override() -> None:
    recorder = Recorder([httpx.Response(200, json=WORKSPACE)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["workspaces", "update", "ml-team", "--input-file", "-", "--description", "From CLI"],
        obj=state,
        input=json.dumps({"description": "From stdin"}),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"description": "From CLI"}


def test_update_input_data_payload() -> None:
    recorder = Recorder([httpx.Response(200, json=WORKSPACE)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app, ["workspaces", "update", "ml-team", "--input-data", '{"description": "From data"}'], obj=state
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"description": "From data"}


def test_update_requires_positional_name() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "update", "--description", "x"], obj=state)

    assert result.exit_code == 2
    assert "Missing required argument" in result.stderr or "NAME" in result.stderr
    assert recorder.requests == []


# ---------------------------------------------------------------------------
# workspaces list
# ---------------------------------------------------------------------------


def test_list_without_options_sends_no_query_params() -> None:
    recorder = Recorder([_page([WORKSPACE], 1, 1)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "list"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "GET"
    assert recorder.last.url.path == "/apis/entities/v2/workspaces"
    assert dict(recorder.last.url.params) == {}
    body = json.loads(result.stdout)
    assert [item["name"] for item in body["data"]] == ["ml-team"]
    assert body["pagination"]["total_results"] == 1


def test_list_passes_all_query_params_and_warns_about_more_pages() -> None:
    recorder = Recorder([_page([WORKSPACE], 1, 2)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["workspaces", "list", "--page", "1", "--page-size", "1", "--sort", "-name", "--filter", 'name~"ml"'],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert dict(recorder.last.url.params) == {"page": "1", "page_size": "1", "sort": "-name", "filter": 'name~"ml"'}
    assert json.loads(result.stdout)["pagination"]["total_pages"] == 2
    assert "More pages" in result.stderr


def test_list_rejects_unknown_sort_value() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "list", "--sort", "bogus"], obj=state)

    assert result.exit_code == 2
    assert recorder.requests == []


def test_list_all_pages_follows_every_page() -> None:
    recorder = Recorder([_page([WORKSPACE], 1, 2), _page([{**WORKSPACE, "name": "second"}], 2, 2)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "list", "--page-size", "1", "--all-pages"], obj=state)

    assert result.exit_code == 0, result.output
    assert [dict(r.url.params).get("page") for r in recorder.requests] == [None, "2"]
    body = json.loads(result.stdout)
    assert [item["name"] for item in body["data"]] == ["ml-team", "second"]
    assert body["pagination"]["total_results"] == 2
    assert body["pagination"]["total_pages"] == 1
    assert "More pages" not in result.stderr


def test_list_table_output_uses_default_columns() -> None:
    recorder = Recorder([_page([WORKSPACE], 1, 1)])
    runner, state = make_runner(recorder)
    state.overrides["output_format"] = "table"

    result = runner.invoke(app, ["workspaces", "list", "-f", "table"], obj=state)

    assert result.exit_code == 0, result.output
    output = result.stdout.lower()
    assert "name" in output and "description" in output and "created_at" in output
    assert "created_by" not in output


def test_list_stream_emits_one_json_record_per_item() -> None:
    recorder = Recorder([_page([WORKSPACE, {**WORKSPACE, "name": "second"}], 1, 1)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "list", "-f", "json", "--stream"], obj=state)

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert [record["name"] for record in records] == ["ml-team", "second"]


def test_list_stream_requires_json_or_raw_before_any_request() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "list", "-f", "table", "--stream"], obj=state)

    assert result.exit_code == 2
    assert "--stream requires --output json or --output raw" in result.stderr
    assert recorder.requests == []


def test_list_does_not_need_a_default_workspace() -> None:
    recorder = Recorder([_page([WORKSPACE], 1, 1)])
    runner, state = make_runner(recorder, workspace=None)

    result = runner.invoke(app, ["workspaces", "list"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/entities/v2/workspaces"


# ---------------------------------------------------------------------------
# workspaces members
# ---------------------------------------------------------------------------


def test_members_list_uses_client_default_workspace() -> None:
    recorder = Recorder([httpx.Response(200, json={"data": [MEMBER]})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "members", "list"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "GET"
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/default/members"
    assert dict(recorder.last.url.params) == {}
    assert [m["principal"] for m in json.loads(result.stdout)["data"]] == ["user@example.com"]


def test_members_list_explicit_workspace_overrides_default() -> None:
    recorder = Recorder([httpx.Response(200, json={"data": []})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "members", "list", "--workspace", "ml-team"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/ml-team/members"
    assert json.loads(result.stdout)["data"] == []


def test_members_list_without_any_workspace_is_usage_error() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder, workspace=None)

    result = runner.invoke(app, ["workspaces", "members", "list"], obj=state)

    assert result.exit_code == 2
    assert "Missing workspace" in result.stderr
    assert recorder.requests == []


def test_members_list_table_output_uses_default_columns() -> None:
    recorder = Recorder([httpx.Response(200, json={"data": [MEMBER]})])
    runner, state = make_runner(recorder)
    state.overrides["output_format"] = "table"

    result = runner.invoke(app, ["workspaces", "members", "list", "-f", "table"], obj=state)

    assert result.exit_code == 0, result.output
    output = result.stdout.lower()
    for column in ("principal", "roles", "granted_by", "granted_at"):
        assert column in output


def test_members_list_stream_emits_one_record_per_member() -> None:
    recorder = Recorder([httpx.Response(200, json={"data": [MEMBER, {**MEMBER, "principal": "b@example.com"}]})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "members", "list", "-f", "json", "--stream"], obj=state)

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert [record["principal"] for record in records] == ["user@example.com", "b@example.com"]


def test_members_create_sends_principal_and_roles() -> None:
    recorder = Recorder([httpx.Response(201, json={**MEMBER, "roles": ["Editor", "Viewer"]})])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "workspaces",
            "members",
            "create",
            "--principal",
            "user@example.com",
            "--roles",
            "Editor",
            "--roles",
            "Viewer",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert len(recorder.requests) == 1
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/default/members"
    assert dict(recorder.last.url.params) == {}
    assert json.loads(recorder.last.content) == {"principal": "user@example.com", "roles": ["Editor", "Viewer"]}
    assert json.loads(result.stdout)["roles"] == ["Editor", "Viewer"]


def test_members_create_without_roles_omits_them_so_the_server_default_applies() -> None:
    recorder = Recorder([httpx.Response(201, json=MEMBER)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app, ["workspaces", "members", "create", "--principal", "user@example.com", "--workspace", "ml-team"], obj=state
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/ml-team/members"
    assert json.loads(recorder.last.content) == {"principal": "user@example.com"}


def test_members_create_wait_role_propagation_is_a_query_param() -> None:
    recorder = Recorder([httpx.Response(201, json=MEMBER)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["workspaces", "members", "create", "--principal", "user@example.com", "--wait-role-propagation"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert dict(recorder.last.url.params) == {"wait_role_propagation": "true"}
    assert json.loads(recorder.last.content) == {"principal": "user@example.com"}


def test_members_create_payload_from_stdin_including_workspace() -> None:
    recorder = Recorder([httpx.Response(201, json=MEMBER)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["workspaces", "members", "create", "--input-file", "-"],
        obj=state,
        input=json.dumps(
            {
                "workspace": "ml-team",
                "principal": "user@example.com",
                "roles": ["Viewer"],
                "wait_role_propagation": False,
            }
        ),
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/ml-team/members"
    assert dict(recorder.last.url.params) == {"wait_role_propagation": "false"}
    assert json.loads(recorder.last.content) == {"principal": "user@example.com", "roles": ["Viewer"]}


def test_members_create_flags_override_input_data() -> None:
    recorder = Recorder([httpx.Response(201, json=MEMBER)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "workspaces",
            "members",
            "create",
            "--input-data",
            '{"principal": "from-data", "roles": ["Viewer"]}',
            "--principal",
            "user@example.com",
            "--roles",
            "Admin",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"principal": "user@example.com", "roles": ["Admin"]}


def test_members_create_requires_principal() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "members", "create", "--roles", "Editor"], obj=state)

    assert result.exit_code == 2
    assert "Missing required options" in result.stderr
    assert "--principal" in result.stderr
    assert recorder.requests == []


def test_members_create_without_any_workspace_is_usage_error() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder, workspace=None)

    result = runner.invoke(app, ["workspaces", "members", "create", "--principal", "user@example.com"], obj=state)

    assert result.exit_code == 2
    assert "Missing workspace" in result.stderr
    assert recorder.requests == []


def test_members_update_sends_roles_body() -> None:
    recorder = Recorder([httpx.Response(200, json={**MEMBER, "roles": ["Viewer", "Editor"]})])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["workspaces", "members", "update", "user@example.com", "--roles", "Viewer", "--roles", "Editor"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "PUT"
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/default/members/user@example.com"
    assert recorder.last.url.raw_path == b"/apis/entities/v2/workspaces/default/members/user%40example.com"
    assert dict(recorder.last.url.params) == {}
    assert json.loads(recorder.last.content) == {"roles": ["Viewer", "Editor"]}
    assert json.loads(result.stdout)["roles"] == ["Viewer", "Editor"]


def test_members_update_with_workspace_and_wait_role_propagation() -> None:
    recorder = Recorder([httpx.Response(200, json=MEMBER)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "workspaces",
            "members",
            "update",
            "user-123",
            "--workspace",
            "ml-team",
            "--roles",
            "Admin",
            "--wait-role-propagation",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/ml-team/members/user-123"
    assert dict(recorder.last.url.params) == {"wait_role_propagation": "true"}
    assert json.loads(recorder.last.content) == {"roles": ["Admin"]}


def test_members_update_roles_from_stdin_payload() -> None:
    recorder = Recorder([httpx.Response(200, json=MEMBER)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["workspaces", "members", "update", "user-123", "--input-file", "-"],
        obj=state,
        input=json.dumps({"roles": ["Viewer"], "workspace": "ml-team"}),
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/ml-team/members/user-123"
    assert json.loads(recorder.last.content) == {"roles": ["Viewer"]}


@pytest.mark.parametrize("roles", ["Editor", 5])
def test_members_update_rejects_scalar_roles_before_sending(roles: object) -> None:
    """A bare string must not be split into per-character roles (that revoked the member's real role)."""
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["workspaces", "members", "update", "user-123", "--input-data", json.dumps({"roles": roles})],
        obj=state,
    )

    assert result.exit_code == 2, result.output
    assert "Invalid input" in result.stderr
    assert "roles" in result.stderr
    assert recorder.requests == []


def test_members_create_rejects_scalar_roles_before_sending() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["workspaces", "members", "create", "--input-data", json.dumps({"principal": "u", "roles": "Editor"})],
        obj=state,
    )

    assert result.exit_code == 2, result.output
    assert "Invalid input" in result.stderr
    assert recorder.requests == []


def test_unknown_input_field_is_a_usage_error() -> None:
    """A typo in --input-data fails loudly instead of being dropped by pydantic's extra=ignore."""
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["workspaces", "create", "--input-data", json.dumps({"name": "ml-team", "descripton": "typo"})],
        obj=state,
    )

    assert result.exit_code == 2, result.output
    assert "Unknown input fields: descripton" in result.stderr
    assert "Accepted fields: description, name" in result.stderr
    assert recorder.requests == []


def test_members_update_requires_roles() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "members", "update", "user-123"], obj=state)

    assert result.exit_code == 2
    assert "Missing required options" in result.stderr
    assert "--roles" in result.stderr
    assert recorder.requests == []


def test_members_update_not_found_maps_to_remote_error_exit_code() -> None:
    recorder = Recorder([httpx.Response(404, json={"detail": "Member 'ghost' not found"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "members", "update", "ghost", "--roles", "Viewer"], obj=state)

    assert result.exit_code == 3
    assert "Not found: (404) Member 'ghost' not found" in result.stderr
    assert "nemo workspaces members list" in result.stderr


def test_members_delete() -> None:
    recorder = Recorder([httpx.Response(200, json={"message": "deleted", "id": "user-123", "deleted_at": None})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "members", "delete", "user-123"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "DELETE"
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/default/members/user-123"
    assert dict(recorder.last.url.params) == {}
    assert "Deleted successfully" in result.stdout


def test_members_delete_with_workspace_and_wait_role_propagation() -> None:
    recorder = Recorder([httpx.Response(200, json={"message": "deleted", "id": "user-123", "deleted_at": None})])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["workspaces", "members", "delete", "user-123", "--workspace", "ml-team", "--wait-role-propagation"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/ml-team/members/user-123"
    assert dict(recorder.last.url.params) == {"wait_role_propagation": "true"}


def test_members_delete_without_any_workspace_is_usage_error() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder, workspace=None)

    result = runner.invoke(app, ["workspaces", "members", "delete", "user-123"], obj=state)

    assert result.exit_code == 2
    assert "Missing workspace" in result.stderr
    assert recorder.requests == []


# ---------------------------------------------------------------------------
# -f code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "expected_call", "expected_type_import"),
    [
        (["workspaces", "list", "--page-size", "5", "--sort", "-name"], "client.list_workspaces(", None),
        (["workspaces", "get", "ml-team"], 'client.get_workspace(name="ml-team")', None),
        (
            ["workspaces", "create", "ml-team", "--description", "d", "--exist-ok"],
            "client.create_workspace(",
            "CreateWorkspaceRequest",
        ),
        (
            ["workspaces", "update", "ml-team", "--description", "d"],
            "client.update_workspace(",
            "UpdateWorkspaceRequest",
        ),
        (["workspaces", "members", "list", "--workspace", "ml-team"], "client.list_workspace_members(", None),
        (
            ["workspaces", "members", "create", "--principal", "user@example.com", "--roles", "Viewer"],
            "client.create_workspace_member(",
            "CreateWorkspaceMemberRequest",
        ),
        (
            ["workspaces", "members", "update", "user-123", "--roles", "Viewer", "--wait-role-propagation"],
            "client.update_workspace_member(",
            "UpdateWorkspaceMemberRequest",
        ),
    ],
    ids=lambda value: " ".join(value) if isinstance(value, list) else str(value),
)
def test_code_output_renders_typed_client_call_without_sending(
    argv: list[str], expected_call: str, expected_type_import: str | None
) -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, [*argv, "-f", "code"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "from nemo_platform_plugin.workspaces.client import WorkspacesClient" in result.stdout
    assert 'client = WorkspacesClient(base_url="http://test/")' in result.stdout
    assert expected_call in result.stdout
    if expected_type_import is not None:
        assert f"from nemo_platform_plugin.workspaces.types import {expected_type_import}" in result.stdout
    assert "NeMoPlatform" not in result.stdout
    assert "nemo_platform." not in result.stdout


def test_code_output_for_create_includes_query_params_and_exist_ok() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["workspaces", "create", "ml-team", "--exist-ok", "--wait-role-propagation", "-f", "code"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert 'CreateWorkspaceRequest(name="ml-team")' in result.stdout
    assert 'query_params={"wait_role_propagation": True}' in result.stdout
    assert "exist_ok=True" in result.stdout


def test_code_output_for_list_renders_query_params() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["workspaces", "list", "--page", "2", "--sort", "name", "-f", "code"], obj=state)

    assert result.exit_code == 0, result.output
    assert 'client.list_workspaces(query_params={"page": 2, "sort": "name"})' in result.stdout
    assert "for item in response.page().items:" in result.stdout
