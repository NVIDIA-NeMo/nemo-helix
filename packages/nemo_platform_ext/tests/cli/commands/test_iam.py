# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire-level tests for ``nemo iam role-bindings`` against a scripted typed client.

Each test drives the real Typer command with a ``NemoClient`` whose transport
records the HTTP requests, so the assertions pin the exact method, path, query
string, and JSON body the command sends.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx
import pytest
from nemo_platform_ext.cli.app import app
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_plugin.client.client import NemoClient
from typer.testing import CliRunner

BINDING = {
    "id": "id-1",
    "name": "rb-123",
    "principal": "user@example.com",
    "workspace": "default",
    "role": "Viewer",
    "granted_by": "service:test",
    "granted_at": "2026-01-01T00:00:00Z",
    "revoked_at": None,
}

ROLE_BINDINGS_PATH = "/apis/auth/v2/iam/role-bindings"


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
            "sort": "created_at",
            "filter": {},
        },
    )


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


def test_get_is_platform_scoped() -> None:
    recorder = Recorder([httpx.Response(200, json=BINDING)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["iam", "role-bindings", "get", "rb-123"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "GET"
    assert recorder.last.url.path == f"{ROLE_BINDINGS_PATH}/rb-123"
    assert dict(recorder.last.url.params) == {}
    assert json.loads(result.stdout)["name"] == "rb-123"


def test_get_does_not_need_a_workspace() -> None:
    recorder = Recorder([httpx.Response(200, json=BINDING)])
    runner, state = make_runner(recorder, workspace=None)

    result = runner.invoke(app, ["iam", "role-bindings", "get", "rb-123"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == f"{ROLE_BINDINGS_PATH}/rb-123"


def test_get_not_found_maps_to_remote_error_exit_code() -> None:
    recorder = Recorder([httpx.Response(404, json={"detail": "Role binding not found"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["iam", "role-bindings", "get", "nope"], obj=state)

    assert result.exit_code == 3
    assert "Not found: (404) Role binding not found" in result.stderr
    assert "nemo iam role-bindings list" in result.stderr


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def test_create_sends_only_provided_fields() -> None:
    recorder = Recorder([httpx.Response(200, json=BINDING)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["iam", "role-bindings", "create", "--principal", "user@example.com", "--role", "Viewer"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert len(recorder.requests) == 1
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == ROLE_BINDINGS_PATH
    assert dict(recorder.last.url.params) == {}
    assert json.loads(recorder.last.content) == {"principal": "user@example.com", "role": "Viewer"}
    assert json.loads(result.stdout)["name"] == "rb-123"


def test_create_with_workspace_and_wait_role_propagation() -> None:
    recorder = Recorder([httpx.Response(200, json=BINDING)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "iam",
            "role-bindings",
            "create",
            "--principal",
            "user@example.com",
            "--role",
            "Viewer",
            "--workspace",
            "ml-team",
            "--wait-role-propagation",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert dict(recorder.last.url.params) == {"wait_role_propagation": "true"}
    assert json.loads(recorder.last.content) == {
        "principal": "user@example.com",
        "workspace": "ml-team",
        "role": "Viewer",
    }


def test_create_workspace_is_body_not_path_and_ignores_client_default() -> None:
    recorder = Recorder([httpx.Response(200, json=BINDING)])
    runner, state = make_runner(recorder, workspace="default")

    result = runner.invoke(
        app, ["iam", "role-bindings", "create", "--principal", "u@example.com", "--role", "Admin"], obj=state
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == ROLE_BINDINGS_PATH
    assert "workspace" not in json.loads(recorder.last.content)


def test_create_requires_principal_and_role() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["iam", "role-bindings", "create", "--principal", "u@example.com"], obj=state)

    assert result.exit_code == 2
    assert "role" in result.stderr
    assert recorder.requests == []


def test_create_from_input_data_with_flag_override() -> None:
    recorder = Recorder([httpx.Response(200, json=BINDING)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "iam",
            "role-bindings",
            "create",
            "--input-data",
            '{"principal": "json@example.com", "role": "Viewer", "workspace": "ws", "wait_role_propagation": false}',
            "--role",
            "Editor",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert dict(recorder.last.url.params) == {"wait_role_propagation": "false"}
    assert json.loads(recorder.last.content) == {"principal": "json@example.com", "workspace": "ws", "role": "Editor"}


def test_create_from_stdin_file() -> None:
    recorder = Recorder([httpx.Response(200, json=BINDING)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["iam", "role-bindings", "create", "--input-file", "-"],
        obj=state,
        input=json.dumps({"principal": "piped@example.com", "role": "Viewer"}),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"principal": "piped@example.com", "role": "Viewer"}


def test_create_conflict_is_remote_error() -> None:
    recorder = Recorder([httpx.Response(409, json={"detail": "Active role binding already exists"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app, ["iam", "role-bindings", "create", "--principal", "u@example.com", "--role", "Viewer"], obj=state
    )

    assert result.exit_code == 3
    assert "Conflict" in result.stderr


# ---------------------------------------------------------------------------
# delete (revoke)
# ---------------------------------------------------------------------------


def test_delete_revokes_binding() -> None:
    recorder = Recorder([httpx.Response(200, json={"message": "Resource deleted successfully.", "id": "id-1"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["iam", "role-bindings", "delete", "rb-123"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "DELETE"
    assert recorder.last.url.path == f"{ROLE_BINDINGS_PATH}/rb-123"
    assert dict(recorder.last.url.params) == {}
    assert "Deleted successfully" in result.stdout


def test_delete_passes_wait_role_propagation() -> None:
    recorder = Recorder([httpx.Response(200, json={"message": "Resource deleted successfully.", "id": "id-1"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["iam", "role-bindings", "delete", "rb-123", "--wait-role-propagation"], obj=state)

    assert result.exit_code == 0, result.output
    assert dict(recorder.last.url.params) == {"wait_role_propagation": "true"}


def test_delete_not_found_is_remote_error() -> None:
    recorder = Recorder([httpx.Response(404, json={"detail": "Role binding not found"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["iam", "role-bindings", "delete", "nope"], obj=state)

    assert result.exit_code == 3
    assert "Not found" in result.stderr


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_without_options_sends_no_query_params() -> None:
    recorder = Recorder([_page([BINDING], 1, 1)])
    runner, state = make_runner(recorder, workspace=None)

    result = runner.invoke(app, ["iam", "role-bindings", "list"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "GET"
    assert recorder.last.url.path == ROLE_BINDINGS_PATH
    assert dict(recorder.last.url.params) == {}
    body = json.loads(result.stdout)
    assert [item["name"] for item in body["data"]] == ["rb-123"]
    assert "More pages" not in result.stderr


def test_list_passes_pagination_and_sort_and_warns() -> None:
    recorder = Recorder([_page([BINDING], 1, 2)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["iam", "role-bindings", "list", "--page", "1", "--page-size", "1", "--sort", "-created_at"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert dict(recorder.last.url.params) == {"page": "1", "page_size": "1", "sort": "-created_at"}
    assert json.loads(result.stdout)["pagination"]["total_pages"] == 2
    assert "More pages" in result.stderr


def test_list_json_output_keeps_server_sort_and_filter() -> None:
    recorder = Recorder([_page([BINDING], 1, 1)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["iam", "role-bindings", "list", "--sort", "created_at"], obj=state)

    assert result.exit_code == 0, result.output
    output = json.loads(result.stdout)
    assert output["sort"] == "created_at"
    assert output["filter"] == {}
    assert list(output) == ["data", "sort", "filter", "pagination"]


def test_list_field_filters_are_sent_as_json_filter() -> None:
    recorder = Recorder([_page([BINDING], 1, 1)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "iam",
            "role-bindings",
            "list",
            "--filter.principal",
            "user@example.com",
            "--filter.workspace",
            "default",
            "--filter.role",
            "Viewer",
            "--filter.granted-by",
            "service:test",
            "--filter.is-active",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    params = dict(recorder.last.url.params)
    assert set(params) == {"filter"}
    assert json.loads(params["filter"]) == {
        "principal": "user@example.com",
        "workspace": "default",
        "role": "Viewer",
        "granted_by": "service:test",
        "is_active": True,
    }


def test_list_text_filter_is_forwarded_verbatim() -> None:
    recorder = Recorder([_page([BINDING], 1, 1)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["iam", "role-bindings", "list", "--filter", 'role:"Viewer"'], obj=state)

    assert result.exit_code == 0, result.output
    assert dict(recorder.last.url.params) == {"filter": 'role:"Viewer"'}


def test_list_json_filter_merges_with_field_options() -> None:
    recorder = Recorder([_page([BINDING], 1, 1)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "iam",
            "role-bindings",
            "list",
            "--filter",
            '{"granted_at": {"$gte": "2026-01-01T00:00:00Z"}, "role": "Editor"}',
            "--filter.role",
            "Viewer",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert json.loads(dict(recorder.last.url.params)["filter"]) == {
        "granted_at": {"$gte": "2026-01-01T00:00:00Z"},
        "role": "Viewer",
    }


def test_list_text_filter_with_field_options_is_usage_error() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app, ["iam", "role-bindings", "list", "--filter", 'role:"Viewer"', "--filter.principal", "x"], obj=state
    )

    assert result.exit_code == 2
    assert recorder.requests == []


def test_list_all_pages_follows_every_page() -> None:
    recorder = Recorder([_page([BINDING], 1, 2), _page([{**BINDING, "name": "rb-456"}], 2, 2)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["iam", "role-bindings", "list", "--page-size", "1", "--all-pages"], obj=state)

    assert result.exit_code == 0, result.output
    assert [dict(r.url.params).get("page") for r in recorder.requests] == [None, "2"]
    body = json.loads(result.stdout)
    assert [item["name"] for item in body["data"]] == ["rb-123", "rb-456"]
    assert body["pagination"]["total_results"] == 2
    assert body["pagination"]["total_pages"] == 1
    assert "More pages" not in result.stderr


def test_list_table_output_uses_default_columns() -> None:
    recorder = Recorder([_page([BINDING], 1, 1)])
    runner, state = make_runner(recorder)
    state.overrides["output_format"] = "table"

    result = runner.invoke(app, ["iam", "role-bindings", "list", "-f", "table"], obj=state)

    assert result.exit_code == 0, result.output
    output = result.stdout.lower()
    assert "name" in output and "workspace" in output and "granted_at" in output
    assert "rb-123" in output and "2026-01-01" in output
    assert "principal" not in output and "user@example.com" not in output


def test_list_stream_emits_one_record_per_line() -> None:
    recorder = Recorder([_page([BINDING, {**BINDING, "name": "rb-456"}], 1, 1)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["iam", "role-bindings", "list", "-f", "json", "--stream"], obj=state)

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert [record["name"] for record in records] == ["rb-123", "rb-456"]


def test_list_forbidden_is_remote_error() -> None:
    recorder = Recorder([httpx.Response(403, json={"detail": "Forbidden"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["iam", "role-bindings", "list"], obj=state)

    assert result.exit_code == 3
    assert "(403)" in result.stderr


# ---------------------------------------------------------------------------
# -f code
# ---------------------------------------------------------------------------


def test_code_output_does_not_send_request() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "iam",
            "role-bindings",
            "create",
            "--principal",
            "user@example.com",
            "--role",
            "Viewer",
            "--workspace",
            "ml-team",
            "--wait-role-propagation",
            "-f",
            "code",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "from nemo_platform_plugin.iam.client import IAMClient" in result.stdout
    assert "from nemo_platform_plugin.iam.types import RoleBindingInput" in result.stdout
    assert 'client = IAMClient(base_url="http://test/")' in result.stdout
    assert 'RoleBindingInput(principal="user@example.com", workspace="ml-team", role="Viewer")' in result.stdout
    assert 'query_params={"wait_role_propagation": True}' in result.stdout
    assert "NeMoPlatform" not in result.stdout


def test_code_output_for_list_renders_filter_query() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app, ["iam", "role-bindings", "list", "--filter.role", "Viewer", "--page", "2", "-f", "code"], obj=state
    )

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "client.list_role_bindings(" in result.stdout
    assert '"page": 2' in result.stdout
    assert "Viewer" in result.stdout
    assert "for item in response.page().items:" in result.stdout
    assert "NeMoPlatform" not in result.stdout
