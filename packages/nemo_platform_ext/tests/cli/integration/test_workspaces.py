# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for ``nemo workspaces`` against the in-process Entities service."""

import json
import uuid

from nemo_platform_ext.cli.app import app


def _name() -> str:
    return f"ws-{uuid.uuid4().hex[:8]}"


def test_workspaces_lifecycle(runner, assert_exit_code) -> None:
    name = _name()

    result = runner.invoke(app, ["workspaces", "create", name, "--description", "demo"])
    assert_exit_code(result, 0)
    created = json.loads(result.stdout)
    assert created["name"] == name
    assert created["description"] == "demo"
    assert created["id"]

    result = runner.invoke(app, ["workspaces", "get", name])
    assert_exit_code(result, 0)
    fetched = json.loads(result.stdout)
    assert fetched["id"] == created["id"]

    result = runner.invoke(app, ["workspaces", "list", "--filter", f'name:"{name}"'])
    assert_exit_code(result, 0)
    listed = json.loads(result.stdout)
    assert [item["name"] for item in listed["data"]] == [name]
    assert listed["pagination"]["total_results"] == 1

    result = runner.invoke(app, ["workspaces", "update", name, "--description", "changed"])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["description"] == "changed"
    result = runner.invoke(app, ["workspaces", "get", name])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["description"] == "changed"

    result = runner.invoke(app, ["workspaces", "delete", name])
    assert_exit_code(result, 0)
    assert "Deleted successfully" in result.stdout

    result = runner.invoke(app, ["workspaces", "get", name])
    assert_exit_code(result, 3)
    assert "Not found" in result.stderr

    result = runner.invoke(app, ["workspaces", "list", "--filter", f'name:"{name}"'])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["data"] == []


def test_workspaces_create_conflict_and_exist_ok(runner, assert_exit_code) -> None:
    name = _name()
    assert_exit_code(runner.invoke(app, ["workspaces", "create", name, "--description", "first"]), 0)

    result = runner.invoke(app, ["workspaces", "create", name])
    assert_exit_code(result, 3)
    assert "Conflict" in result.stderr

    result = runner.invoke(app, ["workspaces", "create", name, "--exist-ok"])
    assert_exit_code(result, 0)
    existing = json.loads(result.stdout)
    assert existing["name"] == name
    assert existing["description"] == "first"


def test_workspaces_create_rejects_invalid_name(runner, assert_exit_code) -> None:
    result = runner.invoke(app, ["workspaces", "create", "Invalid_Name"])
    assert_exit_code(result, 3)
    assert "(422)" in result.stderr


def test_workspaces_create_wait_role_propagation_from_input_data(runner, assert_exit_code) -> None:
    name = _name()
    result = runner.invoke(
        app, ["workspaces", "create", name, "--input-data", '{"wait_role_propagation": false, "description": "bulk"}']
    )
    assert_exit_code(result, 0)
    created = json.loads(result.stdout)
    assert created["name"] == name
    assert created["description"] == "bulk"


def test_workspaces_list_pagination_and_all_pages(runner, assert_exit_code) -> None:
    prefix = f"pg{uuid.uuid4().hex[:6]}"
    names = sorted(f"{prefix}-{i}" for i in range(3))
    for name in names:
        assert_exit_code(runner.invoke(app, ["workspaces", "create", name]), 0)
    name_filter = f'name~"{prefix}"'

    result = runner.invoke(app, ["workspaces", "list", "--filter", name_filter, "--page-size", "2", "--sort", "name"])
    assert_exit_code(result, 0)
    first_page = json.loads(result.stdout)
    assert [item["name"] for item in first_page["data"]] == names[:2]
    assert first_page["pagination"]["total_pages"] == 2
    assert first_page["pagination"]["total_results"] == 3
    assert "More pages" in result.stderr

    result = runner.invoke(
        app, ["workspaces", "list", "--filter", name_filter, "--page-size", "2", "--sort", "name", "--page", "2"]
    )
    assert_exit_code(result, 0)
    assert [item["name"] for item in json.loads(result.stdout)["data"]] == names[2:]

    result = runner.invoke(
        app, ["workspaces", "list", "--filter", name_filter, "--page-size", "2", "--sort", "name", "--all-pages"]
    )
    assert_exit_code(result, 0)
    all_pages = json.loads(result.stdout)
    assert [item["name"] for item in all_pages["data"]] == names
    assert all_pages["pagination"]["total_results"] == 3
    assert all_pages["pagination"]["total_pages"] == 1
    assert "More pages" not in result.stderr


def test_workspaces_list_table_and_stream(runner, assert_exit_code) -> None:
    name = _name()
    assert_exit_code(runner.invoke(app, ["workspaces", "create", name, "--description", "tabular"]), 0)

    result = runner.invoke(app, ["workspaces", "list", "--filter", f'name:"{name}"', "-f", "table"])
    assert_exit_code(result, 0)
    assert name in result.stdout
    assert "tabular" in result.stdout

    result = runner.invoke(app, ["workspaces", "list", "--filter", f'name:"{name}"', "-f", "json", "--stream"])
    assert_exit_code(result, 0)
    records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert [record["name"] for record in records] == [name]


def test_workspace_members_lifecycle(runner, random_workspace: str, assert_exit_code) -> None:
    result = runner.invoke(app, ["workspaces", "members", "list", "--workspace", random_workspace])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["data"] == []

    result = runner.invoke(
        app,
        [
            "workspaces",
            "members",
            "create",
            "--workspace",
            random_workspace,
            "--principal",
            "user@example.com",
            "--roles",
            "Editor",
            "--roles",
            "Viewer",
        ],
    )
    assert_exit_code(result, 0)
    member = json.loads(result.stdout)
    assert member["principal"] == "user@example.com"
    assert sorted(member["roles"]) == ["Editor", "Viewer"]
    assert member["granted_at"]

    result = runner.invoke(app, ["workspaces", "members", "list", "--workspace", random_workspace])
    assert_exit_code(result, 0)
    members = json.loads(result.stdout)["data"]
    assert [m["principal"] for m in members] == ["user@example.com"]
    assert sorted(members[0]["roles"]) == ["Editor", "Viewer"]

    result = runner.invoke(
        app,
        ["workspaces", "members", "update", "user@example.com", "--workspace", random_workspace, "--roles", "Viewer"],
    )
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["roles"] == ["Viewer"]
    result = runner.invoke(app, ["workspaces", "members", "list", "--workspace", random_workspace])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["data"][0]["roles"] == ["Viewer"]

    result = runner.invoke(
        app, ["workspaces", "members", "delete", "user@example.com", "--workspace", random_workspace]
    )
    assert_exit_code(result, 0)
    assert "Deleted successfully" in result.stdout

    result = runner.invoke(app, ["workspaces", "members", "list", "--workspace", random_workspace])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["data"] == []


def test_workspace_members_create_from_stdin_with_default_roles(
    runner, random_workspace: str, assert_exit_code
) -> None:
    result = runner.invoke(
        app,
        ["workspaces", "members", "create", "--input-file", "-"],
        input=json.dumps({"workspace": random_workspace, "principal": "piped@example.com"}),
    )
    assert_exit_code(result, 0)
    member = json.loads(result.stdout)
    assert member["principal"] == "piped@example.com"
    assert member["roles"] == ["Editor"]

    result = runner.invoke(app, ["workspaces", "members", "list", "--workspace", random_workspace, "-f", "table"])
    assert_exit_code(result, 0)
    assert "piped@example.com" in result.stdout
    assert "Editor" in result.stdout


def test_workspace_members_unknown_workspace_is_not_found(runner, assert_exit_code) -> None:
    result = runner.invoke(app, ["workspaces", "members", "list", "--workspace", "does-not-exist"])
    assert_exit_code(result, 3)
    assert "Not found" in result.stderr

    result = runner.invoke(
        app, ["workspaces", "members", "create", "--workspace", "does-not-exist", "--principal", "x@example.com"]
    )
    assert_exit_code(result, 3)
    assert "Not found" in result.stderr


def test_workspace_members_missing_workspace_is_usage_error(runner, assert_exit_code) -> None:
    # The injected client has no default workspace and none was passed.
    result = runner.invoke(app, ["workspaces", "members", "list"])
    assert_exit_code(result, 2)
    assert "Missing workspace" in result.stderr


def test_workspaces_code_output_does_not_create(runner, assert_exit_code) -> None:
    name = _name()
    result = runner.invoke(app, ["workspaces", "create", name, "--description", "code-only", "-f", "code"])
    assert_exit_code(result, 0)
    assert "from nemo_platform_plugin.workspaces.client import WorkspacesClient" in result.stdout
    assert "client.create_workspace(" in result.stdout
    assert "NeMoPlatform" not in result.stdout

    result = runner.invoke(app, ["workspaces", "get", name])
    assert_exit_code(result, 3)
