# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for ``nemo projects`` against the in-process Entities service."""

import json
import uuid

from nemo_platform_ext.cli.app import app


def _name() -> str:
    return f"proj-{uuid.uuid4().hex[:8]}"


def test_projects_lifecycle(runner, random_workspace: str, assert_exit_code) -> None:
    name = _name()

    result = runner.invoke(app, ["projects", "create", name, "--description", "demo", "--workspace", random_workspace])
    assert_exit_code(result, 0)
    created = json.loads(result.stdout)
    assert created["name"] == name
    assert created["workspace"] == random_workspace
    assert created["description"] == "demo"
    assert created["id"].startswith("project-")

    result = runner.invoke(app, ["projects", "get", name, "--workspace", random_workspace])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["id"] == created["id"]

    result = runner.invoke(app, ["projects", "list", "--workspace", random_workspace])
    assert_exit_code(result, 0)
    listed = json.loads(result.stdout)
    assert [item["name"] for item in listed["data"]] == [name]
    assert listed["pagination"]["total_results"] == 1

    result = runner.invoke(
        app, ["projects", "update", name, "--description", "changed", "--workspace", random_workspace]
    )
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["description"] == "changed"
    result = runner.invoke(app, ["projects", "get", name, "--workspace", random_workspace])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["description"] == "changed"

    result = runner.invoke(app, ["projects", "delete", name, "--workspace", random_workspace])
    assert_exit_code(result, 0)
    assert "Deleted successfully" in result.stdout

    result = runner.invoke(app, ["projects", "get", name, "--workspace", random_workspace])
    assert_exit_code(result, 3)
    assert "Not found" in result.stderr

    result = runner.invoke(app, ["projects", "list", "--workspace", random_workspace])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["data"] == []


def test_projects_create_conflict_and_exist_ok(runner, random_workspace: str, assert_exit_code) -> None:
    name = _name()
    assert_exit_code(
        runner.invoke(app, ["projects", "create", name, "--description", "first", "--workspace", random_workspace]), 0
    )

    result = runner.invoke(app, ["projects", "create", name, "--workspace", random_workspace])
    assert_exit_code(result, 3)
    assert "Conflict" in result.stderr

    result = runner.invoke(app, ["projects", "create", name, "--workspace", random_workspace, "--exist-ok"])
    assert_exit_code(result, 0)
    existing = json.loads(result.stdout)
    assert existing["name"] == name
    assert existing["description"] == "first"


def test_projects_create_rejects_invalid_name(runner, random_workspace: str, assert_exit_code) -> None:
    result = runner.invoke(app, ["projects", "create", "Invalid_Name", "--workspace", random_workspace])
    assert_exit_code(result, 3)
    assert "(422)" in result.stderr


def test_projects_create_from_stdin_with_workspace_in_payload(runner, random_workspace: str, assert_exit_code) -> None:
    name = _name()
    result = runner.invoke(
        app,
        ["projects", "create", "--input-file", "-"],
        input=json.dumps({"name": name, "workspace": random_workspace, "description": "piped"}),
    )
    assert_exit_code(result, 0)
    created = json.loads(result.stdout)
    assert created["name"] == name
    assert created["workspace"] == random_workspace
    assert created["description"] == "piped"


def test_projects_missing_workspace_is_usage_error(runner, assert_exit_code) -> None:
    # The injected client has no default workspace and none was passed.
    result = runner.invoke(app, ["projects", "list"])
    assert_exit_code(result, 2)
    assert "Missing workspace" in result.stderr


def test_projects_list_filter_sort_and_pages(runner, random_workspace: str, assert_exit_code) -> None:
    prefix = f"pg{uuid.uuid4().hex[:6]}"
    names = sorted(f"{prefix}-{i}" for i in range(3))
    for name in names:
        assert_exit_code(runner.invoke(app, ["projects", "create", name, "--workspace", random_workspace]), 0)
    assert_exit_code(runner.invoke(app, ["projects", "create", "other-project", "--workspace", random_workspace]), 0)
    name_filter = f'name~"{prefix}"'

    result = runner.invoke(
        app,
        [
            "projects",
            "list",
            "--workspace",
            random_workspace,
            "--filter",
            name_filter,
            "--page-size",
            "2",
            "--sort",
            "name",
        ],
    )
    assert_exit_code(result, 0)
    first_page = json.loads(result.stdout)
    assert [item["name"] for item in first_page["data"]] == names[:2]
    assert first_page["pagination"]["total_pages"] == 2
    assert first_page["pagination"]["total_results"] == 3
    assert "More pages" in result.stderr

    result = runner.invoke(
        app,
        [
            "projects",
            "list",
            "--workspace",
            random_workspace,
            "--filter",
            name_filter,
            "--page-size",
            "2",
            "--sort",
            "name",
            "--page",
            "2",
        ],
    )
    assert_exit_code(result, 0)
    assert [item["name"] for item in json.loads(result.stdout)["data"]] == names[2:]

    result = runner.invoke(
        app,
        [
            "projects",
            "list",
            "--workspace",
            random_workspace,
            "--filter",
            name_filter,
            "--page-size",
            "2",
            "--sort",
            "-name",
            "--all-pages",
        ],
    )
    assert_exit_code(result, 0)
    all_pages = json.loads(result.stdout)
    assert [item["name"] for item in all_pages["data"]] == list(reversed(names))
    assert all_pages["pagination"]["total_results"] == 3
    assert all_pages["pagination"]["total_pages"] == 1
    assert "More pages" not in result.stderr


def test_projects_list_table_and_stream(runner, random_workspace: str, assert_exit_code) -> None:
    name = _name()
    assert_exit_code(
        runner.invoke(app, ["projects", "create", name, "--description", "tabular", "--workspace", random_workspace]),
        0,
    )

    result = runner.invoke(app, ["projects", "list", "--workspace", random_workspace, "-f", "table"])
    assert_exit_code(result, 0)
    assert name in result.stdout
    assert "tabular" in result.stdout

    result = runner.invoke(app, ["projects", "list", "--workspace", random_workspace, "-f", "json", "--stream"])
    assert_exit_code(result, 0)
    records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert [record["name"] for record in records] == [name]


def test_projects_code_output_does_not_create(runner, random_workspace: str, assert_exit_code) -> None:
    name = _name()
    result = runner.invoke(
        app,
        ["projects", "create", name, "--description", "code-only", "--workspace", random_workspace, "-f", "code"],
    )
    assert_exit_code(result, 0)
    assert "from nemo_platform_plugin.projects.client import ProjectsClient" in result.stdout
    assert "client.create_project(" in result.stdout
    assert "NeMoPlatform" not in result.stdout

    result = runner.invoke(app, ["projects", "get", name, "--workspace", random_workspace])
    assert_exit_code(result, 3)
