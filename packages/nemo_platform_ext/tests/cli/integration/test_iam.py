# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for ``nemo iam role-bindings`` against the in-process Auth service.

The test app runs with authorization disabled, so the Platform Admin gate is a
no-op and ``wait_role_propagation`` returns immediately.
"""

import json
import uuid

from nemo_platform_ext.cli.app import app


def _principal() -> str:
    return f"user-{uuid.uuid4().hex[:8]}@example.com"


def test_role_bindings_lifecycle(runner, random_workspace: str, assert_exit_code) -> None:
    principal = _principal()

    result = runner.invoke(
        app,
        [
            "iam",
            "role-bindings",
            "create",
            "--principal",
            principal,
            "--role",
            "Viewer",
            "--workspace",
            random_workspace,
        ],
    )
    assert_exit_code(result, 0)
    created = json.loads(result.stdout)
    assert created["principal"] == principal
    assert created["role"] == "Viewer"
    assert created["workspace"] == random_workspace
    assert created["name"].startswith("rb-")
    assert created["revoked_at"] is None
    name = created["name"]

    result = runner.invoke(app, ["iam", "role-bindings", "get", name])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["id"] == created["id"]

    result = runner.invoke(app, ["iam", "role-bindings", "list", "--filter.principal", principal])
    assert_exit_code(result, 0)
    listed = json.loads(result.stdout)
    assert [item["name"] for item in listed["data"]] == [name]
    assert listed["pagination"]["total_results"] == 1

    result = runner.invoke(app, ["iam", "role-bindings", "delete", name])
    assert_exit_code(result, 0)
    assert "Deleted successfully" in result.stdout

    # Revocation is a soft delete: the binding stays readable with revoked_at set.
    result = runner.invoke(app, ["iam", "role-bindings", "get", name])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["revoked_at"] is not None

    result = runner.invoke(app, ["iam", "role-bindings", "delete", name])
    assert_exit_code(result, 3)
    assert "Conflict" in result.stderr


def test_role_bindings_create_conflict_on_duplicate(runner, random_workspace: str, assert_exit_code) -> None:
    principal = _principal()
    args = [
        "iam",
        "role-bindings",
        "create",
        "--principal",
        principal,
        "--role",
        "Editor",
        "--workspace",
        random_workspace,
    ]
    assert_exit_code(runner.invoke(app, args), 0)

    result = runner.invoke(app, args)
    assert_exit_code(result, 3)
    assert "Conflict" in result.stderr


def test_role_bindings_platform_level_from_stdin(runner, assert_exit_code) -> None:
    principal = _principal()
    result = runner.invoke(
        app,
        ["iam", "role-bindings", "create", "--input-file", "-"],
        input=json.dumps({"principal": principal, "role": "Admin", "wait_role_propagation": False}),
    )
    assert_exit_code(result, 0)
    created = json.loads(result.stdout)
    assert created["principal"] == principal
    assert created["role"] == "Admin"
    # Platform-level bindings are stored under the "system" workspace.
    assert created["workspace"] == "system"


def test_role_bindings_create_requires_principal_and_role(runner, assert_exit_code) -> None:
    result = runner.invoke(app, ["iam", "role-bindings", "create", "--role", "Viewer"])
    assert_exit_code(result, 2)
    assert "principal" in result.stderr


def test_role_bindings_get_unknown_is_not_found(runner, assert_exit_code) -> None:
    result = runner.invoke(app, ["iam", "role-bindings", "get", "rb-does-not-exist"])
    assert_exit_code(result, 3)
    assert "Not found" in result.stderr

    result = runner.invoke(app, ["iam", "role-bindings", "delete", "rb-does-not-exist"])
    assert_exit_code(result, 3)
    assert "Not found" in result.stderr


def test_role_bindings_list_filters_and_pages(runner, random_workspace: str, assert_exit_code) -> None:
    principals = sorted(_principal() for _ in range(3))
    for principal in principals:
        assert_exit_code(
            runner.invoke(
                app,
                [
                    "iam",
                    "role-bindings",
                    "create",
                    "--principal",
                    principal,
                    "--role",
                    "Viewer",
                    "--workspace",
                    random_workspace,
                ],
            ),
            0,
        )
    assert_exit_code(
        runner.invoke(
            app,
            [
                "iam",
                "role-bindings",
                "create",
                "--principal",
                principals[0],
                "--role",
                "Editor",
                "--workspace",
                random_workspace,
            ],
        ),
        0,
    )

    result = runner.invoke(
        app, ["iam", "role-bindings", "list", "--filter.workspace", random_workspace, "--filter.role", "Viewer"]
    )
    assert_exit_code(result, 0)
    listed = json.loads(result.stdout)
    assert sorted(item["principal"] for item in listed["data"]) == principals
    assert listed["pagination"]["total_results"] == 3

    result = runner.invoke(
        app,
        ["iam", "role-bindings", "list", "--filter", f'workspace:"{random_workspace}" AND role:"Editor"'],
    )
    assert_exit_code(result, 0)
    assert [item["principal"] for item in json.loads(result.stdout)["data"]] == [principals[0]]

    result = runner.invoke(
        app,
        [
            "iam",
            "role-bindings",
            "list",
            "--filter",
            json.dumps({"workspace": random_workspace, "principal": {"$like": f"%{principals[1][5:13]}%"}}),
        ],
    )
    assert_exit_code(result, 0)
    assert [item["principal"] for item in json.loads(result.stdout)["data"]] == [principals[1]]

    result = runner.invoke(
        app, ["iam", "role-bindings", "list", "--filter.workspace", random_workspace, "--page-size", "2"]
    )
    assert_exit_code(result, 0)
    first_page = json.loads(result.stdout)
    assert len(first_page["data"]) == 2
    assert first_page["pagination"]["total_pages"] == 2
    assert first_page["pagination"]["total_results"] == 4
    assert "More pages" in result.stderr

    result = runner.invoke(
        app,
        ["iam", "role-bindings", "list", "--filter.workspace", random_workspace, "--page-size", "2", "--all-pages"],
    )
    assert_exit_code(result, 0)
    all_pages = json.loads(result.stdout)
    assert len(all_pages["data"]) == 4
    assert all_pages["pagination"]["total_results"] == 4
    assert all_pages["pagination"]["total_pages"] == 1
    assert "More pages" not in result.stderr


def test_role_bindings_list_table_and_stream(runner, random_workspace: str, assert_exit_code) -> None:
    principal = _principal()
    assert_exit_code(
        runner.invoke(
            app,
            [
                "iam",
                "role-bindings",
                "create",
                "--principal",
                principal,
                "--role",
                "Viewer",
                "--workspace",
                random_workspace,
            ],
        ),
        0,
    )

    result = runner.invoke(app, ["iam", "role-bindings", "list", "--filter.principal", principal, "-f", "table"])
    assert_exit_code(result, 0)
    assert "rb-" in result.stdout
    assert random_workspace in result.stdout

    result = runner.invoke(
        app, ["iam", "role-bindings", "list", "--filter.principal", principal, "-f", "json", "--stream"]
    )
    assert_exit_code(result, 0)
    records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert [record["principal"] for record in records] == [principal]


def test_role_bindings_code_output_does_not_create(runner, random_workspace: str, assert_exit_code) -> None:
    principal = _principal()
    result = runner.invoke(
        app,
        [
            "iam",
            "role-bindings",
            "create",
            "--principal",
            principal,
            "--role",
            "Viewer",
            "--workspace",
            random_workspace,
            "-f",
            "code",
        ],
    )
    assert_exit_code(result, 0)
    assert "from nemo_platform_plugin.iam.client import IAMClient" in result.stdout
    assert "client.create_role_binding(" in result.stdout
    assert "NeMoPlatform" not in result.stdout

    result = runner.invoke(app, ["iam", "role-bindings", "list", "--filter.principal", principal])
    assert_exit_code(result, 0)
    assert json.loads(result.stdout)["data"] == []
