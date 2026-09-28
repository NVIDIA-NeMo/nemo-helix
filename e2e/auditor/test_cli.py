# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI smoke tests for the auditor plugin.

Exercises ``nemo auditor configs`` and ``nemo auditor targets`` subcommands
against a live platform via ``NHX_BASE_URL``, using ``run_nemo_local`` so the
CLI picks up platform credentials from env without reading the developer's own
config file.
"""

import json

from nemo_helix_plugin.client.client import NemoClient
from nhx.testing import assert_exit_0, run_nemo_local

from e2e.auditor.utils import minimal_audit_config, minimal_audit_target, unique_name


def test_cli_config_create_list_delete(client: NemoClient, workspace: str) -> None:
    name = unique_name("cli-cfg")
    base_url = client.base_url

    result = run_nemo_local(
        "auditor",
        "configs",
        "create",
        name,
        "--data",
        json.dumps(minimal_audit_config(description="cli smoke")),
        "--workspace",
        workspace,
        base_url=base_url,
    )
    assert_exit_0(result, "CLI create config")
    assert name in result.stdout

    result = run_nemo_local("auditor", "configs", "list", "--workspace", workspace, base_url=base_url)
    assert_exit_0(result, "CLI list configs")
    assert name in result.stdout

    result = run_nemo_local("auditor", "configs", "delete", name, "--workspace", workspace, base_url=base_url)
    assert_exit_0(result, "CLI delete config")
    assert name in result.stdout

    result = run_nemo_local("auditor", "configs", "list", "--workspace", workspace, base_url=base_url)
    assert_exit_0(result, "CLI list after delete")
    listed = json.loads(result.stdout)
    assert all(item["name"] != name for item in listed["data"])


def test_cli_target_create_list_delete(client: NemoClient, workspace: str) -> None:
    name = unique_name("cli-tgt")
    base_url = client.base_url

    result = run_nemo_local(
        "auditor",
        "targets",
        "create",
        name,
        "--data",
        json.dumps(minimal_audit_target(description="cli target smoke")),
        "--workspace",
        workspace,
        base_url=base_url,
    )
    assert_exit_0(result, "CLI create target")
    assert name in result.stdout

    result = run_nemo_local("auditor", "targets", "list", "--workspace", workspace, base_url=base_url)
    assert_exit_0(result, "CLI list targets")
    assert name in result.stdout

    result = run_nemo_local("auditor", "targets", "delete", name, "--workspace", workspace, base_url=base_url)
    assert_exit_0(result, "CLI delete target")


def test_cli_config_update(client: NemoClient, workspace: str) -> None:
    name = unique_name("cli-upd")
    base_url = client.base_url

    result = run_nemo_local(
        "auditor",
        "configs",
        "create",
        name,
        "--data",
        json.dumps(minimal_audit_config(description="before update")),
        "--workspace",
        workspace,
        base_url=base_url,
    )
    assert_exit_0(result, "CLI create config for update")

    updated_body = minimal_audit_config(description="after update")
    result = run_nemo_local(
        "auditor",
        "configs",
        "update",
        name,
        "--data",
        json.dumps(updated_body),
        "--workspace",
        workspace,
        base_url=base_url,
    )
    assert_exit_0(result, "CLI update config")

    result = run_nemo_local("auditor", "configs", "get", name, "--workspace", workspace, base_url=base_url)
    assert_exit_0(result, "CLI get updated config")
    payload = json.loads(result.stdout)
    assert payload["description"] == "after update"
