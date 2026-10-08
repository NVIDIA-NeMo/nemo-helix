# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every installed command must be describable offline.

``nemo describe`` builds the command tree the same way running a command does,
so this also guards the contract that loading a command group (including plugin
groups and generated job/function commands) has no network side effects. Each
top-level command is walked in a fresh interpreter so import-time side effects
are not hidden by modules another test already imported.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from nemo_helix_ext.cli.app import _build_top_level_lazy_entries

# Loading these groups opens a network connection today. Each is strict, so the
# test fails once the group loads offline and the entry can be removed.
_KNOWN_NETWORK_ON_LOAD = {
    # nemo_insights_plugin.cli imports nemo_helix_plugin.nooa_model_client, which
    # imports litellm at module level; litellm fetches its model cost map from GitHub.
    "insights": "litellm fetches its remote model cost map on import",
}

_WALK_SCRIPT = """
import json
import socket
import sys

import click
import typer

connections = []
_connect = socket.socket.connect


def _refuse_connection(sock, address):
    # Local IPC such as the Docker socket is not network access.
    if sock.family == socket.AF_UNIX:
        return _connect(sock, address)
    connections.append(repr(address))
    raise OSError("network access is not allowed while describing commands")


socket.socket.connect = _refuse_connection

from nemo_helix_ext.cli.app import app
from nemo_helix_ext.cli.core.command_description import describe_command, resolve_command_path

root = typer.main.get_command(app)
problems = []


def walk(path):
    resolved = resolve_command_path(root, path)
    description = describe_command(resolved)
    json.dumps(description)
    if description["command"] != " ".join(["nemo", *path]):
        problems.append(f"{path} described as {description['command']!r}")
    command = resolved.command
    if isinstance(command, click.Group):
        for name in command.list_commands(resolved.context):
            if command.get_command(resolved.context, name) is None:
                problems.append(f"{[*path, name]} is listed but cannot be loaded")
                continue
            walk([*path, name])


walk([sys.argv[1]])
print(json.dumps({"connections": connections, "problems": problems}))
"""


def _top_level_params() -> list[object]:
    params: list[object] = []
    for name in sorted(entry.name for entry in _build_top_level_lazy_entries()):
        reason = _KNOWN_NETWORK_ON_LOAD.get(name)
        marks = [pytest.mark.xfail(reason=reason, strict=True)] if reason else []
        params.append(pytest.param(name, marks=marks, id=name))
    return params


@pytest.mark.parametrize("top_level_name", _top_level_params())
def test_every_command_describes_offline(top_level_name: str):
    result = subprocess.run(
        [sys.executable, "-c", _WALK_SCRIPT, top_level_name],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]

    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["problems"] == []
    assert report["connections"] == []
