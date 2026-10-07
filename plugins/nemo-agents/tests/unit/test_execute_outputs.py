# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
from nemo_agents_plugin.jobs.execute import ExecuteAgentJobConfig
from nemo_agents_plugin.tasks.execute.outputs import AgentOutputFile, select_output_files


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "10"])
def test_requires_a_positive_integer_size_limit(limit: object) -> None:
    with pytest.raises(ValueError):
        AgentOutputFile(path="result.json", max_bytes=limit)


def test_rejects_duplicate_paths() -> None:
    output = AgentOutputFile(path="result.json", max_bytes=10)
    with pytest.raises(ValueError, match="unique"):
        ExecuteAgentJobConfig(agent="test", input="test", output_files=[output, output])


@pytest.mark.parametrize("path", ["", "/etc/passwd", "../outside", "a/../b", "a//b", "./result.json", "a\\b", "a\x00b"])
def test_rejects_unsafe_output_paths(path: str) -> None:
    with pytest.raises(ValueError, match="relative path"):
        AgentOutputFile(path=path, max_bytes=10)


@pytest.mark.parametrize("kind", ["symlink", "parent-symlink", "directory", "fifo", "oversized"])
def test_rejects_non_regular_and_oversized_outputs(tmp_path: Path, kind: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = workspace / "result.json"
    if kind == "symlink":
        output.symlink_to(tmp_path / "outside")
    elif kind == "parent-symlink":
        (workspace / "link").symlink_to(tmp_path, target_is_directory=True)
        (tmp_path / "outside").write_text("private")
        output = workspace / "link/outside"
    elif kind == "directory":
        output.mkdir()
    elif kind == "fifo":
        os.mkfifo(output)
    else:
        output.write_bytes(b"x" * 11)
    destination = tmp_path / "export"
    destination.mkdir()
    with pytest.raises(ValueError):
        select_output_files(
            workspace,
            destination,
            [AgentOutputFile(path=str(output.relative_to(workspace)), max_bytes=10)],
            require_files=True,
        )
    assert list(destination.iterdir()) == []


def test_copies_only_declared_files_and_enforces_required_outputs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "checks.log").write_text("checked")
    scratch = workspace / "tmp"
    scratch.mkdir()
    for number in range(2000):
        (scratch / str(number)).write_text("scratch")
    destination = tmp_path / "export"
    destination.mkdir()
    files = [
        AgentOutputFile(path="result.json", required=True, max_bytes=10),
        AgentOutputFile(path="checks.log", max_bytes=10),
    ]
    with pytest.raises(ValueError, match="Required output file is missing"):
        select_output_files(workspace, destination, files, require_files=True)
    select_output_files(workspace, destination, files, require_files=False)
    assert [path.name for path in destination.iterdir()] == ["checks.log"]
    assert len(list(scratch.iterdir())) == 2000
