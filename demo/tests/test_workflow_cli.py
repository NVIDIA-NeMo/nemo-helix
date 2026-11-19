# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command parsing and watch delegation without live services."""

import subprocess
import sys

import pytest

from demo import workflows
from demo.workflows import parser, watch_run


def test_choose_workflow_and_explicit_run_name() -> None:
    args = parser().parse_args(["create", "montage", "my-images", "--secret", "existing-key", "--watch"])
    assert args.workflow == "montage" and args.name == "my-images" and args.watch


def test_manual_trigger_accepts_multiple_nodes() -> None:
    args = parser().parse_args(["trigger", "my-dag", "--step", "worker-03", "--step", "worker-04"])
    assert args.step == ["worker-03", "worker-04"]


def test_upload_and_invalidation_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        parser().parse_args(["trigger", "my-dag", "--input", "new.txt", "--step", "worker-03"])


@pytest.mark.parametrize("no_tui", [False, True])
def test_watch_delegates_to_same_python_and_preserves_exit_status(
    monkeypatch: pytest.MonkeyPatch, no_tui: bool
) -> None:
    seen: list[list[str]] = []

    def run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert not check
        seen.append(command)
        return subprocess.CompletedProcess(command, 130)

    monkeypatch.setattr(workflows, "run", run)
    assert watch_run("my-images", workspace="another-workspace", no_tui=no_tui) == 130
    assert seen == [
        [
            sys.executable,
            "-m",
            "nemo_helix_ext.cli.app",
            "jobs",
            "watch",
            "my-images",
            "--workspace",
            "another-workspace",
            "--no-tui" if no_tui else "--tui",
        ]
    ]
