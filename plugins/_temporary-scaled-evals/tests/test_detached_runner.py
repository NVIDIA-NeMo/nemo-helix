# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

try:
    from scaled_evals.dispatch import detached_runner
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


def test_detached_runner_persists_exit_before_removing_identity(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    pid_path = tmp_path / "runner.pid"
    exit_path = tmp_path / "runner.exit.json"
    token = "unique-token"
    pid_path.write_text(json.dumps({"pid": os.getpid(), "token": token}))
    monkeypatch.setattr(
        detached_runner.subprocess,
        "run",
        lambda argv, check: MagicMock(returncode=7),
    )

    assert detached_runner.run(pid_path, exit_path, token, ["harbor", "run"]) == 7

    terminal = json.loads(exit_path.read_text())
    assert terminal["token"] == token
    assert terminal["exit_code"] == 7
    assert terminal["finished_at"]
    assert not pid_path.exists()


def test_detached_runner_does_not_remove_replaced_identity(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    pid_path = tmp_path / "runner.pid"
    exit_path = tmp_path / "runner.exit.json"
    token = "first-token"
    pid_path.write_text(json.dumps({"pid": os.getpid(), "token": token}))

    def replace_identity(argv, check):  # noqa: ANN001, ARG001
        pid_path.write_text(json.dumps({"pid": 999, "token": "replacement-token"}))
        return MagicMock(returncode=0)

    monkeypatch.setattr(detached_runner.subprocess, "run", replace_identity)

    assert detached_runner.run(pid_path, exit_path, token, ["harbor", "run"]) == 0
    assert json.loads(pid_path.read_text())["token"] == "replacement-token"
