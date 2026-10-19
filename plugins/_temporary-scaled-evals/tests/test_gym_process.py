# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("scaled_evals")
import scaled_evals.dispatch.gym.process as gym_process
from scaled_evals.dispatch.gym.process import (
    make_gym_process_status_reader,
    make_gym_process_terminator,
)
from scaled_evals.dispatch.runtime_backend import LaunchHandle


def _handle(eval_id: str, work: Path) -> LaunchHandle:
    return LaunchHandle(
        backend="gym_sandbox_opensandbox",
        external_id=eval_id,
        raw={"exit_code_path": str(work / eval_id / "exit-code")},
    )


def test_process_status_converts_successful_rollouts(tmp_path: Path) -> None:
    eval_id = "ev_gym"
    eval_work = tmp_path / eval_id
    eval_work.mkdir()
    (eval_work / "exit-code").write_text("0\n", encoding="utf-8")
    (eval_work / "rollouts.jsonl").write_text('{"reward": 1, "task_id": "smoke"}\n', encoding="utf-8")

    status = make_gym_process_status_reader(work_dir=str(tmp_path))(_handle(eval_id, tmp_path))

    assert status.phase == "succeeded"
    assert status.raw["n_total_trials"] == 1


def test_process_status_reports_nonzero_exit_with_redacted_size_tail(tmp_path: Path) -> None:
    eval_id = "ev_gym"
    eval_work = tmp_path / eval_id
    eval_work.mkdir()
    (eval_work / "exit-code").write_text("7\n", encoding="utf-8")
    (eval_work / "gym.log").write_text("runner failed\n", encoding="utf-8")

    status = make_gym_process_status_reader(work_dir=str(tmp_path))(_handle(eval_id, tmp_path))

    assert status.phase == "failed"
    assert "exited 7" in str(status.detail)
    assert "runner failed" in str(status.detail)


def test_process_status_rejects_resume_in_replacement_pod(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    eval_id = "ev_gym"
    (tmp_path / eval_id).mkdir()
    monkeypatch.setenv("HOSTNAME", "replacement-pod")
    handle = LaunchHandle(
        backend="gym_sandbox_opensandbox",
        external_id=eval_id,
        raw={
            "process_owner_pod": "original-pod",
            "process_pid": 42,
            "process_start_identity": "start-1",
        },
    )

    status = make_gym_process_status_reader(work_dir=str(tmp_path))(handle)

    assert status.phase == "failed"
    assert "replacement pod" in str(status.detail)


def test_process_terminator_refuses_reused_pid(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("HOSTNAME", "owner-pod")
    monkeypatch.setattr(gym_process, "_process_start_identity", lambda pid: "new-start")

    def unexpected_killpg(pid: int, signal: int) -> None:
        raise AssertionError(f"must not terminate reused pid {pid} with signal {signal}")

    monkeypatch.setattr(os, "killpg", unexpected_killpg)
    handle = LaunchHandle(
        backend="gym_sandbox_opensandbox",
        external_id="ev_gym",
        raw={
            "process_owner_pod": "owner-pod",
            "process_pid": 42,
            "process_start_identity": "old-start",
        },
    )

    make_gym_process_terminator()(handle)
