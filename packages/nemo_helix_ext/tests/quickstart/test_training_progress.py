# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from nemo_helix_ext.quickstart.progress import describe_training_progress, status_text, wait_for_training


def _status(state: str, steps: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(status=state, steps=steps or [])


def _step(name: str, state: str, details: dict) -> SimpleNamespace:
    task = SimpleNamespace(status_details=details)
    return SimpleNamespace(name=name, status=state, tasks=[task])


def test_status_text_uses_enum_value() -> None:
    assert status_text(SimpleNamespace(value="active")) == "active"
    assert status_text("completed") == "completed"


def test_describe_training_progress_includes_percentage_and_loss() -> None:
    status = _status(
        "active",
        [
            _step(
                "training",
                "active",
                {
                    "step": 8,
                    "max_steps": 94,
                    "epoch": 1,
                    "num_epochs": 2,
                    "phase": "training",
                    "train_loss": 2.5,
                    "metrics": {"train_loss": [{"step": 4, "value": 3.0}, {"step": 8, "value": 2.5}]},
                },
            )
        ],
    )

    text = describe_training_progress(status)

    assert "step 8/94 — 8.5%" in text
    assert "epoch 1/2" in text
    assert "train_loss: 2.5000" in text


def test_wait_for_training_stops_when_the_job_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    statuses = [
        _status("active", [_step("training", "active", {"step": 1, "max_steps": 2, "train_loss": 1.0})]),
        _status("completed", [_step("training", "completed", {"step": 2, "max_steps": 2, "train_loss": 0.2})]),
    ]

    class _Jobs:
        def get_status(self, name: str, workspace: str) -> SimpleNamespace:
            assert name == "job-a"
            assert workspace == "default"
            return statuses.pop(0)

    monkeypatch.setattr("nemo_helix_ext.quickstart.progress.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("nemo_helix_ext.quickstart.progress._clear_output", lambda: None)
    with patch("nemo_helix_ext.quickstart.progress._plot_loss"):
        finished = wait_for_training(SimpleNamespace(jobs=_Jobs()), "job-a", timeout_seconds=5, poll_interval=0)

    assert finished.status == "completed"
    assert statuses == []


def test_wait_for_training_raises_when_the_job_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Jobs:
        def get_status(self, name: str, workspace: str) -> SimpleNamespace:
            return _status("failed")

    monkeypatch.setattr("nemo_helix_ext.quickstart.progress.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("nemo_helix_ext.quickstart.progress._clear_output", lambda: None)
    with patch("nemo_helix_ext.quickstart.progress._plot_loss"), pytest.raises(RuntimeError, match="failed"):
        wait_for_training(SimpleNamespace(jobs=_Jobs()), "job-a", poll_interval=0)
