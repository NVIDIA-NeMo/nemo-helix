# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from nhx.rl.app.constants import DEFAULT_TRAINING_RESULT_FILE_NAME
from nhx.rl.app.jobs.training.schemas import TrainingResult
from nhx.rl.tasks.training.protocol import TrainingPaused
from nhx.rl.tasks.training.runner import TrainingRunner


def _runner(tmp_path: Path, synced: list[str], completed: list[str]) -> TrainingRunner:
    runner = TrainingRunner.__new__(TrainingRunner)
    runner._config = SimpleNamespace(seed=1)
    runner._progress = SimpleNamespace(
        training_duration_seconds=3.0,
        report_running=lambda *args, **kwargs: None,
        report_completed=completed.append,
        report_error=lambda *args, **kwargs: None,
    )
    runner._dist_ctx = SimpleNamespace(is_coordinator=True, sync_point=synced.append, signal_failure=lambda: None)
    runner._workspace_path = tmp_path
    runner._output_path = tmp_path / "output"
    runner._backend = MagicMock()
    runner._preprocessing_phase = lambda: None  # type: ignore[method-assign]
    runner._compile_config_phase = lambda: MagicMock()  # type: ignore[method-assign]
    return runner


def test_a_paused_run_releases_workers_and_skips_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nhx.rl.tasks.training.runner.get_gpu_info", lambda: None)
    synced: list[str] = []
    completed: list[str] = []
    runner = _runner(tmp_path, synced, completed)
    runner._training_phase = MagicMock(side_effect=TrainingPaused())  # type: ignore[method-assign]

    result = runner.run()

    assert result.paused and not result.success
    # Workers wait at this barrier for the coordinator, so a pause must still reach it.
    assert synced == ["training_complete"]
    assert completed == []
    runner._backend.find_best_checkpoint.assert_not_called()
    runner._backend.process_checkpoint.assert_not_called()
    written = json.loads((tmp_path / DEFAULT_TRAINING_RESULT_FILE_NAME).read_text())
    assert written["paused"] is True
    assert written["training_duration_seconds"] == 3.0


def test_a_paused_run_exits_with_the_pause_code(monkeypatch: pytest.MonkeyPatch) -> None:
    from nhx.rl.tasks.training import __main__ as entrypoint

    class _Runner:
        def __enter__(self) -> "_Runner":
            return self

        def __exit__(self, *exc: object) -> None: ...

        def run(self) -> TrainingResult:
            return TrainingResult(success=False, paused=True)

    monkeypatch.setattr(entrypoint, "TrainingRunner", _Runner)

    assert entrypoint.main() == 75
