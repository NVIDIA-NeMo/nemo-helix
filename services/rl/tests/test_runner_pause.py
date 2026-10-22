# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from nhx.customization_common.training.errors.classifier import classify_output
from nhx.rl.app.constants import DEFAULT_TRAINING_RESULT_FILE_NAME
from nhx.rl.app.jobs.training.schemas import TrainingResult, TrainingStepConfig
from nhx.rl.tasks.training.distributed import DistributedContext
from nhx.rl.tasks.training.progress import JobsServiceProgressReporter
from nhx.rl.tasks.training.protocol import TrainingPaused
from nhx.rl.tasks.training.runner import TrainingRunner


def _runner(tmp_path: Path, synced: list[str], completed: list[str]) -> TrainingRunner:
    runner = TrainingRunner.__new__(TrainingRunner)
    runner._config = cast(TrainingStepConfig, SimpleNamespace(seed=1))
    runner._progress = cast(
        JobsServiceProgressReporter,
        SimpleNamespace(
            training_duration_seconds=3.0,
            report_running=lambda *args, **kwargs: None,
            report_completed=completed.append,
            report_error=lambda *args, **kwargs: None,
        ),
    )
    runner._dist_ctx = cast(
        DistributedContext,
        SimpleNamespace(is_coordinator=True, sync_point=synced.append, signal_failure=lambda: None),
    )
    runner._workspace_path = tmp_path
    runner._output_path = tmp_path / "output"
    runner._backend = MagicMock()
    setattr(runner, "_preprocessing_phase", lambda: None)
    setattr(runner, "_compile_config_phase", lambda: MagicMock())
    return runner


def test_a_paused_run_releases_workers_and_skips_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nhx.rl.tasks.training.runner.get_gpu_info", lambda: None)
    synced: list[str] = []
    completed: list[str] = []
    runner = _runner(tmp_path, synced, completed)
    setattr(runner, "_training_phase", MagicMock(side_effect=TrainingPaused()))

    result = runner.run()

    assert result.paused and not result.success
    # Workers wait at this barrier for the coordinator, so a pause must still reach it.
    assert synced == ["training_complete"]
    assert completed == []
    backend = cast(MagicMock, runner._backend)
    backend.find_best_checkpoint.assert_not_called()
    backend.process_checkpoint.assert_not_called()
    written = json.loads((tmp_path / DEFAULT_TRAINING_RESULT_FILE_NAME).read_text())
    assert written["paused"] is True
    assert written["training_duration_seconds"] == 3.0


def test_a_failure_reports_whether_a_rerun_would_repeat_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nhx.rl.tasks.training.runner.get_gpu_info", lambda: None)
    runner = _runner(tmp_path, [], [])
    report_error = MagicMock()
    runner._progress.report_error = report_error
    driver_output = [
        "RuntimeError: unable to allocate shared memory(shm) for file </torch_1_2_3>: No space left on device (28)",
        "torch.distributed.DistBackendError: NCCL error in: ProcessGroupNCCL.cpp:3866, remote process exited",
    ]
    runner._training_phase = MagicMock(side_effect=classify_output(driver_output, 1))  # type: ignore[method-assign]

    result = runner.run()

    error_details, status_details = report_error.call_args.args
    assert error_details["cause"] == "shared_memory_exhausted"
    assert status_details == {"resumable": False, "non_resumable_reason": result.error_message}


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
