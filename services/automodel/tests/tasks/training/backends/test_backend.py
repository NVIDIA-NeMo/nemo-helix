# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AutomodelBackend embedding model type selection."""

import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

# Mock nemo_automodel before importing backend/config modules
# (nemo_automodel is only available in the training container)
sys.modules["nemo_automodel"] = MagicMock()
sys.modules["nemo_automodel._transformers"] = MagicMock()
sys.modules["nemo_automodel._transformers.registry"] = MagicMock()

from nhx.automodel.tasks.training.backends.backend import AutomodelBackend  # noqa: E402
from nhx.automodel.tasks.training.backends.checkpoints import ModelType  # noqa: E402
from nhx.automodel.tasks.training.progress import JobsServiceProgressReporter  # noqa: E402
from nhx.automodel.tasks.training.protocol import LibraryConfig  # noqa: E402
from nhx.automodel.tasks.training.schemas import TrainingRecipe  # noqa: E402
from nhx.customization_common.training.progress import TrainingWallClock  # noqa: E402


class TestAutomodelBackend:
    """Tests for AutomodelBackend."""

    def test_find_checkpoints_uses_model_embedding_flag(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """ModelType should be EMBEDDING when model.is_embedding_model is True."""
        backend = AutomodelBackend(job_ctx=MagicMock())
        customizer_config = MagicMock()
        customizer_config.model.checkpoint_head_type = "unknown"
        customizer_config.model.is_embedding_model = True
        customizer_config.model.name = "meta/llama-3.1-8b-instruct"

        expected = {"best": tmp_path / "best.ckpt"}
        mock_find_selected = mocker.patch(
            "nhx.automodel.tasks.training.backends.backend.find_selected_checkpoints",
            return_value=expected,
        )

        result = backend.find_checkpoints(tmp_path, customizer_config)

        assert result == expected
        mock_find_selected.assert_called_once_with(
            tmp_path,
            customizer_config,
            model_type=ModelType.EMBEDDING,
        )

    def test_process_checkpoints_uses_model_embedding_flag_not_model_name(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """ModelType should stay LLM when model.is_embedding_model is False."""
        backend = AutomodelBackend(job_ctx=MagicMock())
        customizer_config = MagicMock()
        customizer_config.model.checkpoint_head_type = "unknown"
        customizer_config.model.is_embedding_model = False
        customizer_config.model.name = "nvidia/llama-nemotron-embed-1b-v2"

        checkpoint_info = MagicMock()
        mock_process = mocker.patch(
            "nhx.automodel.tasks.training.backends.backend.process_selected_checkpoints",
            return_value=checkpoint_info,
        )

        checkpoints = {"best": tmp_path / "checkpoint"}
        output_path = tmp_path / "output_model"
        result = backend.process_checkpoints(
            checkpoints=checkpoints,
            output_path=output_path,
            workspace_dir=tmp_path,
            customizer_config=customizer_config,
            library_config=None,
        )

        assert result == checkpoint_info
        mock_process.assert_called_once_with(
            checkpoints,
            output_path,
            tmp_path,
            customizer_config,
            model_type=ModelType.LLM,
            resolved_chat_template=None,
        )

    def test_cross_encoder_recipe_uses_cross_encoder_checkpoint_processing(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        backend = AutomodelBackend(job_ctx=MagicMock())
        customizer_config = MagicMock()
        customizer_config.training.recipe = TrainingRecipe.CROSS_ENCODER

        expected = {"best": tmp_path / "best.ckpt"}
        mock_find_selected = mocker.patch(
            "nhx.automodel.tasks.training.backends.backend.find_selected_checkpoints",
            return_value=expected,
        )

        assert backend.find_checkpoints(tmp_path, customizer_config) == expected
        mock_find_selected.assert_called_once_with(
            tmp_path,
            customizer_config,
            model_type=ModelType.CROSS_ENCODER,
        )

    def test_auto_recipe_cross_encoder_head_uses_cross_encoder_checkpoint_processing(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        backend = AutomodelBackend(job_ctx=MagicMock())
        customizer_config = MagicMock()
        customizer_config.training.recipe = TrainingRecipe.AUTO
        customizer_config.model.is_embedding_model = False
        customizer_config.model.checkpoint_head_type = "cross_encoder"

        mock_process = mocker.patch(
            "nhx.automodel.tasks.training.backends.backend.process_selected_checkpoints",
            return_value=MagicMock(),
        )

        backend.process_checkpoints(
            checkpoints={"best": tmp_path / "checkpoint"},
            output_path=tmp_path / "output_model",
            workspace_dir=tmp_path,
            customizer_config=customizer_config,
            library_config=None,
        )

        assert mock_process.call_args.kwargs["model_type"] == ModelType.CROSS_ENCODER


class Proc:
    def __init__(self, *, fail: str | None = None, on_wait: Callable[[], None] | None = None) -> None:
        self.returncode = 0
        self.fail = fail
        self.on_wait = on_wait
        self.waits = 0
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        self.waits += 1
        if self.on_wait is not None:
            self.on_wait()
        if self.fail == "timeout" and self.waits == 1:
            raise subprocess.TimeoutExpired("torchrun", 1)
        if self.fail == "exit":
            self.returncode = 1
        return self.returncode

    def kill(self) -> None:
        self.killed = True

    def send_signal(self, signum: int) -> None:
        return None


class Thread:
    def __init__(self, *args: object, **kwargs: object) -> None:
        return None

    def start(self) -> None:
        return None

    def is_alive(self) -> bool:
        return False

    def join(self, timeout: float | None = None) -> None:
        return None


class RecordingProgress(JobsServiceProgressReporter):
    """Records the progress it would send, and times training with a scripted clock."""

    def __init__(self, start: float, end: float) -> None:
        # The base constructor builds a Jobs client; nothing here is sent anywhere.
        self.training_duration_seconds = None
        self.sent: list[dict] = []
        self._ticks = iter((start, end))

    def update_task(
        self,
        status: str = "active",
        status_details: dict[str, Any] | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> None:
        self.sent.append(status_details or {})

    def report_running(self, phase: str, **details: Any) -> None:
        self.sent.append({"phase": phase, **details})

    def training_wall_clock(self, clock: Callable[[], float] | None = None) -> TrainingWallClock:
        return super().training_wall_clock(clock=lambda: next(self._ticks))


def run_execute(monkeypatch: pytest.MonkeyPatch, progress: JobsServiceProgressReporter, proc: Proc):
    monkeypatch.setattr("nhx.automodel.tasks.training.backends.backend.subprocess.Popen", lambda *a, **k: proc)
    monkeypatch.setattr("nhx.automodel.tasks.training.backends.backend.threading.Thread", Thread)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    config = MagicMock()
    config.parallelism.num_nodes = 1
    config.training_timeout = None
    library = LibraryConfig(config_dict={}, config_path=Path("/tmp/automodel_config.yaml"))
    try:
        return AutomodelBackend(job_ctx=MagicMock()).execute_training(config, library, progress)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


class TestExecuteTrainingWallClock:
    def test_duration_is_the_torchrun_wait(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        progress = RecordingProgress(10.0, 25.0)

        def check_training_is_running() -> None:
            assert any("training_started_at" in item for item in progress.sent)
            assert all("training_finished_at" not in item for item in progress.sent)

        proc = Proc(on_wait=check_training_is_running)
        with caplog.at_level("INFO"):
            result = run_execute(monkeypatch, progress, proc)

        assert result.total_steps == 0
        assert progress.training_duration_seconds == 15.0
        finished = next(item for item in progress.sent if "training_finished_at" in item)
        assert finished["training_duration_seconds"] == 15.0
        assert "Training finished in 15.0 seconds" in caplog.text

    def test_timeout_still_reports_the_duration(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        progress = RecordingProgress(10.0, 25.0)
        with caplog.at_level("INFO"), pytest.raises(subprocess.TimeoutExpired):
            run_execute(monkeypatch, progress, Proc(fail="timeout"))

        assert progress.training_duration_seconds == 15.0
        assert any("training_finished_at" in item for item in progress.sent)
        assert "Training finished in 15.0 seconds" in caplog.text

    def test_nonzero_exit_still_reports_the_duration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        progress = RecordingProgress(10.0, 25.0)
        with pytest.raises(Exception):
            run_execute(monkeypatch, progress, Proc(fail="exit"))

        assert progress.training_duration_seconds == 15.0
        assert any(item.get("training_duration_seconds") == 15.0 for item in progress.sent)
