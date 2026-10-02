# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TrainingProgressCallback metric accumulation."""

from unittest.mock import MagicMock

from nhx.unsloth.tasks.training.backends.callbacks import TrainingProgressCallback


class TestTrainingProgressCallback:
    def _make_callback(self, prior_metrics: dict | None = None) -> tuple[TrainingProgressCallback, MagicMock]:
        mock_reporter = MagicMock()
        mock_reporter.fetch_current_metrics.return_value = prior_metrics or {
            "train_loss": [],
            "val_loss": [],
        }
        callback = TrainingProgressCallback(mock_reporter, min_report_interval_seconds=0)
        return callback, mock_reporter

    def _last_report_kwargs(self, mock_reporter: MagicMock) -> dict:
        return mock_reporter.report_running.call_args.kwargs

    def test_train_step_accumulates_metrics(self):
        callback, reporter = self._make_callback()

        callback.report_train_step(step=1, epoch=1, metrics={"loss": 3.21})
        callback.report_train_step(step=2, epoch=1, metrics={"loss": 2.89})
        callback.report_train_step(step=3, epoch=1, metrics={"loss": 2.56})

        kwargs = self._last_report_kwargs(reporter)
        assert kwargs["metrics"]["train_loss"] == [
            {"step": 1, "epoch": 1, "value": 3.21},
            {"step": 2, "epoch": 1, "value": 2.89},
            {"step": 3, "epoch": 1, "value": 2.56},
        ]

    def test_train_step_uses_train_loss_flat_field(self):
        callback, reporter = self._make_callback()

        callback.report_train_step(step=1, epoch=1, metrics={"loss": 3.21})

        kwargs = self._last_report_kwargs(reporter)
        assert kwargs["train_loss"] == 3.21
        assert kwargs["backend"] == "unsloth"
        assert "loss" not in kwargs

    def test_train_step_passes_optional_fields(self):
        callback, reporter = self._make_callback()

        callback.report_train_step(step=1, epoch=1, metrics={"loss": 3.21, "lr": 0.0002, "grad_norm": 1.5})

        kwargs = self._last_report_kwargs(reporter)
        assert kwargs["train_lr"] == 0.0002
        assert kwargs["train_grad_norm"] == 1.5

    def test_report_training_start_delegates(self):
        callback, reporter = self._make_callback()

        callback.report_training_start(max_steps=500, num_epochs=2)

        reporter.configure_progress_tracking.assert_called_once_with(500, 2)
        # No `step`: it fires before the first one, and a literal 0 would write
        # 0% over whatever progress the task had stored.
        reporter.report_running.assert_called_once_with(
            phase="training",
            max_steps=500,
            num_epochs=2,
            backend="unsloth",
        )

    def test_close_delegates(self):
        callback, reporter = self._make_callback()
        callback.close()
        reporter.close.assert_called_once()
