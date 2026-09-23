# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TrainingProgressCallback metric accumulation."""

from unittest.mock import MagicMock

from nhx.customization_common.training.callbacks import TrainingProgressCallback


class TestTrainingProgressCallback:
    """Tests for metric accumulation and status_details reporting."""

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

    def test_validation_accumulates_metrics(self):
        callback, reporter = self._make_callback()

        callback.report_validation(step=250, epoch=1, metrics={"loss": 3.19})
        callback.report_validation(step=500, epoch=2, metrics={"loss": 2.09})

        kwargs = self._last_report_kwargs(reporter)
        assert kwargs["metrics"]["val_loss"] == [
            {"step": 250, "epoch": 1, "value": 3.19},
            {"step": 500, "epoch": 2, "value": 2.09},
        ]
        assert kwargs["val_loss"] == 2.09

    def test_mixed_train_and_val_both_present(self):
        callback, reporter = self._make_callback()

        callback.report_train_step(step=1, epoch=1, metrics={"loss": 3.21})
        callback.report_train_step(step=2, epoch=1, metrics={"loss": 2.89})
        callback.report_validation(step=2, epoch=1, metrics={"loss": 3.19})
        callback.report_train_step(step=3, epoch=1, metrics={"loss": 2.56})

        kwargs = self._last_report_kwargs(reporter)
        assert len(kwargs["metrics"]["train_loss"]) == 3
        assert len(kwargs["metrics"]["val_loss"]) == 1

    def test_metrics_included_in_every_update(self):
        callback, reporter = self._make_callback()

        callback.report_train_step(step=1, epoch=1, metrics={"loss": 3.21})
        first_call_kwargs = reporter.report_running.call_args_list[0].kwargs
        assert len(first_call_kwargs["metrics"]["train_loss"]) == 1
        assert first_call_kwargs["metrics"]["val_loss"] == []

        callback.report_train_step(step=2, epoch=1, metrics={"loss": 2.89})
        second_call_kwargs = reporter.report_running.call_args_list[1].kwargs
        assert len(second_call_kwargs["metrics"]["train_loss"]) == 2

    def test_train_step_uses_train_loss_flat_field(self):
        callback, reporter = self._make_callback()

        callback.report_train_step(step=1, epoch=1, metrics={"loss": 3.21})

        kwargs = self._last_report_kwargs(reporter)
        assert kwargs["train_loss"] == 3.21
        assert "loss" not in kwargs

    def test_train_step_passes_optional_fields(self):
        callback, reporter = self._make_callback()

        callback.report_train_step(step=1, epoch=1, metrics={"loss": 3.21, "lr": 0.0002, "grad_norm": 1.5})

        kwargs = self._last_report_kwargs(reporter)
        assert kwargs["train_lr"] == 0.0002
        assert kwargs["train_grad_norm"] == 1.5

    def test_seeds_from_server_on_init(self):
        prior = {
            "train_loss": [
                {"step": 1, "epoch": 1, "value": 3.21},
                {"step": 2, "epoch": 1, "value": 2.89},
            ],
            "val_loss": [
                {"step": 2, "epoch": 1, "value": 3.19},
            ],
        }
        callback, reporter = self._make_callback(prior_metrics=prior)

        assert len(callback._metrics["train_loss"]) == 2
        assert len(callback._metrics["val_loss"]) == 1
        reporter.fetch_current_metrics.assert_called_once()

    def test_seeded_metrics_included_in_first_report(self):
        prior = {
            "train_loss": [{"step": 1, "epoch": 1, "value": 3.21}],
            "val_loss": [],
        }
        callback, reporter = self._make_callback(prior_metrics=prior)

        callback.report_train_step(step=2, epoch=1, metrics={"loss": 2.89})

        kwargs = self._last_report_kwargs(reporter)
        assert kwargs["metrics"]["train_loss"] == [
            {"step": 1, "epoch": 1, "value": 3.21},
            {"step": 2, "epoch": 1, "value": 2.89},
        ]

    def test_seeded_val_metrics_preserved_across_train_steps(self):
        prior = {
            "train_loss": [{"step": 1, "epoch": 1, "value": 3.21}],
            "val_loss": [{"step": 1, "epoch": 1, "value": 3.50}],
        }
        callback, reporter = self._make_callback(prior_metrics=prior)

        callback.report_train_step(step=2, epoch=1, metrics={"loss": 2.89})

        kwargs = self._last_report_kwargs(reporter)
        assert len(kwargs["metrics"]["val_loss"]) == 1
        assert kwargs["metrics"]["val_loss"][0]["value"] == 3.50

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
        )

    def test_report_checkpoint_saved_delegates(self):
        callback, reporter = self._make_callback()

        callback.report_checkpoint_saved(step=100, epoch=1, checkpoint_path="/tmp/ckpt")

        reporter.report_running.assert_called_once_with(
            phase="checkpoint_saved",
            step=100,
            epoch=1,
            checkpoint_path="/tmp/ckpt",
        )

    def test_checkpoint_report_leaves_the_accumulated_series_alone(self):
        """The Jobs service merges status_details, so an omitted key is not an erasure.

        Checkpoint saves fire mid-training (finetune.py calls this from the save
        hook) and have nothing to add to the curves. Resending every series on
        each one would be pure upload; omitting the key leaves the stored series
        exactly as the last train step left it.
        """
        callback, reporter = self._make_callback()

        callback.report_train_step(step=1, epoch=1, metrics={"loss": 3.21})
        callback.report_checkpoint_saved(step=1, epoch=1, checkpoint_path="/tmp/ckpt")

        assert "metrics" not in self._last_report_kwargs(reporter)

    def test_epoch_end_report_leaves_the_accumulated_series_alone(self):
        callback, reporter = self._make_callback()

        callback.report_train_step(step=1, epoch=1, metrics={"loss": 3.21})
        callback.report_epoch_end(step=1, epoch=1)

        assert "metrics" not in self._last_report_kwargs(reporter)

    def test_close_delegates(self):
        callback, reporter = self._make_callback()
        callback.close()
        reporter.close.assert_called_once()
