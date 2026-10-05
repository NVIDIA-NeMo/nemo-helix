# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge HuggingFace Trainer callbacks to Jobs-service progress reporting."""

import math
from typing import Any

from nhx.unsloth.tasks.training.backends.callbacks import TrainingProgressCallback


def _epoch_from_value(raw_epoch: float | int, num_epochs: int) -> int:
    """Map HF fractional epoch values to a 1-based epoch index."""
    return max(1, min(num_epochs, math.ceil(float(raw_epoch))))


def _as_float(value: Any) -> Any:
    """Coerce a logged value to a float, leaving it alone if it will not go.

    The Trainer does not promise a builtin: some paths log ``grad_norm`` as a
    0-dim tensor rather than calling ``.item()`` on it, and a tensor is not a
    ``numbers.Real``, so the callback's chartable filter drops it and the curve
    silently never appears. ``float()`` converts anything with ``__float__``,
    which covers those.

    A value it cannot convert is passed through rather than raised on or dropped
    here: deciding what belongs in a series is the callback's job, and it already
    drops what it cannot chart. ``None`` goes through untouched for the same
    reason -- the trainer omits ``learning_rate`` and ``grad_norm`` on some
    steps, and that is not an error.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def create_hf_trainer_progress_callback(
    progress_callback: TrainingProgressCallback,
    *,
    backend: str = "unsloth",
) -> Any:
    """Build a HuggingFace :class:`~transformers.TrainerCallback` for Jobs reporting.

    Import is deferred so this module stays importable without ``transformers``.
    """
    from transformers import TrainerCallback

    class HfTrainerProgressCallback(TrainerCallback):
        def __init__(self) -> None:
            self._progress = progress_callback
            self._backend = backend
            self._num_epochs = 1

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self._num_epochs = max(1, int(args.num_train_epochs))
            max_steps = max(1, int(state.max_steps))
            self._progress.report_training_start(
                max_steps=max_steps,
                num_epochs=self._num_epochs,
                backend=self._backend,
            )

        def on_log(
            self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any
        ) -> None:
            if not logs or "loss" not in logs:
                return

            epoch_raw = logs.get("epoch", state.epoch if state.epoch is not None else 0)
            self._progress.report_train_step(
                step=int(state.global_step),
                epoch=_epoch_from_value(epoch_raw, self._num_epochs),
                # `learning_rate` is renamed to the `lr` every other backend uses,
                # so the series is `train_lr` regardless of who reported it. The
                # callback drops whichever of these the trainer did not produce.
                metrics={
                    "loss": _as_float(logs["loss"]),
                    "lr": _as_float(logs.get("learning_rate")),
                    "grad_norm": _as_float(logs.get("grad_norm")),
                },
                backend=self._backend,
            )

        def on_evaluate(
            self,
            args: Any,
            state: Any,
            control: Any,
            metrics: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            if not metrics or "eval_loss" not in metrics:
                return

            epoch_raw = metrics.get("epoch", state.epoch if state.epoch is not None else 0)
            self._progress.report_validation(
                step=int(state.global_step),
                epoch=_epoch_from_value(epoch_raw, self._num_epochs),
                metrics={"loss": _as_float(metrics["eval_loss"])},
                backend=self._backend,
            )

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            checkpoint_path = kwargs.get("checkpoint_folder", args.output_dir)
            epoch_raw = state.epoch if state.epoch is not None else 0
            self._progress.report_checkpoint_saved(
                step=int(state.global_step),
                epoch=_epoch_from_value(epoch_raw, self._num_epochs),
                checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
                backend=self._backend,
            )

    return HfTrainerProgressCallback()
