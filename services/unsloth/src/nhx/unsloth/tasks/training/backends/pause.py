# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pause and resume for the Unsloth trainer.

A pause saves a checkpoint at the end of the current optimizer step and stops
training. Resume, and a rerun after a crash, continue from the newest complete
checkpoint.
"""

import json
import logging
import shutil
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CHECKPOINT_PREFIX = "checkpoint-"
_TRAINER_STATE_FILE = "trainer_state.json"


def _saved_step(checkpoint: Path) -> int | None:
    """The step a checkpoint holds, or None when its trainer state is missing or not valid JSON.

    Other read errors, such as a permissions error or an I/O timeout, are raised: the checkpoint may
    be complete, so it must not be removed.
    """
    try:
        state = json.loads((checkpoint / _TRAINER_STATE_FILE).read_text())
    except (FileNotFoundError, ValueError):
        return None
    step = state.get("global_step") if isinstance(state, dict) else None
    return step if isinstance(step, int) else None


def newest_complete_checkpoint(checkpoint_dir: Path) -> tuple[Path, int] | None:
    """Return the newest complete ``checkpoint-N`` and its step, after removing incomplete ones.

    The trainer writes ``trainer_state.json`` last, so a checkpoint whose trainer
    state is missing or not valid JSON was interrupted mid-save. It cannot be
    resumed from, so it is removed.
    """
    if not checkpoint_dir.is_dir():
        return None
    newest: tuple[Path, int] | None = None
    for path in checkpoint_dir.iterdir():
        if not path.is_dir() or not path.name.startswith(_CHECKPOINT_PREFIX):
            continue
        step = _saved_step(path)
        if step is None:
            logger.warning(f"Removing incomplete checkpoint {path}")
            shutil.rmtree(path, ignore_errors=True)
            continue
        if newest is None or step > newest[1]:
            newest = (path, step)
    return newest


def create_pause_callback(requested: threading.Event) -> Any:
    """Build a :class:`~transformers.TrainerCallback` that saves and stops once a pause is requested.

    Import is deferred so this module stays importable without ``transformers``.
    The trainer runs it after its own flow callback, so its settings win.
    """
    from transformers import TrainerCallback

    class PauseCallback(TrainerCallback):
        def __init__(self) -> None:
            #: Set once training stopped for the pause. The trainer saves before it stops.
            self.stopped = False

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            # A pause in the last step loses the race: training finishes and publishes as usual.
            if self.stopped or not requested.is_set() or state.global_step >= state.max_steps:
                return control
            self.stopped = True
            control.should_save = True
            control.should_training_stop = True
            control.should_evaluate = False
            # Report the loss since the last log; the resumed run starts a new average.
            control.should_log = True
            return control

        def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            # Stopping ends the epoch, and an end-of-epoch save or evaluation would repeat the pause step.
            if self.stopped:
                control.should_save = False
                control.should_evaluate = False
            return control

    return PauseCallback()
