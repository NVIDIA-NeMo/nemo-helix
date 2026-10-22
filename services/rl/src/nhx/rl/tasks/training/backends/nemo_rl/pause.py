# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stop a NeMo-RL training loop for a pause.

The GRPO and DPO loops save a checkpoint and return when their
``TimeoutChecker.check_save()`` is true. While training runs, the drivers swap in
a checker that is also true once the step is ``pausing``. A pause then ends the
loop at the next step boundary with a complete ``step_N`` checkpoint, and resume
restores it the same way NeMo-RL restores after its own timeout save.
"""

import contextlib
import threading
from collections.abc import Iterator
from types import ModuleType
from typing import Any

from nhx.customization_common.service.context import NHXJobContext
from nhx.customization_common.training.pause import PausePoller
from nhx.rl.app.constants import SERVICE_NAME


class PauseStop:
    """Ends a training loop at its next step boundary once a pause is requested."""

    def __init__(self, requested: threading.Event) -> None:
        self._requested = requested
        #: Set once the loop saved a checkpoint because of the pause, after which it returns.
        self.saved = False

    @contextlib.contextmanager
    def patch(self, algorithm: ModuleType) -> Iterator[None]:
        """Swap ``algorithm.TimeoutChecker`` for a pause-aware subclass while the block runs.

        The training function looks the name up when it is called, so patching the
        module is enough.
        """
        original: Any = algorithm.TimeoutChecker
        stop = self

        class PauseAwareTimeoutChecker(original):
            def would_save(self) -> bool:
                if not self.last_saved and stop._requested.is_set():
                    return True
                return super().would_save()

            def check_save(self) -> bool:
                saving = super().check_save()
                # Read after the decision: a request is never withdrawn, so a save it caused is seen here.
                if saving and stop._requested.is_set():
                    stop.saved = True
                return saving

        algorithm.TimeoutChecker = PauseAwareTimeoutChecker
        try:
            yield
        finally:
            algorithm.TimeoutChecker = original

    def stopped_training(self, completed_steps: int, planned_steps: int) -> bool:
        """Whether the loop returned for the pause rather than by finishing.

        A pause that arrives at the last step loses the race: that save is the
        final one, so training finished and publishes as usual.
        """
        return self.saved and completed_steps < planned_steps


def planned_steps(max_num_steps: int, max_num_epochs: int, steps_per_epoch: int | None) -> int:
    """How many steps the loop runs before it finishes on its own.

    ``steps_per_epoch`` is the training dataloader's length, or None when epochs do
    not bound the run.
    """
    if steps_per_epoch is None:
        return max_num_steps
    return min(max_num_steps, max_num_epochs * steps_per_epoch)


@contextlib.contextmanager
def stop_for_pause(algorithm: ModuleType, job_ctx: NHXJobContext) -> Iterator[PauseStop]:
    """Watch this step for a pause, and end ``algorithm``'s training loop when one comes."""
    with PausePoller(job_ctx, SERVICE_NAME) as poller:
        stop = PauseStop(poller.requested)
        with stop.patch(algorithm):
            yield stop
