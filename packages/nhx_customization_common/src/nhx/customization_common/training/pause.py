# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Notice a pause request from inside a training process.

Pausing a job moves its running step to ``pausing``. Nothing is sent to the
workload, so the training process has to notice on its own, write a checkpoint,
and exit with ``PAUSE_EXIT_CODE``. :class:`PausePoller` is the part that notices.

The paused run also leaves a marker naming the step it stopped at. Resume starts
a new process that is otherwise told nothing, and the marker is how that process
knows it must restore a checkpoint rather than start from scratch.
"""

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import NoReturn, Self

import httpx
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.constants import PAUSE_EXIT_CODE
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nhx.common.client_factory import get_task_nemo_client
from nhx.customization_common.service.context import NHXJobContext

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_SECONDS = 15.0
# A slow read only delays noticing the pause by one interval, so fail fast.
_POLL_READ_TIMEOUT_SECONDS = 10.0
_POLL_TIMEOUT = httpx.Timeout(_POLL_READ_TIMEOUT_SECONDS, connect=5.0)


#: Written when training stops for a pause; holds the step it stopped at.
PAUSE_MARKER_FILE_NAME = "PAUSED"


class ResumeCheckpointMissingError(RuntimeError):
    """A resume found no checkpoint to continue from."""


def exit_paused() -> NoReturn:
    """Exit with the code that tells the Jobs service the pause checkpoint is written."""
    raise SystemExit(PAUSE_EXIT_CODE)


def pause_marker_path(checkpoint_dir: Path) -> Path:
    """Where a pause is recorded: beside the checkpoint directory, so it outlives that directory."""
    return checkpoint_dir.parent / PAUSE_MARKER_FILE_NAME


def write_pause_marker(path: Path, step: int) -> None:
    """Record that training stopped for a pause at ``step``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{step}\n", encoding="utf-8")


def take_pause_marker(path: Path) -> int | None:
    """Remove the marker an earlier run's pause left, and return the step it paused at.

    None means that run did not pause, so this one is a first run or a rerun, not a
    resume. A marker that holds no step was still left by a pause, so it reads as
    step 0 and the resume still has to restore a checkpoint.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    path.unlink(missing_ok=True)
    try:
        return int(text.strip())
    except ValueError:
        logger.warning(f"Pause marker {path} holds no step: {text!r}")
        return 0


def check_resumed_from_pause(paused_step: int | None, restored_step: int, checkpoint_dir: object) -> None:
    """Fail a resume that restored no checkpoint, before it trains from step 0.

    Every backend starts from scratch when it finds no checkpoint, which is right for
    a first run. After a pause it would discard the paused progress and the metric
    history recorded for it. Restoring an older checkpoint than the pause saved is
    fine: training repeats those steps.

    Args:
        paused_step: The step from :func:`take_pause_marker`, or None when this run is not a resume.
        restored_step: The step the backend restored, 0 when it found no checkpoint.
        checkpoint_dir: Where the backend looked, for the error message.

    Raises:
        ResumeCheckpointMissingError: This run resumes a pause and restored nothing.
    """
    if paused_step is None or restored_step > 0:
        return
    raise ResumeCheckpointMissingError(
        f"Cannot resume: training paused at step {paused_step}, "
        f"but no complete checkpoint was found in {checkpoint_dir}."
    )


class PausePoller:
    """Poll this task's step, and act once when the step is ``pausing``.

    It runs on a daemon thread because training callbacks only run when the loop
    reaches them, and one training step can take minutes. A failed read is
    logged and retried on the next interval. When the step is ``pausing``,
    :attr:`requested` is set, ``on_pause`` runs once, and polling stops.

    Outside a real job (no job context), :meth:`start` does nothing.
    """

    def __init__(
        self,
        job_ctx: NHXJobContext,
        service_name: str,
        *,
        interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        on_pause: Callable[[], None] | None = None,
        jobs: JobsClient | None = None,
    ) -> None:
        """Build the poller.

        Args:
            job_ctx: The task's job context, which names the step to watch.
            service_name: Selects the task credentials, as for progress reporting.
            interval_seconds: Time between reads of the step.
            on_pause: Called once, on the polling thread, when the step is pausing.
            jobs: Jobs client to read with. When omitted, the poller builds one from
                the task credentials and closes it when it stops.
        """
        self._job_ctx = job_ctx
        self._interval_seconds = interval_seconds
        self._on_pause = on_pause
        self._owned_client = None
        if jobs is None and job_ctx.is_configured:
            self._owned_client = get_task_nemo_client(service_name)
            jobs = JobsClient.from_client(self._owned_client).with_options(timeout=_POLL_TIMEOUT)
        self._jobs = jobs
        #: Set once the step is pausing. Training code can check it directly.
        self.requested = threading.Event()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> Self:
        if self._jobs is None or not self._job_ctx.is_configured or self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, args=(self._jobs,), name="pause-poller", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop polling. Safe to call more than once, and from ``on_pause``."""
        self._stopped.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self._interval_seconds + _POLL_READ_TIMEOUT_SECONDS)
        if self._owned_client is not None:
            self._owned_client.close()
            self._owned_client = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def _run(self, jobs: JobsClient) -> None:
        # Check before the first wait: the pause may have come while the process started.
        while not self._stopped.is_set():
            if self._step_is_pausing(jobs):
                self._handle_pause()
                return
            self._stopped.wait(self._interval_seconds)

    def _step_is_pausing(self, jobs: JobsClient) -> bool:
        try:
            step = jobs.get_job_step(
                name=self._job_ctx.step,
                workspace=self._job_ctx.workspace,
                job=self._job_ctx.job_id,
            ).data()
        except Exception as e:
            logger.warning(f"Could not read the step status to check for a pause: {e}")
            return False
        return step.status == HelixJobStatus.PAUSING

    def _handle_pause(self) -> None:
        logger.info("Pause requested for step '%s'; training will stop at the next checkpoint", self._job_ctx.step)
        self.requested.set()
        if self._on_pause is not None:
            try:
                self._on_pause()
            except Exception:
                logger.exception("Pause handler failed")
