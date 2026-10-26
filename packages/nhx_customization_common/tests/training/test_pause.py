# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nhx.customization_common.service.context import NHXJobContext
from nhx.customization_common.training.errors.classifier import classify_exception
from nhx.customization_common.training.errors.exceptions import CheckpointError
from nhx.customization_common.training.pause import (
    PausePoller,
    ResumeCheckpointMissingError,
    check_resumed_from_pause,
    exit_paused,
    pause_marker_path,
    take_pause_marker,
    write_pause_marker,
)


class _JobCtx:
    workspace = "default"
    job_id = "job-1"
    step = "training"
    is_configured = True


class _Jobs:
    """Answers step reads in order, repeating the last answer. An exception answer is raised."""

    def __init__(self, *answers: HelixJobStatus | Exception) -> None:
        self._answers = list(answers)
        self.reads: list[dict[str, Any]] = []
        self.read_twice = threading.Event()

    def get_job_step(self, **kwargs: Any) -> SimpleNamespace:
        self.reads.append(kwargs)
        if len(self.reads) >= 2:
            self.read_twice.set()
        answer = self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(data=lambda: SimpleNamespace(status=answer))


def _poller(jobs: _Jobs, **kwargs: Any) -> PausePoller:
    return PausePoller(
        cast(NHXJobContext, _JobCtx()),
        service_name="customizer",
        interval_seconds=0.01,
        jobs=cast(JobsClient, jobs),
        **kwargs,
    )


def test_a_pausing_step_sets_the_event_and_calls_the_handler_once() -> None:
    calls: list[str] = []
    jobs = _Jobs(HelixJobStatus.ACTIVE, HelixJobStatus.ACTIVE, HelixJobStatus.PAUSING)

    with _poller(jobs, on_pause=lambda: calls.append("paused")) as poller:
        assert poller.requested.wait(timeout=5)

    assert calls == ["paused"]
    assert jobs.reads[0] == {"name": "training", "workspace": "default", "job": "job-1"}


def test_polling_stops_after_the_pause_is_handled() -> None:
    jobs = _Jobs(HelixJobStatus.PAUSING)

    with _poller(jobs) as poller:
        assert poller.requested.wait(timeout=5)
        reads = len(jobs.reads)
        threading.Event().wait(0.1)

    assert len(jobs.reads) == reads == 1


def test_a_failed_read_is_retried_on_the_next_interval() -> None:
    jobs = _Jobs(ConnectionError("jobs service unavailable"), HelixJobStatus.PAUSING)

    with _poller(jobs) as poller:
        assert poller.requested.wait(timeout=5)


def test_a_handler_error_does_not_escape_the_thread() -> None:
    def fail() -> None:
        raise RuntimeError("signal failed")

    with _poller(_Jobs(HelixJobStatus.PAUSING), on_pause=fail) as poller:
        assert poller.requested.wait(timeout=5)


def test_stop_ends_polling_when_no_pause_comes() -> None:
    jobs = _Jobs(HelixJobStatus.ACTIVE)
    poller = _poller(jobs).start()
    assert jobs.read_twice.wait(timeout=5)

    poller.stop()
    reads = len(jobs.reads)
    threading.Event().wait(0.1)

    assert not poller.requested.is_set()
    assert len(jobs.reads) == reads


def test_outside_a_job_nothing_is_polled() -> None:
    ctx = _JobCtx()
    ctx.is_configured = False
    jobs = _Jobs(HelixJobStatus.PAUSING)

    poller = PausePoller(cast(NHXJobContext, ctx), service_name="customizer", jobs=cast(JobsClient, jobs)).start()
    poller.stop()

    assert jobs.reads == []


def test_exit_paused_uses_the_pause_exit_code() -> None:
    with pytest.raises(SystemExit) as exc_info:
        exit_paused()

    assert exc_info.value.code == 75


# --------------------------------------------------------------------------- #
# The pause marker, and the check that a resume restored a checkpoint
# --------------------------------------------------------------------------- #


def test_the_marker_sits_beside_the_checkpoints_so_it_outlives_them(tmp_path: Path) -> None:
    assert pause_marker_path(tmp_path / "training" / "checkpoints") == tmp_path / "training" / "PAUSED"


def test_a_resume_takes_the_step_the_pause_wrote(tmp_path: Path) -> None:
    """Taken, not read: a later run must not mistake this pause for its own."""
    marker = tmp_path / "training" / "PAUSED"
    write_pause_marker(marker, 9)

    assert take_pause_marker(marker) == 9
    assert not marker.exists()


def test_without_a_marker_the_run_is_not_a_resume(tmp_path: Path) -> None:
    assert take_pause_marker(tmp_path / "PAUSED") is None


def test_a_marker_without_a_step_still_counts_as_a_pause(tmp_path: Path) -> None:
    """Reading it as no pause would let the resume start from scratch."""
    marker = tmp_path / "PAUSED"
    marker.write_text("", encoding="utf-8")

    assert take_pause_marker(marker) == 0
    assert not marker.exists()


def test_a_resume_that_restored_nothing_fails(tmp_path: Path) -> None:
    with pytest.raises(ResumeCheckpointMissingError, match=r"paused at step 9, but no complete checkpoint .* in /ckpt"):
        check_resumed_from_pause(9, 0, "/ckpt")


@pytest.mark.parametrize(
    "paused_step,restored_step",
    [
        (None, 0),  # a first run starts from scratch
        (None, 8),  # a rerun after a crash continues from its newest checkpoint
        (9, 9),  # a resume restores the pause checkpoint
        (9, 8),  # or an older one, when the newest was incomplete; training repeats the step
    ],
)
def test_a_first_run_a_rerun_and_a_restored_resume_continue(paused_step: int | None, restored_step: int) -> None:
    check_resumed_from_pause(paused_step, restored_step, "/ckpt")


def test_a_resume_that_restored_nothing_tells_the_user_to_create_a_new_job() -> None:
    """In process (Unsloth) the classifier sees the exception itself, not a printed line."""
    error = classify_exception(ResumeCheckpointMissingError("Cannot resume: training paused at step 9"))

    assert isinstance(error, CheckpointError)
    assert error.cause == "resume_checkpoint_missing"
    assert not error.rerunnable
    assert "Create a new job" in error.message
