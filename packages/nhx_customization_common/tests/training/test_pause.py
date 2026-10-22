# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace
from typing import Any, cast

import pytest
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nhx.customization_common.service.context import NHXJobContext
from nhx.customization_common.training.pause import PausePoller, exit_paused


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
