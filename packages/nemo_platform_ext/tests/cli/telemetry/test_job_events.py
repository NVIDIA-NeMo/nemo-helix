# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from nemo_platform_ext.cli.core import waiters
from nemo_platform_ext.cli.telemetry.events import TaskStatusEnum
from nemo_platform_plugin.jobs.schemas import (
    PlatformJobStatus,
    PlatformJobStatusResponse,
    PlatformJobStepStatusResponse,
)

WAITERS_MODULE = "nemo_platform_ext.cli.core.waiters"
WATCH_MODULE = "nemo_platform_plugin.jobs.watch"
EMIT_TARGET = "nemo_platform_ext.cli.telemetry.emit.emit_event"
JOB_TIMESTAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _DummyLive:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __enter__(self) -> _DummyLive:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def update(self, *args: object) -> None:
        pass

    def stop(self) -> None:
        pass

    def start(self) -> None:
        pass


class _StatusResponse:
    def __init__(self, status: PlatformJobStatusResponse) -> None:
        self._status = status

    def data(self) -> PlatformJobStatusResponse:
        return self._status


def _step(name: str) -> PlatformJobStepStatusResponse:
    return PlatformJobStepStatusResponse(
        id=name,
        name=name,
        status=PlatformJobStatus.COMPLETED,
        status_details={},
        error_details=None,
        tasks=[],
        created_at=JOB_TIMESTAMP,
        updated_at=JOB_TIMESTAMP,
    )


def _status_response(
    status: str | PlatformJobStatus,
    *,
    steps: list[str] | None = None,
    status_details: dict[str, object] | None = None,
    created_at: datetime = JOB_TIMESTAMP,
) -> _StatusResponse:
    return _StatusResponse(
        PlatformJobStatusResponse(
            id="job-a",
            name="job-a",
            status=PlatformJobStatus(status),
            status_details=status_details or {},
            error_details=None,
            steps=[_step(step) for step in steps or []],
            created_at=created_at,
            updated_at=created_at,
        )
    )


@pytest.fixture(autouse=True)
def _quiet_rich_output() -> Iterator[None]:
    with (
        patch(f"{WAITERS_MODULE}.Live", _DummyLive),
        patch(f"{WAITERS_MODULE}.console.print"),
    ):
        yield


@pytest.fixture
def frozen_time() -> Iterator[MagicMock]:
    with patch(f"{WAITERS_MODULE}.time.time", return_value=0) as time:
        yield time


@pytest.fixture
def waiter_pause() -> Iterator[MagicMock]:
    with patch(f"{WAITERS_MODULE}._pause") as pause:
        yield pause


def _completed_status() -> _StatusResponse:
    return _status_response(
        "completed",
        steps=[
            "audit-job",
            "evaluate",
            "evaluate-suite",
            "customer-project-step",
        ],
        status_details={"input_tokens": 512, "output_tokens": 2048, "model": "nemotron-super-49b"},
    )


def test_completed_emits_single_job_run_event(frozen_time: MagicMock) -> None:
    jobs = MagicMock()
    jobs.get_job_status.return_value = _completed_status()

    with patch(EMIT_TARGET) as emit_event:
        assert waiters.wait_for_platform_job(jobs, "job-a", workspace="default", resource_label="customization") is True

    emit_event.assert_called_once()
    event = emit_event.call_args.args[0]
    assert event.job_type == "customization"
    assert event.task_status is TaskStatusEnum.COMPLETED
    assert event.input_tokens == 512
    assert event.output_tokens == 2048
    assert event.model == "defined"
    assert event.plugins == []
    assert "customer-project-step" not in event.job_type


def test_static_step_name_is_not_emitted_as_job_type(frozen_time: MagicMock) -> None:
    jobs = MagicMock()
    jobs.get_job_status.return_value = _status_response("completed", steps=["audit-job"])

    with patch(EMIT_TARGET) as emit_event:
        assert waiters.wait_for_platform_job(jobs, "job-a", workspace="default", resource_label="audit") is True

    event = emit_event.call_args.args[0]
    assert event.job_type == "audit"
    assert event.plugins == []


def test_job_type_falls_back_to_resource_label_without_steps(frozen_time: MagicMock) -> None:
    jobs = MagicMock()
    jobs.get_job_status.return_value = _status_response("completed")

    with patch(EMIT_TARGET) as emit_event:
        assert waiters.wait_for_platform_job(jobs, "job-a", workspace="default", resource_label="customization") is True

    event = emit_event.call_args.args[0]
    assert event.job_type == "customization"
    assert event.plugins == []


def test_unsafe_resource_label_falls_back_to_custom(frozen_time: MagicMock) -> None:
    jobs = MagicMock()
    jobs.get_job_status.return_value = _status_response("completed", steps=["private-customer-step"])

    with patch(EMIT_TARGET) as emit_event:
        assert (
            waiters.wait_for_platform_job(jobs, "job-a", workspace="default", resource_label="customer alpha project")
            is True
        )

    event = emit_event.call_args.args[0]
    assert event.job_type == "custom"
    assert event.plugins == []


def test_status_details_defaults_when_absent(frozen_time: MagicMock) -> None:
    jobs = MagicMock()
    jobs.get_job_status.return_value = _status_response("completed")

    with patch(EMIT_TARGET) as emit_event:
        assert waiters.wait_for_platform_job(jobs, "job-a", workspace="default") is True

    event = emit_event.call_args.args[0]
    assert event.input_tokens == -1
    assert event.output_tokens == -1
    assert event.model == "undefined"
    assert event.plugins == []


def test_null_status_details_still_emits(frozen_time: MagicMock) -> None:
    """Explicit nulls must not drop the event; a real 0 token count must survive."""
    jobs = MagicMock()
    jobs.get_job_status.return_value = _status_response(
        "completed",
        status_details={"model": None, "input_tokens": 0, "output_tokens": None},
    )

    with patch(EMIT_TARGET) as emit_event:
        assert waiters.wait_for_platform_job(jobs, "job-a", workspace="default") is True

    emit_event.assert_called_once()
    event = emit_event.call_args.args[0]
    assert event.model == "undefined"
    assert event.input_tokens == 0
    assert event.output_tokens == -1


def test_non_string_model_details_still_emit_safe_bucket(frozen_time: MagicMock) -> None:
    jobs = MagicMock()
    jobs.get_job_status.return_value = _status_response(
        "completed",
        status_details={"model": {"name": "private-model"}, "input_tokens": 7, "output_tokens": 9},
    )

    with patch(EMIT_TARGET) as emit_event:
        assert waiters.wait_for_platform_job(jobs, "job-a", workspace="default") is True

    emit_event.assert_called_once()
    event = emit_event.call_args.args[0]
    assert event.model == "defined"
    assert event.input_tokens == 7
    assert event.output_tokens == 9


def test_duration_uses_job_created_at_when_available() -> None:
    jobs = MagicMock()
    jobs.get_job_status.return_value = _status_response(
        "completed",
        created_at=datetime.fromtimestamp(90.0, tz=timezone.utc),
    )
    # Elapsed display uses the monotonic clock; the duration compares the job's
    # created_at against wall-clock time, which is what the event reports.
    with (
        patch(f"{WAITERS_MODULE}.time.time", return_value=130.0),
        patch(EMIT_TARGET) as emit_event,
    ):
        assert waiters.wait_for_platform_job(jobs, "job-a", workspace="default") is True

    event = emit_event.call_args.args[0]
    assert event.duration_sec == 40.0


def test_error_status_maps_to_error(frozen_time: MagicMock) -> None:
    jobs = MagicMock()
    jobs.get_job_status.return_value = _status_response("error")

    with patch(EMIT_TARGET) as emit_event:
        assert waiters.wait_for_platform_job(jobs, "job-a", workspace="default") is False

    emit_event.assert_called_once()
    assert emit_event.call_args.args[0].task_status is TaskStatusEnum.ERROR


def test_cancelled_status_maps_to_canceled(frozen_time: MagicMock) -> None:
    jobs = MagicMock()
    jobs.get_job_status.return_value = _status_response("cancelled")

    with patch(EMIT_TARGET) as emit_event:
        assert waiters.wait_for_platform_job(jobs, "job-a", workspace="default") is False

    emit_event.assert_called_once()
    assert emit_event.call_args.args[0].task_status is TaskStatusEnum.CANCELED


def test_timeout_emits_nothing_and_does_not_crash() -> None:
    jobs = MagicMock()
    jobs.get_job_status.return_value = _status_response("active")

    watch_clock = iter([0.0, 0.0, 4.0, 5.0])

    def monotonic() -> float:
        caller = sys._getframe(1).f_globals["__name__"]
        return next(watch_clock) if caller == WATCH_MODULE else 0.0

    with (
        patch("time.monotonic", monotonic),
        patch(f"{WATCH_MODULE}.time.sleep"),
        patch(EMIT_TARGET) as emit_event,
    ):
        assert waiters.wait_for_platform_job(jobs, "job-a", workspace="default", timeout=5, poll_interval=10) is False

    emit_event.assert_not_called()
