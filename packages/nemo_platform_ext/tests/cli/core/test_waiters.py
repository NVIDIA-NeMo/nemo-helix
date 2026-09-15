# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest
from nemo_platform_ext.cli.core import waiters
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.client.errors import InternalServerError
from nemo_platform_plugin.jobs.schemas import PlatformJobStatus, PlatformJobStatusResponse

WAITERS_MODULE = "nemo_platform_ext.cli.core.waiters"
WATCH_MODULE = "nemo_platform_plugin.jobs.watch"
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


class _RecordingLive(_DummyLive):
    instances: list[_RecordingLive] = []

    def __init__(self, renderable: object | None = None, *_args: object, **_kwargs: object) -> None:
        self.renderable = renderable
        self.updates: list[object] = []
        self.instances.append(self)

    def update(self, renderable: object, *_args: object, **_kwargs: object) -> None:
        self.updates.append(renderable)


class _StatusResponse:
    def __init__(self, status: PlatformJobStatusResponse) -> None:
        self._status = status

    def data(self) -> PlatformJobStatusResponse:
        return self._status


def _deployment_json(status: str, *, status_message: str = "", model_provider_id: str | None = None) -> dict:
    return {
        "id": "dep-1",
        "name": "deployment-a",
        "workspace": "default",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "entity_version": 1,
        "config": "cfg",
        "config_version": 1,
        "status": status,
        "status_message": status_message,
        "status_history": [],
        "model_provider_id": model_provider_id,
    }


def _scripted_client(steps: list[httpx.Response | Exception]) -> NemoClient:
    """A real NemoClient whose transport answers each request with the next scripted step.

    A step that is an exception is raised from the transport, which the client
    surfaces as ``NemoTransportError``. Retries are disabled so every request
    consumes exactly one step.
    """
    remaining = list(steps)

    def handler(request: httpx.Request) -> httpx.Response:
        if not remaining:
            raise AssertionError(f"unexpected request {request.method} {request.url}")
        step = remaining.pop(0)
        if isinstance(step, Exception):
            raise step
        step.request = request
        return step

    return NemoClient(
        base_url="http://test",
        workspace="default",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _status_response(status: str | PlatformJobStatus) -> _StatusResponse:
    return _StatusResponse(
        PlatformJobStatusResponse(
            id="job-a",
            name="job-a",
            status=PlatformJobStatus(status),
            status_details={},
            error_details=None,
            steps=[],
            created_at=JOB_TIMESTAMP,
            updated_at=JOB_TIMESTAMP,
        )
    )


@pytest.fixture(autouse=True)
def _quiet_rich_output() -> Iterator[None]:
    with (
        patch(f"{WAITERS_MODULE}.Live", _DummyLive),
        patch(f"{WAITERS_MODULE}.console.print"),
    ):
        yield


@pytest.fixture(autouse=True)
def _no_telemetry() -> Iterator[None]:
    """Neutralize best-effort job_run telemetry so waiter tests never do real I/O."""
    with patch("nemo_platform_ext.cli.telemetry.emit.emit_event"):
        yield


@pytest.fixture
def frozen_time() -> Iterator[MagicMock]:
    with patch(f"{WAITERS_MODULE}.time.monotonic", return_value=0) as time:
        yield time


@pytest.fixture
def waiter_pause() -> Iterator[MagicMock]:
    with patch(f"{WAITERS_MODULE}._pause") as pause:
        yield pause


@pytest.fixture
def gateway_wait() -> Iterator[MagicMock]:
    with patch(f"{WAITERS_MODULE}.wait_for_gateway", return_value=True) as wait_for_gateway:
        yield wait_for_gateway


def test_wait_for_platform_job_returns_true_on_completed(frozen_time: MagicMock) -> None:
    jobs = MagicMock()
    jobs.get_job_status.return_value = _status_response("completed")

    assert waiters.wait_for_platform_job(jobs, "job-a", workspace="default") is True

    jobs.get_job_status.assert_called_once_with(workspace="default", name="job-a")
    frozen_time.assert_called()


def test_wait_for_platform_job_returns_false_on_error(frozen_time: MagicMock) -> None:
    jobs = MagicMock()
    jobs.get_job_status.return_value = _status_response("error")

    assert waiters.wait_for_platform_job(jobs, "job-a", workspace="default") is False
    frozen_time.assert_called()


def test_platform_job_wait_live_display_recomputes_elapsed() -> None:
    display = waiters._PlatformJobWaitLiveDisplay(start_time=100.0, timeout=1200, poll_interval=3)

    with (
        patch(f"{WAITERS_MODULE}.datetime") as datetime_mock,
        patch(f"{WAITERS_MODULE}.time.monotonic", side_effect=[101.0, 109.0]),
    ):
        datetime_mock.now.return_value.strftime.return_value = "12:34:56"

        assert "Wait: 1s" in display.__rich__().plain
        assert "Wait: 9s" in display.__rich__().plain


def test_wait_for_platform_job_uses_dynamic_live_display_for_unchanged_status_polls() -> None:
    jobs = MagicMock()
    jobs.get_job_status.side_effect = [
        _status_response("active"),
        _status_response("active"),
        _status_response("completed"),
    ]
    _RecordingLive.instances = []

    with (
        patch(f"{WAITERS_MODULE}.Live", _RecordingLive),
        patch(f"{WATCH_MODULE}.time.sleep"),
    ):
        assert waiters.wait_for_platform_job(jobs, "job-a", workspace="default") is True

    live = _RecordingLive.instances[0]
    assert isinstance(live.renderable, waiters._PlatformJobWaitLiveDisplay)
    assert live.updates == [live.renderable, live.renderable]
    assert jobs.get_job_status.call_count == 3


def test_wait_for_inference_deployment_uses_remaining_timeout_for_gateway(gateway_wait: MagicMock) -> None:
    client = _scripted_client([httpx.Response(200, json=_deployment_json("READY"))])

    with patch(f"{WAITERS_MODULE}.time.monotonic", side_effect=[100.0, 104.0, 104.0, 104.0]):
        assert waiters.wait_for_inference_deployment(
            client,
            "deployment-a",
            workspace="default",
            timeout=10,
            poll_interval=2,
        )

    gateway_wait.assert_called_once()
    assert gateway_wait.call_args.args[:3] == (client, "deployment-a", "default")
    assert gateway_wait.call_args.kwargs["timeout"] == pytest.approx(6.0)
    assert gateway_wait.call_args.kwargs["poll_interval"] == 2


def test_wait_for_inference_deployment_uses_model_provider_id_for_gateway(gateway_wait: MagicMock) -> None:
    client = _scripted_client(
        [httpx.Response(200, json=_deployment_json("READY", model_provider_id="provider-workspace/generated-provider"))]
    )

    with patch(f"{WAITERS_MODULE}.time.monotonic", side_effect=[100.0, 104.0, 104.0, 104.0]):
        assert waiters.wait_for_inference_deployment(
            client,
            "deployment-a",
            workspace="default",
            timeout=10,
            poll_interval=2,
        )

    assert gateway_wait.call_args.args[:3] == (client, "generated-provider", "provider-workspace")


def test_wait_for_inference_deployment_quiet_mode_uses_quiet_gateway(gateway_wait: MagicMock) -> None:
    client = _scripted_client([httpx.Response(200, json=_deployment_json("READY"))])

    with patch(f"{WAITERS_MODULE}.time.monotonic", side_effect=[100.0, 104.0, 104.0, 104.0]):
        assert waiters.wait_for_inference_deployment(
            client,
            "deployment-a",
            workspace="default",
            timeout=10,
            poll_interval=2,
            verbose=False,
        )

    assert gateway_wait.call_args.kwargs["verbose"] is False


def test_wait_for_inference_deployment_retries_transient_status_error(
    frozen_time: MagicMock, waiter_pause: MagicMock, gateway_wait: MagicMock
) -> None:
    client = _scripted_client(
        [
            httpx.ConnectError("connection refused"),
            httpx.Response(200, json=_deployment_json("READY")),
        ]
    )

    assert waiters.wait_for_inference_deployment(
        client,
        "deployment-a",
        workspace="default",
        timeout=10,
        poll_interval=1,
    )

    frozen_time.assert_called()
    waiter_pause.assert_called_once_with(1)
    gateway_wait.assert_called_once()


def test_wait_for_inference_deployment_returns_false_on_error_status(frozen_time: MagicMock) -> None:
    client = _scripted_client([httpx.Response(200, json=_deployment_json("ERROR", status_message="boom"))])

    assert (
        waiters.wait_for_inference_deployment(
            client,
            "deployment-a",
            workspace="default",
            timeout=10,
            poll_interval=2,
        )
        is False
    )
    frozen_time.assert_called()


def test_wait_for_inference_deployment_does_not_sleep_past_timeout(waiter_pause: MagicMock) -> None:
    client = _scripted_client([httpx.Response(200, json=_deployment_json("PENDING"))] * 3)

    with (
        patch(f"{WAITERS_MODULE}.time.monotonic", side_effect=[0.0, 0.0, 0.0, 4.0, 5.0, 5.0]),
    ):
        assert (
            waiters.wait_for_inference_deployment(
                client,
                "deployment-a",
                workspace="default",
                timeout=5,
                poll_interval=10,
                check_gateway=False,
            )
            is False
        )

    waiter_pause.assert_called_once_with(1.0)


def test_wait_for_platform_job_does_not_sleep_past_timeout() -> None:
    jobs = MagicMock()
    jobs.get_job_status.return_value = _status_response("active")

    # ``time.monotonic`` is one global attribute, so script it per caller: the
    # waiter's own elapsed reads stay at 0 while the watch loop sees the clock
    # advance to 4s and then 5s of its 5s deadline.
    watch_clock = iter([0.0, 0.0, 4.0, 5.0])

    def monotonic() -> float:
        caller = sys._getframe(1).f_globals["__name__"]
        return next(watch_clock) if caller == WATCH_MODULE else 0.0

    with (
        patch("time.monotonic", monotonic),
        patch(f"{WATCH_MODULE}.time.sleep") as watch_sleep,
    ):
        assert waiters.wait_for_platform_job(jobs, "job-a", workspace="default", timeout=5, poll_interval=10) is False

    watch_sleep.assert_called_once_with(1.0)


def test_wait_for_platform_job_retries_transient_status_error(frozen_time: MagicMock) -> None:
    jobs = MagicMock()
    request = httpx.Request("GET", "http://test")
    response = httpx.Response(503, request=request)
    jobs.get_job_status.side_effect = [
        InternalServerError(response),
        _status_response("completed"),
    ]

    with patch(f"{WATCH_MODULE}.time.sleep") as watch_sleep:
        assert waiters.wait_for_platform_job(jobs, "job-a", workspace="default", timeout=10, poll_interval=1) is True

    frozen_time.assert_called()
    watch_sleep.assert_called_once_with(1)


def test_wait_for_gateway_does_not_sleep_past_timeout(waiter_pause: MagicMock) -> None:
    client = _scripted_client([httpx.ConnectError("connection refused")] * 3)

    with patch(f"{WAITERS_MODULE}.time.monotonic", side_effect=[0.0, 0.0, 0.0, 4.0, 5.0, 5.0]):
        assert (
            waiters.wait_for_gateway(
                client,
                "provider-a",
                workspace="default",
                timeout=5,
                poll_interval=10,
            )
            is False
        )

    waiter_pause.assert_called_once_with(1.0)


def test_wait_for_gateway_returns_false_on_non_transient_status_error(frozen_time: MagicMock) -> None:
    client = _scripted_client([httpx.Response(401, json={"detail": "unauthorized"})])

    assert waiters.wait_for_gateway(client, "provider-a", workspace="default") is False
    frozen_time.assert_called()


def test_wait_for_gateway_returns_true_when_provider_ready(frozen_time: MagicMock) -> None:
    client = _scripted_client(
        [
            httpx.Response(404, json={"detail": "Model provider not found for default/provider-a"}),
            httpx.Response(503, json={"detail": "warming up"}),
            httpx.Response(200, json={"workspace": "default", "name": "provider-a"}),
        ]
    )

    with patch(f"{WAITERS_MODULE}._sleep_until_next_poll", return_value=True):
        assert waiters.wait_for_gateway(client, "provider-a", workspace="default") is True
    frozen_time.assert_called()


def test_wait_for_gateway_reraises_unexpected_errors(frozen_time: MagicMock) -> None:
    client = _scripted_client([RuntimeError("boom")])

    with pytest.raises(RuntimeError, match="boom"):
        waiters.wait_for_gateway(client, "provider-a", workspace="default")
    frozen_time.assert_called()


def test_sleep_until_next_poll_rejects_non_positive_poll_interval() -> None:
    with pytest.raises(ValueError, match=r"_sleep_until_next_poll.*poll_interval"):
        waiters._sleep_until_next_poll(0.0, 10.0, 0)
