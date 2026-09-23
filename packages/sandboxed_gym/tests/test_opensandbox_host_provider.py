# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readiness diagnostics of the OpenSandbox job-host provider.

These need no OpenSandbox SDK: the module keeps its SDK types under ``TYPE_CHECKING`` and builds
the driver lazily, so the provider imports and constructs from a plain checkout.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from email.message import Message
from typing import Any
from urllib.error import HTTPError

import pytest
from sandboxed_gym.host.models import (
    GymHostBootstrapFailed,
    GymHostHandle,
    GymHostSpec,
    GymHostVolumeMount,
    render_host_error,
)
from sandboxed_gym.host.opensandbox import OpenSandboxGymHostProvider

HEALTH_URL = "https://sandbox.example/gym-1/health"
ROLLOUT_URL = "https://sandbox.example/gym-1/rollouts/run"


def _raising(error: Exception):
    def _raise(*args: Any, **kwargs: Any):
        raise error

    return _raise


@pytest.fixture
def provider() -> OpenSandboxGymHostProvider:
    return OpenSandboxGymHostProvider(connection={"domain": "x", "api_key": "k"})


def test_readiness_timeout_names_the_url_it_polled(
    provider: OpenSandboxGymHostProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scheme or port mismatch is otherwise indistinguishable from a slow sandbox.

    The host id alone cannot be turned back into the URL the poll used, so a misconfigured
    protocol reads as "the sandbox is taking too long" and sends the reader to the wrong place.
    """
    handle = GymHostHandle(host_id="gym-1", health_url=HEALTH_URL, rollout_url=ROLLOUT_URL)

    def _refuse(url: str, headers: Any) -> dict[str, Any]:
        raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr(provider, "_get_json", _refuse)

    with pytest.raises(TimeoutError) as excinfo:
        asyncio.run(provider.wait_ready(handle, timeout_s=0.01))

    message = str(excinfo.value)
    assert HEALTH_URL in message
    # The underlying error too: knowing which URL was polled is only half of it.
    assert "connection refused" in message


def test_a_created_host_logs_the_urls_it_resolved(
    provider: OpenSandboxGymHostProvider,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``create_host`` is the only place the resolved URLs are knowable.

    They come back from the SDK's route resolution onto a handle the caller may never print, so
    without this every downstream connectivity failure is a timeout against an address that
    appears nowhere in the logs.
    """
    routes = _Routes(health_url=HEALTH_URL, rollout_url=ROLLOUT_URL, headers={})
    monkeypatch.setattr(provider, "_provider_for_spec", lambda spec: _StubDriver())
    monkeypatch.setattr(provider, "_to_sandbox_spec", lambda spec: spec)
    monkeypatch.setattr(provider, "_resolve_routes", _returning(routes))

    with caplog.at_level(logging.INFO, logger="sandboxed_gym.host.opensandbox"):
        asyncio.run(provider.create_host(_spec()))

    assert HEALTH_URL in caplog.text
    assert ROLLOUT_URL in caplog.text
    assert "gym-1" in caplog.text


class _Routes:
    def __init__(self, *, health_url: str, rollout_url: str, headers: dict[str, str]) -> None:
        self.health_url = health_url
        self.rollout_url = rollout_url
        self.headers = headers


class _ResourceHandle:
    sandbox_id = "gym-1"
    raw = object()


class _StubDriver:
    """Stands in for the OpenSandbox SDK driver, which is the only part needing a real cluster."""

    async def create(self, spec: GymHostSpec) -> _ResourceHandle:
        return _ResourceHandle()


def _returning(routes: _Routes):
    async def _resolve(raw: Any, port: int) -> _Routes:
        return routes

    return _resolve


def _spec() -> GymHostSpec:
    return GymHostSpec(
        job_id="job-1",
        runtime_image="example/gym:latest",
        environment_mount=GymHostVolumeMount(pvc_claim="env", mount_path="/job/environment", read_only=True),
        workspace_mount=GymHostVolumeMount(pvc_claim="work", mount_path="/job/work"),
    )


def test_a_host_that_reports_a_failed_bootstrap_stops_the_poll_immediately(
    provider: OpenSandboxGymHostProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bootstrap failure is terminal, so polling it to the readiness deadline reports a timeout.

    The reader then sees "did not become ready" for a host that answered on the first poll and
    said exactly what was wrong.
    """
    handle = GymHostHandle(host_id="gym-1", health_url=HEALTH_URL, rollout_url=ROLLOUT_URL)
    polls = 0

    def _failed(url: str, headers: Any) -> dict[str, Any]:
        nonlocal polls
        polls += 1
        return {
            "status": "failed",
            "error": {
                "code": "bootstrap_failed",
                "message": "the policy endpoint rejected the configured credential (HTTP 401)",
                "host_output_tail": ["Traceback (most recent call last):"],
            },
        }

    monkeypatch.setattr(provider, "_get_json", _failed)

    with pytest.raises(GymHostBootstrapFailed) as excinfo:
        asyncio.run(provider.wait_ready(handle, timeout_s=30))

    assert polls == 1
    message = str(excinfo.value)
    assert "gym-1" in message
    assert "rejected the configured credential (HTTP 401)" in message
    assert "Traceback (most recent call last):" in message


def test_a_503_health_response_keeps_the_body_it_carried(
    provider: OpenSandboxGymHostProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host reports a failed bootstrap as 503, so discarding that body discards the diagnosis."""
    body = json.dumps({"status": "failed", "error": {"message": "boom"}}).encode("utf-8")
    error = HTTPError(HEALTH_URL, 503, "Service Unavailable", Message(), io.BytesIO(body))
    monkeypatch.setattr("sandboxed_gym.host.opensandbox.urlopen", _raising(error))

    assert provider._get_json(HEALTH_URL, {}) == {"status": "failed", "error": {"message": "boom"}}


def test_a_503_health_response_without_a_body_still_reads_as_starting(
    provider: OpenSandboxGymHostProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sandbox proxy 503s before the host exists at all, with no JSON of its own."""
    error = HTTPError(HEALTH_URL, 503, "Service Unavailable", Message(), io.BytesIO(b"<html>no route</html>"))
    monkeypatch.setattr("sandboxed_gym.host.opensandbox.urlopen", _raising(error))

    assert provider._get_json(HEALTH_URL, {}) == {"status": "starting"}


def test_a_rendered_error_keeps_every_line_the_host_sent() -> None:
    """The host bounds the tail against its own response budget before sending it.

    A second, smaller bound here would silently drop diagnostics that already survived the wire,
    which is the opposite of what shipping them was for.
    """
    tail = [f"line {index}" for index in range(80)]

    rendered = render_host_error({"code": "bootstrap_failed", "host_output_tail": tail})

    assert "line 0" in rendered
    assert "line 79" in rendered
    assert "(80 lines)" in rendered
