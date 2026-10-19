# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The orchestrator proxy forwards a rollout body as it arrives, rather than after it completes.

The Gym host answers ``200`` before its batch finishes and pads the body with whitespace while it
works, so no hop mistakes a long rollout for a dead one. A proxy that read the response to
completion would absorb every one of those bytes and re-emit them at the end -- handing its own
caller exactly the silence the padding exists to prevent.
"""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.error
import urllib.request
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sandboxed_gym.config import BrokerEndpoint
from sandboxed_gym.host.models import GymHostHandle
from sandboxed_gym.orchestrator import SandboxedGymSession
from sandboxed_gym.proxy_app import _forward, build_proxy_app
from sandboxed_gym.serve_config import SandboxedGymServeConfig

ROLLOUT_BODY = {"results": [{"reward": 1.0}]}


def _session(**sandbox_overrides: Any) -> SandboxedGymSession:
    cfg = SandboxedGymServeConfig.model_validate(
        {
            "job_id": "job-1",
            "sandbox": {
                "image": "runtime:dev",
                "network_policy": {"egress_allow": []},
                "environment_pvc_claim": "env",
                "workspace_pvc_claim": "work",
                **sandbox_overrides,
            },
        }
    )
    return SandboxedGymSession(
        cfg=cfg,
        broker_server=MagicMock(),
        broker=BrokerEndpoint(url="http://broker:1", host="broker", port=1, token="t"),
        host_provider=MagicMock(),
        host=GymHostHandle(
            host_id="h1",
            health_url="http://host/health",
            rollout_url="http://host/rollouts/run",
            headers={},
        ),
    )


class _DrippingResponse:
    """An upstream that emits a heartbeat, stalls until released, then sends its payload."""

    status = 200
    headers = {"Content-Type": "application/json"}

    def __init__(self, release: threading.Event, payload: bytes) -> None:
        self._release = release
        self._chunks = [b" ", payload]
        self.closed = False

    def read1(self, size: int) -> bytes:
        if not self._chunks:
            return b""
        if len(self._chunks) == 1:
            # The payload is withheld until the test releases it, standing in for a batch still
            # running. Raising rather than proceeding on timeout: a consumer that read to
            # completion would otherwise just look slow instead of failing.
            if not self._release.wait(5):
                raise AssertionError("the payload was read before the batch was released")
        return self._chunks.pop(0)

    def close(self) -> None:
        self.closed = True


async def test_the_heartbeat_is_forwarded_before_the_payload_exists() -> None:
    """The property the whole change exists for.

    The first chunk has to come out while the upstream is still working on the rest. Asserted on
    the generator rather than through ``TestClient``: the ASGI test transport hands back the body
    as one piece, so a buffered and a streamed proxy are indistinguishable from that side.
    """
    release = threading.Event()
    payload = json.dumps(ROLLOUT_BODY).encode()
    chunks = _forward(_DrippingResponse(release, payload), max_response_bytes=1024)

    # Would block -- and then trip the stub's assertion -- against a proxy that read to completion.
    assert await anext(chunks) == b" "

    release.set()
    assert json.loads(b"".join([chunk async for chunk in chunks])) == ROLLOUT_BODY


def test_the_whole_body_still_round_trips_through_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streaming must not change what the caller ends up with, only when it starts arriving."""
    release = threading.Event()
    release.set()
    payload = json.dumps(ROLLOUT_BODY).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _DrippingResponse(release, payload))

    client = TestClient(build_proxy_app(_session()))
    response = client.post("/rollouts/run", json={"examples": [{"id": 1}]})

    assert response.status_code == 200
    assert response.json() == ROLLOUT_BODY


def test_the_upstream_status_still_decides_the_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opened before streaming starts, so a refusal is still a refusal and not a 200 with a body."""
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        _raising(urllib.error.HTTPError("http://host", 503, "nope", {}, None)),  # ty: ignore[invalid-argument-type]
    )

    client = TestClient(build_proxy_app(_session()))

    assert client.post("/rollouts/run", json={"examples": [{"id": 1}]}).status_code == 503


def test_an_upstream_that_never_connects_is_a_502(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exception text stays server-side; it can name the host's internal address."""
    monkeypatch.setattr(urllib.request, "urlopen", _raising(urllib.error.URLError("refused")))

    client = TestClient(build_proxy_app(_session()))
    response = client.post("/rollouts/run", json={"examples": [{"id": 1}]})

    assert response.status_code == 502
    assert "refused" not in response.text


def _raising(exc: BaseException):
    def _raise(*args: Any, **kwargs: Any):
        raise exc

    return _raise


async def test_an_oversize_body_fails_loudly_rather_than_truncating() -> None:
    """A truncated rollout body must not be able to look like a complete one.

    Ending the stream cleanly on overflow would send a 200 carrying a prefix -- and a prefix of
    ``{"results": [...]}`` can itself be valid JSON describing a *smaller* result set, which the
    SDK consumer would accept as a successful run that simply produced fewer rollouts. Aborting
    mid-body is what makes it a read error instead.
    """
    release = threading.Event()
    release.set()
    response = _DrippingResponse(release, b"x" * 100)

    with pytest.raises(RuntimeError, match="exceeded max_response_bytes"):
        async for _ in _forward(response, max_response_bytes=8):
            pass

    assert response.closed, "the upstream response was left open"


def test_a_prefix_of_a_results_body_would_have_parsed_as_a_smaller_batch() -> None:
    """Why the abort above matters, stated as the property that makes truncation dangerous."""
    full = b'{"results": [{"reward": 1.0}, {"reward": 0.0}]}'
    prefix = b'{"results": [{"reward": 1.0}]}'

    assert full.startswith(prefix[:20])
    # The prefix is not merely unparseable garbage -- it is a valid, wrong answer.
    assert json.loads(prefix) == {"results": [{"reward": 1.0}]}


async def test_the_upstream_is_closed_once_the_body_is_forwarded() -> None:
    """The generator owns the response, so nothing else is positioned to close it."""
    release = threading.Event()
    release.set()
    response = _DrippingResponse(release, b'{"results": []}')

    async for _ in _forward(response, max_response_bytes=1024):
        pass

    assert response.closed


async def test_a_disconnect_closes_an_upstream_still_blocked_in_a_read() -> None:
    """A client that leaves mid-batch must not pin the upstream connection open.

    This is why ``_forward`` is an async generator despite every read being blocking. Starlette
    runs a *sync* iterator in a worker thread, and cancellation cannot reach a thread parked in
    ``read1`` -- the cleanup would wait for the host's next byte, which for an abandoned rollout
    means the rest of the batch. Awaiting the read gives the cancellation somewhere to land.
    """
    release = threading.Event()
    response = _DrippingResponse(release, b'{"results": []}')
    chunks = _forward(response, max_response_bytes=1024)

    assert await anext(chunks) == b" "

    # Now blocked on the payload, exactly as it would be mid-rollout.
    pending = asyncio.ensure_future(anext(chunks))
    await asyncio.sleep(0.05)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await chunks.aclose()

    assert response.closed, "the upstream was left open after the caller disconnected"
    release.set()
