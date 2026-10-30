# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``OpenSandboxDriver``.

They run against ``opensandbox_mocks``, an in-memory OpenSandbox SDK and server, so they need
neither a server nor the SDK. The mock records every create and kill, which is how the tests check
that no sandbox is left running after a failed, retried or cancelled create.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

import pytest
from nhx_sandbox.core.errors import ExecInterrupted
from nhx_sandbox.harbor_opensandbox_driver import (
    CREATE_ATTEMPT_METADATA_KEY,
    CREATE_ATTEMPTS,
    Connection,
    CreateRequest,
    OpenSandboxDriver,
    SandboxExpiredError,
    harbor_backoff_sec,
)
from opensandbox_mocks import SandboxApiException, Server, execution, mock_sdk


class SandboxReadyTimeoutException(Exception):
    """Same name as the SDK's transient readiness error."""


@pytest.fixture
def server() -> Server:
    """A fresh mock OpenSandbox server."""
    return Server()


def _driver(server: Server, *, retry_wait: float = 0, **kwargs: Any) -> OpenSandboxDriver:
    """A driver on the mock server that waits ``retry_wait`` seconds between create attempts."""
    return OpenSandboxDriver(
        Connection(domain="opensandbox.example", api_key="", request_timeout_sec=90),
        sdk=mock_sdk(server),
        retry_wait=lambda attempt: retry_wait,
        **kwargs,
    )


def _request(**overrides: Any) -> CreateRequest:
    """A small valid create request; ``overrides`` replace individual fields."""
    values: dict[str, Any] = {
        "image": "registry.example/hello:1",
        "env": {"A": "1"},
        "metadata": {"owner": "eval-1"},
        "resource": {"cpu": "2"},
        "network_policy": {"defaultAction": "deny", "egress": []},
        "extensions": {},
        "sandbox_timeout_sec": 3600,
        "ready_timeout_sec": 5,
        "health_check_poll_interval_sec": 0,
    }
    return CreateRequest(**{**values, **overrides})


async def test_create_sends_the_request_and_labels_the_attempt(server: Server) -> None:
    """Create sends every request field, skips the SDK's health check, and labels the attempt."""
    sandbox = await _driver(server).create(_request())

    sent = server.creates[-1]
    assert sent["image"] == "registry.example/hello:1"
    assert sent["env"] == {"A": "1"}
    assert sent["resource"] == {"cpu": "2"}
    assert sent["network_policy"].payload == {"defaultAction": "deny", "egress": []}
    assert sent["skip_health_check"] is True
    assert sent["entrypoint"] is None
    assert sent["connection_config"].request_timeout.total_seconds() == 90
    assert sent["metadata"]["owner"] == "eval-1"
    assert len(sent["metadata"][CREATE_ATTEMPT_METADATA_KEY]) == 32
    assert server.live() == {sandbox.id}


async def test_create_sends_an_entrypoint_and_extensions(server: Server) -> None:
    """An entrypoint replaces the server's default command; extensions pass through unchanged."""
    await _driver(server).create(
        _request(entrypoint=("/app/start.sh", "sh", "-c", "sleep infinity"), extensions={"nemo.ext": "{}"})
    )

    sent = server.creates[-1]
    assert sent["entrypoint"] == ["/app/start.sh", "sh", "-c", "sleep infinity"]
    assert sent["extensions"] == {"nemo.ext": "{}"}


async def test_transient_errors_are_retried_and_orphans_killed(server: Server) -> None:
    """A lost response is retried under the same label, and the sandbox it orphaned is killed."""
    server.create_failures.append(SandboxApiException("lost response"))

    sandbox = await _driver(server).create(_request())

    assert len(server.creates) == 2
    labels = {create["metadata"][CREATE_ATTEMPT_METADATA_KEY] for create in server.creates}
    assert len(labels) == 1
    assert server.live() == {sandbox.id}


async def test_each_retry_gets_only_the_time_left_before_the_expiry(
    server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lifetime is cut again before every attempt, so a retried sandbox doesn't outlive ``expires_at`` by
    the time the failed attempts took."""
    expires_at = datetime(2026, 10, 23, 13, tzinfo=UTC)
    now = [expires_at.timestamp() - 100]
    monkeypatch.setattr(time, "time", lambda: now[0])
    server.create_failures.append(SandboxApiException("timed out"))

    def a_minute_later(attempt: int) -> float:
        now[0] += 60  # the failed attempt and its backoff
        return 0

    driver = OpenSandboxDriver(
        Connection(domain="opensandbox.example", api_key="", request_timeout_sec=90),
        sdk=mock_sdk(server),
        retry_wait=a_minute_later,
    )
    await driver.create(_request(expires_at=expires_at))

    assert [create["timeout"].total_seconds() for create in server.creates] == [100, 40]


async def test_a_retry_after_the_expiry_is_not_sent(server: Server, monkeypatch: pytest.MonkeyPatch) -> None:
    """A retry that would start past ``expires_at`` fails the create instead of making a one-second sandbox."""
    expires_at = datetime(2026, 10, 23, 13, tzinfo=UTC)
    now = [expires_at.timestamp() - 30]
    monkeypatch.setattr(time, "time", lambda: now[0])
    server.create_failures.append(SandboxApiException("timed out"))

    def a_minute_later(attempt: int) -> float:
        now[0] += 60
        return 0

    driver = OpenSandboxDriver(
        Connection(domain="opensandbox.example", api_key="", request_timeout_sec=90),
        sdk=mock_sdk(server),
        retry_wait=a_minute_later,
    )
    with pytest.raises(SandboxExpiredError):
        await driver.create(_request(expires_at=expires_at))

    assert [create["timeout"].total_seconds() for create in server.creates] == [30]
    assert server.live() == set()


async def test_gives_up_after_the_last_attempt_and_kills_everything(server: Server) -> None:
    """After the last attempt fails, the error surfaces and every sandbox the attempts made is killed."""
    server.create_failures.extend(SandboxApiException("lost response") for _ in range(CREATE_ATTEMPTS))

    with pytest.raises(SandboxApiException):
        await _driver(server).create(_request())

    assert len(server.creates) == CREATE_ATTEMPTS
    assert server.live() == set()


async def test_input_errors_are_not_retried(server: Server) -> None:
    """A non-transient error fails the create at once, and its sandbox is still killed."""
    server.create_failures.append(ValueError("bad request"))

    with pytest.raises(ValueError, match="bad request"):
        await _driver(server).create(_request())

    assert len(server.creates) == 1
    assert server.live() == set()


async def test_a_rejected_create_leaves_nothing(server: Server) -> None:
    """A create the server rejects outright makes no sandbox."""
    server.rejections.append(ValueError("invalid image"))

    with pytest.raises(ValueError, match="invalid image"):
        await _driver(server).create(_request())

    assert server.sandboxes == {}


async def test_cancelling_during_retry_backoff_kills_earlier_attempts(server: Server) -> None:
    """Cancelling while waiting to retry still kills the sandbox a lost response left behind."""
    server.create_failures.append(SandboxApiException("lost response"))
    task = asyncio.create_task(_driver(server, retry_wait=30).create(_request()))
    # Let the first attempt fail; the driver then sleeps before retrying.
    await asyncio.sleep(0.1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(server.creates) == 1
    assert server.live() == set()


async def test_cancelling_during_the_readiness_wait_kills_the_sandbox(server: Server) -> None:
    """Cancelling while the sandbox is still starting kills it."""
    server.healthy = False
    task = asyncio.create_task(
        _driver(server).create(_request(ready_timeout_sec=30, health_check_poll_interval_sec=0.01))
    )
    await asyncio.sleep(0.1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(server.creates) == 1
    assert server.live() == set()


async def test_a_terminal_state_fails_fast_with_the_server_reason(server: Server) -> None:
    """A sandbox that enters Failed fails the create at once, with the server's message."""
    server.states = ["Pending", "Failed"]
    server.healthy = False

    with pytest.raises(RuntimeError, match="entered Failed before becoming ready: image pull failed"):
        await _driver(server).create(_request())

    assert len(server.creates) == 1
    assert server.live() == set()


async def test_a_failure_check_fails_the_create(server: Server) -> None:
    """A failure the caller's check reports fails the create and kills the sandbox."""

    async def failing(sandbox: Any) -> str | None:
        """Report a failure on the first poll, as a crashed service would."""
        return f"{sandbox.id}: service crashed"

    with pytest.raises(RuntimeError, match="sbx-1: service crashed"):
        await _driver(server).create(_request(), failure_check=failing)

    assert server.live() == set()


async def test_probe_errors_keep_waiting_instead_of_recreating(server: Server) -> None:
    """Health-probe errors count as not ready yet; they don't trigger a second create."""
    server.health_errors = 2

    sandbox = await _driver(server).create(_request())

    assert len(server.creates) == 1
    assert server.live() == {sandbox.id}


async def test_an_unready_sandbox_times_out_and_is_killed(server: Server) -> None:
    """A sandbox that never becomes healthy times out and is killed."""
    server.healthy = False

    with pytest.raises(TimeoutError, match="did not become ready within 0.01s"):
        await _driver(server).create(_request(ready_timeout_sec=0.01))

    assert server.live() == set()


async def test_transient_readiness_errors_retry_the_create(server: Server, monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient error from the readiness wait retries the whole create."""
    driver = _driver(server)
    original = driver._wait_until_ready
    calls = 0

    async def flaky(sandbox: Any, request: CreateRequest, failure_check: Any) -> None:
        """Fail the first readiness wait with a transient error, then wait normally."""
        nonlocal calls
        calls += 1
        if calls == 1:
            raise SandboxReadyTimeoutException("not ready")
        await original(sandbox, request, failure_check)

    monkeypatch.setattr(driver, "_wait_until_ready", flaky)

    sandbox = await driver.create(_request())

    assert len(server.creates) == 2
    assert server.live() == {sandbox.id}


async def test_kill_matching_keeps_the_named_sandbox(server: Server) -> None:
    """Only sandboxes with matching labels are killed, and ``keep`` is spared."""
    server.sandboxes = {"a": {"x": "1"}, "b": {"x": "1"}, "c": {"x": "2"}}

    killed = await _driver(server).kill_matching({"x": "1"}, keep="a")

    assert killed == ["b"]
    assert server.live() == {"a", "c"}


async def test_kill_matching_refuses_an_empty_selector(server: Server) -> None:
    """An empty selector would match every sandbox on the server, so it is refused."""
    with pytest.raises(ValueError, match="at least one label"):
        await _driver(server).kill_matching({})


async def test_kill_matching_walks_every_page(server: Server) -> None:
    """Matches on every page of the listing are killed, not just the first page."""
    server.page_size = 2
    server.sandboxes = {f"s{n}": {"x": "1"} for n in range(5)}

    killed = await _driver(server).kill_matching({"x": "1"})

    assert sorted(killed) == [f"s{n}" for n in range(5)]
    assert server.live() == set()


async def test_kill_matching_continues_past_a_failed_kill(server: Server) -> None:
    """One failed kill doesn't stop the rest, and isn't reported as killed."""
    server.sandboxes = {"a": {"x": "1"}, "b": {"x": "1"}}
    server.unkillable = {"a"}

    killed = await _driver(server).kill_matching({"x": "1"})

    assert killed == ["b"]
    assert server.live() == {"a"}


async def test_safe_kill_never_raises_and_still_closes(server: Server) -> None:
    """``safe_kill`` swallows a failed kill and still releases the client."""
    driver = _driver(server)
    sandbox = await driver.create(_request())
    server.unkillable = {sandbox.id}

    await driver.safe_kill(sandbox)

    assert sandbox.closed
    assert server.live() == {sandbox.id}


async def test_run_passes_options_and_joins_output(server: Server) -> None:
    """``run`` sends cwd, env, timeout and uid, and returns both streams and the exit code."""
    driver = _driver(server)
    sandbox = await driver.create(_request())
    sandbox.commands.results["make"] = execution(stdout=["a", "b"], stderr=["warn"], exit_code=2)

    result = await driver.run(sandbox, "make", cwd="/app", env={"K": "v"}, timeout_sec=30, uid=0)

    assert (result.stdout, result.stderr, result.exit_code) == ("a\nb", "warn", 2)
    _, opts = sandbox.commands.runs[-1]
    assert opts["working_directory"] == "/app"
    assert opts["envs"] == {"K": "v"}
    assert opts["timeout"].total_seconds() == 30
    assert opts["uid"] == 0


async def test_run_without_a_uid_sends_none(server: Server) -> None:
    """Without a uid, none is sent, so the command runs as the image's default user."""
    driver = _driver(server)
    sandbox = await driver.create(_request())

    await driver.run(sandbox, "whoami")

    assert "uid" not in sandbox.commands.runs[-1][1]


async def test_run_reports_an_error_without_an_exit_code_as_1(server: Server) -> None:
    driver = _driver(server)
    sandbox = await driver.create(_request())
    sandbox.commands.results["x"] = execution(exit_code=None, error="boom")

    assert (await driver.run(sandbox, "x")).exit_code == 1


async def test_run_raises_when_the_output_ends_without_an_exit_code(server: Server) -> None:
    """What the SDK returns when the sandbox's container is OOM-killed or evicted under the command: it never succeeded."""
    driver = _driver(server)
    sandbox = await driver.create(_request())
    sandbox.commands.results["x"] = execution(stderr=["400+0 records out"], exit_code=None)

    with pytest.raises(ExecInterrupted, match="ended before it reported an exit code"):
        await driver.run(sandbox, "x")


async def test_run_keeps_line_breaks_including_blank_lines(server: Server) -> None:
    """execd strips each line's ending and sends a blank line as "\\n", as seen on a live server."""
    driver = _driver(server)
    sandbox = await driver.create(_request())
    sandbox.commands.results["printf"] = execution(stdout=["one", "\n", "three"], stderr=["e1", "e2"])

    result = await driver.run(sandbox, "printf")

    assert (result.stdout, result.stderr) == ("one\n\nthree", "e1\ne2")


def test_connection_repr_hides_the_api_key() -> None:
    """Printing or logging a ``Connection`` never shows the API key."""
    connection = Connection(domain="opensandbox.example", api_key="secret-key")

    assert "secret-key" not in repr(connection)
    assert "opensandbox.example" in repr(connection)


def test_connection_config_keeps_the_sdk_timeout_when_unset(server: Server) -> None:
    """Without a configured request timeout, the SDK's default is left alone."""
    driver = OpenSandboxDriver(Connection(domain="d", api_key="k"), sdk=mock_sdk(server))

    config = driver.connection_config()

    assert not hasattr(config, "request_timeout")
    assert (config.domain, config.api_key, config.protocol, config.use_server_proxy) == ("d", "k", "https", False)


def test_default_backoff_matches_harbor() -> None:
    """The default wait between create attempts doubles from 1 s and caps at 10 s, as in Harbor 0.20."""
    assert [harbor_backoff_sec(attempt) for attempt in range(1, 6)] == [1.0, 2.0, 4.0, 8.0, 10.0]
