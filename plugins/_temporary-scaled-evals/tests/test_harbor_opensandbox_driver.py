# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenSandboxDriver against a mock OpenSandbox SDK."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("scaled_evals", reason="scaled-evals plugin not installed")

from opensandbox_mocks import SandboxApiException, Server, execution, mock_sdk
from scaled_evals.harbor_opensandbox_driver import (
    CREATE_ATTEMPT_METADATA_KEY,
    CREATE_ATTEMPTS,
    Connection,
    CreateRequest,
    OpenSandboxDriver,
    harbor_backoff_sec,
)


class SandboxReadyTimeoutException(Exception):
    """Same name as the SDK's transient readiness error."""


@pytest.fixture
def server() -> Server:
    """A fresh mock OpenSandbox server."""
    return Server()


def _driver(server: Server, **kwargs: Any) -> OpenSandboxDriver:
    """A driver on the mock server that doesn't wait between create attempts."""
    return OpenSandboxDriver(
        Connection(domain="opensandbox.example", api_key="", request_timeout_sec=90),
        sdk=mock_sdk(server),
        retry_wait=lambda attempt: 0,
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
    sandbox = await _driver(server).create(_request())

    sent = server.creates[-1]
    assert sent["image"] == "registry.example/hello:1"
    assert sent["env"] == {"A": "1"}
    assert sent["resource"] == {"cpu": "2"}
    assert sent["network_policy"].payload == {"defaultAction": "deny", "egress": []}
    assert sent["skip_health_check"] is True
    assert sent["connection_config"].request_timeout.total_seconds() == 90
    assert sent["metadata"]["owner"] == "eval-1"
    assert len(sent["metadata"][CREATE_ATTEMPT_METADATA_KEY]) == 32
    assert server.live() == {sandbox.id}


async def test_transient_errors_are_retried_and_orphans_killed(server: Server) -> None:
    server.create_failures.append(SandboxApiException("lost response"))

    sandbox = await _driver(server).create(_request())

    assert len(server.creates) == 2
    labels = {create["metadata"][CREATE_ATTEMPT_METADATA_KEY] for create in server.creates}
    assert len(labels) == 1
    assert server.live() == {sandbox.id}


async def test_gives_up_after_the_last_attempt_and_kills_everything(server: Server) -> None:
    server.create_failures.extend(SandboxApiException("lost response") for _ in range(CREATE_ATTEMPTS))

    with pytest.raises(SandboxApiException):
        await _driver(server).create(_request())

    assert len(server.creates) == CREATE_ATTEMPTS
    assert server.live() == set()


async def test_input_errors_are_not_retried(server: Server) -> None:
    server.create_failures.append(ValueError("bad request"))

    with pytest.raises(ValueError, match="bad request"):
        await _driver(server).create(_request())

    assert len(server.creates) == 1
    assert server.live() == set()


async def test_a_rejected_create_leaves_nothing(server: Server) -> None:
    server.rejections.append(ValueError("invalid image"))

    with pytest.raises(ValueError, match="invalid image"):
        await _driver(server).create(_request())

    assert server.sandboxes == {}


async def test_a_terminal_state_fails_fast_with_the_server_reason(server: Server) -> None:
    server.states = ["Pending", "Failed"]
    server.healthy = False

    with pytest.raises(RuntimeError, match="entered Failed before becoming ready: image pull failed"):
        await _driver(server).create(_request())

    assert len(server.creates) == 1
    assert server.live() == set()


async def test_a_failure_check_fails_the_create(server: Server) -> None:
    async def failing(sandbox: Any) -> str | None:
        """Report a failure on the first poll, as a crashed service would."""
        return f"{sandbox.id}: service crashed"

    with pytest.raises(RuntimeError, match="sbx-1: service crashed"):
        await _driver(server).create(_request(), failure_check=failing)

    assert server.live() == set()


async def test_probe_errors_keep_waiting_instead_of_recreating(server: Server) -> None:
    server.health_errors = 2

    sandbox = await _driver(server).create(_request())

    assert len(server.creates) == 1
    assert server.live() == {sandbox.id}


async def test_an_unready_sandbox_times_out_and_is_killed(server: Server) -> None:
    server.healthy = False

    with pytest.raises(TimeoutError, match="did not become ready within 0.01s"):
        await _driver(server).create(_request(ready_timeout_sec=0.01))

    assert server.live() == set()


async def test_transient_readiness_errors_retry_the_create(server: Server, monkeypatch: pytest.MonkeyPatch) -> None:
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
    server.sandboxes = {"a": {"x": "1"}, "b": {"x": "1"}, "c": {"x": "2"}}

    killed = await _driver(server).kill_matching({"x": "1"}, keep="a")

    assert killed == ["b"]
    assert server.live() == {"a", "c"}


async def test_kill_matching_refuses_an_empty_selector(server: Server) -> None:
    with pytest.raises(ValueError, match="at least one label"):
        await _driver(server).kill_matching({})


async def test_run_passes_options_and_joins_output(server: Server) -> None:
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
    driver = _driver(server)
    sandbox = await driver.create(_request())

    await driver.run(sandbox, "whoami")

    assert "uid" not in sandbox.commands.runs[-1][1]


@pytest.mark.parametrize(("error", "expected"), [(None, 0), ("boom", 1)])
async def test_run_infers_a_missing_exit_code(server: Server, error: str | None, expected: int) -> None:
    driver = _driver(server)
    sandbox = await driver.create(_request())
    sandbox.commands.results["x"] = execution(exit_code=None, error=error)

    assert (await driver.run(sandbox, "x")).exit_code == expected


async def test_run_keeps_line_breaks_including_blank_lines(server: Server) -> None:
    """execd strips each line's ending and sends a blank line as "\\n", as seen on a live server."""
    driver = _driver(server)
    sandbox = await driver.create(_request())
    sandbox.commands.results["printf"] = execution(stdout=["one", "\n", "three"], stderr=["e1", "e2"])

    result = await driver.run(sandbox, "printf")

    assert (result.stdout, result.stderr) == ("one\n\nthree", "e1\ne2")


def test_connection_repr_hides_the_api_key() -> None:
    connection = Connection(domain="opensandbox.example", api_key="secret-key")

    assert "secret-key" not in repr(connection)
    assert "opensandbox.example" in repr(connection)


def test_connection_config_keeps_the_sdk_timeout_when_unset(server: Server) -> None:
    driver = OpenSandboxDriver(Connection(domain="d", api_key="k"), sdk=mock_sdk(server))

    config = driver.connection_config()

    assert not hasattr(config, "request_timeout")
    assert (config.domain, config.api_key, config.protocol, config.use_server_proxy) == ("d", "k", "https", False)


def test_default_backoff_matches_harbor() -> None:
    assert [harbor_backoff_sec(attempt) for attempt in range(1, 6)] == [1.0, 2.0, 4.0, 8.0, 10.0]


async def test_kill_matching_walks_every_page(server: Server) -> None:
    server.page_size = 2
    server.sandboxes = {f"s{n}": {"x": "1"} for n in range(5)}

    killed = await _driver(server).kill_matching({"x": "1"})

    assert sorted(killed) == [f"s{n}" for n in range(5)]
    assert server.live() == set()


async def test_kill_matching_continues_past_a_failed_kill(server: Server) -> None:
    server.sandboxes = {"a": {"x": "1"}, "b": {"x": "1"}}
    server.unkillable = {"a"}

    killed = await _driver(server).kill_matching({"x": "1"})

    assert killed == ["b"]
    assert server.live() == {"a"}


async def test_safe_kill_never_raises_and_still_closes(server: Server) -> None:
    driver = _driver(server)
    sandbox = await driver.create(_request())
    server.unkillable = {sandbox.id}

    await driver.safe_kill(sandbox)

    assert sandbox.closed
    assert server.live() == {sandbox.id}
