# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The driver-side sweep that destroys a job's sandboxes on cancel or exit.

The actor's own teardown does not run when Ray kills that worker, which is what
a cancel does. These tests invoke the hooks directly: delivering a real signal
would take the session down with the handler's re-raise.
"""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable, Mapping
from typing import Any

import pytest
import sandboxed_gym.orchestrator as orchestrator_module
from sandboxed_gym.config import JOB_ID_METADATA_KEY
from sandboxed_gym.job_reaper import reap_job_sandboxes
from sandboxed_gym.orchestrator import (
    _INSTALLED,
    _TERMINATION_SHUTDOWNS,
    TERMINATION_SIGNALS,
    install_job_sandbox_reaper,
)


@pytest.fixture(autouse=True)
def restore_process_state(monkeypatch: pytest.MonkeyPatch):
    """Undo the process-global handlers, atexit hooks, and the once-per-job guard."""
    original = {signum: signal.getsignal(signum) for signum in TERMINATION_SIGNALS}
    # Tests replace signal.signal through the same monkeypatch, which is undone only after this
    # teardown, so restore the handlers with the real function.
    set_handler = signal.signal
    registered: list[Callable[..., object]] = []
    installed = set(_INSTALLED)
    prior_shutdowns = list(_TERMINATION_SHUTDOWNS)
    _TERMINATION_SHUTDOWNS.clear()

    # Record the hooks instead of registering them, so the test process never runs them at exit.
    monkeypatch.setattr(orchestrator_module, "register_atexit", registered.append)
    try:
        yield registered
    finally:
        for signum, handler in original.items():
            set_handler(signum, handler)
        _INSTALLED.clear()
        _INSTALLED.update(installed)
        _TERMINATION_SHUTDOWNS.clear()
        _TERMINATION_SHUTDOWNS.extend(prior_shutdowns)


class _Driver:
    """A fake OpenSandbox driver that records the metadata it is asked to match."""

    def __init__(self, *, connection: Mapping[str, str] | None = None, **kwargs: object) -> None:
        self.connection = dict(connection or {})
        self.metadata: dict[str, str] = {}
        self.error: BaseException | None = None

    async def destroy_sandboxes_matching(self, metadata: Mapping[str, str]) -> tuple[str, ...]:
        """Hang long past any caller timeout."""
        self.metadata = dict(metadata)
        if self.error is not None:
            raise self.error
        return ("sbx-1", "sbx-2")


def test_reap_destroys_every_sandbox_labeled_with_the_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancel has no actor handle left. The sweep is by the job id on the BatchSandbox."""
    created: list[_Driver] = []

    def _build(*, connection: Mapping[str, str] | None = None, **kwargs: object) -> _Driver:
        driver = _Driver(connection=connection, **kwargs)
        created.append(driver)
        return driver

    monkeypatch.setattr("sandboxed_gym.backends._opensandbox_driver.OpenSandboxDriver", _build)

    removed = reap_job_sandboxes(
        "rl-e6b8b524a645",
        host_provider="opensandbox",
        host_provider_options={"connection": {"protocol": "http"}},
    )

    assert removed == ("sbx-1", "sbx-2")
    assert created[0].metadata == {JOB_ID_METADATA_KEY: "rl-e6b8b524a645"}
    assert created[0].connection["protocol"] == "http"
    assert created[0].connection["use_server_proxy"] is True


def test_reap_logs_a_control_plane_failure_and_returns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A sweep that raises must not turn process exit into a hung teardown."""

    def _build(*, connection: Mapping[str, str] | None = None, **kwargs: object) -> _Driver:
        driver = _Driver(connection=connection)
        driver.error = RuntimeError("opensandbox unreachable")
        return driver

    monkeypatch.setattr("sandboxed_gym.backends._opensandbox_driver.OpenSandboxDriver", _build)

    with caplog.at_level("ERROR"):
        assert reap_job_sandboxes("rl-1", host_provider="opensandbox") == ()

    assert "failed to reap sandboxes for job rl-1" in caplog.text


def test_reap_gives_up_after_the_callers_timeout(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A caller inside a termination grace period bounds the sweep so its own teardown still runs."""

    class _Hung(_Driver):
        """A driver whose sweep never finishes."""

        async def destroy_sandboxes_matching(self, metadata: Mapping[str, str]) -> tuple[str, ...]:
            """Hang long past any caller timeout."""
            await asyncio.sleep(60)
            return ()

    monkeypatch.setattr(
        "sandboxed_gym.backends._opensandbox_driver.OpenSandboxDriver",
        lambda *, connection=None, **kwargs: _Hung(connection=connection),
    )

    with caplog.at_level("ERROR"):
        assert reap_job_sandboxes("rl-1", host_provider="opensandbox", timeout_s=0.05) == ()

    assert "failed to reap sandboxes for job rl-1" in caplog.text


def test_reap_logs_an_unknown_provider_and_returns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("ERROR"):
        assert reap_job_sandboxes("rl-1", host_provider="no-such-provider") == ()

    assert "failed to reap sandboxes for job rl-1" in caplog.text


class _FakeHostProvider:
    def __init__(self) -> None:
        self.destroyed: list[str] = []

    async def destroy_job_sandboxes(self, job_id: str) -> tuple[str, ...]:
        await asyncio.sleep(0)
        self.destroyed.append(job_id)
        return ("gym-host-1",)


def test_reap_completes_when_called_inside_a_running_event_loop(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A SIGINT to ``serve --mode orchestrator`` runs the reaper while Uvicorn's loop is running.

    ``asyncio.run`` refuses to nest there, so the destroy was never awaited and the Gym host
    container outlived the CLI.
    """
    provider = _FakeHostProvider()

    def _get(name: str, options: Mapping[str, Any] | None = None) -> _FakeHostProvider:
        return provider

    monkeypatch.setattr("sandboxed_gym.job_reaper.get_host_provider", _get)

    async def _as_uvicorn_would() -> tuple[str, ...]:
        return reap_job_sandboxes("rl-1", host_provider="docker")

    with caplog.at_level("ERROR"):
        removed = asyncio.run(_as_uvicorn_would())

    assert removed == ("gym-host-1",)
    assert provider.destroyed == ["rl-1"]
    assert "failed to reap" not in caplog.text


def test_exit_and_cancel_both_reap(monkeypatch: pytest.MonkeyPatch, restore_process_state) -> None:
    """atexit covers a failed process. SIGTERM covers a cancel, which does not run atexit."""
    calls: list[str] = []
    monkeypatch.setattr(orchestrator_module, "kill", lambda pid, sig: calls.append(f"kill:{sig}"))
    monkeypatch.setattr(
        "sandboxed_gym.job_reaper.reap_job_sandboxes",
        lambda job_id, **kwargs: calls.append(job_id) or (),
    )

    install_job_sandbox_reaper(
        "rl-1",
        host_provider="opensandbox",
        host_provider_options={"connection": {"protocol": "http"}},
    )
    install_job_sandbox_reaper("rl-1", host_provider="opensandbox")

    assert restore_process_state, "nothing was registered with atexit"
    for hook in restore_process_state:
        hook()

    handler = signal.getsignal(signal.SIGTERM)
    assert not isinstance(handler, int) and handler is not None
    handler(signal.SIGTERM, None)

    assert calls == ["rl-1", "rl-1", "kill:15"]
