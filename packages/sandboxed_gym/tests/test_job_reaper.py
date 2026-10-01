# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The driver-side sweep that destroys a job's sandboxes on cancel or exit.

The actor's own teardown does not run when Ray kills that worker, which is what
a cancel does. These tests invoke the hooks directly: delivering a real signal
would take the session down with the handler's re-raise.
"""

from __future__ import annotations

import atexit
import signal
from collections.abc import Callable, Mapping

import pytest
from sandboxed_gym.config import JOB_ID_METADATA_KEY
from sandboxed_gym.job_reaper import reap_job_sandboxes
from sandboxed_gym.orchestrator import (
    _INSTALLED,
    _TERMINATION_SHUTDOWNS,
    TERMINATION_SIGNALS,
    install_job_sandbox_reaper,
)


@pytest.fixture(autouse=True)
def restore_process_state():
    """Undo the process-global handlers, atexit hooks, and the once-per-job guard."""
    original = {signum: signal.getsignal(signum) for signum in TERMINATION_SIGNALS}
    registered: list[Callable[..., object]] = []
    real_register = atexit.register
    installed = set(_INSTALLED)
    prior_shutdowns = list(_TERMINATION_SHUTDOWNS)
    _TERMINATION_SHUTDOWNS.clear()

    def _tracking_register(func, *args, **kwargs):
        registered.append(func)
        return real_register(func, *args, **kwargs)

    atexit.register = _tracking_register  # ty: ignore[invalid-assignment]
    try:
        yield registered
    finally:
        atexit.register = real_register
        for func in registered:
            atexit.unregister(func)
        for signum, handler in original.items():
            signal.signal(signum, handler)
        _INSTALLED.clear()
        _INSTALLED.update(installed)
        _TERMINATION_SHUTDOWNS.clear()
        _TERMINATION_SHUTDOWNS.extend(prior_shutdowns)


class _Driver:
    def __init__(self, *, connection: Mapping[str, str] | None = None, **kwargs: object) -> None:
        self.connection = dict(connection or {})
        self.metadata: dict[str, str] = {}
        self.error: BaseException | None = None

    async def destroy_sandboxes_matching(self, metadata: Mapping[str, str]) -> tuple[str, ...]:
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


def test_reap_logs_an_unknown_provider_and_returns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("ERROR"):
        assert reap_job_sandboxes("rl-1", host_provider="no-such-provider") == ()

    assert "failed to reap sandboxes for job rl-1" in caplog.text


def test_exit_and_cancel_both_reap(monkeypatch: pytest.MonkeyPatch, restore_process_state) -> None:
    """atexit covers a failed process. SIGTERM covers a cancel, which does not run atexit."""
    calls: list[str] = []
    monkeypatch.setattr("os.kill", lambda pid, sig: calls.append(f"kill:{sig}"))
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
