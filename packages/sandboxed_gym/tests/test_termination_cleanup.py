# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Destroying the Gym host when the owning process is terminated rather than shut down.

The handlers are invoked directly rather than by delivering a real signal: the handler's final act
is to re-raise with the default action installed, which would take the test session down with it.
"""

from __future__ import annotations

import atexit
import signal
from collections.abc import Callable

import pytest
from sandboxed_gym.orchestrator import (
    _TERMINATION_SHUTDOWNS,
    TERMINATION_SIGNALS,
    install_termination_cleanup,
)


@pytest.fixture(autouse=True)
def restore_process_state():
    """Undo the process-global handlers and atexit hooks each test installs."""
    original = {signum: signal.getsignal(signum) for signum in TERMINATION_SIGNALS}
    registered: list[Callable[[], object]] = []
    real_register = atexit.register
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
        _TERMINATION_SHUTDOWNS.clear()
        _TERMINATION_SHUTDOWNS.extend(prior_shutdowns)


@pytest.mark.parametrize("signum", TERMINATION_SIGNALS)
def test_a_termination_signal_destroys_the_host(signum, monkeypatch: pytest.MonkeyPatch) -> None:
    """SIGTERM is what Kubernetes sends before SIGKILL, and the last chance to free the sandbox."""
    calls: list[str] = []
    monkeypatch.setattr("os.kill", lambda pid, sig: calls.append(f"kill:{sig}"))

    install_termination_cleanup(lambda: calls.append("shutdown"))
    handler = signal.getsignal(signum)
    assert not isinstance(handler, int) and handler is not None
    handler(signum, None)

    assert calls == ["shutdown", f"kill:{signum}"]


@pytest.mark.parametrize("signum", TERMINATION_SIGNALS)
def test_the_signal_is_re_raised_with_the_default_action(signum, monkeypatch: pytest.MonkeyPatch) -> None:
    """The exit status must still report the signal, or a cancelled job reads as a clean stop."""
    monkeypatch.setattr("os.kill", lambda pid, sig: None)

    install_termination_cleanup(lambda: None)
    handler = signal.getsignal(signum)
    assert not isinstance(handler, int) and handler is not None
    handler(signum, None)

    assert signal.getsignal(signum) is signal.SIG_DFL


@pytest.mark.parametrize("signum", TERMINATION_SIGNALS)
def test_the_default_action_is_restored_before_cleanup_runs(signum, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second signal during a slow destroy must terminate, not re-enter this handler.

    Restoring afterwards would leave the handler installed for the whole of ``shutdown()`` -- the
    slowest part, since it waits on the sandbox being destroyed -- so an impatient second SIGTERM
    would stack another destroy on top of the in-flight one.
    """
    monkeypatch.setattr("os.kill", lambda pid, sig: None)
    observed: list[object] = []

    install_termination_cleanup(lambda: observed.append(signal.getsignal(signum)))
    handler = signal.getsignal(signum)
    assert not isinstance(handler, int) and handler is not None
    handler(signum, None)

    assert observed == [signal.SIG_DFL]


@pytest.mark.parametrize("signum", TERMINATION_SIGNALS)
def test_the_signal_is_re_raised_even_when_cleanup_fails(signum, monkeypatch: pytest.MonkeyPatch) -> None:
    """A destroy that raises must not turn a terminated job into a hung one.

    Without the re-raise the exception escapes into whatever the handler interrupted, and the
    process keeps running with its exit status never reporting the signal.
    """
    killed: list[int] = []
    monkeypatch.setattr("os.kill", lambda pid, sig: killed.append(sig))

    def _explode() -> None:
        raise RuntimeError("destroy_host failed")

    install_termination_cleanup(_explode)
    handler = signal.getsignal(signum)
    assert not isinstance(handler, int) and handler is not None

    with pytest.raises(RuntimeError, match="destroy_host failed"):
        handler(signum, None)

    assert killed == [signum]


def test_shutdown_runs_at_exit(restore_process_state) -> None:
    """Covers the path a signal cannot: an ordinary interpreter exit with no teardown call.

    Also the only cover when handler installation is refused off the main thread.
    """
    calls: list[str] = []
    install_termination_cleanup(lambda: calls.append("shutdown"))

    assert restore_process_state, "nothing was registered with atexit"
    for hook in restore_process_state:
        hook()
    assert calls == ["shutdown"]


def test_handlers_are_optional_off_the_main_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ray does not promise to run an actor method on the main thread.

    ``signal.signal`` raises ValueError there. Registration must degrade to atexit rather than
    failing spinup, since a sandbox that leaks on SIGTERM still beats one that never starts.
    """

    def _refuse(signum: int, handler: object) -> None:
        raise ValueError("signal only works in main thread")

    monkeypatch.setattr(signal, "signal", _refuse)

    install_termination_cleanup(lambda: None)
