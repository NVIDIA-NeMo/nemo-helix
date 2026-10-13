# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle tests for the Docker Compose sandbox provider."""

from __future__ import annotations

import asyncio
import os
import select
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import pytest
from nhx_evals_sdk.agent_eval.runtimes.sandbox.base import SandboxCreateError, SandboxSpec, SandboxStatus
from nhx_evals_sdk.agent_eval.runtimes.sandbox.providers import _compose_cli as compose_cli
from nhx_evals_sdk.agent_eval.runtimes.sandbox.providers import _compose_lifecycle as compose_lifecycle
from nhx_evals_sdk.agent_eval.runtimes.sandbox.providers import _compose_state as compose_state
from nhx_evals_sdk.agent_eval.runtimes.sandbox.providers.compose import ComposeCleanupError, ComposeServiceTopology

from packages.nhx_evals_sdk.tests.agent_eval._compose_testkit import _compose_suffix, _create, _provider, _Runner


async def test_environment_precedence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    monkeypatch.setenv("PORT", "from-host")
    provider = _provider(tmp_path, environment_defaults={"PORT": "default", "ONLY_DEFAULT": "yes"})
    handle = await _create(
        monkeypatch,
        provider,
        runner,
        SandboxSpec(env={"PORT": "from-spec"}),
    )

    up_environment = next(env for argv, env, _ in runner.calls if _compose_suffix(argv)[:1] == ("up",))
    assert up_environment["PORT"] == "from-spec"
    assert up_environment["ONLY_DEFAULT"] == "yes"
    await provider.close(handle)


async def test_teardown_hook_is_constrained_and_runs_before_down(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    events: list[str] = []

    async def hook(context) -> None:
        assert await context.service_is_running("agent")
        events.append("hook")
        stopped = await context.stop_service("agent")
        assert stopped.ok
        result = await context.exec_service("redis", ("redis-cli", "PING"))
        assert result.ok

    provider = _provider(tmp_path, teardown_hook=hook)
    handle = await _create(monkeypatch, provider, runner)
    await provider.close(handle)

    suffixes = [_compose_suffix(argv) for argv, _, _ in runner.calls if argv[:2] == ("docker", "compose")]
    assert events == ["hook"]
    assert next(i for i, args in enumerate(suffixes) if args[:1] == ("stop",)) < next(
        i for i, args in enumerate(suffixes) if args[:1] == ("down",)
    )


async def test_hook_failure_is_reported_but_down_still_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()

    async def hook(_context) -> None:
        raise RuntimeError("wipe failed")

    provider = _provider(tmp_path, teardown_hook=hook)
    handle = await _create(monkeypatch, provider, runner)
    with pytest.raises(ComposeCleanupError, match="wipe failed"):
        await provider.close(handle)
    assert any(_compose_suffix(argv)[:1] == ("down",) for argv, _, _ in runner.calls)


async def test_cancelled_close_finishes_down_then_restores_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hook(_context) -> None:
        entered.set()
        await release.wait()

    provider = _provider(tmp_path, teardown_hook=hook)
    handle = await _create(monkeypatch, provider, runner)
    close_task = asyncio.create_task(provider.close(handle))
    await entered.wait()
    close_task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await close_task
    assert any(_compose_suffix(argv)[:1] == ("down",) for argv, _, _ in runner.calls)


async def test_cancelled_startup_diagnostics_finishes_cleanup_before_restoring_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    runner.up_failure = True
    provider = _provider(tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def capture_diagnostics(
        _environment: Mapping[str, str],
        *,
        reason: str,
    ) -> None:
        assert reason in {"startup-failure", "shutdown"}
        entered.set()
        await release.wait()

    monkeypatch.setattr(compose_cli, "_run_command", runner)
    monkeypatch.setattr(provider, "_capture_diagnostics", capture_diagnostics)
    create_task = asyncio.create_task(provider.create(SandboxSpec()))
    await entered.wait()
    create_task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await create_task

    assert any(_compose_suffix(argv)[:1] == ("down",) for argv, _, _ in runner.calls)
    assert provider._session is None
    recovered_lock = compose_state._ComposeProjectLock.acquire(provider.lock_path)
    recovered_lock.release()


@pytest.mark.parametrize("exit_type", [KeyboardInterrupt, SystemExit])
async def test_startup_process_exit_is_preserved_after_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exit_type: type[BaseException],
) -> None:
    runner = _Runner()
    provider = _provider(tmp_path)

    async def fail_readiness(_environment: Mapping[str, str]) -> None:
        raise exit_type("process exit")

    monkeypatch.setattr(provider, "_assert_ready", fail_readiness)

    with pytest.raises(exit_type, match="process exit"):
        await _create(monkeypatch, provider, runner)

    assert any(_compose_suffix(argv)[:1] == ("down",) for argv, _, _ in runner.calls)
    assert provider._session is None
    recovered_lock = compose_state._ComposeProjectLock.acquire(provider.lock_path)
    recovered_lock.release()


@pytest.mark.parametrize("exit_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("phase", ["hook", "down"])
async def test_teardown_process_exit_is_preserved_after_retirement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exit_type: type[BaseException],
    phase: str,
) -> None:
    runner = _Runner()

    async def hook(_context) -> None:
        if phase == "hook":
            raise exit_type("process exit")

    provider = _provider(tmp_path, teardown_hook=hook)
    handle = await _create(monkeypatch, provider, runner)

    if phase == "down":

        async def fail_down(*_args: object, **_kwargs: object) -> None:
            raise exit_type("process exit")

        monkeypatch.setattr(compose_lifecycle, "_compose_down", fail_down)

    with pytest.raises(exit_type, match="process exit"):
        await provider.close(handle)

    if phase == "hook":
        assert any(_compose_suffix(argv)[:1] == ("down",) for argv, _, _ in runner.calls)
    assert provider._session is None
    recovered_lock = compose_state._ComposeProjectLock.acquire(provider.lock_path)
    recovered_lock.release()


@pytest.mark.parametrize(("remove_volumes", "expected"), [(False, False), (True, True)])
async def test_volume_removal_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    remove_volumes: bool,
    expected: bool,
) -> None:
    runner = _Runner()
    provider = _provider(tmp_path, remove_project_volumes=remove_volumes)
    handle = await _create(monkeypatch, provider, runner)
    await provider.close(handle)
    down = next(_compose_suffix(argv) for argv, _, _ in runner.calls if _compose_suffix(argv)[:1] == ("down",))
    assert ("--volumes" in down) is expected


@pytest.mark.parametrize(
    ("remove_volumes", "expected_kinds", "expected_errors"),
    [
        (
            False,
            ["container", "network"],
            [
                "Managed Compose containers remain after teardown: container-a",
                "network query failed",
            ],
        ),
        (
            True,
            ["container", "network", "volume"],
            [
                "Managed Compose containers remain after teardown: container-a",
                "network query failed",
                "Managed Compose volumes remain after teardown: volume-a",
            ],
        ),
    ],
)
async def test_project_destruction_verification_preserves_kind_and_error_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    remove_volumes: bool,
    expected_kinds: list[str],
    expected_errors: list[str],
) -> None:
    provider = _provider(tmp_path, remove_project_volumes=remove_volumes)
    calls: list[str] = []
    responses = {
        "container": (["container-a"], None),
        "network": ([], "network query failed"),
        "volume": (["volume-a"], None),
    }

    async def managed_resource_names(
        kind: str,
        _environment: Mapping[str, str],
    ) -> tuple[list[str], str | None]:
        calls.append(kind)
        return responses[kind]

    monkeypatch.setattr(provider, "_managed_resource_names", managed_resource_names)

    errors = await provider._verify_project_destroyed(os.environ)

    assert calls == expected_kinds
    assert errors == expected_errors


async def test_project_destruction_verification_stops_after_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = _provider(tmp_path, remove_project_volumes=True)
    calls: list[str] = []

    async def managed_resource_names(
        kind: str,
        _environment: Mapping[str, str],
    ) -> tuple[list[str], str | None]:
        calls.append(kind)
        if kind == "network":
            raise OSError("inspection failed")
        return [], None

    monkeypatch.setattr(provider, "_managed_resource_names", managed_resource_names)

    with pytest.raises(OSError, match="inspection failed"):
        await provider._verify_project_destroyed(os.environ)

    assert calls == ["container", "network"]


async def test_closed_handle_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    provider = _provider(tmp_path)
    handle = await _create(monkeypatch, provider, runner)
    await provider.close(handle)

    with pytest.raises(ValueError, match="not the active Compose session"):
        await provider.exec(handle, "true")


async def test_stale_handle_cannot_alias_recreated_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    provider = _provider(tmp_path)
    first = await _create(monkeypatch, provider, runner, SandboxSpec(env={"SESSION": "first"}))
    await provider.close(first)
    second = await provider.create(SandboxSpec(env={"SESSION": "second"}))

    try:
        assert first.sandbox_id != second.sandbox_id
        with pytest.raises(ValueError, match="not the active Compose session"):
            await provider.close(first)
        assert await provider.status(second) == SandboxStatus.RUNNING
    finally:
        await provider.close(second)


async def test_active_session_freezes_project_scope_and_target_topology(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    provider = _provider(tmp_path, project_name="original", profiles=("original-profile",))
    handle = await _create(monkeypatch, provider, runner)
    original_scope = provider._command_scope()
    replacement_directory = tmp_path / "replacement"
    replacement_directory.mkdir()
    replacement_file = replacement_directory / "compose.yaml"
    replacement_file.write_text("services: {}\n", encoding="utf-8")

    provider.docker_bin = "replacement-docker"
    provider.project_directory = replacement_directory
    provider.compose_files = (replacement_file,)
    provider.project_name = "replacement"
    provider.profiles = ("replacement-profile",)
    provider.target_service = "replacement-service"
    provider.service_topology = ComposeServiceTopology(
        target_service="replacement-service",
        long_running_services=frozenset({"replacement-service"}),
    )

    assert provider._command_scope() == original_scope
    assert await provider.status(handle) == SandboxStatus.RUNNING
    await provider.exec(handle, "true")
    await provider.close(handle)

    compose_calls = [(argv, cwd) for argv, _, cwd in runner.calls if argv[:2] == ("docker", "compose")]
    assert compose_calls
    assert all(argv[argv.index("--project-name") + 1] == "original" for argv, _ in compose_calls)
    assert all(argv[0] == "docker" and cwd == tmp_path for argv, cwd in compose_calls)
    assert any(_compose_suffix(argv)[-4:] == ("agent", "sh", "-lc", "true") for argv, _ in compose_calls)
    assert provider._command_scope().project_name == "replacement"
    assert provider._command_scope().docker_bin == "replacement-docker"


def test_project_lock_and_generated_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    opened_fds: list[int] = []
    closed_fds: list[int] = []
    open_fd = os.open
    close_fd = os.close
    first = _provider(tmp_path, project_name="shared")
    second = _provider(tmp_path, project_name="shared")
    lock_paths = {first.lock_path, second.lock_path}

    def record_open(path: Path, flags: int, mode: int = 0o600) -> int:
        fd = open_fd(path, flags, mode)
        if Path(path) in lock_paths:
            opened_fds.append(fd)
        return fd

    def record_close(fd: int) -> None:
        if Counter(closed_fds)[fd] < Counter(opened_fds)[fd]:
            closed_fds.append(fd)
        close_fd(fd)

    monkeypatch.setattr(compose_state.os, "open", record_open)
    monkeypatch.setattr(compose_state.os, "close", record_close)
    first_lock = compose_state._ComposeProjectLock.acquire(first.lock_path)
    try:
        with pytest.raises(SandboxCreateError, match="Another Compose sandbox"):
            compose_state._ComposeProjectLock.acquire(second.lock_path)
    finally:
        first_lock.release()
    assert first_lock.fd is None

    second_lock = compose_state._ComposeProjectLock.acquire(second.lock_path)
    second_lock.release()
    assert second_lock.fd is None
    assert Counter(closed_fds) == Counter(opened_fds)

    generated = _provider(tmp_path, lock_path=tmp_path / "other.lock")
    assert generated.project_name.startswith("nemo-eval-")
    assert generated.project_name != first.project_name


async def test_create_wraps_lock_filesystem_errors(
    tmp_path: Path,
) -> None:
    lock_directory = tmp_path / "lock-directory"
    lock_directory.mkdir()
    provider = _provider(tmp_path, lock_path=lock_directory)

    with pytest.raises(SandboxCreateError, match="Could not acquire Compose project lock") as caught:
        await provider.create(SandboxSpec())

    assert isinstance(caught.value.__cause__, IsADirectoryError)
    assert provider._session is None


def test_project_lock_excludes_another_process(tmp_path: Path) -> None:
    lock_path = tmp_path / "compose.lock"
    script = "\n".join(
        [
            "import fcntl",
            "import os",
            "import sys",
            "fd = os.open(sys.argv[1], os.O_CREAT | os.O_RDWR, 0o600)",
            "fcntl.flock(fd, fcntl.LOCK_EX)",
            "print('locked', flush=True)",
            "sys.stdin.read(1)",
        ]
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(lock_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        ready, _, _ = select.select([process.stdout], [], [], 5)
        assert ready, "lock holder did not become ready"
        assert process.stdout.readline().strip() == "locked"

        with pytest.raises(SandboxCreateError, match="Another Compose sandbox"):
            compose_state._ComposeProjectLock.acquire(lock_path)
    finally:
        if process.stdin is not None:
            process.stdin.close()
        process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()
    assert process.returncode == 0

    contender_lock = compose_state._ComposeProjectLock.acquire(lock_path)
    contender_lock.release()
    assert contender_lock.fd is None
