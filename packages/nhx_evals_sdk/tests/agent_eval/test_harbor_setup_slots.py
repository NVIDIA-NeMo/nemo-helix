# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Machine-wide slots that cap how many local Docker trials start and set up their agent at once."""

import asyncio
import os
import subprocess
import sys
import textwrap
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from nhx_evals_sdk.agent_eval.runtimes.harbor import setup_slots
from nhx_evals_sdk.agent_eval.runtimes.harbor.setup_slots import (
    DEFAULT_MAX_CONCURRENT_SETUPS,
    MAX_CONCURRENT_SETUPS_ENV,
    SetupSlots,
    default_slot_dir,
    max_concurrent_setups,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="no flock on Windows")

Hook = Callable[[SimpleNamespace], Awaitable[None]]


class _Job:
    """Records the hooks the slots register, by Harbor's event name."""

    def __init__(self) -> None:
        self.hooks: dict[str, Hook] = {}

    def on_environment_started(self, callback: Hook) -> None:
        self.hooks["environment-start"] = callback

    def on_agent_started(self, callback: Hook) -> None:
        self.hooks["agent-start"] = callback

    def on_trial_ended(self, callback: Hook) -> None:
        self.hooks["end"] = callback

    def on_trial_cancelled(self, callback: Hook) -> None:
        self.hooks["cancel"] = callback


def _event(name: str, *, import_path: str | None = None, kind: str | None = "docker") -> SimpleNamespace:
    return SimpleNamespace(
        trial_name=name, config=SimpleNamespace(environment=SimpleNamespace(type=kind, import_path=import_path))
    )


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup_slots, "POLL_INTERVAL_SEC", 0.01)


def _attached(tmp_path: Path, slots: int) -> _Job:
    job = _Job()
    SetupSlots(slots, tmp_path / "slots").attach(job)
    return job


async def _blocks(awaitable: Awaitable[None]) -> bool:
    try:
        await asyncio.wait_for(awaitable, timeout=0.2)
    except TimeoutError:
        return True
    return False


async def test_a_trial_waits_for_a_slot_until_another_trials_agent_starts(tmp_path: Path) -> None:
    job = _attached(tmp_path, slots=1)
    await job.hooks["environment-start"](_event("a"))
    waiting = asyncio.ensure_future(job.hooks["environment-start"](_event("b")))
    assert await _blocks(asyncio.shield(waiting))
    await job.hooks["agent-start"](_event("a"))
    await asyncio.wait_for(waiting, timeout=2)


@pytest.mark.parametrize("hook", ["end", "cancel"])
async def test_a_trial_that_ends_before_its_agent_starts_gives_its_slot_back(tmp_path: Path, hook: str) -> None:
    job = _attached(tmp_path, slots=1)
    await job.hooks["environment-start"](_event("a"))
    await job.hooks[hook](_event("a"))
    await asyncio.wait_for(job.hooks["environment-start"](_event("b")), timeout=2)


async def test_slots_are_shared_with_other_processes_on_the_machine(tmp_path: Path) -> None:
    """Another run holding the only slot keeps this run's trial waiting until that process lets go."""
    directory = tmp_path / "slots"
    directory.mkdir()
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(f"""
                import fcntl, sys
                handle = open({str(directory / "slot-0.lock")!r}, "ab")
                fcntl.flock(handle, fcntl.LOCK_EX)
                print("held", flush=True)
                sys.stdin.read()
            """),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
        job = _Job()
        SetupSlots(1, directory).attach(job)
        waiting = asyncio.ensure_future(job.hooks["environment-start"](_event("a")))
        assert await _blocks(asyncio.shield(waiting))
    finally:
        holder.kill()  # the kernel drops a dead process's lock
        holder.wait()
    await asyncio.wait_for(waiting, timeout=2)


async def test_n_slots_let_n_trials_set_up_at_once(tmp_path: Path) -> None:
    job = _attached(tmp_path, slots=2)
    await job.hooks["environment-start"](_event("a"))
    await asyncio.wait_for(job.hooks["environment-start"](_event("b")), timeout=2)
    assert await _blocks(job.hooks["environment-start"](_event("c")))


@pytest.mark.parametrize(
    "event",
    [_event("a", import_path="nemo_evals.harbor.opensandbox_environment:Env"), _event("a", kind="daytona")],
    ids=["import-path", "other-provider"],
)
async def test_trials_outside_local_docker_take_no_slot(tmp_path: Path, event: SimpleNamespace) -> None:
    job = _attached(tmp_path, slots=1)
    await job.hooks["environment-start"](event)
    await asyncio.wait_for(job.hooks["environment-start"](_event("b")), timeout=2)


async def test_an_import_path_environment_built_on_harbors_docker_takes_a_slot(tmp_path: Path) -> None:
    """Such as the agent-layer environment: an ``import_path`` environment whose trials still run on local Docker."""
    pytest.importorskip("harbor")
    job = _attached(tmp_path, slots=1)
    docker = "harbor.environments.docker.docker:DockerEnvironment"

    await job.hooks["environment-start"](_event("a", import_path=docker))

    assert await _blocks(job.hooks["environment-start"](_event("b", import_path=docker)))


def test_zero_slots_registers_no_hooks(tmp_path: Path) -> None:
    assert _attached(tmp_path, slots=0).hooks == {}


@pytest.mark.parametrize(("raw", "expected"), [(None, DEFAULT_MAX_CONCURRENT_SETUPS), ("", 4), ("0", 0), (" 8 ", 8)])
def test_the_slot_count_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: int):
    if raw is None:
        monkeypatch.delenv(MAX_CONCURRENT_SETUPS_ENV, raising=False)
    else:
        monkeypatch.setenv(MAX_CONCURRENT_SETUPS_ENV, raw)
    assert max_concurrent_setups() == expected


@pytest.mark.parametrize("raw", ["-1", "four", "1.5"])
def test_a_bad_slot_count_is_refused(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv(MAX_CONCURRENT_SETUPS_ENV, raw)
    with pytest.raises(ValueError, match=MAX_CONCURRENT_SETUPS_ENV):
        max_concurrent_setups()


def test_each_user_has_one_slot_directory_whatever_their_temp_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Another user can't open this one's lock files, and runs with different TMPDIRs must share slots."""
    monkeypatch.setenv("TMPDIR", "/elsewhere")
    assert default_slot_dir() == Path("/tmp") / f"nemo-harbor-setup-slots-{os.getuid()}"


async def _uncapped(directory: Path) -> bool:
    """Whether one slot in ``directory`` lets a second trial start while the first holds it."""
    job = _Job()
    SetupSlots(1, directory).attach(job)
    await job.hooks["environment-start"](_event("a"))
    return not await _blocks(job.hooks["environment-start"](_event("b")))


async def test_a_slot_directory_others_can_write_to_isnt_used(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Another user could plant a FIFO at a slot's path there: trials set up without the cap instead."""
    directory = tmp_path / "slots"
    directory.mkdir()
    directory.chmod(0o777)
    assert await _uncapped(directory)
    assert "isn't this user's own" in caplog.text
    assert list(directory.iterdir()) == []


async def test_a_slot_directory_another_user_owns_isnt_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "slots"
    directory.mkdir(mode=0o700)
    owner = directory.stat().st_uid
    monkeypatch.setattr(os, "getuid", lambda: owner + 1)
    assert await _uncapped(directory)


async def test_a_linked_slot_directory_isnt_used(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir(mode=0o700)
    (tmp_path / "slots").symlink_to(target)
    assert await _uncapped(tmp_path / "slots")


async def test_a_fifo_at_a_slots_path_is_skipped_without_blocking(tmp_path: Path) -> None:
    """Opening a FIFO to write waits for a reader: the slot is skipped instead, and the event loop never stalls."""
    directory = tmp_path / "slots"
    directory.mkdir(mode=0o700)
    os.mkfifo(directory / "slot-0.lock")
    job = _Job()
    SetupSlots(2, directory).attach(job)
    await asyncio.wait_for(job.hooks["environment-start"](_event("a")), timeout=2)  # takes slot 1
    assert await _blocks(job.hooks["environment-start"](_event("b")))  # slot 0 is no slot
