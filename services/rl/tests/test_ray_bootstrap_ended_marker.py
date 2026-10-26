# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A resumed run's pods never act on the ENDED marker that the paused run left on shared storage."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from nhx.rl.tasks.training.backends.nemo_rl import ray_bootstrap
from nhx.rl.tasks.training.backends.nemo_rl.ray_bootstrap import RayClusterBootstrap

# A resume restarts the paused step in the same attempt, so both runs use the same marker path.
ATTEMPT = "attempt-1"


def _node(tmp_path: Path, rank: int) -> RayClusterBootstrap:
    return RayClusterBootstrap(
        rank=rank, world_size=2, master_addr="head", log_dir=tmp_path, attempt_id=ATTEMPT, retry_sleep=0
    )


@pytest.fixture(autouse=True)
def short_pauses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ray_bootstrap, "_pause", lambda seconds: time.sleep(0.01))


def _wait_until(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)


def test_a_resumed_worker_that_starts_before_the_head_runs_until_this_run_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker starts while the paused run's marker is still there, before the head has cleared it.

    It must keep trying to connect rather than exit, which would fail the resumed job. A worker connects
    only once the head's Ray is up, and the head clears the marker before it starts Ray, so the worker
    then stops only on the marker this run's head writes when it ends.
    """
    marker = tmp_path / f"ENDED.{ATTEMPT}"
    marker.touch()
    head, worker = _node(tmp_path, 0), _node(tmp_path, 1)
    head_ray_up, worker_connected = threading.Event(), threading.Event()
    events: list[str] = []

    def start_head() -> SimpleNamespace:
        events.append("head starts ray" if not marker.exists() else "head starts ray with the old marker")
        head_ray_up.set()
        return SimpleNamespace(returncode=0)

    def start_worker() -> SimpleNamespace | None:
        events.append("worker starts ray")
        # `ray start --address` connects only once the head's Ray is up.
        if not head_ray_up.wait(timeout=10):
            return None
        events.append("worker connected")
        worker_connected.set()
        return SimpleNamespace(returncode=0)

    def run_driver(driver_script: str, driver_args: list[str]) -> int:
        events.append("driver ran")
        return 0

    monkeypatch.setattr(head, "_start_head_process", start_head)
    monkeypatch.setattr(head, "_wait_for_workers", lambda: worker_connected.wait(timeout=10))
    monkeypatch.setattr(head, "_log_ray_status", lambda: None)
    monkeypatch.setattr(head, "_run_driver", run_driver)
    monkeypatch.setattr(head, "_stop_ray", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "_start_worker_process", start_worker)
    monkeypatch.setattr(worker, "_stop_ray", lambda *args, **kwargs: events.append("worker stopped"))

    worker_exit: list[int] = []
    worker_thread = threading.Thread(target=lambda: worker_exit.append(worker._run_as_worker()))
    worker_thread.start()
    _wait_until(lambda: "worker starts ray" in events or not worker_thread.is_alive())
    head_exit = head._run_head_with_driver("driver.py", [])
    worker_thread.join(timeout=10)

    assert head_exit == 0
    assert worker_exit == [0]
    assert events == ["worker starts ray", "head starts ray", "worker connected", "driver ran", "worker stopped"]


def test_the_head_does_not_start_ray_while_the_old_marker_remains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Workers would read the marker once connected, so a head that cannot clear it fails instead."""
    (tmp_path / f"ENDED.{ATTEMPT}").touch()
    head = _node(tmp_path, 0)
    started: list[bool] = []

    def refuse_unlink(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError("read-only")

    monkeypatch.setattr(ray_bootstrap.Path, "unlink", refuse_unlink)
    monkeypatch.setattr(head, "_start_head_process", lambda: started.append(True))
    monkeypatch.setattr(head, "_stop_ray", lambda *args, **kwargs: None)

    assert head._run_head_with_driver("driver.py", []) == 1
    assert started == []
    assert any("Could not clear the ENDED marker" in line for line in head.driver_output)
