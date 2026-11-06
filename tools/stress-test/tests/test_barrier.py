# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

from bundle import barrier_exit_path, barrier_ready_path, barrier_release_path
from names import workspace_names

WORKER = Path(__file__).resolve().parents[1] / "worker" / "barrier.py"


def _worker_barrier():
    spec = importlib.util.spec_from_file_location("stress_worker_barrier", WORKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Files:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def write(self, path: str, data: bytes) -> None:
        self.objects[path] = data

    def read(self, path: str) -> bytes | None:
        return self.objects.get(path)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.objects_release()

    def objects_release(self) -> None:
        return None


def test_workspace_names_keep_the_single_workspace_unsuffixed() -> None:
    assert workspace_names("stress", 1) == ["stress-stress"]
    assert workspace_names("stress", 2) == ["stress-stress-1", "stress-stress-2"]


def test_release_is_exit_zero_and_timeout_is_exit_one() -> None:
    barrier = _worker_barrier()
    assert barrier.ready_path("barrier-10", "job") == barrier_ready_path("barrier-10", "job")
    assert barrier.release_path("barrier-10") == barrier_release_path("barrier-10")
    assert barrier.exit_path("barrier-10", "job") == barrier_exit_path("barrier-10", "job")
    files = _Files()
    clock = _Clock()

    def release_on_sleep(seconds: float) -> None:
        del seconds
        clock.now += 1
        files.objects[barrier.release_path("barrier-10")] = b"release\n"

    reason = barrier.wait_for_barrier(
        step="barrier-10",
        job="job",
        timeout=10,
        clock=clock,
        sleep=release_on_sleep,
        files=files,
        poll=1,
    )
    assert reason == "barrier"
    assert files.objects[barrier.exit_path("barrier-10", "job")] == b"barrier\n"

    files = _Files()
    clock = _Clock()

    def expire(seconds: float) -> None:
        del seconds
        clock.now += 10

    reason = barrier.wait_for_barrier(
        step="barrier-10",
        job="job",
        timeout=5,
        clock=clock,
        sleep=expire,
        files=files,
        poll=1,
    )
    assert reason == "timeout"
    assert files.objects[barrier.exit_path("barrier-10", "job")] == b"timeout\n"
