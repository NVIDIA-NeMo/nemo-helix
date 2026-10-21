# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from cluster import ClusterWriteError, JobView
from config import PodSize, Step, Sweep
from controller import execute_steps, selected_steps
from records import marker_path, write_json

PODS = {
    "exit": PodSize("100m", "256Mi", "500m", "512Mi"),
    "idle": PodSize("100m", "256Mi", "500m", "512Mi"),
    "hammer": PodSize("500m", "512Mi", "1", "1Gi"),
}


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeCluster:
    def __init__(self) -> None:
        self.jobs_state: dict[str, str] = {}
        self.created: list[str] = []
        self.files: dict[str, bytes] = {}
        self.removed: list[str] = []
        self.refuse_after: int | None = None
        self.complete_on_sleep = False
        self.fail_writes = False

    def create_job(self, name: str, environment: dict[str, str], pod: PodSize, profile_name: str) -> bool:
        del environment, pod, profile_name
        if self.refuse_after is not None and len(self.created) >= self.refuse_after:
            self.created.append(name)
            return False
        self.created.append(name)
        self.jobs_state[name] = "pending" if self.complete_on_sleep else "completed"
        return True

    def jobs(self, prefix: str) -> list[JobView]:
        return [JobView(name, status) for name, status in sorted(self.jobs_state.items()) if name.startswith(prefix)]

    def cancel(self, name: str) -> None:
        if name in self.jobs_state and self.jobs_state[name] not in {"completed", "error", "cancelled"}:
            self.jobs_state[name] = "cancelled"

    def delete(self, name: str) -> None:
        self.jobs_state.pop(name, None)

    def read_result(self, name: str) -> bytes | None:
        return b'{"record":"header","code_fetch_seconds":0.1}\n'

    def write_file(self, path: str, data: bytes) -> None:
        if self.fail_writes:
            raise ClusterWriteError(path)
        self.files[path] = data

    def remove_prefix(self, prefix: str) -> None:
        self.removed.append(prefix)
        for path in list(self.files):
            if not prefix or path.startswith(prefix):
                del self.files[path]

    def sleep(self, seconds: float, clock: Clock) -> None:
        clock.sleep(seconds)
        if self.complete_on_sleep:
            for name, status in list(self.jobs_state.items()):
                if status == "pending":
                    self.jobs_state[name] = "completed"


def _sweep(*steps: Step) -> Sweep:
    return Sweep(run_key="stress", payload_bytes=8, pods=PODS, steps=steps)


def _exit(step_id: str, count: int) -> Step:
    return Step(step_id, "exit", count, None, None, None)


def _run(tmp_path: Path, sweep: Sweep, cluster: FakeCluster, only: str | None, budget: int = 100) -> int:
    clock = Clock()
    return execute_steps(
        sweep,
        tmp_path,
        selected_steps(sweep, tmp_path, only),
        cluster,
        profile_name="default",
        fileset="stress-stress",
        ttl_before=budget // 2,
        ttl_active=budget // 2,
        clock=clock,
        sleep=lambda seconds: cluster.sleep(seconds, clock),
        poll_seconds=2,
    )


def test_resume_skips_a_finished_step_and_runs_a_new_id(tmp_path: Path) -> None:
    first = _exit("exit-01", 1)
    second = _exit("exit-10", 1)
    write_json(marker_path(tmp_path, first.id), dict(first.measurement()))
    cluster = FakeCluster()
    assert _run(tmp_path, _sweep(first, second), cluster, None) == 0
    assert cluster.created == ["stress-exit-10-1"]


def test_changed_count_replaces_the_step(tmp_path: Path) -> None:
    step = _exit("exit-01", 2)
    stale = dict(step.measurement())
    stale["count"] = 1
    write_json(marker_path(tmp_path, step.id), stale)
    cluster = FakeCluster()
    assert _run(tmp_path, _sweep(step), cluster, None) == 0
    assert cluster.created == ["stress-exit-01-1", "stress-exit-01-2"]


def test_missing_marker_runs_the_step(tmp_path: Path) -> None:
    step = _exit("exit-01", 1)
    cluster = FakeCluster()
    assert _run(tmp_path, _sweep(step), cluster, None) == 0
    assert cluster.created == ["stress-exit-01-1"]
    assert marker_path(tmp_path, step.id).is_file()


def test_finalize_failure_does_not_submit_the_next_step(tmp_path: Path) -> None:
    first = _exit("exit-01", 1)
    second = _exit("exit-10", 1)
    cluster = FakeCluster()
    cluster.complete_on_sleep = False
    cluster.jobs_state["stress-exit-01-1"] = "pending"
    code = _run(tmp_path, _sweep(first, second), cluster, None, budget=4)
    assert code == 1
    assert cluster.created == []
    assert not marker_path(tmp_path, first.id).is_file()


def test_leftover_jobs_are_deleted_before_another_submit(tmp_path: Path) -> None:
    first = _exit("exit-01", 1)
    second = _exit("exit-10", 1)
    cluster = FakeCluster()
    cluster.complete_on_sleep = True
    cluster.jobs_state["stress-exit-01-1"] = "pending"
    assert _run(tmp_path, _sweep(first, second), cluster, None, budget=20) == 0
    assert "stress-exit-01-1" not in cluster.created
    assert cluster.created == ["stress-exit-10-1"]
    assert "stress-exit-01-1" not in cluster.jobs_state


def test_replacing_one_step_does_not_duplicate_another(tmp_path: Path) -> None:
    first = _exit("exit-01", 1)
    second = _exit("exit-10", 1)
    write_json(marker_path(tmp_path, first.id), dict(first.measurement()))
    write_json(marker_path(tmp_path, second.id), dict(second.measurement()))
    cluster = FakeCluster()
    assert _run(tmp_path, _sweep(first, second), cluster, "exit-01") == 0
    assert cluster.created == ["stress-exit-01-1"]


def test_refused_create_finalizes_accepted_jobs_and_stops(tmp_path: Path) -> None:
    step = _exit("exit-01", 2)
    cluster = FakeCluster()
    cluster.refuse_after = 1
    assert _run(tmp_path, _sweep(step), cluster, None) == 1
    assert cluster.created == ["stress-exit-01-1", "stress-exit-01-2"]
    assert "stress-exit-01-1" not in cluster.jobs_state
    assert not marker_path(tmp_path, step.id).is_file()
