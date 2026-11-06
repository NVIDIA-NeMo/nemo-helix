# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from cluster import ClusterWriteError, JobView
from config import PodSize, Step, Sweep
from controller import execute_steps, selected_steps
from records import marker_path, read_json, record_path, write_json

PODS = {
    "exit": PodSize("100m", "256Mi", "500m", "512Mi"),
    "idle": PodSize("100m", "256Mi", "500m", "512Mi"),
    "hammer": PodSize("500m", "512Mi", "1", "1Gi"),
    "barrier": PodSize("100m", "256Mi", "500m", "512Mi"),
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
        self.images: list[str | None] = []
        self.files: dict[str, bytes] = {}
        self.removed: list[str] = []
        self.refuse_after: int | None = None
        self.complete_on_sleep = False
        self.fail_writes = False
        self.end_status = "completed"
        self.ready_on_create = False
        self.exit_on_release = False
        self.result: bytes | None = b'{"record":"header","code_fetch_seconds":0.1}\n'

    def create_job(
        self,
        name: str,
        environment: dict[str, str],
        pod: PodSize,
        profile_name: str,
        provider: str = "cpu",
        image: str | None = None,
    ) -> bool:
        del pod, profile_name, provider
        self.images.append(image)
        if self.refuse_after is not None and len(self.created) >= self.refuse_after:
            self.created.append(name)
            return False
        self.created.append(name)
        self.jobs_state[name] = "pending" if self.complete_on_sleep else self.end_status
        if self.ready_on_create:
            self.jobs_state[name] = "active"
            step_id = environment["STRESS_STEP_ID"]
            self.files[f"barrier/{step_id}/ready/{name}"] = b"ready\n"
        return True

    def jobs(self, prefix: str) -> list[JobView]:
        return [JobView(name, status) for name, status in sorted(self.jobs_state.items()) if name.startswith(prefix)]

    def cancel(self, name: str) -> None:
        if name in self.jobs_state and self.jobs_state[name] not in {"completed", "error", "cancelled"}:
            self.jobs_state[name] = "cancelled"

    def delete(self, name: str) -> None:
        self.jobs_state.pop(name, None)

    def read_result(self, name: str) -> bytes | None:
        return self.result

    def diagnose(self, name: str) -> dict[str, object]:
        return {"errors": [f"image pull failed for {name}"], "log_tail": ["Error: pull access denied"]}

    def read_file(self, path: str) -> bytes | None:
        return self.files.get(path)

    def write_file(self, path: str, data: bytes) -> None:
        if self.fail_writes:
            raise ClusterWriteError(path)
        self.files[path] = data
        if self.exit_on_release and path.endswith("/release"):
            step_id = path.split("/")[1]
            for name, status in list(self.jobs_state.items()):
                self.files[f"barrier/{step_id}/exit/{name}"] = b"barrier\n"
                if status == "active":
                    self.jobs_state[name] = "completed"

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


def _run(
    tmp_path: Path,
    sweep: Sweep,
    cluster: FakeCluster,
    only: str | None,
    budget: int = 100,
    image: str | None = None,
    force: bool = False,
) -> int:
    clock = Clock()
    return execute_steps(
        sweep,
        tmp_path,
        selected_steps(sweep, tmp_path, only, force=force),
        cluster,
        profile_name="default",
        provider="cpu",
        fileset="stress-stress",
        ttl_before=budget // 2,
        ttl_active=budget // 2,
        clock=clock,
        sleep=lambda seconds: cluster.sleep(seconds, clock),
        poll_seconds=2,
        image=image,
    )


def test_image_override_is_sent_with_the_job(tmp_path: Path) -> None:
    step = _exit("exit-01", 1)
    cluster = FakeCluster()
    assert _run(tmp_path, _sweep(step), cluster, None, image="local/nhx-tasks:dev") == 0
    assert cluster.images == ["local/nhx-tasks:dev"]


def test_force_reruns_a_finished_step(tmp_path: Path) -> None:
    step = _exit("exit-01", 1)
    write_json(marker_path(tmp_path, step.id), dict(step.measurement()))
    cluster = FakeCluster()
    assert _run(tmp_path, _sweep(step), cluster, None, force=True) == 0
    assert cluster.created == ["stress-exit-01-1"]


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


def test_failed_job_logs_an_error_and_keeps_the_log_tail_for_debug(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    step = _exit("exit-01", 1)
    cluster = FakeCluster()
    cluster.end_status = "error"
    cluster.result = None
    with caplog.at_level(logging.DEBUG, logger="stress"):
        assert _run(tmp_path, _sweep(step), cluster, None) == 0
    info = [record.message for record in caplog.records if record.levelno == logging.INFO]
    errors = [record.message for record in caplog.records if record.levelno >= logging.ERROR]
    debug = [record.message for record in caplog.records if record.levelno == logging.DEBUG]
    assert info == [
        "running 1 step(s): exit-01",
        "preparing step 1/1 exit-01 exit count=1",
        "submitting 1 job(s) for exit-01 on cpu/default",
        "started stress-exit-01-1",
        "waiting for 1 job(s) in exit-01",
        "step exit-01: 1 submitted, 1 error, 0 with results; marker written",
    ]
    assert errors == ["error stress-exit-01-1: image pull failed for stress-exit-01-1"]
    assert "stress-exit-01-1: Error: pull access denied" in debug
    record = read_json(record_path(tmp_path, step.id, "stress-exit-01-1"))
    assert record["platform_errors"] == ["image pull failed for stress-exit-01-1"]


def test_completed_exit_does_not_require_a_result(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    step = _exit("exit-01", 1)
    cluster = FakeCluster()
    cluster.result = None
    with caplog.at_level(logging.INFO, logger="stress"):
        assert _run(tmp_path, _sweep(step), cluster, None) == 0
    messages = [record.message for record in caplog.records if record.levelno >= logging.INFO]
    assert messages == [
        "running 1 step(s): exit-01",
        "preparing step 1/1 exit-01 exit count=1",
        "submitting 1 job(s) for exit-01 on cpu/default",
        "started stress-exit-01-1",
        "waiting for 1 job(s) in exit-01",
        "stopped stress-exit-01-1",
        "step exit-01: 1 submitted, 1 completed, 0 with results; marker written",
    ]
    saved = read_json(record_path(tmp_path, step.id, "stress-exit-01-1"))
    assert saved["finished_at"] == 0


def test_completed_job_logs_started_and_stopped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    step = _exit("exit-01", 1)
    with caplog.at_level(logging.INFO, logger="stress"):
        assert _run(tmp_path, _sweep(step), FakeCluster(), None) == 0
    assert [record.message for record in caplog.records] == [
        "running 1 step(s): exit-01",
        "preparing step 1/1 exit-01 exit count=1",
        "submitting 1 job(s) for exit-01 on cpu/default",
        "started stress-exit-01-1",
        "waiting for 1 job(s) in exit-01",
        "stopped stress-exit-01-1",
        "step exit-01: 1 submitted, 1 completed, 0 with results; marker written",
    ]


def test_sweep_logs_the_next_step_before_its_jobs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    first = _exit("exit-01", 1)
    second = _exit("exit-10", 2)
    with caplog.at_level(logging.INFO, logger="stress"):
        assert _run(tmp_path, _sweep(first, second), FakeCluster(), None) == 0
    messages = [record.message for record in caplog.records if record.levelno == logging.INFO]
    assert messages[0] == "running 2 step(s): exit-01, exit-10"
    prepare_second = messages.index("preparing step 2/2 exit-10 exit count=2")
    assert messages.index("step exit-01: 1 submitted, 1 completed, 0 with results; marker written") < prepare_second
    assert messages[prepare_second + 1] == "submitting 2 job(s) for exit-10 on cpu/default"
    assert "waiting for 2 job(s) in exit-10" in messages


def test_barrier_splits_jobs_and_records_a_barrier_exit(tmp_path: Path) -> None:
    step = Step("barrier-10", "barrier", 2, None, None, None, timeout_seconds=30)
    sweep = Sweep("stress", 8, PODS, (step,))
    first = FakeCluster()
    second = FakeCluster()
    first.ready_on_create = second.ready_on_create = True
    first.exit_on_release = second.exit_on_release = True
    clock = Clock()
    code = execute_steps(
        sweep,
        tmp_path,
        [(step, True)],
        first,
        profile_name="default",
        provider="cpu",
        fileset="stress-stress",
        ttl_before=30,
        ttl_active=30,
        clock=clock,
        sleep=lambda seconds: clock.sleep(seconds),
        poll_seconds=1,
        clusters=[first, second],
        workspaces=["stress-stress-1", "stress-stress-2"],
    )
    assert code == 0
    assert first.created == ["stress-barrier-10-1", "stress-barrier-10-3"]
    assert second.created == ["stress-barrier-10-2", "stress-barrier-10-4"]
    cohort = read_json(tmp_path / "steps" / "barrier-10" / "cohort.json")
    by_workspace = cohort["by_workspace"]
    assert isinstance(by_workspace, dict)
    first_workspace = by_workspace["stress-stress-1"]
    second_workspace = by_workspace["stress-stress-2"]
    assert isinstance(first_workspace, dict)
    assert isinstance(second_workspace, dict)
    assert cohort["released"] is True
    assert cohort["barrier"] == 4
    assert cohort["timeout"] == 0
    assert cohort["peak_active"] == 4
    assert first_workspace["accepted"] == 2
    assert second_workspace["barrier"] == 2
    saved = read_json(record_path(tmp_path, step.id, "stress-barrier-10-1"))
    assert saved["exit_reason"] == "barrier"
    assert saved["workspace"] == "stress-stress-1"
