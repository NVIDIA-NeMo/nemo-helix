# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The RL training wall clock wraps the driver, not Ray cluster bring-up."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from nhx.rl.app.constants import DEFAULT_TRAINING_RESULT_FILE_NAME
from nhx.rl.app.jobs.training.schemas import TrainingMetrics
from nhx.rl.tasks.training.backends.nemo_rl.ray_bootstrap import RayClusterBootstrap
from nhx.rl.tasks.training.runner import TrainingRunner

PROXY_VARS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")


@pytest.fixture
def bootstrap(tmp_path: Path) -> Iterator[RayClusterBootstrap]:
    saved = {name: os.environ.get(name) for name in PROXY_VARS}
    cluster = RayClusterBootstrap(
        rank=0,
        world_size=1,
        master_addr="127.0.0.1",
        log_dir=tmp_path,
    )
    yield cluster
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


class Clock:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __enter__(self) -> "Clock":
        self.events.append("clock-start")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.events.append("clock-end")
        return False


class Progress:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def training_wall_clock(self) -> Clock:
        return Clock(self.events)


class Proc:
    def __init__(self, events: list[str], *, fail: bool = False) -> None:
        self.events = events
        self.fail = fail
        self.returncode = 0
        self.pid = 7

    def wait(self) -> int:
        self.events.append("wait")
        if self.fail:
            raise RuntimeError("driver failed")
        return 0

    def poll(self) -> int:
        return 0

    def send_signal(self, signum: int) -> None:
        return None


def test_driver_clock_starts_at_popen_and_ends_after_wait(
    bootstrap: RayClusterBootstrap, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        "nhx.rl.tasks.training.backends.nemo_rl.ray_bootstrap.read_subprocess_output",
        lambda proc, buffer: None,
    )

    def popen(*args: object, **kwargs: object) -> Proc:
        events.append("popen")
        return Proc(events)

    monkeypatch.setattr("nhx.rl.tasks.training.backends.nemo_rl.ray_bootstrap.subprocess.Popen", popen)
    bootstrap.progress = Progress(events)  # type: ignore[assignment]

    assert bootstrap._run_driver("driver.py", ["--config", "cfg.yaml"]) == 0
    assert events == ["clock-start", "popen", "wait", "clock-end"]


def test_driver_clock_closes_when_the_driver_fails(
    bootstrap: RayClusterBootstrap, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        "nhx.rl.tasks.training.backends.nemo_rl.ray_bootstrap.read_subprocess_output",
        lambda proc, buffer: None,
    )

    def popen(*args: object, **kwargs: object) -> Proc:
        events.append("popen")
        return Proc(events, fail=True)

    monkeypatch.setattr("nhx.rl.tasks.training.backends.nemo_rl.ray_bootstrap.subprocess.Popen", popen)
    bootstrap.progress = Progress(events)  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="driver failed"):
        bootstrap._run_driver("driver.py", [])

    assert events == ["clock-start", "popen", "wait", "clock-end"]


def test_ray_worker_wait_is_outside_the_driver(bootstrap: RayClusterBootstrap, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nhx.rl.tasks.training.backends.nemo_rl.ray_bootstrap._pause", lambda seconds: None)
    order: list[str] = []
    bootstrap._start_head_background = lambda: order.append("head") or True  # type: ignore[method-assign]
    bootstrap._wait_for_workers = lambda: order.append("wait-workers")  # type: ignore[method-assign]
    bootstrap._log_ray_status = lambda: order.append("status")  # type: ignore[method-assign]
    bootstrap._run_driver = lambda script, args: order.append("driver") or 0  # type: ignore[method-assign]
    bootstrap._cleanup_with_timeout = lambda timeout=30: order.append("cleanup")  # type: ignore[method-assign]

    assert bootstrap._run_head_with_driver("driver.py", []) == 0
    assert order.index("wait-workers") < order.index("driver")


def runner_with_duration(tmp_path: Path, duration: float | None) -> TrainingRunner:
    runner = TrainingRunner.__new__(TrainingRunner)
    runner._config = SimpleNamespace(seed=1)
    runner._progress = SimpleNamespace(
        training_duration_seconds=duration,
        report_running=lambda *args, **kwargs: None,
        report_completed=lambda *args, **kwargs: None,
        report_error=lambda *args, **kwargs: None,
    )
    runner._dist_ctx = SimpleNamespace(
        is_coordinator=True,
        sync_point=lambda name: None,
        signal_failure=lambda: None,
    )
    runner._workspace_path = tmp_path
    runner._output_path = tmp_path / "output"
    runner._backend = MagicMock()
    runner._preprocessing_phase = lambda: None  # type: ignore[method-assign]
    return runner


def written_duration(tmp_path: Path) -> float | None:
    payload = json.loads((tmp_path / DEFAULT_TRAINING_RESULT_FILE_NAME).read_text())
    return payload["training_duration_seconds"]


def test_result_json_copies_the_recorded_training_duration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nhx.rl.tasks.training.runner.get_gpu_info", lambda: None)
    runner = runner_with_duration(tmp_path, 4.5)
    runner._compile_config_phase = lambda: MagicMock()  # type: ignore[method-assign]
    runner._training_phase = lambda library: TrainingMetrics()  # type: ignore[method-assign]
    runner._backend.find_best_checkpoint.return_value = tmp_path / "ckpt"
    runner._backend.process_checkpoint.return_value = None

    result = runner.run()

    assert result.training_duration_seconds == 4.5
    assert written_duration(tmp_path) == 4.5


def test_result_json_keeps_the_recorded_duration_when_training_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nhx.rl.tasks.training.runner.get_gpu_info", lambda: None)
    runner = runner_with_duration(tmp_path, 8.0)
    runner._compile_config_phase = lambda: MagicMock()  # type: ignore[method-assign]
    runner._training_phase = MagicMock(side_effect=RuntimeError("train failed"))  # type: ignore[method-assign]

    result = runner.run()

    assert result.success is False
    assert result.training_duration_seconds == 8.0
    assert written_duration(tmp_path) == 8.0


def test_result_json_omits_duration_when_training_never_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nhx.rl.tasks.training.runner.get_gpu_info", lambda: None)
    runner = runner_with_duration(tmp_path, None)
    runner._compile_config_phase = MagicMock(side_effect=RuntimeError("compile failed"))  # type: ignore[method-assign]

    result = runner.run()

    assert result.training_duration_seconds is None
    assert written_duration(tmp_path) is None
