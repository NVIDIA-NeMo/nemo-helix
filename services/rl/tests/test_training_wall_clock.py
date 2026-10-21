# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The RL training wall clock wraps the driver, not Ray cluster bring-up."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from nhx.customization_common.service.context import NHXJobContext
from nhx.customization_common.training.progress import JobsServiceProgressReporter
from nhx.rl.app.constants import DEFAULT_TRAINING_RESULT_FILE_NAME
from nhx.rl.app.jobs.training.schemas import (
    DPOConfig,
    ModelConfig,
    TrainingMetrics,
    TrainingStepConfig,
    TrainingType,
)
from nhx.rl.tasks.training.backends.nemo_rl.ray_bootstrap import RayClusterBootstrap
from nhx.rl.tasks.training.distributed import DistributedContext, DistributedRole
from nhx.rl.tasks.training.progress import JobsServiceProgressReporter as RLProgressReporter
from nhx.rl.tasks.training.protocol import LibraryConfig, TrainingBackend
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


class EventProgress(JobsServiceProgressReporter):
    """Records when the real training wall clock reports start and finish, and sends nothing."""

    def __init__(self, events: list[str]) -> None:
        # The base constructor builds a Jobs client; nothing here is sent anywhere.
        self.training_duration_seconds = None
        self.events = events

    def update_task(
        self,
        status: str = "active",
        status_details: dict[str, Any] | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> None:
        details = status_details or {}
        if "training_started_at" in details:
            self.events.append("clock-start")
        if "training_finished_at" in details:
            self.events.append("clock-end")


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
    bootstrap.progress = EventProgress(events)

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
    bootstrap.progress = EventProgress(events)

    with pytest.raises(RuntimeError, match="driver failed"):
        bootstrap._run_driver("driver.py", [])

    assert events == ["clock-start", "popen", "wait", "clock-end"]


@dataclass
class OrderRecordingBootstrap(RayClusterBootstrap):
    """A head-node bootstrap whose cluster steps only record the order they ran in."""

    order: list[str] = field(default_factory=list)

    def _start_head_background(self) -> bool:
        self.order.append("head")
        return True

    def _wait_for_workers(self) -> None:
        self.order.append("wait-workers")

    def _log_ray_status(self) -> None:
        self.order.append("status")

    def _run_driver(self, driver_script: str, driver_args: list[str]) -> int:
        self.order.append("driver")
        return 0

    def _cleanup_with_timeout(self, timeout: int = 30) -> None:
        self.order.append("cleanup")


def test_ray_worker_wait_is_outside_the_driver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nhx.rl.tasks.training.backends.nemo_rl.ray_bootstrap._pause", lambda seconds: None)
    bootstrap = OrderRecordingBootstrap(rank=0, world_size=1, master_addr="127.0.0.1", log_dir=tmp_path)

    assert bootstrap._run_head_with_driver("driver.py", []) == 0
    assert bootstrap.order.index("wait-workers") < bootstrap.order.index("driver")


class ScriptedRunner(TrainingRunner):
    """A coordinator runner built from in-memory parts, whose config and training phases the test supplies.

    It skips the environment-reading constructor. Progress goes to a real reporter whose job context
    isn't configured, so it reports nothing, and the distributed context is a real single-node one.
    """

    def __init__(
        self,
        tmp_path: Path,
        *,
        duration: float | None,
        compile_config: Callable[[], LibraryConfig],
        train: Callable[[LibraryConfig], TrainingMetrics] | None = None,
        backend: TrainingBackend,
    ) -> None:
        self._job_ctx = NHXJobContext(
            workspace="default",
            job_id="",
            attempt_id="",
            step="",
            task="",
            jobs_url=None,
            files_url=None,
            storage_path=tmp_path,
            config_path=tmp_path / "config.json",
        )
        self._config = TrainingStepConfig(
            model=ModelConfig(path="/model"),
            dataset=TrainingStepConfig.DatasetConfig(path="/data"),
            training=TrainingStepConfig.TrainingConfig(training_type=TrainingType.DPO, dpo=DPOConfig()),
            schedule=TrainingStepConfig.ScheduleConfig(),
            batch=TrainingStepConfig.BatchConfig(),
            optimizer=TrainingStepConfig.OptimizerConfig(),
            parallelism=TrainingStepConfig.ParallelismConfig(),
            output_model="out",
            seed=1,
        )
        self._progress = RLProgressReporter(self._job_ctx)
        self._progress.training_duration_seconds = duration
        barrier_dir = tmp_path / "barriers"
        barrier_dir.mkdir()
        self._dist_ctx = DistributedContext(
            role=DistributedRole.COORDINATOR, rank=0, world_size=1, barrier_dir=barrier_dir
        )
        self._backend = backend
        self._workspace_path = tmp_path
        self._output_path = tmp_path / "output"
        self._compile_config = compile_config
        self._train = train

    def _preprocessing_phase(self) -> None:
        return None

    def _compile_config_phase(self) -> LibraryConfig:
        return self._compile_config()

    def _training_phase(self, library_config: LibraryConfig) -> TrainingMetrics:
        assert self._train is not None, "training was not expected to start"
        return self._train(library_config)


def written_duration(tmp_path: Path) -> float | None:
    payload = json.loads((tmp_path / DEFAULT_TRAINING_RESULT_FILE_NAME).read_text())
    return payload["training_duration_seconds"]


def test_result_json_copies_the_recorded_training_duration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nhx.rl.tasks.training.runner.get_gpu_info", lambda: None)
    backend = MagicMock()
    backend.find_best_checkpoint.return_value = tmp_path / "ckpt"
    backend.process_checkpoint.return_value = None
    runner = ScriptedRunner(
        tmp_path,
        duration=4.5,
        compile_config=MagicMock,
        train=lambda library: TrainingMetrics(),
        backend=backend,
    )

    result = runner.run()

    assert result.training_duration_seconds == 4.5
    assert written_duration(tmp_path) == 4.5


def test_result_json_keeps_the_recorded_duration_when_training_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nhx.rl.tasks.training.runner.get_gpu_info", lambda: None)
    runner = ScriptedRunner(
        tmp_path,
        duration=8.0,
        compile_config=MagicMock,
        train=MagicMock(side_effect=RuntimeError("train failed")),
        backend=MagicMock(),
    )

    result = runner.run()

    assert result.success is False
    assert result.training_duration_seconds == 8.0
    assert written_duration(tmp_path) == 8.0


def test_result_json_omits_duration_when_training_never_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nhx.rl.tasks.training.runner.get_gpu_info", lambda: None)
    runner = ScriptedRunner(
        tmp_path,
        duration=None,
        compile_config=MagicMock(side_effect=RuntimeError("compile failed")),
        backend=MagicMock(),
    )

    result = runner.run()

    assert result.training_duration_seconds is None
    assert written_duration(tmp_path) is None


def test_postprocessing_reports_that_it_is_finalizing_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Converting or copying a large checkpoint can run for minutes after the last step."""
    sent: list[str] = []
    backend = MagicMock()
    backend.find_best_checkpoint.return_value = tmp_path / "ckpt"
    backend.process_checkpoint.side_effect = lambda *a, **k: sent.append("process")
    runner = ScriptedRunner(tmp_path, duration=1.0, compile_config=MagicMock, backend=backend)
    monkeypatch.setattr(runner._progress, "report_running", lambda reported, **details: sent.append(reported))

    runner._postprocess_phase(None, TrainingMetrics(), MagicMock())

    assert sent == ["finalizing_checkpoint", "process"]
