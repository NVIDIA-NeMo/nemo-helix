# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Automodel training runner."""

import json
import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

sys.modules["nemo_automodel"] = MagicMock()
sys.modules["nemo_automodel._transformers"] = MagicMock()
sys.modules["nemo_automodel._transformers.registry"] = MagicMock()

from nhx.automodel.entities.values import FinetuningType, TrainingType  # noqa: E402
from nhx.automodel.tasks.training.backends.backend import AutomodelBackend  # noqa: E402
from nhx.automodel.tasks.training.distributed import DistributedContext, DistributedRole  # noqa: E402
from nhx.automodel.tasks.training.progress import JobsServiceProgressReporter  # noqa: E402
from nhx.automodel.tasks.training.protocol import LibraryConfig  # noqa: E402
from nhx.automodel.tasks.training.runner import TrainingRunner  # noqa: E402
from nhx.automodel.tasks.training.schemas import (  # noqa: E402
    DistillationConfig,
    ModelConfig,
    TrainingMetrics,
    TrainingResult,
    TrainingStepConfig,
)
from nhx.customization_common.service.context import NHXJobContext  # noqa: E402


def _job_context(storage_path: Path) -> NHXJobContext:
    return NHXJobContext(
        workspace="default",
        job_id="job-1",
        attempt_id="attempt-0",
        step="training",
        task="task-1",
        jobs_url=None,
        files_url=None,
        storage_path=storage_path,
        config_path=storage_path / "config.json",
    )


def _config() -> TrainingStepConfig:
    return TrainingStepConfig(
        model=ModelConfig(path="/run/scratch/job/model"),
        dataset=TrainingStepConfig.DatasetConfig(path="/run/scratch/job/dataset"),
        training=TrainingStepConfig.TrainingConfig(
            training_type=TrainingType.DISTILLATION,
            kd=DistillationConfig(teacher_model=ModelConfig(path="/run/scratch/job/teacher_model")),
        ),
        schedule=TrainingStepConfig.ScheduleConfig(),
        batch=TrainingStepConfig.BatchConfig(),
        optimizer=TrainingStepConfig.OptimizerConfig(),
        parallelism=TrainingStepConfig.ParallelismConfig(),
        output_model="trained-model",
        workspace_path="/run/scratch/job/training",
        output_path="/run/scratch/job/output_model",
    )


def test_normalizes_legacy_job_storage_paths_to_runtime_mount(tmp_path: Path) -> None:
    storage = tmp_path / "job"
    runner = TrainingRunner.__new__(TrainingRunner)
    runner._job_ctx = _job_context(storage)

    normalized = runner._normalize_storage_paths(_config())

    assert normalized.model.path == str(storage / "model")
    assert normalized.dataset.path == str(storage / "dataset")
    assert normalized.training.kd is not None
    assert normalized.training.kd.teacher_model.path == str(storage / "teacher_model")
    assert normalized.workspace_path == str(storage / "training")
    assert normalized.output_path == str(storage / "output_model")


class _ScriptedRunner(TrainingRunner):
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
        backend: AutomodelBackend,
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
        self._config = _config()
        self._progress = JobsServiceProgressReporter(self._job_ctx)
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

    def _compile_config_phase(self) -> LibraryConfig:
        return self._compile_config()

    def _training_phase(self, library_config: LibraryConfig) -> TrainingMetrics:
        assert self._train is not None, "training was not expected to start"
        return self._train(library_config)


def written_duration(tmp_path: Path) -> float | None:
    payload = json.loads((tmp_path / "customizer_training_result.json").read_text())
    return payload["training_duration_seconds"]


def test_result_json_copies_the_recorded_training_duration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nhx.automodel.tasks.training.runner.get_gpu_info", lambda: None)
    backend = MagicMock()
    backend.find_checkpoints.return_value = {}
    backend.process_checkpoints.return_value = None
    runner = _ScriptedRunner(
        tmp_path,
        duration=17.5,
        compile_config=MagicMock,
        train=lambda library: TrainingMetrics(),
        backend=backend,
    )

    result = runner.run()

    assert result.success is True
    assert result.training_duration_seconds == 17.5
    assert written_duration(tmp_path) == 17.5


def test_result_json_keeps_the_recorded_duration_when_training_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nhx.automodel.tasks.training.runner.get_gpu_info", lambda: None)
    runner = _ScriptedRunner(
        tmp_path,
        duration=9.25,
        compile_config=MagicMock,
        train=MagicMock(side_effect=RuntimeError("train failed")),
        backend=MagicMock(),
    )

    result = runner.run()

    assert result.success is False
    assert result.training_duration_seconds == 9.25
    assert written_duration(tmp_path) == 9.25


def test_a_failure_reports_whether_a_rerun_would_repeat_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nhx.automodel.tasks.training.runner.get_gpu_info", lambda: None)
    runner = _ScriptedRunner(
        tmp_path,
        duration=1.0,
        compile_config=MagicMock,
        train=MagicMock(side_effect=RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")),
        backend=MagicMock(),
    )
    report_error = MagicMock()
    monkeypatch.setattr(runner._progress, "report_error", report_error)

    result = runner.run()

    error_details, status_details = report_error.call_args.args
    assert error_details["cause"] == "gpu_out_of_memory"
    assert error_details["type"] == "GpuOutOfMemoryError"
    assert status_details["resumable"] is False
    assert status_details["non_resumable_reason"] == error_details["message"] == result.error_message


def test_result_json_omits_duration_when_training_never_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nhx.automodel.tasks.training.runner.get_gpu_info", lambda: None)
    runner = _ScriptedRunner(
        tmp_path,
        duration=None,
        compile_config=MagicMock(side_effect=RuntimeError("compile failed")),
        backend=MagicMock(),
    )

    result = runner.run()

    assert result.training_duration_seconds is None
    assert written_duration(tmp_path) is None


@pytest.mark.parametrize(
    ("finetuning_type", "phase"),
    [
        (FinetuningType.LORA_MERGED, "merging_adapter"),
        (FinetuningType.LORA, "finalizing_checkpoint"),
        (FinetuningType.ALL_WEIGHTS, "finalizing_checkpoint"),
    ],
)
def test_postprocessing_reports_the_work_it_is_doing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finetuning_type: FinetuningType, phase: str
) -> None:
    """A LoRA merge loads the whole base model and can take as long as training did."""
    sent: list[str] = []
    backend = MagicMock()
    backend.find_checkpoints.return_value = {}
    backend.process_checkpoints.side_effect = lambda *a, **k: sent.append("process")
    runner = _ScriptedRunner(tmp_path, duration=1.0, compile_config=MagicMock, backend=backend)
    runner._config.training.finetuning_type = finetuning_type
    monkeypatch.setattr(runner._progress, "report_running", lambda reported, **details: sent.append(reported))

    runner._postprocess_phase(None, TrainingMetrics(), MagicMock())

    assert sent == [phase, "process"]


def test_a_paused_run_skips_publication_and_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Training wrote the pause marker, so nothing is selected, published, or reported completed."""
    monkeypatch.setattr("nhx.automodel.tasks.training.runner.get_gpu_info", lambda: None)
    backend = MagicMock()

    def train(library: LibraryConfig) -> TrainingMetrics:
        (tmp_path / "PAUSED").write_text("5\n")
        return TrainingMetrics()

    runner = _ScriptedRunner(tmp_path, duration=5.0, compile_config=MagicMock, train=train, backend=backend)
    synced: list[str] = []
    completed: list[str] = []
    monkeypatch.setattr(runner._dist_ctx, "sync_point", synced.append)
    monkeypatch.setattr(runner._progress, "report_completed", lambda *args, **kwargs: completed.append("completed"))

    result = runner.run()

    assert result.paused and not result.success
    assert synced == []
    assert completed == []
    backend.find_checkpoints.assert_not_called()
    backend.process_checkpoints.assert_not_called()


def _paused_multi_node_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: DistributedRole) -> TrainingResult:
    monkeypatch.setattr("nhx.automodel.tasks.training.runner.get_gpu_info", lambda: None)

    def train(library: LibraryConfig) -> TrainingMetrics:
        (tmp_path / "PAUSED").write_text("5\n")
        return TrainingMetrics()

    runner = _ScriptedRunner(tmp_path, duration=5.0, compile_config=MagicMock, train=train, backend=MagicMock())
    rank = 0 if role == DistributedRole.COORDINATOR else 1
    runner._dist_ctx = DistributedContext(role=role, rank=rank, world_size=2, barrier_dir=tmp_path / "barriers")
    (tmp_path / "barriers" / "config_ready.rank0.ready").touch()
    return runner.run()


def test_a_paused_coordinator_clears_the_barriers_for_the_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume reuses the attempt's barriers, and a worker that starts first must not take this run's marker."""
    result = _paused_multi_node_run(tmp_path, monkeypatch, DistributedRole.COORDINATOR)

    assert result.paused
    assert not (tmp_path / "barriers").exists()


def test_a_paused_worker_leaves_the_barriers_to_the_coordinator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _paused_multi_node_run(tmp_path, monkeypatch, DistributedRole.WORKER)

    assert result.paused
    assert (tmp_path / "barriers" / "config_ready.rank0.ready").exists()


def test_a_pause_fails_when_the_barriers_cannot_be_cleared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The job then ends in error, and a rerun starts a new attempt with its own barriers."""
    monkeypatch.setattr(
        "nhx.automodel.tasks.training.distributed.shutil.rmtree", MagicMock(side_effect=PermissionError("denied"))
    )

    result = _paused_multi_node_run(tmp_path, monkeypatch, DistributedRole.COORDINATOR)

    assert not result.paused and not result.success


def test_the_coordinator_clears_an_earlier_pause_marker_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise a run that finishes after a resume would read the old marker and exit as paused."""
    (tmp_path / "PAUSED").write_text("5\n")
    backend = MagicMock()
    backend.compile_config.return_value = {}
    runner = _ScriptedRunner(tmp_path, duration=None, compile_config=MagicMock, backend=backend)
    signals: list[str] = []
    monkeypatch.setattr(runner._dist_ctx, "signal", signals.append)

    # The real compile phase, which _ScriptedRunner replaces.
    TrainingRunner._compile_config_phase(runner)

    assert not (tmp_path / "PAUSED").exists()
    assert signals == ["config_ready"]


def test_a_resume_tells_training_which_step_it_paused_at(tmp_path: Path) -> None:
    """The marker is gone once training starts, so the step goes in the config training reads."""
    (tmp_path / "PAUSED").write_text("9\n")
    backend = MagicMock()
    backend.compile_config.return_value = {}
    runner = _ScriptedRunner(tmp_path, duration=None, compile_config=MagicMock, backend=backend)

    library = TrainingRunner._compile_config_phase(runner)

    assert library.config_dict["_paused_at_step"] == 9
    assert yaml.safe_load(library.config_path.read_text())["_paused_at_step"] == 9


def test_a_first_run_is_not_told_it_resumes(tmp_path: Path) -> None:
    backend = MagicMock()
    backend.compile_config.return_value = {}
    runner = _ScriptedRunner(tmp_path, duration=None, compile_config=MagicMock, backend=backend)

    library = TrainingRunner._compile_config_phase(runner)

    assert "_paused_at_step" not in library.config_dict


def test_a_paused_run_exits_with_the_pause_code(monkeypatch: pytest.MonkeyPatch) -> None:
    from nhx.automodel.tasks.training import __main__ as entrypoint
    from nhx.automodel.tasks.training.schemas import TrainingResult

    class _Runner:
        def __enter__(self) -> "_Runner":
            return self

        def __exit__(self, *exc: object) -> None: ...

        def run(self) -> TrainingResult:
            return TrainingResult(success=False, paused=True)

    monkeypatch.setattr(entrypoint, "TrainingRunner", _Runner)

    assert entrypoint.run() == 75
