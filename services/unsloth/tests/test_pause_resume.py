# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pause and resume for the Unsloth trainer."""

from __future__ import annotations

import json
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from nemo_helix_plugin.job_context import JobContext
from nhx.customization_common.training.pause import ResumeCheckpointMissingError
from nhx.unsloth.schemas import DatasetSpec, ModelLoadSpec, OutputResponse, TrainingSpec, UnslothJobOutput
from nhx.unsloth.tasks.training.backends import unsloth_sft
from nhx.unsloth.tasks.training.backends.callbacks import TrainingProgressCallback
from nhx.unsloth.tasks.training.backends.pause import create_pause_callback, newest_complete_checkpoint


def _checkpoint(root: Path, step: int, *, trainer_state: str | None) -> Path:
    path = root / f"checkpoint-{step}"
    path.mkdir(parents=True)
    if trainer_state is not None:
        (path / "trainer_state.json").write_text(trainer_state)
    return path


def _complete(root: Path, step: int) -> Path:
    return _checkpoint(root, step, trainer_state=json.dumps({"global_step": step}))


def test_the_newest_complete_checkpoint_is_resumed_and_incomplete_ones_are_removed(tmp_path: Path) -> None:
    _complete(tmp_path, 20)
    newest = _complete(tmp_path, 40)
    no_state = _checkpoint(tmp_path, 60, trainer_state=None)
    torn_state = _checkpoint(tmp_path, 80, trainer_state='{"global_st')

    assert newest_complete_checkpoint(tmp_path) == (newest, 40)
    assert not no_state.exists()
    assert not torn_state.exists()


@pytest.mark.parametrize("error", [PermissionError(13, "Permission denied"), TimeoutError(110, "Timed out")])
def test_a_trainer_state_that_cannot_be_read_fails_without_removing_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: OSError
) -> None:
    checkpoint = _complete(tmp_path, 40)
    read_text = Path.read_text

    def failing_read(path: Path, *args: Any, **kwargs: Any) -> str:
        if path.name == "trainer_state.json":
            raise error
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", failing_read)

    with pytest.raises(type(error)):
        newest_complete_checkpoint(tmp_path)
    assert checkpoint.exists()


def test_without_a_checkpoint_training_starts_from_scratch(tmp_path: Path) -> None:
    assert newest_complete_checkpoint(tmp_path / "missing") is None
    assert newest_complete_checkpoint(tmp_path) is None


class _Flow:
    """The trainer's callback handling: its flow callback first, then ours."""

    def __init__(self, *, save_strategy: str = "epoch", max_steps: int = 30) -> None:
        pytest.importorskip("transformers")
        from transformers.trainer_callback import CallbackHandler, DefaultFlowCallback, TrainerControl, TrainerState
        from transformers.training_args import TrainingArguments

        self.requested = threading.Event()
        self.pause = create_pause_callback(self.requested)
        self.handler = CallbackHandler([DefaultFlowCallback, self.pause], None, None, None, None)
        # Only the fields DefaultFlowCallback reads.
        self.args = cast(
            TrainingArguments,
            SimpleNamespace(
                logging_first_step=False,
                logging_strategy="steps",
                eval_strategy="epoch",
                eval_delay=0,
                save_strategy=save_strategy,
            ),
        )
        self.state = TrainerState(max_steps=max_steps, logging_steps=10, eval_steps=10, save_steps=10)
        self.control = TrainerControl()

    def step_end(self, step: int) -> Any:
        self.state.global_step = step
        self.control = self.handler.on_step_end(self.args, self.state, self.control)
        return self.control

    def epoch_end(self) -> Any:
        self.state.epoch = 1.0
        self.control = self.handler.on_epoch_end(self.args, self.state, self.control)
        return self.control


def test_a_requested_pause_saves_and_stops_at_the_end_of_the_step() -> None:
    flow = _Flow()
    assert not flow.step_end(5).should_save

    flow.requested.set()
    control = flow.step_end(6)

    assert flow.pause.stopped
    assert control.should_save and control.should_training_stop and control.should_log
    assert not control.should_evaluate


def test_the_pause_step_is_saved_once() -> None:
    """Stopping ends the epoch, and an end-of-epoch save would write the same step again."""
    flow = _Flow(save_strategy="epoch")
    flow.requested.set()
    flow.step_end(6)
    flow.control.should_save = False  # the trainer saved, and its on_save cleared this

    control = flow.epoch_end()

    assert not control.should_save
    assert not control.should_evaluate


def test_an_epoch_end_without_a_pause_still_saves() -> None:
    flow = _Flow(save_strategy="epoch")

    assert flow.epoch_end().should_save


def test_a_pause_in_the_last_step_lets_training_finish() -> None:
    flow = _Flow(max_steps=30)
    flow.requested.set()

    control = flow.step_end(30)

    assert not flow.pause.stopped
    assert control.should_training_stop


class _Progress:
    def __init__(self) -> None:
        self.reported: list[str] = []
        self.reporter = SimpleNamespace(
            training_wall_clock=lambda: _NullClock(),
            report_completed=lambda message="Completed": self.reported.append("completed"),
        )
        self.rewound_to: list[int] = []

    def rewind_to(self, step: int) -> None:
        self.rewound_to.append(step)

    def report_phase(self, phase: str) -> None:
        self.reported.append(str(phase))

    def close(self) -> None:
        return None


class _NullClock:
    def __enter__(self) -> _NullClock:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _Poller:
    """Reports a pause as soon as training starts, when ``pause`` is set."""

    pause = False

    def __init__(self, job_ctx: object, service_name: str) -> None:
        self.requested = threading.Event()
        if self.pause:
            self.requested.set()

    def __enter__(self) -> _Poller:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@pytest.fixture
def run_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Run ``train_sft`` against stand-ins for unsloth, trl and datasets; record what reaches the trainer."""
    seen: dict[str, Any] = {"saved": False}

    unsloth_mod = types.ModuleType("unsloth")
    setattr(unsloth_mod, "FastLanguageModel", SimpleNamespace(from_pretrained=lambda **kwargs: (object(), object())))
    monkeypatch.setitem(sys.modules, "unsloth", unsloth_mod)
    datasets_mod = types.ModuleType("datasets")
    setattr(datasets_mod, "Dataset", object)
    setattr(datasets_mod, "load_dataset", lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)

    class SFTConfig:
        def __init__(self, **kwargs: Any) -> None:
            seen["args"] = kwargs

    class SFTTrainer:
        def __init__(self, **kwargs: Any) -> None:
            self.callbacks = kwargs["callbacks"]
            self.state = SimpleNamespace(global_step=0)

        def train(self, resume_from_checkpoint: str | None = None) -> SimpleNamespace:
            seen["resume_from_checkpoint"] = resume_from_checkpoint
            pause = self.callbacks[-1]
            if _Poller.pause:
                pause.stopped = True
                self.state.global_step = 7
            return SimpleNamespace(training_loss=0.5)

    trl_mod = types.ModuleType("trl")
    setattr(trl_mod, "SFTConfig", SFTConfig)
    setattr(trl_mod, "SFTTrainer", SFTTrainer)
    monkeypatch.setitem(sys.modules, "trl", trl_mod)

    monkeypatch.setattr(unsloth_sft.NHXJobContext, "from_env", staticmethod(lambda: SimpleNamespace()))
    monkeypatch.setattr(unsloth_sft, "apply_integrations_to_sft_config", lambda **kwargs: ([], {}, {}))
    monkeypatch.setattr(unsloth_sft, "_load_datasets", lambda **kwargs: (["row"], None))
    monkeypatch.setattr(unsloth_sft, "_save_model", lambda *args, **kwargs: seen.update(saved=True) or args[2])
    monkeypatch.setattr(
        "nhx.unsloth.tasks.training.backends.hf_trainer_callback.create_hf_trainer_progress_callback",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(unsloth_sft, "PausePoller", _Poller)
    monkeypatch.setattr(unsloth_sft, "create_pause_callback", lambda requested: SimpleNamespace(stopped=False))
    monkeypatch.setattr(_Poller, "pause", False)

    spec = UnslothJobOutput(
        model=ModelLoadSpec(name="meta/llama", load_in_4bit=False),
        dataset=DatasetSpec(path="train"),
        training=TrainingSpec(finetuning_type="all_weights"),
        output=OutputResponse(name="out", type="model", save_method="merged_16bit", fileset="out"),
    )
    checkpoints = tmp_path / "training" / "checkpoints"
    output = tmp_path / "output_model"

    def run(*, pause: bool = False) -> tuple[dict[str, Any], dict[str, Any], _Progress]:
        _Poller.pause = pause
        progress = _Progress()
        # Kept for a test whose run raises before returning it.
        setattr(run, "progress", progress)
        result = unsloth_sft.train_sft(
            spec,
            cast(JobContext, SimpleNamespace(job_id="", workspace="default")),
            output_path=str(output),
            checkpoint_path=str(checkpoints),
            progress_callback=cast(TrainingProgressCallback, progress),
        )
        return result, seen, progress

    setattr(run, "checkpoints", checkpoints)
    setattr(run, "output", output)
    return run


def test_checkpoints_are_kept_out_of_the_published_model(run_training: Any) -> None:
    result, seen, progress = run_training()

    assert seen["args"]["output_dir"] == str(run_training.checkpoints)
    assert seen["args"]["save_total_limit"] == 1
    assert result["paused"] is False
    assert seen["saved"] and result["output_path"] == str(run_training.output)
    assert progress.reported[-1] == "completed"


def test_training_resumes_from_the_newest_complete_checkpoint(run_training: Any) -> None:
    _complete(run_training.checkpoints, 20)
    newest = _complete(run_training.checkpoints, 40)

    _, seen, progress = run_training()

    assert seen["resume_from_checkpoint"] == str(newest)
    # A rerun after a crash can restore an earlier step than the last one reported.
    assert progress.rewound_to == [40]


def test_a_fresh_run_starts_from_scratch(run_training: Any) -> None:
    _, seen, progress = run_training()

    assert seen["resume_from_checkpoint"] is None
    assert progress.rewound_to == [0]


def test_a_paused_run_skips_the_final_save(run_training: Any) -> None:
    result, seen, progress = run_training(pause=True)

    assert result["paused"] is True
    assert result["global_step"] == 7
    assert not seen["saved"]
    # A pause is not the end of training, so neither the save phase nor completion is reported.
    assert progress.reported == []


def test_a_pause_tells_the_resume_where_it_stopped(run_training: Any) -> None:
    """Beside the checkpoint directory, not in it, so losing the checkpoints does not lose the marker."""
    run_training(pause=True)

    assert (run_training.checkpoints.parent / "PAUSED").read_text(encoding="utf-8") == "7\n"


def test_a_resume_without_a_checkpoint_fails_instead_of_starting_over(run_training: Any) -> None:
    """Starting over would train from step 0 and drop every recorded metric point past it."""
    run_training(pause=True)

    with pytest.raises(ResumeCheckpointMissingError, match="paused at step 7"):
        run_training()

    assert run_training.progress.rewound_to == []
    # The failed resume took the marker, so a rerun of the job starts over.
    result, _, progress = run_training()
    assert result["paused"] is False
    assert progress.rewound_to == [0]


def test_a_resume_continues_from_the_pause_checkpoint(run_training: Any) -> None:
    run_training(pause=True)
    pause_checkpoint = _complete(run_training.checkpoints, 7)

    _, seen, progress = run_training()

    assert seen["resume_from_checkpoint"] == str(pause_checkpoint)
    assert progress.rewound_to == [7]
    assert not (run_training.checkpoints.parent / "PAUSED").exists()
