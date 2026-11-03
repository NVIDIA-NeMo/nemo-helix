# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unsloth's wall clock covers model load and training, and stops before save."""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults
from nhx.customization_common.training.progress import JobsServiceProgressReporter
from nhx.unsloth.schemas import DatasetSpec, ModelLoadSpec, OutputResponse, TrainingSpec, UnslothJobOutput
from nhx.unsloth.tasks.training.backends import unsloth_sft
from nhx.unsloth.tasks.training.backends.callbacks import TrainingProgressCallback


class RecordingReporter(JobsServiceProgressReporter):
    """Records the task updates training sends, and closing, instead of calling the Jobs service."""

    def __init__(self, events: list[str]) -> None:
        # No super().__init__: that builds a Jobs-service client this test never uses.
        self.events = events
        self.training_duration_seconds = None

    def update_task(
        self,
        status: str = "active",
        status_details: dict[str, Any] | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> None:
        details = status_details or {}
        if status == "completed":
            self.events.append("completed")
        elif "phase" in details:
            self.events.append(str(details["phase"]))
        if "training_started_at" in details:
            self.events.append("clock-start")
        if "training_finished_at" in details:
            self.events.append("clock-end")

    def fetch_current_metrics(self) -> None:
        return None

    def close(self) -> None:
        self.events.append("close")


class _IdlePoller:
    def __init__(self, job_ctx: object, service_name: str) -> None:
        self.requested = threading.Event()

    def __enter__(self) -> "_IdlePoller":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_wall_clock_wraps_load_and_train_but_not_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    unsloth_mod = types.ModuleType("unsloth")

    class FastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs: object) -> tuple[object, object]:
            events.append("load")
            return object(), object()

    unsloth_mod.__dict__.update(FastLanguageModel=FastLanguageModel)
    monkeypatch.setitem(sys.modules, "unsloth", unsloth_mod)

    datasets_mod = types.ModuleType("datasets")
    datasets_mod.__dict__.update(Dataset=object, load_dataset=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)

    trl_mod = types.ModuleType("trl")

    class SFTConfig:
        def __init__(self, **kwargs: object) -> None:
            return None

    class SFTTrainer:
        def __init__(self, **kwargs: object) -> None:
            return None

        def train(self, resume_from_checkpoint: str | None = None) -> SimpleNamespace:
            events.append("train")
            return SimpleNamespace(training_loss=0.2)

    trl_mod.__dict__.update(SFTConfig=SFTConfig, SFTTrainer=SFTTrainer)
    monkeypatch.setitem(sys.modules, "trl", trl_mod)

    monkeypatch.setattr(unsloth_sft.NHXJobContext, "from_env", staticmethod(lambda: SimpleNamespace()))
    monkeypatch.setattr(
        unsloth_sft,
        "apply_integrations_to_sft_config",
        lambda **kwargs: ([], {}, {}),
    )
    monkeypatch.setattr(unsloth_sft, "_load_datasets", lambda **kwargs: (["row"], None))
    monkeypatch.setattr(
        unsloth_sft,
        "_save_model",
        lambda *args, **kwargs: events.append("save") or tmp_path / "saved",
    )
    monkeypatch.setattr(
        "nhx.unsloth.tasks.training.backends.hf_trainer_callback.create_hf_trainer_progress_callback",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(unsloth_sft, "PausePoller", _IdlePoller)
    monkeypatch.setattr(unsloth_sft, "create_pause_callback", lambda requested: SimpleNamespace(stopped=False))

    spec = UnslothJobOutput(
        model=ModelLoadSpec(name="meta/llama", load_in_4bit=False),
        dataset=DatasetSpec(path="train"),
        training=TrainingSpec(finetuning_type="all_weights"),
        output=OutputResponse(name="out", type="model", save_method="merged_16bit", fileset="out"),
    )
    result = unsloth_sft.train_sft(
        spec,
        JobContext(
            workspace="default",
            storage=StoragePaths(ephemeral=tmp_path / "ephemeral", persistent=tmp_path / "persistent"),
            results=LocalJobResults(tmp_path / "results"),
        ),
        output_path=str(tmp_path / "output"),
        checkpoint_path=str(tmp_path / "checkpoints"),
        progress_callback=TrainingProgressCallback(RecordingReporter(events)),
    )

    assert events.index("clock-start") < events.index("load") < events.index("train") < events.index("clock-end")
    assert (
        events.index("clock-end")
        < events.index("merging_adapter")
        < events.index("save")
        < events.index("completed")
        < events.index("close")
    )
    assert result["loss"] == 0.2
