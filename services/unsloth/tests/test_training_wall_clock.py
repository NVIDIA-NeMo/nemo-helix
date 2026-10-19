# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unsloth's wall clock covers model load and training, and stops before save."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from nhx.unsloth.schemas import DatasetSpec, ModelLoadSpec, OutputResponse, TrainingSpec, UnslothJobOutput
from nhx.unsloth.tasks.training.backends import unsloth_sft


class Clock:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __enter__(self) -> "Clock":
        self.events.append("clock-start")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.events.append("clock-end")
        return False


class Reporter:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def training_wall_clock(self) -> Clock:
        return Clock(self.events)


class Callback:
    def __init__(self, events: list[str]) -> None:
        self.reporter = Reporter(events)
        self.events = events

    def close(self) -> None:
        self.events.append("close")


class _FakeUnslothModule(types.ModuleType):
    """Stand-in ``unsloth`` exposing the model loader the training backend imports."""

    FastLanguageModel: type


class _FakeDatasetsModule(types.ModuleType):
    """Stand-in ``datasets`` exposing what the training backend imports."""

    Dataset: type
    load_dataset: Callable[..., object]


class _FakeTrlModule(types.ModuleType):
    """Stand-in ``trl`` exposing the SFT config and trainer the training backend imports."""

    SFTConfig: type
    SFTTrainer: type


def test_wall_clock_wraps_load_and_train_but_not_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    unsloth_mod = _FakeUnslothModule("unsloth")

    class FastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs: object) -> tuple[object, object]:
            events.append("load")
            return object(), object()

    unsloth_mod.FastLanguageModel = FastLanguageModel
    monkeypatch.setitem(sys.modules, "unsloth", unsloth_mod)

    datasets_mod = _FakeDatasetsModule("datasets")
    datasets_mod.Dataset = object
    datasets_mod.load_dataset = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)

    trl_mod = _FakeTrlModule("trl")

    class SFTConfig:
        def __init__(self, **kwargs: object) -> None:
            return None

    class SFTTrainer:
        def __init__(self, **kwargs: object) -> None:
            return None

        def train(self) -> SimpleNamespace:
            events.append("train")
            return SimpleNamespace(training_loss=0.2)

    trl_mod.SFTConfig = SFTConfig
    trl_mod.SFTTrainer = SFTTrainer
    monkeypatch.setitem(sys.modules, "trl", trl_mod)

    monkeypatch.setattr(unsloth_sft.NHXJobContext, "from_env", staticmethod(lambda: SimpleNamespace()))
    monkeypatch.setattr(
        unsloth_sft,
        "apply_integrations_to_sft_config",
        lambda **kwargs: ([], {}, {}),
    )
    monkeypatch.setattr(unsloth_sft, "_load_training_dataset", lambda **kwargs: ["row"])
    monkeypatch.setattr(
        unsloth_sft,
        "_save_model",
        lambda *args, **kwargs: events.append("save") or tmp_path / "saved",
    )
    monkeypatch.setattr(
        "nhx.unsloth.tasks.training.backends.hf_trainer_callback.create_hf_trainer_progress_callback",
        lambda *args, **kwargs: object(),
    )

    spec = UnslothJobOutput(
        model=ModelLoadSpec(name="meta/llama", load_in_4bit=False),
        dataset=DatasetSpec(path="train"),
        training=TrainingSpec(finetuning_type="all_weights"),
        output=OutputResponse(name="out", type="model", save_method="merged_16bit", fileset="out"),
    )
    result = unsloth_sft.train_sft(
        spec,
        SimpleNamespace(job_id="", workspace="default"),  # type: ignore[arg-type]
        output_path=str(tmp_path / "output"),
        progress_callback=Callback(events),  # type: ignore[arg-type]
    )

    assert events.index("clock-start") < events.index("load") < events.index("train") < events.index("clock-end")
    assert events.index("clock-end") < events.index("close") < events.index("save")
    assert result["loss"] == 0.2
