# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the canonical Unsloth schemas.

Submitter-side validation (mutexes, required fields, defaults) lives
with ``UnslothJobInput`` in the plugin. These tests pin only the
canonical-shape contract that ``train_sft`` and ``compile`` consume.
"""

from __future__ import annotations

import pytest
from nhx.unsloth.schemas import (
    DatasetSpec,
    LoRAParams,
    ModelLoadSpec,
    OutputResponse,
    ScheduleSpec,
    TrainingSpec,
    UnslothJobOutput,
)
from pydantic import ValidationError


def _canonical_dict() -> dict[str, object]:
    return {
        "model": {"name": "unsloth/Qwen2.5-0.5B-Instruct"},
        "dataset": {"path": "/d.jsonl"},
        "training": {"lora": {"rank": 8, "alpha": 16}},
        "schedule": {"max_steps": 60},
        "output": {
            "name": "run-1",
            "type": "adapter",
            "save_method": "lora",
            "fileset": "run-1",
        },
    }


class TestUnslothJobOutput:
    def test_minimal_canonical_validates(self) -> None:
        out = UnslothJobOutput.model_validate(_canonical_dict())
        assert out.output.name == "run-1"
        assert out.output.type == "adapter"
        assert out.output.save_method == "lora"

    def test_output_required(self) -> None:
        payload = _canonical_dict()
        del payload["output"]
        with pytest.raises(ValidationError):
            UnslothJobOutput.model_validate(payload)

    def test_lora_default_target_modules_count(self) -> None:
        # Unsloth's recommended 7-module set lives in this canonical shape.
        lora = LoRAParams()
        assert len(lora.target_modules) == 7

    def test_extra_forbidden_top_level(self) -> None:
        payload = _canonical_dict()
        payload["mystery_field"] = "boom"
        with pytest.raises(ValidationError):
            UnslothJobOutput.model_validate(payload)


class TestSubShapesIndependently:
    def test_dataset_path_required(self) -> None:
        with pytest.raises(ValidationError):
            DatasetSpec.model_validate({})

    def test_model_name_required(self) -> None:
        with pytest.raises(ValidationError):
            ModelLoadSpec.model_validate({})

    def test_schedule_defaults_pass_through(self) -> None:
        # Consistent with Automodel: epochs defaults to 1; max_steps (when set) overrides it.
        sched = ScheduleSpec()
        assert sched.epochs == 1
        assert sched.max_steps is None

    def test_the_hf_logging_cadence_is_not_a_jobs_reporting_knob(self) -> None:
        """`logging_steps` drives stdout and W&B; it never gated Jobs reporting.

        Before the shared naming rule landed it accidentally was the only thing
        throttling Jobs reports, so raising it from 1 gave proportionally fewer.
        Nothing throttles now -- every step the trainer logs is reported -- so
        this is purely HuggingFace's knob again.
        """
        assert ScheduleSpec().logging_steps == 1

    def test_a_misspelled_reporting_field_is_rejected(self) -> None:
        """The fragment is not a NamespacedModel, so it had to forbid extras itself.

        Every model it is embedded in inherits `extra="forbid"`, and this module
        states the contract: typos become validation errors, not silently-ignored
        fields. Being the one nested object that ignored them made
        `time_series_metric` a spelling that parsed and then did nothing.
        """
        with pytest.raises(ValidationError):
            ScheduleSpec.model_validate({"progress_reporting": {"time_series_metric": ["*_loss"]}})

    def test_the_time_series_metrics_default_to_the_backends_choice(self) -> None:
        """Narrowing is opt-in, so adding the knob changed nobody's curves."""
        assert ScheduleSpec().progress_reporting.time_series_metrics is None
        assert ScheduleSpec(
            progress_reporting={"time_series_metrics": ["*_loss"]}
        ).progress_reporting.time_series_metrics == ["*_loss"]

    def test_training_defaults(self) -> None:
        t = TrainingSpec()
        assert t.training_type == "sft"
        assert t.finetuning_type == "lora"
        # finetuning_type defaults to 'lora', so the schema auto-fills a default
        # LoRAParams block — downstream (build_peft_kwargs) can rely on it.
        assert t.lora == LoRAParams()

    def test_lora_finetuning_autofills_lora(self) -> None:
        # Explicit lora type with no block → default LoRAParams, never None.
        t = TrainingSpec(finetuning_type="lora", lora=None)
        assert t.lora == LoRAParams()

    def test_all_weights_rejects_lora_block(self) -> None:
        with pytest.raises(ValidationError, match="training.lora must be unset"):
            TrainingSpec(finetuning_type="all_weights", lora=LoRAParams())

    def test_all_weights_allows_no_lora(self) -> None:
        t = TrainingSpec(finetuning_type="all_weights")
        assert t.lora is None

    def test_output_response_extras_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            OutputResponse.model_validate(
                {
                    "name": "x",
                    "type": "adapter",
                    "save_method": "lora",
                    "fileset": "x",
                    "junk": 1,
                }
            )

    def test_output_response_requires_fileset(self) -> None:
        with pytest.raises(ValidationError, match="fileset"):
            OutputResponse.model_validate({"name": "x", "type": "adapter", "save_method": "lora"})
