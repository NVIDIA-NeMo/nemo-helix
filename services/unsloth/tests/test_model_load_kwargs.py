# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``build_model_load_kwargs`` / ``build_peft_kwargs`` in the unsloth SFT driver.

These exercise the torch-free kwargs assembly only, so the module imports
fine on a CPU box without ``unsloth``/``torch`` installed.
"""

from __future__ import annotations

from typing import Any

from nhx.unsloth.schemas import (
    DatasetSpec,
    LoRAParams,
    ModelLoadSpec,
    OutputResponse,
    TrainingSpec,
    UnslothJobOutput,
)
from nhx.unsloth.tasks.training.backends.unsloth_sft import build_model_load_kwargs, build_peft_kwargs


def _spec(
    *,
    finetuning_type: str = "lora",
    load_in_4bit: bool = True,
    model_extra: dict[str, Any] | None = None,
    lora: LoRAParams | None = None,
) -> UnslothJobOutput:
    return UnslothJobOutput(
        model=ModelLoadSpec(name="meta/llama-3.1-8b", load_in_4bit=load_in_4bit, **(model_extra or {})),
        dataset=DatasetSpec(path="ws/train"),
        training=TrainingSpec(finetuning_type=finetuning_type, lora=lora),
        output=OutputResponse(name="out", type="model", save_method="lora", fileset="out"),
    )


def test_all_weights_sets_full_finetuning() -> None:
    # all-weights FT loads in 16-bit (validator forbids 4/8-bit upstream) and
    # MUST request full_finetuning so Unsloth uses the full-FT load path.
    kwargs = build_model_load_kwargs(_spec(finetuning_type="all_weights", load_in_4bit=False), "/local/model")
    assert kwargs["full_finetuning"] is True
    assert kwargs["load_in_4bit"] is False
    assert kwargs["model_name"] == "/local/model"


def test_lora_does_not_set_full_finetuning() -> None:
    kwargs = build_model_load_kwargs(_spec(finetuning_type="lora", load_in_4bit=True), "/local/model")
    assert kwargs["full_finetuning"] is False
    assert kwargs["load_in_4bit"] is True


def test_dtype_not_included() -> None:
    # dtype mapping needs torch and stays in train_sft; the helper must not emit it.
    kwargs = build_model_load_kwargs(_spec(finetuning_type="lora", load_in_4bit=True), "/local/model")
    assert "dtype" not in kwargs


def test_rope_scaling_omitted_when_none() -> None:
    kwargs = build_model_load_kwargs(_spec(), "/local/model")
    assert "rope_scaling" not in kwargs


def test_rope_scaling_passed_when_set() -> None:
    spec = _spec(model_extra={"rope_scaling": {"type": "linear", "factor": 2.0}})
    kwargs = build_model_load_kwargs(spec, "/local/model")
    assert kwargs["rope_scaling"] == {"type": "linear", "factor": 2.0}


def test_peft_kwargs_defaults_preserve_behavior() -> None:
    # Default LoRAParams → library-default knobs; optional ones omitted (not None).
    kwargs = build_peft_kwargs(_spec(lora=LoRAParams()), gradient_checkpointing="unsloth")
    assert kwargs["r"] == 16
    assert kwargs["use_dora"] is False
    assert kwargs["init_lora_weights"] is True
    assert kwargs["use_gradient_checkpointing"] == "unsloth"
    for omitted in ("loftq_config", "modules_to_save", "layers_to_transform", "layer_replication"):
        assert omitted not in kwargs


def test_peft_kwargs_emits_optional_fields_when_set() -> None:
    lora = LoRAParams(
        use_dora=True,
        init_lora_weights="pissa",
        modules_to_save=["embed_tokens", "lm_head"],
        layers_to_transform=[0, 1, 2],
        layer_replication=[[0, 16], [8, 24]],
        loftq_config={"loftq_bits": 4},
    )
    kwargs = build_peft_kwargs(_spec(lora=lora), gradient_checkpointing=True)
    assert kwargs["use_dora"] is True
    assert kwargs["init_lora_weights"] == "pissa"
    assert kwargs["modules_to_save"] == ["embed_tokens", "lm_head"]
    assert kwargs["layers_to_transform"] == [0, 1, 2]
    assert kwargs["layer_replication"] == [[0, 16], [8, 24]]
    assert kwargs["loftq_config"] == {"loftq_bits": 4}
    assert kwargs["use_gradient_checkpointing"] is True
