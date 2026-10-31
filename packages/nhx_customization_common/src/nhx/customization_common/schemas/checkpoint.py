# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint traits the model-spec task stored, as the job compilers pass them on.

The models service inspects the checkpoint once and writes the result onto the
model entity. These helpers only read that spec. A missing vision field is a
text checkpoint: vision-language models were not loaded before these fields
existed, so an older entity is not one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Applied only when the caller named neither target_modules nor exclude_modules.
# ``*proj`` is the API backfill for that same unset case, so it is not a caller filter.
VLM_LORA_EXCLUDE_MODULES = [
    "*vision_tower*",
    "*vision_model*",
    "*visual*",
    "*image_encoder*",
    "*multi_modal_projector*",
    "*mlp1*",
    "*lm_head*",
]


@dataclass(frozen=True)
class CheckpointTraits:
    """What a training compiler needs from the model spec."""

    has_vision_tower: bool = False
    has_audio_tower: bool = False
    expert_count: int | None = None
    num_nextn_predict_layers: int | None = None
    checkpoint_model_name: str | None = None

    @property
    def declares_nextn_predict_layers(self) -> bool:
        return self.num_nextn_predict_layers is not None


def lora_uses_default_targets(target_modules: list[str] | None, exclude_modules: list[str] | None) -> bool:
    """True when the caller did not name a LoRA filter.

    The automodel API writes ``["*proj"]`` when both lists were empty. That
    default would adapt the vision tower, so a vision job replaces it.
    """
    if exclude_modules:
        return False
    if not target_modules:
        return True
    return list(target_modules) == ["*proj"]


def _fields_set(spec: Any) -> set[str]:
    fields = getattr(spec, "model_fields_set", getattr(spec, "__fields_set__", set()))
    return set(fields or ())


def traits_from_model_spec(spec: Any) -> CheckpointTraits:
    """Read traits off a model entity spec.

    A missing vision field stays false. Expert count still falls back to
    ``moe_config``, which older text checkpoints already carry.
    """
    if spec is None:
        return CheckpointTraits()

    fields = _fields_set(spec)
    nextn = (
        _optional_int(getattr(spec, "num_nextn_predict_layers", None)) if "num_nextn_predict_layers" in fields else None
    )
    name = getattr(spec, "checkpoint_model_name", None)
    return CheckpointTraits(
        has_vision_tower=bool(getattr(spec, "has_vision_tower", False)),
        has_audio_tower=bool(getattr(spec, "has_audio_tower", False)),
        expert_count=_expert_count(spec, fields),
        num_nextn_predict_layers=nextn,
        checkpoint_model_name=name if isinstance(name, str) else None,
    )


def traits_from_step_model(model: Any) -> CheckpointTraits:
    """Read the traits the job compiler copied onto the training step."""
    return CheckpointTraits(
        has_vision_tower=bool(getattr(model, "has_vision_tower", False)),
        has_audio_tower=bool(getattr(model, "has_audio_tower", False)),
        expert_count=_optional_int(getattr(model, "num_experts", None)),
        num_nextn_predict_layers=_optional_int(getattr(model, "num_nextn_predict_layers", None)),
        checkpoint_model_name=_optional_str(getattr(model, "checkpoint_model_name", None)),
    )


def reject_incompatible_checkpoint(
    traits: CheckpointTraits,
    training_type: str,
    recipe: str | None,
    sequence_packing: bool,
    expert_parallel_size: int | None,
    num_gpus_per_node: int,
    num_nodes: int,
) -> None:
    """Reject pairings the training image would otherwise discover after download."""
    if traits.has_vision_tower:
        if training_type == "distillation":
            raise ValueError(
                "Knowledge distillation is not supported for a vision-language checkpoint; "
                "use the sft recipe without a teacher model."
            )
        if sequence_packing:
            raise ValueError("Sequence packing is not supported for a vision-language checkpoint.")
        if recipe in {"bi_encoder", "cross_encoder"}:
            raise ValueError(f"Recipe '{recipe}' cannot train a checkpoint with a vision tower. Use 'sft' or 'auto'.")
        if expert_parallel_size is not None and expert_parallel_size > num_gpus_per_node:
            raise ValueError(
                f"expert_parallel_size ({expert_parallel_size}) must stay inside the node "
                f"(num_gpus_per_node={num_gpus_per_node})."
            )

    experts = traits.expert_count
    if experts is not None and experts > 1 and expert_parallel_size not in (None, 0) and expert_parallel_size > 1:
        if experts % expert_parallel_size != 0:
            raise ValueError(f"expert_parallel_size ({expert_parallel_size}) must divide num_experts ({experts}).")

    # A vision MoE job leaves expert parallel unset and the compiler fills it with the
    # GPUs on the node. A text MoE job has no such default, so reject it here.
    total_gpus = num_nodes * num_gpus_per_node
    if (
        experts is not None
        and experts > 1
        and total_gpus > 1
        and not traits.has_vision_tower
        and (expert_parallel_size is None or expert_parallel_size <= 1)
    ):
        raise ValueError(
            f"MoE model detected (num_experts={experts}) but expert_parallel_size "
            f"is {expert_parallel_size or 'not set'}. Multi-GPU MoE training requires expert_parallel_size > 1."
        )


def _expert_count(spec: Any, fields: set[str]) -> int | None:
    if "num_experts" in fields:
        return _optional_int(getattr(spec, "num_experts", None))
    moe = getattr(spec, "moe_config", None)
    if moe is None:
        return None
    return _optional_int(getattr(moe, "num_experts", None))


def _optional_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return None
