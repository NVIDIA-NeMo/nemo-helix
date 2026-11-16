# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read vision, audio, expert, and multi-token-prediction traits from a checkpoint config.

The model-spec task stores the result on the model entity. Training compilers do not
open ``config.json`` again; they read the fields the job compiler copied onto the step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# One level of names Hugging Face uses when the language model is not the root.
# Nemotron Omni keeps ``n_routed_experts`` on the nested language config.
_NESTED_CONFIG_KEYS = (
    "text_config",
    "thinker_config",
    "language_config",
    "llm_config",
)

_EXPERT_COUNT_KEYS = ("num_local_experts", "num_experts", "n_routed_experts")


@dataclass(frozen=True)
class CheckpointInspection:
    """Facts the model spec records from a checkpoint config."""

    has_vision_tower: bool = False
    has_audio_tower: bool = False
    expert_count: int | None = None
    num_nextn_predict_layers: int | None = None


def inspect_checkpoint(config: Any) -> CheckpointInspection:
    """Inspect a config object or a ``config.json`` dict."""
    if not config:
        return CheckpointInspection()
    blocks = iter_config_blocks(config)
    return CheckpointInspection(
        has_vision_tower=any(_is_present_block(_get(block, "vision_config")) for block in blocks),
        has_audio_tower=any(_is_present_block(_get(block, "audio_config")) for block in blocks),
        expert_count=_expert_count(blocks),
        num_nextn_predict_layers=_nextn_predict_layers(blocks),
    )


def iter_config_blocks(root: Any) -> list[Any]:
    """Yield the root config and nested language configs, root first."""
    found = [root]
    if isinstance(root, (str, int, float, bool)):
        return found
    seen = {id(root)}
    pending = [root]
    while pending:
        current = pending.pop()
        for name in _NESTED_CONFIG_KEYS:
            child = _get(current, name)
            if child is None or id(child) in seen:
                continue
            if isinstance(child, (str, int, float, bool, list, tuple)):
                continue
            seen.add(id(child))
            found.append(child)
            pending.append(child)
    return found


def _expert_count(blocks: list[Any]) -> int | None:
    for block in blocks:
        for key in _EXPERT_COUNT_KEYS:
            value = _get(block, key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 1:
                return value
    return None


def _nextn_predict_layers(blocks: list[Any]) -> int | None:
    for block in blocks:
        value = _get(block, "num_nextn_predict_layers")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _get(config: Any, name: str) -> Any:
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _is_present_block(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return False
    if isinstance(value, dict):
        return bool(value)
    return True
