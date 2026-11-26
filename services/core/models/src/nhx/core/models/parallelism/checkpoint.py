# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read vision, audio, expert, and multi-token-prediction traits from a checkpoint config.

The model-spec task stores the result on the model entity. Training compilers do not
open ``config.json`` again; they read the fields the job compiler copied onto the step.
"""

from __future__ import annotations

import json
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
    vision_patch_size: int | None = None


_PROCESSOR_CONFIG_FILES = ("processor_config.json", "preprocessor_config.json")
_NESTED_PROCESSOR_BLOCKS = ("image_processor", "video_processor")


def processor_config_keys(config: Any) -> list[str]:
    """Keys a processor config file sets, including those inside its nested image and video blocks."""
    if not isinstance(config, dict):
        return []
    keys = {key for key in config if key not in _NESTED_PROCESSOR_BLOCKS}
    for block in _NESTED_PROCESSOR_BLOCKS:
        if isinstance(config.get(block), dict):
            keys.update(config[block])
    return sorted(keys)


def read_processor_keys(pretrained_or_path: str) -> list[str] | None:
    """Keys from the processor config files of a local directory or hub id; ``None`` when none can be read."""
    from transformers.utils import cached_file

    keys: set[str] = set()
    for name in _PROCESSOR_CONFIG_FILES:
        try:
            path = cached_file(pretrained_or_path, name, _raise_exceptions_for_missing_entries=False)
            if path:
                with open(path, encoding="utf-8") as handle:
                    keys.update(processor_config_keys(json.load(handle)))
        except (OSError, ValueError):
            continue
    return sorted(keys) or None


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
        vision_patch_size=_vision_patch_size(blocks),
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


def _vision_patch_size(blocks: list[Any]) -> int | None:
    """Side length, in pixels, of one image token.

    A Qwen-style vision config declares ``patch_size`` and ``spatial_merge_size``.
    One token covers ``patch_size * spatial_merge_size`` pixels on a side.
    A Nemotron-style root config declares ``downsample_ratio`` instead, the fraction by which
    the patch grid shrinks per side, so the merge is its inverse (0.5 merges 2x2 patches).
    """
    for block in blocks:
        vision = _get(block, "vision_config")
        if not _is_present_block(vision):
            continue
        patch = _positive_int(_get(vision, "patch_size"))
        if patch is None:
            continue
        merge = (
            _positive_int(_get(vision, "spatial_merge_size"))
            or _positive_int(_get(vision, "merge_size"))
            or _downsample_merge(blocks)
            or 1
        )
        return patch * merge
    return None


def _downsample_merge(blocks: list[Any]) -> int | None:
    """Patches merged per side, from a ``downsample_ratio`` in (0, 1)."""
    for block in blocks:
        ratio = _get(block, "downsample_ratio")
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 < ratio < 1:
            continue
        merge = round(1 / ratio)
        if merge > 1:
            return merge
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


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _is_present_block(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return False
    if isinstance(value, dict):
        return bool(value)
    return True
