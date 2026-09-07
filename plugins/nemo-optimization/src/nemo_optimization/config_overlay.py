# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dotted-path config overlay and Fabric profile overlay helpers."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

_OPTIMIZER_ONLY_TOP_LEVEL_KEYS = frozenset({"optimizer", "optimizable_params"})


def set_by_dotted_path(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set ``value`` on ``config`` at a dotted path, creating intermediate dicts."""
    keys = dotted_path.split(".")
    cursor = config
    for key in keys[:-1]:
        existing = cursor.get(key)
        if existing is None:
            cursor[key] = {}
        elif not isinstance(existing, dict):
            raise KeyError(
                f"Cannot set {dotted_path!r}: segment {key!r} is not a mapping (got {type(existing).__name__})."
            )
        cursor = cursor[key]
    cursor[keys[-1]] = value


def get_by_dotted_path(config: Mapping[str, Any], dotted_path: str) -> Any:
    """Return the value at ``dotted_path`` without creating intermediate nodes."""

    keys = dotted_path.split(".")
    if not keys or any(not key for key in keys):
        raise KeyError(f"Invalid dotted path {dotted_path!r}.")

    cursor: Any = config
    for key in keys:
        if not isinstance(cursor, Mapping):
            raise KeyError(
                f"Cannot resolve {dotted_path!r}: segment {key!r} is under a {type(cursor).__name__}, not a mapping."
            )
        if key not in cursor:
            raise KeyError(f"Cannot resolve {dotted_path!r}: missing segment {key!r}.")
        cursor = cursor[key]
    return cursor


def nest_dotted_paths(flat: Mapping[str, Any]) -> dict[str, Any]:
    """Convert ``{'models.default.temperature': 0.2}`` into nested mappings."""
    root: dict[str, Any] = {}
    for dotted_path, value in flat.items():
        keys = dotted_path.split(".")
        cursor = root
        for key in keys[:-1]:
            child = cursor.get(key)
            if child is None:
                child = {}
                cursor[key] = child
            elif not isinstance(child, dict):
                raise KeyError(f"Cannot nest {dotted_path!r}: segment {key!r} is not a mapping.")
            cursor = child
        cursor[keys[-1]] = value
    return root


def strip_optimizer_only_fields(config: dict[str, Any]) -> None:
    """Remove optimizer metadata from a trial config artifact in place."""
    for key in _OPTIMIZER_ONLY_TOP_LEVEL_KEYS:
        config.pop(key, None)
    optimizer = config.get("optimizer")
    if isinstance(optimizer, dict):
        optimizer.pop("search_space", None)
        optimizer.pop("optimizable_params", None)


def apply_suggestions(
    base_config: Mapping[str, Any],
    suggestions: Mapping[str, Any],
    *,
    strip_optimizer: bool = True,
) -> dict[str, Any]:
    """Return a deep copy of ``base_config`` with trial suggestions overlaid."""
    trial_config = copy.deepcopy(dict(base_config))
    for dotted_path, value in suggestions.items():
        set_by_dotted_path(trial_config, dotted_path, value)
    if strip_optimizer:
        strip_optimizer_only_fields(trial_config)
    return trial_config


def suggestions_to_profile_overlay(suggestions: Mapping[str, Any], trial_number: int) -> dict[str, Any]:
    """Build a Fabric profile overlay dict for per-trial HPO parameters."""
    overlay: dict[str, Any] = {
        "schema_version": "fabric.profile/v1alpha1",
        "metadata": {"name": f"trial-{trial_number:03d}"},
    }
    nested = nest_dotted_paths(suggestions)
    for key, value in nested.items():
        if key in overlay and isinstance(overlay[key], dict) and isinstance(value, dict):
            overlay[key] = {**overlay[key], **value}
        else:
            overlay[key] = value
    return overlay
