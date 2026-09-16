# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply tuned study parameters onto a ``nemo-agents-spec-v1`` agent config.

Search-space paths are author-declared against the Fabric package (e.g.
``models.default.temperature``), but the optimized agent is persisted as a
platform spec, where that value lives on the selected harness's model.  The
Fabric ``models.default`` block is synthesized by the translator and frequently
has no counterpart in the stored config, so this overlay writes only to
locations that already exist and raises when a tuned parameter matches nothing.
Inventing the key instead would produce an agent that looks optimized and is
not.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Declared fields of ``nemo_agents_plugin.agent_config.ModelConfig``.  A tuned
#: leaf outside this set is not a model field and belongs in ``settings``.
_MODEL_FIELDS = frozenset({"provider", "model", "api_key_env", "base_url", "temperature"})

_FABRIC_DEFAULT_MODEL_PREFIX = "models.default."


class SpecOverlayError(RuntimeError):
    """Raised when a tuned parameter cannot be applied to the agent spec."""


def apply_tuned_params_to_spec(spec_config: dict[str, Any], by_path: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *spec_config* with each tuned path in *by_path* applied."""
    optimized = copy.deepcopy(spec_config)
    for dotted_path, value in by_path.items():
        if dotted_path.startswith(_FABRIC_DEFAULT_MODEL_PREFIX):
            leaf = dotted_path[len(_FABRIC_DEFAULT_MODEL_PREFIX) :]
            applied = _apply_to_model_blocks(optimized, leaf, value)
        else:
            applied = _set_if_exists(optimized, dotted_path.split("."), value)
        if not applied:
            raise SpecOverlayError(
                f"Tuned parameter {dotted_path!r} matched nothing in the agent spec. "
                "Search-space paths must address a value the stored agent actually has — "
                "for a model parameter, the selected harness's model block."
            )
        logger.info("Applied tuned parameter %s=%r to the agent spec", dotted_path, value)
    return optimized


def _apply_to_model_blocks(spec_config: dict[str, Any], leaf: str, value: Any) -> bool:
    """Write *leaf* onto every model block the spec actually has.

    Mirrors switchyard's ``_rewrite_model``: the default harness's model is the
    real target, and an explicit ``models.default`` is updated too when present
    so the two cannot drift apart.
    """
    applied = False
    targets: list[dict[str, Any]] = []

    harnesses = spec_config.get("harnesses")
    default_harness = spec_config.get("default_harness")
    if isinstance(harnesses, dict) and isinstance(harnesses.get(default_harness), dict):
        model = harnesses[default_harness].get("model")
        if isinstance(model, dict):
            targets.append(model)

    models = spec_config.get("models")
    if isinstance(models, dict) and isinstance(models.get("default"), dict):
        targets.append(models["default"])

    for target in targets:
        if leaf in _MODEL_FIELDS:
            target[leaf] = value
        else:
            settings = target.setdefault("settings", {})
            if not isinstance(settings, dict):
                continue
            settings[leaf] = value
        applied = True
    return applied


def _set_if_exists(config: dict[str, Any], keys: list[str], value: Any) -> bool:
    """Set a dotted path only when every intermediate mapping already exists."""
    cursor: Any = config
    for key in keys[:-1]:
        if not isinstance(cursor, dict) or not isinstance(cursor.get(key), dict):
            return False
        cursor = cursor[key]
    if not isinstance(cursor, dict) or keys[-1] not in cursor:
        return False
    cursor[keys[-1]] = value
    return True
