# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clone a registered agent's config with the gateway model swapped.

Eval only runs ``nemo-agents-spec-v1``. The model string lives on
``models.<name>.model`` and on ``harnesses.<name>.model.model``. ``base_url``,
``provider``, and ``settings`` stay as they are. The gateway reads the model
string as the request body on the shared route.
"""

from __future__ import annotations

import copy
from typing import Any

from nemo_distillation_plugin.models import FABRIC_AGENT_CONFIG_FORMAT, GraphError


def require_fabric(config_format: str, config: dict[str, Any]) -> None:
    declared = config.get("config_format", config_format)
    if config_format != FABRIC_AGENT_CONFIG_FORMAT or declared != FABRIC_AGENT_CONFIG_FORMAT:
        raise GraphError(f"agent eval only runs {FABRIC_AGENT_CONFIG_FORMAT} agents, got {config_format!r}")


def clone_config(config: dict[str, Any], *, name: str, model: str | None) -> dict[str, Any]:
    """Deep-copy ``config``, set its name, and point every model entry at ``model``."""
    cloned: dict[str, Any] = copy.deepcopy(config)
    cloned["name"] = name
    cloned["config_format"] = FABRIC_AGENT_CONFIG_FORMAT
    if model is None:
        return cloned
    models = cloned.get("models")
    if isinstance(models, dict):
        for entry in models.values():
            if isinstance(entry, dict):
                entry["model"] = model
    harnesses = cloned.get("harnesses")
    if isinstance(harnesses, dict):
        for harness in harnesses.values():
            if not isinstance(harness, dict):
                continue
            model_cfg = harness.get("model")
            if isinstance(model_cfg, dict):
                model_cfg["model"] = model
            elif isinstance(model_cfg, str):
                harness["model"] = model
    return cloned
