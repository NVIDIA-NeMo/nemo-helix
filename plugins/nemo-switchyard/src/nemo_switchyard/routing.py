# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure helpers for the ``switchyard`` strategy: model-pair combinations and agent config rewrites."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from nemo_helix_plugin.entity_naming import NAME_MAX_LENGTH
from nemo_helix_plugin.errors import LocalRunError
from nemo_helix_plugin.refs import parse_entity_ref
from nemo_switchyard.schemas.optimize import SwitchyardOptimizeSpec

#: Placeholder credential the platform swaps for a gateway token at run time (duplicated from
#: nemo_agents_plugin.fabric.gateway_credentials so this module stays free of nemo-agents).
IGW_API_KEY_ENV = "NEMO_AGENTS_IGW_API_KEY"
#: Providers for which the platform binds the Inference Gateway URL when base_url is unset.
GATEWAY_PROVIDERS = frozenset({"openai", "nvidia", "openai-compatible"})


@dataclass(frozen=True)
class Combination:
    virtual_model: str
    config_type: str
    capable: str
    efficient: str
    judge: str | None
    config: dict[str, Any]

    @property
    def models(self) -> list[str]:
        return list(dict.fromkeys(ref for ref in (self.judge, self.capable, self.efficient) if ref))


def build_combinations(spec: SwitchyardOptimizeSpec, *, agent_name: str) -> list[Combination]:
    refs = [_qualify(ref, spec.workspace) for ref in spec.models]
    judge = _qualify(spec.judge_model, spec.workspace) if spec.judge_model else None
    built: list[Combination] = []
    for capable, efficient in combinations(refs, 2):
        for config_type in spec.routing_strategies:
            pair_judge = (judge or capable) if config_type == "llm_classifier" else None
            config = middleware_config(spec, config_type, capable=capable, efficient=efficient, judge=pair_judge)
            # Keyed on the routing itself so a re-run with the same request reuses the VirtualModel
            # and a different model list or setting never collides with an earlier run's.
            virtual_model = f"{agent_name}-{config_type.replace('_', '-')}-{_routing_digest(config_type, config)}"
            # Checked up front so a long agent name fails before any VirtualModel is created.
            if len(virtual_model) > NAME_MAX_LENGTH:
                raise LocalRunError(
                    f"VirtualModel name {virtual_model!r} exceeds {NAME_MAX_LENGTH} characters; use a shorter agent name."
                )
            built.append(
                Combination(
                    virtual_model=virtual_model,
                    config_type=config_type,
                    capable=capable,
                    efficient=efficient,
                    judge=pair_judge,
                    config=config,
                )
            )
    return built


def _routing_digest(config_type: str, config: dict[str, Any]) -> str:
    payload = json.dumps({"config_type": config_type, "config": config}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def middleware_config(
    spec: SwitchyardOptimizeSpec, config_type: str, *, capable: str, efficient: str, judge: str | None
) -> dict[str, Any]:
    if config_type == "random_routing":
        return {
            "strong": {"model": capable},
            "weak": {"model": efficient},
            "strong_probability": spec.strong_probability,
        }
    if config_type == "stage_router":
        return {
            "picker": "efficient_first",
            "confidence_threshold": spec.confidence_threshold,
            "models": {"capable": [capable], "efficient": [efficient]},
        }
    return {
        "mode": "capability",
        "base_threshold": spec.base_threshold,
        "models": {"judge": [judge], "capable": [capable], "efficient": [efficient]},
    }


def rewrite_agent_config(source: dict[str, Any], model_ref: str) -> dict[str, Any]:
    """A deep copy of *source* whose default and harness models all point at *model_ref* through the gateway."""
    rewritten = copy.deepcopy(source)
    default = rewritten.get("models", {}).get("default")
    blocks = [default] if default else []
    blocks.extend(harness["model"] for harness in rewritten.get("harnesses", {}).values() if harness.get("model"))
    if not blocks:
        raise LocalRunError(
            "The agent config declares no models.default or harness model to route; nothing to rewrite."
        )
    for block in blocks:
        block["model"] = model_ref
        block.pop("base_url", None)
        block.get("settings", {}).pop("base_url", None)
        if block.get("provider") not in GATEWAY_PROVIDERS:
            block["provider"] = "openai"
        block["api_key_env"] = IGW_API_KEY_ENV
    return rewritten


def _qualify(ref: str, workspace: str) -> str:
    parsed = parse_entity_ref(ref, default_workspace=workspace)
    return f"{parsed.workspace}/{parsed.name}"
