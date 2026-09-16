# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared routing config and helpers for the ``switchyard`` agent optimization job.

Creates a VirtualModel that routes between a strong and a weak model via the
``nemo-switchyard`` request middleware. Routing is a two-tier weighted coin flip because that
is what the Inference Gateway actually wires up today — see
``_CONFIG_TYPE_TO_SY_NAME`` in :mod:`nemo_switchyard._factory`.
"""

from __future__ import annotations

from typing import Any, Literal

from nemo_platform.types.inference.middleware_call_param import MiddlewareCallParam
from nemo_platform.types.inference.virtual_model_inference_config_param import VirtualModelInferenceConfigParam
from pydantic import BaseModel, Field

RESULT_NAME = "switchyard_results"
MIDDLEWARE_NAME = "nemo-switchyard"
CONFIG_TYPE = "random_routing"

#: Platform/IGW wire-format names, as accepted by ``VirtualModel.models[].backend_format``
#: (``nemo_platform_plugin.inference_middleware.BackendFormat``).
PlatformBackendFormat = Literal["OPENAI_CHAT", "ANTHROPIC_MESSAGES"]
DEFAULT_BACKEND_FORMAT: PlatformBackendFormat = "OPENAI_CHAT"

#: Platform format names → the lowercase names switchyard's own ``BackendTier`` accepts.
#: The two vocabularies are genuinely different: IGW validates the VirtualModel's ``models``
#: entries against the uppercase enum, while ``RandomRoutingFactory.validate`` parses the
#: middleware ``config`` with switchyard's lowercase ``BackendFormat``. Mirrors
#: ``_NEMO_TO_SWITCHYARD_FORMAT`` in :mod:`nemo_switchyard._format`, duplicated here so that
#: loading this job (which ``nemo agents optimize`` does for every run) does not import
#: the openai/anthropic SDKs that module pulls in transitively.
_PLATFORM_TO_SWITCHYARD_FORMAT: dict[str, str] = {
    "OPENAI_CHAT": "openai",
    "ANTHROPIC_MESSAGES": "anthropic",
}


class BackendTier(BaseModel):
    """One routing tier. Mirrors switchyard's ``BackendTier``."""

    model: str = Field(min_length=1, description="Model entity ref ('name' or 'workspace/name').")
    backend_format: PlatformBackendFormat = Field(
        default=DEFAULT_BACKEND_FORMAT,
        description="Wire format of the tier's backend.",
    )

    def as_virtual_model_entry(self) -> VirtualModelInferenceConfigParam:
        """This tier as a ``VirtualModel.models`` entry (platform vocabulary)."""
        return {"model": self.model, "backend_format": self.backend_format}

    def as_switchyard_tier(self) -> dict[str, str]:
        """This tier as a switchyard ``BackendTier`` mapping (switchyard vocabulary)."""
        return {"model": self.model, "backend_format": _PLATFORM_TO_SWITCHYARD_FORMAT[self.backend_format]}


class SwitchyardConfig(BaseModel):
    """Strategy config for ``--strategy switchyard``."""

    virtual_model: str = Field(min_length=1, description="Name of the VirtualModel to create.")
    strong: BackendTier
    weak: BackendTier
    strong_probability: float = Field(ge=0.0, le=1.0, description="Probability of routing to the strong tier.")
    rng_seed: int | None = Field(default=None, description="Seed for deterministic routing; None uses a fresh RNG.")
    enable_stats: bool = Field(default=False, description="Record per-tier routing stats in the gateway.")


def _routing_middleware(parsed: SwitchyardConfig) -> MiddlewareCallParam:
    """The ``nemo-switchyard`` request-middleware entry that does the routing.

    ``SwitchyardMiddleware`` parses ``config`` with ``RandomRoutingFactory.validate`` when the
    VirtualModel is upserted, so the tiers go out in switchyard's own vocabulary — a mismatch
    here is a 400 at VirtualModel create time, not a runtime surprise.
    """
    return {
        "name": MIDDLEWARE_NAME,
        "config_type": CONFIG_TYPE,
        "config": {
            "strong": parsed.strong.as_switchyard_tier(),
            "weak": parsed.weak.as_switchyard_tier(),
            "strong_probability": parsed.strong_probability,
            "rng_seed": parsed.rng_seed,
            "enable_stats": parsed.enable_stats,
        },
    }


def _virtual_model_entries(parsed: SwitchyardConfig) -> list[VirtualModelInferenceConfigParam]:
    """The tiers as ``VirtualModel.models`` entries, deduplicated by model ref.

    IGW keys backend formats by model name, so listing one model twice buys nothing — and a
    strong/weak pair pointing at the same entity is a legitimate config (an A/B of the same
    model under different routing weights).
    """
    entries: list[VirtualModelInferenceConfigParam] = []
    seen: set[str] = set()
    for tier in (parsed.strong, parsed.weak):
        if tier.model in seen:
            continue
        seen.add(tier.model)
        entries.append(tier.as_virtual_model_entry())
    return entries


#: Mirrors ``nemo_agents_plugin.fabric.gateway_credentials``, duplicated here for the same
#: reason as ``_PLATFORM_TO_SWITCHYARD_FORMAT`` above: nemo-switchyard does not depend on
#: nemo-agents-plugin, and this module is imported for every ``nemo agents optimize`` run.
PLATFORM_IGW_PATH_MARKER = "/apis/inference-gateway/"
PLATFORM_IGW_API_KEY_ENV = "NEMO_AGENTS_IGW_API_KEY"


def _is_gateway_routed(model: dict[str, Any]) -> bool:
    """Whether *model* already talks to a platform Inference Gateway route."""
    base_url = model.get("base_url")
    if isinstance(base_url, str) and PLATFORM_IGW_PATH_MARKER in base_url:
        return True
    settings = model.get("settings")
    legacy = settings.get("base_url") if isinstance(settings, dict) else None
    return isinstance(legacy, str) and PLATFORM_IGW_PATH_MARKER in legacy


def _rewrite_model(agent_config: dict[str, Any], routed_model: str, *, gateway_base_url: str) -> bool:
    """Point every model parameter the agent actually reads at *routed_model*.

    The routed model is a VirtualModel entity ref (``workspace/name``), which only exists
    behind the workspace's Inference Gateway — so the endpoint has to move with the name.
    Rewriting the name alone leaves the agent asking its original provider (e.g.
    ``inference-api.nvidia.com``) for a model that provider has never heard of.  The job
    persists this config as a live agent entity, so an unrunnable result would not be
    caught by a human reading a YAML diff.

    ``api_key_env`` moves too: the gateway takes the platform's placeholder credential,
    not the provider key the original ``api_key_env`` named.  A model that was already
    gateway-routed keeps whatever ``api_key_env`` it declared.

    Returns whether anything was rewritten, so a config shape this strategy does not
    understand fails loudly instead of yielding an unchanged "optimized" config.
    """
    targets: list[dict[str, Any]] = []
    models = agent_config.get("models")
    if isinstance(models, dict) and isinstance(models.get("default"), dict):
        targets.append(models["default"])
    harnesses = agent_config.get("harnesses")
    if isinstance(harnesses, dict):
        for harness in harnesses.values():
            if isinstance(harness, dict) and isinstance(harness.get("model"), dict):
                targets.append(harness["model"])

    for model in targets:
        already_routed = _is_gateway_routed(model)
        model["model"] = routed_model
        model["base_url"] = gateway_base_url
        if not already_routed or not model.get("api_key_env"):
            model["api_key_env"] = PLATFORM_IGW_API_KEY_ENV
        # A legacy settings.base_url is only read when base_url is unset; leaving the old
        # provider URL behind it would make the persisted agent read two ways.
        settings = model.get("settings")
        if isinstance(settings, dict):
            settings.pop("base_url", None)
    return bool(targets)
