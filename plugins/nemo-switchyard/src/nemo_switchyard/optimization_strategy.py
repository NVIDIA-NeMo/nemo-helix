# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``switchyard`` implementation of ``nemo agents optimize --strategy``.

Creates a VirtualModel that routes between a strong and a weak model via the
``nemo-switchyard`` request middleware, then rewrites the agent's model
parameter to point at it. Routing is a two-tier weighted coin flip because that
is what the Inference Gateway actually wires up today — see
``_CONFIG_TYPE_TO_SY_NAME`` in :mod:`nemo_switchyard._factory`.
"""

from __future__ import annotations

import copy
import json
from typing import Any, ClassVar, Literal

import yaml
from nemo_agent_optimization_plugin.strategies import PRIMARY_ARTIFACT_KEY
from nemo_platform import NeMoPlatform
from nemo_platform.types.inference.middleware_call_param import MiddlewareCallParam
from nemo_platform.types.inference.virtual_model_inference_config_param import VirtualModelInferenceConfigParam
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.run_dependencies import LocalRunError
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
#: loading this strategy (which ``nemo agents optimize`` does for every run) does not import
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


class SwitchyardOptimizationStrategy:
    """Point a Fabric agent's model at a Switchyard-routed VirtualModel."""

    name: ClassVar[str] = "switchyard"

    def validate_config(self, config: dict[str, Any], *, agent: str | None) -> None:
        if agent is None:
            raise ValueError("The switchyard strategy requires --agent.")
        SwitchyardConfig.model_validate(config)

    def run(
        self,
        *,
        agent_config: dict[str, Any] | None,
        source_agent_config: dict[str, Any] | None = None,
        config: dict[str, Any],
        ctx: JobContext,
        workspace: str,
        sdk: NeMoPlatform | None = None,
    ) -> dict[str, Any]:
        parsed = SwitchyardConfig.model_validate(config)
        if agent_config is None:
            raise LocalRunError("The switchyard strategy requires --agent: it rewrites an agent's model parameter.")
        if sdk is None:
            raise LocalRunError(
                "The switchyard strategy creates a VirtualModel on the platform, which requires a platform "
                "SDK ('sdk: NeMoPlatform'). Set NMP_BASE_URL or pass sdk via NemoJobScheduler.run_local(sdk=...)."
            )

        virtual_model = sdk.inference.virtual_models.create(
            workspace=workspace,
            name=parsed.virtual_model,
            models=_virtual_model_entries(parsed),
            request_middleware=[_routing_middleware(parsed)],
            exist_ok=True,
        )

        # `VirtualModel.name` is Optional[str] in the generated SDK, so falling back to the name we
        # asked for keeps this from silently producing a "<workspace>/None" model reference.
        # `exist_ok=True` returns the existing VirtualModel under that same name, so the requested
        # name is canonical either way.
        routed_model = f"{workspace}/{virtual_model.name or parsed.virtual_model}"
        optimized_config = copy.deepcopy(source_agent_config or agent_config)
        if not _rewrite_model(optimized_config, routed_model):
            raise LocalRunError(
                f"The switchyard strategy found no model parameter to rewrite in the agent config: "
                f"neither 'models.default.model' nor any 'harnesses.<name>.model' block is present. "
                f"VirtualModel {routed_model!r} was created before this check, so fixing the agent "
                f"config and re-running is safe — the VirtualModel is reused, not duplicated."
            )

        output_dir = ctx.storage.persistent / "results" / RESULT_NAME
        output_dir.mkdir(parents=True, exist_ok=True)
        optimized_path = output_dir / "optimized_config.yml"
        optimized_path.write_text(yaml.safe_dump(optimized_config, sort_keys=False), encoding="utf-8")

        summary = {
            "status": "completed",
            "strategy": self.name,
            "virtual_model": routed_model,
            "optimized_config": optimized_path.name,
        }
        (output_dir / "switchyard_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        ref = ctx.results.save(RESULT_NAME, output_dir)
        return {**summary, "result": ref.model_dump(mode="json"), PRIMARY_ARTIFACT_KEY: str(optimized_path)}


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


def _rewrite_model(agent_config: dict[str, Any], routed_model: str) -> bool:
    """Point every model parameter the agent actually reads at *routed_model*.

    Returns whether anything was rewritten, so a config shape this strategy does not
    understand fails loudly instead of yielding an unchanged "optimized" config.
    """
    rewritten = False
    models = agent_config.get("models")
    if isinstance(models, dict) and isinstance(models.get("default"), dict):
        models["default"]["model"] = routed_model
        rewritten = True
    harnesses = agent_config.get("harnesses")
    if isinstance(harnesses, dict):
        for harness in harnesses.values():
            if isinstance(harness, dict) and isinstance(harness.get("model"), dict):
                harness["model"]["model"] = routed_model
                rewritten = True
    return rewritten
