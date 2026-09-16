# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``switchyard`` agent optimization: route the agent through a new VirtualModel."""

from __future__ import annotations

import copy
from typing import Any, ClassVar

from nemo_agent_optimization_plugin.job_base import AgentOptimizeJob
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.run_dependencies import LocalRunError
from nemo_switchyard.routing import (
    PLATFORM_IGW_PATH_MARKER,
    SwitchyardConfig,
    _rewrite_model,
    _routing_middleware,
    _virtual_model_entries,
)


class SwitchyardOptimizeJob(AgentOptimizeJob):
    """Point the agent's model at a probabilistically routed VirtualModel."""

    name: ClassVar[str] = "agent_optimize"
    strategy: ClassVar[str] = "switchyard"
    description: ClassVar[str] = "Route an agent's model through a Switchyard VirtualModel."
    task_module: ClassVar[str] = "nemo_switchyard.tasks.agent_optimize"

    def optimize(
        self,
        *,
        source_agent_config: dict[str, Any],
        config: dict[str, Any],
        ctx: JobContext,
        workspace: str,
        sdk: NeMoPlatform,
    ) -> dict[str, Any]:
        del ctx
        parsed = SwitchyardConfig.model_validate(config)

        # Resolved before the VirtualModel is created: without a gateway endpoint the
        # rewritten agent could not reach the router, and this job registers its output
        # as a live agent entity rather than handing a human a YAML diff to review.
        gateway_base_url = _gateway_base_url(sdk, workspace)

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

        optimized = copy.deepcopy(source_agent_config)
        if not _rewrite_model(optimized, routed_model, gateway_base_url=gateway_base_url):
            raise LocalRunError(
                "The switchyard strategy found no model parameter to rewrite in the agent "
                "config: neither 'models.default.model' nor any 'harnesses.<name>.model' block "
                "is present."
            )
        return optimized


def _gateway_base_url(sdk: NeMoPlatform, workspace: str) -> str:
    """The workspace's Inference Gateway OpenAI route, where the VirtualModel lives."""
    try:
        base_url = str(sdk.models.get_openai_route_base_url(workspace=workspace))
    except (AttributeError, ValueError) as exc:
        raise LocalRunError(
            "The switchyard strategy could not resolve the Inference Gateway URL for workspace "
            f"{workspace!r} from the platform SDK. The routed VirtualModel only exists behind the "
            "gateway, so rewriting the agent's model name without its endpoint would register an "
            "agent that cannot run."
        ) from exc
    if PLATFORM_IGW_PATH_MARKER not in base_url:
        raise LocalRunError(
            f"The platform SDK resolved {base_url!r} as the OpenAI route for workspace "
            f"{workspace!r}, which is not an Inference Gateway URL (no "
            f"{PLATFORM_IGW_PATH_MARKER!r} segment). Refusing rather than registering an agent "
            "pointed at an endpoint that does not serve the routed VirtualModel."
        )
    return base_url
