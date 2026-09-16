# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo agents optimize`` — route to the selected strategy's job.

This job owns no optimization logic and never touches a bundle or an agent.
It resolves ``--strategy`` to an installed :class:`AgentOptimizeJob` and hands
off: ``compile`` delegates to the strategy job and returns its steps, so the
strategy's work *is* the run — one job record, no child submission, no polling.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from nemo_agent_optimization_plugin.discovery import discover_agent_optimize_jobs
from nemo_agent_optimization_plugin.job_base import AgentOptimizeJob
from nemo_agent_optimization_plugin.schemas.optimize import (
    AgentOptimizeSpec,
    OptimizeSpec,
    OptimizeSubmitSpec,
)
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.jobs.api_factory import PlatformJobSpec
from nemo_platform_plugin.run_dependencies import LocalRunError
from nemo_platform_plugin.scheduler import NemoJobScheduler
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class OptimizeJob(NemoJob):
    """Dispatch an optimization run to the strategy job that implements it."""

    name: ClassVar[str] = "optimize"
    description: ClassVar[str] = "Optimize a platform agent with an installed optimization strategy."
    container: ClassVar[str] = "cpu-tasks"
    job_collection_path: ClassVar[str | None] = None
    generate_legacy_verbs: ClassVar[bool] = False
    spec_schema: ClassVar[type[BaseModel]] = OptimizeSpec
    input_spec_schema: ClassVar[type[BaseModel]] = OptimizeSubmitSpec

    @classmethod
    async def to_spec(  # ty: ignore[invalid-method-override]
        cls,
        input_spec: OptimizeSubmitSpec,
        *,
        workspace: str,
        entity_client: object,
        async_sdk: AsyncNeMoPlatform,
        is_local: bool,
    ) -> OptimizeSpec:
        del entity_client, async_sdk, is_local
        payload = input_spec.model_dump(mode="json")
        payload["workspace"] = workspace
        return OptimizeSpec.model_validate(payload)

    @classmethod
    async def compile(  # ty: ignore[invalid-method-override]
        cls,
        *,
        workspace: str,
        spec: OptimizeSpec,
        entity_client: object,
        job_name: str | None,
        async_sdk: object,
        profile: str | None = None,
        options: dict | None = None,
    ) -> PlatformJobSpec:
        target = _resolve_strategy(spec.strategy)
        logger.info("Dispatching agents optimize to strategy job %s", target.__qualname__)
        compiled = await target.compile(
            workspace=workspace,
            spec=_child_spec(spec),
            entity_client=entity_client,
            job_name=job_name,
            async_sdk=async_sdk,
            profile=profile,
            options=options,
        )
        return PlatformJobSpec(steps=compiled.steps)

    def run(self, config: dict, *, ctx: JobContext, sdk: NeMoPlatform | None = None) -> dict:
        del ctx
        spec = OptimizeSpec.model_validate(config)
        target = _resolve_strategy(spec.strategy)
        return NemoJobScheduler().run_local(
            target,
            _child_spec(spec).model_dump(mode="json"),
            workspace=spec.workspace,
            sdk=sdk,
        )


def _resolve_strategy(strategy: str) -> type[AgentOptimizeJob]:
    """The installed job implementing *strategy*, or a listing of what is installed."""
    installed = discover_agent_optimize_jobs()
    target = installed.get(strategy)
    if target is None:
        raise LocalRunError(
            f"Optimization strategy {strategy!r} is not installed. Available strategies: {sorted(installed)}"
        )
    return target


def _child_spec(spec: OptimizeSpec) -> AgentOptimizeSpec:
    """The router's spec minus ``strategy``, which the strategy job already knows."""
    payload = spec.model_dump(mode="json")
    payload.pop("strategy", None)
    return AgentOptimizeSpec.model_validate(payload)
