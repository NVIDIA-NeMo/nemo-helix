# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo agents optimize run-strategy`` — route a run to the strategy that implements it.

This job owns no optimization logic.  It resolves ``--strategy`` to an installed
job, hands the rest of the spec to that job's own schema, and delegates:
:meth:`compile` returns the strategy's steps verbatim, so the strategy's work
*is* this run — one job record, no child submission, no polling.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, ClassVar

from nemo_agent_optimization_plugin.discovery import discover_strategy_jobs
from nemo_agent_optimization_plugin.schemas.optimize import RunStrategySpec, RunStrategySubmitSpec
from nemo_helix_plugin.client.adapter import AsyncHelixClient
from nemo_helix_plugin.errors import LocalRunError
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.api_factory import HelixJobSpec
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class RunStrategyJob(NemoJob):
    """Dispatch an optimization run to the strategy job that implements it."""

    name: ClassVar[str] = "run-strategy"
    description: ClassVar[str] = "Optimize a platform agent with an installed optimization strategy."
    container: ClassVar[str] = "cpu-tasks"
    job_collection_path: ClassVar[str | None] = None
    generate_legacy_verbs: ClassVar[bool] = False
    spec_schema: ClassVar[type[BaseModel]] = RunStrategySpec
    input_spec_schema: ClassVar[type[BaseModel]] = RunStrategySubmitSpec

    @classmethod
    async def to_spec(  # ty: ignore[invalid-method-override]
        cls,
        input_spec: RunStrategySubmitSpec,
        *,
        workspace: str,
        entity_client: object,
        async_sdk: AsyncHelixClient,
        is_local: bool,
    ) -> RunStrategySpec:
        del entity_client, async_sdk, is_local
        payload = input_spec.model_dump(mode="json")
        payload["workspace"] = workspace
        return RunStrategySpec.model_validate(payload)

    @classmethod
    async def compile(  # ty: ignore[invalid-method-override]
        cls,
        *,
        workspace: str,
        spec: RunStrategySpec,
        entity_client: object,
        job_name: str | None,
        async_sdk: AsyncHelixClient,
        profile: str | None = None,
        options: dict | None = None,
    ) -> HelixJobSpec:
        target = _resolve_strategy(spec.strategy, error=HelixJobCompilationError)
        logger.info("Dispatching agents optimize to strategy job %s", target.__qualname__)
        compiled = await target.compile(
            workspace=workspace,
            spec=_strategy_spec(target, spec, error=HelixJobCompilationError),
            entity_client=entity_client,
            job_name=job_name,
            async_sdk=async_sdk,
            profile=profile,
            options=options,
        )
        # Returned whole rather than re-wrapped around ``steps``: a strategy may also declare
        # secrets its steps reference, and dropping those would fail the run at execution time.
        return compiled if isinstance(compiled, HelixJobSpec) else HelixJobSpec.model_validate(compiled)

    def run(self, config: dict, *, ctx: JobContext, **dependencies: Any) -> dict:
        """Run the selected strategy in-process.

        Only the task entrypoint reaches this: a platform submission runs the
        strategy's own steps, because :meth:`compile` returned them. *ctx* is
        forwarded so the strategy writes its artifacts into the context the
        caller set up rather than a fresh one.

        The router does not know which framework-managed dependencies a
        strategy declares, so it forwards only the ones that strategy's
        ``run`` actually accepts -- the same rule the CLI applies when it
        invokes a job locally.
        """
        spec = RunStrategySpec.model_validate(config)
        target = _resolve_strategy(spec.strategy, error=LocalRunError)
        accepted = inspect.signature(target.run).parameters
        forwarded = {name: value for name, value in dependencies.items() if name in accepted}
        return target().run(_strategy_payload(spec), ctx=ctx, **forwarded)


def _resolve_strategy(strategy: str, *, error: type[Exception]) -> type[NemoJob]:
    """The installed job implementing *strategy*, or *error* listing what is installed."""
    installed = discover_strategy_jobs()
    target = installed.get(strategy)
    if target is None:
        raise error(
            f"Optimization strategy {strategy!r} is not installed. "
            f"Available strategies: {sorted(installed) or ['<none>']}"
        )
    return target


def _strategy_payload(spec: RunStrategySpec) -> dict[str, Any]:
    """The router's fields minus ``strategy``, which the strategy job already knows."""
    payload = spec.model_dump(mode="json")
    payload.pop("strategy", None)
    return payload


def _strategy_spec(target: type[NemoJob], spec: RunStrategySpec, *, error: type[Exception]) -> Any:
    """Re-validate the forwarded fields against *target*'s own spec schema.

    The strategy owns what its inputs mean, so this is where a bad combination
    (a config path that has to be fileset-relative, an output target the
    strategy cannot write) is caught.  Surfacing it as *error* keeps it a clean
    422 on the submit rather than a 500 from an unhandled ``ValidationError``.
    """
    payload = _strategy_payload(spec)
    schema = target.spec_schema
    if schema is None:
        return payload
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        raise error(f"Spec is not valid for optimization strategy {spec.strategy!r}: {exc}") from exc
