# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo agents optimize run-strategy`` — route a run to the strategy that implements it.

This job owns no optimization logic.  It resolves ``--strategy`` to an installed
job, puts the rest of the spec through that job's own submit lifecycle (its
input schema, then its ``to_spec``), and delegates: :meth:`compile` returns the
strategy's steps verbatim, so the strategy's work *is* this run — one job
record, no child submission, no polling.
"""

from __future__ import annotations

import asyncio
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
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError, HelixJobDependencyUnavailableError
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
        # Resolution imports every installed plugin's job modules the first time it runs,
        # so it is kept off the event loop; later calls answer from the discovery cache.
        target = await asyncio.to_thread(_resolve_strategy, spec.strategy, error=HelixJobCompilationError)
        logger.info("Dispatching agents optimize to strategy job %s", target.__qualname__)
        compiled = await target.compile(
            workspace=workspace,
            spec=await _strategy_spec(
                target, spec, workspace=workspace, entity_client=entity_client, async_sdk=async_sdk
            ),
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
        invokes a job locally.  A strategy whose ``run`` takes ``**kwargs``
        accepts anything, so it receives every dependency: the signature
        names only the catch-all, not what it might read from it.
        """
        spec = RunStrategySpec.model_validate(config)
        target = _resolve_strategy(spec.strategy, error=LocalRunError)
        return target().run(_strategy_payload(spec), ctx=ctx, **_accepted_dependencies(target, dependencies))


def _accepted_dependencies(target: type[NemoJob], dependencies: dict[str, Any]) -> dict[str, Any]:
    """The subset of *dependencies* that ``target.run`` can be called with."""
    params = inspect.signature(target.run).parameters
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return dict(dependencies)
    return {name: value for name, value in dependencies.items() if name in params}


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


async def _strategy_spec(
    target: type[NemoJob],
    spec: RunStrategySpec,
    *,
    workspace: str,
    entity_client: object,
    async_sdk: AsyncHelixClient,
) -> Any:
    """Put the forwarded fields through *target*'s own submit lifecycle.

    The plugin service gives every job the same treatment on submit: the body is
    validated against ``input_spec_schema`` (or ``spec_schema`` when the two
    coincide), then ``to_spec`` turns that into the canonical shape ``compile``
    expects.  The route layer already did that for the router's *own* spec; this
    repeats it for the strategy, so one that resolves references or applies
    remote-only rules in ``to_spec`` sees exactly what a direct submission would
    have given it.  ``is_local`` is ``False`` for the same reason the route
    adapter passes it: ``compile`` only ever runs on the plugin service.

    Failures surface as :class:`HelixJobCompilationError` so the submit answers
    with a clean 422 -- the treatment the route layer gives a ``to_spec`` that
    fails there, re-created one layer down where the factory only maps its own
    typed errors.  Those typed errors pass through untouched, so a strategy that
    reports a dependency outage or a permission problem keeps its status code.
    """
    payload = _strategy_payload(spec)
    schema = target.input_spec_schema or target.spec_schema
    if schema is None:
        return payload
    try:
        submitted = schema.model_validate(payload)
    except ValidationError as exc:
        raise HelixJobCompilationError(f"Spec is not valid for optimization strategy {spec.strategy!r}: {exc}") from exc
    try:
        return await target.to_spec(
            submitted,
            workspace=workspace,
            entity_client=entity_client,
            async_sdk=async_sdk,
            is_local=False,
        )
    except (PermissionError, HelixJobDependencyUnavailableError, HelixJobCompilationError):
        raise
    except Exception as exc:  # noqa: BLE001 -- mirrors the route layer: a failing to_spec is a bad submit, not a crash
        raise HelixJobCompilationError(
            f"Optimization strategy {spec.strategy!r} could not prepare its spec: {exc}"
        ) from exc
