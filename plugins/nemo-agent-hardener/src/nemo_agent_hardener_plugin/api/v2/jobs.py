# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job-submission routes for the Agent Hardener service.

Mounts the platform's standard job-collection endpoints (POST/GET/DELETE + status/cancel/logs) via
:func:`job_route_factory` for three jobs: the ``agent-hardener.war-game`` job (``/jobs``), the
``agent-hardener.synth`` benign-suite job (``/synth-benign/jobs``) and the ``agent-hardener.victim-context``
staging job (``/victim-context/jobs``). Each compiler delegates to the job's own ``compile``, so the submitted
(Studio) path and the local (CLI) path share one spec.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from nemo_agent_hardener_plugin.authz import scope
from nemo_agent_hardener_plugin.jobs.run import AgentHardenerRunJob
from nemo_agent_hardener_plugin.jobs.spec import WarGameSpec
from nemo_agent_hardener_plugin.jobs.synth_benign import AgentHardenerSynthBenignJob, SynthBenignSpec
from nemo_agent_hardener_plugin.jobs.victim_context import AgentHardenerVictimContextJob, VictimContextSpec
from nemo_helix_plugin.api.filter import ComparisonOperation, FilterOperator, LogicalOperation
from nemo_helix_plugin.client.adapter import AsyncHelixClient
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.jobs.api_factory import HelixJobSpec, job_route_factory
from pydantic import BaseModel


class _CompilableJob(Protocol):
    """The ``compile`` signature this plugin's jobs share (``NemoJob.compile`` itself is typed loosely)."""

    async def compile(
        self, *, workspace: str, spec: BaseModel, entity_client: object, job_name: str | None, async_sdk: object
    ) -> HelixJobSpec: ...


def _compiler_for(job_type: _CompilableJob) -> Callable[..., Awaitable[HelixJobSpec]]:
    """A route compiler that delegates a submission to *job_type*'s own ``compile``."""

    async def compile_submission(
        workspace: str,
        original_spec: BaseModel,
        transformed_spec: BaseModel,
        entity_client: EntityClient,
        job_name: str | None,
        sdk: AsyncHelixClient,
    ) -> HelixJobSpec:
        del original_spec
        return await job_type.compile(
            workspace=workspace,
            spec=transformed_spec,
            entity_client=entity_client,
            job_name=job_name,
            async_sdk=sdk,
        )

    return compile_submission


# Historical factory submissions persisted model_dump(mode="json"), including defaults:
# war-game has stop_after_synth (even False); synth has interview. Driver and
# manifest_id overlap and cannot identify a kind. Reject ambiguous/unknown shapes
# rather than guess from names or schema validation (WarGame accepts synth fields).
# Use the same predicate in SQL lists (before pagination) and named route checks.
def _legacy_kind_filter(own_field: str, sibling_field: str) -> LogicalOperation:
    return LogicalOperation(
        operator=FilterOperator.AND,
        operations=[
            ComparisonOperation(operator=FilterOperator.HAS_KEY, field="spec", value=own_field),
            LogicalOperation(
                operator=FilterOperator.NOT,
                operations=[ComparisonOperation(operator=FilterOperator.HAS_KEY, field="spec", value=sibling_field)],
            ),
        ],
    )


router = job_route_factory(
    service_name="agent-hardener",
    job_type="WarGame",
    job_input=WarGameSpec,
    platform_job_config_compiler=_compiler_for(AgentHardenerRunJob),
    authz=scope.child("jobs"),
    job_discriminator="war-game",
    legacy_job_filter=_legacy_kind_filter("stop_after_synth", "interview"),
)


# Mounted under a distinct ``/synth-benign`` prefix (see service.py) so its ``/jobs`` paths don't collide
# with the war-game router's.
synth_router = job_route_factory(
    service_name="agent-hardener",
    job_type="SynthBenign",
    job_input=SynthBenignSpec,
    platform_job_config_compiler=_compiler_for(AgentHardenerSynthBenignJob),
    authz=scope.child("jobs"),
    job_discriminator="synth-benign",
    legacy_job_filter=_legacy_kind_filter("interview", "stop_after_synth"),
)

# Mounted under ``/victim-context``; new with its discriminator, so it has no unmarked historical jobs to claim.
victim_context_router = job_route_factory(
    service_name="agent-hardener",
    job_type="VictimContext",
    job_input=VictimContextSpec,
    platform_job_config_compiler=_compiler_for(AgentHardenerVictimContextJob),
    authz=scope.child("jobs"),
    job_discriminator="victim-context",
)
