# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job-submission routes for the Agent Hardener service.

Mounts the platform's standard job-collection endpoints (POST/GET/DELETE + status/cancel/logs) via
:func:`job_route_factory` for two jobs: the ``agent-hardener.war-game`` job (``/jobs``) and the
``agent-hardener.synth`` benign-suite job (``/synth-benign/jobs``). Each compiler delegates to the job's own
``compile``, so the submitted (Studio) path and the local (CLI) path share one spec.
"""

from __future__ import annotations

from nemo_agent_hardener_plugin.authz import scope
from nemo_agent_hardener_plugin.jobs.run import AgentHardenerRunJob
from nemo_agent_hardener_plugin.jobs.spec import WarGameSpec
from nemo_agent_hardener_plugin.jobs.synth_benign import AgentHardenerSynthBenignJob, SynthBenignSpec
from nemo_helix_plugin.client.adapter import AsyncHelixClient
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperator, LogicalOperation
from nemo_helix_plugin.jobs.api_factory import HelixJobSpec, job_route_factory


async def _compile_war_game(
    workspace: str,
    original_spec: WarGameSpec,
    transformed_spec: WarGameSpec,
    entity_client: EntityClient,
    job_name: str | None,
    sdk: AsyncHelixClient,
) -> HelixJobSpec:
    """Compile a war-game submission into a platform job (delegates to the job's own compile)."""
    del original_spec
    return await AgentHardenerRunJob.compile(
        workspace=workspace,
        spec=transformed_spec,
        entity_client=entity_client,
        job_name=job_name,
        async_sdk=sdk,
    )


def _collection_filter(own_field: str, sibling_field: str) -> LogicalOperation:
    # Schema defaults have been serialized since before the Agent Hardener rename. Key presence,
    # not truthiness, retains historical failed jobs and false stop_after_synth values. Requiring the
    # sibling key to be absent avoids guessing ambiguous/corrupt specs into either typed collection.
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
    job_collection_filter=_collection_filter("stop_after_synth", "interview"),
    job_input=WarGameSpec,
    platform_job_config_compiler=_compile_war_game,
    authz=scope.child("jobs"),
)


async def _compile_synth_benign(
    workspace: str,
    original_spec: SynthBenignSpec,
    transformed_spec: SynthBenignSpec,
    entity_client: EntityClient,
    job_name: str | None,
    sdk: AsyncHelixClient,
) -> HelixJobSpec:
    """Compile a benign-suite synthesis submission into a platform job (delegates to the job's own compile)."""
    del original_spec
    return await AgentHardenerSynthBenignJob.compile(
        workspace=workspace,
        spec=transformed_spec,
        entity_client=entity_client,
        job_name=job_name,
        async_sdk=sdk,
    )


# Mounted under a distinct ``/synth-benign`` prefix (see service.py) so its ``/jobs`` paths don't collide
# with the war-game router's.
synth_router = job_route_factory(
    service_name="agent-hardener",
    job_type="SynthBenign",
    job_collection_filter=_collection_filter("interview", "stop_after_synth"),
    job_input=SynthBenignSpec,
    platform_job_config_compiler=_compile_synth_benign,
    authz=scope.child("jobs"),
)
