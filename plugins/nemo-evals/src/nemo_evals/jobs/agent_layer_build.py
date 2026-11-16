# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent-layer Job: builds one agent's layer of one task revision through NeMo Builder.

Two steps share the Job's storage. ``fetch-installer`` runs the ``nhx-agent-installer`` image and copies its
installer archive there, so the cluster's pull credentials, not an anonymous download, fetch it. ``build-layer`` runs
in ``nhx-tasks`` and does the rest (:func:`~nemo_evals.agent_layers.runner.run_agent_layer_build`).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from nemo_evals.agent_layers.models import AGENT_LAYER_BUILD_JOB_SOURCE
from nemo_evals.agent_layers.runner import AgentLayerBuildSpec, run_agent_layer_build
from nemo_evals.image_builds.builder import HttpImageBuilder
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    HelixJobSpec,
    HelixJobStep,
    ResourcesLimitsSpec,
    ResourcesRequestsSpec,
    ResourcesSpec,
)
from nemo_helix_plugin.jobs.constants import DEFAULT_JOB_STORAGE_PATH, PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_helix_plugin.jobs.image import get_qualified_image
from nemo_helix_plugin.jobs.spec import HelixJobEnvironmentVariable
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest

AGENT_LAYER_BUILD_MODULE = "nemo_evals.tasks.agent_layer_build"
INSTALLER_IMAGE = "nhx-agent-installer"
INSTALLER_ARCHIVE = "nhx-agent-installer.tar.gz"
"""The installer archive's name, at the image's root and on the Job's storage."""

AGENT_LAYER_BUILD_RESOURCES = ResourcesSpec(
    requests=ResourcesRequestsSpec(cpu="100m", memory="256Mi"),
    limits=ResourcesLimitsSpec(cpu="1", memory="1Gi"),
)
_STORAGE = [HelixJobEnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value=DEFAULT_JOB_STORAGE_PATH)]


def installer_image() -> str:
    """The installer image this deployment's layers are built with; its reference is part of every layer's key."""
    return get_qualified_image(INSTALLER_IMAGE)


class AgentLayerBuildJob(NemoJob):
    """Build one agent's layer of one task revision."""

    name = "agent-layer-build"
    description = "Build a task revision's image with an agent installed, with NeMo Builder."
    container = "nhx-tasks"
    spec_schema = AgentLayerBuildSpec

    def run(self, config: dict, *, ctx: JobContext, async_client: AsyncNemoClient) -> dict:
        """Run the layer to a ready or failed record."""
        spec = AgentLayerBuildSpec.model_validate(config)
        storage = Path(os.environ.get(PERSISTENT_JOB_STORAGE_PATH_ENVVAR, DEFAULT_JOB_STORAGE_PATH))
        layer = asyncio.run(
            run_agent_layer_build(
                spec,
                entities=EntityClient(AsyncEntitiesClient.from_client(async_client)),
                files=AsyncFilesClient.from_client(async_client),
                builder=HttpImageBuilder(async_client, workspace=spec.workspace),
                installer=storage / INSTALLER_ARCHIVE,
            )
        )
        return {"status": "completed" if layer.status == "ready" else "failed", "layer_status": layer.status}


def agent_layer_build_job_request(
    spec: AgentLayerBuildSpec, *, name: str, profile: str = "default"
) -> CreateHelixJobRequest:
    """The Jobs request that starts the agent-layer Job ``name`` for ``spec``."""
    fetch = HelixJobStep(
        name="fetch-installer",
        executor=CPUExecutionProviderSpec(
            profile=profile,
            provider="cpu",
            container=ContainerSpec(
                image=installer_image(),
                entrypoint=["/bin/cp"],
                command=[f"/{INSTALLER_ARCHIVE}", f"{DEFAULT_JOB_STORAGE_PATH}/{INSTALLER_ARCHIVE}"],
            ),
            resources=AGENT_LAYER_BUILD_RESOURCES,
        ),
        environment=_STORAGE,
    )
    build = HelixJobStep(
        name="build-layer",
        executor=CPUExecutionProviderSpec(
            profile=profile,
            provider="cpu",
            container=ContainerSpec(
                image=get_qualified_image("nhx-tasks"), entrypoint=["python", "-m"], command=[AGENT_LAYER_BUILD_MODULE]
            ),
            resources=AGENT_LAYER_BUILD_RESOURCES,
        ),
        environment=_STORAGE,
        config=spec.model_dump(mode="json"),
    )
    return CreateHelixJobRequest(
        name=name,
        description=f"Agent layer {spec.attempt} of {spec.workspace}/{spec.task}",
        source=AGENT_LAYER_BUILD_JOB_SOURCE,
        spec=spec.model_dump(mode="json"),
        platform_spec=HelixJobSpec(steps=[fetch, build]),
        custom_fields={"agent_layer": _layer_key(spec)},
    )


def is_job_for(custom_fields: dict | None, spec: AgentLayerBuildSpec) -> bool:
    """Whether a Job's custom fields say it runs ``spec``'s layer."""
    return (custom_fields or {}).get("agent_layer") == _layer_key(spec)


def _layer_key(spec: AgentLayerBuildSpec) -> str:
    return f"{spec.workspace}/{spec.task}#{spec.revision_digest}/{spec.agent_key}/{spec.attempt}"
