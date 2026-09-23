# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The builder's request surface, registered under ``nemo.services``.

``POST /builds`` is hand-written rather than generated from ``job_collection_path``. The
generated route would have been cheaper, and it was the wrong trade: it returns a job, and the
thing a caller needs to poll is the **images**. A set of ten produces ten rows that reach
``ready`` independently, and a single job status cannot say which. Hand-writing also keeps the
compiler a pure function called from here, rather than I/O smuggled into ``compile()``.

Every route carries ``@scope.*`` and ``@path_rule``; without them the OPA bundle build fails.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from fastapi import APIRouter, Depends, HTTPException
from nemo_builder_plugin._perms import BuildPerms, ContainerImagePerms
from nemo_builder_plugin.authz import scope
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.plan import BuildCompileError
from nemo_builder_plugin.schema import BuildSet
from nemo_builder_plugin.submit import submit_build_set
from nemo_helix import AsyncNeMoHelix
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.dependencies import get_sdk_client
from nemo_helix_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityNotFoundError,
    get_entity_client,
)
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.service import NemoService, RouterSpec
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SubmitBuildResponse(BaseModel):
    """``job`` for operators, ``images`` for callers.

    Both, because they answer different questions: the job is where a human goes to read logs,
    and the images are what a consumer polls and eventually pins.
    """

    job: str = Field(description="Name of the build job, for logs and debugging.")
    images: list[ContainerImage] = Field(
        description="One row per build spec, created `pending`. POLL THESE, not the job."
    )


class BuilderService(NemoService):
    """Registered under ``nemo.services``; mounted at ``/apis/builder``."""

    name: ClassVar[str] = "builder"
    dependencies: ClassVar[list[str]] = ["entities", "jobs"]

    def get_routers(self) -> list[RouterSpec]:
        return [
            RouterSpec(
                _build_router(),
                tag="Builder",
                description="Submit container image builds and read the images they produce.",
                prefix="/v2/workspaces/{workspace}",
            )
        ]


def _build_router() -> APIRouter:
    router = APIRouter()

    @router.post("/builds", response_model=SubmitBuildResponse, status_code=201, tags=["Builder"])
    @scope.write
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[BuildPerms.CREATE])
    async def submit_build(
        workspace: str,
        body: BuildSet,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
        sdk: AsyncNeMoHelix = Depends(get_sdk_client),
    ) -> SubmitBuildResponse:
        """Submit a set of images to build.

        Creates one `pending` ``ContainerImage`` per spec, then the job that builds them. A
        deployment that is unconfigured -- or that has the execution backend disabled -- fails
        here with a 4xx, rather than producing a job that dies in a pod later.
        """
        config = BuilderConfig.get()
        jobs_client = client_from_platform(sdk, AsyncJobsClient)

        async def create_job(request):
            return await jobs_client.create_job(workspace=workspace, body=request)

        try:
            result = await submit_build_set(
                body,
                config=config,
                workspace=workspace,
                entity_client=entity_client,
                create_job=create_job,
            )
        except BuildCompileError as exc:
            # The kill switch and every unconfigured-deployment case land here. 409 rather than
            # 400: the request is well-formed, the deployment cannot currently satisfy it.
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("failed to submit build set %r", body.name)
            raise HTTPException(status_code=500, detail="Failed to submit build.") from exc

        return SubmitBuildResponse(job=result.job, images=result.images)

    @router.get("/container-images", response_model=list[ContainerImage], tags=["Builder"])
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[ContainerImagePerms.LIST])
    async def list_container_images(
        workspace: str,
        page: int = 1,
        page_size: int = 100,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
    ) -> list[ContainerImage]:
        """List the images in a workspace, in whatever state they are in.

        ``.data`` rather than ``list(...)``: the client returns a ``ListResponse``, and iterating
        a pydantic model yields ``(field, value)`` pairs, not its contents.
        """
        response = await entity_client.list(ContainerImage, workspace=workspace, page=page, page_size=page_size)
        return response.data

    @router.get("/container-images/{name}", response_model=ContainerImage, tags=["Builder"])
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[ContainerImagePerms.READ])
    async def get_container_image(
        workspace: str,
        name: str,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
    ) -> ContainerImage:
        """Read one image. This is what a caller polls until `status` leaves `pending`."""
        try:
            return await entity_client.get(ContainerImage, name=name, workspace=workspace)
        except NemoEntityNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Container image '{name}' not found.") from exc

    return router
