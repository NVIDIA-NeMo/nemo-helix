# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The builder's request surface, registered under ``nemo.services``."""

from __future__ import annotations

import logging
from typing import ClassVar, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from nemo_builder_plugin._perms import BuildPerms, ContainerImagePerms
from nemo_builder_plugin.authz import scope
from nemo_builder_plugin.backend import BackendRejectedError
from nemo_builder_plugin.backends import load_backend
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.schema import BuildSet
from nemo_builder_plugin.submit import BuildConflict, InvalidBuildRequest, MissingSecrets, submit_build_set
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NemoHTTPError, NotFoundError, PermissionDeniedError
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityNotFoundError,
    get_entity_client,
)
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.log_utils import sanitize_for_log
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nemo_helix_plugin.service import NemoService, RouterSpec
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SubmitBuildResponse(BaseModel):
    """The submitted build's job and images."""

    job: str = Field(description="Name of the build job, for logs and debugging.")
    images: list[ContainerImage] = Field(
        description="One row per build spec, created `pending`. POLL THESE, not the job."
    )


class BuilderService(NemoService):
    name: ClassVar[str] = "builder"
    dependencies: ClassVar[list[str]] = ["entities", "jobs", "secrets"]

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
        client: AsyncNemoClient = Depends(get_nemo_client),
    ) -> SubmitBuildResponse:
        """Submit a set of images to build.

        Creates one `pending` ``ContainerImage`` per spec, then the job that builds them.
        """
        jobs_client = AsyncJobsClient.from_client(client)

        async def create_job(request):
            return await jobs_client.create_job(workspace=workspace, body=request)

        async def get_job_fields(name: str):
            return (await jobs_client.retrieve(name, workspace=workspace)).custom_fields or {}

        secrets_client = AsyncSecretsClient.from_client(client)

        async def secret_readable(secret_workspace: str, name: str) -> bool:
            try:
                (await secrets_client.get_secret(workspace=secret_workspace, name=name)).data()
            except (NotFoundError, PermissionDeniedError):
                return False
            return True

        try:
            # The sandbox runs where the build's profiles put its steps.
            backend = load_backend(BuilderConfig.get(), await jobs_client.list_execution_profiles())
            result = await submit_build_set(
                body,
                backend=backend,
                workspace=workspace,
                entity_client=entity_client,
                create_job=create_job,
                get_job_fields=get_job_fields,
                secret_readable=secret_readable,
            )
        except InvalidBuildRequest as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (BackendRejectedError, BuildConflict, MissingSecrets) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except PermissionDeniedError as exc:
            # From Jobs, which lists the profiles and creates the job as the caller. A refused listing writes nothing;
            # a refused job leaves the rows written before it pending.
            raise HTTPException(status_code=403, detail=f"Not permitted: {exc.detail}") from exc
        except NemoHTTPError as exc:
            logger.exception("a platform service refused build set %r", sanitize_for_log(body.name))
            raise HTTPException(
                status_code=502, detail=f"A platform service refused the build (HTTP {exc.status_code})."
            ) from exc
        except Exception as exc:
            logger.exception("failed to submit build set %r", sanitize_for_log(body.name))
            raise HTTPException(status_code=500, detail="Failed to submit build.") from exc

        return SubmitBuildResponse(job=result.job, images=result.images)

    @router.get("/container-images", response_model=list[ContainerImage], tags=["Builder"])
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[ContainerImagePerms.LIST])
    async def list_container_images(
        workspace: str,
        job: str | None = Query(
            default=None,
            max_length=128,
            pattern=r"^[^/]+$",
            description="Only the images this build job produces: the `job` a submit returned, in this workspace.",
        ),
        status: Literal["pending", "ready", "failed"] | None = Query(
            default=None, description="Only images in this state."
        ),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=100, ge=1, le=100),
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
    ) -> list[ContainerImage]:
        """List the images in a workspace, in whatever state they are in, or only one job's."""
        filters: dict[str, object] = {}
        if job is not None:
            filters["provenance.job"] = f"{workspace}/{job}"
        if status is not None:
            filters["status"] = status
        response = await entity_client.list(
            ContainerImage, workspace=workspace, filter_obj=filters or None, page=page, page_size=page_size
        )
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
