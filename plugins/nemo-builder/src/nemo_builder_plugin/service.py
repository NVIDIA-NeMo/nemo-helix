# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The builder's request surface, registered under ``nemo.services``."""

from __future__ import annotations

import logging
from typing import ClassVar, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from nemo_builder_plugin._perms import BuildPerms, ContainerImagePerms
from nemo_builder_plugin.authz import scope
from nemo_builder_plugin.backend import BackendRefused
from nemo_builder_plugin.backends import load_backend
from nemo_builder_plugin.completion import (
    Caller,
    CompleteRequest,
    CompletionConflict,
    caller_refusal,
    complete,
    current_caller,
)
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.schema import BuildSet
from nemo_builder_plugin.submit import BuildConflict, InvalidBuildRequest, MissingSecrets, submit_build_set
from nemo_helix import AsyncNeMoHelix
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import NemoHTTPError, NotFoundError, PermissionDeniedError
from nemo_helix_plugin.dependencies import get_sdk_client
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


async def _get_row(entity_client: NemoEntitiesClient, workspace: str, name: str) -> ContainerImage:
    try:
        return await entity_client.get(ContainerImage, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Container image '{name}' not found.") from exc


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
        sdk: AsyncNeMoHelix = Depends(get_sdk_client),
    ) -> SubmitBuildResponse:
        """Submit a set of images to build.

        Creates one `pending` ``ContainerImage`` per spec, then the job that builds them.
        """
        backend = load_backend(BuilderConfig.get())
        jobs_client = client_from_platform(sdk, AsyncJobsClient)

        async def create_job(request):
            return await jobs_client.create_job(workspace=workspace, body=request)

        async def get_job_fields(name: str):
            return (await jobs_client.retrieve(name, workspace=workspace)).custom_fields or {}

        secrets_client = client_from_platform(sdk, AsyncSecretsClient)

        async def secret_readable(secret_workspace: str, name: str) -> bool:
            try:
                (await secrets_client.get_secret(workspace=secret_workspace, name=name)).data()
            except (NotFoundError, PermissionDeniedError):
                return False
            return True

        try:
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
        except (BackendRefused, BuildConflict, MissingSecrets) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except PermissionDeniedError as exc:
            # The rows and the job are created as the caller, so this refusal is the caller's.
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

    @router.post("/container-images/{name}/complete", response_model=ContainerImage, tags=["Builder"])
    @scope.write
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[ContainerImagePerms.COMPLETE])
    async def complete_container_image(
        workspace: str,
        name: str,
        body: CompleteRequest,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
        caller: Caller = Depends(current_caller),
    ) -> ContainerImage:
        """Make an image `ready` at the digest its push step pushed and signed.

        Completing an image already `ready` at that digest returns it; 409 if it settled otherwise.
        """
        row = await _get_row(entity_client, workspace, name)
        refusal = caller_refusal(row, caller)
        if refusal is not None:
            logger.warning(
                "completing %s/%s refused: %s",
                sanitize_for_log(workspace),
                sanitize_for_log(name),
                sanitize_for_log(refusal),
            )
            raise HTTPException(status_code=403, detail=refusal)
        try:
            ready = await complete(entity_client, workspace=workspace, name=name, digest=body.digest)
        except NemoEntityNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Container image '{name}' not found.") from exc
        except CompletionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        logger.info(
            "image %s/%s ready: %s",
            sanitize_for_log(workspace),
            sanitize_for_log(name),
            sanitize_for_log(ready.image_ref),
        )
        return ready

    return router
