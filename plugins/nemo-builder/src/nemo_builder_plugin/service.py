# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The builder's request surface, registered under ``nemo.services``."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import ClassVar, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from nemo_builder_plugin._perms import BuildPerms, ContainerImagePerms
from nemo_builder_plugin.authz import scope
from nemo_builder_plugin.backend import BackendRejectedError
from nemo_builder_plugin.backends import load_backend
from nemo_builder_plugin.completion import (
    CompleteRequest,
    CompletionConflict,
    complete,
)
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.identity import ImageReference
from nemo_builder_plugin.schema import BuildSet, SubmitBuildResponse
from nemo_builder_plugin.submit import (
    BuildConflict,
    InvalidBuildRequest,
    MissingSecrets,
    UpstreamUnavailable,
    submit_build_set,
)
from nemo_builder_plugin.upstream import UpstreamRegistry
from nemo_helix_plugin.auth import RuntimeAuthClient, current_auth_client
from nemo_helix_plugin.auth.decisions import DecisionRequest, ImageFacts, OperationRule, ResourceTarget
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NemoHTTPError, NotFoundError, PermissionDeniedError
from nemo_helix_plugin.client_provider import get_async_nemo_client
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityNotFoundError,
    get_entity_client,
)
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.log_utils import sanitize_for_log
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nemo_helix_plugin.service import NemoService, RouterSpec

logger = logging.getLogger(__name__)


async def _get_row(entity_client: NemoEntitiesClient, workspace: str, name: str) -> ContainerImage:
    try:
        return await entity_client.get(ContainerImage, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Container image '{name}' not found.") from exc


class BuilderService(NemoService):
    name: ClassVar[str] = "builder"
    dependencies: ClassVar[list[str]] = ["entities", "jobs", "secrets"]

    def authorization_operations(self) -> dict[str, OperationRule]:
        return {
            "builder.build.submit": OperationRule(
                owner="builder",
                predicate="builder_submit",
                permissions=[BuildPerms.CREATE.id],
                callers=["principal", "service_principal"],
                scopes=["builder:write"],
            ),
            "builder.image.read": OperationRule(
                owner="builder",
                predicate="builder_read",
                permissions=[ContainerImagePerms.READ.id],
                callers=["principal", "service_principal"],
                scopes=["builder:read"],
            ),
            "builder.image.complete": OperationRule(
                owner="builder",
                predicate="builder_complete",
                permissions=[ContainerImagePerms.COMPLETE.id],
                callers=["principal", "service_principal"],
                scopes=["builder:write"],
            ),
        }

    def get_routers(self) -> list[RouterSpec]:
        return [
            RouterSpec(
                _build_router(),
                tag="Builder",
                description="Submit container image builds and read the images they produce.",
                prefix="/v2/workspaces/{workspace}",
            )
        ]


async def get_service_entity_client(
    request_client: AsyncNemoClient = Depends(get_nemo_client),
) -> AsyncIterator[NemoEntitiesClient]:
    """Use builder credentials for records containing the verified creator snapshot."""
    async with get_async_nemo_client(as_service="builder", internal=True, http_client=request_client._http) as client:
        yield NemoEntitiesClient(AsyncEntitiesClient.from_client(client))


def _build_router() -> APIRouter:
    router = APIRouter()

    @router.post("/builds", response_model=SubmitBuildResponse, status_code=201, tags=["Builder"])
    @scope.write
    # A service too, for a platform service that builds for a user, acting for that user.
    @path_rule(callers=[CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL], permissions=[BuildPerms.CREATE])
    async def submit_build(
        workspace: str,
        body: BuildSet,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
        service_entities: NemoEntitiesClient = Depends(get_service_entity_client),
        client: AsyncNemoClient = Depends(get_nemo_client),
        authorizer: RuntimeAuthClient = Depends(current_auth_client),
    ) -> SubmitBuildResponse:
        """Submit a set of images to build.

        Creates one `pending` ``ContainerImage`` per spec, then the job that builds them. With
        `max_concurrent_builds` set, the images are `queued` instead, and the job starts once a slot is free.
        """
        if not await authorizer.has_operation(
            DecisionRequest(
                operation="builder.build.submit",
                target=ResourceTarget(workspace=workspace, resource_type="container_image", name=""),
            )
        ):
            raise HTTPException(status_code=403, detail="Build submission denied")
        jobs_client = AsyncJobsClient.from_client(client)

        async def create_job(request):
            return await jobs_client.create_job(workspace=workspace, body=request)

        async def get_job_fields(name: str):
            return (await jobs_client.get_job(workspace=workspace, name=name)).data().custom_fields or {}

        secrets_client = AsyncSecretsClient.from_client(client)

        async def secret_readable(secret_workspace: str, name: str) -> bool:
            try:
                (await secrets_client.get_secret(workspace=secret_workspace, name=name)).data()
            except (NotFoundError, PermissionDeniedError):
                return False
            return True

        async def resolve_upstream(reference: ImageReference, platform: str) -> str:
            registry = UpstreamRegistry()
            try:
                return (await asyncio.to_thread(registry.resolve, reference, platform=platform)).manifest_digest
            finally:
                registry.close()

        try:
            # The sandbox runs where the build's profiles put its steps.
            config = BuilderConfig.get()
            backend = load_backend(config, await jobs_client.list_execution_profiles())
            result = await submit_build_set(
                body,
                backend=backend,
                workspace=workspace,
                entity_client=entity_client,
                create_queued_build=service_entities.create,
                create_job=create_job,
                get_job_fields=get_job_fields,
                secret_readable=secret_readable,
                resolve_upstream=resolve_upstream,
                default_platform=config.default_platform,
                queue=config.max_concurrent_builds is not None,
            )
        except InvalidBuildRequest as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (BackendRejectedError, BuildConflict, MissingSecrets) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except UpstreamUnavailable as exc:
            raise HTTPException(status_code=502, detail=f"Couldn't resolve an image to copy: {exc}") from exc
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
        build_set: str | None = Query(
            default=None, max_length=128, description="Only the images of the build set with this `name`."
        ),
        revision: int | None = Query(default=None, ge=1, description="Only the images of this `revision`."),
        status: Literal["queued", "pending", "ready", "failed"] | None = Query(
            default=None, description="Only images in this state."
        ),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=100, ge=1, le=100),
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
    ) -> list[ContainerImage]:
        """List the images in a workspace, in whatever state they are in, or only one job's or one build set's."""
        filters: dict[str, object] = {}
        if job is not None:
            filters["provenance.job"] = f"{workspace}/{job}"
        if build_set is not None:
            filters["provenance.build_set"] = build_set
        if revision is not None:
            filters["provenance.revision"] = revision
        if status is not None:
            filters["status"] = status
        response = await entity_client.list(
            ContainerImage, workspace=workspace, filter_obj=filters or None, page=page, page_size=page_size
        )
        return response.data

    @router.get("/container-images/{name}", response_model=ContainerImage, tags=["Builder"])
    @scope.read
    # A service too, for the push step, which calls as `service:builder` without workload token exchange.
    @path_rule(callers=[CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL], permissions=[ContainerImagePerms.READ])
    async def get_container_image(
        workspace: str,
        name: str,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
        authorizer: RuntimeAuthClient = Depends(current_auth_client),
    ) -> ContainerImage:
        """Read one image. This is what a caller polls until `status` leaves `pending`."""
        row = await _get_row(entity_client, workspace, name)
        if not await authorizer.has_operation(
            DecisionRequest(
                operation="builder.image.read",
                target=ResourceTarget(workspace=workspace, resource_type="container_image", name=name),
            )
        ):
            raise HTTPException(status_code=403, detail="Image read denied")
        return row

    @router.post("/container-images/{name}/complete", response_model=ContainerImage, tags=["Builder"])
    @scope.write
    @path_rule(callers=[CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL], permissions=[ContainerImagePerms.COMPLETE])
    async def complete_container_image(
        workspace: str,
        name: str,
        body: CompleteRequest,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
        authorizer: RuntimeAuthClient = Depends(current_auth_client),
    ) -> ContainerImage:
        """Make an image `ready` at the digest its push step pushed and signed.

        Completing an image already `ready` at that digest returns it; 409 if it settled otherwise.
        """
        row = await _get_row(entity_client, workspace, name)
        if not await authorizer.has_operation(
            DecisionRequest(
                operation="builder.image.complete",
                target=ResourceTarget(workspace=workspace, resource_type="container_image", name=name),
                facts=ImageFacts(created_by=row.created_by or ""),
            )
        ):
            raise HTTPException(status_code=403, detail="Image completion denied")
        try:
            ready = await complete(
                entity_client,
                workspace=workspace,
                name=name,
                digest=body.digest,
                runtime_config=body.runtime_config,
            )
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
