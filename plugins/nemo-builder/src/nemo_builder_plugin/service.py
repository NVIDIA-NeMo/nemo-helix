# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The builder's request surface, registered under ``nemo.services``.

``POST /builds`` is hand-written rather than generated from ``job_collection_path``. The
generated route would have been cheaper, and it was the wrong trade: it returns a job, and the
thing a caller needs to poll is the **images**. A set of ten produces ten rows that reach
``ready`` independently, and a single job status cannot say which. Hand-writing also keeps the
compiler a pure function called from here, rather than I/O smuggled into ``compile()``.

``POST /container-images/{name}/signature`` is how a row goes ``ready``: the build's last step
delivers the signed payload and signature the credential broker returned, and the route verifies
them against the backend's trust root before writing anything (``completion.py``). The caller is
authorized for the workspace, but what the route trusts is the signature.

Every route carries ``@scope.*`` and ``@path_rule``; without them the OPA bundle build fails.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from nemo_builder_plugin._perms import BuildPerms, ContainerImagePerms
from nemo_builder_plugin.authz import scope
from nemo_builder_plugin.backend import BackendRefused, SignatureRefused
from nemo_builder_plugin.backends import load_backend
from nemo_builder_plugin.completion import CompletionConflict, SignatureDelivery, complete
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.schema import BuildSet
from nemo_builder_plugin.submit import BuildConflict, InvalidBuildRequest, submit_build_set
from nemo_helix import AsyncNeMoHelix
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import NemoHTTPError, PermissionDeniedError
from nemo_helix_plugin.dependencies import get_sdk_client
from nemo_helix_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityNotFoundError,
    get_entity_client,
)
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.log_utils import sanitize_for_log
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
        deployment that is unconfigured -- or whose backend is disabled, or cannot honor the
        request -- fails here with a 4xx, rather than producing a job that dies in a pod later.
        """
        backend = load_backend(BuilderConfig.get())
        jobs_client = client_from_platform(sdk, AsyncJobsClient)

        async def create_job(request):
            return await jobs_client.create_job(workspace=workspace, body=request)

        async def get_job_fields(name: str):
            return (await jobs_client.retrieve(name, workspace=workspace)).custom_fields or {}

        try:
            result = await submit_build_set(
                body,
                backend=backend,
                workspace=workspace,
                entity_client=entity_client,
                create_job=create_job,
                get_job_fields=get_job_fields,
            )
        except InvalidBuildRequest as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (BackendRefused, BuildConflict) as exc:
            # An unconfigured deployment, what the backend cannot build, and a reused revision.
            # 409 rather than 400: the request is well-formed; the deployment or the revision's
            # history is what stands in the way.
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except PermissionDeniedError as exc:
            # The job and rows are created as the caller, so a caller allowed to submit builds but
            # not to create jobs in this workspace lands here. Theirs to fix, not ours.
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
        """List the images in a workspace, in whatever state they are in, or only one job's.

        ``job`` and ``status`` are how the credential broker finds what a job may publish -- its
        ``pending`` rows -- without reading the whole workspace.

        ``.data`` rather than ``list(...)``: the client returns a ``ListResponse``, and iterating
        a pydantic model yields ``(field, value)`` pairs, not its contents.
        """
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

    @router.post("/container-images/{name}/signature", response_model=ContainerImage, tags=["Builder"])
    @scope.write
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[ContainerImagePerms.COMPLETE])
    async def deliver_signature(
        workspace: str,
        name: str,
        body: SignatureDelivery,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
    ) -> ContainerImage:
        """Deliver an image's signature; the row goes `ready` if it verifies.

        A signature for a row already `ready` with the same digest is accepted again, so a step that
        reruns still completes. 422 for a signature that does not verify or is not this row's; 409
        for a row that has settled otherwise.
        """
        config = BuilderConfig.get()
        try:
            policy = load_backend(config).signature_policy()
        except BackendRefused as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        try:
            row = await complete(
                entity_client,
                workspace=workspace,
                name=name,
                delivered=body,
                policy=policy,
            )
        except NemoEntityNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Container image '{name}' not found.") from exc
        except SignatureRefused as exc:
            logger.warning(
                "signature for %s/%s refused: %s",
                sanitize_for_log(workspace),
                sanitize_for_log(name),
                sanitize_for_log(exc),
            )
            raise HTTPException(status_code=422, detail=f"Signature refused: {exc}") from exc
        except CompletionConflict as exc:
            logger.warning(
                "signature for %s/%s conflicts: %s",
                sanitize_for_log(workspace),
                sanitize_for_log(name),
                sanitize_for_log(exc),
            )
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        logger.info(
            "image %s/%s ready: %s (verified against %s)",
            sanitize_for_log(workspace),
            sanitize_for_log(name),
            sanitize_for_log(row.image_ref),
            policy.trust_root,
        )
        return row

    return router
