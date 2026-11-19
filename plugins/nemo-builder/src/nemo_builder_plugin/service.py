# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The builder's request surface, registered under ``nemo.services``."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import ClassVar, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from nemo_builder_plugin._perms import BuildPerms, ContainerImagePerms
from nemo_builder_plugin.authz import scope
from nemo_builder_plugin.backend import BackendRejectedError
from nemo_builder_plugin.backends import load_backend
from nemo_builder_plugin.completion import (
    Caller,
    CompleteRequest,
    CompletionConflict,
    caller_refusal,
    complete,
    current_caller,
    read_refusal,
    submit_refusal,
)
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import MAX_REVISION, ContainerImage
from nemo_builder_plugin.identity import ImageReference
from nemo_builder_plugin.schema import (
    CODE_BACKEND_CHANGED,
    CODE_BACKEND_REJECTED,
    CODE_MISSING_SECRETS,
    BuildSet,
    SubmitBuildResponse,
)
from nemo_builder_plugin.settle import job_status_from, settle
from nemo_builder_plugin.submit import (
    BackendChanged,
    BuildConflict,
    InvalidBuildRequest,
    MissingSecrets,
    UpstreamUnavailable,
    submit_build_set,
)
from nemo_builder_plugin.upstream import UpstreamRegistry
from nemo_helix_plugin.api.pagination import PageParams, make_page_dep
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NemoHTTPError, NotFoundError, PermissionDeniedError
from nemo_helix_plugin.client_provider import get_async_nemo_client
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.entities import EntityClient
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
from pydantic import ValidationError
from pydantic_core import ErrorDetails

logger = logging.getLogger(__name__)


async def _get_row(entity_client: NemoEntitiesClient, workspace: str, name: str) -> ContainerImage:
    try:
        return await entity_client.get(ContainerImage, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Container image '{name}' not found.") from exc


def _refusal(code: str, exc: Exception) -> JSONResponse:
    """A 409 that says which kind it is.

    ``detail`` is the sentence it always was. ``code`` is for a caller that must tell a refusal, which the same
    request meets every time, from a revision another request holds, which the next revision gets past.
    """
    return JSONResponse(status_code=409, content={"detail": str(exc), "code": code})


def _setting(location: tuple[int | str, ...]) -> str:
    """The setting a validation error is about. A check on the section as a whole has no location: `builder`."""
    return ".".join(map(str, location)) or "builder"


def _without_input(error: ErrorDetails) -> str:
    """A validation error's message, with the value it refused taken out if the message repeats it."""
    message, refused = error["msg"], error.get("input")
    # As a validator would most likely have written it, quoted, then bare.
    for form in (repr(refused), str(refused)) if isinstance(refused, str | int | float) else ():
        if form:
            message = message.replace(form, "<the value>")
    return message


def _load_config() -> BuilderConfig:
    try:
        return BuilderConfig.get()
    except ValidationError as exc:
        # Named, not a bare 500: an operator who misspelt a key finds it in the answer. The answer says which
        # settings and what kind of problem. A validator's own sentence goes to the log only, and without the
        # value it refused, should it quote that: a setting may hold a credential, such as a registry given
        # with a login.
        errors = exc.errors(include_url=False)
        logger.error(
            "the builder's configuration is invalid: %s",
            sanitize_for_log("; ".join(f"{_setting(error['loc'])}: {_without_input(error)}" for error in errors)),
        )
        problems = "; ".join(
            f"{_setting(error['loc'])}: "
            + ("invalid value" if error["type"] in ("value_error", "assertion_error") else error["msg"])
            for error in errors
        )
        raise BackendRejectedError(
            f"the builder's configuration is invalid ({problems}); the API log says more"
        ) from exc
    except ValueError as exc:
        # Not a setting that fails its check, but a section that can't be read at all: `builder:` left empty, or
        # an environment variable that isn't the JSON it must be. The message may quote what it couldn't read.
        logger.error("the builder's configuration can't be read: %s", sanitize_for_log(str(exc)))
        raise BackendRejectedError("the builder's configuration can't be read; the API log says why") from exc


#: ``image`` as it stands once settled: failed if its job abandoned it.
Settle = Callable[[ContainerImage], Awaitable[ContainerImage]]


async def _settle_as_builder(image: ContainerImage) -> ContainerImage:
    # As the builder, not the reader: a reader who may not see the job must not have it taken for missing, and one
    # who may not write the image must not be refused, nor have the failure written in their name.
    try:
        async with get_async_nemo_client(as_service="builder", internal=True) as client:
            rows = EntityClient(AsyncEntitiesClient.from_client(client))
            return (await settle(rows, [image], job_status_from(AsyncJobsClient.from_client(client))))[0]
    except Exception:
        # Settling is best effort, as `settle` is: the read still returns the row, and the next read or
        # builder-settle pass settles it.
        logger.exception(
            "could not settle image %s/%s as the builder; returning it as stored",
            sanitize_for_log(image.workspace),
            sanitize_for_log(image.name),
        )
        return image


def get_settle() -> Settle:
    """How reading an image settles it, if its job abandoned it."""
    return _settle_as_builder


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
    # A service too, for a platform service that builds for a user, acting for that user.
    @path_rule(callers=[CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL], permissions=[BuildPerms.CREATE])
    async def submit_build(
        workspace: str,
        body: BuildSet,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
        client: AsyncNemoClient = Depends(get_nemo_client),
        caller: Caller = Depends(current_caller),
    ) -> SubmitBuildResponse | JSONResponse:
        """Submit a set of images to build.

        Creates one `pending` ``ContainerImage`` per spec, then the job that builds them. Jobs starts the job once
        its build profiles have room (their `max_active_jobs`).
        """
        refusal = submit_refusal(caller)
        if refusal is not None:
            raise HTTPException(status_code=403, detail=refusal)
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
            config = _load_config()
            backend = load_backend(config, await jobs_client.list_execution_profiles())
            result = await submit_build_set(
                body,
                backend=backend,
                workspace=workspace,
                entity_client=entity_client,
                create_job=create_job,
                get_job_fields=get_job_fields,
                secret_readable=secret_readable,
                resolve_upstream=resolve_upstream,
                default_platform=config.default_platform,
            )
        except InvalidBuildRequest as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except BackendRejectedError as exc:
            return _refusal(CODE_BACKEND_REJECTED, exc)
        except MissingSecrets as exc:
            return _refusal(CODE_MISSING_SECRETS, exc)
        except BackendChanged as exc:
            return _refusal(CODE_BACKEND_CHANGED, exc)
        except BuildConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except UpstreamUnavailable as exc:
            raise HTTPException(status_code=502, detail=f"Couldn't resolve an image to copy: {exc}") from exc
        except PermissionDeniedError as exc:
            # From Jobs, which lists the profiles and creates the job as the caller. A refused listing writes nothing;
            # a refused job leaves the rows written before it pending, until a read finds them with no job.
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
        revision: int | None = Query(
            default=None, ge=1, le=MAX_REVISION, description="Only the images of this `revision`."
        ),
        status: Literal["queued", "pending", "ready", "failed"] | None = Query(
            default=None,
            description=(
                "Only images in this stored state. The `builder-settle` controller fails an abandoned `pending` "
                "image within a minute or so, so until then `pending` includes it and `failed` doesn't. `queued` is "
                "deprecated: only 0.10's builder wrote it."
            ),
        ),
        page_params: PageParams = Depends(make_page_dep(max_size=100)),
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
    ) -> list[ContainerImage]:
        """List the images in a workspace, in whatever state they are stored in, or only one job's or one build set's.

        Listing doesn't settle: the `builder-settle` controller fails the abandoned ones.
        """
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
            ContainerImage,
            workspace=workspace,
            filter_obj=filters or None,
            page=page_params.page,
            page_size=page_params.page_size,
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
        caller: Caller = Depends(current_caller),
        settle_image: Settle = Depends(get_settle),
    ) -> ContainerImage:
        """Read one image. This is what a caller polls until `status` leaves `pending`.

        A `pending` image whose job has ended, or that is five minutes old and has no job, is failed as it is read.
        """
        row = await _get_row(entity_client, workspace, name)
        refusal = read_refusal(row, caller)
        if refusal is not None:
            raise HTTPException(status_code=403, detail=refusal)
        return await settle_image(row)

    @router.post("/container-images/{name}/complete", response_model=ContainerImage, tags=["Builder"])
    @scope.write
    @path_rule(callers=[CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL], permissions=[ContainerImagePerms.COMPLETE])
    async def complete_container_image(
        workspace: str,
        name: str,
        body: CompleteRequest,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
        caller: Caller = Depends(current_caller),
    ) -> ContainerImage:
        """Make an image `ready` at the digest the step that built it published.

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
            ready = await complete(
                entity_client,
                workspace=workspace,
                name=name,
                digest=body.digest,
                runtime_config=body.runtime_config,
                external=body.external,
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
