# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The builder's request surface, registered under ``nemo.services``."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Iterator
from typing import ClassVar, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from nemo_builder_plugin._perms import BuildPerms, ContainerImagePerms
from nemo_builder_plugin.authz import scope
from nemo_builder_plugin.backend import BackendRejectedError
from nemo_builder_plugin.backends import load_backend
from nemo_builder_plugin.completion import (
    Caller,
    CompleteRequest,
    CompletionConflict,
    JobNotRecorded,
    JobOf,
    RecordedStep,
    caller_refusal,
    complete,
    current_caller,
    current_delegation,
    read_refusal,
    step_refusal,
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
    CreatedJob,
    ExistingJob,
    InvalidBuildRequest,
    MissingSecrets,
    UpstreamUnavailable,
    submit_build_set,
)
from nemo_builder_plugin.upstream import UpstreamRegistry
from nemo_helix_plugin.api.pagination import PageParams, make_page_dep
from nemo_helix_plugin.auth.workload_delegations import WorkloadDelegationEntity, WorkloadDelegationStore
from nemo_helix_plugin.auth.workload_identity import is_workload_identity_token_exchange_enabled
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
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nemo_helix_plugin.log_utils import sanitize_for_log
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nemo_helix_plugin.service import ExceptionHandler, NemoService, RouterSpec
from pydantic import ValidationError
from pydantic_core import ErrorDetails

logger = logging.getLogger(__name__)


async def _get_row(entity_client: NemoEntitiesClient, workspace: str, name: str) -> ContainerImage:
    try:
        return await entity_client.get(ContainerImage, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Container image '{name}' not found.") from exc


class Refused(Exception):
    """A submit the builder refuses with a 409 that says which kind it is.

    ``detail`` is a sentence, as a 409's always is. ``code`` is for a caller that must tell which it met.
    ``backend_rejected`` and ``missing_secrets`` refuse the request itself, under any revision.
    ``backend_changed``, like the 409 with no code for a revision another request holds, is one the next
    revision gets past.
    """

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _refused(_request: Request, refused: Exception) -> JSONResponse:
    """The answer to a :class:`Refused`: what an ``HTTPException`` would answer, with ``code`` beside ``detail``."""
    if not isinstance(refused, Refused):
        raise refused
    return JSONResponse(status_code=409, content={"detail": refused.detail, "code": refused.code})


#: The shortest value taken out of a logged validation message.
_MIN_REFUSED_CHARS = 4


def _setting(location: tuple[int | str, ...]) -> str:
    """The setting a validation error is about. A check on the section as a whole has no location: `builder`."""
    return ".".join(map(str, location)) or "builder"


def _refused_values(refused: object, depth: int = 0) -> Iterator[str]:
    """Each string in what a setting was given, and in a list or mapping of them, as a message might repeat it."""
    if isinstance(refused, str):
        # As a validator would most likely have written it, quoted, then bare. One trimmed of what a validator
        # trims first, too. Not a short one: it would be found in words that have nothing to do with it.
        for spelling in (refused, refused.strip(), refused.strip("/")):
            for form in (repr(spelling), repr(spelling)[1:-1], spelling):
                if len(form) >= _MIN_REFUSED_CHARS:
                    yield form
    elif depth < 4 and isinstance(refused, list | tuple | set | frozenset):
        for item in refused:
            yield from _refused_values(item, depth + 1)
    elif depth < 4 and isinstance(refused, dict):
        # Its values, not its keys: a check on a whole section is given the section, and names its settings.
        for item in refused.values():
            yield from _refused_values(item, depth + 1)


def _without_input(error: ErrorDetails) -> str:
    """A validation error's message, for the log, without the value it refused.

    Only a validator's own sentence could repeat what it was given. The builder's validators of a registry,
    the settings most likely to be written with a login in them, are written not to. This is for any that does:
    a setting may hold a credential.
    """
    message = error["msg"]
    if error["type"] not in ("value_error", "assertion_error"):
        # Pydantic's own sentence, which doesn't quote the value. Left as it is.
        return message
    for form in sorted(set(_refused_values(error.get("input"))), key=len, reverse=True):
        # As a word of its own: a check on a whole section is given every value in it, and one of them may be
        # part of a setting's name, which the sentence is there to give.
        message = re.sub(rf"(?<![\w-]){re.escape(form)}(?![\w-])", "<the value>", message)
    return message


def _unreadable(exc: ValueError) -> str:
    """What kind of error kept the section from being read, for the log, and which setting it names, if any.

    Never the error's own sentence. A parser's quotes what it stopped at, and the configuration file holds every
    section's settings: a value there may be a credential.
    """
    kinds = [type(error).__name__ for error in (exc, exc.__cause__) if error is not None]
    # Only a name the builder gave a setting is repeated, where the error quotes one: which variable wasn't JSON.
    named = [name for name in BuilderConfig.model_fields if f'"{name}"' in str(exc)]
    return ", caused by ".join(kinds) + (f", for {', '.join(named)}" if named else "")


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
        refusal = BackendRejectedError(f"the builder's configuration is invalid ({problems}); the API log says more")
    except ValueError as exc:
        # Not a setting that fails its check, but a section that can't be read at all: `builder:` left empty, or
        # an environment variable that isn't the JSON it must be. The error's own sentence may quote what it
        # couldn't read, so the log is given its kind instead.
        logger.error(
            "the builder's configuration can't be read (%s): the configuration file must be a YAML mapping with a "
            "mapping under `builder`, and a NEMO_BUILDER_* variable that holds a list or a section must be JSON",
            _unreadable(exc),
        )
        refusal = BackendRejectedError("the builder's configuration can't be read; the API log says why")
    # Raised out here, where no error is being handled, so it carries none as its cause or its context. A caller
    # may log the refusal with its traceback, as completing an image does, and the error under it would be
    # printed there: a validation error's own text repeats each value it refused.
    raise refusal


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


def _step_on_record(record: WorkloadDelegationEntity | None) -> RecordedStep | None:
    """What ``record`` says of the job step it was made for: the job's workspace and name, as an image's row
    names its build job, and the attempt of the job the step belongs to, as the row records the one its submit
    created. None if it was made for something else, or no longer stands.

    Jobs records these for each step's pod, and withdraws the record once the step has ended.
    """
    if record is None or not record.is_active() or record.workload_kind != "job":
        return None
    if not record.job_id or not record.attempt_id:
        return None
    # The platform doesn't record the job's own ID for a step's pod yet, only its name. Read where a later
    # platform does: it would tell a job created under the name from a rerun of the job itself.
    job_uid = getattr(record, "job_uid", None)
    return RecordedStep(
        workspace=record.workload_workspace,
        job=record.job_id,
        attempt_id=record.attempt_id,
        job_uid=job_uid if isinstance(job_uid, str) and job_uid else None,
    )


async def _job_of_as_builder(delegation: str) -> RecordedStep | None:
    # As the builder, not the caller: the record is the platform's own, kept where no submitter reads. By its
    # ID, which names one record: a later workload's is another, though it may be found under the same name.
    async with get_async_nemo_client(as_service="builder", internal=True) as client:
        records = WorkloadDelegationStore(EntityClient(AsyncEntitiesClient.from_client(client)))
        return _step_on_record(await records.get_by_id(delegation))


def get_job_of() -> JobOf:
    """How the builder learns which job step a caller's workload delegation was made for."""
    return _job_of_as_builder


def get_token_exchange() -> Callable[[], bool]:
    """How the builder learns whether the platform mints its workloads' tokens: workload token exchange.

    Asked only once a caller is known to be authenticated, of a row a build service built. It raises where the
    platform's auth configuration can't be read.
    """
    return is_workload_identity_token_exchange_enabled


def _unverified_completion_allowed() -> bool:
    return _load_config().remote.allow_unverified_completion


def get_allow_unverified() -> Callable[[], bool]:
    """How the builder learns whether the operator accepted completions it cannot verify
    (`builder.remote.allow_unverified_completion`). It raises where the builder's configuration can't be read."""
    return _unverified_completion_allowed


class BuilderService(NemoService):
    name: ClassVar[str] = "builder"
    dependencies: ClassVar[list[str]] = ["entities", "jobs", "secrets"]

    def get_exception_handlers(self) -> dict[type[Exception], ExceptionHandler]:
        return {Refused: _refused}

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
    ) -> SubmitBuildResponse:
        """Submit a set of images to build.

        Creates one `pending` ``ContainerImage`` per spec, then the job that builds them. Jobs starts the job once
        its build profiles have room (their `max_active_jobs`).
        """
        refusal = submit_refusal(caller)
        if refusal is not None:
            raise HTTPException(status_code=403, detail=refusal)
        jobs_client = AsyncJobsClient.from_client(client)

        async def create_job(request: CreateHelixJobRequest) -> CreatedJob:
            # What Jobs answers the create with, and gives no other job: kept, for the rows to record.
            job = (await jobs_client.create_job(workspace=workspace, body=request)).data()
            return CreatedJob(uid=job.id, attempt_id=job.attempt_id)

        async def get_job(name: str) -> ExistingJob:
            job = (await jobs_client.get_job(workspace=workspace, name=name)).data()
            return ExistingJob(uid=job.id, custom_fields=job.custom_fields or {})

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
                get_job=get_job,
                secret_readable=secret_readable,
                resolve_upstream=resolve_upstream,
                default_platform=config.default_platform,
            )
        except InvalidBuildRequest as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except BackendRejectedError as exc:
            raise Refused(CODE_BACKEND_REJECTED, str(exc)) from exc
        except MissingSecrets as exc:
            raise Refused(CODE_MISSING_SECRETS, str(exc)) from exc
        except BackendChanged as exc:
            raise Refused(CODE_BACKEND_CHANGED, str(exc)) from exc
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
        delegation: str | None = Depends(current_delegation),
        job_of: JobOf = Depends(get_job_of),
        token_exchange: Callable[[], bool] = Depends(get_token_exchange),
        allow_unverified: Callable[[], bool] = Depends(get_allow_unverified),
    ) -> ContainerImage:
        """Make an image `ready` at the digest the step that built it published.

        Completing an image already `ready` at that digest, with the same `external` and the same
        `runtime_config`, returns it, as does one that carries no `runtime_config` for an image this cluster
        built; 409 if it settled otherwise, if the request doesn't fit the image's backend, or if another
        image took its name meanwhile. For an image a build service built, with
        platform auth on: under workload token exchange, 403 to a caller that is not a step of the attempt the
        image records, and 503 while it records none; without workload token exchange, 403 to every caller
        unless `builder.remote.allow_unverified_completion` is set. With platform auth off no caller is checked.
        """

        async def admit(row: ContainerImage) -> None:
            """Raise what the route answers a caller who may not complete ``row``: 403, or 503 while that
            can't be told. Asked of the row the route reads, and again of any that is read in its place."""
            refusal = caller_refusal(row, caller)
            if refusal is None:
                try:
                    refusal = await step_refusal(
                        row,
                        caller,
                        delegation,
                        job_of,
                        token_exchange=token_exchange,
                        allow_unverified=allow_unverified,
                    )
                except JobNotRecorded as exc:
                    # The submit that created the build job may still be writing it on the row. A step waits
                    # this out, as it does an outage.
                    logger.warning(
                        "completing %s/%s put off: its row records no build job yet",
                        sanitize_for_log(workspace),
                        sanitize_for_log(name),
                    )
                    raise HTTPException(status_code=503, detail=f"{exc}; try again.") from exc
                except Exception as exc:
                    # No one is admitted on a record, or a deployment's settings, that couldn't be read.
                    # Answered as an outage, which a step waits out: its image is built, and only this call
                    # is left.
                    logger.exception(
                        "could not tell which job is completing %s/%s",
                        sanitize_for_log(workspace),
                        sanitize_for_log(name),
                    )
                    raise HTTPException(
                        status_code=503, detail="Couldn't check which job the caller is a step of; try again."
                    ) from exc
            if refusal is not None:
                logger.warning(
                    "completing %s/%s refused: %s",
                    sanitize_for_log(workspace),
                    sanitize_for_log(name),
                    sanitize_for_log(refusal),
                )
                raise HTTPException(status_code=403, detail=refusal)

        row = await _get_row(entity_client, workspace, name)
        await admit(row)
        try:
            # The row that was let through is the one written: not whatever is under its name by then.
            ready = await complete(
                entity_client,
                workspace=workspace,
                name=name,
                digest=body.digest,
                runtime_config=body.runtime_config,
                external=body.external,
                row=row,
                admit=admit,
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
