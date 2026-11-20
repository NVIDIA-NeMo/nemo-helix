# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Completion: who may make a row ``ready``, and the one write that does, at the digest the step that built the
image published."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from nemo_builder_plugin.entities import IN_CLUSTER_BACKEND, ContainerImage, ExternalImage, ImageRuntimeConfig
from nemo_builder_plugin.identity import DIGEST_PATTERN
from nemo_helix_plugin.auth import (
    current_auth_client,
    current_auth_context,
    is_service_principal_id,
    platform_auth_enabled,
)
from nemo_helix_plugin.entities import EntityConflictError
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.requests import Request


class CompleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    digest: str = Field(
        pattern=DIGEST_PATTERN,
        description=(
            "The manifest digest the image is published at: the one the push step pushed under the image's system "
            "tag and signed, or the one a remote build service published."
        ),
    )
    runtime_config: ImageRuntimeConfig | None = Field(
        default=None,
        description=(
            "How the image starts by default: read from the config the push step pushed, or what a remote build "
            "service's client reported for the image it built. None when neither is known."
        ),
    )
    external: ExternalImage | None = Field(
        default=None,
        description="What a remote build service calls the image. Only a remote-build step sends it.",
    )

    @field_validator("external", mode="before")
    @classmethod
    def _external_has_no_unknown_fields(cls, value: object) -> object:
        """A misspelt key would otherwise make a row ready with no handle. The stored model stays lenient, so a
        row written by a later builder still reads."""
        if isinstance(value, dict) and (unknown := sorted(set(value) - set(ExternalImage.model_fields))):
            raise ValueError(f"unknown fields: {', '.join(map(str, unknown))}")
        return value


@dataclass(frozen=True, slots=True)
class Caller:
    """Who is asking. With platform auth off, nobody in particular."""

    auth: bool
    #: The principal the request came as. For a job step that is a service its headers name or, with a token,
    #: the ServiceAccount its pod runs as: neither says which job the step belongs to.
    actor: str | None = None
    #: Whom the actor acts for, when it is delegated.
    submitter: str | None = None


def current_caller() -> Caller:
    """The caller of the current request, from the platform's auth context."""
    if not platform_auth_enabled():
        return Caller(auth=False)
    context = current_auth_context()
    if context is None:
        return Caller(auth=True)
    return Caller(auth=True, actor=context.principal_id, submitter=context.principal_on_behalf_of)


#: What the platform's auth calls a token that its own auth service minted for a workload.
_WORKLOAD_TOKEN = "workload_access_token"
#: Where in such a token's claims the auth service names the delegation it minted the token from.
_DELEGATION_CLAIM = ("nhx_service_attributes", "auth", "delegation_id")


def verified_delegation(auth_client: object) -> str | None:
    """The ID of the workload delegation the request's token was minted from. None if it came with no such token.

    ``auth_client`` is what the platform's middleware leaves for a route once it has authenticated the request.
    Under workload token exchange, Jobs records a delegation for each step's pod, and the platform's auth
    service mints the step a token, in the delegation's name, only once the pod has proved which pod it is. The
    middleware has checked that token's signature before a route runs, so the ID is the platform's word for
    which workload is calling. It is read from that token's claims and nowhere else: not from a header or the
    request's body, which a caller writes, nor from a token of any other kind, which a caller may have had
    minted with claims of its choosing. A token that acts for no one is not a delegation's.
    """
    token = getattr(auth_client, "resolved_bearer_token", None)
    claims = getattr(token, "claims", None)
    if getattr(token, "token_kind", None) != _WORKLOAD_TOKEN or getattr(claims, "actor", None) is None:
        return None
    found: object = getattr(claims, "raw_claims", None)
    for key in _DELEGATION_CLAIM:
        found = found.get(key) if isinstance(found, Mapping) else None
    return found if isinstance(found, str) and found else None


def current_delegation(request: Request) -> str | None:
    """The ID of the workload delegation the current request's token was minted from, read from what the
    platform's middleware left for the request. None if it came with no such token, or platform auth is off."""
    if not platform_auth_enabled() or current_auth_context() is None:
        return None
    return verified_delegation(current_auth_client(request))


def caller_refusal(row: ContainerImage, caller: Caller) -> str | None:
    """Why ``caller`` may not complete ``row``, or None if it may. With platform auth off, anyone may.

    A user's own token is refused: the caller must act for the image's submitter, as the step that built it does.
    That doesn't show the caller is that step. Without workload token exchange the platform takes a request's
    identity from its headers, which anything inside the cluster, a build's own ``RUN`` included, can set. With
    it, a step's token says whom the step acts for and which ServiceAccount its pod runs as, and any other
    workload acting for the submitter passes this too. Which job the caller is a step of is asked next, of a
    row a build service built (:func:`step_refusal`).
    """
    if not caller.auth:
        return None
    if caller.submitter is None:
        return "only a job step acting for the image's submitter may complete it"
    if caller.submitter != row.created_by:
        return f"{row.name} was submitted by someone else"
    return None


#: ``<workspace>/<name>`` of the job that the workload delegation with this ID was made for a step of. None if
#: the platform has no such delegation on record any more, or it is not a job step's.
JobOf = Callable[[str], Awaitable[str | None]]


async def step_refusal(row: ContainerImage, delegation: str | None, job_of: JobOf) -> str | None:
    """Why a caller whose token names ``delegation`` is not a step of the job that builds ``row``, or None if
    it is, or nothing verified says.

    Asked of a row a build service built: its completion records the digest, the build and the runtime
    configuration the caller sends, and the builder can check none of them against the service. Where the
    request came with a token that names its workload's delegation, the job on that record must be the row's.
    The record names a job as the row does, by workspace and name, so a job put under the name of a build's
    job passes: once that job is deleted, or before the build is submitted, where the submit takes it for its own.

    A caller whose token names no delegation is not asked. Without workload token exchange no caller has one,
    and with it only a platform service's own token is without. A row built in this cluster is completed as it
    was before the builder asked this.
    """
    if row.provenance.backend == IN_CLUSTER_BACKEND or delegation is None:
        return None
    job = await job_of(delegation)
    if job is None:
        return f"only a step of its build job may complete {row.name}: the caller is not on record as a job step"
    if job != row.provenance.job:
        return f"only a step of its build job may complete {row.name}: the caller is a step of another job"
    return None


def read_refusal(row: ContainerImage, caller: Caller) -> str | None:
    """Why ``caller`` may not read ``row``, or None if it may.

    A person needs only the permission, which the policy checks. A service must act for a person, though not
    necessarily the image's submitter, so that one person's build can be reused for another. The policy admits a
    service on its own permissions, whoever it acts for, so the calling service must check that its user may read
    images here.
    """
    if not caller.auth or caller.actor is None or not is_service_principal_id(caller.actor):
        return None
    if caller.submitter is None or is_service_principal_id(caller.submitter):
        return f"a service may read {row.name} only acting for a person"
    return None


def submit_refusal(caller: Caller) -> str | None:
    """Why ``caller`` may not submit builds, or None if it may.

    A service must act for a person, as a platform service building for a user does. The entity store records the
    rows as created by that person; acting for no one, or for a service, they would be the builder's own, which no push
    step acts for, so none would ever complete. The policy admits a service on its own permissions, whoever it acts
    for.
    """
    if not caller.auth or caller.actor is None or not is_service_principal_id(caller.actor):
        return None
    if caller.submitter is None or is_service_principal_id(caller.submitter):
        return "a service may submit builds only acting for a person"
    return None


class CompletionConflict(Exception):
    """The completion can't be applied: the row settled otherwise, or the request doesn't fit the row (409)."""


class _Rows(Protocol):
    async def get(self, entity_type: type[ContainerImage], *, name: str, workspace: str) -> ContainerImage: ...

    async def update(self, entity: ContainerImage) -> ContainerImage: ...


def _starts_as_recorded(row: ContainerImage, runtime_config: ImageRuntimeConfig | None) -> bool:
    """Whether a completion carrying ``runtime_config`` would have written what ``row`` records of how its image
    starts.

    A row a build service built records what its one step sent, and that step sends the same on every try: a
    repeat carries exactly that, none included. A row built in this cluster may have been completed by a builder
    from before it recorded this, or the image's ``USER``. A push step whose completion is repeated across the
    upgrade sends them for an image that is published, so it is held only to what the row does record.
    """
    recorded = row.runtime_config
    if row.provenance.backend == IN_CLUSTER_BACKEND:
        if recorded is None:
            return True
        if recorded.user is None and runtime_config is not None:
            runtime_config = runtime_config.model_copy(update={"user": None})
    return recorded == runtime_config


def settled(
    row: ContainerImage,
    digest: str,
    external: ExternalImage | None = None,
    runtime_config: ImageRuntimeConfig | None = None,
) -> bool:
    """Whether ``row`` is already ``ready`` as a completion carrying these would have written it: at ``digest``,
    with ``external`` as the build service's handle for it, starting as ``runtime_config`` says.

    Raises :class:`CompletionConflict` if it settled otherwise.
    """
    if row.status == "ready":
        if row.digest != digest:
            raise CompletionConflict(f"{row.name} is already ready as {row.digest}, not {digest}")
        if row.external != external:
            # Returned as it is, the caller would take the row for the one it meant to write.
            raise CompletionConflict(f"{row.name} is already ready at {digest}, as another build's image")
        if not _starts_as_recorded(row, runtime_config):
            # And how the row says the image starts, which a consumer runs, for what it sent.
            raise CompletionConflict(f"{row.name} is already ready at {digest}, with another runtime configuration")
        return True
    if row.status == "failed":
        raise CompletionConflict(f"{row.name} has already failed ({row.status_detail}); ready is not reachable from it")
    return False


async def complete(
    rows: _Rows,
    *,
    workspace: str,
    name: str,
    digest: str,
    runtime_config: ImageRuntimeConfig | None = None,
    external: ExternalImage | None = None,
) -> ContainerImage:
    """Write the row ``name`` ``ready`` at ``digest``, conditional on the version it was read at.

    ``runtime_config`` is written with it: how the image starts by default, from its config. So is ``external``:
    what a remote build service calls the image. A row already ``ready`` as these would have written it is
    returned as it is.

    Raises :class:`CompletionConflict` if the row settled otherwise (it failed, or is ``ready`` at another digest,
    as another build's image or with another runtime configuration) or the request doesn't fit the row (its
    backend, or the digest it copies), or the store's not-found error.
    """
    for _ in range(2):
        row = await rows.get(ContainerImage, name=name, workspace=workspace)
        in_cluster = row.provenance.backend == IN_CLUSTER_BACKEND
        if external is not None and in_cluster:
            raise CompletionConflict(f"{name} is built in this cluster; no build service has a handle for it")
        if not in_cluster and (external is None or external.build_id is None):
            # A remote row is ready only with the name of the build that made it. It shows which build, not who
            # is calling: any caller can name one. Who is calling is asked before this (`step_refusal`).
            raise CompletionConflict(f"{name} is built by a build service; its completion must name the build")
        if settled(row, digest, external, runtime_config):
            return row
        upstream = row.provenance.upstream
        if upstream is not None and digest != upstream.manifest_digest:
            raise CompletionConflict(f"{name} copies {upstream.manifest_digest}, not {digest}")
        row.digest = digest
        row.runtime_config = runtime_config
        row.external = external
        row.status = "ready"
        row.status_detail = None
        try:
            return await rows.update(row)
        except EntityConflictError:
            continue
    raise CompletionConflict(f"{name} kept changing while it was being completed")
