# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Completion: who may make a row ``ready``, and the one write that does, at the digest the step that built the
image published."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from nemo_builder_plugin.entities import IN_CLUSTER_BACKEND, ContainerImage, ExternalImage, ImageRuntimeConfig
from nemo_builder_plugin.identity import DIGEST_PATTERN
from nemo_helix_plugin.auth import current_auth_context, is_service_principal_id, platform_auth_enabled
from nemo_helix_plugin.entities import EntityConflictError
from pydantic import BaseModel, ConfigDict, Field, field_validator


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
            "How the image starts by default, read from the config the push step pushed. A remote-build step "
            "sends none."
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
    #: The principal the request came as: with a job's token, the job step.
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


def caller_refusal(row: ContainerImage, caller: Caller) -> str | None:
    """Why ``caller`` may not complete ``row``, or None if it may. With platform auth off, anyone may.

    A user's own token is refused: the caller must act for the image's submitter, as its push step does.
    That doesn't show the caller is the push step: the platform also takes a request's identity from its
    headers, which anything inside the cluster, a build's own ``RUN`` included, can set.
    """
    if not caller.auth:
        return None
    if caller.submitter is None:
        return "only a job step acting for the image's submitter may complete it"
    if caller.submitter != row.created_by:
        return f"{row.name} was submitted by someone else"
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


def settled(row: ContainerImage, digest: str, external: ExternalImage | None = None) -> bool:
    """Whether ``row`` is already ``ready`` at ``digest``, with ``external`` as the build service's handle for it.

    Raises :class:`CompletionConflict` if it settled otherwise.
    """
    if row.status == "ready":
        if row.digest != digest:
            raise CompletionConflict(f"{row.name} is already ready as {row.digest}, not {digest}")
        if row.external != external:
            # Returned as it is, the caller would take the row for the one it meant to write.
            raise CompletionConflict(f"{row.name} is already ready at {digest}, as another build's image")
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
    what a remote build service calls the image.

    Raises :class:`CompletionConflict` if the row settled otherwise or the request doesn't fit the row (its
    backend, or the digest it copies), or the store's not-found error.
    """
    for _ in range(2):
        row = await rows.get(ContainerImage, name=name, workspace=workspace)
        in_cluster = row.provenance.backend == IN_CLUSTER_BACKEND
        if external is not None and in_cluster:
            raise CompletionConflict(f"{name} is built in this cluster; no build service has a handle for it")
        if not in_cluster and (external is None or external.build_id is None):
            # A remote row is ready only with the name of the build that made it. It shows which build, not who
            # is calling: any step acting for the submitter can name one.
            raise CompletionConflict(f"{name} is built by a build service; its completion must name the build")
        if settled(row, digest, external):
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
