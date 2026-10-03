# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Completion: who may make a row ``ready``, and the one write that does, at the digest its push step pushed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.identity import DIGEST_PATTERN
from nemo_helix_plugin.auth import current_auth_context, is_service_principal_id, platform_auth_enabled
from nemo_helix_plugin.entities import EntityConflictError
from pydantic import BaseModel, ConfigDict, Field


class CompleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    digest: str = Field(
        pattern=DIGEST_PATTERN,
        description="The manifest digest the push step pushed under the image's system tag, and signed.",
    )


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

    A person needs only the permission, which the policy checks. A service must act for the image's submitter, as
    the push step does without workload token exchange: the policy admits a service on its own permissions, whoever
    it acts for.
    """
    if not caller.auth or caller.actor is None or not is_service_principal_id(caller.actor):
        return None
    if caller.submitter != row.created_by:
        return f"a service may read {row.name} only acting for its submitter"
    return None


class CompletionConflict(Exception):
    """The row has already settled, and not as this completion would settle it (409)."""


class _Rows(Protocol):
    async def get(self, entity_type: type[ContainerImage], *, name: str, workspace: str) -> ContainerImage: ...

    async def update(self, entity: ContainerImage) -> ContainerImage: ...


def settled(row: ContainerImage, digest: str) -> bool:
    """Whether ``row`` is already ``ready`` at ``digest``. Raises :class:`CompletionConflict` if it settled otherwise."""
    if row.status == "ready":
        if row.digest == digest:
            return True
        raise CompletionConflict(f"{row.name} is already ready as {row.digest}, not {digest}")
    if row.status == "failed":
        raise CompletionConflict(f"{row.name} has already failed ({row.status_detail}); ready is not reachable from it")
    return False


async def complete(rows: _Rows, *, workspace: str, name: str, digest: str) -> ContainerImage:
    """Write the row ``name`` ``ready`` at ``digest``, conditional on the version it was read at.

    Raises :class:`CompletionConflict` if the row settled otherwise, or the store's not-found error.
    """
    for _ in range(2):
        row = await rows.get(ContainerImage, name=name, workspace=workspace)
        if settled(row, digest):
            return row
        row.digest = digest
        row.status = "ready"
        row.status_detail = None
        try:
            return await rows.update(row)
        except EntityConflictError:
            continue
    raise CompletionConflict(f"{name} kept changing while it was being completed")
