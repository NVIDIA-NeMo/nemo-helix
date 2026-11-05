# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Completion: who may make a row ``ready``, and the one write that does, at the digest its push step pushed."""

from __future__ import annotations

from typing import Protocol

from nemo_builder_plugin.entities import ContainerImage, ImageRuntimeConfig
from nemo_builder_plugin.identity import DIGEST_PATTERN
from nemo_helix_plugin.entities import EntityConflictError
from pydantic import BaseModel, ConfigDict, Field


class CompleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    digest: str = Field(
        pattern=DIGEST_PATTERN,
        description="The manifest digest the push step pushed under the image's system tag, and signed.",
    )
    runtime_config: ImageRuntimeConfig | None = Field(
        default=None, description="How the image starts by default, read from the config the push step pushed."
    )


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


async def complete(
    rows: _Rows, *, workspace: str, name: str, digest: str, runtime_config: ImageRuntimeConfig | None = None
) -> ContainerImage:
    """Write the row ``name`` ``ready`` at ``digest``, conditional on the version it was read at.

    ``runtime_config`` is written with it: how the image starts by default, from its config.

    Raises :class:`CompletionConflict` if the row settled otherwise, or the store's not-found error.
    """
    for _ in range(2):
        row = await rows.get(ContainerImage, name=name, workspace=workspace)
        if settled(row, digest):
            return row
        upstream = row.provenance.upstream
        if upstream is not None and digest != upstream.manifest_digest:
            raise CompletionConflict(f"{name} copies {upstream.manifest_digest}, not {digest}")
        row.digest = digest
        row.runtime_config = runtime_config
        row.status = "ready"
        row.status_detail = None
        try:
            return await rows.update(row)
        except EntityConflictError:
            continue
    raise CompletionConflict(f"{name} kept changing while it was being completed")
