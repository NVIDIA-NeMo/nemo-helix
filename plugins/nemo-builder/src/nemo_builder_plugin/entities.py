# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``ContainerImage``: one row per image a build publishes.

The submit route creates each row ``pending``, before anything is built. A row becomes ``ready``
when the platform verifies a signature over the image's digest, and ``failed`` when its build job
ends without one. Build pods never write rows. A ``ready`` or ``failed`` row never changes; a
rebuild makes a new row.
"""

from __future__ import annotations

from typing import Literal

from nemo_builder_plugin.identity import DIGEST_PATTERN
from nemo_helix_plugin.entity import NemoEntity
from nemo_helix_plugin.refs import ENTITY_REF_PATTERN
from pydantic import BaseModel, Field


class Provenance(BaseModel):
    """The build that produces the image. Written at submit, and never changed."""

    build_set: str = Field(description="The `BuildSet.name` this row came from.")
    revision: int = Field(ge=1, description="The `BuildSet.revision` this row came from.")
    job: str = Field(pattern=ENTITY_REF_PATTERN, description="`<workspace>/<name>` of the build job.")
    system_tag: str = Field(description="The tag the build pushes and the broker signs.")
    request_digest: str = Field(
        pattern=DIGEST_PATTERN,
        description=(
            "sha256 of the canonical JSON of the `BuildSet`. Tells a retry of the same request from a "
            "different request that reuses its revision. Every signature carries it."
        ),
    )


class Signature(BaseModel):
    """The verified signature a row was made ``ready`` on. The signature itself is in the registry."""

    verified_against: str = Field(description="The trust root the signature was verified against.")
    signer: dict[str, str] = Field(
        default_factory=dict,
        description="The signed claims that tie the signature to this row.",
    )


class ContainerImage(NemoEntity, entity_type="container_image"):
    """One image. ``name`` is ``<set>-<revision>-<index>``, the index being the spec's position in the set."""

    status: Literal["pending", "ready", "failed"] = "pending"
    status_detail: str | None = None

    # Written at submit.
    registry: str = Field(description="Registry host the image is published to.")
    repository: str = Field(description="Repository path within the registry.")
    provenance: Provenance
    platform: str = Field(default="linux/amd64", pattern=r"^[a-z0-9]+/[a-z0-9]+(/[a-z0-9]+)?$")

    # Written once, when the row becomes `ready`.
    digest: str | None = Field(
        default=None,
        pattern=DIGEST_PATTERN,
        description="The manifest digest the verified signature covers.",
    )
    signature: Signature | None = Field(default=None, description="The signature the row was made `ready` on.")

    @property
    def image_ref(self) -> str | None:
        """``<registry>/<repository>@<digest>``, once the row is ``ready``."""
        if self.digest is None:
            return None
        return f"{self.registry}/{self.repository}@{self.digest}"
