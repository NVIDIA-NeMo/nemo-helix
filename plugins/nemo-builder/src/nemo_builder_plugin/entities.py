# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``ContainerImage``: one row per image a build publishes.

Only the builder writes rows; ``ready`` means the image's push step pushed its digest and signed it.
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
    spec: str | None = Field(
        default=None, description="The `BuildSpec.name` this row came from, or None for the set's unnamed spec."
    )
    job: str = Field(pattern=ENTITY_REF_PATTERN, description="`<workspace>/<name>` of the build job.")
    request_digest: str = Field(
        pattern=DIGEST_PATTERN,
        description=(
            "sha256 of the canonical JSON of the `BuildSet`. Tells a retry of the same request from a "
            "different request that reuses its revision. Every signature carries it."
        ),
    )


class ContainerImage(NemoEntity, entity_type="container_image"):
    """One image. ``name`` is ``<set>-<revision>.<spec>``, or ``<set>-<revision>`` for the set's unnamed spec."""

    status: Literal["pending", "ready", "failed"] = "pending"
    status_detail: str | None = None

    registry: str = Field(description="Registry host the image is published to.")
    repository: str = Field(description="Repository path within the registry.")
    provenance: Provenance
    platform: str = Field(default="linux/amd64", pattern=r"^[a-z0-9]+/[a-z0-9]+(/[a-z0-9]+)?$")

    # Written once, when the row becomes `ready`.
    digest: str | None = Field(
        default=None,
        pattern=DIGEST_PATTERN,
        description="The manifest digest the push step pushed under the system tag, and signed.",
    )

    @property
    def image_ref(self) -> str | None:
        if self.digest is None:
            return None
        return f"{self.registry}/{self.repository}@{self.digest}"
