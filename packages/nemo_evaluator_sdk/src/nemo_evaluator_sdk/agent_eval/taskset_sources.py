# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared taskset source contracts."""

from pathlib import Path
from typing import Annotated, Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class TasksetSourceMaterialization(BaseModel):
    """Receipt returned after a taskset source has been materialized."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_uri: str
    materialized_root: Path
    revision_digest: Digest | None

    @field_validator("source_uri")
    @classmethod
    def _validate_source_uri(cls, value: str) -> str:
        try:
            scheme = urlsplit(value).scheme
        except ValueError as exc:
            raise ValueError("source_uri must be a nonempty absolute URI") from exc
        if not value or not scheme or any(character.isspace() for character in value):
            raise ValueError("source_uri must be a nonempty absolute URI")
        return value

    @field_validator("materialized_root")
    @classmethod
    def _validate_materialized_root(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("materialized_root must be an absolute path")
        return value


class TasksetSourceAdapter(Protocol):
    schemes: frozenset[str]

    async def materialize(self, source_uri: str, *, destination_root: Path) -> TasksetSourceMaterialization: ...
