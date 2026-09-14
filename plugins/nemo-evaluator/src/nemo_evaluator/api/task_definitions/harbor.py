# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stored Harbor tasks reference a verified, manifest-backed Fileset tree."""

import unicodedata
from typing import Annotated, Any, Literal, Self

from filesets import parse_fileset_ref
from nemo_evaluator.content_hash import DIGEST_PATTERN
from nemo_platform_plugin.refs import FILESET_REF_PATTERN
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TreeDigest = Annotated[str, Field(pattern=DIGEST_PATTERN, min_length=64, max_length=64)]


def validate_tree_path(value: str) -> str:
    """Require an unambiguous relative POSIX path before any normalization."""
    if (
        not value
        or len(value.encode("utf-8")) > 4096
        or any(part in {"", ".", ".."} or part.endswith((" ", ".")) for part in value.split("/"))
        or any(ord(char) < 32 or ord(char) == 127 or char in "\\%?#:" for char in value)
        or unicodedata.normalize("NFC", value) != value
    ):
        raise ValueError(f"Unsafe or ambiguous tree path: {value!r}")
    return value


class HarborTreeSource(BaseModel):
    """Exact manifest and complete task content, independently verified by the worker."""

    model_config = ConfigDict(extra="forbid")
    format: Literal["fileset-tree-v1"] = "fileset-tree-v1"
    root_ref: str = Field(pattern=FILESET_REF_PATTERN, description="Qualified Files reference to the file prefix.")
    manifest_ref: str = Field(pattern=FILESET_REF_PATTERN, description="Qualified Files reference to the manifest.")
    manifest_digest: TreeDigest = Field(description="SHA-256 of the exact canonical manifest bytes.")
    tree_digest: TreeDigest = Field(description="Strict digest_harbor_tree checksum of the complete task directory.")

    @field_validator("root_ref", "manifest_ref")
    @classmethod
    def _reference(cls, value: str) -> str:
        workspace, name, path = parse_fileset_ref(value, workspace_fallback=None)
        if not workspace or not name or value != f"{workspace}/{name}#{path}":
            raise ValueError("Tree references must be qualified and canonical")
        validate_tree_path(workspace)
        validate_tree_path(name)
        validate_tree_path(path)
        return value

    @model_validator(mode="after")
    def _manifest_outside_tree(self) -> Self:
        if self.manifest_ref == self.root_ref or self.manifest_ref.startswith(self.root_ref + "/"):
            raise ValueError("Manifest must be outside the executable tree prefix")
        return self


class HarborTaskDefinition(BaseModel):
    """Task files and a queryable projection of Harbor's config; the target selects the agent."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["harbor"] = Field(description="Task kind discriminator.")
    tree: HarborTreeSource
    instruction: str | None = Field(default=None, description="Task instruction text, when present.")
    # Excluded from revision identity: verified task.toml is authoritative at execution.
    config: dict[str, Any] = Field(default_factory=dict, description="Queryable projection of Harbor task.toml.")
