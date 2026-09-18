# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained Harbor task archive references."""

import unicodedata
from typing import Annotated, Any, Literal

from filesets import parse_fileset_ref
from nemo_evaluator.api.fields import MetricRefOrInline
from nemo_evaluator.content_hash import DIGEST_PATTERN
from nemo_evaluator_sdk.agent_eval.tasks import SemanticView
from nemo_platform_plugin.refs import FILESET_REF_PATTERN
from pydantic import BaseModel, ConfigDict, Field, field_validator

ArchiveDigest = Annotated[str, Field(pattern=DIGEST_PATTERN, min_length=64, max_length=64)]


def validate_archive_path(value: str) -> str:
    """Require an unambiguous relative POSIX path before normalization."""
    if (
        not value
        or len(value.encode("utf-8")) > 4096
        or any(part in {"", ".", ".."} or part.endswith((" ", ".")) for part in value.split("/"))
        or any(ord(char) < 32 or ord(char) == 127 or char in "\\%?#:" for char in value)
        or unicodedata.normalize("NFC", value) != value
    ):
        raise ValueError(f"Unsafe or ambiguous archive path: {value!r}")
    return value


class HarborArchiveSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["fileset-archive"] = "fileset-archive"
    fileset_ref: str = Field(pattern=FILESET_REF_PATTERN, description="Exact qualified archive object reference.")
    files_hash: ArchiveDigest = Field(description="SHA-256 of the exact compressed archive bytes.")
    archive_format: Literal["tar-gzip-v1"] = "tar-gzip-v1"

    @field_validator("fileset_ref")
    @classmethod
    def _reference(cls, value: str) -> str:
        workspace, name, path = parse_fileset_ref(value, workspace_fallback=None)
        if not workspace or not name or value != f"{workspace}/{name}#{path}":
            raise ValueError("Archive references must be qualified and canonical")
        for part in (workspace, name, path):
            validate_archive_path(part)
        return value


class HarborTaskHash(BaseModel):
    """Producer fingerprint for future use; never an integrity or execution gate."""

    model_config = ConfigDict(extra="forbid")
    digest: ArchiveDigest
    method: Literal["harbor-packager-content-v1"] = "harbor-packager-content-v1"
    harbor_version: str = Field(min_length=1, max_length=128)
    source_commit: str | None = Field(default=None, max_length=128)


class HarborTaskDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["harbor"]
    native_task_id: str = Field(
        min_length=1, description="Native task identity, independently verified from the archive."
    )
    source: HarborArchiveSource
    harbor_hash: HarborTaskHash
    instruction: str | None = None
    # Verified task.toml is authoritative, and this projection is excluded from revision identity.
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: list[MetricRefOrInline] = Field(
        default_factory=list,
        description="Additional metrics, appended to the mandatory HarborRewardMetric. Inline bundles are "
        "normalized to stored metric references on registration. Do not include HarborRewardMetric here.",
    )
    views: dict[str, SemanticView] = Field(
        default_factory=dict,
        description="Reporting views over the primary Harbor reward and declared additional metric outputs.",
    )
