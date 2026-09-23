# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained Harbor task archive references."""

from typing import Annotated, Any, Literal

from filesets import parse_fileset_ref
from nemo_evaluator.api.fields import MetricInline, MetricRefOrInline
from nemo_evaluator.api.task_definitions.provenance import TaskProvenance
from nemo_evaluator.content_hash import DIGEST_PATTERN
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_archive import validate_archive_path
from nemo_evaluator_sdk.agent_eval.tasks import SemanticView
from nemo_helix_plugin.refs import FILESET_REF_PATTERN
from pydantic import BaseModel, ConfigDict, Field, field_validator

ArchiveDigest = Annotated[str, Field(pattern=DIGEST_PATTERN, min_length=64, max_length=64)]


class HarborArchiveSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["fileset-archive"] = "fileset-archive"
    fileset_ref: str = Field(
        pattern=FILESET_REF_PATTERN,
        description=(
            "Exact qualified archive object reference: workspace/fileset#relative/path. "
            "Workspace, fileset name, and path must each be NFC-normalized and at most 4096 UTF-8 bytes. "
            "Path segments must be nonempty, must not be '.' or '..', and must not end in a space or dot. "
            "Control characters (U+0000–U+001F and U+007F), backslashes, %, ?, #, and : are forbidden "
            "within each component. Absolute paths, a//b, a/../b, and trailing slashes are rejected. "
            "These canonicalization constraints are enforced by the server in addition to the pattern."
        ),
        examples=["default/harbor-tasks#suite/task/task_archive"],
    )
    files_hash: ArchiveDigest = Field(description="SHA-256 of the exact compressed archive bytes.")
    archive_format: Literal["tar-gzip-v1"] = "tar-gzip-v1"

    @field_validator("fileset_ref")
    @classmethod
    def _reference(cls, value: str) -> str:
        """Require a fully qualified Fileset reference whose components are safe archive paths."""
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


class _HarborTaskDefinitionCommon(BaseModel):
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
    views: dict[str, SemanticView] = Field(
        default_factory=dict,
        description="Reporting views over the primary Harbor reward and declared additional metric outputs.",
    )


class HarborTaskDefinition(_HarborTaskDefinitionCommon):
    metrics: list[MetricRefOrInline] = Field(
        default_factory=list,
        description="Additional metrics, appended to the mandatory HarborRewardMetric. Inline bundles are "
        "normalized to stored metric references on registration. Do not include HarborRewardMetric here.",
    )


class ResolvedHarborTaskDefinition(_HarborTaskDefinitionCommon):
    """Job snapshot with expanded metrics and the original stored revision."""

    provenance: TaskProvenance
    metrics: list[MetricInline] = Field(
        default_factory=list,
        description="Resolved additional metrics, appended to the mandatory HarborRewardMetric.",
    )
