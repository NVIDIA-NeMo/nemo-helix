# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen public identities of the executions included in a benchmark export."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class BenchmarkArchiveMember(BaseModel):
    id: str
    task_id: str
    task_revision: int
    task_slug: str | None = None
    task_name: str | None = None
    status: str
    current_execution: int
    archive_object_key: str
    archive_built_at: datetime
    archive_size_bytes: int


class ArchiveFile(BaseModel):
    """Checksum of one regular file, relative to its containing experiment."""

    model_config = ConfigDict(extra="forbid")

    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BenchmarkArchiveArtifact(ArchiveFile):
    object_key: str
    updated_at: str | None = None


class ArchivedTrial(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    exported: str


class BenchmarkArchiveExportMember(BenchmarkArchiveMember):
    """Source identity, verified inputs, and original public execution provenance."""

    model_config = ConfigDict(extra="forbid")

    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_integrity: Literal["verified", "manifest_missing"] = "manifest_missing"
    provenance_path: str | None = None
    # The producer owns this versioned JSON schema. Preserve all of its public
    # fields (including future runner/bundle metadata), not a lossy projection.
    provenance: dict[str, JsonValue] | None = None
    trials: list[ArchivedTrial] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class UnavailableBenchmarkEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_id: str
    kind: str


class BenchmarkArchiveManifest(BaseModel):
    """Typed analysis-export manifest; it never contains private DB snapshots."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["scaled-evals-benchmark-archive-v1"] = "scaled-evals-benchmark-archive-v1"
    benchmark_run_id: str
    generation: str
    harbor_job_id: UUID | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    members: list[BenchmarkArchiveExportMember] = Field(default_factory=list)
    benchmark_artifacts: list[BenchmarkArchiveArtifact] = Field(default_factory=list)
    unavailable_benchmark_evidence: list[UnavailableBenchmarkEvidence] = Field(default_factory=list)
    # Computed after rewriting Harbor metadata; excludes this manifest itself.
    files: list[ArchiveFile] = Field(default_factory=list)
    partial: bool = False
    n_exported_trial_results: int = Field(default=0, ge=0)
    n_declared_trials: int = Field(default=0, ge=0)
    notes: list[str] = Field(
        default_factory=lambda: [
            "Analysis export; original member job metadata is under _scaled_evals/evaluations.",
            "Custom metrics and pass@k are not aggregated; recompute them from trial results.",
            "Token/cost totals include recorded values only; absent values remain unknown.",
            "Member artifact_integrity reports producer-manifest verification; legacy inputs may lack that manifest.",
            "File checksums cover final exported bytes, excluding this manifest; the API provides the outer tarball checksum.",
        ]
    )


BenchmarkEvidenceCheck = Callable[
    [Path, list[BenchmarkArchiveExportMember], list[BenchmarkArchiveArtifact]],
    list[UnavailableBenchmarkEvidence],
]
