# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-independent integrity and identity checks for analysis exports."""

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ValidationError, field_validator

from scaled_evals.models.benchmark_archives import ArchiveFile

ARTIFACT_MANIFEST_NAME = "scaled-evals-manifest.json"


class BenchmarkArchiveError(ValueError):
    """An input cannot be safely or consistently included in an export."""


class ProducerArtifactFile(ArchiveFile):
    @field_validator("sha256", mode="before")
    @classmethod
    def normalize_digest(cls, value: str) -> str:
        return value.removeprefix("sha256:") if isinstance(value, str) else value


class ProducerArtifactManifest(BaseModel):
    schema_version: Literal["scaled-evals-artifacts-v1"]
    files: list[ProducerArtifactFile]


def read_archive_json(path: Path) -> dict:
    """Bound metadata reads independently of the archive's total byte budget."""
    if path.stat().st_size > 64 * 1024 * 1024:
        raise BenchmarkArchiveError(f"JSON metadata exceeds 64 MiB: {path.name}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise BenchmarkArchiveError(f"expected JSON object: {path.name}")
    return value


def validate_archive_id(value: str) -> str:
    """Run/evaluation IDs become directory names, so require one safe component.

    Production IDs such as bmr_<uuid> and ev_<uuid> are opaque identifiers, not
    filesystem paths. This is a traversal guard, not a list of test identifiers.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise BenchmarkArchiveError("invalid archive identifier: expected one safe path component")
    return value


def benchmark_archive_object_key(run_id: str, generation: str, claim_token: str) -> str:
    """One object per claim; revoked workers cannot overwrite a retry's upload."""
    return (
        f"benchmark-runs/{validate_archive_id(run_id)}/archives/"
        f"{validate_archive_id(generation)}/{validate_archive_id(claim_token)}.tar.gz"
    )


def harbor_job_uuid(run_id: str, generation: str) -> UUID:
    """Give Harbor a UUID stable across retries of one export generation.

    The benchmark run ID is not itself a UUID. UUID5 preserves the association
    while a forced rebuild's new generation gets a distinct Harbor job identity.
    The worker claim token deliberately does not participate in this identity.
    """
    return uuid5(NAMESPACE_URL, f"scaled-evals:{run_id}:{generation}")


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_artifact_manifest(root: Path, check_claim: Callable[[], None]) -> Literal["verified", "manifest_missing"]:
    """Cross-check the downloaded files before rewriting their Harbor metadata.

    These are producer-recorded checksums, not signatures. A legacy archive with
    no producer manifest remains readable but must not be labelled verified.
    """
    manifest_path = root / ARTIFACT_MANIFEST_NAME
    if not manifest_path.is_file():
        return "manifest_missing"
    try:
        manifest = ProducerArtifactManifest.model_validate(read_archive_json(manifest_path))
    except (ValidationError, ValueError) as exc:
        raise BenchmarkArchiveError("invalid producer artifact manifest") from exc
    expected: set[str] = set()
    for entry in manifest.files:
        check_claim()
        path = PurePosixPath(entry.path)
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in entry.path
            or path.as_posix() != entry.path
            or entry.path in expected
            or entry.path == ARTIFACT_MANIFEST_NAME
        ):
            raise BenchmarkArchiveError("unsafe or duplicate producer manifest path")
        expected.add(entry.path)
        source = root / entry.path
        if not source.is_file() or source.stat().st_size != entry.size_bytes or file_sha256(source) != entry.sha256:
            raise BenchmarkArchiveError(f"producer manifest checksum mismatch: {entry.path}")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path != manifest_path}
    if actual != expected:
        raise BenchmarkArchiveError("producer manifest does not cover all archive files")
    return "verified"
