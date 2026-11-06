# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit, workspace-bounded Job addressing shared by clients and services."""

from dataclasses import dataclass
from enum import StrEnum


class JobReferenceKind(StrEnum):
    ID = "id"
    NAME = "name"


class JobMetadataReferenceKind(StrEnum):
    ID = "id"
    NAME = "name"
    LEGACY = "legacy"


@dataclass(frozen=True)
class JobById:
    job_id: str


@dataclass(frozen=True)
class JobByName:
    job_name: str


@dataclass(frozen=True)
class LegacyJobSelector:
    reference: str


JobSelector = JobById | JobByName | LegacyJobSelector


@dataclass(frozen=True)
class ResolvedJobIdentity:
    job_id: str
    job_name: str
    workspace: str
    parent_id: str | None


def job_selector(reference: str, kind: JobReferenceKind | None) -> JobSelector:
    """Normalize an existing overloaded HTTP entry point once."""
    if kind is None:
        return LegacyJobSelector(reference)
    match JobReferenceKind(kind):
        case JobReferenceKind.ID:
            return JobById(reference)
        case JobReferenceKind.NAME:
            return JobByName(reference)


def metadata_selector(reference: str, kind: JobMetadataReferenceKind) -> JobSelector:
    match JobMetadataReferenceKind(kind):
        case JobMetadataReferenceKind.ID:
            return JobById(reference)
        case JobMetadataReferenceKind.NAME:
            return JobByName(reference)
        case JobMetadataReferenceKind.LEGACY:
            return LegacyJobSelector(reference)
