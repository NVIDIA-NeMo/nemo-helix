# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Visibility = Literal["private", "team", "org", "public"]
RevisionStatus = Literal["pending", "uploading", "building", "ready", "failed"]

SLUG_MAX_LEN = 63
SLUG_PATTERN = rf"^[a-z0-9][a-z0-9-]{{0,{SLUG_MAX_LEN - 1}}}$"


# Request body: POST /v1/tasks
class TaskCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    slug: str | None = Field(default=None, pattern=SLUG_PATTERN)
    description: str | None = None
    visibility: Visibility = "private"


# Request body: PATCH /v1/tasks/{id}. Only identity/metadata is mutable;
# revision contents (tasks, build) are immutable and not patchable here.
class TaskUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    slug: str | None = Field(default=None, pattern=SLUG_PATTERN)
    description: str | None = None
    visibility: Visibility | None = None


# Nested in POST /v1/tasks response (upload field)
class TaskUpload(BaseModel):
    method: Literal["PUT"] = "PUT"
    url: str
    headers: dict[str, str] = Field(default_factory=lambda: {"Content-Type": "application/gzip"})
    # "put" is an ordinary S3-compatible presigned PUT. "gcs_resumable" is a
    # GCS upload session URL; clients must include Content-Range for the full
    # tarball bytes.
    mode: Literal["put", "gcs_resumable"] = "put"


# Response: GET /v1/tasks/{id}, /by-slug/{slug}, and list items
class Task(BaseModel):
    id: str
    name: str
    slug: str
    description: str | None
    visibility: Visibility
    current_revision: int | None
    created_at: datetime
    updated_at: datetime


# Response: GET /v1/tasks/{id} — task plus its latest revision's
# build state (docs/API.md: detail includes "revision, tasks, build status").
# All revision fields are null until a revision exists / a build completes.
class TaskDetail(Task):
    revision: int | None = None
    status: RevisionStatus | None = None
    image_ref: str | None = None
    image_digest: str | None = None
    # Set only for tasks whose verifier runs in its own sandbox, built from tests/.
    verifier_image_ref: str | None = None
    verifier_image_digest: str | None = None
    build_error: str | None = None
    tarball_size_bytes: int | None = None
    tarball_sha256: str | None = None


# Response body: POST /v1/tasks
class TaskCreateResponse(Task):
    revision: int
    status: RevisionStatus
    upload: TaskUpload
    links: dict[str, str]


class TaskFinalizeResponse(BaseModel):
    id: str
    revision: int
    status: RevisionStatus


# Request body: POST /v1/tasks/{id}/finalize
class TaskFinalizeRequest(BaseModel):
    """Optional finalize body selecting reuse of an already signed image.

    Omit the body to build and sign the already-uploaded task pack through the
    configured image-builder-service path, or through local BuildKit when the
    image-builder service is not configured.
    """

    model_config = ConfigDict(extra="forbid")

    # Reuse an externally produced image that is already signed and pullable.
    # Import orchestration pins this so a later upload cannot redirect finalize.
    revision: int | None = Field(default=None, ge=1)
    image_ref: str | None = None
    image_digest: str | None = None
    # Prebuilt image for a verifier that runs in its own sandbox ([verifier]
    # environment_mode = "separate"). Only accepted together with image_ref.
    verifier_image_ref: str | None = None
    verifier_image_digest: str | None = None
    tarball_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class TaskRevisionCreateResponse(TaskFinalizeResponse):
    upload: TaskUpload
    links: dict[str, str]


class TaskPackReconciliationItem(BaseModel):
    task_id: str
    revision: int
    status: RevisionStatus
    object_key: str
    missing: bool
    repaired: bool = False


class TaskPackReconciliationResponse(BaseModel):
    owner_id: str
    checked: int
    missing: int
    repaired: int
    items: list[TaskPackReconciliationItem]
