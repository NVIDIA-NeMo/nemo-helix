# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episode provisioning broker contract, shared by NeMo-Gym and NeMo-RL.

Vendored from NeMo-Gym rather than imported. The definitions live on an unmerged branch
(``soluwalana/Gym@nhx/customizer``, commit ``f2a47392``), which is 180 commits behind upstream and
carries no release — so depending on them would pin the platform to a personal fork. Upstream
``NVIDIA-NeMo/Gym`` has no ``sandbox.broker`` module at all.

This is a *contract*, not an implementation: request/response models, path and header constants, and
validators. ``BROKER_PROTOCOL_VERSION`` is what detects drift between the two copies — bump it there
and here together, and check it on the wire.
"""

import base64
import binascii
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sandboxed_gym.sandbox_types import SandboxStatus

# Bump when an already-deployed client can no longer be assumed to interoperate -- either because
# a change breaks it, or because a newer client needs to know whether the broker has a capability
# it wants. Reported by ``GET /health`` so a job-sandbox image built against a different NeMo-Gym
# revision than the broker's NeMo-RL image fails loudly instead of drifting.
#
# "2" added the directory transfer endpoints. Nothing in "1" broke: a v1 client talks to a v2
# broker unchanged. A v2 client against a v1 broker would get 404s from `/dirs`, which is what the
# version check exists to turn into a clear failure.
BROKER_PROTOCOL_VERSION = "2"

BROKER_AUTH_HEADER = "OPENSANDBOX-EPISODE-BROKER-AUTH"

# Auth header for the *orchestrator proxy* (`/health`, `/rollouts/run`) -- a different surface from
# the broker above, and a different token. Declared here, with no server dependency, so a rollout
# client can agree on the name without importing the proxy app and its web stack.
PROXY_AUTH_HEADER = "X-Sandboxed-Gym-Token"

# How the job-sandbox runtime tells NeMo-Gym's public sandbox API to route to a broker. Named here,
# in the contract both sides share, so the process that sets them and the process that reads them
# cannot drift apart. Setting the URL is what turns brokered mode on; see
# ``nemo_gym.sandbox.api`` for why that switch is a compatibility mechanism and not a security one.
BROKER_URL_ENV = "NEMO_GYM_SANDBOX_BROKER_URL"
BROKER_TOKEN_ENV = "NEMO_GYM_SANDBOX_BROKER_TOKEN"

HEALTH_PATH = "/health"
EPISODES_PATH = "/episodes"
EPISODE_PATH = "/episodes/{episode_id}"
EPISODE_EXEC_PATH = "/episodes/{episode_id}/exec"
EPISODE_FILES_PATH = "/episodes/{episode_id}/files"
EPISODE_DIRS_PATH = "/episodes/{episode_id}/dirs"


class BrokerErrorCode(str, Enum):
    """Machine-readable reason for a broker rejection.

    Clients map these onto NeMo-Gym sandbox exceptions so an environment fails fast with an
    actionable message instead of a generic HTTP error mid-rollout.
    """

    UNAUTHORIZED = "unauthorized"
    INVALID_REQUEST = "invalid_request"
    FIELD_NOT_ALLOWED = "field_not_allowed"
    IMAGE_NOT_APPROVED = "image_not_approved"
    EPISODE_NOT_FOUND = "episode_not_found"
    QUOTA_EXCEEDED = "quota_exceeded"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    BACKEND_ERROR = "backend_error"
    # The broker is tearing down and will not provision anything further. Existing episodes can
    # still be operated on and closed; only creation is refused.
    SHUTTING_DOWN = "shutting_down"


def validate_base64(value: str) -> str:
    """Return ``value`` if it is standard base64, else raise ``ValueError``."""
    try:
        base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError("content must be standard base64") from e
    return value


def validate_absolute_path(value: str) -> str:
    """Return ``value`` if it is an absolute POSIX path with no traversal, else raise ``ValueError``.

    Traversal inside an episode is not a cluster-boundary escape (the episode is itself isolated),
    but the broker refuses to relay ambiguous paths so a backend can never resolve one differently
    than the caller intended.
    """
    if "\x00" in value:
        raise ValueError("path must not contain NUL bytes")
    if not value.startswith("/"):
        raise ValueError("path must be absolute")
    if any(part == ".." for part in PurePosixPath(value).parts):
        raise ValueError("path must not contain '..' segments")
    return value


class EpisodeResources(BaseModel):
    """Resource request for one episode; mirrors :class:`nemo_gym.sandbox.SandboxResources`.

    Typed rather than a free-form mapping so unknown keys cannot ride through to a backend SDK and
    so the broker has a fixed set of fields to cap.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cpu: float | None = Field(default=None, gt=0)
    memory_mib: int | None = Field(default=None, gt=0)
    disk_gib: int | None = Field(default=None, gt=0)
    gpu: int | None = Field(default=None, ge=0)
    gpu_type: str | None = None


class EpisodeCreateRequest(BaseModel):
    """``POST /episodes`` -- the job sandbox asking for one episode sandbox.

    ``image`` is required even though :class:`nemo_gym.sandbox.SandboxSpec` allows ``None``: it is
    what the broker's approved-image policy is evaluated against, and a request the broker cannot
    evaluate is a request it must not forward.

    ``ttl_s`` left unset means "use the broker's default"; lifetime policy belongs to the trusted
    side. ``provider_options`` is accepted so a client never has to guess what is permitted, but
    the broker rejects any key it has not explicitly allowed rather than silently dropping it.
    """

    model_config = ConfigDict(extra="forbid")

    image: str = Field(min_length=1)
    ttl_s: float | None = Field(default=None, gt=0)
    ready_timeout_s: float | None = Field(default=None, gt=0)
    workdir: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    resources: EpisodeResources = Field(default_factory=EpisodeResources)
    entrypoint: list[str] | None = None
    files_b64: dict[str, str] = Field(default_factory=dict)
    provider_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("workdir")
    @classmethod
    def _check_workdir(cls, value: str | None) -> str | None:
        return value if value is None else validate_absolute_path(value)

    @field_validator("files_b64")
    @classmethod
    def _check_files(cls, value: dict[str, str]) -> dict[str, str]:
        for path, content in value.items():
            validate_absolute_path(path)
            validate_base64(content)
        return value


class EpisodeCreateResponse(BaseModel):
    """``POST /episodes`` result. ``episode_id`` is a broker-owned opaque handle."""

    model_config = ConfigDict(frozen=True)

    episode_id: str
    status: SandboxStatus = SandboxStatus.RUNNING


class EpisodeStatusResponse(BaseModel):
    """``GET /episodes/{episode_id}`` result."""

    model_config = ConfigDict(frozen=True)

    status: SandboxStatus


class EpisodeExecRequest(BaseModel):
    """``POST /episodes/{episode_id}/exec`` -- run one command inside an episode.

    ``user`` defaults to ``None`` (the backend's own default) rather than ``"root"``: privileged
    execution is something a call site opts into explicitly. Backends that cannot honour a
    requested user reject the call with ``UNSUPPORTED_OPERATION`` instead of quietly downgrading.
    """

    model_config = ConfigDict(extra="forbid")

    command: str = Field(min_length=1)
    cwd: str | None = None
    env: dict[str, str] | None = None
    user: str | int | None = None
    timeout_s: float | None = Field(default=None, gt=0)

    @field_validator("cwd")
    @classmethod
    def _check_cwd(cls, value: str | None) -> str | None:
        return value if value is None else validate_absolute_path(value)


class EpisodeExecResponse(BaseModel):
    """``POST /episodes/{episode_id}/exec`` result; mirrors ``SandboxExecResult``."""

    model_config = ConfigDict(frozen=True)

    stdout: str | None
    stderr: str | None
    return_code: int
    error_type: str | None = None


class EpisodeFileUploadRequest(BaseModel):
    """``PUT /episodes/{episode_id}/files`` -- write one file into an episode.

    A successful upload answers ``204 No Content``; there is no response body to parse.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    content_b64: str

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        return validate_absolute_path(value)

    @field_validator("content_b64")
    @classmethod
    def _check_content(cls, value: str) -> str:
        return validate_base64(value)


class EpisodeFileDownloadRequest(BaseModel):
    """``GET /episodes/{episode_id}/files`` query parameters."""

    model_config = ConfigDict(extra="forbid")

    path: str

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        return validate_absolute_path(value)


class EpisodeFileDownloadResponse(BaseModel):
    """``GET /episodes/{episode_id}/files`` result."""

    model_config = ConfigDict(frozen=True)

    content_b64: str


class EpisodeDirUploadRequest(BaseModel):
    """``PUT /episodes/{episode_id}/dirs`` -- unpack a directory tree into an episode.

    The payload is a **gzipped tar archive**, not a file-by-file mapping. Per-file transfer would
    drop mode bits, symlinks and empty directories, and a seeded workspace whose scripts arrive
    without their execute bit fails at run time rather than at upload. Archive members are relative
    to ``path``.

    A successful upload answers ``204 No Content``. The whole archive travels in one request, so it
    is bounded by ``max_request_bytes`` -- see :data:`EpisodeDirDownloadResponse` on why this
    contract has no chunked form yet.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    archive_b64: str

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        return validate_absolute_path(value)

    @field_validator("archive_b64")
    @classmethod
    def _check_archive(cls, value: str) -> str:
        return validate_base64(value)


class EpisodeDirDownloadRequest(BaseModel):
    """``GET /episodes/{episode_id}/dirs`` query parameters."""

    model_config = ConfigDict(extra="forbid")

    path: str

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        return validate_absolute_path(value)


class EpisodeDirDownloadResponse(BaseModel):
    """``GET /episodes/{episode_id}/dirs`` result: a gzipped tar of ``path``'s contents.

    Single-shot, like the upload. A chunked form would need transfer sessions, offsets, resumption
    and orphan cleanup -- a protocol of its own -- so the broker instead refuses an oversized
    transfer with ``413`` and names the limit. Nothing on the current path is known to exceed it;
    revisit when something does.
    """

    model_config = ConfigDict(frozen=True)

    archive_b64: str


class EpisodeCloseResponse(BaseModel):
    """``DELETE /episodes/{episode_id}`` result."""

    model_config = ConfigDict(frozen=True)

    closed: bool = True


class BrokerHealthResponse(BaseModel):
    """``GET /health`` result. Authenticated like every other route -- an unauthenticated readiness
    endpoint is a free oracle for anything that can reach the leader pod.
    """

    model_config = ConfigDict(frozen=True)

    status: str = "ok"
    job_id: str
    protocol_version: str = BROKER_PROTOCOL_VERSION


class BrokerErrorResponse(BaseModel):
    """Error body returned for every non-2xx broker response."""

    model_config = ConfigDict(frozen=True)

    error: str
    code: BrokerErrorCode
