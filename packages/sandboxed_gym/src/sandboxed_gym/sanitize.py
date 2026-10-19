# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rebuild untrusted broker requests into specs the broker is willing to forward.

This is the broker's trust boundary. Its governing rule is **allowlist reconstruction**: build a
fresh spec out of known-safe fields rather than deleting dangerous ones from the caller's request.
You cannot forward what you never read, so a backend that later grows a new passthrough field
cannot reopen a path that was closed here.

The concrete levers this closes, in the OpenSandbox reference backend, are volumes (host bind
mounts, a neighbour's PVC), ``platform``, snapshot references that would sidestep the image
policy, and ``extensions`` -- notably ``bootstrap.execd.isolation``, which grants ``CAP_SYS_ADMIN``
with unconfined seccomp and AppArmor. None of them are fields on a sanitized spec.

The broader reason the broker exists is that isolation must not depend on backend configuration we
do not control. OpenSandbox does reject host bind mounts server-side by default, but "by default"
is a setting; the guarantee here does not rely on it.
"""

import base64
import binascii

from pydantic import BaseModel, ConfigDict, Field

from sandboxed_gym.backends.base import SanitizedEpisodeSpec
from sandboxed_gym.config import (
    JOB_ID_METADATA_KEY,
    K8S_LABEL_KEY_RE,
    PLATFORM_LABEL_PREFIX,
    RESERVED_ENV_PREFIX,
    RESERVED_METADATA_PREFIX,
    EpisodeBrokerConfig,
)
from sandboxed_gym.errors import BrokerRequestError
from sandboxed_gym.wire import (
    BrokerErrorCode,
    EpisodeCreateRequest,
    EpisodeExecRequest,
    EpisodeResources,
)


class SanitizedExecCall(BaseModel):
    """An exec request with broker-owned limits already applied."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    command: str
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    timeout_s: float
    user: str | int | None = None


def _reject(code: BrokerErrorCode, message: str, status_code: int) -> None:
    raise BrokerRequestError(code, message, status_code=status_code)


def _check_provider_options(request: EpisodeCreateRequest) -> None:
    """Reject any provider options at all.

    The field exists on the wire so a client never has to guess what is permitted, but nothing
    downstream reads it: a sanitized spec has no provider-options field and the backend builds its
    create request from named fields only. Accepting a key and then ignoring it would be the exact
    failure this is meant to prevent -- an environment grading under conditions it did not request.
    So the answer is always no, and it says which keys were refused.
    """
    if not request.provider_options:
        return
    _reject(
        BrokerErrorCode.FIELD_NOT_ALLOWED,
        "provider_options are not accepted by the broker; "
        f"refused key(s): {', '.join(sorted(request.provider_options))}",
        400,
    )


def _check_image(image: str, config: EpisodeBrokerConfig) -> None:
    """Enforce the approved-image policy. Fails closed when nothing is approved."""
    if any(char.isspace() for char in image):
        _reject(
            BrokerErrorCode.INVALID_REQUEST,
            "episode image must not contain whitespace",
            400,
        )
    if image in config.approved_images:
        return
    if any(image.startswith(prefix) for prefix in config.approved_image_prefixes):
        return
    _reject(
        BrokerErrorCode.IMAGE_NOT_APPROVED,
        f"episode image is not approved: {image}",
        403,
    )


def _sanitize_metadata(request: EpisodeCreateRequest, config: EpisodeBrokerConfig) -> dict[str, str]:
    """Copy caller metadata and stamp the owning job id.

    Broker-owned keys are refused rather than overwritten so a backend label always identifies the
    job that really owns an episode, including during reconciliation after a broker restart.
    """
    reserved = sorted(
        key for key in request.metadata if key.startswith((RESERVED_METADATA_PREFIX, PLATFORM_LABEL_PREFIX))
    )
    if reserved:
        _reject(
            BrokerErrorCode.FIELD_NOT_ALLOWED,
            f"metadata key(s) reserved by the broker: {', '.join(reserved)}",
            400,
        )
    # OpenSandbox turns sandbox metadata into Kubernetes pod labels, so a key that is not a valid
    # label key fails server-side as an opaque backend error. Catching it here gives the
    # environment something it can act on.
    invalid = sorted(key for key in request.metadata if not K8S_LABEL_KEY_RE.match(key))
    if invalid:
        _reject(
            BrokerErrorCode.INVALID_REQUEST,
            f"metadata key(s) are not valid Kubernetes label keys: {', '.join(invalid)}",
            400,
        )
    return {**request.metadata, JOB_ID_METADATA_KEY: config.job_id}


def _sanitize_env(env: dict[str, str]) -> dict[str, str]:
    """Validate environment variable names and values.

    There is deliberately no name allowlist here. Episode environment lands inside a sandbox that
    is already isolated, so a caller setting ``LD_PRELOAD`` or a proxy variable only affects its
    own episode; and a caller setting a credential name supplies its own worthless value rather
    than obtaining ours. An allowlist would have to enumerate every variable a grading image
    legitimately reads, which is unbounded, in exchange for no boundary it actually moves.

    What is refused is narrower and real: names that could be reinterpreted as structure by a
    backend that builds environment strings itself, and the prefix the broker reserves so a caller
    cannot pre-empt a variable the broker may later need to inject.
    """
    for key, value in env.items():
        if not key or "=" in key or "\x00" in key or "\x00" in value:
            _reject(
                BrokerErrorCode.INVALID_REQUEST,
                f"invalid environment variable name: {key!r}",
                400,
            )
    reserved = sorted(key for key in env if key.startswith(RESERVED_ENV_PREFIX))
    if reserved:
        _reject(
            BrokerErrorCode.FIELD_NOT_ALLOWED,
            f"environment variable name(s) reserved by the broker: {', '.join(reserved)}",
            400,
        )
    return dict(env)


def _check_resources(resources: EpisodeResources, config: EpisodeBrokerConfig) -> None:
    """Enforce per-episode resource caps so a caller cannot exhaust the cluster."""
    for label, requested, cap in (
        ("cpu", resources.cpu, config.max_cpu),
        ("memory_mib", resources.memory_mib, config.max_memory_mib),
        ("disk_gib", resources.disk_gib, config.max_disk_gib),
        ("gpu", resources.gpu, config.max_gpu),
    ):
        if requested is not None and cap is not None and requested > cap:
            _reject(
                BrokerErrorCode.QUOTA_EXCEEDED,
                f"requested {label}={requested} exceeds the per-episode cap of {cap}",
                400,
            )


def _decode_files(request: EpisodeCreateRequest) -> dict[str, bytes]:
    """Decode staged file content. Paths were validated by the wire model."""
    files: dict[str, bytes] = {}
    for path, content in request.files_b64.items():
        try:
            files[path] = base64.b64decode(content, validate=True)
        except (binascii.Error, ValueError):
            _reject(
                BrokerErrorCode.INVALID_REQUEST,
                f"file content is not valid base64: {path}",
                400,
            )
    return files


def sanitize_create_request(request: EpisodeCreateRequest, config: EpisodeBrokerConfig) -> SanitizedEpisodeSpec:
    """Rebuild an episode create request as a spec the broker will forward.

    Args:
        request: The parsed, still-untrusted request from the job sandbox.
        config: Trusted-side policy.

    Returns:
        A :class:`SanitizedEpisodeSpec` built only from permitted fields.

    Raises:
        BrokerRequestError: If the request asks for anything this deployment does not allow.
    """
    _check_provider_options(request)
    _check_image(request.image, config)
    _check_resources(request.resources, config)

    ready_timeout_s = None
    if request.ready_timeout_s is not None:
        ready_timeout_s = min(request.ready_timeout_s, config.max_ready_timeout_s)

    return SanitizedEpisodeSpec(
        job_id=config.job_id,
        image=request.image,
        ttl_s=min(request.ttl_s or config.default_ttl_s, config.max_ttl_s),
        ready_timeout_s=ready_timeout_s,
        workdir=request.workdir,
        env=_sanitize_env(request.env),
        metadata=_sanitize_metadata(request, config),
        resources=request.resources,
        entrypoint=tuple(request.entrypoint) if request.entrypoint is not None else None,
        files=_decode_files(request),
        # Mounts are platform-owned; a caller has no way to name one. No shipped NeMo-Gym agent
        # mounts anything into an episode -- content arrives through staged files -- so mounts
        # stay empty until something concrete needs them. Egress is platform-owned too, but it is
        # not per-request: it belongs to the backend that applies it.
        mounts=(),
    )


def sanitize_exec_request(request: EpisodeExecRequest, config: EpisodeBrokerConfig) -> SanitizedExecCall:
    """Apply broker-owned limits to an exec request.

    Args:
        request: The parsed, still-untrusted request from the job sandbox.
        config: Trusted-side policy.

    Returns:
        A :class:`SanitizedExecCall` with a bounded timeout.

    Raises:
        BrokerRequestError: If the request carries an invalid environment.
    """
    return SanitizedExecCall(
        command=request.command,
        cwd=request.cwd,
        env=_sanitize_env(request.env or {}),
        timeout_s=min(
            request.timeout_s or config.default_exec_timeout_s,
            config.max_exec_timeout_s,
        ),
        user=request.user,
    )
