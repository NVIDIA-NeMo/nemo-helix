# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve Platform secret references at substrate execution boundaries."""

from __future__ import annotations

import os

from nemo_deployments_plugin.entities import DeploymentConfig, EnvVar, SecretRef
from nemo_helix import AsyncNeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import NemoClientError, NotFoundError
from nemo_helix_plugin.config import get_platform_config
from nemo_helix_plugin.secrets.client import AsyncSecretsClient


class SecretResolutionError(RuntimeError):
    """A required environment secret could not be resolved."""


def platform_ngc_secret_ref() -> SecretRef | None:
    """Return the configured NGC secret reference without reading its value."""
    config = get_platform_config()
    parts = config.ngc_api_key_secret.strip().split("/")
    if len(parts) != 2 or not all(parts):
        return None
    return SecretRef(workspace=parts[0], name=parts[1])


async def resolve_secret_ref(sdk: AsyncNeMoHelix, secret_ref: SecretRef) -> str | None:
    """Resolve a Platform secret value without logging reference or value data."""
    try:
        secrets = client_from_platform(sdk, AsyncSecretsClient)
        response = (await secrets.access_secret(name=secret_ref.name, workspace=secret_ref.workspace)).data()
        if response.value:
            return response.value
    except NotFoundError:
        return None
    except NemoClientError as exc:
        raise SecretResolutionError("Platform secret access failed") from exc
    return None


async def resolve_deployment_config_secrets(sdk: AsyncNeMoHelix, config: DeploymentConfig) -> DeploymentConfig:
    """Return an execution-only copy whose secret references have plaintext values.

    Used by substrates that cannot mount a managed secret object (docker,
    openshell): every ``secret_ref`` env var is resolved to a plaintext
    ``EnvVar.value``. Vars that resolve to ``None`` (best-effort NGC only) are
    omitted so mock/local images can still start.
    """
    resolved = config.model_copy(deep=True)
    for container in (*resolved.init_containers, *resolved.containers):
        env: list[EnvVar] = []
        for item in container.env:
            resolved_item = await _resolve_env_var(sdk, item)
            if resolved_item is not None:
                env.append(resolved_item)
        container.env = env
    return resolved


async def resolve_deployment_secret_env(sdk: AsyncNeMoHelix, config: DeploymentConfig) -> dict[str, str]:
    """Collect resolved secret values for every ``secret_ref`` env var.

    Used by the k8s substrate to materialize a single per-deployment ``Secret``
    that is mounted via ``envFrom``, so plaintext never lands in the pod
    manifest. The returned mapping is keyed by the env var name across all
    containers. Later containers win on duplicate names, matching the
    single-Secret-per-deployment projection.

    Best-effort NGC semantics are preserved: a ``secret_ref`` that resolves to
    ``None`` is omitted rather than raising. Unauthorized references and
    secret-service access failures remain hard errors via ``_resolve_secret_value``.
    """
    secret_env: dict[str, str] = {}
    for container in (*config.init_containers, *config.containers):
        for item in container.env:
            if item.secret_ref is None:
                continue
            value = await _resolve_secret_value(sdk, item)
            if value is not None:
                secret_env[item.name] = value
    return secret_env


async def _resolve_env_var(sdk: AsyncNeMoHelix, item: EnvVar) -> EnvVar | None:
    """Resolve a secret-backed environment variable to a plaintext ``EnvVar``.

    Non-secret vars pass through unchanged. Secret vars that resolve to ``None``
    (best-effort NGC only) are omitted (returns ``None``).
    """
    if item.secret_ref is None:
        return item
    value = await _resolve_secret_value(sdk, item)
    if value is None:
        return None
    return EnvVar(name=item.name, value=value)


async def _resolve_secret_value(sdk: AsyncNeMoHelix, item: EnvVar) -> str | None:
    """Resolve the plaintext value for a secret-backed env var.

    NGC credentials are best-effort: when neither the configured secret nor the
    process-environment fallback is available, return ``None`` so callers can
    omit the variable and mock/local NIM images can still start. All other
    references are resolved via the Secrets service; a missing secret is a hard
    error, as is any secret-service access failure.
    """
    if item.secret_ref is None:
        raise SecretResolutionError(f"Environment variable {item.name!r} has no secret reference to resolve")
    value = await resolve_secret_ref(sdk, item.secret_ref)
    if value is not None:
        return value

    # Only reached when the Secrets service had no value: fall back to the NGC
    # process-env only for the platform NGC key, otherwise it is a hard error.
    is_ngc = item.name == "NGC_API_KEY" and item.secret_ref == platform_ngc_secret_ref()
    if is_ngc:
        return os.environ.get(get_platform_config().ngc_api_key_env_var)
    raise SecretResolutionError(
        f"Secret {item.secret_ref.workspace}/{item.secret_ref.name} for environment "
        f"variable {item.name!r} could not be resolved"
    )
